"""
Finite-temperature Thomas--Fermi average-atom solver.

This module implements the ion-sphere construction written explicitly in
Starrett and Saumon, :cite:`StarrettSaumon2014`, High Energy Density Physics
10, 35--42 (2014):

* full TF density and potential: Eqs. (2)--(5);
* fixed-chemical-potential external system: Eqs. (6)--(7);
* pseudoatom and screening densities: Eqs. (8)--(11).

The output deliberately follows the existing full/external KS payload so the
same pseudoatom QOZ/HNC code can consume either electronic approximation.
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, root
from scipy.special import expit

from otter.data.elements import element as element_info
from otter.data.helpers import ion_density_bohr3, mu_guess_from_density
from otter.electronic.potential import (
    effective_potential_external,
    effective_potential_full,
)
from otter.electronic.xc import (
    radial_core_diagnostics,
    resolve_gga_core_radius,
    xc_potential,
    xc_provenance,
)
from otter.literature import CitationMixin, citation_keys_for_xc_model
from otter.io.results import save_full_external_data
from otter.ionic.correlation import ion_sphere_radius_from_density
from otter.numerics.constants import EV_TO_HA
from otter.numerics.grids import create_sqrt_grid


_SQRT2_OVER_PI2 = np.sqrt(2.0) / np.pi**2


def _krylov_root(fun: Any, x0: np.ndarray, *, fatol: float, maxiter: int) -> Any:
    """Run SciPy's Krylov polish without its benign zero-step ratio warning.

    SciPy evaluates ``dx_norm / x_norm`` after a converged, exactly zero
    update; ``0 / 0`` then emits a RuntimeWarning even though the nonlinear
    residual and returned status are valid.  Suppress only that implementation
    warning.  All Otter residual, success, and finiteness checks remain active.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
            module=r"scipy\.optimize\._nonlin",
        )
        return root(
            fun,
            np.asarray(x0, dtype=float),
            method="krylov",
            options={"fatol": float(fatol), "maxiter": int(maxiter)},
        )


def _radial_charge(r: np.ndarray, density: np.ndarray) -> float:
    """Return ``4 pi integral r^2 n(r) dr`` on an arbitrary radial grid."""
    return float(4.0 * np.pi * np.trapezoid(density * r**2, r))


def _radial_charge_through(
    r: np.ndarray,
    density: np.ndarray,
    radius: float,
) -> float:
    """Integrate a density through an interpolated physical radius."""
    radius = float(radius)
    stop = int(np.searchsorted(r, radius, side="left"))
    if stop == 0:
        return float(4.0 * np.pi * density[0] * radius**3 / 3.0)
    if stop >= r.size:
        return _radial_charge(r, density)
    f = density * r**2
    f_radius = float(np.interp(radius, r[stop - 1 : stop + 1], f[stop - 1 : stop + 1]))
    r_int = np.concatenate((r[:stop], np.asarray([radius])))
    f_int = np.concatenate((f[:stop], np.asarray([f_radius])))
    return float(4.0 * np.pi * np.trapezoid(f_int, r_int))


def _fermi_half_sommerfeld(eta: np.ndarray) -> np.ndarray:
    """Sommerfeld expansion of the unnormalized ``I_{1/2}`` integral."""
    eta = np.asarray(eta, dtype=float)
    return (
        (2.0 / 3.0) * eta**1.5
        + (np.pi**2 / 12.0) * eta**-0.5
        + (7.0 * np.pi**4 / 960.0) * eta**-2.5
    )


def _fermi_half_boltzmann_series(eta: np.ndarray) -> np.ndarray:
    """Three-term fugacity series for the unnormalized ``I_{1/2}``."""
    z = np.exp(np.asarray(eta, dtype=float))
    gamma_3_2 = 0.5 * np.sqrt(np.pi)
    return gamma_3_2 * (
        z - z**2 / 2.0**1.5 + z**3 / 3.0**1.5
    )


class FermiHalfEvaluator:
    """Fast, controlled evaluator for Starrett's unnormalized ``I_{1/2}``."""

    def __init__(
        self,
        *,
        eta_min: float = -20.0,
        eta_max: float = 40.0,
        table_points: int = 12001,
        quadrature_order: int = 96,
    ) -> None:
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.eta_grid = np.linspace(
            self.eta_min, self.eta_max, int(table_points)
        )
        nodes, weights = leggauss(int(quadrature_order))
        u_max = np.sqrt(np.maximum(self.eta_grid, 0.0) + 45.0)
        u = 0.5 * (nodes[None, :] + 1.0) * u_max[:, None]
        integrand = 2.0 * u**2 * expit(
            self.eta_grid[:, None] - u**2
        )
        self.values = (
            0.5
            * u_max
            * np.sum(weights[None, :] * integrand, axis=1)
        )
        self._interpolator = PchipInterpolator(
            self.eta_grid, self.values, extrapolate=False
        )

    def __call__(self, eta: np.ndarray | float) -> np.ndarray:
        eta_arr = np.asarray(eta, dtype=float)
        flat = eta_arr.reshape(-1)
        out = np.empty_like(flat)
        low = flat < self.eta_min
        high = flat > self.eta_max
        mid = ~(low | high)
        if np.any(low):
            out[low] = _fermi_half_boltzmann_series(flat[low])
        if np.any(high):
            out[high] = _fermi_half_sommerfeld(flat[high])
        if np.any(mid):
            out[mid] = self._interpolator(flat[mid])
        return out.reshape(eta_arr.shape)


def incomplete_fermi_half_lower(
    eta: np.ndarray,
    upper: np.ndarray,
    *,
    quadrature_order: int = 96,
) -> np.ndarray:
    """
    Evaluate ``integral_0^upper sqrt(y)/(exp(y-eta)+1) dy``.

    In Starrett--Saumon Eq. (11), ``upper=-beta V_eff``.  This lower
    incomplete integral is exactly ``I_{1/2}-J_{1/2}`` with their convention
    for ``J``.
    """
    eta_arr, upper_arr = np.broadcast_arrays(
        np.asarray(eta, dtype=float),
        np.maximum(np.asarray(upper, dtype=float), 0.0),
    )
    nodes, weights = leggauss(int(quadrature_order))
    umax = np.sqrt(upper_arr.reshape(-1))
    u = 0.5 * (nodes[None, :] + 1.0) * umax[:, None]
    integrand = 2.0 * u**2 * expit(
        eta_arr.reshape(-1, 1) - u**2
    )
    values = (
        0.5
        * umax
        * np.sum(weights[None, :] * integrand, axis=1)
    )
    return values.reshape(eta_arr.shape)


def incomplete_fermi_half_upper(
    eta: np.ndarray,
    lower: np.ndarray,
    *,
    fermi_half: FermiHalfEvaluator,
    quadrature_order: int = 96,
) -> np.ndarray:
    """
    Evaluate Starrett's ``J_{1/2}(eta, lower)`` without subtraction.

    Near the nucleus both ``I_{1/2}`` and its lower incomplete part are very
    large while their difference (the positive-energy density) is modest.
    Forming ``I-J`` and then subtracting it back from the full density loses
    many digits.  With ``y=lower+x``, the directly evaluated upper integral is

    ``integral_0^inf sqrt(lower+x) expit(eta-lower-x) dx``.
    """
    eta_arr, lower_arr = np.broadcast_arrays(
        np.asarray(eta, dtype=float),
        np.maximum(np.asarray(lower, dtype=float), 0.0),
    )
    flat_eta = eta_arr.reshape(-1)
    flat_lower = lower_arr.reshape(-1)
    values = np.empty_like(flat_eta)
    # For a small lower limit, subtracting the tiny lower integral is stable
    # and avoids the square-root endpoint that an x-space quadrature would
    # otherwise see as lower -> 0.  At large lower limit (the nuclear region)
    # direct upper integration is essential because I and I-J are enormous.
    small = flat_lower < 1.0
    if np.any(small):
        values[small] = (
            fermi_half(flat_eta[small])
            - incomplete_fermi_half_lower(
                flat_eta[small],
                flat_lower[small],
                quadrature_order=int(quadrature_order),
            )
        )
    positive = ~small
    if np.any(positive):
        eta_p = flat_eta[positive]
        lower_p = flat_lower[positive]
        delta = eta_p - lower_p
        x_max = np.maximum(delta, 0.0) + 45.0
        nodes, weights = leggauss(int(quadrature_order))
        x = 0.5 * (nodes[None, :] + 1.0) * x_max[:, None]
        integrand = np.sqrt(lower_p[:, None] + x) * expit(
            delta[:, None] - x
        )
        values[positive] = (
            0.5
            * x_max
            * np.sum(weights[None, :] * integrand, axis=1)
        )
    return values.reshape(eta_arr.shape)


def _starrett_cutoff(r: np.ndarray, r_ws: float, c: float) -> np.ndarray:
    """Starrett et al. (2013) Eq. (80), reused in 2014 Eq. (11)."""
    if float(c) <= 0.0:
        return (r <= float(r_ws)).astype(float)
    exponent = (r - float(r_ws)) / (float(c) * float(r_ws))
    return (1.0 + np.exp(-1.0 / float(c))) * np.exp(-np.logaddexp(0.0, exponent))


@dataclass
class ThomasFermiConfig(CitationMixin):
    """Numerical controls for the finite-temperature TF ion-sphere solver."""

    element: int | str
    temperature_ev: float
    rho_g_cc: float
    xc_model: str = "dirac"
    gga_core_mode: str = "finite"
    gga_core_zr: float = 0.05
    r_ws_override_bohr: float | None = None
    n_i_override_bohr3: float | None = None
    run_mode: str = "full+ext"
    rmax_mult: float = 15.0
    n_points: int = 2**12
    mix: float = 0.18
    mixing_history: int = 6
    mixing_regularization: float = 5.0e-4
    max_iter: int = 300
    tol: float = 2.0e-7
    polish_tol: float = 1.0e-10
    polish_max_iter: int = 80
    mu_tol: float = 1.0e-10
    mu_bounds: tuple[float, float] = (-200.0, 200.0)
    mu_guess_zbar: float = 2.0
    ion_cut_c: float = 0.05
    quadrature_order: int = 160
    show_progress: bool = False
    verbose: bool = False
    v_full_init: np.ndarray | None = None
    v_full_init_r: np.ndarray | None = None
    v_ext_init: np.ndarray | None = None
    v_ext_init_r: np.ndarray | None = None
    full_fixed_mu_ha: float | None = None
    # Starrett--Saumon (2014), Sec. 2.4: the SC calculation retains the
    # chemical potential found by the preceding ion-sphere calculation.
    g_ii_override: np.ndarray | None = None
    g_ii_override_r: np.ndarray | None = None
    # Tabulated QOZ/HNC ion distribution replacing the IS step in Eqs. (4)
    # and (7).  This is required for the SC model of Sec. 2.4.
    v_corr_full: np.ndarray | None = None
    v_corr_full_r: np.ndarray | None = None
    v_corr_ext: np.ndarray | None = None
    v_corr_ext_r: np.ndarray | None = None
    # Additive V_Ie^C of Starrett--Saumon (2014), Eqs. (19)--(20), for the
    # full and external potential maps.
    save_data: bool = False
    save_output_dir: str | Path = "outputs"
    save_suffix: str = ""

    def __post_init__(self) -> None:
        self.gga_core_mode = str(self.gga_core_mode).strip().lower()
        if self.gga_core_mode not in {"finite", "strict"}:
            raise ValueError("gga_core_mode must be 'finite' or 'strict'.")
        if (
            not np.isfinite(float(self.gga_core_zr))
            or float(self.gga_core_zr) <= 0.0
        ):
            raise ValueError("gga_core_zr must be finite and positive.")
        if float(self.temperature_ev) <= 0.0:
            raise ValueError("temperature_ev must be positive for TF.")
        if float(self.rho_g_cc) <= 0.0:
            raise ValueError("rho_g_cc must be positive.")
        if float(self.rmax_mult) <= 1.0:
            raise ValueError("rmax_mult must exceed 1.")
        if int(self.n_points) < 64:
            raise ValueError("n_points must be at least 64.")
        if not (0.0 < float(self.mix) <= 1.0):
            raise ValueError("mix must lie in (0, 1].")
        if int(self.mixing_history) < 1:
            raise ValueError("mixing_history must be at least 1.")
        if float(self.mixing_regularization) <= 0.0:
            raise ValueError("mixing_regularization must be positive.")
        if int(self.max_iter) < 1 or int(self.polish_max_iter) < 1:
            raise ValueError("max_iter and polish_max_iter must be positive.")
        if (
            float(self.tol) <= 0.0
            or float(self.polish_tol) <= 0.0
            or float(self.mu_tol) <= 0.0
        ):
            raise ValueError("TF convergence tolerances must be positive.")
        if int(self.quadrature_order) < 16:
            raise ValueError("quadrature_order must be at least 16.")
        if float(self.ion_cut_c) < 0.0:
            raise ValueError("ion_cut_c must be non-negative.")
        if self.full_fixed_mu_ha is not None and not np.isfinite(
            float(self.full_fixed_mu_ha)
        ):
            raise ValueError("full_fixed_mu_ha must be finite when supplied.")

    @property
    def citation_keys(self) -> tuple[str, ...]:
        """Primary papers for the TF pseudoatom and selected XC model."""
        return (
            "StarrettSaumon2014",
            *citation_keys_for_xc_model(self.xc_model),
        )


def _resolve_geometry(
    cfg: ThomasFermiConfig,
) -> tuple[Any, float, float]:
    elem = element_info(cfg.element)
    if cfg.r_ws_override_bohr is not None:
        r_ws = float(cfg.r_ws_override_bohr)
        n_i = (
            float(cfg.n_i_override_bohr3)
            if cfg.n_i_override_bohr3 is not None
            else 3.0 / (4.0 * np.pi * r_ws**3)
        )
    elif cfg.n_i_override_bohr3 is not None:
        n_i = float(cfg.n_i_override_bohr3)
        r_ws = float(ion_sphere_radius_from_density(n_i))
    else:
        n_i = float(
            ion_density_bohr3(float(cfg.rho_g_cc), float(elem.atomic_mass))
        )
        r_ws = float(ion_sphere_radius_from_density(n_i))
    if n_i <= 0.0 or r_ws <= 0.0:
        raise ValueError("TF ion density and R_ws must be positive.")
    return elem, n_i, r_ws


def _resample_initial(
    values: np.ndarray | None,
    source_r: np.ndarray | None,
    target_r: np.ndarray,
) -> np.ndarray | None:
    if values is None:
        return None
    vals = np.asarray(values, dtype=float)
    if source_r is None:
        if vals.shape != target_r.shape:
            return None
        return vals.copy()
    src = np.asarray(source_r, dtype=float)
    if src.shape != vals.shape or src.ndim != 1:
        return None
    return np.interp(target_r, src, vals, left=vals[0], right=0.0)


def _resample_profile(
    values: np.ndarray | None,
    source_r: np.ndarray | None,
    target_r: np.ndarray,
    *,
    name: str,
    right: float,
) -> np.ndarray | None:
    """Validate and resample one radial SC-feedback profile."""
    if values is None:
        if source_r is not None:
            raise ValueError(f"{name}_r requires {name}.")
        return None
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1 or vals.size < 2 or not np.all(np.isfinite(vals)):
        raise ValueError(f"{name} must be a finite one-dimensional array.")
    if source_r is None:
        if vals.shape != target_r.shape:
            raise ValueError(
                f"{name} must match the TF grid when {name}_r is omitted."
            )
        return vals.copy()
    src = np.asarray(source_r, dtype=float)
    if (
        src.ndim != 1
        or src.shape != vals.shape
        or not np.all(np.isfinite(src))
        or np.any(np.diff(src) <= 0.0)
    ):
        raise ValueError(
            f"{name}_r must be finite, strictly increasing, and match {name}."
        )
    return np.interp(
        target_r,
        src,
        vals,
        left=float(vals[0]),
        right=float(right),
    )


def _resolve_sc_profiles(
    cfg: ThomasFermiConfig,
    *,
    r: np.ndarray,
    r_ws: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Resolve the IS or SC ionic background and Eq. (19) potentials.

    The analytic sharp-background potential is allowed only for the literal
    ion-sphere step.  A tabulated ``g_ii_override`` must be integrated as the
    actual source in Starrett--Saumon (2014), Eqs. (4) and (7); replacing it
    with the ion-sphere analytic shortcut would silently remove the SC
    feedback from the TF fixed-point map.
    """
    g_override = _resample_profile(
        cfg.g_ii_override,
        cfg.g_ii_override_r,
        r,
        name="g_ii_override",
        right=1.0,
    )
    if g_override is None:
        g_ii = (r >= float(r_ws)).astype(float)
        analytic_radius: float | None = float(r_ws)
    else:
        if np.any(g_override < 0.0):
            raise ValueError("g_ii_override must be non-negative.")
        g_ii = np.asarray(g_override, dtype=float)
        analytic_radius = None

    v_corr_full = _resample_profile(
        cfg.v_corr_full,
        cfg.v_corr_full_r,
        r,
        name="v_corr_full",
        right=0.0,
    )
    v_corr_ext = _resample_profile(
        cfg.v_corr_ext,
        cfg.v_corr_ext_r,
        r,
        name="v_corr_ext",
        right=0.0,
    )
    if v_corr_full is None:
        v_corr_full = np.zeros_like(r)
    if v_corr_ext is None:
        v_corr_ext = np.zeros_like(r)
    return (
        np.asarray(g_ii, dtype=float),
        np.asarray(v_corr_full, dtype=float),
        np.asarray(v_corr_ext, dtype=float),
        analytic_radius,
    )


def _tf_density(
    potential: np.ndarray,
    *,
    mu: float,
    temperature: float,
    fermi_half: FermiHalfEvaluator,
) -> np.ndarray:
    eta = (float(mu) - np.asarray(potential, dtype=float)) / float(temperature)
    return (
        _SQRT2_OVER_PI2
        * float(temperature) ** 1.5
        * fermi_half(eta)
    )


def _field_free_density(
    mu: float,
    temperature: float,
    fermi_half: FermiHalfEvaluator,
) -> float:
    return float(
        _SQRT2_OVER_PI2
        * float(temperature) ** 1.5
        * fermi_half(np.asarray(float(mu) / float(temperature)))
    )


def _neutral_mu(
    potential: np.ndarray,
    *,
    r: np.ndarray,
    r_ws: float,
    z_nuc: int,
    temperature: float,
    guess: float,
    bounds: tuple[float, float],
    tol: float,
    fermi_half: FermiHalfEvaluator,
) -> float:
    """Solve Starrett--Saumon Eq. (3) at fixed TF potential."""

    def residual(mu: float) -> float:
        density = _tf_density(
            potential,
            mu=float(mu),
            temperature=float(temperature),
            fermi_half=fermi_half,
        )
        return _radial_charge_through(r, density, r_ws) - float(z_nuc)

    lo_cfg, hi_cfg = (float(bounds[0]), float(bounds[1]))
    width = max(2.0 * float(temperature), 1.0)
    lo = max(lo_cfg, float(guess) - width)
    hi = min(hi_cfg, float(guess) + width)
    f_lo = residual(lo)
    f_hi = residual(hi)
    for _ in range(32):
        if f_lo <= 0.0 <= f_hi:
            break
        width *= 1.8
        if f_lo > 0.0:
            lo = max(lo_cfg, float(guess) - width)
            f_lo = residual(lo)
        if f_hi < 0.0:
            hi = min(hi_cfg, float(guess) + width)
            f_hi = residual(hi)
        if lo <= lo_cfg and hi >= hi_cfg:
            break
    if not (f_lo <= 0.0 <= f_hi):
        raise RuntimeError(
            "TF ion-sphere neutrality could not bracket mu: "
            f"Q(lo)-Z={f_lo:.3e}, Q(hi)-Z={f_hi:.3e}, "
            f"bounds=({lo_cfg:.3e}, {hi_cfg:.3e}) Ha."
        )
    return float(
        brentq(
            residual,
            lo,
            hi,
            xtol=float(tol),
            rtol=max(float(tol), 4.0 * np.finfo(float).eps),
            maxiter=100,
        )
    )


def _fixed_point_error(
    r: np.ndarray,
    new: np.ndarray,
    old: np.ndarray,
    scale: float,
) -> float:
    return float(
        np.max(np.abs((np.asarray(new) - np.asarray(old)) * r))
        / max(abs(float(scale)), 1.0)
    )


def _eyert_update(
    x_in: np.ndarray,
    x_out: np.ndarray,
    *,
    r: np.ndarray,
    mix: float,
    regularization: float,
    x_prev: np.ndarray | None,
    f_prev: np.ndarray | None,
    dx_history: deque[np.ndarray],
    df_history: deque[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the short-history Eyert/Anderson update used by the KS-AA solver.

    See V. Eyert, J. Comput. Phys. 124, 271 (1996), and Eqs. (59)--(63)
    of Starrett et al., Comput. Phys. Commun. 235, 50 (2019).
    """
    x_in = np.asarray(x_in, dtype=float)
    f_now = np.asarray(x_out, dtype=float) - x_in
    if x_prev is not None and f_prev is not None:
        dx_history.append(x_in - x_prev)
        df_history.append(f_now - f_prev)
    x_next = x_in + float(mix) * f_now
    if dx_history:
        count = len(dx_history)
        matrix = np.empty((count, count), dtype=float)
        rhs = np.empty(count, dtype=float)
        for i in range(count):
            rhs[i] = np.trapezoid(df_history[i] * f_now, r)
            for j in range(count):
                matrix[i, j] = np.trapezoid(
                    df_history[i] * df_history[j], r
                )
            matrix[i, i] += float(regularization) ** 2
        try:
            weights = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            weights = None
        if weights is not None and np.all(np.isfinite(weights)):
            correction = np.zeros_like(x_in)
            for weight, dx, df in zip(
                weights, dx_history, df_history, strict=True
            ):
                correction += weight * (dx + float(mix) * df)
            candidate = x_next - correction
            if np.all(np.isfinite(candidate)):
                x_next = candidate
    return x_next, x_in.copy(), f_now.copy()


def _solve_full(
    cfg: ThomasFermiConfig,
    *,
    r: np.ndarray,
    r_ws: float,
    n_i: float,
    z_nuc: int,
    temperature: float,
    fermi_half: FermiHalfEvaluator,
    g_ii: np.ndarray,
    v_corr: np.ndarray,
    analytic_ion_sphere_radius: float | None,
) -> tuple[np.ndarray, np.ndarray, float, float, list[dict[str, float]], bool]:
    v_init = _resample_initial(cfg.v_full_init, cfg.v_full_init_r, r)
    if v_init is None:
        v = -float(z_nuc) * np.exp(-r / float(r_ws)) / r
    else:
        v = np.asarray(v_init, dtype=float)
    fixed_mu = (
        None
        if cfg.full_fixed_mu_ha is None
        else float(cfg.full_fixed_mu_ha)
    )
    mu = (
        float(mu_guess_from_density(n_i, zbar=float(cfg.mu_guess_zbar)))
        if fixed_mu is None
        else float(fixed_mu)
    )
    density_prev: np.ndarray | None = None
    history: list[dict[str, float]] = []
    x_prev: np.ndarray | None = None
    f_prev: np.ndarray | None = None
    dx_history: deque[np.ndarray] = deque(maxlen=max(int(cfg.mixing_history), 1))
    df_history: deque[np.ndarray] = deque(maxlen=max(int(cfg.mixing_history), 1))
    coarse_converged = False
    coarse_tol = max(float(cfg.tol), 5.0e-5)

    for iteration in range(int(cfg.max_iter)):
        if fixed_mu is None:
            mu = _neutral_mu(
                v,
                r=r,
                r_ws=r_ws,
                z_nuc=z_nuc,
                temperature=temperature,
                guess=mu,
                bounds=cfg.mu_bounds,
                tol=cfg.mu_tol,
                fermi_half=fermi_half,
            )
        else:
            mu = float(fixed_mu)
        density = _tf_density(v, mu=mu, temperature=temperature, fermi_half=fermi_half)
        n0 = _field_free_density(mu, temperature, fermi_half)
        v_map = effective_potential_full(
            r,
            density,
            n0,
            g_ii,
            float(z_nuc),
            xc_model=cfg.xc_model,
            ion_sphere_radius=analytic_ion_sphere_radius,
            gga_core_mode=cfg.gga_core_mode,
            gga_core_zr=cfg.gga_core_zr,
        )
        v_map = np.asarray(v_map, dtype=float) + np.asarray(v_corr, dtype=float)
        v_map -= float(np.mean(v_map[-min(32, v_map.size) :]))
        error = _fixed_point_error(r, v_map, v, float(z_nuc))
        if density_prev is None:
            dn_rel = np.inf
        else:
            numerator = _radial_charge(r, np.abs(density - density_prev))
            denominator = max(_radial_charge(r, density), 1.0)
            dn_rel = float(numerator / denominator)
        charge_error = abs(_radial_charge_through(r, density, r_ws) - float(z_nuc))
        history.append(
            {
                "iter": float(iteration),
                "err": float(error),
                "dv_rel": float(error),
                "dn_rel": float(dn_rel),
                "mu": float(mu),
                "n0": float(n0),
                "q_ws_error": float(charge_error),
                "mix": float(cfg.mix),
            }
        )
        if cfg.show_progress or cfg.verbose:
            if iteration == 0 or (iteration + 1) % 10 == 0:
                print(
                    "[TF-full] "
                    f"iter={iteration + 1} err={error:.3e} "
                    f"dn={dn_rel:.3e} mu={mu:.8f} Ha "
                    f"Qws-Z={charge_error:.3e}"
                )
        if (
            error <= coarse_tol
            and (not np.isfinite(dn_rel) or dn_rel <= 5.0 * coarse_tol)
            and (fixed_mu is not None or charge_error <= 1.0e-7)
        ):
            v = v_map
            coarse_converged = True
            break
        x_in = v * r / float(z_nuc)
        x_out = v_map * r / float(z_nuc)
        x_next, x_prev, f_prev = _eyert_update(
            x_in,
            x_out,
            r=r,
            mix=float(cfg.mix),
            regularization=float(cfg.mixing_regularization),
            x_prev=x_prev,
            f_prev=f_prev,
            dx_history=dx_history,
            df_history=df_history,
        )
        v = x_next * float(z_nuc) / r
        v -= float(np.mean(v[-min(32, v.size) :]))
        density_prev = density

    # A matrix-free Newton--Krylov polish removes the small long-range
    # fixed-point residual left by damped mixing.  That residual is harmless
    # for local densities but is amplified by the r^2 and r^4 moments entering
    # q(k -> 0), so it must not be passed to QOZ.
    mu_holder = [float(mu)]

    def residual_x(x_value: np.ndarray) -> np.ndarray:
        potential = np.asarray(x_value, dtype=float) * float(z_nuc) / r
        if fixed_mu is None:
            mu_value = _neutral_mu(
                potential,
                r=r,
                r_ws=r_ws,
                z_nuc=z_nuc,
                temperature=temperature,
                guess=mu_holder[0],
                bounds=cfg.mu_bounds,
                tol=cfg.mu_tol,
                fermi_half=fermi_half,
            )
        else:
            mu_value = float(fixed_mu)
        mu_holder[0] = float(mu_value)
        density_value = _tf_density(
            potential,
            mu=mu_value,
            temperature=temperature,
            fermi_half=fermi_half,
        )
        n0_value = _field_free_density(mu_value, temperature, fermi_half)
        mapped = effective_potential_full(
            r,
            density_value,
            n0_value,
            g_ii,
            float(z_nuc),
            xc_model=cfg.xc_model,
            ion_sphere_radius=analytic_ion_sphere_radius,
            gga_core_mode=cfg.gga_core_mode,
            gga_core_zr=cfg.gga_core_zr,
        )
        mapped = np.asarray(mapped, dtype=float) + np.asarray(v_corr, dtype=float)
        mapped -= float(np.mean(mapped[-min(32, mapped.size) :]))
        return mapped * r / float(z_nuc) - np.asarray(x_value, dtype=float)

    polish = _krylov_root(
        residual_x,
        np.asarray(v * r / float(z_nuc), dtype=float),
        fatol=float(cfg.polish_tol),
        maxiter=int(cfg.polish_max_iter),
    )
    polish_error = float(np.max(np.abs(np.asarray(polish.fun, dtype=float))))
    if bool(polish.success) and np.isfinite(polish_error):
        v = np.asarray(polish.x, dtype=float) * float(z_nuc) / r
        mu = float(mu_holder[0])
    converged = bool(
        bool(polish.success)
        and np.isfinite(polish_error)
        and polish_error <= max(10.0 * float(cfg.polish_tol), float(cfg.tol))
    )
    if not converged:
        converged = bool(coarse_converged and polish_error <= coarse_tol)
    history.append(
        {
            "iter": float(len(history)),
            "err": float(polish_error),
            "dv_rel": float(polish_error),
            "dn_rel": np.nan,
            "mu": float(mu_holder[0]),
            "n0": float(_field_free_density(mu_holder[0], temperature, fermi_half)),
            "q_ws_error": np.nan,
            "mix": 0.0,
            "phase": "krylov_polish",
            "success": bool(polish.success),
            "nfev": int(polish.nfev),
        }
    )

    if fixed_mu is None:
        mu = _neutral_mu(
            v,
            r=r,
            r_ws=r_ws,
            z_nuc=z_nuc,
            temperature=temperature,
            guess=mu,
            bounds=cfg.mu_bounds,
            tol=cfg.mu_tol,
            fermi_half=fermi_half,
        )
    else:
        mu = float(fixed_mu)
    density = _tf_density(
        v, mu=mu, temperature=temperature, fermi_half=fermi_half
    )
    n0 = _field_free_density(mu, temperature, fermi_half)
    return density, v, mu, n0, history, converged


def _solve_external(
    cfg: ThomasFermiConfig,
    *,
    r: np.ndarray,
    r_ws: float,
    z_nuc: int,
    temperature: float,
    mu: float,
    n0: float,
    fermi_half: FermiHalfEvaluator,
    g_ii: np.ndarray,
    v_corr: np.ndarray,
    analytic_ion_sphere_radius: float | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], bool]:
    v_init = _resample_initial(cfg.v_ext_init, cfg.v_ext_init_r, r)
    if v_init is None:
        density0 = np.full_like(r, float(n0))
        v = effective_potential_external(
            r,
            density0,
            float(n0),
            g_ii,
            xc_model=cfg.xc_model,
            ion_sphere_radius=analytic_ion_sphere_radius,
            nuclear_charge=float(z_nuc),
            gga_core_mode=cfg.gga_core_mode,
            gga_core_zr=cfg.gga_core_zr,
        )
        v = np.asarray(v, dtype=float) + np.asarray(v_corr, dtype=float)
    else:
        v = np.asarray(v_init, dtype=float)
    v = np.asarray(v, dtype=float)
    v -= float(np.mean(v[-min(32, v.size) :]))
    density_prev: np.ndarray | None = None
    history: list[dict[str, float]] = []
    x_prev: np.ndarray | None = None
    f_prev: np.ndarray | None = None
    dx_history: deque[np.ndarray] = deque(maxlen=max(int(cfg.mixing_history), 1))
    df_history: deque[np.ndarray] = deque(maxlen=max(int(cfg.mixing_history), 1))
    coarse_converged = False
    coarse_tol = max(float(cfg.tol), 5.0e-5)

    for iteration in range(int(cfg.max_iter)):
        density = _tf_density(v, mu=mu, temperature=temperature, fermi_half=fermi_half)
        v_map = effective_potential_external(
            r,
            density,
            float(n0),
            g_ii,
            xc_model=cfg.xc_model,
            ion_sphere_radius=analytic_ion_sphere_radius,
            nuclear_charge=float(z_nuc),
            gga_core_mode=cfg.gga_core_mode,
            gga_core_zr=cfg.gga_core_zr,
        )
        v_map = np.asarray(v_map, dtype=float) + np.asarray(v_corr, dtype=float)
        v_map -= float(np.mean(v_map[-min(32, v_map.size) :]))
        error = _fixed_point_error(r, v_map, v, float(z_nuc))
        if density_prev is None:
            dn_rel = np.inf
        else:
            numerator = _radial_charge(r, np.abs(density - density_prev))
            denominator = max(_radial_charge(r, density), 1.0)
            dn_rel = float(numerator / denominator)
        history.append(
            {
                "iter": float(iteration),
                "err": float(error),
                "dv_rel": float(error),
                "dn_rel": float(dn_rel),
                "mu": float(mu),
                "n0": float(n0),
                "mix": float(cfg.mix),
            }
        )
        if cfg.show_progress or cfg.verbose:
            if iteration == 0 or (iteration + 1) % 10 == 0:
                print(
                    "[TF-ext] "
                    f"iter={iteration + 1} err={error:.3e} "
                    f"dn={dn_rel:.3e}"
                )
        if error <= coarse_tol and (
            not np.isfinite(dn_rel) or dn_rel <= 5.0 * coarse_tol
        ):
            v = v_map
            coarse_converged = True
            break
        v, x_prev, f_prev = _eyert_update(
            v,
            v_map,
            r=r,
            mix=float(cfg.mix),
            regularization=float(cfg.mixing_regularization),
            x_prev=x_prev,
            f_prev=f_prev,
            dx_history=dx_history,
            df_history=df_history,
        )
        v -= float(np.mean(v[-min(32, v.size) :]))
        density_prev = density

    def residual_v(potential: np.ndarray) -> np.ndarray:
        density_value = _tf_density(
            potential,
            mu=mu,
            temperature=temperature,
            fermi_half=fermi_half,
        )
        mapped = effective_potential_external(
            r,
            density_value,
            float(n0),
            g_ii,
            xc_model=cfg.xc_model,
            ion_sphere_radius=analytic_ion_sphere_radius,
            nuclear_charge=float(z_nuc),
            gga_core_mode=cfg.gga_core_mode,
            gga_core_zr=cfg.gga_core_zr,
        )
        mapped = np.asarray(mapped, dtype=float) + np.asarray(v_corr, dtype=float)
        mapped -= float(np.mean(mapped[-min(32, mapped.size) :]))
        return mapped - np.asarray(potential, dtype=float)

    polish = _krylov_root(
        residual_v,
        np.asarray(v, dtype=float),
        fatol=float(cfg.polish_tol),
        maxiter=int(cfg.polish_max_iter),
    )
    polish_error = float(np.max(np.abs(np.asarray(polish.fun, dtype=float))))
    if bool(polish.success) and np.isfinite(polish_error):
        v = np.asarray(polish.x, dtype=float)
    converged = bool(
        bool(polish.success)
        and np.isfinite(polish_error)
        and polish_error <= max(10.0 * float(cfg.polish_tol), float(cfg.tol))
    )
    if not converged:
        converged = bool(coarse_converged and polish_error <= coarse_tol)
    history.append(
        {
            "iter": float(len(history)),
            "err": float(polish_error),
            "dv_rel": float(polish_error),
            "dn_rel": np.nan,
            "mu": float(mu),
            "n0": float(n0),
            "mix": 0.0,
            "phase": "krylov_polish",
            "success": bool(polish.success),
            "nfev": int(polish.nfev),
        }
    )

    density = _tf_density(
        v, mu=mu, temperature=temperature, fermi_half=fermi_half
    )
    return density, v, history, converged


def solve_thomas_fermi_full_then_external(
    cfg: ThomasFermiConfig,
) -> dict[str, Any]:
    """Solve finite-T TF full and external IS or SC systems.

    With the default configuration this is the ion-sphere construction of
    Starrett and Saumon (2014), Eqs. (2)--(11).  Supplying
    ``full_fixed_mu_ha``, ``g_ii_override``, and the two correlation
    potentials activates the electronic part of their Sec. 2.4 SC iteration:
    the IS chemical potential is retained, the QOZ/HNC ``g_II`` replaces the
    sharp step in Eqs. (4)/(7), and ``V_Ie^C`` from Eqs. (19)/(20) is added to
    both maps.
    """
    if float(cfg.temperature_ev) <= 0.0:
        raise ValueError("Thomas-Fermi solver requires temperature_ev > 0.")
    run_mode = str(cfg.run_mode).strip().lower()
    if run_mode not in {"full", "full+ext", "full_ext"}:
        raise ValueError("TF run_mode must be 'full' or 'full+ext'.")
    elem, n_i, r_ws = _resolve_geometry(cfg)
    z_nuc = int(elem.z)
    temperature = float(cfg.temperature_ev) * EV_TO_HA
    rmax = float(cfg.rmax_mult) * float(r_ws)
    r = np.asarray(
        create_sqrt_grid(rmax, int(cfg.n_points)).r, dtype=float
    )
    fermi_half = FermiHalfEvaluator(
        quadrature_order=int(cfg.quadrature_order)
    )
    g_ii, v_corr_full, v_corr_ext, analytic_radius = _resolve_sc_profiles(
        cfg,
        r=r,
        r_ws=r_ws,
    )

    n_full, v_full, mu, n0, history, full_converged = _solve_full(
        cfg,
        r=r,
        r_ws=r_ws,
        n_i=n_i,
        z_nuc=z_nuc,
        temperature=temperature,
        fermi_half=fermi_half,
        g_ii=g_ii,
        v_corr=v_corr_full,
        analytic_ion_sphere_radius=analytic_radius,
    )
    do_external = run_mode in {"full+ext", "full_ext"}
    if do_external:
        n_ext, v_ext, ext_history, ext_converged = _solve_external(
            cfg,
            r=r,
            r_ws=r_ws,
            z_nuc=z_nuc,
            temperature=temperature,
            mu=mu,
            n0=n0,
            fermi_half=fermi_half,
            g_ii=g_ii,
            v_corr=v_corr_ext,
            analytic_ion_sphere_radius=analytic_radius,
        )
    else:
        n_ext = np.full_like(r, float(n0))
        v_ext = np.zeros_like(r)
        ext_history = []
        ext_converged = True

    beta = 1.0 / float(temperature)
    eta = beta * (float(mu) - v_full)
    upper = np.maximum(-beta * v_full, 0.0)
    n_free = (
        _SQRT2_OVER_PI2
        * float(temperature) ** 1.5
        * incomplete_fermi_half_upper(
            eta,
            upper,
            fermi_half=fermi_half,
            quadrature_order=int(cfg.quadrature_order),
        )
    )
    # The lower (negative-energy) part is best obtained from the already
    # evaluated total only once.  More importantly, n_scr below is assembled
    # from the direct upper integral, avoiding the catastrophic
    # n_full - n_negative cancellation at small r.
    n_negative = np.maximum(n_full - n_free, 0.0)
    cutoff = _starrett_cutoff(r, r_ws, float(cfg.ion_cut_c))
    n_ion = np.asarray(n_negative * cutoff, dtype=float)
    n_pa = np.asarray(n_full - n_ext, dtype=float)
    n_scr = np.asarray(
        n_free - n_ext + (1.0 - cutoff) * n_negative,
        dtype=float,
    )
    n_free_total = np.asarray(n_full - n_ion, dtype=float)
    q_ion = _radial_charge(r, n_ion)
    zbar_partition = float(z_nuc) - float(q_ion)
    zbar_background = float(n0 / n_i)

    n0_arr = np.full_like(r, float(n0))
    v_nuc = -float(z_nuc) / r
    gga_core_radius = resolve_gga_core_radius(
        cfg.xc_model,
        nuclear_charge=float(z_nuc),
        mode=cfg.gga_core_mode,
        core_zr=cfg.gga_core_zr,
        r=r,
    )
    v_xc = xc_potential(
        n_full,
        model=cfg.xc_model,
        r=r,
        gga_core_radius_bohr=gga_core_radius,
    ) - xc_potential(
        n0_arr,
        model=cfg.xc_model,
        r=r,
        gga_core_radius_bohr=gga_core_radius,
    )
    v_h = np.asarray(v_full - v_nuc - v_xc - v_corr_full, dtype=float)
    v_xc_ext = xc_potential(
        n_ext,
        model=cfg.xc_model,
        r=r,
        gga_core_radius_bohr=gga_core_radius,
    ) - xc_potential(
        n0_arr,
        model=cfg.xc_model,
        r=r,
        gga_core_radius_bohr=gga_core_radius,
    )
    v_h_ext = np.asarray(v_ext - v_xc_ext - v_corr_ext, dtype=float)
    core_diagnostics = radial_core_diagnostics(
        r,
        n_full,
        v_xc,
        nuclear_charge=float(z_nuc),
        core_radius_bohr=gga_core_radius,
    )

    xc_provenance_record = xc_provenance(cfg.xc_model)
    meta = {
        "element": str(elem.symbol),
        "Z": int(z_nuc),
        "atomic_mass": float(elem.atomic_mass),
        "rho_g_cc": float(cfg.rho_g_cc),
        "temperature_ev": float(cfg.temperature_ev),
        "temperature_ha": float(temperature),
        "n_i_bohr3": float(n_i),
        "r_ws_bohr": float(r_ws),
        "r_max_bohr": float(r[-1]),
        "electronic_model": "thomas_fermi",
        "tf_density_equation": "Starrett_Saumon_2014_Eq2_Eq6",
        "tf_ion_density_equation": "Starrett_Saumon_2014_Eq11",
        "tf_xc_model": str(cfg.xc_model),
        "xc_provenance": xc_provenance_record,
        "gga_core_mode": str(cfg.gga_core_mode),
        "gga_core_zr": float(cfg.gga_core_zr),
        "gga_core_radius_bohr": (
            float(gga_core_radius) if gga_core_radius is not None else np.nan
        ),
        "gga_core_points": int(core_diagnostics["core_points"]),
        "gga_radial_operator": "sqrt-grid-discrete-adjoint-v1",
        "density_cusp_rel_error": float(
            core_diagnostics["density_cusp_rel_error"]
        ),
        "v_xc_core_turn_count": int(
            core_diagnostics["potential_turn_count"]
        ),
        "tf_quadrature_order": int(cfg.quadrature_order),
        "tf_scf_tol": float(cfg.tol),
        "tf_full_converged": bool(full_converged),
        "tf_external_converged": bool(ext_converged),
        "structure_model": (
            "SC_feedback_electronic_step" if cfg.g_ii_override is not None else "IS"
        ),
        "fixed_is_mu_used": bool(cfg.full_fixed_mu_ha is not None),
        "fixed_is_mu_ha": (
            float(cfg.full_fixed_mu_ha) if cfg.full_fixed_mu_ha is not None else np.nan
        ),
        "g_ii_override_used": bool(cfg.g_ii_override is not None),
        "analytic_ion_sphere_background": bool(analytic_radius is not None),
        "v_corr_full_used": bool(cfg.v_corr_full is not None),
        "v_corr_ext_used": bool(cfg.v_corr_ext is not None),
        "sc_reference": (
            "Starrett_Saumon_2014_Sec2.4_Eqs19_20"
            if cfg.g_ii_override is not None
            else "not_applicable"
        ),
        "q_full_ws": float(_radial_charge_through(r, n_full, r_ws)),
        "q_ion_all": float(q_ion),
        "zbar_partition": float(zbar_partition),
        "zbar_background": float(zbar_background),
    }
    result: dict[str, Any] = {
        "r": r,
        "r_ws": float(r_ws),
        "n_i": float(n_i),
        "n0": float(n0),
        "mu": float(mu),
        "n_full": np.asarray(n_full, dtype=float),
        "n_bound": np.asarray(n_ion, dtype=float),
        "n_cont": np.asarray(n_free_total, dtype=float),
        "n_free": np.asarray(n_free_total, dtype=float),
        "n_positive_energy_tf": np.asarray(n_free, dtype=float),
        "n_ext": np.asarray(n_ext, dtype=float),
        "n_pa": np.asarray(n_pa, dtype=float),
        "n_ion": np.asarray(n_ion, dtype=float),
        "n_negative_tf": np.asarray(n_negative, dtype=float),
        "n_scr": np.asarray(n_scr, dtype=float),
        "ion_cutoff": np.asarray(cutoff, dtype=float),
        "v_full": np.asarray(v_full, dtype=float),
        "v_scf": np.asarray(v_full, dtype=float),
        "v_ext": np.asarray(v_ext, dtype=float),
        "v_nuc": np.asarray(v_nuc, dtype=float),
        "v_H": np.asarray(v_h, dtype=float),
        "v_xc": np.asarray(v_xc, dtype=float),
        "xc_model": str(cfg.xc_model),
        "xc_provenance": xc_provenance_record,
        "v_H_ext": np.asarray(v_h_ext, dtype=float),
        "v_xc_ext": np.asarray(v_xc_ext, dtype=float),
        "gga_core_mode": str(cfg.gga_core_mode),
        "gga_core_zr": float(cfg.gga_core_zr),
        "gga_core_radius_bohr": (
            float(gga_core_radius) if gga_core_radius is not None else np.nan
        ),
        "gga_core_points": int(core_diagnostics["core_points"]),
        "gga_radial_operator": "sqrt-grid-discrete-adjoint-v1",
        "density_cusp_rel_error": float(
            core_diagnostics["density_cusp_rel_error"]
        ),
        "v_xc_core_turn_count": int(
            core_diagnostics["potential_turn_count"]
        ),
        "v_xc_core_max_abs_ha": float(
            core_diagnostics["max_abs_potential_ha"]
        ),
        "g_ii_background": np.asarray(g_ii, dtype=float),
        "v_corr_full": np.asarray(v_corr_full, dtype=float),
        "v_corr_ext": np.asarray(v_corr_ext, dtype=float),
        "zbar": float(zbar_background),
        "zbar_partition": float(zbar_partition),
        "q_ion_all": float(q_ion),
        "history": list(history),
        "stage1_history": [],
        "stage1_iters": 0,
        "stage1_converged": True,
        "stage2_iters": int(len(history)),
        "stage2_converged": bool(full_converged),
        "converged": bool(full_converged),
        "threshold_state_status": "not_applicable_tf",
        "ext_status": {
            "enabled": bool(do_external),
            "iters": int(len(ext_history)),
            "err": (
                float(ext_history[-1]["err"])
                if ext_history
                else 0.0
            ),
            "converged": bool(ext_converged),
            "history": list(ext_history),
        },
        "meta": meta,
    }
    if bool(cfg.save_data):
        result["saved_paths"] = save_full_external_data(
            output_dir=cfg.save_output_dir,
            element_symbol=str(elem.symbol),
            z=int(z_nuc),
            temperature_ev=float(cfg.temperature_ev),
            rho_g_cc=float(cfg.rho_g_cc),
            suffix=str(cfg.save_suffix),
            result=result,
            metadata=meta,
        )
    return result


__all__ = [
    "FermiHalfEvaluator",
    "ThomasFermiConfig",
    "incomplete_fermi_half_lower",
    "incomplete_fermi_half_upper",
    "solve_thomas_fermi_full_then_external",
]
