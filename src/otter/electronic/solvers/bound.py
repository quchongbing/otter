"""
otter/electronic/solvers/bound.py

Purpose
-------
Sparse Numerov solver for bound states using a generalized eigenproblem.

Methods
-------
- Matrix Numerov discretization on sqrt grids.
- Sparse generalized eigenvalue solve with shift-invert (ARPACK).
- Coarse-grid estimate for sigma (target eigenvalue).
  For sqrt grids, the generalized eigenproblem is non-symmetric.

Equations
---------
Matrix Numerov form (atomic units), sqrt grid (x = sqrt(r)):
  H c = E S c
  H = T + B V,    S = B
  T = -0.5 p A,   p = 0.25 x^-2
  V = V_eff + 3/(32 x^4) + l(l+1)/(2 x^4)
  y(r) = sqrt(r) R(r) = u(x) / sqrt(x)
A,B are tridiagonal Numerov matrices.

References
----------
- :cite:`PillaiGoglioWalker2012`.
- :cite:`StarrettSaumon2014`, Appendix A/B.

The sparse shift-invert tolerances and origin safeguards are Otter numerical
choices, not claims made by either reference.
"""
from typing import Tuple, Optional
import warnings
import numpy as np
from numba import njit

try:
    from scipy import sparse as sp
    from scipy.integrate import quad
    from scipy.optimize import brentq
    from scipy.special import kve
    from scipy.sparse.linalg import eigs
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    sp = None
    quad = None
    brentq = None
    kve = None
    eigs = None


def _zero_tail_outer_reduced_ratio(
    energy: float,
    l: int,
    r_inner: float,
    r_outer: float,
) -> float:
    """Return ``P_l(r_inner) / P_l(r_outer)`` for the decaying free tail.

    ``P_l`` is the reduced radial wave function.  For ``E < 0`` and a
    potential which is zero outside the numerical sphere,

    ``P_l(r) \\propto r k_l(kappa r)``, ``kappa = sqrt(-2 E)``.

    The ratio is evaluated with the exponentially scaled modified Bessel
    function ``kve`` so that the common ``exp(-kappa*r)`` factor does not
    underflow.  This is the non-relativistic form of the negative-energy
    exterior matching condition in Starrett et al. (2019), Eqs. (21)-(22).
    """
    if not _HAVE_SCIPY or kve is None:
        raise ImportError("scipy is required for zero-tail bound-state matching.")
    energy = float(energy)
    r_inner = float(r_inner)
    r_outer = float(r_outer)
    l = int(l)
    if not energy < 0.0:
        raise ValueError("zero-tail bound matching requires energy < 0.")
    if not (0.0 < r_inner < r_outer):
        raise ValueError("Require 0 < r_inner < r_outer for tail matching.")
    if l < 0:
        raise ValueError("l must be non-negative.")

    kappa = float(np.sqrt(-2.0 * energy))
    z_inner = kappa * r_inner
    z_outer = kappa * r_outer
    order = float(l) + 0.5
    scaled_inner = float(kve(order, z_inner))
    scaled_outer = float(kve(order, z_outer))
    ratio = (
        np.sqrt(r_inner / r_outer)
        * np.exp(z_outer - z_inner)
        * scaled_inner
        / scaled_outer
    )
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise FloatingPointError(
            "Could not evaluate the decaying zero-potential bound tail."
        )
    return float(ratio)


@njit(cache=True)
def _propagate_zero_tail_pair(
    x: np.ndarray,
    numerov_f: np.ndarray,
    denom: np.ndarray,
    power: float,
    tail_ratio: float,
    match_index: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Propagate regular/exterior Numerov solutions on one energy shell.

    The recurrence is sequential and dominated the logarithmic threshold
    scout when written as Python loops.  Keeping it in this small compiled
    helper makes per-SCF pole detection inexpensive without changing the
    matching equations.
    """
    u_out = np.zeros_like(x)
    u_out[0] = (x[0] / x[1]) ** power
    u_out[1] = 1.0
    for idx in range(1, match_index + 1):
        u_out[idx + 1] = (
            2.0 * (1.0 + 5.0 * numerov_f[idx] / 12.0) * u_out[idx]
            - denom[idx - 1] * u_out[idx - 1]
        ) / denom[idx + 1]
        scale = max(abs(u_out[idx - 1]), abs(u_out[idx]), abs(u_out[idx + 1]))
        if scale > 1.0e100:
            u_out[: idx + 2] /= scale

    u_in = np.zeros_like(x)
    u_in[-1] = 1.0 / np.sqrt(x[-1])
    u_in[-2] = tail_ratio / np.sqrt(x[-2])
    for idx in range(x.size - 2, match_index - 1, -1):
        u_in[idx - 1] = (
            2.0 * (1.0 + 5.0 * numerov_f[idx] / 12.0) * u_in[idx]
            - denom[idx + 1] * u_in[idx + 1]
        ) / denom[idx - 1]
        scale = max(abs(u_in[idx - 1]), abs(u_in[idx]), abs(u_in[idx + 1]))
        if scale > 1.0e100:
            u_in[idx - 1 :] /= scale

    du_out = (u_out[match_index + 1] - u_out[match_index - 1]) / (
        2.0 * (x[1] - x[0])
    )
    du_in = (u_in[match_index + 1] - u_in[match_index - 1]) / (
        2.0 * (x[1] - x[0])
    )
    return u_out, u_in, du_out, du_in


def zero_tail_bound_matching_residual(
    v_eff: np.ndarray,
    grid_r: np.ndarray,
    grid_dx: float,
    l: int,
    energy: float,
    *,
    match_index: int | None = None,
) -> float:
    """Wronskian residual for an infinite-domain negative-energy state.

    A regular solution is propagated outwards on the uniform ``sqrt(r)``
    Numerov coordinate.  Independently, the analytic exponentially decaying
    solution for ``V_eff=0`` is initialized at the last two grid points and
    propagated inwards.  Their normalized Wronskian at an interior point
    vanishes at a bound eigenenergy.

    Unlike placing a Dirichlet wall at ``grid_r[-1]``, this residual imposes
    the infinite-domain logarithmic derivative.  It is therefore useful for
    frozen-potential checks of very shallow states without constructing a
    radial box several decay lengths long.

    Notes
    -----
    The exterior condition follows Starrett et al., *Computer Physics
    Communications* **235**, 50-62 (2019), Eqs. (21)-(22).  Wilson et al.,
    *JQSRT* **99**, 658 (2006), Appendix A.4, use the same interior/exterior
    matching to demonstrate continuity when a bound state reaches the continuum.

    The caller must ensure that ``V_eff`` is negligible near the outer grid;
    :func:`find_zero_tail_bound_energy` supplies an explicit tail check.
    """
    if not _HAVE_SCIPY:
        raise ImportError("scipy is required for zero-tail bound-state matching.")
    v = np.asarray(v_eff, dtype=float)
    r = np.asarray(grid_r, dtype=float)
    dx = float(grid_dx)
    l = int(l)
    energy = float(energy)
    if v.ndim != 1 or r.ndim != 1 or v.shape != r.shape:
        raise ValueError("v_eff and grid_r must be aligned one-dimensional arrays.")
    if r.size < 9 or np.any(np.diff(r) <= 0.0):
        raise ValueError("grid_r must contain at least nine increasing points.")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(r)):
        raise ValueError("v_eff and grid_r must be finite.")
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("grid_dx must be finite and positive.")
    if l < 0:
        raise ValueError("l must be non-negative.")
    if not np.isfinite(energy) or energy >= 0.0:
        raise ValueError("energy must be finite and negative.")

    x = np.sqrt(r)
    measured_dx = np.diff(x)
    if not np.allclose(measured_dx, dx, rtol=2.0e-5, atol=2.0e-12):
        raise ValueError("zero-tail matching requires a uniform sqrt(r) grid.")
    if match_index is None:
        match_index = int(r.size // 2)
    match_index = int(match_index)
    if match_index < 2 or match_index > r.size - 3:
        raise ValueError("match_index must leave two grid points on each side.")

    # The sparse matrix solver uses u=sqrt(x)*y=P/sqrt(x).  Its transformed
    # equation is u''(x)=8*x^2*(V_corr-E)*u(x), on the uniform x=sqrt(r) grid.
    v_corr = (
        v
        + 3.0 / (32.0 * x**4)
        + float(l) * (float(l) + 1.0) / (2.0 * x**4)
    )
    numerov_f = dx * dx * 8.0 * x * x * (v_corr - energy)
    denom = 1.0 - numerov_f / 12.0
    if np.any(np.abs(denom) < 1.0e-13):
        raise FloatingPointError("Singular Numerov denominator in bound matching.")

    # Set P(R)=1 and obtain the preceding point from the exact free decaying
    # tail.  Converting P to u supplies two values for inward Numerov.
    tail_ratio = _zero_tail_outer_reduced_ratio(
        energy,
        l,
        float(r[-2]),
        float(r[-1]),
    )
    u_out, u_in, du_out, du_in = _propagate_zero_tail_pair(
        x,
        numerov_f,
        denom,
        2.0 * float(l) + 1.5,
        float(tail_ratio),
        int(match_index),
    )
    norm_out = float(np.hypot(u_out[match_index], du_out))
    norm_in = float(np.hypot(u_in[match_index], du_in))
    if norm_out <= 0.0 or norm_in <= 0.0:
        raise FloatingPointError("Degenerate solution in bound matching.")
    residual = (
        u_out[match_index] * du_in - du_out * u_in[match_index]
    ) / (norm_out * norm_in)
    if not np.isfinite(residual):
        raise FloatingPointError("Non-finite zero-tail bound matching residual.")
    return float(residual)


def find_zero_tail_bound_energy(
    v_eff: np.ndarray,
    grid_r: np.ndarray,
    grid_dx: float,
    l: int,
    energy_bracket: tuple[float, float],
    *,
    match_index: int | None = None,
    tail_fraction: float = 0.1,
    tail_v_rel_tol: float | None = 0.1,
    xtol: float = 1.0e-12,
) -> float:
    """Locate one frozen-potential bound level with an analytic zero tail.

    This is deliberately a bracketed diagnostic/refinement API rather than a
    replacement for the production sparse eigensolver.  A supplied bracket
    must isolate one simple zero of :func:`zero_tail_bound_matching_residual`.
    Before solving, the outer potential is checked against the weaker binding
    energy in the bracket.  Set ``tail_v_rel_tol=None`` only for controlled
    diagnostic experiments.
    """
    if not _HAVE_SCIPY or brentq is None:
        raise ImportError("scipy is required for zero-tail bound-state matching.")
    v = np.asarray(v_eff, dtype=float)
    r = np.asarray(grid_r, dtype=float)
    if len(energy_bracket) != 2:
        raise ValueError("energy_bracket must contain two negative energies.")
    e_lo, e_hi = map(float, energy_bracket)
    if not (np.isfinite(e_lo) and np.isfinite(e_hi) and e_lo < e_hi < 0.0):
        raise ValueError("Require a finite bracket e_lo < e_hi < 0.")
    tail_fraction = float(tail_fraction)
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must lie in (0, 1].")
    if v.shape != r.shape or v.ndim != 1:
        raise ValueError("v_eff and grid_r must be aligned one-dimensional arrays.")
    tail_start = float(r[-1]) * (1.0 - tail_fraction)
    tail_mask = r >= tail_start
    tail_max = float(np.max(np.abs(v[tail_mask])))
    tail_ratio = tail_max / max(abs(e_hi), np.finfo(float).tiny)
    if tail_v_rel_tol is not None:
        tol = float(tail_v_rel_tol)
        if not np.isfinite(tol) or tol <= 0.0:
            raise ValueError("tail_v_rel_tol must be positive or None.")
        if tail_ratio > tol:
            raise ValueError(
                "V_eff is not negligible relative to the candidate binding "
                f"energy in the outer tail (ratio={tail_ratio:.3e}, tol={tol:.3e})."
            )

    def residual(energy: float) -> float:
        return zero_tail_bound_matching_residual(
            v,
            r,
            grid_dx,
            l,
            energy,
            match_index=match_index,
        )

    f_lo = residual(e_lo)
    f_hi = residual(e_hi)
    if f_lo == 0.0:
        return float(e_lo)
    if f_hi == 0.0:
        return float(e_hi)
    if np.signbit(f_lo) == np.signbit(f_hi):
        raise ValueError(
            "energy_bracket does not straddle a zero-tail matching root; "
            f"residuals are {f_lo:.6e} and {f_hi:.6e}."
        )
    return float(brentq(residual, e_lo, e_hi, xtol=float(xtol), rtol=4.0 * np.finfo(float).eps))


def solve_zero_tail_bound_state(
    v_eff: np.ndarray,
    grid_r: np.ndarray,
    grid_dx: float,
    l: int,
    energy_bracket: tuple[float, float],
    *,
    match_index: int | None = None,
    tail_fraction: float = 0.1,
    tail_v_rel_tol: float | None = 0.1,
    xtol: float = 1.0e-12,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Return a zero-tail matched orbital normalized over all space.

    This complements :func:`find_zero_tail_bound_energy`: after locating the
    pole, the regular solution propagated from the origin is joined to the
    analytic decaying exterior solution.  The returned
    ``y=sqrt(r) R=P/sqrt(r)`` is normalized with both the sampled interior
    probability and the analytic probability beyond ``grid_r[-1]``.

    The all-space normalization is essential at pressure ionization.  For an
    s state approaching threshold, the probability in every fixed interior
    region vanishes continuously as ``kappa=sqrt(-2E)`` tends to zero.  A
    Dirichlet-box-normalized orbital cannot reproduce that limit when the box
    is shorter than several decay lengths (Wilson et al., JQSRT 99, 658
    (2006), Appendix A.4; Starrett et al., CPC 235, 50--62 (2019),
    Eqs. 21--22).

    Notes
    -----
    This routine is valid only when the sampled outer tail is already
    negligible.  The same explicit tail check as the energy-only API is
    applied before constructing the orbital.
    """
    if not _HAVE_SCIPY or quad is None:
        raise ImportError("scipy is required for zero-tail bound-state matching.")
    v = np.asarray(v_eff, dtype=float)
    r = np.asarray(grid_r, dtype=float)
    dx = float(grid_dx)
    l = int(l)
    energy = find_zero_tail_bound_energy(
        v,
        r,
        dx,
        l,
        energy_bracket,
        match_index=match_index,
        tail_fraction=tail_fraction,
        tail_v_rel_tol=tail_v_rel_tol,
        xtol=xtol,
    )
    x = np.sqrt(r)
    if match_index is None:
        match_index = int(r.size // 2)
    match_index = int(match_index)

    v_corr = (
        v
        + 3.0 / (32.0 * x**4)
        + float(l) * (float(l) + 1.0) / (2.0 * x**4)
    )
    numerov_f = dx * dx * 8.0 * x * x * (v_corr - energy)
    denom = 1.0 - numerov_f / 12.0
    if np.any(np.abs(denom) < 1.0e-13):
        raise FloatingPointError("Singular Numerov denominator in bound matching.")

    tail_ratio = _zero_tail_outer_reduced_ratio(
        energy,
        l,
        float(r[-2]),
        float(r[-1]),
    )
    u_out, u_in, du_out, du_in = _propagate_zero_tail_pair(
        x,
        numerov_f,
        denom,
        2.0 * float(l) + 1.5,
        float(tail_ratio),
        int(match_index),
    )
    pair_in = np.asarray([u_in[match_index], du_in], dtype=float)
    pair_out = np.asarray([u_out[match_index], du_out], dtype=float)
    scale_den = float(np.dot(pair_in, pair_in))
    if not np.isfinite(scale_den) or scale_den <= 0.0:
        raise FloatingPointError("Degenerate inward solution at the match point.")
    scale_in = float(np.dot(pair_in, pair_out) / scale_den)
    u = u_out.copy()
    u[match_index + 1 :] = scale_in * u_in[match_index + 1 :]
    y = u / np.sqrt(x)

    # P=rR=sqrt(r)*y.  The exterior norm is analytic for l=0 and is
    # integrated from the exponentially scaled Bessel ratio for l>0.
    p_edge = float(np.sqrt(r[-1]) * y[-1])
    kappa = float(np.sqrt(-2.0 * energy))
    radial_probability = y * y * r
    interior_norm = float(np.trapezoid(radial_probability, r))
    interior_first_moment = float(np.trapezoid(radial_probability * r, r))
    interior_second_moment = float(np.trapezoid(radial_probability * r * r, r))
    if l == 0:
        exterior_norm = float(p_edge * p_edge / (2.0 * kappa))
        exterior_first_moment = float(
            exterior_norm * (float(r[-1]) + 1.0 / (2.0 * kappa))
        )
        exterior_second_moment = float(
            exterior_norm
            * (
                float(r[-1]) ** 2
                + float(r[-1]) / kappa
                + 1.0 / (2.0 * kappa * kappa)
            )
        )
    else:
        order = float(l) + 0.5
        z_edge = kappa * float(r[-1])
        scaled_edge = float(kve(order, z_edge))

        def relative_p_squared(radius: float) -> float:
            z = kappa * float(radius)
            ratio = (
                np.sqrt(float(radius) / float(r[-1]))
                * np.exp(-(z - z_edge))
                * float(kve(order, z))
                / scaled_edge
            )
            return float(ratio * ratio)

        relative_tail_norm, _ = quad(
            relative_p_squared,
            float(r[-1]),
            np.inf,
            epsabs=1.0e-11,
            epsrel=1.0e-9,
            limit=200,
        )
        exterior_norm = float(p_edge * p_edge * relative_tail_norm)
        relative_tail_first, _ = quad(
            lambda radius: float(radius) * relative_p_squared(radius),
            float(r[-1]),
            np.inf,
            epsabs=1.0e-11,
            epsrel=1.0e-9,
            limit=200,
        )
        relative_tail_second, _ = quad(
            lambda radius: float(radius) ** 2 * relative_p_squared(radius),
            float(r[-1]),
            np.inf,
            epsabs=1.0e-11,
            epsrel=1.0e-9,
            limit=200,
        )
        exterior_first_moment = float(p_edge * p_edge * relative_tail_first)
        exterior_second_moment = float(p_edge * p_edge * relative_tail_second)
    total_norm = float(interior_norm + exterior_norm)
    if not np.isfinite(total_norm) or total_norm <= 0.0:
        raise FloatingPointError("Non-finite all-space bound-state norm.")
    y = y / np.sqrt(total_norm)
    all_space_mean = float(
        (interior_first_moment + exterior_first_moment) / total_norm
    )
    all_space_rms = float(
        np.sqrt(
            max(
                (interior_second_moment + exterior_second_moment) / total_norm,
                0.0,
            )
        )
    )
    return energy, y, {
        "kappa_bohr_inv": kappa,
        "interior_probability": float(interior_norm / total_norm),
        "exterior_probability": float(exterior_norm / total_norm),
        "all_space_mean_radius_bohr": all_space_mean,
        "all_space_rms_radius_bohr": all_space_rms,
        "numeric_box_conditional_mean_radius_bohr": float(
            interior_first_moment / interior_norm
        ),
        "numeric_box_conditional_rms_radius_bohr": float(
            np.sqrt(max(interior_second_moment / interior_norm, 0.0))
        ),
        "match_index": float(match_index),
        "match_radius_bohr": float(r[match_index]),
    }


def find_shallowest_zero_tail_bound_state(
    v_eff: np.ndarray,
    grid_r: np.ndarray,
    grid_dx: float,
    l: int,
    *,
    min_binding: float = 1.0e-10,
    max_binding: float = 1.0e-2,
    n_scan: int = 72,
    match_index: int | None = None,
    tail_fraction: float = 0.1,
    tail_v_rel_tol: float | None = 0.1,
    xtol: float = 1.0e-12,
) -> tuple[float, np.ndarray, dict[str, float]] | None:
    """Scout the shallowest negative-energy zero-tail pole in one channel.

    The logarithmic scan is intentionally confined to a caller-supplied
    near-threshold binding window.  Returning ``None`` means that no *resolved
    sign-changing* Wronskian root was found; callers must not use that outcome
    alone to delete an orbital from a less reliable finite-box solve.
    """
    min_binding = float(min_binding)
    max_binding = float(max_binding)
    n_scan = int(n_scan)
    if not (0.0 < min_binding < max_binding):
        raise ValueError("Require 0 < min_binding < max_binding.")
    if n_scan < 3:
        raise ValueError("n_scan must be at least three.")
    bindings = np.geomspace(min_binding, max_binding, n_scan)
    residual_prev: float | None = None
    energy_prev: float | None = None
    for binding in bindings:
        energy = -float(binding)
        try:
            residual = zero_tail_bound_matching_residual(
                v_eff,
                grid_r,
                grid_dx,
                int(l),
                energy,
                match_index=match_index,
            )
        except (FloatingPointError, ValueError):
            residual_prev = None
            energy_prev = None
            continue
        if residual == 0.0 and energy_prev is not None:
            return solve_zero_tail_bound_state(
                v_eff,
                grid_r,
                grid_dx,
                int(l),
                (float(energy), float(energy_prev)),
                match_index=match_index,
                tail_fraction=tail_fraction,
                tail_v_rel_tol=tail_v_rel_tol,
                xtol=xtol,
            )
        if residual_prev == 0.0 and energy_prev is not None:
            return solve_zero_tail_bound_state(
                v_eff,
                grid_r,
                grid_dx,
                int(l),
                (float(energy), float(energy_prev)),
                match_index=match_index,
                tail_fraction=tail_fraction,
                tail_v_rel_tol=tail_v_rel_tol,
                xtol=xtol,
            )
        if (
            residual_prev is not None
            and energy_prev is not None
            and np.signbit(residual_prev) != np.signbit(residual)
        ):
            bracket = (float(energy), float(energy_prev))
            return solve_zero_tail_bound_state(
                v_eff,
                grid_r,
                grid_dx,
                int(l),
                bracket,
                match_index=match_index,
                tail_fraction=tail_fraction,
                tail_v_rel_tol=tail_v_rel_tol,
                xtol=xtol,
            )
        residual_prev = residual
        energy_prev = energy
    return None


def _solve_single_l_coarse_sqrt(v_eff: np.ndarray,
                                grid_x: np.ndarray,
                                grid_dx: float,
                                l: float,
                                boundary: str = "dirichlet") -> float:
    """
    Coarse sqrt-grid eigenvalue estimate using matrix Numerov.
    """
    if boundary != "dirichlet":
        raise ValueError("Coarse estimate supports only dirichlet boundary.")

    N = v_eff.shape[0]
    dx = grid_dx

    B_main = (10.0 / 12.0) * np.ones(N)
    B_off = (1.0 / 12.0) * np.ones(N - 1)
    A_main = -2.0 * np.ones(N) / (dx ** 2)
    A_off = 1.0 * np.ones(N - 1) / (dx ** 2)

    B = np.diag(B_main) + np.diag(B_off, 1) + np.diag(B_off, -1)
    A = np.diag(A_main) + np.diag(A_off, 1) + np.diag(A_off, -1)

    x = grid_x
    p = 0.25 * x ** -2
    T = -0.5 * np.diag(p) @ A
    v_corr = 3.0 / (32.0 * x ** 4) + l * (l + 1.0) / (2.0 * x ** 4)
    V = np.diag(v_eff + v_corr)

    H = T + B @ V

    if N < 2:
        return -0.5

    eigs = np.linalg.eigvals(np.linalg.solve(B, H))
    eigs = np.sort(eigs.real)
    return float(eigs[0])


def _coarse_sigma_guess(v_eff: np.ndarray,
                        grid_r: np.ndarray,
                        grid_dx: float,
                        l: float,
                        n_coarse: int = 200) -> float:
    """
    Estimate sigma from a coarse grid to accelerate shift-invert.
    """
    N = v_eff.shape[0]
    n_coarse = min(n_coarse, N)
    if N <= n_coarse or n_coarse < 10:
        x_full = np.sqrt(grid_r)
        return _solve_single_l_coarse_sqrt(v_eff, x_full, grid_dx, l)

    x_full = np.sqrt(grid_r)
    x_coarse = np.linspace(x_full[0], x_full[-1], n_coarse)
    v_coarse = np.interp(x_coarse, x_full, v_eff)
    dx_coarse = (x_coarse[-1] - x_coarse[0]) / (n_coarse - 1)
    return _solve_single_l_coarse_sqrt(v_coarse, x_coarse, dx_coarse, l)


def _heuristic_sigma_guess(v_eff: np.ndarray,
                           grid_r: np.ndarray,
                           l: float) -> float:
    """
    Cheap shift-invert target from the diagonal effective potential.

    Notes
    -----
    This estimates the lowest bound-state scale with the diagonal part of the
    sqrt-grid Hamiltonian,

    ``V_eff + 3/(32 x^4) + l(l+1)/(2 x^4)``,

    and avoids the dense coarse-grid generalized eigenproblem that was
    previously evaluated for every l-channel. The sparse solve still falls back
    to the coarse estimator if ARPACK rejects this heuristic shift.
    """
    x = np.sqrt(np.asarray(grid_r, dtype=float))
    inv_x2 = x ** -2
    inv_x4 = inv_x2 ** 2
    v_diag = (
        np.asarray(v_eff, dtype=float)
        + 3.0 / 32.0 * inv_x4
        + float(l) * (float(l) + 1.0) / 2.0 * inv_x4
    )
    sigma = float(np.min(v_diag))
    return sigma if np.isfinite(sigma) else -0.5


def _regularize_origin_series(y: np.ndarray,
                              grid_r: np.ndarray,
                              l: float,
                              n_fix: int = 4,
                              fit_start: int = 4,
                              fit_stop: int = 10,
                              nuclear_charge: float | None = None,
                              core_zr: float = 0.05) -> np.ndarray:
    """
    Regularize the first few bound-state samples with the regular-origin series.

    Parameters
    ----------
    y : ndarray
        Bound-state radial function on the sqrt grid, with
        ``y(r) = sqrt(r) R(r)``.
    grid_r : ndarray
        Radial grid in Bohr.
    l : float
        Angular momentum quantum number.
    n_fix : int, optional
        Number of leading grid points replaced by the fitted regular series.
    fit_start : int, optional
        First index used to fit the small-r series.
    fit_stop : int, optional
        One-past-last index used to fit the small-r series.
    nuclear_charge : float or None, optional
        Nuclear charge used to impose the Coulomb cusp. If omitted, retain
        the historical unconstrained two-term repair.
    core_zr : float, optional
        Dimensionless outer radius ``Z*r`` of the cusp reconstruction.

    Returns
    -------
    ndarray
        Regularized copy of ``y``.

    Notes
    -----
    The physical regular solution obeys

    ``y(r) = r^(l + 1/2) [a0 + a1 r + O(r^2)]``.

    For a finite Hartree/XC core and nuclear potential ``-Z/r``, the radial
    Coulomb cusp fixes ``a1/a0 = -Z/(l+1)``. The constrained path fits higher
    powers just outside the repaired core, then joins the series to the raw
    Numerov vector with a C2 smoothstep.

    The sparse sqrt-grid generalized eigenproblem does not impose an explicit
    origin row, so the first one or two samples can show a small stencil-driven
    foldback even when the bound-state energy is converged. This helper fits the
    leading regular series on the next few interior points and overwrites only
    the first ``n_fix`` samples. The caller renormalizes afterwards.
    """
    y = np.asarray(y, dtype=float)
    r = np.asarray(grid_r, dtype=float)
    n_pts = y.size
    if n_pts < 8:
        return y.copy()

    n_fix = max(0, min(int(n_fix), n_pts))
    fit_lo = max(int(fit_start), n_fix)
    fit_hi = min(int(fit_stop), n_pts)
    if n_fix == 0 or fit_hi - fit_lo < 2:
        return y.copy()

    power = float(l) + 0.5
    if nuclear_charge is not None:
        charge = float(nuclear_charge)
        zr = float(core_zr)
        if not np.isfinite(charge) or charge <= 0.0:
            raise ValueError("nuclear_charge must be finite and positive.")
        if not np.isfinite(zr) or zr <= 0.0:
            raise ValueError("core_zr must be finite and positive.")
        core_radius = zr / charge
        fit_mask = (r >= core_radius) & (r <= 2.0 * core_radius)
        fit_indices = np.flatnonzero(fit_mask)
        if fit_indices.size >= 4:
            scaled_fit = r[fit_indices] / core_radius
            reduced = y[fit_indices] / r[fit_indices] ** power
            cusp_scaled = charge * core_radius / (float(l) + 1.0)
            fit_mat = np.column_stack(
                (
                    1.0 - cusp_scaled * scaled_fit,
                    scaled_fit**2,
                    scaled_fit**3,
                    scaled_fit**4,
                )
            )
            coef, _, _, _ = np.linalg.lstsq(fit_mat, reduced, rcond=None)
            if np.all(np.isfinite(coef)):
                scaled = r / core_radius
                series = (
                    coef[0] * (1.0 - cusp_scaled * scaled)
                    + coef[1] * scaled**2
                    + coef[2] * scaled**3
                    + coef[3] * scaled**4
                )
                reduced_raw = y / r**power
                blend = np.clip((scaled - 0.75) / 0.25, 0.0, 1.0)
                blend = blend**3 * (10.0 - 15.0 * blend + 6.0 * blend**2)
                repaired = np.where(
                    r < core_radius,
                    (1.0 - blend) * series + blend * reduced_raw,
                    reduced_raw,
                )
                return r**power * repaired

    basis0 = r[fit_lo:fit_hi] ** power
    basis1 = basis0 * r[fit_lo:fit_hi]
    fit_mat = np.column_stack((basis0, basis1))
    coef, _, _, _ = np.linalg.lstsq(fit_mat, y[fit_lo:fit_hi], rcond=None)
    if not np.all(np.isfinite(coef)):
        return y.copy()

    y_reg = y.copy()
    basis_fix = r[:n_fix] ** power
    y_reg[:n_fix] = coef[0] * basis_fix + coef[1] * basis_fix * r[:n_fix]
    return y_reg


def _solve_single_l_sparse(v_eff: np.ndarray,
                           grid_r: np.ndarray,
                           grid_dx: float,
                           l: float,
                           n_states: int,
                           boundary: str = "dirichlet",
                           sigma_guess: Optional[float] = None,
                           nuclear_charge: float | None = None,
                           origin_core_zr: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve a single-l channel with sparse Numerov.
    """
    if not _HAVE_SCIPY:
        raise ImportError("scipy is required for sparse Numerov.")
    if boundary != "dirichlet":
        raise ValueError("Sparse Numerov currently supports only dirichlet boundary.")

    N = v_eff.shape[0]
    if N < 5:
        raise ValueError("Grid too small for sparse Numerov.")

    diag_main_A = -2.0 * np.ones(N) / (grid_dx ** 2)
    diag_off_A = 1.0 * np.ones(N - 1) / (grid_dx ** 2)
    diag_main_B = (10.0 / 12.0) * np.ones(N)
    diag_off_B = (1.0 / 12.0) * np.ones(N - 1)

    A_sparse = sp.diags([diag_off_A, diag_main_A, diag_off_A], offsets=[-1, 0, 1], format="csr")
    B_sparse = sp.diags([diag_off_B, diag_main_B, diag_off_B], offsets=[-1, 0, 1], format="csr")

    x = np.sqrt(grid_r)
    p = 0.25 * x ** -2
    T_sparse = -0.5 * sp.diags([p], offsets=[0], format="csr") @ A_sparse
    v_corr = 3.0 / (32.0 * x ** 4) + l * (l + 1.0) / (2.0 * x ** 4)
    V_mat = sp.diags([v_eff + v_corr], offsets=[0], format="csr")
    H_sparse = T_sparse + B_sparse @ V_mat
    S_sparse = B_sparse

    sigma_used = float(sigma_guess) if sigma_guess is not None else _heuristic_sigma_guess(v_eff, grid_r, l)
    if not np.isfinite(sigma_used):
        sigma_used = -0.5

    k_solve = min(max(n_states + 2, 2), N - 2)

    H_use = H_sparse[:-1, :-1]
    S_use = S_sparse[:-1, :-1]

    try:
        vals, vecs = eigs(
            H_use,
            k=k_solve,
            M=S_use,
            sigma=sigma_used,
            which="LM",
            tol=1e-10,
        )
    except Exception as exc:
        if sigma_guess is None:
            sigma_fallback = _coarse_sigma_guess(v_eff, grid_r, grid_dx, l)
            if not np.isfinite(sigma_fallback):
                sigma_fallback = -0.5
            try:
                vals, vecs = eigs(
                    H_use,
                    k=k_solve,
                    M=S_use,
                    sigma=sigma_fallback,
                    which="LM",
                    tol=1e-10,
                )
            except Exception as exc_retry:
                warnings.warn(
                    f"Sparse Numerov failed with heuristic sigma={sigma_used:.6e} "
                    f"and coarse sigma={sigma_fallback:.6e}: {exc_retry}"
                )
                raise
        else:
            warnings.warn(f"Sparse Numerov failed: {exc}")
            raise

    idx = np.argsort(vals)
    vals = np.array(vals[idx], dtype=np.complex128)
    vecs = np.array(vecs[:, idx], dtype=np.complex128)
    if np.max(np.abs(vals.imag)) > 1e-8:
        warnings.warn("Eigenvalues have non-negligible imaginary parts in sqrt-grid solve.")
    vals = vals.real.astype(np.float64)
    vecs = vecs.real.astype(np.float64)

    vals = vals[:n_states]
    vecs = vecs[:, :n_states]

    vecs_pad = np.zeros((N, n_states), dtype=np.float64)
    vecs_pad[:-1, :] = vecs
    vecs = vecs_pad

    x = np.sqrt(grid_r)
    y = vecs / np.sqrt(x[:, None])
    weight = 2.0 * x ** 3
    for i in range(n_states):
        # (1) Remove the small-r stencil foldback by enforcing the regular
        # origin series on only the first few grid samples.
        y[:, i] = _regularize_origin_series(
            y[:, i],
            grid_r,
            l,
            nuclear_charge=nuclear_charge,
            core_zr=origin_core_zr,
        )
        # (2) Renormalize in the physical r-space measure.
        norm = np.sum(y[:, i] ** 2 * weight) * grid_dx
        y[:, i] /= np.sqrt(max(norm, 1e-30))
    vecs = y

    return vals, vecs


def _normalize_n_states_per_l(
    l_list: np.ndarray,
    n_states: int | np.ndarray,
) -> np.ndarray:
    """
    Return per-l bound-state caps as a 1D integer array.

    Parameters
    ----------
    l_list : ndarray
        Angular-momentum channels.
    n_states : int or ndarray
        Either one shared radial-state cap or one cap per l channel.

    Returns
    -------
    ndarray
        Integer radial-state caps with one entry per l in ``l_list``.
    """
    l_arr = np.asarray(l_list, dtype=int)
    if np.isscalar(n_states):
        return np.full(l_arr.shape, max(int(n_states), 1), dtype=int)
    n_arr = np.asarray(n_states, dtype=int)
    if n_arr.ndim != 1 or n_arr.size != l_arr.size:
        raise ValueError("n_states array must have the same length as l_list.")
    if np.any(n_arr < 1):
        raise ValueError("n_states values must be positive.")
    return n_arr


def solve_bound_states_sparse_numerov(v_eff: np.ndarray,
                                      grid_r: np.ndarray,
                                      grid_dx: float,
                                      l_list: np.ndarray,
                                      n_states: int | np.ndarray = 5,
                                      boundary: str = "dirichlet",
                                      n_jobs: Optional[int] = None,
                                      *,
                                      nuclear_charge: float | None = None,
                                      origin_core_zr: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve bound states for multiple l values using sparse Numerov.

    Parameters
    ----------
    v_eff : ndarray
        Effective potential on the grid.
    grid_r : ndarray
        Radial grid points.
    grid_dx : float
        Grid spacing in sqrt-grid coordinate xi=sqrt(r).
    l_list : ndarray
        Angular momenta to solve.
    n_states : int or ndarray
        Number of bound states per l. When an array is supplied it must match
        ``l_list`` and gives one radial-state cap per angular channel.
    boundary : str
        Boundary condition (dirichlet only for now).
    n_jobs : int or None
        Compatibility argument retained for older callers. The bound solver now
        always runs serially.
    nuclear_charge : float or None
        Nuclear charge used by the Coulomb-cusp origin reconstruction. The
        legacy unconstrained repair is retained when omitted.
    origin_core_zr : float
        Dimensionless outer radius ``Z*r`` of the origin reconstruction.

    Returns
    -------
    eigenvalues, eigenvectors

    Notes
    -----
    The solver only returns physically bound levels. For each ``l`` channel,
    any non-negative eigenvalues are discarded and the corresponding output
    slots are padded with ``+inf`` energies and zero wavefunctions. The solver
    also stops launching higher-``l`` channels once the lowest state of one
    channel is already non-bound, because centrifugal barrier growth makes
    still higher channels even less likely to bind.
    """
    v_eff_np = np.asarray(v_eff)
    r_np = np.asarray(grid_r)
    dx = float(grid_dx)
    l_np = np.asarray(l_list, dtype=int)
    n_states_by_l = _normalize_n_states_per_l(l_np, n_states)
    if l_np.size == 0:
        return np.empty((0, 0), dtype=float), np.empty((0, r_np.size, 0), dtype=float)

    n_states_max = int(np.max(n_states_by_l))
    vals = np.full((l_np.size, n_states_max), np.inf, dtype=np.float64)
    vecs = np.zeros((l_np.size, r_np.size, n_states_max), dtype=np.float64)

    stop_higher_l = False
    for idx, lv in enumerate(l_np):
        if stop_higher_l:
            break
        vals_l, vecs_l = _solve_single_l_sparse(
            v_eff_np,
            r_np,
            dx,
            float(lv),
            int(n_states_by_l[idx]),
            boundary,
            nuclear_charge=nuclear_charge,
            origin_core_zr=origin_core_zr,
        )
        n_keep = int(np.sum(np.asarray(vals_l, dtype=float) < 0.0))
        if n_keep > 0:
            vals[idx, :n_keep] = np.asarray(vals_l[:n_keep], dtype=np.float64)
            vecs[idx, :, :n_keep] = np.asarray(vecs_l[:, :n_keep], dtype=np.float64)
        if vals_l.size == 0 or float(vals_l[0]) >= 0.0:
            # For a central attractive potential the lowest energy grows with l,
            # so once the first state is non-bound we stop the remaining higher-l channels.
            stop_higher_l = True
    return vals, vecs
