"""
otter/ionic/qoz.py

Purpose
-------
Reusable one- and multicomponent QOZ building blocks.

Scope
-----
This module owns the core QOZ pieces:
  - jellium response ingredients (`chi0`, `G_ee`, `chi_ee`)
  - effective ion-ion potentials constructed from `n_scr`
  - one- and multicomponent OZ + HNC solvers

Physical references
-------------------
The one-component reduction follows C. E. Starrett and D. Saumon,
High Energy Density Physics 10, 35--42 (2014), Eqs. (14)--(18),
DOI 10.1016/j.hedp.2013.12.001.  The common-response mixture extension
follows C. E. Starrett et al., Physical Review E 90, 033110 (2014),
especially Eq. (20), DOI 10.1103/PhysRevE.90.033110.  Chabrier's
finite-temperature jellium LFC is the production default; model-specific
references and implementation provenance are recorded in
``otter.ionic.lfc``.  The canonical keys are
:cite:`StarrettSaumon2014,StarrettEtAl2014,Mermin1970,Chabrier1990`.

The ``lindhard_fd`` path evaluates the collisionless static
finite-temperature Lindhard response.  Mermin's relaxation-time construction
:cite:`Mermin1970` is cited for the dielectric-response context; this path
does not introduce Mermin's finite collision frequency.

Anderson mixing, matrix-free Newton iterations, adaptive potential
continuation, high-k tapering, and strict DST charge closure are Otter
numerical methods.  They are not algorithms claimed by those papers.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
try:
    from numba import njit
except Exception:  # pragma: no cover - numba is optional at import time
    def njit(*args, **kwargs):  # type: ignore[misc]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(func):
            return func

        return _decorator

from otter.electronic.continuum.ideal import ideal_unbound_density
from otter.electronic.continuum.tail import tail_parameters
from otter.numerics.transforms import (
    RadialTransform,
    dst_lattice_zero_moment,
    radial_forward,
    radial_inverse,
)
from otter.ionic.lfc import cee_from_gee, gee_jellium
from otter.literature import (
    CitationMixin,
    citation_keys_for_chi0_model,
    citation_keys_for_lfc_model,
)


@dataclass(frozen=True)
class QOZResponseOptions(CitationMixin):
    """
    Controls for the electron-response part of the QOZ chain.

    These parameters enter Eq.(17)/(18)-style constructions through:
      - the non-interacting response `chi0`
      - the electron-electron local field correction `G_ee`

    ``lfc_model="chabrier1990"`` selects Chabrier's finite-temperature
    extension of the full Utsumi--Ichimaru form.  It is distinct from the
    simpler ``"chabrier_hubbard"`` comparison form retained for diagnostics.
    """

    chi0_model: str = "lindhard_fd"
    lfc_model: str = "chabrier1990"
    electron_temperature_ha: float | None = None
    mu_jellium_override_ha: float | None = None
    lindhard_p_points: int = 4096
    lindhard_p_max_mult: float = 8.0
    lindhard_p_max_extra: float = 20.0

    @property
    def citation_keys(self) -> tuple[str, ...]:
        return (
            "StarrettSaumon2014",
            "StarrettEtAl2014",
            *citation_keys_for_chi0_model(self.chi0_model),
            *citation_keys_for_lfc_model(self.lfc_model),
        )


@dataclass(frozen=True)
class QOZPotentialOptions(CitationMixin):
    """
    Controls for building the effective ion-ion potential from `n_scr`.

    The default values match the currently working test configuration.
    """

    response: QOZResponseOptions = QOZResponseOptions()
    high_k_taper_start_frac: float | None = 0.9

    @property
    def citation_keys(self) -> tuple[str, ...]:
        return tuple(self.response.citation_keys)


@dataclass(frozen=True)
class EffectivePotentialResult:
    """
    Container for the Starrett-style QOZ potential assembly.

    Exposing the intermediate arrays keeps the API usable for diagnostics and
    later development without forcing users back into the test script.
    """

    vii_r: np.ndarray
    vii_k: np.ndarray
    v_ie_k: np.ndarray
    v_ee_k: np.ndarray
    c_ie_k: np.ndarray
    c_ee_k: np.ndarray
    chi_ee_k: np.ndarray
    chi0_k: np.ndarray
    gee_k: np.ndarray
    n_scr_k: np.ndarray
    mu_jellium_ha: float
    nbar_e0: float


@dataclass(frozen=True)
class ScreeningChargeConsistencyResult:
    """
    Screening-density profile after optional `Q_scr -> Zbar` enforcement.

    The reduced QOZ mapping is especially sensitive to the low-k content of
    `n_scr(k)`.  This helper keeps native-grid interpolation and discrete-grid
    closure explicit and records both trapezoidal and DST diagnostics.
    """

    n_scr_r: np.ndarray
    q_scr_raw: float
    q_scr_used: float
    q_scr_rel: float
    normalization_measure: str = "trapezoid"
    scale_factor: float = 1.0
    q_scr_trapz_raw: float = np.nan
    q_scr_trapz_used: float = np.nan
    q_scr_dst_raw: float = np.nan
    q_scr_dst_used: float = np.nan


@dataclass(frozen=True)
class MultiComponentScreeningChargeConsistencyResult:
    """
    Per-species screening-charge diagnostics for a multicomponent mixture.

    The reduced mixture-QOZ mapping uses one screening cloud per species. This
    helper applies the same scalar `Q_scr -> Zbar` correction independently to
    each species and stores the corresponding before/after charge checks.
    """

    n_scr_r: np.ndarray
    q_scr_raw: np.ndarray
    q_scr_used: np.ndarray
    q_scr_rel: np.ndarray
    normalization_measure: str = "trapezoid"
    scale_factor: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    q_scr_trapz_raw: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    q_scr_trapz_used: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    q_scr_dst_raw: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    q_scr_dst_used: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))


@dataclass(frozen=True)
class MultiComponentEffectivePotentialResult:
    """
    Container for the multicomponent Starrett-style QOZ potential assembly.

    Arrays use the axis order `(species_i, species_j, grid)` for pair
    quantities and `(species, grid)` for one-body screening quantities.
    """

    vij_r: np.ndarray
    vij_k: np.ndarray
    v_ie_k: np.ndarray
    v_ee_k: np.ndarray
    c_ie_k: np.ndarray
    c_ee_k: np.ndarray
    chi_ee_k: np.ndarray
    chi0_k: np.ndarray
    gee_k: np.ndarray
    n_scr_k: np.ndarray
    mu_jellium_ha: float
    nbar_e0: float
    n_i_species: np.ndarray
    zbar_species: np.ndarray


@njit(cache=True)
def _high_k_cosine_taper_numba(k_arr: np.ndarray, start_frac: float) -> np.ndarray:
    """
    Return one cosine taper that smoothly damps only the highest-k tail.

    Notes
    -----
    The strict DST lattice makes the `V(k) -> V(r)` inverse transform very
    efficient, but finite `k_max` can leave a visible Nyquist-scale ringing in
    `V(r)` and, downstream, in `g(r)`. A gentle cosine taper on only the last
    few percent of the reciprocal grid suppresses that artifact without
    materially changing the physically relevant low-k content.
    """
    out = np.ones_like(k_arr)
    if k_arr.size == 0:
        return out
    frac = min(max(float(start_frac), 0.0), 1.0)
    if frac >= 1.0:
        return out
    k_max = max(float(k_arr[-1]), 1.0e-12)
    k_start = frac * k_max
    width = max(k_max - k_start, 1.0e-12)
    for idx in range(k_arr.size):
        kv = float(k_arr[idx])
        if kv <= k_start:
            out[idx] = 1.0
        else:
            x = (kv - k_start) / width
            if x >= 1.0:
                out[idx] = 0.0
            elif x <= 0.0:
                out[idx] = 1.0
            else:
                out[idx] = 0.5 * (1.0 + np.cos(np.pi * x))
    return out


def radial_charge_trapezoid(r: np.ndarray, density_r: np.ndarray) -> float | np.ndarray:
    """Integrate one or more spherical number densities on their native grid."""
    r_arr = np.asarray(r, dtype=float)
    density_arr = np.asarray(density_r, dtype=float)
    if r_arr.ndim != 1 or density_arr.ndim == 0 or density_arr.shape[-1] != r_arr.size:
        raise ValueError("density_r must end in the one-dimensional r grid.")
    return 4.0 * np.pi * np.trapezoid((r_arr**2) * density_arr, r_arr, axis=-1)


def qoz_zbar_from_nscr(
    r: np.ndarray,
    n_scr: np.ndarray,
    *,
    partition_zbar: float | np.ndarray | None = None,
    electronic_zbar: float | np.ndarray | None = None,
    mode: str = "pseudoatom_partition",
) -> float | np.ndarray:
    r"""Select the ionic charge used by the reduced QOZ mapping.

    ``pseudoatom_partition`` is the production choice.  It enforces the joint
    Starrett--Saumon identities (2013, Eqs. 73--76; 2014, Eqs. 9, 10, 15)

    .. math::

       \int n_e^{\rm PA}\,d^3r=Z,\qquad
       n_e^{\rm scr}=n_e^{\rm PA}-n_e^{\rm ion},

    and therefore uses

    .. math:: \bar Z = Z-\int n_e^{\rm ion}(r)\,d^3r.

    The downstream charge-closure step then normalizes the screening cloud on
    the *actual DST lattice* so that Eq. (15),

    .. math:: \bar Z = \int n_e^{\rm scr}(r)\,d^3r.

    is also satisfied numerically.  This distinction matters when a finite
    full/external box leaves a small pseudoatom-neutrality residual.

    ``screening_integral`` returns the uncorrected native-grid integral.  It
    remains useful for auditing that residual, but treating it as the physical
    charge would silently accept a non-neutral numerical pseudoatom.

    ``electronic`` retains the former behavior and is provided only for
    controlled comparisons with existing results.  The electronic value is
    commonly the AA diagnostic ``Z - Q_ion(R_WS)`` and must not be silently
    conflated with the QOZ/TCP definition above.
    """
    mode_key = str(mode).strip().lower().replace("-", "_")
    if mode_key in {
        "pseudoatom_partition",
        "partition",
        "pa_partition",
        "z_minus_qion_all",
    }:
        if partition_zbar is None:
            raise ValueError(
                "partition_zbar is required when qoz zbar mode is "
                "'pseudoatom_partition'."
            )
        value = np.asarray(partition_zbar, dtype=float)
    elif mode_key in {"screening_integral", "nscr_integral", "eq15_raw"}:
        value = np.asarray(radial_charge_trapezoid(r, n_scr), dtype=float)
    elif mode_key in {"electronic", "aa", "legacy"}:
        if electronic_zbar is None:
            raise ValueError("electronic_zbar is required when qoz zbar mode is 'electronic'.")
        value = np.asarray(electronic_zbar, dtype=float)
    else:
        raise ValueError(
            "qoz zbar mode must be 'pseudoatom_partition', "
            "'screening_integral', or 'electronic'."
        )
    if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError("The selected QOZ screening charge must be finite and positive.")
    return float(value) if value.ndim == 0 else value


def enforce_screening_charge_consistency(
    r: np.ndarray,
    n_scr: np.ndarray,
    *,
    zbar: float,
    renormalize: bool = True,
    transform: RadialTransform | None = None,
) -> ScreeningChargeConsistencyResult:
    """
    Optionally rescale `n_scr(r)` so its integrated screening charge matches `Zbar`.

    Parameters
    ----------
    r
        Real-space grid in Bohr.
    n_scr
        Screening density on `r` in Bohr^-3.
    zbar
        Net ionic charge used by the reduced QOZ mapping.
    renormalize
        If True, scale `n_scr` so its selected discrete charge is `Zbar`.
    transform
        Optional strict DST lattice.  When supplied, the normalization uses
        the DST's own discrete zero moment.  When omitted, the legacy/native
        trapezoidal integral is retained for backward-compatible utilities.

    Returns
    -------
    ScreeningChargeConsistencyResult
        Corrected profile together with the raw charge, the final charge after
        any rescaling, and the raw relative mismatch against `Zbar`.

    Notes
    -----
    This helper does not invent new tail structure.  In production the target
    is the all-space pseudoatom partition ``Z-Q_ion(all)``; this final scalar
    factor repairs any pseudoatom-neutrality residual together with
    interpolation and strict-DST quadrature differences.  The raw trap and DST
    charges and the applied scale remain explicit diagnostics.
    """
    r = np.asarray(r, dtype=float)
    n_scr_arr = np.asarray(n_scr, dtype=float)
    if r.shape != n_scr_arr.shape:
        raise ValueError("r and n_scr must have the same shape.")

    if transform is not None:
        if transform.r.shape != r.shape or not np.allclose(
            transform.r,
            r,
            rtol=1.0e-12,
            atol=max(1.0e-14, abs(float(transform.dr)) * 1.0e-12),
        ):
            raise ValueError("r must match transform.r when a DST transform is supplied.")

    if not np.isfinite(float(zbar)) or float(zbar) <= 0.0:
        raise ValueError("zbar must be finite and positive for screening-charge closure.")
    zbar_val = max(abs(float(zbar)), 1.0e-12)
    q_trap_raw = float(radial_charge_trapezoid(r, n_scr_arr))
    q_dst_raw = (
        float(dst_lattice_zero_moment(n_scr_arr, transform))
        if transform is not None
        else np.nan
    )
    normalization_measure = "dst_zero_moment" if transform is not None else "trapezoid"
    q_scr_raw = q_dst_raw if transform is not None else q_trap_raw
    q_scr_rel = abs(float(q_scr_raw) - float(zbar)) / zbar_val

    n_scr_use = np.asarray(n_scr_arr, dtype=float).copy()
    scale_factor = 1.0
    if renormalize:
        if not np.isfinite(q_scr_raw) or abs(q_scr_raw) <= 1.0e-12:
            raise ValueError(
                "Cannot normalize n_scr to zbar: its selected discrete charge "
                f"is {q_scr_raw!r}."
            )
        scale_factor = float(zbar) / float(q_scr_raw)
        n_scr_use *= scale_factor

    q_trap_used = float(radial_charge_trapezoid(r, n_scr_use))
    q_dst_used = (
        float(dst_lattice_zero_moment(n_scr_use, transform))
        if transform is not None
        else np.nan
    )
    q_scr_used = q_dst_used if transform is not None else q_trap_used
    return ScreeningChargeConsistencyResult(
        n_scr_r=np.asarray(n_scr_use, dtype=float),
        q_scr_raw=float(q_scr_raw),
        q_scr_used=float(q_scr_used),
        q_scr_rel=float(q_scr_rel),
        normalization_measure=str(normalization_measure),
        scale_factor=float(scale_factor),
        q_scr_trapz_raw=float(q_trap_raw),
        q_scr_trapz_used=float(q_trap_used),
        q_scr_dst_raw=float(q_dst_raw),
        q_scr_dst_used=float(q_dst_used),
    )


def enforce_screening_charge_consistency_many(
    r: np.ndarray,
    n_scr: np.ndarray,
    *,
    zbar: np.ndarray,
    renormalize: bool = True,
    transform: RadialTransform | None = None,
) -> MultiComponentScreeningChargeConsistencyResult:
    """
    Apply per-species `Q_scr -> Zbar` enforcement for a mixture.

    Parameters
    ----------
    r
        Shared real-space grid in Bohr.
    n_scr
        Screening densities with shape `(n_species, n_r)` in Bohr^-3.
    zbar
        Net ionic charges for each species.
    renormalize
        If True, scale each species screening cloud independently so that
        `4*pi*int r^2 n_scr_i(r) dr = Zbar_i`.

    Returns
    -------
    MultiComponentScreeningChargeConsistencyResult
        Corrected profiles and one scalar charge diagnostic per species.
    """
    r_arr = np.asarray(r, dtype=float)
    n_scr_arr = np.asarray(n_scr, dtype=float)
    zbar_arr = np.asarray(zbar, dtype=float)
    if n_scr_arr.ndim != 2:
        raise ValueError("n_scr must have shape (n_species, n_r).")
    if zbar_arr.shape != (n_scr_arr.shape[0],):
        raise ValueError("zbar must have one entry per species.")

    n_fix = np.zeros_like(n_scr_arr)
    q_raw = np.zeros(n_scr_arr.shape[0], dtype=float)
    q_use = np.zeros(n_scr_arr.shape[0], dtype=float)
    q_rel = np.zeros(n_scr_arr.shape[0], dtype=float)
    scale = np.ones(n_scr_arr.shape[0], dtype=float)
    q_trap_raw = np.zeros(n_scr_arr.shape[0], dtype=float)
    q_trap_use = np.zeros(n_scr_arr.shape[0], dtype=float)
    q_dst_raw = np.full(n_scr_arr.shape[0], np.nan, dtype=float)
    q_dst_use = np.full(n_scr_arr.shape[0], np.nan, dtype=float)
    for idx in range(n_scr_arr.shape[0]):
        one = enforce_screening_charge_consistency(
            r_arr,
            n_scr_arr[idx],
            zbar=float(zbar_arr[idx]),
            renormalize=bool(renormalize),
            transform=transform,
        )
        n_fix[idx] = one.n_scr_r
        q_raw[idx] = float(one.q_scr_raw)
        q_use[idx] = float(one.q_scr_used)
        q_rel[idx] = float(one.q_scr_rel)
        scale[idx] = float(one.scale_factor)
        q_trap_raw[idx] = float(one.q_scr_trapz_raw)
        q_trap_use[idx] = float(one.q_scr_trapz_used)
        q_dst_raw[idx] = float(one.q_scr_dst_raw)
        q_dst_use[idx] = float(one.q_scr_dst_used)
    return MultiComponentScreeningChargeConsistencyResult(
        n_scr_r=n_fix,
        q_scr_raw=q_raw,
        q_scr_used=q_use,
        q_scr_rel=q_rel,
        normalization_measure=("dst_zero_moment" if transform is not None else "trapezoid"),
        scale_factor=scale,
        q_scr_trapz_raw=q_trap_raw,
        q_scr_trapz_used=q_trap_use,
        q_scr_dst_raw=q_dst_raw,
        q_scr_dst_used=q_dst_use,
    )


def mu_jellium_from_nbar(nbar_e0: float, temperature_ha: float) -> float:
    """
    Infer the jellium chemical potential from the average free-electron density.

    Starrett's QOZ response functions use the average density

      nbar_e0 = n_I^0 * Zbar

    for a one-component plasma, and

      nbar_e0 = sum_i n_i^0 * Zbar_i = n_mix * sum_i x_i Zbar_i

    for a mixture, rather than any species-local AA quantity such as `n_e^0`
    at the ion-sphere boundary or center. This helper inverts the ideal
    finite-T density relation once, so all subsequent response functions are
    built from the correct jellium reference state.
    """
    nbar = max(float(nbar_e0), 1e-16)
    temperature = max(float(temperature_ha), 1e-12)

    def _density_mismatch(mu: float) -> float:
        return ideal_unbound_density(mu, temperature, method="exact") - nbar

    # Use a free-electron estimate as the center of the search bracket, then
    # enlarge the interval until the finite-T density mismatch changes sign.
    # This keeps the inversion robust over both degenerate and classical cases.
    kf = (3.0 * np.pi**2 * nbar) ** (1.0 / 3.0)
    ef = 0.5 * kf * kf
    mu_lo = -50.0 * temperature - 10.0
    mu_hi = max(ef + 50.0 * temperature + 10.0, 10.0)
    f_lo = _density_mismatch(mu_lo)
    f_hi = _density_mismatch(mu_hi)
    n_expand = 0
    while f_lo * f_hi > 0.0 and n_expand < 20:
        mu_lo -= 20.0 * temperature + 5.0
        mu_hi += 20.0 * temperature + 5.0
        f_lo = _density_mismatch(mu_lo)
        f_hi = _density_mismatch(mu_hi)
        n_expand += 1
    if f_lo * f_hi > 0.0:
        return float(ef)

    from scipy.optimize import brentq

    return float(brentq(_density_mismatch, mu_lo, mu_hi, xtol=1e-10, rtol=1e-10, maxiter=200))


def chi0_lindhard_finite_t(
    k: np.ndarray,
    mu_ha: float,
    temperature_ha: float,
    *,
    spin_deg: float = 2.0,
    p_points: int = 4096,
    p_max_mult: float = 8.0,
    p_max_extra: float = 20.0,
) -> np.ndarray:
    """
    Static finite-temperature Lindhard response for a 3D uniform electron gas.

      chi0(k) = -(g_s/(2*pi^2*k)) * integral_0^inf
                [ p f(p) ln|(k+2p)/(k-2p)| ] dp

    The singular point p = k/2 is handled with a principal-value-friendly
    change of variables on both sides of the logarithmic kernel. The k -> 0
    limit is replaced by -dn/dmu from a finite-difference derivative of the
    ideal-gas density.

    References
    ----------
    N. D. Mermin, Phys. Rev. B 1, 2362 (1970),
    DOI 10.1103/PhysRevB.1.2362.  That paper gives the conserving
    relaxation-time construction based on the Lindhard dielectric function.
    This routine evaluates only its collisionless, static non-interacting
    input and does not implement a finite relaxation rate.
    """
    k_arr = np.ascontiguousarray(np.asarray(k, dtype=np.float64))
    if np.any(k_arr < 0.0):
        raise ValueError("k must be non-negative.")

    mu = float(mu_ha)
    temperature = max(float(temperature_ha), 1e-12)
    g_s = float(spin_deg)
    if g_s <= 0.0:
        raise ValueError("spin_deg must be positive.")

    n_side = max(int(p_points) // 2, 256)

    p_char = np.sqrt(max(2.0 * max(mu, 0.0), 1e-12))
    p_max = max(float(p_max_mult) * max(p_char, np.sqrt(2.0 * temperature)), float(p_max_extra))
    if k_arr.size > 0:
        p_max = max(p_max, 0.5 * float(np.max(k_arr)) + 2.0)

    eps_tail = max(mu + 35.0 * temperature, 35.0 * temperature)
    eps_tail = max(eps_tail, 1.0)
    p_max_n = max(p_max, np.sqrt(2.0 * eps_tail))
    n_p_n = max(int(p_points), 2048)
    tiny = 1e-14
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    def _fd_scalar(eps: float, inv_t: float) -> float:
        x = (eps - mu) * inv_t
        if x > 80.0:
            return 0.0
        if x < -80.0:
            return 1.0
        ex = np.exp(x)
        return 1.0 / (1.0 + ex)

    def _density_from_mu(mu_val: float, t_val: float, g_val: float, p_max_val: float, n_p_val: int) -> float:
        inv_t = 1.0 / t_val
        n_p_val = max(int(n_p_val), 2)
        dp = p_max_val / (n_p_val - 1)
        integ = 0.0
        val_prev = 0.0
        for j in range(1, n_p_val):
            p = j * dp
            x = (0.5 * p * p - mu_val) * inv_t
            if x > 80.0:
                f = 0.0
            elif x < -80.0:
                f = 1.0
            else:
                ex = np.exp(x)
                f = 1.0 / (1.0 + ex)
            val = p * p * f
            integ += 0.5 * (val_prev + val) * dp
            val_prev = val
        return g_val * integ / (2.0 * np.pi**2)

    def _chi0_k0(mu_val: float, t_val: float, g_val: float, p_max_val: float, n_p_val: int) -> float:
        dmu = max(1e-6, 1e-4 * max(abs(mu_val), 1.0))
        dmu = max(dmu, 5e-3 * t_val)
        n_plus = _density_from_mu(mu_val + dmu, t_val, g_val, p_max_val, n_p_val)
        n_minus = _density_from_mu(mu_val - dmu, t_val, g_val, p_max_val, n_p_val)
        return -(n_plus - n_minus) / (2.0 * dmu)

    def _one_k(kv: float, inv_t: float, pref_common: float, small_k: float, chi0_0_val: float) -> float:
        if kv <= small_k or kv <= tiny:
            return chi0_0_val

        p0 = 0.5 * kv

        if p0 <= tiny or p0 >= p_max - tiny:
            n_p = max(2 * n_side, 512)
            p_min = 1e-10
            if p_max <= p_min:
                return chi0_0_val
            p_grid = np.linspace(p_min, p_max, n_p)
            eps = 0.5 * p_grid * p_grid
            x = np.clip((eps - mu) * inv_t, -80.0, 80.0)
            f = 1.0 / (1.0 + np.exp(x))
            den = np.maximum(np.abs(kv - 2.0 * p_grid), tiny)
            vals = p_grid * f * np.log((kv + 2.0 * p_grid) / den)
            return pref_common * trapz(vals, p_grid) / kv

        u_grid = np.linspace(0.0, np.sqrt(p0), n_side)
        uu = np.maximum(u_grid * u_grid, tiny)
        p_left = np.maximum(p0 - uu, 0.0)
        eps_left = 0.5 * p_left * p_left
        x_left = np.clip((eps_left - mu) * inv_t, -80.0, 80.0)
        f_left = 1.0 / (1.0 + np.exp(x_left))
        ratio_left = np.maximum((2.0 * p0 - uu) / uu, tiny)
        vals_left = 2.0 * u_grid * p_left * f_left * np.log(ratio_left)
        left = trapz(vals_left, u_grid)

        v_grid = np.linspace(0.0, np.sqrt(p_max - p0), n_side)
        vv = np.maximum(v_grid * v_grid, tiny)
        p_right = p0 + vv
        eps_right = 0.5 * p_right * p_right
        x_right = np.clip((eps_right - mu) * inv_t, -80.0, 80.0)
        f_right = 1.0 / (1.0 + np.exp(x_right))
        ratio_right = np.maximum((2.0 * p0 + vv) / vv, tiny)
        vals_right = 2.0 * v_grid * p_right * f_right * np.log(ratio_right)
        right = trapz(vals_right, v_grid)

        return pref_common * (left + right) / kv

    chi0_0 = _chi0_k0(mu, temperature, g_s, float(p_max_n), int(n_p_n))

    p_scale = np.sqrt(max(2.0 * max(mu, 0.0), 2.0 * temperature, 1e-12))
    small_k = max(1e-7, 1e-3 * p_scale)
    inv_t = 1.0 / temperature
    pref_common = -(g_s / (2.0 * np.pi**2))

    out = np.empty_like(k_arr)
    for i, kv in enumerate(k_arr):
        out[i] = _one_k(float(kv), inv_t, pref_common, small_k, chi0_0)
    return out


def chi_ee_from_eq17(
    k: np.ndarray,
    *,
    nbar_e0: float,
    electron_temperature_ha: float,
    mu_jellium_ha: float,
    response_options: QOZResponseOptions | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build `chi_ee(k)` from Starrett--Saumon (2014), Eqs. (17)--(18).

    The implementation is intentionally split into the same three logical
    stages used in the paper:

      1) choose chi0(k)
      2) choose G_ee(k)
      3) build C_ee(k) from G_ee(k), then assemble chi_ee(k)

      chi_ee = chi0 / (1 + chi0 * C_ee / beta)
      C_ee   = -beta * (4*pi/k^2) * (1 - G_ee)

    Returns
    -------
    chi_ee, chi0, gee
    """
    opts = response_options or QOZResponseOptions()
    k = np.asarray(k, dtype=float)
    te_ha = max(float(electron_temperature_ha), 1e-12)
    beta = 1.0 / te_ha

    # Step 1: choose the non-interacting response model chi0(k).
    if str(opts.chi0_model).lower().strip() == "lindhard_fd":
        chi0 = chi0_lindhard_finite_t(
            k,
            mu_jellium_ha,
            te_ha,
            p_points=opts.lindhard_p_points,
            p_max_mult=opts.lindhard_p_max_mult,
            p_max_extra=opts.lindhard_p_max_extra,
        )
    else:
        _, _, k_tf = tail_parameters(mu_jellium_ha, te_ha)
        chi0 = -max(k_tf * k_tf, 1e-14) / (4.0 * np.pi) * np.ones_like(k)

    # Step 2: choose the electron-electron local field correction G_ee(k).
    gee = gee_jellium(
        k,
        nbar_e0,
        model=opts.lfc_model,
        t_e_ha=te_ha,
    )

    # Step 3: build C_ee(k) explicitly from G_ee(k), then assemble chi_ee(k).
    # The QOZ/DST lattice starts at the first non-zero k mode, so k=0 is not
    # part of the production path; `cee_from_gee` still handles it defensively.
    c_ee = cee_from_gee(k, gee, te_ha)
    denom = 1.0 + chi0 * c_ee / beta
    small = np.abs(denom) < 1e-10
    if np.any(small):
        denom = denom.copy()
        denom[small] = np.where(denom[small] >= 0.0, 1e-10, -1e-10)
    chi_ee = chi0 / denom
    chi_ee = np.minimum(chi_ee, -1e-12)
    return chi_ee, chi0, gee


def electron_interaction_channels_k(
    *,
    k: np.ndarray,
    n_scr_k: np.ndarray,
    chi_ee_k: np.ndarray,
    gee_k: np.ndarray,
    electron_temperature_ha: float,
    ion_temperature_ha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Return the electron--ion and electron--electron QOZ channels.

    The definitions follow Starrett--Saumon (2014), Eqs. (16)--(18):

    .. math::

       V_{Ie}(k) &= \frac{n_e^{\rm scr}(k)}{\chi_{ee}(k)},
       &C_{Ie}(k) &= -\beta_i V_{Ie}(k),\\
       V_{ee}(k) &= \frac{4\pi}{k^2}[1-G_{ee}(k)],
       &C_{ee}(k) &= -\beta_e V_{ee}(k).

    ``n_scr_k`` may be one-dimensional for a single species or have shape
    ``(n_species, n_k)`` for a mixture.  The returned ``V_Ie`` and ``C_Ie``
    preserve that shape; the common electron channels are one-dimensional.

    Returns
    -------
    v_ie_k, v_ee_k, c_ie_k, c_ee_k
        Fourier-space potentials in Hartree Bohr^3 and direct-correlation
        functions in Bohr^3.
    """
    k_arr = np.asarray(k, dtype=float)
    n_scr_arr = np.asarray(n_scr_k, dtype=float)
    chi_arr = np.asarray(chi_ee_k, dtype=float)
    gee_arr = np.asarray(gee_k, dtype=float)
    if k_arr.ndim != 1 or chi_arr.shape != k_arr.shape or gee_arr.shape != k_arr.shape:
        raise ValueError("k, chi_ee_k, and gee_k must share one reciprocal grid.")
    if n_scr_arr.ndim not in (1, 2) or n_scr_arr.shape[-1] != k_arr.size:
        raise ValueError("n_scr_k must have shape (n_k,) or (n_species, n_k).")

    te_ha = max(float(electron_temperature_ha), 1.0e-12)
    ti_ha = max(float(ion_temperature_ha), 1.0e-12)
    c_ee_k = cee_from_gee(k_arr, gee_arr, te_ha)
    v_ee_k = -te_ha * c_ee_k
    v_ie_k = n_scr_arr / chi_arr
    c_ie_k = -v_ie_k / ti_ha
    return v_ie_k, v_ee_k, c_ie_k, c_ee_k


def _stable_effective_pair_potential_k(
    *,
    n_scr_k: np.ndarray,
    zbar: np.ndarray,
    k: np.ndarray,
    chi0_k: np.ndarray,
    gee_k: np.ndarray,
) -> np.ndarray:
    r"""Assemble Eq. (14) without subtracting two divergent Coulomb terms.

    With ``q_i(k) = n_scr_i(k)`` and ``d_i = q_i - Z_i``, the usual reduced
    QOZ expression can be rearranged exactly as

    .. math::

       V_{ij} = v_c[-Z_i d_j-Z_j d_i-d_i d_j+G q_iq_j]
                + q_iq_j/\chi_0.

    Charge closure makes ``d_i = O(k**2)``.  This form therefore avoids the
    former direct subtraction of two separately ``O(1/k**2)`` quantities,
    which magnified roundoff in the first reciprocal-space modes.
    """
    charge_remainder, lfc_term, chi0_term = decompose_effective_pair_potential_k(
        n_scr_k=n_scr_k,
        zbar=zbar,
        k=k,
        chi0_k=chi0_k,
        gee_k=gee_k,
    )
    return charge_remainder + lfc_term + chi0_term


def decompose_effective_pair_potential_k(
    *,
    n_scr_k: np.ndarray,
    zbar: np.ndarray,
    k: np.ndarray,
    chi0_k: np.ndarray,
    gee_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return the three finite pieces of the stable Starrett potential.

    With ``q_i=n_scr_i`` and ``d_i=q_i-Z_i``, this returns

    .. math::

       V_{\rm charge} &= v_c[-Z_i d_j-Z_j d_i-d_i d_j],\\
       V_{\rm LFC}    &= v_c G q_iq_j,\\
       V_{\chi_0}    &= q_iq_j/\chi_0.

    Their sum is exactly :func:`_stable_effective_pair_potential_k`. Keeping
    the pieces finite permits low-``k`` attribution for both one- and
    multicomponent calculations without reintroducing the subtraction of two
    separately divergent Coulomb terms.
    """
    q = np.asarray(n_scr_k, dtype=float)
    z = np.asarray(zbar, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    chi0 = np.asarray(chi0_k, dtype=float)
    gee = np.asarray(gee_k, dtype=float)
    if q.ndim != 2 or z.shape != (q.shape[0],):
        raise ValueError("n_scr_k must have shape (n_species, n_k) with matching zbar.")
    if q.shape[1] != k_arr.size or chi0.shape != k_arr.shape or gee.shape != k_arr.shape:
        raise ValueError("k, chi0_k, gee_k, and n_scr_k must share the reciprocal grid.")

    k2 = np.maximum(k_arr**2, 1.0e-24)
    v_c = 4.0 * np.pi / k2
    delta = q - z[:, np.newaxis]
    q_i = q[:, np.newaxis, :]
    q_j = q[np.newaxis, :, :]
    d_i = delta[:, np.newaxis, :]
    d_j = delta[np.newaxis, :, :]
    z_i = z[:, np.newaxis, np.newaxis]
    z_j = z[np.newaxis, :, np.newaxis]
    charge_remainder = v_c * (-z_i * d_j - z_j * d_i - d_i * d_j)
    lfc_term = v_c * gee * q_i * q_j
    chi0_term = (q_i * q_j) / chi0
    return charge_remainder, lfc_term, chi0_term


def build_effective_vii_from_nscr(
    r: np.ndarray,
    n_scr: np.ndarray,
    zbar: float,
    n_i: float,
    ion_temperature_ha: float,
    k: np.ndarray,
    transform: RadialTransform,
    options: QOZPotentialOptions | None = None,
) -> EffectivePotentialResult:
    """
    Build the effective ion-ion potential from `n_scr`.

    The physical reduction is Starrett--Saumon (2014), Eqs. (14)--(18),
    DOI 10.1016/j.hedp.2013.12.001.

    Key quantities
    --------------
      C_Ie(k) = -beta * n_scr(k) / chi_ee(k)
      beta*V_ii(k) = 4*pi*beta*Zbar^2/k^2 - n_scr(k) * C_Ie(k)

    Temperature convention
    ----------------------
    `ion_temperature_ha` is the ion thermal energy used in `beta V_ii`.
    The electron response uses `T_e`, taken from
    `options.response.electron_temperature_ha` when provided; otherwise it
    falls back to the same value as `ion_temperature_ha`.

    The Otter production path evaluates this construction as follows:
      1) transform n_scr(r) -> n_scr(k)
      2) build chi_ee(k)
      3) assemble C_Ie(k)
      4) assemble Eq.(14) in its charge-separated stable form
      5) inverse transform back to V_ii(r)

    The only k-space post-processing retained in the production path is one
    optional high-k cosine taper on `V_ii(k)` before the inverse transform.
    This targets the visible DST ringing in `V_ii(r)` and `g_ii(r)` while
    leaving the low-k physics untouched.
    """
    opts = options or QOZPotentialOptions()
    n_i = float(n_i)
    zbar = float(zbar)
    nbar_e0 = max(n_i * zbar, 1e-16)
    te_ha = float(
        opts.response.electron_temperature_ha
        if opts.response.electron_temperature_ha is not None
        else ion_temperature_ha
    )
    mu_jel = float(
        opts.response.mu_jellium_override_ha
        if opts.response.mu_jellium_override_ha is not None
        else mu_jellium_from_nbar(nbar_e0, te_ha)
    )

    n_scr_k = radial_forward(n_scr, transform)
    chi_ee, chi0_k, gee_k = chi_ee_from_eq17(
        k,
        nbar_e0=nbar_e0,
        electron_temperature_ha=te_ha,
        mu_jellium_ha=mu_jel,
        response_options=opts.response,
    )
    v_ie_k, v_ee_k, c_ie_k, c_ee_k = electron_interaction_channels_k(
        k=np.asarray(k, dtype=float),
        n_scr_k=n_scr_k,
        chi_ee_k=chi_ee,
        gee_k=gee_k,
        electron_temperature_ha=te_ha,
        ion_temperature_ha=ion_temperature_ha,
    )
    vii_k = _stable_effective_pair_potential_k(
        n_scr_k=n_scr_k[np.newaxis, :],
        zbar=np.asarray([zbar], dtype=float),
        k=np.asarray(k, dtype=float),
        chi0_k=chi0_k,
        gee_k=gee_k,
    )[0, 0]
    if opts.high_k_taper_start_frac is not None:
        taper_start = float(opts.high_k_taper_start_frac)
        if not (0.0 < taper_start < 1.0):
            raise ValueError("high_k_taper_start_frac must lie in (0, 1) when provided.")
        vii_k = vii_k * _high_k_cosine_taper_numba(np.asarray(k, dtype=float), taper_start)
    vii_r = radial_inverse(vii_k, transform)
    vii_r = vii_r - float(vii_r[-1])
    return EffectivePotentialResult(
        vii_r=vii_r,
        vii_k=vii_k,
        v_ie_k=v_ie_k,
        v_ee_k=v_ee_k,
        c_ie_k=c_ie_k,
        c_ee_k=c_ee_k,
        chi_ee_k=chi_ee,
        chi0_k=chi0_k,
        gee_k=gee_k,
        n_scr_k=n_scr_k,
        mu_jellium_ha=mu_jel,
        nbar_e0=nbar_e0,
    )


def build_effective_vij_from_nscr(
    r: np.ndarray,
    n_scr: np.ndarray,
    zbar: np.ndarray,
    n_i: np.ndarray,
    ion_temperature_ha: float,
    k: np.ndarray,
    transform: RadialTransform,
    options: QOZPotentialOptions | None = None,
) -> MultiComponentEffectivePotentialResult:
    """
    Build multicomponent effective ion-ion potentials from species screening clouds.

    Parameters
    ----------
    r
        Working linear real-space grid in Bohr.
    n_scr
        Screening densities with shape `(n_species, n_r)` in Bohr^-3.
    zbar
        Per-species net ionic charges.
    n_i
        Per-species ion number densities in Bohr^-3.
    ion_temperature_ha
        Ion temperature used in `beta V_ij`.
    k
        Reciprocal-space grid in Bohr^-1.
    transform
        Radial transform context for the same `r/k` grid.
    options
        Electron-response controls.

    Returns
    -------
    MultiComponentEffectivePotentialResult
        All pair potentials and intermediate screening / response arrays.

    Notes
    -----
    For a common electron response `chi_ee(k)` and one screening cloud per
    species, Starrett et al. (2014), Eq. (20),
    DOI 10.1103/PhysRevE.90.033110, gives

      C_ie^i(k) = -beta * n_scr^i(k) / chi_ee(k)
      V_ij(k)   = 4*pi*Z_i*Z_j/k^2 - [C_ie^i(k)/beta] * n_scr^j(k)

    which is symmetric once both species use the same `chi_ee(k)`.

    As in the one-component path, an optional high-k cosine taper can be
    applied to the final `V_ij(k)` before the inverse transform. This reduces
    the small Nyquist-scale ringing that otherwise appears in `V_ij(r)` and in
    the downstream HNC pair correlations.
    """
    opts = options or QOZPotentialOptions()
    r_arr = np.asarray(r, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    n_scr_arr = np.asarray(n_scr, dtype=float)
    zbar_arr = np.asarray(zbar, dtype=float)
    n_i_arr = np.asarray(n_i, dtype=float)
    if n_scr_arr.ndim != 2:
        raise ValueError("n_scr must have shape (n_species, n_r).")
    if n_scr_arr.shape[1] != r_arr.size:
        raise ValueError("n_scr second axis must match the real-space grid.")
    if zbar_arr.shape != (n_scr_arr.shape[0],):
        raise ValueError("zbar must have one entry per species.")
    if n_i_arr.shape != (n_scr_arr.shape[0],):
        raise ValueError("n_i must have one entry per species.")

    # Starrett's mixture response uses the bulk free-electron density
    # nbar_e0 = sum_i n_i^0 Zbar_i = n_mix * sum_i x_i Zbar_i.
    nbar_e0 = max(float(np.sum(n_i_arr * zbar_arr)), 1e-16)
    te_ha = float(
        opts.response.electron_temperature_ha
        if opts.response.electron_temperature_ha is not None
        else ion_temperature_ha
    )
    mu_jel = float(
        opts.response.mu_jellium_override_ha
        if opts.response.mu_jellium_override_ha is not None
        else mu_jellium_from_nbar(nbar_e0, te_ha)
    )
    chi_ee, chi0_k, gee_k = chi_ee_from_eq17(
        k_arr,
        nbar_e0=nbar_e0,
        electron_temperature_ha=te_ha,
        mu_jellium_ha=mu_jel,
        response_options=opts.response,
    )

    n_species = n_scr_arr.shape[0]
    n_scr_k = np.zeros((n_species, k_arr.size), dtype=float)
    for idx in range(n_species):
        n_scr_k[idx] = radial_forward(n_scr_arr[idx], transform)

    v_ie_k, v_ee_k, c_ie_k, c_ee_k = electron_interaction_channels_k(
        k=k_arr,
        n_scr_k=n_scr_k,
        chi_ee_k=chi_ee,
        gee_k=gee_k,
        electron_temperature_ha=te_ha,
        ion_temperature_ha=ion_temperature_ha,
    )

    vij_k = _stable_effective_pair_potential_k(
        n_scr_k=n_scr_k,
        zbar=zbar_arr,
        k=k_arr,
        chi0_k=chi0_k,
        gee_k=gee_k,
    )
    vij_r = np.zeros((n_species, n_species, r_arr.size), dtype=float)
    vij_k = 0.5 * (vij_k + np.swapaxes(vij_k, 0, 1))
    if opts.high_k_taper_start_frac is not None:
        taper_start = float(opts.high_k_taper_start_frac)
        if not (0.0 < taper_start < 1.0):
            raise ValueError("high_k_taper_start_frac must lie in (0, 1) when provided.")
        taper = _high_k_cosine_taper_numba(k_arr, taper_start)
        vij_k = vij_k * taper[np.newaxis, np.newaxis, :]
    for i in range(n_species):
        for j in range(n_species):
            vij_r[i, j] = radial_inverse(vij_k[i, j], transform)
            vij_r[i, j] = vij_r[i, j] - float(vij_r[i, j, -1])
    vij_r = 0.5 * (vij_r + np.swapaxes(vij_r, 0, 1))
    return MultiComponentEffectivePotentialResult(
        vij_r=vij_r,
        vij_k=vij_k,
        v_ie_k=v_ie_k,
        v_ee_k=v_ee_k,
        c_ie_k=c_ie_k,
        c_ee_k=c_ee_k,
        chi_ee_k=chi_ee,
        chi0_k=chi0_k,
        gee_k=gee_k,
        n_scr_k=n_scr_k,
        mu_jellium_ha=mu_jel,
        nbar_e0=nbar_e0,
        n_i_species=n_i_arr.copy(),
        zbar_species=zbar_arr.copy(),
    )


def hnc_solver(
    r: np.ndarray,
    k: np.ndarray,
    v_ii_r: np.ndarray,
    transform: RadialTransform,
    n_i: float,
    temperature_ha: float,
    *,
    mix: float = 0.12,
    tol: float = 1e-5,
    max_iter: int = 400,
    mixing_scheme: str = "anderson",
    anderson_m: int = 5,
    anderson_w0: float = 5e-4,
    arg_min: float = -80.0,
    arg_max: float = 80.0,
    c_map_clip: float = 200.0,
    s_min_floor: float = 1e-6,
    s_max_ceil: float = 1e3,
    g_tail_min: float = 0.4,
    g_tail_max: float = 2.5,
    enforce_h_tail_zero: bool = True,
    tail_points: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Solve the one-component OZ equation with the HNC closure.

    Parameters
    ----------
    r, k
        Working linear real-space and reciprocal-space grids.
    v_ii_r
        Effective ion-ion potential in real space.
    transform
        Precomputed radial-transform context built from the same `r` and `k`
        grids. The QOZ/OZ tests now use the strict DST lattice backend.
    n_i
        Ion number density in Bohr^-3.
    temperature_ha
        Temperature in Hartree.

    Returns
    -------
    g_r, s_k, h_r, c_r, residual_history

    Notes
    -----
    The iteration variable is the nodal term

      N(r) = h(r) - c(r)

    and the closure map is

      g = exp[-beta V + N]
      h = g - 1
      c = h - N
      h(k) = c(k) / (1 - n_i c(k))
      N_new(k) = h(k) - c(k)

    The OZ/HNC closure is the one used in Starrett--Saumon (2014),
    Eqs. (14)--(18).  Anderson/Picard mixing and the physical iterate guards
    are Otter numerical choices, not prescriptions from that reference.
    """
    r = np.asarray(r, dtype=float)
    k = np.asarray(k, dtype=float)
    v_ii_r = np.asarray(v_ii_r, dtype=float)
    if r.shape != transform.r.shape or k.shape != transform.k.shape:
        raise ValueError("r/k must match the supplied radial transform context.")
    if v_ii_r.shape != r.shape:
        raise ValueError("v_ii_r must have the same shape as r.")

    beta = 1.0 / max(float(temperature_ha), 1e-12)
    n_i = float(n_i)
    alpha = float(mix)
    res_hist: list[float] = []

    scheme = str(mixing_scheme).lower().strip()
    if scheme not in ("picard", "anderson"):
        raise ValueError("mixing_scheme must be 'picard' or 'anderson'.")

    beta_v_r = beta * v_ii_r
    n_tail = min(max(int(tail_points), 8), int(r.size))
    c_cap = float(c_map_clip)

    # Anderson history stores x_i and F(x_i) from previous steps. Only a short
    # window is useful here: the HNC map is stiff, and stale history tends to
    # damage the least-squares extrapolation more than it helps.
    y_hist: deque[np.ndarray] = deque(maxlen=max(int(anderson_m), 1))
    x_hist: deque[np.ndarray] = deque(maxlen=max(int(anderson_m), 1))

    def _map_n(x_n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply one HNC fixed-point map to the current nodal term N(r).

        Input
        -----
        x_n : current guess for N(r)=h(r)-c(r)

        Output
        ------
        n_new : updated nodal term after one OZ+closure cycle
        h_k   : total correlation in k-space
        h     : total correlation in r-space
        g     : pair distribution function
        c_k   : direct correlation in k-space

        This helper is the core of the HNC solver: every Picard/Anderson step
        is ultimately a different way of combining repeated evaluations of
        this map.
        """
        # Closure step: g = exp[-beta V + N]. The clipping avoids overflow in
        # the exponential while still allowing strong short-range repulsion.
        arg = np.clip(-beta_v_r + x_n, float(arg_min), float(arg_max))
        g = np.exp(arg)
        g = np.maximum(g, 1e-300)
        h = g - 1.0
        c = h - x_n

        # OZ step in k-space:
        #   h(k) = c(k) / (1 - n_i c(k))
        c_k = radial_forward(c, transform)
        denom = 1.0 - n_i * c_k
        small = np.abs(denom) < 1e-10
        if np.any(small):
            denom = denom.copy()
            denom[small] = np.where(denom[small] >= 0.0, 1e-10, -1e-10)
        h_k = c_k / denom
        n_k = h_k - c_k
        n_new = radial_inverse(n_k, transform)
        if enforce_h_tail_zero and n_new.size >= n_tail:
            # Legacy option: finite boxes can leave a small offset in N=h-c.
            # Subtracting it can reduce ringing, but it also perturbs the exact
            # OZ/closure identity; production therefore leaves this disabled.
            n_new = n_new - float(np.mean(n_new[-n_tail:]))
        if c_cap > 0.0:
            # The cap is purely numerical: it prevents a single bad trial step
            # from sending Anderson mixing into a huge-amplitude branch.
            n_new = np.clip(n_new, -c_cap, c_cap)
        return n_new, h_k, h, g, c_k

    def _candidate_physical(
        cand_n: np.ndarray,
    ) -> tuple[
        bool,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    ]:
        """
        Reject obviously non-physical HNC iterates before accepting them.

        Why this exists
        ---------------
        Plain fixed-point / Anderson updates occasionally jump to a branch with
        negative S(k), huge spikes, or a tail that no longer tends to 1. Once
        the iteration enters such a branch it rarely recovers on its own.

        This helper acts as a guard:
          - S(k) must stay finite and mostly positive
          - the far tail of g(r) must stay near 1

        If a trial point fails these tests, the outer loop backtracks instead
        of accepting it.
        """
        if not np.all(np.isfinite(cand_n)):
            return False, None
        mapped = _map_n(cand_n)
        _, h_k_loc, _, g_loc, _ = mapped
        s_loc = 1.0 + n_i * h_k_loc
        if not np.all(np.isfinite(s_loc)):
            return False, None
        if float(np.min(s_loc)) < float(s_min_floor):
            return False, None
        if float(np.max(s_loc)) > float(s_max_ceil):
            return False, None
        g_tail = float(np.mean(g_loc[-n_tail:]))
        if g_tail < float(g_tail_min) or g_tail > float(g_tail_max):
            return False, None
        return True, mapped

    best_res = np.inf
    n_cur = np.zeros_like(beta_v_r)
    best_n = n_cur.copy()
    best_ok = False

    for _ in range(int(max_iter)):
        n_map, _, _, _, _ = _map_n(n_cur)
        n_picard = (1.0 - alpha) * n_cur + alpha * n_map
        n_next = n_picard
        n_next_ok = False
        mapped_next: (
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            | None
        ) = None

        if scheme == "anderson":
            # Anderson mixing builds an extrapolated candidate from a short
            # history of residuals r_i = F(x_i) - x_i, with the linear
            # constraint sum(c_i)=1.
            x_hist.append(n_cur.copy())
            y_hist.append(n_map.copy())
            p = len(x_hist)
            if p >= 2:
                r_cols = [y_hist[i] - x_hist[i] for i in range(p)]
                r_mat = np.vstack([col.reshape(-1) for col in r_cols]).T
                try:
                    gram = r_mat.T @ r_mat
                    a_aug = np.block(
                        [
                            [gram + (float(anderson_w0) ** 2) * np.eye(p), np.ones((p, 1))],
                            [np.ones((1, p)), np.array([[0.0]])],
                        ]
                    )
                    b_aug = np.zeros(p + 1, dtype=float)
                    b_aug[-1] = 1.0
                    coeff = np.linalg.lstsq(a_aug, b_aug, rcond=None)[0][:p]
                    y_stack = np.vstack([y.reshape(-1) for y in y_hist]).T
                    n_and = (y_stack @ coeff).reshape(n_cur.shape)
                    n_and_ok, mapped_and = _candidate_physical(n_and)
                    if n_and_ok:
                        n_next = n_and
                        n_next_ok = True
                        mapped_next = mapped_and
                except np.linalg.LinAlgError:
                    n_next = n_picard

        if not n_next_ok:
            n_next_ok, mapped_next = _candidate_physical(n_next)
        if not n_next_ok:
            # If the extrapolated point is unphysical, repeatedly average it
            # back toward the previous iterate until it becomes acceptable.
            # This is a simple line search in function space.
            n_work = n_next.copy()
            accepted = False
            for _ls in range(12):
                n_work = 0.5 * (n_work + n_cur)
                n_work_ok, mapped_work = _candidate_physical(n_work)
                if n_work_ok:
                    n_next = n_work
                    accepted = True
                    n_next_ok = True
                    mapped_next = mapped_work
                    break
            if not accepted:
                # Final fallback: take only a tiny Picard step. If even that is
                # unphysical, freeze the iterate and let the residual test
                # decide whether more progress is possible.
                n_try = (1.0 - 0.02) * n_cur + 0.02 * n_map
                n_try_ok, mapped_try = _candidate_physical(n_try)
                if n_try_ok:
                    n_next = n_try
                    n_next_ok = True
                    mapped_next = mapped_try
                else:
                    n_next = n_cur.copy()
                    n_next_ok, mapped_next = _candidate_physical(n_next)

        # Score the *accepted* candidate, rather than the previous iterate.
        # This keeps the residual attached to the same N(r) from which the
        # returned g/h/c/S arrays are built.  In particular, a rejected or
        # frozen trial must not inherit an apparently good residual from a
        # different point.
        if mapped_next is None:
            mapped_next = _map_n(n_next)
        n_next_map = mapped_next[0]
        num = np.linalg.norm(n_next_map - n_next)
        den = max(np.linalg.norm(n_next), np.linalg.norm(n_next_map), 1e-14)
        res = float(num / den)
        res_hist.append(res)

        if n_next_ok and np.isfinite(res) and res < best_res:
            best_res = res
            best_n = n_next.copy()
            best_ok = True

        n_cur = n_next
        if res < float(tol):
            break

    if best_ok:
        n_cur = best_n

    _, h_k, h_r, g_r, c_k = _map_n(n_cur)
    c_r = radial_inverse(c_k, transform)
    s_ii_k = 1.0 + n_i * h_k
    return g_r, s_ii_k, h_r, c_r, res_hist


def hnc_solver_multicomponent(
    r: np.ndarray,
    k: np.ndarray,
    v_ij_r: np.ndarray,
    transform: RadialTransform,
    n_i: np.ndarray,
    temperature_ha: float,
    *,
    n_init_r: np.ndarray | None = None,
    mix: float = 0.12,
    tol: float = 1e-5,
    max_iter: int = 400,
    mixing_scheme: str = "anderson",
    anderson_m: int = 5,
    anderson_w0: float = 5e-4,
    arg_min: float = -80.0,
    arg_max: float = 80.0,
    c_map_clip: float = 200.0,
    s_min_floor: float = 1e-6,
    s_max_ceil: float = 1e3,
    s_projection_mode: str = "clip",
    g_tail_min: float = 0.4,
    g_tail_max: float = 2.5,
    enforce_h_tail_zero: bool = True,
    tail_points: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """
    Solve the classical multicomponent OZ + HNC problem on one shared radial grid.

    Parameters
    ----------
    r, k
        Working linear real-space and reciprocal-space grids.
    v_ij_r
        Pair potentials with shape `(n_species, n_species, n_r)`.
    transform
        Precomputed radial-transform context built from the same `r` and `k`.
    n_i
        Per-species ion number densities in Bohr^-3.
    temperature_ha
        Common ion temperature in Hartree.
    n_init_r
        Optional initial guess for the multicomponent nodal term
        `N_ij(r) = h_ij(r) - c_ij(r)` with shape `(n_species, n_species, n_r)`.
    s_projection_mode
        ``"none"`` solves the raw OZ matrix equation. ``"clip"`` retains the
        historical PSD eigenvalue projection for diagnostics and backward
        compatibility, but changes the fixed point and must not be used to
        claim a physical root.

    Returns
    -------
    g_r, s_k, h_r, c_r, residual_history

    Notes
    -----
    This implements the mixture OZ/HNC closure of Starrett et al. (2014),
    DOI 10.1103/PhysRevE.90.033110.  For each `k` mode it solves the matrix
    OZ relation

      H(k) = C(k) [I - D C(k)]^{-1}

    with `D = diag(n_i)`, and applies the HNC closure on every pair channel
    simultaneously.  The nodal-term iteration and optional PSD projection
    are Otter numerical/diagnostic choices; production disables projection
    because it changes the physical fixed point.
    """
    r_arr = np.asarray(r, dtype=float)
    k_arr = np.asarray(k, dtype=float)
    v_arr = np.asarray(v_ij_r, dtype=float)
    n_i_arr = np.asarray(n_i, dtype=float)
    if r_arr.shape != transform.r.shape or k_arr.shape != transform.k.shape:
        raise ValueError("r/k must match the supplied radial transform context.")
    if v_arr.ndim != 3 or v_arr.shape[2] != r_arr.size or v_arr.shape[0] != v_arr.shape[1]:
        raise ValueError("v_ij_r must have shape (n_species, n_species, n_r).")
    if n_i_arr.shape != (v_arr.shape[0],):
        raise ValueError("n_i must have one entry per species.")

    beta = 1.0 / max(float(temperature_ha), 1e-12)
    alpha = float(mix)
    res_hist: list[float] = []
    n_species = v_arr.shape[0]
    scheme = str(mixing_scheme).lower().strip()
    if scheme not in ("picard", "anderson", "newton_krylov"):
        raise ValueError(
            "mixing_scheme must be 'picard', 'anderson', or 'newton_krylov'."
        )
    projection_mode = str(s_projection_mode).lower().strip().replace("-", "_")
    if projection_mode not in ("clip", "none", "raw"):
        raise ValueError("s_projection_mode must be 'clip' or 'none'.")
    use_s_projection = projection_mode == "clip"

    beta_v_r = beta * v_arr
    n_tail = min(max(int(tail_points), 8), int(r_arr.size))
    c_cap = float(c_map_clip)
    if scheme == "newton_krylov" and (use_s_projection or c_cap > 0.0):
        raise ValueError(
            "newton_krylov requires s_projection_mode='none' and c_map_clip=0; "
            "otherwise Newton would solve a clipped surrogate rather than OZ/HNC."
        )
    sqrt_n_vec = np.sqrt(np.maximum(n_i_arr, 1.0e-300))
    sqrt_n = np.outer(sqrt_n_vec, sqrt_n_vec)
    inv_sqrt_n = 1.0 / np.maximum(sqrt_n, 1.0e-300)

    y_hist: deque[np.ndarray] = deque(maxlen=max(int(anderson_m), 1))
    x_hist: deque[np.ndarray] = deque(maxlen=max(int(anderson_m), 1))

    def _symmetrize_pair(arr: np.ndarray) -> np.ndarray:
        return 0.5 * (arr + np.swapaxes(arr, 0, 1))

    def _pair_forward(arr_r: np.ndarray) -> np.ndarray:
        return _symmetrize_pair(radial_forward(arr_r, transform))

    def _pair_inverse(arr_k: np.ndarray) -> np.ndarray:
        return _symmetrize_pair(radial_inverse(arr_k, transform))

    def _matrix_oz(c_k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the multicomponent OZ relation in the symmetric AL basis.

        Notes
        -----
        The raw species-space form `H = C + C D H` uses the non-symmetric
        matrix `D C`, which is poorly conditioned for unlike-species channels.
        We instead work with

          C_tilde = sqrt(D) C sqrt(D)
          S       = (I - C_tilde)^(-1)
          H       = sqrt(D)^(-1) (S - I) sqrt(D)^(-1)

        so each `k` mode is solved as a symmetric problem before mapping back
        to the unweighted pair-correlation basis.
        """
        eye = np.eye(n_species, dtype=float)
        eye_batch = np.broadcast_to(eye, (k_arr.size, n_species, n_species))
        c_batch = np.moveaxis(_symmetrize_pair(c_k), -1, 0)
        c_tilde_batch = c_batch * sqrt_n[np.newaxis, :, :]
        a_batch = eye_batch - c_tilde_batch
        try:
            s_batch = np.linalg.solve(a_batch, eye_batch)
        except np.linalg.LinAlgError:
            s_batch = np.linalg.solve(a_batch + 1.0e-10 * eye_batch, eye_batch)
        s_batch = 0.5 * (s_batch + np.swapaxes(s_batch, -1, -2))
        s_k_raw = _symmetrize_pair(np.moveaxis(s_batch, 0, -1))
        s_k = _project_s_matrix(s_k_raw) if use_s_projection else s_k_raw
        h_k = (s_k - eye[:, :, np.newaxis]) * inv_sqrt_n[:, :, np.newaxis]
        h_k = _symmetrize_pair(h_k)
        return h_k, s_k

    def _project_s_matrix(s_k: np.ndarray) -> np.ndarray:
        """
        Project each partial-structure matrix onto the physical PSD cone.

        Notes
        -----
        For multicomponent systems the diagonal-floor trick used in the
        one-component solver is no longer enough. A tiny negative eigenvalue of
        `S_ij(k)` can destabilize the whole OZ/HNC iteration. We therefore
        project each symmetric `S(k)` onto the cone of positive-semidefinite
        matrices with eigenvalues clipped to `[s_min_floor, s_max_ceil]`.
        """
        s_batch = np.moveaxis(_symmetrize_pair(s_k), -1, 0)
        eigval, eigvec = np.linalg.eigh(s_batch)
        eig_clip = np.clip(eigval, float(s_min_floor), float(s_max_ceil))
        s_proj_batch = np.matmul(eigvec * eig_clip[:, np.newaxis, :], np.swapaxes(eigvec, -1, -2))
        return _symmetrize_pair(np.moveaxis(s_proj_batch, 0, -1))

    def _map_n(x_n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # (1) Apply the HNC closure on every pair channel.
        arg = np.clip(-beta_v_r + x_n, float(arg_min), float(arg_max))
        g = np.exp(arg)
        g = np.maximum(g, 1e-300)
        g = _symmetrize_pair(g)
        h = g - 1.0
        c_r = _symmetrize_pair(h - x_n)

        # (2) Solve the matrix OZ relation for each k mode.
        c_k = _pair_forward(c_r)
        h_k, s_k = _matrix_oz(c_k)
        n_k = _symmetrize_pair(h_k - c_k)

        # (3) Transform the nodal term back to real space.  The optional tail
        # shift remains the direct-API legacy default, but the high-level
        # production workflow disables it because it perturbs the exact
        # OZ/closure identity.
        n_new = _pair_inverse(n_k)
        if enforce_h_tail_zero and n_new.shape[-1] >= n_tail:
            n_new = n_new - np.mean(n_new[..., -n_tail:], axis=-1, keepdims=True)
        if c_cap > 0.0:
            n_new = np.clip(n_new, -c_cap, c_cap)
        n_new = _symmetrize_pair(n_new)
        return n_new, h_k, h, g, c_k

    def _candidate_physical(
        cand_n: np.ndarray,
    ) -> tuple[bool, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None]:
        if not np.all(np.isfinite(cand_n)):
            return False, None
        mapped = _map_n(cand_n)
        _, h_k_loc, _, g_loc, _ = mapped
        if not np.all(np.isfinite(h_k_loc)) or not np.all(np.isfinite(g_loc)):
            return False, None
        if not use_s_projection:
            s_loc = (
                np.eye(n_species, dtype=float)[:, :, np.newaxis]
                + sqrt_n[:, :, np.newaxis] * h_k_loc
            )
            eig_loc = np.linalg.eigvalsh(
                np.moveaxis(_symmetrize_pair(s_loc), -1, 0)
            )
            if (
                not np.all(np.isfinite(eig_loc))
                or float(np.min(eig_loc)) < float(s_min_floor)
                or float(np.max(eig_loc)) > float(s_max_ceil)
            ):
                return False, None
        g_tail = np.mean(g_loc[:, :, -n_tail:], axis=2)
        if float(np.min(g_tail)) < float(g_tail_min) or float(np.max(g_tail)) > float(g_tail_max):
            return False, None
        return True, mapped

    best_res = np.inf
    if n_init_r is None:
        n_cur = np.zeros_like(beta_v_r)
    else:
        n_cur = np.asarray(n_init_r, dtype=float)
        if n_cur.shape != beta_v_r.shape:
            raise ValueError("n_init_r must have the same shape as v_ij_r.")
        n_cur = _symmetrize_pair(n_cur)
        if c_cap > 0.0:
            n_cur = np.clip(n_cur, -c_cap, c_cap)
    best_n = n_cur.copy()
    best_ok = False
    mapped_cur: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    if scheme == "newton_krylov":
        # Starrett & Saumon (2014), Sec. III, note that their strongly coupled
        # HNC cases require a Newton iteration.  Work only with the independent
        # upper-triangular pair channels and use a matrix-free Krylov Jacobian;
        # forming the dense Jacobian would scale as (N_r N_pair)^2.
        from scipy.optimize import NoConvergence, newton_krylov

        tri_i, tri_j = np.triu_indices(n_species)

        def _pack_pair(arr: np.ndarray) -> np.ndarray:
            return np.asarray(arr[tri_i, tri_j, :], dtype=float).reshape(-1)

        def _unpack_pair(vec: np.ndarray) -> np.ndarray:
            packed = np.asarray(vec, dtype=float).reshape(tri_i.size, r_arr.size)
            arr = np.empty_like(beta_v_r)
            arr[tri_i, tri_j, :] = packed
            arr[tri_j, tri_i, :] = packed
            return arr

        best_vec = _pack_pair(n_cur)

        def _raw_residual(vec: np.ndarray) -> np.ndarray:
            n_trial = _unpack_pair(vec)
            n_map_trial = _map_n(n_trial)[0]
            return _pack_pair(n_trial - n_map_trial)

        def _newton_callback(vec: np.ndarray, residual: np.ndarray) -> None:
            nonlocal best_res, best_vec, best_ok
            vec_norm = max(float(np.linalg.norm(vec)), 1.0e-14)
            res_rel = float(np.linalg.norm(residual) / vec_norm)
            res_hist.append(res_rel)
            candidate = _unpack_pair(vec)
            ok, _ = _candidate_physical(candidate)
            if ok and np.isfinite(res_rel) and res_rel < best_res:
                best_res = res_rel
                best_vec = np.asarray(vec, dtype=float).copy()
                best_ok = True

        x0 = _pack_pair(n_cur)
        try:
            solved_vec = np.asarray(
                newton_krylov(
                    _raw_residual,
                    x0,
                    method="lgmres",
                    inner_maxiter=40,
                    outer_k=max(int(anderson_m), 3),
                    maxiter=int(max_iter),
                    f_tol=max(float(tol) * 1.0e-2, 1.0e-10),
                    x_tol=np.inf,
                    x_rtol=1.0e300,
                    line_search="armijo",
                    callback=_newton_callback,
                    verbose=False,
                ),
                dtype=float,
            )
        except NoConvergence as exc:
            solved_vec = np.asarray(exc.args[0], dtype=float)

        solved_n = _unpack_pair(solved_vec)
        solved_residual = _raw_residual(solved_vec)
        solved_rel = float(
            np.linalg.norm(solved_residual)
            / max(float(np.linalg.norm(solved_vec)), 1.0e-14)
        )
        if not res_hist or solved_rel != res_hist[-1]:
            res_hist.append(solved_rel)
        solved_ok, _ = _candidate_physical(solved_n)
        if solved_ok and solved_rel < best_res:
            best_res = solved_rel
            best_vec = solved_vec.copy()
            best_ok = True
        n_cur = _unpack_pair(best_vec if best_ok else solved_vec)
        _, h_k, h_r, g_r, c_k = _map_n(n_cur)
        c_r = _pair_inverse(c_k)
        s_ij_k = np.eye(n_species)[:, :, np.newaxis] + sqrt_n[:, :, np.newaxis] * h_k
        return g_r, s_ij_k, h_r, c_r, res_hist

    for _ in range(int(max_iter)):
        if mapped_cur is None:
            mapped_cur = _map_n(n_cur)
        n_map, _, _, _, _ = mapped_cur
        n_picard = (1.0 - alpha) * n_cur + alpha * n_map
        n_next = n_picard
        n_next_ok = False
        mapped_next: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

        if scheme == "anderson":
            # Use the same short-history Anderson logic as the one-component solver.
            x_hist.append(n_cur.copy())
            y_hist.append(n_map.copy())
            p = len(x_hist)
            if p >= 2:
                r_cols = [y_hist[i] - x_hist[i] for i in range(p)]
                r_mat = np.vstack([col.reshape(-1) for col in r_cols]).T
                try:
                    gram = r_mat.T @ r_mat
                    a_aug = np.block(
                        [
                            [gram + (float(anderson_w0) ** 2) * np.eye(p), np.ones((p, 1))],
                            [np.ones((1, p)), np.array([[0.0]])],
                        ]
                    )
                    b_aug = np.zeros(p + 1, dtype=float)
                    b_aug[-1] = 1.0
                    coeff = np.linalg.lstsq(a_aug, b_aug, rcond=None)[0][:p]
                    y_stack = np.vstack([y.reshape(-1) for y in y_hist]).T
                    n_and = (y_stack @ coeff).reshape(n_cur.shape)
                    n_and = _symmetrize_pair(n_and)
                    n_and_ok, mapped_and = _candidate_physical(n_and)
                    if n_and_ok:
                        n_next = n_and
                        n_next_ok = True
                        mapped_next = mapped_and
                except np.linalg.LinAlgError:
                    n_next = n_picard

        if not n_next_ok:
            n_next_ok, mapped_next = _candidate_physical(n_next)
        if not n_next_ok:
            # Backtrack any unphysical multicomponent extrapolation toward the previous iterate.
            n_work = n_next.copy()
            accepted = False
            for _ls in range(12):
                n_work = 0.5 * (n_work + n_cur)
                n_work = _symmetrize_pair(n_work)
                n_work_ok, mapped_work = _candidate_physical(n_work)
                if n_work_ok:
                    n_next = n_work
                    accepted = True
                    n_next_ok = True
                    mapped_next = mapped_work
                    break
            if not accepted:
                n_try = (1.0 - 0.02) * n_cur + 0.02 * n_map
                n_try = _symmetrize_pair(n_try)
                n_try_ok, mapped_try = _candidate_physical(n_try)
                if n_try_ok:
                    n_next = n_try
                    n_next_ok = True
                    mapped_next = mapped_try
                else:
                    n_next = n_cur.copy()
                    n_next_ok = True
                    mapped_next = mapped_cur

        # ``mapped_next`` belongs to the accepted candidate.  Use its own
        # fixed-point defect so the stored best iterate and the reported
        # residual always describe the same point.
        if mapped_next is None:
            mapped_next = _map_n(n_next)
        n_next_map = mapped_next[0]
        num = np.linalg.norm(n_next_map - n_next)
        den = max(np.linalg.norm(n_next), np.linalg.norm(n_next_map), 1e-14)
        res = float(num / den)
        res_hist.append(res)

        if n_next_ok and np.isfinite(res) and res < best_res:
            best_res = res
            best_n = n_next.copy()
            best_ok = True

        n_cur = n_next
        mapped_cur = mapped_next
        if res < float(tol):
            break

    if best_ok:
        n_cur = best_n

    _, h_k, h_r, g_r, c_k = _map_n(n_cur)
    c_r = _pair_inverse(c_k)
    s_ij_k = np.eye(n_species)[:, :, np.newaxis] + sqrt_n[:, :, np.newaxis] * h_k
    return g_r, s_ij_k, h_r, c_r, res_hist


def hnc_solver_multicomponent_continuation(
    r: np.ndarray,
    k: np.ndarray,
    v_ij_r: np.ndarray,
    transform: RadialTransform,
    n_i: np.ndarray,
    temperature_ha: float,
    *,
    potential_scales: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    stage_max_iter: int | None = None,
    newton_max_iter: int | None = None,
    adaptive: bool = False,
    min_scale_step: float = 2.0e-3,
    max_stage_attempts: int = 32,
    require_converged: bool = False,
    fallback_mixing_scheme: str | None = None,
    **kwargs,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[float],
    list[dict[str, float | bool | str]],
]:
    """
    Solve multicomponent HNC by ramping the pair potential strength in stages.

    Parameters
    ----------
    r, k, v_ij_r, transform, n_i, temperature_ha
        Same physical inputs as `hnc_solver_multicomponent`.
    potential_scales
        Monotone sequence of positive scale factors applied to `v_ij_r`.
        Each stage warm-starts the next one with the previous nodal term.
    stage_max_iter
        Optional per-stage iteration cap. If omitted, reuse `max_iter`.
    newton_max_iter
        Optional, usually smaller cap for matrix-free Newton stages.
    adaptive
        Bisect a rejected potential-strength increment down to
        `min_scale_step` instead of carrying a failed stage forward.
    require_converged
        Raise when no raw, positive, closure-consistent root can be continued
        to the requested scale.
    fallback_mixing_scheme
        Optional solver used after the primary scheme rejects a stage.
    **kwargs
        Forwarded to `hnc_solver_multicomponent`.

    Returns
    -------
    g_r, s_k, h_r, c_r, residual_history, stage_meta
        Final multicomponent HNC outputs together with one compact summary per
        continuation stage.

    Notes
    -----
    This potential-strength continuation is an Otter numerical method, not a
    procedure stated in Starrett et al. (2014).  Real mixture pair potentials
    can be much stiffer than the weak synthetic
    smoke tests. A short potential-strength continuation frequently reaches a
    physical branch that direct full-strength iteration misses.  A stage is
    accepted only when the fixed-point residual, raw OZ positivity, and the
    independent `S(k) <-> FT[g(r)-1]` identity all pass; projected or clipped
    pseudo-roots are never warm-started into the next production stage.
    """
    scales = tuple(float(val) for val in potential_scales)
    if len(scales) == 0:
        raise ValueError("potential_scales must contain at least one stage.")
    if any(val <= 0.0 for val in scales):
        raise ValueError("potential_scales must be strictly positive.")
    if any(right <= left for left, right in zip(scales[:-1], scales[1:])):
        raise ValueError("potential_scales must be strictly increasing.")
    if float(min_scale_step) <= 0.0:
        raise ValueError("min_scale_step must be positive.")
    if int(max_stage_attempts) < len(scales):
        raise ValueError("max_stage_attempts must be at least len(potential_scales).")

    local_kwargs = dict(kwargs)
    max_iter_local = int(local_kwargs.pop("max_iter", 400))
    tol_local = float(local_kwargs.get("tol", 1.0e-5))
    if stage_max_iter is not None:
        max_iter_local = int(stage_max_iter)
    if newton_max_iter is not None and int(newton_max_iter) < 2:
        raise ValueError("newton_max_iter must be at least 2 when provided.")

    def _iteration_cap(scheme_name: str) -> int:
        if str(scheme_name).strip().lower() == "newton_krylov" and newton_max_iter is not None:
            return min(max_iter_local, int(newton_max_iter))
        return max_iter_local

    n_init = local_kwargs.pop("n_init_r", None)
    stage_meta: list[dict[str, float | bool | str]] = []
    last_out: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]] | None = None
    last_accepted_meta: dict[str, float | bool | str] | None = None
    pending_scales = list(scales)
    last_accepted_scale = 0.0
    active_scheme = str(local_kwargs.get("mixing_scheme", "anderson"))
    attempt = 0
    while pending_scales:
        if attempt >= int(max_stage_attempts):
            raise RuntimeError(
                "Multicomponent HNC continuation exhausted its adaptive stage budget "
                f"after {attempt} attempts; largest accepted potential scale="
                f"{last_accepted_scale:.6f}."
            )
        scale = float(pending_scales.pop(0))
        attempt += 1
        solve_kwargs = dict(local_kwargs)
        solve_kwargs["mixing_scheme"] = active_scheme
        g_r, s_k, h_r, c_r, res_hist = hnc_solver_multicomponent(
            r,
            k,
            float(scale) * np.asarray(v_ij_r, dtype=float),
            transform,
            n_i,
            temperature_ha,
            n_init_r=n_init,
            max_iter=_iteration_cap(active_scheme),
            **solve_kwargs,
        )

        def _stage_diagnostics() -> dict[str, float | bool | str]:
            s_arr = np.asarray(s_k, dtype=float)
            s_batch = np.moveaxis(0.5 * (s_arr + np.swapaxes(s_arr, 0, 1)), -1, 0)
            reported_val, reported_vec = np.linalg.eigh(s_batch)
            reported_eig = reported_val
            s0_eig = reported_eig[0]
            max_mode_flat = int(np.argmax(reported_val))
            max_k_index, max_branch = np.unravel_index(max_mode_flat, reported_val.shape)
            max_mode_vec = reported_vec[max_k_index, :, max_branch]
            c_k_stage = radial_forward(np.asarray(c_r, dtype=float), transform)
            c_k_stage = 0.5 * (c_k_stage + np.swapaxes(c_k_stage, 0, 1))
            sqrt_n_vec = np.sqrt(np.maximum(np.asarray(n_i, dtype=float), 1.0e-300))
            sqrt_n = np.outer(sqrt_n_vec, sqrt_n_vec)
            fraction = np.asarray(n_i, dtype=float) / float(np.sum(n_i))
            number_direction = np.sqrt(np.maximum(fraction, 0.0))
            number_direction /= max(float(np.linalg.norm(number_direction)), 1.0e-300)
            number_overlap = float(abs(np.dot(max_mode_vec, number_direction)))
            instability_channel = (
                "number_like" if number_overlap >= 1.0 / np.sqrt(2.0) else "concentration_like"
            )
            c_tilde = np.moveaxis(c_k_stage, -1, 0) * sqrt_n[np.newaxis, :, :]
            a_batch = np.eye(c_tilde.shape[1], dtype=float)[np.newaxis, :, :] - c_tilde
            a_eig = np.linalg.eigvalsh(0.5 * (a_batch + np.swapaxes(a_batch, 1, 2)))
            h_from_g_k = radial_forward(np.asarray(g_r, dtype=float) - 1.0, transform)
            s_from_g = (
                np.eye(s_arr.shape[0], dtype=float)[:, :, np.newaxis]
                + sqrt_n[:, :, np.newaxis] * h_from_g_k
            )
            closure_max = float(np.max(np.abs(s_arr - s_from_g)))
            residual = float(res_hist[-1]) if res_hist else np.inf
            projection_mode = str(solve_kwargs.get("s_projection_mode", "clip")).lower().strip()
            raw_psd = bool(float(np.min(a_eig)) > 0.0)
            converged = bool(
                np.isfinite(residual)
                and residual <= tol_local
                and raw_psd
                and np.isfinite(closure_max)
                and closure_max <= max(10.0 * tol_local, 1.0e-6)
            )
            return {
                "stage_index": float(attempt - 1),
                "potential_scale": float(scale),
                "n_iter": float(len(res_hist)),
                "res_final": residual,
                "s0_min_eig": float(np.min(s0_eig)),
                "s0_max_eig": float(np.max(s0_eig)),
                "reported_s_min_eig": float(np.min(reported_eig)),
                "reported_s_max_eig": float(np.max(reported_eig)),
                "reported_s_max_k": float(np.asarray(k, dtype=float)[max_k_index]),
                "max_mode_number_overlap_abs": number_overlap,
                "max_mode_channel": instability_channel,
                "oz_denominator_min_eig": float(np.min(a_eig)),
                "closure_transform_max_abs": closure_max,
                "raw_oz_positive": raw_psd,
                "converged": converged,
                "mixing_scheme": active_scheme,
                "s_projection_mode": projection_mode,
            }

        meta = _stage_diagnostics()

        # If the inexpensive fixed-point iteration fails, retry the same
        # state from the last accepted solution with the Newton-Krylov backend.
        fallback = None if fallback_mixing_scheme is None else str(fallback_mixing_scheme).strip()
        if not bool(meta["converged"]) and fallback and active_scheme != fallback:
            active_scheme = fallback
            solve_kwargs = dict(local_kwargs)
            solve_kwargs["mixing_scheme"] = active_scheme
            g_r, s_k, h_r, c_r, res_hist = hnc_solver_multicomponent(
                r,
                k,
                float(scale) * np.asarray(v_ij_r, dtype=float),
                transform,
                n_i,
                temperature_ha,
                n_init_r=n_init,
                max_iter=_iteration_cap(active_scheme),
                **solve_kwargs,
            )
            meta = _stage_diagnostics()

        stage_meta.append(meta)
        if bool(meta["converged"]):
            last_out = (g_r, s_k, h_r, c_r, res_hist)
            n_init = np.asarray(h_r, dtype=float) - np.asarray(c_r, dtype=float)
            last_accepted_scale = float(scale)
            last_accepted_meta = dict(meta)
            continue

        gap = float(scale) - float(last_accepted_scale)
        if bool(adaptive) and 0.5 * gap >= float(min_scale_step):
            midpoint = float(last_accepted_scale) + 0.5 * gap
            pending_scales.insert(0, float(scale))
            pending_scales.insert(0, midpoint)
            continue

        accepted_mode_note = ""
        if last_accepted_meta is not None:
            accepted_mode_note = (
                " At the largest accepted scale, the largest physical S mode was "
                f"{str(last_accepted_meta['max_mode_channel'])} at "
                f"k={float(last_accepted_meta['reported_s_max_k']):.6f} Bohr^-1 "
                f"with lambda_max={float(last_accepted_meta['reported_s_max_eig']):.3e}."
            )
        message = (
            "Multicomponent OZ/HNC did not reach a physical fixed point: "
            f"failed potential scale={scale:.6f}, largest accepted scale="
            f"{last_accepted_scale:.6f}, residual={float(meta['res_final']):.3e}, "
            f"min eig[I-sqrt(D) C sqrt(D)]="
            f"{float(meta['oz_denominator_min_eig']):.3e}, "
            f"closure mismatch={float(meta['closure_transform_max_abs']):.3e}. "
            f"Largest failed-stage S mode is {str(meta['max_mode_channel'])} at "
            f"k={float(meta['reported_s_max_k']):.6f} Bohr^-1 with "
            f"lambda_max={float(meta['reported_s_max_eig']):.3e}."
            f"{accepted_mode_note} "
            "The homogeneous atomic-fluid PA-HNC branch is unresolved or unstable; "
            "PSD projection is not a physical substitute for a root."
        )
        if bool(require_converged):
            raise RuntimeError(message)
        last_out = (g_r, s_k, h_r, c_r, res_hist)
        n_init = np.asarray(h_r, dtype=float) - np.asarray(c_r, dtype=float)
        last_accepted_scale = float(scale)

    if last_out is None:
        raise RuntimeError("multicomponent continuation produced no stages.")
    g_r, s_k, h_r, c_r, res_hist = last_out
    return g_r, s_k, h_r, c_r, res_hist, stage_meta
