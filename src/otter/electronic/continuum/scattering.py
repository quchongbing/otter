"""
otter/electronic/continuum/scattering.py

Purpose
-------
Prototype quantum continuum model using the A3 energy-integral form.

Methods
-------
- Integrate continuum density over energy.
- Sum partial waves using free-electron spherical Bessel functions
  as a baseline (V_eff = 0).
- Adaptive energy integration to resolve sharp resonances (optional).

Equations
---------
  n_c(r) = ∫_0^∞ dε g(ε) Σ_l [ 2(2l+1)/(4π) ] | y_{k,l}(r) / r |^2
  ε = k^2 / 2
  y_{k,l}(r) = sqrt(2k/π) r j_l(k r)  (energy normalization for dE)

References
----------
- :cite:`StarrettSaumon2014`, Eq. (A3). The free-wave baseline, adaptive
  quadrature, and resonance diagnostics are Otter implementation choices.
"""
from typing import Dict, Tuple, Any
import math
import time
import warnings
import numpy as np
import multiprocessing as mp
from numba import njit
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn, loggamma

try:
    from scipy.special import coulombf, coulombg

    _HAVE_COULOMB = True
    _COULOMB_BACKEND = "scipy"
except Exception:
    coulombf = None
    coulombg = None
    _HAVE_COULOMB = False
    _COULOMB_BACKEND = "none"

# The production basis is deliberately deterministic across SciPy builds:
# use the analytic large-r Coulomb phase below even when one installation
# happens to expose exact ``coulombf/g`` functions.  Exact special functions
# can be compared in diagnostics, but an undeclared optional package must not
# silently change a continuum density.
_HAVE_COULOMB_ASYM = True

from .interface import ContinuumModel


def _trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numpy-version-safe trapezoidal integration helper.
    """
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def _init_scatter_perf_accum() -> dict[str, float]:
    """Allocate one accumulator for fine-grained continuum timings."""
    return {
        "plan_s": 0.0,
        "propagate_s": 0.0,
        "match_s": 0.0,
        "accumulate_s": 0.0,
        "eval_total_s": 0.0,
    }


def _scatter_perf_meta(perf_accum: dict[str, float] | None, wall_s: float) -> dict[str, float]:
    """Convert raw scattering accumulators into exported metadata fields."""
    if perf_accum is None:
        return {}
    eval_total = float(perf_accum.get("eval_total_s", 0.0))
    return {
        "perf_plan_s": float(perf_accum.get("plan_s", 0.0)),
        "perf_propagate_s": float(perf_accum.get("propagate_s", 0.0)),
        "perf_match_s": float(perf_accum.get("match_s", 0.0)),
        "perf_accumulate_s": float(perf_accum.get("accumulate_s", 0.0)),
        "perf_eval_total_s": float(eval_total),
        "perf_control_s": float(max(float(wall_s) - eval_total, 0.0)),
        "perf_wall_s": float(wall_s),
    }


def scattering_default_params() -> Dict[str, Any]:
    """
    Return baseline parameters for the scattering continuum model.

    Notes
    -----
    This helper is intended for tests/benchmarks. Callers should still
    supply problem-specific values for l_max, e_min/e_max, and energy grids.
    """
    return {
        "l_pad": 2,
        "match_fraction": 0.2,
        "match_kr_min": 4.0,
        "match_v_tol": 1e-4,
        "match_min_points": 12,
        "match_asymptotic": "auto",
        "match_coulomb_tol": 0.1,
        "match_allow_shift": True,
        "match_fallback": "free",
        "prop_rescale_limit": 1e6,
        "energy_mode": "adaptive",
        "e_tol": 1e-3,
        "e_max_depth": 10,
        "e_min_width": 1e-4,
        "n_e_base": 8,
        "e_base_grid": "linear",
        # Low-energy threshold guard.  A sqrt(E) base mesh can leave a wide
        # first panel even when e_min is small; logarithmic anchors make that
        # panel observable to the ordinary adaptive quadrature.  This is not
        # an l=0 "resonance" classifier.
        "near_zero_log_grid": True,
        "near_zero_log_points_per_decade": 4,
        "near_zero_log_max_nodes": 24,
        "near_zero_log_max_energy": 1.0e-2,
        "adaptive_parallel_mode": "batch",
        "adaptive_shards": None,
        "delta_tol": np.pi / 2.0,
        "delta_mode": "max",
        # Experimental Wilson-inspired narrow-resonance scout.  It remains
        # off unless adaptive_mode="phase-root" (or "theta-scout") is selected
        # explicitly.  This is not Wilson et al.'s exact relativistic Theta.
        "resonance_theta_l_min": 1,
        "resonance_theta_probe_count": 1,
        # A finite phase-root scout cannot guarantee an arbitrarily narrow
        # even pair of roots.  Dyadic levels nevertheless make its achieved
        # energy resolution explicit and reproducible at bounded cost.
        "resonance_theta_scan_depth": 3,
        "resonance_theta_scout_max_extra_nodes": 128,
        "resonance_theta_root_tol": None,
        "resonance_theta_sharpness_min": 2.0,
        "resonance_theta_max_roots": None,
        "resonance_theta_refine_depth": None,
        "tail_match": False,
        "tail_auto_fallback": "fraction",
        "tail_fit_points": 16,
        "tail_blend_points": 0,
    }


def fermi_dirac(energy: np.ndarray, mu: float, temperature: float) -> np.ndarray:
    """
    Fermi-Dirac occupation for continuum energies.
    """
    T = max(float(temperature), 1e-12)
    beta = 1.0 / T
    x = (energy - mu) * beta
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(x))


def unwrap_scattering_phases(phases: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return a continuous scattering-phase branch along ``axis``.

    A partial-wave S matrix depends on ``exp(2j*delta_l)``, so ``delta_l`` is
    defined modulo pi rather than modulo 2*pi.  Unwrapping ``2*delta_l`` and
    dividing by two avoids spurious pi-sized jumps in phase derivatives and
    density-of-states diagnostics.
    """
    phase_arr = np.asarray(phases, dtype=float)
    return 0.5 * np.unwrap(2.0 * phase_arr, axis=int(axis))


def phase_root_resonance_scout(phases: np.ndarray) -> np.ndarray:
    """Return a Wilson-inspired phase-root scout for narrow resonances.

    In the asymptotic matching convention used here, an unnormalised regular
    solution is fitted as

    ``u = c_F F_l + c_G G_l``

    with ``c_F = A cos(delta_l)`` and ``c_G = -A sin(delta_l)``.  Therefore

    ``T_l = c_F / hypot(c_F, c_G) = cos(delta_l)``

    has zeros at a purely irregular exterior solution.  For ``l > 0`` a
    narrow shape resonance produces a sharp zero of this function and a
    simultaneous maximum of the energy-normalised interior charge.  The root
    location is independent of the arbitrary magnitude (and its zero is
    independent of the sign) of the outward-propagated Numerov solution.

    This criterion is *inspired by*, but is not identical to, the smooth
    relativistic ``Theta(epsilon)`` construction used to catch sub-grid
    continuum resonances in Sec. 6 of Wilson et al., *JQSRT* **99**, 658-679
    (2006).  Wilson's expression combines both Dirac components and additional
    Wronskians.  Here we only use the non-relativistic regular/irregular fit
    already available in this code.

    Notes
    -----
    This function is deliberately *not* a generic resonance classifier.  An
    ``l=0`` threshold crossing and a broad ``delta=pi/2`` crossing also give a
    zero.  Production scouting therefore defaults to ``l >= 1`` and estimates
    the local slope before carving out a refinement window.

    The adaptive scout only locates roots bracketed by its base/probe nodes.
    A non-Breit-Wigner feature whose phase rises and falls so that an even
    number of roots lies wholly between two scout nodes can still be missed.
    Increasing ``resonance_theta_scan_depth`` reduces, but cannot eliminate,
    that limitation.  The adaptive integrator reports the smallest and
    largest achieved scout spacing and whether its node budget was exhausted;
    a Green-function contour treatment would be the more complete solution
    for arbitrary ultra-narrow spectra.
    """
    return np.cos(np.asarray(phases, dtype=float))


def free_electron_y(r: np.ndarray, k: float, l: int) -> np.ndarray:
    """
    Free-electron radial function y_{k,l}(r) with energy normalization.

    Notes
    -----
    A3 integrates over energy dE, so the continuum states are δ(E)-normalized.
    This yields ``y ~ sqrt(2k/π) r j_l(kr)``.  The extra factor of ``k``
    relative to k-normalized ``r*j_l`` is required when converting from
    Dirac-delta normalization in ``k`` to Dirac-delta normalization in
    ``E=k^2/2``.  If integrating over ``k`` instead, the normalization changes
    by the corresponding inverse-square-root factor in ``k``.
    """
    if k <= 0.0:
        return np.zeros_like(r)
    return np.sqrt(2.0 * k / np.pi) * r * spherical_jn(l, k * r)


def continuum_density_free(r: np.ndarray,
                           mu: float,
                           temperature: float,
                           e_grid: np.ndarray,
                           l_max: int) -> np.ndarray:
    """
    Compute continuum density with free-electron wavefunctions.

    Notes
    -----
    This is a V_eff = 0 reference implementation for A3.
    """
    r = np.asarray(r)
    e_grid = np.asarray(e_grid)
    k_grid = np.sqrt(2.0 * e_grid)
    occ = fermi_dirac(e_grid, mu, temperature)

    n_e_r = np.zeros((e_grid.size, r.size), dtype=float)
    for i, (e, k) in enumerate(zip(e_grid, k_grid)):
        if k <= 0:
            continue
        sum_l = np.zeros_like(r, dtype=float)
        for l in range(l_max + 1):
            y = free_electron_y(r, k, l)
            factor = (2.0 * (2 * l + 1)) / (4.0 * np.pi)
            sum_l += factor * (np.abs(y / r) ** 2)
        n_e_r[i] = occ[i] * sum_l

    n_r = _trapz(n_e_r, e_grid, axis=0)
    return n_r


def continuum_density_scattering_basis(v_eff: np.ndarray,
                                       r: np.ndarray,
                                       e_grid: np.ndarray,
                                       l_max: int,
                                       grid_kind: str = "sqrt",
                                       grid_step: float | None = None,
                                       l_pad: int = 2,
                                       match_fraction: float = 0.2,
                                       match_slice: tuple[int, int] | None = None,
                                       match_r_cut: float | None = None,
                                       match_fraction_mode: str = "r",
                                       match_width: float | None = None,
                                       match_kr_min: float | None = 4.0,
                                       match_v_tol: float | None = 1e-4,
                                       match_min_points: int = 12,
                                       match_asymptotic: str = "auto",
                                       match_coulomb_tol: float = 0.1,
                                       match_allow_shift: bool = True,
                                       match_fallback: str = "free",
                                       prop_rescale_limit: float | None = 1e6,
                                       l_cap_strategy: str = "match",
                                       energy_cache: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
                                       n_jobs: int | None = None,
                                       return_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Continuum scattering density basis (no Fermi-Dirac occupancy applied).

    This precomputes the energy-resolved density kernel B(E,r) so that
    n_cont(r) = ∫ dE B(E,r) f_FD(E; μ,T). It is used to accelerate inner-μ
    searches by reusing scattering states for many μ values.

    Inputs
    ------
    v_eff, r : ndarray
        Effective potential and radial grid.
    e_grid : ndarray
        Energy grid (Ha) for continuum integration.
    l_max : int
        Maximum angular momentum for partial-wave sum.
    grid_kind, grid_step : str, float
        Grid type and step (sqrt only).
    matching params
        Control asymptotic matching and phase-shift extraction.

    Returns
    -------
    ndarray
        Basis array with shape (n_e, n_r); integrate over E with FD weights.
    """
    r = np.asarray(r)
    v_eff = np.asarray(v_eff)
    e_grid = np.asarray(e_grid)
    if grid_kind != "sqrt":
        raise ValueError("continuum_density_scattering_basis supports only grid_kind='sqrt'.")
    if grid_step is None:
        grid_step = float(np.sqrt(r[1]) - np.sqrt(r[0]))

    t_wall = time.perf_counter() if bool(return_meta) else 0.0
    perf_accum = _init_scatter_perf_accum() if bool(return_meta) else None
    cache_hits = 0
    n_eval_new = 0

    # Allocate basis: rows are energies, columns are radii.
    n_e_r = np.zeros((e_grid.size, r.size), dtype=float)
    cache = energy_cache if energy_cache is not None else {}
    numerov_geom = _prepare_numerov_geometry(r, v_eff)
    r_eval = np.asarray(numerov_geom["r"], dtype=float)
    v_eval = np.asarray(numerov_geom["v_eff"], dtype=float)
    # Parallel branch: distribute energies across processes.
    if n_jobs is not None and int(n_jobs) > 1:
        if cache:
            warnings.warn("energy_cache ignored when n_jobs>1 for continuum scattering.")
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=int(n_jobs),
            initializer=_init_scatter_worker,
            initargs=(
                v_eval,
                r_eval,
                0.0,
                0.0,
                l_max,
                grid_kind,
                grid_step,
                l_pad,
                match_fraction,
                match_slice,
                match_r_cut,
                match_fraction_mode,
                match_width,
                match_kr_min,
                match_v_tol,
                match_min_points,
                match_asymptotic,
                match_coulomb_tol,
                match_allow_shift,
                match_fallback,
                prop_rescale_limit,
                False,
                l_cap_strategy,
            ),
        ) as pool:
            results = pool.map(_scatter_worker, [float(e) for e in e_grid])
        # Collect per-energy densities.
        for i, (e_val, n_e, delta_vec) in enumerate(results):
            n_e_r[i] = n_e
            cache[e_val] = (n_e, delta_vec)
        n_eval_new = int(e_grid.size)
    else:
        # Serial branch with optional cache reuse.
        for i, e in enumerate(e_grid):
            e_val = float(e)
            if e_val in cache:
                cached = cache[e_val]
                n_e_r[i] = cached[0] if isinstance(cached, tuple) else cached
                cache_hits += 1
                continue
            n_e, delta_vec = _scattering_density_and_phase(
                v_eval,
                r_eval,
                0.0,
                0.0,
                e_val,
                l_max,
                grid_kind,
                grid_step,
                l_pad,
                match_fraction,
                match_slice,
                match_r_cut,
                match_fraction_mode,
                match_width,
                match_kr_min,
                match_v_tol,
                match_min_points,
                match_asymptotic,
                match_coulomb_tol,
                match_allow_shift,
                match_fallback,
                prop_rescale_limit,
                apply_occ=False,
                l_cap_strategy=l_cap_strategy,
                numerov_geom=numerov_geom,
                perf_accum=perf_accum,
            )
            cache[e_val] = (n_e, delta_vec)
            n_e_r[i] = n_e
            n_eval_new += 1

    if not bool(return_meta):
        return n_e_r

    wall_s = time.perf_counter() - t_wall
    meta = {
        "n_eval": int(n_eval_new),
        "n_cache_hits": int(cache_hits),
        "n_cache_total": int(len(cache)),
        "n_e_basis": int(e_grid.size),
        "adaptive_mode": "linear",
    }
    meta.update(_scatter_perf_meta(perf_accum, wall_s))
    return n_e_r, meta

@njit(cache=True, fastmath=True)
def _numerov_propagate_sqrt_numba(r: np.ndarray,
                                  v_eff: np.ndarray,
                                  energy: float,
                                  l: int,
                                  dxi: float,
                                  rescale_limit: float = 1e6) -> np.ndarray:
    N = r.size
    Psi = np.zeros(N, dtype=np.float64)
    power = float(l) + 0.75
    origin_charge = max(
        0.0,
        -0.5 * (r[0] * v_eff[0] + r[1] * v_eff[1]),
    )
    cusp_denom = float(l) + 1.0
    Psi[0] = r[0] ** power * (1.0 - origin_charge * r[0] / cusp_denom)
    if Psi[0] < 1e-300:
        Psi[0] = 1e-300
    Psi[1] = r[1] ** power * (1.0 - origin_charge * r[1] / cusp_denom)
    if Psi[1] < 1e-300:
        Psi[1] = 1e-300

    l_term = 4.0 * float(l) * (float(l) + 1.0) + 0.75
    h = (dxi ** 2) / 12.0
    w_prev = 8.0 * r[0] * (energy - v_eff[0]) - l_term / r[0]
    w_curr = 8.0 * r[1] * (energy - v_eff[1]) - l_term / r[1]
    for i in range(1, N - 1):
        w_next = 8.0 * r[i + 1] * (energy - v_eff[i + 1]) - l_term / r[i + 1]
        Psi[i + 1] = (
            2.0 * (1.0 - 5.0 * h * w_curr) * Psi[i]
            - (1.0 + h * w_prev) * Psi[i - 1]
        ) / (1.0 + h * w_next)
        if rescale_limit > 0.0 and abs(Psi[i + 1]) > rescale_limit:
            for j in range(i + 2):
                Psi[j] /= rescale_limit
        w_prev = w_curr
        w_curr = w_next

    u = np.empty(N, dtype=np.float64)
    for i in range(N):
        u[i] = np.sqrt(np.sqrt(r[i])) * Psi[i]
    return u


def _numerov_propagate_sqrt(r: np.ndarray,
                            v_eff: np.ndarray,
                            energy: float,
                            l: int,
                            dxi: float,
                            rescale_limit: float = 1e6) -> np.ndarray:
    r = np.asarray(r)
    if r.size > 0 and r[0] <= 0.0:
        r = r.copy()
        r[0] = 1e-14
    v_eff = np.asarray(v_eff)
    return _numerov_propagate_sqrt_numba(
        r, v_eff, float(energy), int(l), float(dxi), float(rescale_limit)
    )


def _prepare_numerov_geometry(r: np.ndarray,
                              v_eff: np.ndarray) -> dict[str, np.ndarray]:
    """
    Precompute grid-only Numerov factors for one continuum solve.

    Parameters
    ----------
    r : ndarray
        Radial grid in Bohr.
    v_eff : ndarray
        Effective potential on `r` in Ha.

    Returns
    -------
    dict
        Dictionary containing the sanitized `r`, `v_eff`, and grid factors
        reused across all continuum energies on the current SCF iterate.

    Notes
    -----
    The sqrt-grid Numerov recurrence uses

    `W_i(E,l) = 8 r_i (E - V_i) - (4 l(l+1) + 0.75) / r_i`.

    For a fixed SCF iterate, `r_i` and `V_i` are reused across many energies
    and partial waves. This helper precomputes the grid-only pieces once so
    each energy needs only `w_base(E) = 8 r_i E - 8 r_i V_i`, and each
    l-channel needs only the centrifugal correction.
    """
    r_safe = np.asarray(r, dtype=float)
    if r_safe.size > 0 and r_safe[0] <= 0.0:
        r_safe = r_safe.copy()
        r_safe[0] = 1e-14
    v_eff = np.asarray(v_eff, dtype=float)
    origin_count = min(4, int(r_safe.size))
    origin_charge = float(
        max(0.0, np.median(-r_safe[:origin_count] * v_eff[:origin_count]))
    )
    if origin_charge < 1.0e-6:
        origin_charge = 0.0
    return {
        "r": r_safe,
        "v_eff": v_eff,
        "r_quarter": np.sqrt(np.sqrt(r_safe)),
        "inv_r": 1.0 / r_safe,
        "inv_r2": 1.0 / (r_safe * r_safe),
        "r8": 8.0 * r_safe,
        "v_term": -8.0 * r_safe * v_eff,
        "origin_charge": np.asarray(origin_charge, dtype=float),
    }


@njit(cache=True, fastmath=True)
def _numerov_propagate_sqrt_wbase_inplace_numba(r: np.ndarray,
                                                r_quarter: np.ndarray,
                                                inv_r: np.ndarray,
                                                w_base: np.ndarray,
                                                l: int,
                                                dxi: float,
                                                out: np.ndarray,
                                                rescale_limit: float = 1e6,
                                                origin_charge: float = 0.0) -> None:
    """
    Numerov propagation with precomputed grid geometry into a caller buffer.

    Notes
    -----
    `w_base[i] = 8 r_i (E - V_i)` is shared by all l-channels at a fixed
    continuum energy, so the per-channel recurrence only adds the centrifugal
    correction `-(4 l(l+1)+0.75)/r_i`.
    """
    N = r.size
    do_rescale = rescale_limit > 0.0
    power = float(l) + 0.75
    cusp_denom = float(l) + 1.0
    out[0] = r[0] ** power * (1.0 - origin_charge * r[0] / cusp_denom)
    if out[0] < 1e-300:
        out[0] = 1e-300
    out[1] = r[1] ** power * (1.0 - origin_charge * r[1] / cusp_denom)
    if out[1] < 1e-300:
        out[1] = 1e-300

    l_term = 4.0 * float(l) * (float(l) + 1.0) + 0.75
    h = (dxi ** 2) / 12.0
    w_prev = w_base[0] - l_term * inv_r[0]
    w_curr = w_base[1] - l_term * inv_r[1]
    for i in range(1, N - 1):
        w_next = w_base[i + 1] - l_term * inv_r[i + 1]
        out[i + 1] = (
            2.0 * (1.0 - 5.0 * h * w_curr) * out[i]
            - (1.0 + h * w_prev) * out[i - 1]
        ) / (1.0 + h * w_next)
        if do_rescale and abs(out[i + 1]) > rescale_limit:
            for j in range(i + 2):
                out[j] /= rescale_limit
        w_prev = w_curr
        w_curr = w_next

    for i in range(N):
        out[i] *= r_quarter[i]


@njit(cache=True, fastmath=True)
def _numerov_propagate_sqrt_wbase_numba(r: np.ndarray,
                                        r_quarter: np.ndarray,
                                        inv_r: np.ndarray,
                                        w_base: np.ndarray,
                                        l: int,
                                        dxi: float,
                                        rescale_limit: float = 1e6,
                                        origin_charge: float = 0.0) -> np.ndarray:
    """
    Numerov propagation with precomputed grid geometry and per-energy base term.
    """
    out = np.empty(r.size, dtype=np.float64)
    _numerov_propagate_sqrt_wbase_inplace_numba(
        r,
        r_quarter,
        inv_r,
        w_base,
        l,
        dxi,
        out,
        rescale_limit=rescale_limit,
        origin_charge=origin_charge,
    )
    return out


@njit(cache=True, fastmath=True)
def _numerov_propagate_sqrt_wbase_batch_numba(r: np.ndarray,
                                              r_quarter: np.ndarray,
                                              inv_r: np.ndarray,
                                              w_base: np.ndarray,
                                              l_vals: np.ndarray,
                                              dxi: float,
                                              rescale_limit: float = 1e6,
                                              origin_charge: float = 0.0) -> np.ndarray:
    """
    Propagate all partial waves for one energy on a shared radial sweep.

    Parameters
    ----------
    r : ndarray
        Radial grid in Bohr.
    r_quarter : ndarray
        Precomputed ``r^(1/4)`` factor for the sqrt-grid wavefunction map.
    inv_r : ndarray
        Precomputed reciprocal radial grid ``1/r``.
    w_base : ndarray
        Shared per-energy term ``8 r (E - V_eff)``.
    l_vals : ndarray
        Integer partial waves to propagate for this energy.
    dxi : float
        Uniform sqrt-grid spacing.
    rescale_limit : float, optional
        Overflow-protection threshold for the raw Numerov solution ``psi``.

    Returns
    -------
    ndarray
        Array with shape ``(n_l, n_r)`` containing ``u_l(r) = r^(1/4) psi_l``.

    Notes
    -----
    For a fixed continuum energy, all channels share the same radial sweep and
    differ only by the centrifugal correction. This helper evolves all
    requested ``l`` values together so the expensive radial loop is traversed
    once per energy instead of once per channel.
    """
    N = r.size
    n_l = l_vals.size
    psi = np.empty((N, n_l), dtype=np.float64)
    l_terms = np.empty(n_l, dtype=np.float64)
    w_prev = np.empty(n_l, dtype=np.float64)
    w_curr = np.empty(n_l, dtype=np.float64)
    do_rescale = rescale_limit > 0.0
    h = (dxi ** 2) / 12.0

    # (1) Seed each partial wave with the regular-origin power law.
    for j in range(n_l):
        l_float = float(l_vals[j])
        power = l_float + 0.75
        cusp_denom = l_float + 1.0
        psi[0, j] = r[0] ** power * (
            1.0 - origin_charge * r[0] / cusp_denom
        )
        if psi[0, j] < 1e-300:
            psi[0, j] = 1e-300
        psi[1, j] = r[1] ** power * (
            1.0 - origin_charge * r[1] / cusp_denom
        )
        if psi[1, j] < 1e-300:
            psi[1, j] = 1e-300
        l_terms[j] = 4.0 * l_float * (l_float + 1.0) + 0.75
        w_prev[j] = w_base[0] - l_terms[j] * inv_r[0]
        w_curr[j] = w_base[1] - l_terms[j] * inv_r[1]

    # (2) March once over radius and update every l-channel in place.
    for i in range(1, N - 1):
        inv_next = inv_r[i + 1]
        for j in range(n_l):
            w_next = w_base[i + 1] - l_terms[j] * inv_next
            psi[i + 1, j] = (
                2.0 * (1.0 - 5.0 * h * w_curr[j]) * psi[i, j]
                - (1.0 + h * w_prev[j]) * psi[i - 1, j]
            ) / (1.0 + h * w_next)
            if do_rescale and abs(psi[i + 1, j]) > rescale_limit:
                for k in range(i + 2):
                    psi[k, j] /= rescale_limit
            w_prev[j] = w_curr[j]
            w_curr[j] = w_next

    # (3) Return contiguous row-major u_l(r) buffers for downstream matching.
    out = np.empty((n_l, N), dtype=np.float64)
    for i in range(N):
        scale = r_quarter[i]
        for j in range(n_l):
            out[j, i] = psi[i, j] * scale
    return out




@njit(cache=True, inline="always")
def _spherical_j0_exact(x: float) -> float:
    ax = abs(x)
    if ax < 1e-5:
        x2 = x * x
        return 1.0 - x2 / 6.0 + x2 * x2 / 120.0 - x2 * x2 * x2 / 5040.0
    return math.sin(x) / x


@njit(cache=True, inline="always")
def _spherical_j1_exact(x: float) -> float:
    ax = abs(x)
    if ax < 1e-4:
        x2 = x * x
        return x / 3.0 - x * x2 / 30.0 + x * x2 * x2 / 840.0 - x * x2 * x2 * x2 / 45360.0
    return math.sin(x) / (x * x) - math.cos(x) / x


@njit(cache=True, inline="always")
def _spherical_y0_exact(x: float) -> float:
    return -math.cos(x) / x


@njit(cache=True, inline="always")
def _spherical_y1_exact(x: float) -> float:
    return -math.cos(x) / (x * x) - math.sin(x) / x


@njit(cache=True)
def _spherical_jn_yn_all_numba(l_max: int, x: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute j_0..j_lmax and y_0..y_lmax for one real x.

    Notes
    -----
    - j_l(x) uses Miller downward recurrence + normalization.
    - y_l(x) uses upward recurrence.
    """
    j = np.empty(l_max + 1, dtype=np.float64)
    y = np.empty(l_max + 1, dtype=np.float64)

    if x == 0.0:
        j[0] = 1.0
        y[0] = -math.inf
        for l_val in range(1, l_max + 1):
            j[l_val] = 0.0
            y[l_val] = -math.inf
        return j, y

    y[0] = _spherical_y0_exact(x)
    if l_max >= 1:
        y[1] = _spherical_y1_exact(x)
        for l_val in range(1, l_max):
            y[l_val + 1] = ((2.0 * l_val + 1.0) / x) * y[l_val] - y[l_val - 1]

    L = l_max + int(max(50.0, x + 25.0))
    tmp = np.empty(L + 2, dtype=np.float64)
    tmp[L + 1] = 0.0
    tmp[L] = 1.0

    for n_val in range(L, 0, -1):
        prev = ((2.0 * n_val + 1.0) / x) * tmp[n_val] - tmp[n_val + 1]
        ap = abs(prev)
        if ap > 1e200:
            scale = 1e-200
            for m_val in range(n_val, L + 2):
                tmp[m_val] *= scale
            prev *= scale
        elif 0.0 < ap < 1e-200:
            scale = 1e200
            for m_val in range(n_val, L + 2):
                tmp[m_val] *= scale
            prev *= scale
        tmp[n_val - 1] = prev

    j0 = _spherical_j0_exact(x)
    j1 = _spherical_j1_exact(x)
    if abs(j0) >= abs(j1):
        scale = j0 / tmp[0]
    else:
        scale = j1 / tmp[1]

    for l_val in range(l_max + 1):
        j[l_val] = tmp[l_val] * scale

    return j, y


@njit(cache=True)
def _free_bessel_tables_numba(z_grid: np.ndarray, l_max: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build exact free spherical-Bessel tables for one energy grid z = k r.

    Returns
    -------
    j_tab, y_tab : ndarray
        Arrays with shape (l_max + 1, n_z).
    """
    n_z = z_grid.size
    j_tab = np.empty((l_max + 1, n_z), dtype=np.float64)
    y_tab = np.empty((l_max + 1, n_z), dtype=np.float64)
    for i in range(n_z):
        j_vals, y_vals = _spherical_jn_yn_all_numba(l_max, float(z_grid[i]))
        for l_val in range(l_max + 1):
            j_tab[l_val, i] = j_vals[l_val]
            y_tab[l_val, i] = y_vals[l_val]
    return j_tab, y_tab


@njit(cache=True, fastmath=True)
def _solve_match_amplitude_phase_numba(u: np.ndarray,
                                       i0: int,
                                       u_f: np.ndarray,
                                       u_g: np.ndarray) -> tuple[int, float, float]:
    """
    Solve the scaled 2x2 asymptotic matching system for one channel.

    Parameters
    ----------
    u : ndarray
        Raw propagated channel on the full radial grid.
    i0 : int
        Start index of the matching window inside ``u``.
    u_f, u_g : ndarray
        Exact asymptotic basis functions on the matching window.

    Returns
    -------
    status_code : int
        ``0`` for success, ``1`` for invalid propagated amplitude,
        ``2`` for invalid asymptotic basis, ``3`` when the scaled 2x2 system
        is too ill-conditioned and should fall back to Python ``lstsq``,
        and ``4`` for a non-physical fitted amplitude.
    amp : float
        Extracted asymptotic amplitude.
    delta : float
        Extracted phase shift (radians).

    Notes
    -----
    The helper mirrors the production least-squares normalization, but keeps
    the common well-conditioned case inside one numba loop nest to avoid
    repeated tiny NumPy allocations.
    """
    n = u_f.size
    if n < 2 or u_g.size != n or i0 < 0 or (i0 + n) > u.size:
        return 2, np.nan, 0.0

    # (1) Find robust channel scales before forming the 2x2 normal system.
    scale_b = 0.0
    scale_f = 0.0
    scale_g = 0.0
    for j in range(n):
        b_val = u[i0 + j]
        f_val = u_f[j]
        g_val = u_g[j]
        if not math.isfinite(b_val):
            return 1, np.nan, 0.0
        if not math.isfinite(f_val) or not math.isfinite(g_val):
            return 2, np.nan, 0.0
        abs_b = abs(b_val)
        abs_f = abs(f_val)
        abs_g = abs(g_val)
        if abs_b > scale_b:
            scale_b = abs_b
        if abs_f > scale_f:
            scale_f = abs_f
        if abs_g > scale_g:
            scale_g = abs_g

    if not math.isfinite(scale_b) or scale_b <= 0.0:
        return 4, np.nan, 0.0
    if not math.isfinite(scale_f) or not math.isfinite(scale_g):
        return 2, np.nan, 0.0
    if scale_f <= 0.0:
        scale_f = 1.0
    if scale_g <= 0.0:
        scale_g = 1.0

    # (2) Accumulate the scaled normal equations in one pass.
    inv_scale_b = 1.0 / scale_b
    inv_scale_f = 1.0 / scale_f
    inv_scale_g = 1.0 / scale_g
    s00 = 0.0
    s01 = 0.0
    s11 = 0.0
    t0 = 0.0
    t1 = 0.0
    for j in range(n):
        a0 = u_f[j] * inv_scale_f
        a1 = u_g[j] * inv_scale_g
        b = u[i0 + j] * inv_scale_b
        s00 += a0 * a0
        s01 += a0 * a1
        s11 += a1 * a1
        t0 += a0 * b
        t1 += a1 * b

    # (3) Solve the direct 2x2 system when it is well conditioned.
    det = s00 * s11 - s01 * s01
    det_floor = 1e-14 * max(abs(s00 * s11), 1.0)
    if not math.isfinite(det) or abs(det) <= det_floor:
        return 3, np.nan, 0.0

    coeff0 = ((t0 * s11 - t1 * s01) / det) * inv_scale_f
    coeff1 = ((s00 * t1 - s01 * t0) / det) * inv_scale_g
    amp = math.sqrt(coeff0 * coeff0 + coeff1 * coeff1) * scale_b
    if not math.isfinite(amp) or amp <= 0.0:
        return 1, np.nan, 0.0
    delta = math.atan2(-coeff1, coeff0)
    return 0, amp, delta


@njit(cache=True, fastmath=True)
def _solve_match_amplitude_phase_free_numba(u: np.ndarray,
                                            i0: int,
                                            z: np.ndarray,
                                            j_l: np.ndarray,
                                            y_l: np.ndarray) -> tuple[int, float, float]:
    """
    Match against the free asymptotic basis without building z*j_l and z*y_l.
    """
    n = z.size
    if n < 2 or j_l.size != n or y_l.size != n or i0 < 0 or (i0 + n) > u.size:
        return 2, np.nan, 0.0

    scale_b = 0.0
    scale_f = 0.0
    scale_g = 0.0
    for j in range(n):
        b_val = u[i0 + j]
        f_val = z[j] * j_l[j]
        g_val = z[j] * y_l[j]
        if not math.isfinite(b_val):
            return 1, np.nan, 0.0
        if not math.isfinite(f_val) or not math.isfinite(g_val):
            return 2, np.nan, 0.0
        abs_b = abs(b_val)
        abs_f = abs(f_val)
        abs_g = abs(g_val)
        if abs_b > scale_b:
            scale_b = abs_b
        if abs_f > scale_f:
            scale_f = abs_f
        if abs_g > scale_g:
            scale_g = abs_g

    if not math.isfinite(scale_b) or scale_b <= 0.0:
        return 4, np.nan, 0.0
    if not math.isfinite(scale_f) or not math.isfinite(scale_g):
        return 2, np.nan, 0.0
    if scale_f <= 0.0:
        scale_f = 1.0
    if scale_g <= 0.0:
        scale_g = 1.0

    inv_scale_b = 1.0 / scale_b
    inv_scale_f = 1.0 / scale_f
    inv_scale_g = 1.0 / scale_g
    s00 = 0.0
    s01 = 0.0
    s11 = 0.0
    t0 = 0.0
    t1 = 0.0
    for j in range(n):
        a0 = (z[j] * j_l[j]) * inv_scale_f
        a1 = (z[j] * y_l[j]) * inv_scale_g
        b = u[i0 + j] * inv_scale_b
        s00 += a0 * a0
        s01 += a0 * a1
        s11 += a1 * a1
        t0 += a0 * b
        t1 += a1 * b

    det = s00 * s11 - s01 * s01
    det_floor = 1e-14 * max(abs(s00 * s11), 1.0)
    if not math.isfinite(det) or abs(det) <= det_floor:
        return 3, np.nan, 0.0

    coeff0 = ((t0 * s11 - t1 * s01) / det) * inv_scale_f
    coeff1 = ((s00 * t1 - s01 * t0) / det) * inv_scale_g
    amp = math.sqrt(coeff0 * coeff0 + coeff1 * coeff1) * scale_b
    if not math.isfinite(amp) or amp <= 0.0:
        return 1, np.nan, 0.0
    delta = math.atan2(-coeff1, coeff0)
    return 0, amp, delta


def _solve_match_amplitude_phase(u: np.ndarray,
                                 i0: int,
                                 i1: int,
                                 u_f: np.ndarray,
                                 u_g: np.ndarray) -> tuple[float, float, str]:
    """
    Extract the asymptotic amplitude and phase for one matching window.

    Parameters
    ----------
    u : ndarray
        Raw propagated channel on the full radial grid.
    i0, i1 : int
        Matching-window bounds inside ``u``.
    u_f, u_g : ndarray
        Exact asymptotic basis functions on the matching window.

    Returns
    -------
    amp : float
        Extracted asymptotic amplitude.
    delta : float
        Extracted phase shift (radians).
    status : str
        ``"ok"`` on success, otherwise a diagnostic status compatible with the
        existing matching metadata.

    Notes
    -----
    The common well-conditioned path is handled by
    ``_solve_match_amplitude_phase_numba``. Only the rare singular 2x2 case
    falls back to ``np.linalg.lstsq``.
    """
    status_code, amp, delta = _solve_match_amplitude_phase_numba(
        np.asarray(u, dtype=float),
        int(i0),
        np.asarray(u_f, dtype=float),
        np.asarray(u_g, dtype=float),
    )
    if status_code == 0:
        return float(amp), float(delta), "ok"
    if status_code == 1:
        return np.nan, 0.0, "bad_scale"
    if status_code == 2:
        return np.nan, 0.0, "bad_basis"
    if status_code == 4:
        return np.nan, 0.0, "bad_amp"

    b = np.asarray(u[i0:i1], dtype=float)
    scale_b = np.max(np.abs(b))
    if not np.isfinite(scale_b) or scale_b <= 0.0:
        return np.nan, 0.0, "bad_scale"
    b = b / scale_b

    scale_a = np.array(
        [
            np.max(np.abs(u_f)),
            np.max(np.abs(u_g)),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(scale_a)):
        return np.nan, 0.0, "bad_basis"
    scale_a = np.where(scale_a > 0.0, scale_a, 1.0)
    a0 = np.asarray(u_f, dtype=float) / scale_a[0]
    a1 = np.asarray(u_g, dtype=float) / scale_a[1]
    a_scaled = np.column_stack((a0, a1))
    try:
        coeff_scaled, _, _, _ = np.linalg.lstsq(a_scaled, b, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, 0.0, "lstsq_fail"

    coeff = coeff_scaled / scale_a
    amp = float(np.sqrt(coeff[0] ** 2 + coeff[1] ** 2) * scale_b)
    if not np.isfinite(amp) or amp <= 0.0:
        return np.nan, 0.0, "bad_amp"
    delta = float(np.arctan2(-coeff[1], coeff[0]))
    return amp, delta, "ok"


@njit(cache=True, fastmath=True)
def _accumulate_scattering_density_inplace(sum_l: np.ndarray,
                                           u: np.ndarray,
                                           inv_r2: np.ndarray,
                                           coeff: float) -> None:
    """
    In-place channel accumulation without temporary arrays.
    """
    for i in range(sum_l.size):
        ui = u[i]
        sum_l[i] += coeff * ui * ui * inv_r2[i]


def _contiguous_tail_slice(mask: np.ndarray,
                           min_points: int) -> tuple[int, int] | None:
    """
    Return the last contiguous True run with at least min_points entries.
    """
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    splits = np.where(np.diff(idx) != 1)[0] + 1
    runs = np.split(idx, splits)
    for run in reversed(runs):
        if run.size >= min_points:
            return int(run[0]), int(run[-1] + 1)
    return None


def _resolve_match_base_slice(r: np.ndarray,
                              match_slice: tuple[int, int] | None,
                              match_fraction: float,
                              match_r_cut: float | None,
                              match_width: float | None,
                              match_fraction_mode: str) -> tuple[int, int]:
    """
    Resolve the outer tail window before energy- and l-dependent constraints.
    """
    N = r.size
    if match_slice is not None:
        i0, i1 = match_slice
    elif match_r_cut is not None:
        idx_cut = int(np.searchsorted(r, float(match_r_cut)))
        if match_width is not None:
            idx_end = int(np.searchsorted(r, float(match_r_cut + match_width)))
            i0, i1 = idx_cut, idx_end
        else:
            i0, i1 = idx_cut, N
    else:
        mode = str(match_fraction_mode).lower()
        if mode in ("r", "radius", "physical"):
            r_start = float((1.0 - match_fraction) * r[-1])
            i0 = int(np.searchsorted(r, r_start))
        else:
            i0 = int(max(1, (1.0 - match_fraction) * N))
        i1 = N

    i0 = max(1, min(int(i0), N - 2))
    i1 = max(i0 + 1, min(int(i1), N))
    return i0, i1


def _contiguous_true_runs(mask: np.ndarray) -> np.ndarray:
    """
    Return contiguous True runs as an array of shape (n_runs, 2).

    Notes
    -----
    Each row stores a half-open interval `[start, end)` in mask-local indices.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return np.empty((0, 2), dtype=int)

    edge = np.diff(mask.astype(np.int8))
    starts = np.where(edge == 1)[0] + 1
    ends = np.where(edge == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate((np.array([0], dtype=int), starts))
    if mask[-1]:
        ends = np.concatenate((ends, np.array([mask.size], dtype=int)))
    return np.column_stack((starts.astype(int), ends.astype(int)))


def _select_match_window(r: np.ndarray,
                         v_eff: np.ndarray | None,
                         energy: float,
                         l: int,
                         match_slice: tuple[int, int] | None,
                         match_fraction: float,
                         match_r_cut: float | None,
                         match_width: float | None,
                         match_fraction_mode: str,
                         match_kr_min: float | None,
                         match_v_tol: float | None,
                         match_min_points: int) -> tuple[tuple[int, int], dict]:
    """
    Select a tail window for asymptotic matching.

    Returns
    -------
    (i0, i1), meta
        Slice indices and diagnostics for the selected window.

    Notes
    -----
    The window is chosen near the outer grid and can be constrained by:
    - k r >= match_kr_min (oscillatory regime),
    - |V_eff| <= match_v_tol (asymptotic potential).
    match_fraction_mode controls whether match_fraction is interpreted in
    index-space ("index", default) or physical radius ("r").
    match_width sets a finite end point for the window when match_r_cut is used.
    """
    i0, i1 = _resolve_match_base_slice(
        r,
        match_slice,
        match_fraction,
        match_r_cut,
        match_width,
        match_fraction_mode,
    )

    meta = {
        "base_slice": (i0, i1),
        "used_constraints": False,
        "fallback": False,
    }

    if match_kr_min is None and match_v_tol is None:
        return (i0, i1), meta
    if not np.isfinite(energy) or energy <= 0.0:
        return (i0, i1), meta

    k = np.sqrt(2.0 * energy)
    if not np.isfinite(k) or k <= 0.0:
        return (i0, i1), meta

    rm = r[i0:i1]
    mask_kr = np.ones_like(rm, dtype=bool)
    mask_v = np.ones_like(rm, dtype=bool)

    if match_kr_min is not None:
        kr = k * rm
        kr_min = max(float(match_kr_min), float(l + 1))
        mask_kr &= kr >= kr_min

    if match_v_tol is not None and v_eff is not None:
        v_win = v_eff[i0:i1]
        mask_v &= np.abs(v_win) <= float(match_v_tol)

    meta["used_constraints"] = True
    min_points = max(int(match_min_points), 2)
    slice_rel = _contiguous_tail_slice(mask_kr & mask_v, min_points)
    if slice_rel is None and match_kr_min is not None:
        # Exact spherical-Bessel/Coulomb reference functions remain valid in
        # the non-oscillatory kr<kr_min regime.  If the tail potential itself
        # is already asymptotic, keep the propagated interacting solution and
        # relax only the conditioning preference on kr.  Replacing the state
        # by a free wave here creates an artificial phase jump at
        # E=(kr_min/r_match)^2/2 and removes the interacting threshold
        # continuum (as well as any genuine resonant structure).
        slice_rel = _contiguous_tail_slice(mask_v, min_points)
        if slice_rel is not None:
            meta["kr_constraint_relaxed"] = True
            meta["fallback_reason"] = "kr_only"
    if slice_rel is None:
        meta["fallback"] = True
        meta["fallback_reason"] = "no_asymptotic_tail_window"
        return (i0, i1), meta

    j0, j1 = slice_rel
    return (i0 + j0, i0 + j1), meta


def _estimate_coulomb_eta(rm: np.ndarray,
                          v_tail: np.ndarray,
                          k: float,
                          rel_tol: float) -> float | None:
    """
    Estimate Coulomb eta realizeable from a tail region.
    """
    if rm.size < 3 or not np.isfinite(k) or k <= 0.0:
        return None
    rv = rm * v_tail
    med = float(np.median(rv))
    if not np.isfinite(med) or abs(med) <= 0.0:
        return None
    rel_std = float(np.std(rv)) / max(abs(med), 1e-12)
    if rel_std > rel_tol:
        return None
    z_eff = -med
    return float(-z_eff / k)


def _coulomb_asymptotic_basis(rm: np.ndarray,
                              k: float,
                              l: int,
                              eta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Asymptotic Coulomb basis for large rho = k r.

    The DLMF convention satisfies
    ``u'' + [1 - 2*eta/rho - l(l+1)/rho**2] u = 0`` and has phase
    ``rho - eta*log(2*rho) - l*pi/2 + sigma_l``.
    """
    rho = np.maximum(k * rm, 1e-12)
    sigma_l = float(np.imag(loggamma(l + 1.0 + 1j * eta)))
    phase = rho - 0.5 * l * np.pi + sigma_l - eta * np.log(2.0 * rho)
    return np.sin(phase), np.cos(phase)


def _resolve_asymptotic_basis_meta(rm: np.ndarray,
                                   v_tail: np.ndarray | None,
                                   energy: float,
                                   asymptotic: str,
                                   match_v_tol: float,
                                   coulomb_tol: float,
                                   allow_shift: bool) -> dict:
    """
    Resolve asymptotic-basis kind and parameters without building the basis.
    """
    k = np.sqrt(2.0 * energy)
    v_tail = np.asarray(v_tail) if v_tail is not None else None

    kind = str(asymptotic).lower()
    if kind == "auto":
        v_max = 0.0
        if v_tail is not None and v_tail.size > 0:
            v_max = float(np.max(np.abs(v_tail)))
        if v_max <= match_v_tol:
            kind = "free"
        else:
            eta = _estimate_coulomb_eta(rm, v_tail, k, coulomb_tol) if v_tail is not None else None
            if eta is not None and (_HAVE_COULOMB or _HAVE_COULOMB_ASYM):
                kind = "coulomb"
            elif allow_shift:
                kind = "shifted"
            else:
                kind = "free"

    meta = {
        "kind": kind,
        "k_use": float(k),
        "eta": None,
        "v_shift": None,
        "coulomb_backend": None,
    }

    if kind == "coulomb":
        eta = _estimate_coulomb_eta(rm, v_tail, k, coulomb_tol) if v_tail is not None else None
        if eta is None:
            meta["kind"] = "free"
        elif _HAVE_COULOMB_ASYM:
            meta["eta"] = float(eta)
            meta["coulomb_backend"] = "asymptotic"
        elif _HAVE_COULOMB:
            meta["eta"] = float(eta)
            meta["coulomb_backend"] = _COULOMB_BACKEND
        else:
            meta["kind"] = "free"

    if meta["kind"] == "shifted":
        v_shift = float(np.median(v_tail)) if v_tail is not None and v_tail.size > 0 else 0.0
        e_shift = energy - v_shift
        meta["v_shift"] = v_shift
        if e_shift <= 0.0:
            meta["kind"] = "shifted_forbidden"
        else:
            meta["k_use"] = float(np.sqrt(2.0 * e_shift))

    return meta


def _build_asymptotic_basis_from_meta(rm: np.ndarray,
                                      energy: float,
                                      l: int,
                                      meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Build asymptotic basis functions from a resolved metadata dictionary.
    """
    kind = str(meta.get("kind", "free")).lower()
    k = np.sqrt(2.0 * energy)

    if kind == "coulomb":
        eta = float(meta.get("eta", 0.0))
        backend = str(meta.get("coulomb_backend", "none")).lower()
        if backend == "scipy":
            rho = k * rm
            return coulombf(l, eta, rho), coulombg(l, eta, rho)
        return _coulomb_asymptotic_basis(rm, k, l, eta)

    if kind == "shifted":
        k_use = float(meta.get("k_use", k))
        z = k_use * rm
        return z * spherical_jn(l, z), z * spherical_yn(l, z)

    if kind == "shifted_forbidden":
        z = k * rm
        return z * spherical_jn(l, z), z * spherical_yn(l, z)

    z = k * rm
    return z * spherical_jn(l, z), z * spherical_yn(l, z)


def _build_asymptotic_basis(rm: np.ndarray,
                            v_tail: np.ndarray | None,
                            energy: float,
                            l: int,
                            asymptotic: str,
                            match_v_tol: float,
                            coulomb_tol: float,
                            allow_shift: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Construct asymptotic basis functions for matching u(r).
    """
    k = np.sqrt(2.0 * energy)
    meta = _resolve_asymptotic_basis_meta(
        rm,
        v_tail,
        energy,
        asymptotic,
        match_v_tol,
        coulomb_tol,
        allow_shift,
    )
    u_f, u_g = _build_asymptotic_basis_from_meta(
        rm,
        energy,
        l,
        meta,
    )
    return u_f, u_g, meta


def _normalized_free_scattering_state(
    r: np.ndarray,
    energy: float,
    l: int,
) -> tuple[np.ndarray, float]:
    """Return one energy-normalized free radial state and its target amplitude."""
    k = np.sqrt(2.0 * float(energy))
    z = k * np.asarray(r, dtype=float)
    amp_target = np.sqrt(2.0 / np.pi) / np.sqrt(max(k, 1.0e-12))
    return amp_target * z * spherical_jn(int(l), z), float(amp_target)


def _match_scattering_u(u: np.ndarray,
                        r: np.ndarray,
                        v_eff: np.ndarray | None,
                        energy: float,
                        l: int,
                        match_fraction: float = 0.2,
                        match_slice: tuple[int, int] | None = None,
                        match_r_cut: float | None = None,
                        match_fraction_mode: str = "r",
                        match_width: float | None = None,
                        match_kr_min: float | None = 4.0,
                        match_v_tol: float | None = 1e-4,
                        match_min_points: int = 12,
                        match_asymptotic: str = "auto",
                        match_coulomb_tol: float = 0.1,
                        match_allow_shift: bool = True,
                        match_fallback: str = "free") -> tuple[np.ndarray, float, float, dict]:
    """
    Normalize u(r) to delta(E) and extract the phase shift.

    Parameters
    ----------
    u : ndarray
        Unnormalized u(r) = r R(r) on the grid (Bohr).
    r : ndarray
        Radial grid (Bohr).
    v_eff : ndarray or None
        Effective potential on r (Ha).
    energy : float
        Continuum energy (Ha).
    l : int
        Angular momentum.

    Returns
    -------
    u_norm, delta_l, amp_target, meta
        Normalized radial function, phase shift (radians),
        target amplitude, and match diagnostics.

    Notes
    -----
    The normalization targets δ(E) because A3 integrates over energy.
    The asymptotic amplitude is scaled to sqrt(2/pi)/sqrt(k_use).
    """
    if energy <= 0.0:
        return u, 0.0, np.nan, {"status": "energy<=0"}

    k = np.sqrt(2.0 * energy)
    if not np.isfinite(k) or k <= 0.0:
        return u, 0.0, np.nan, {"status": "invalid_k"}

    match_slice, meta = _select_match_window(
        r,
        v_eff,
        energy,
        l,
        match_slice,
        match_fraction,
        match_r_cut,
        match_width,
        match_fraction_mode,
        match_kr_min,
        match_v_tol,
        match_min_points,
    )

    if meta.get("fallback") and match_fallback.lower() == "free":
        k_use = float(k)
        u_free = k_use * r * spherical_jn(l, k_use * r)
        amp_target = np.sqrt(2.0 / np.pi)/ np.sqrt(max(k_use, 1e-12))
        u_norm = amp_target * u_free
        return u_norm, 0.0, amp_target, {
            **meta,
            "status": "fallback_free",
            "kind": "free",
            "k_use": k_use,
        }

    i0, i1 = match_slice
    rm = r[i0:i1]
    if rm.size < 2:
        if str(match_fallback).lower() == "free":
            u_free, amp_target = _normalized_free_scattering_state(r, energy, l)
            return u_free, 0.0, amp_target, {**meta, "status": "short_window_free"}
        return u, 0.0, np.nan, {**meta, "status": "short_window"}

    v_tail = v_eff[i0:i1] if v_eff is not None else None
    u_f, u_g, meta_basis = _build_asymptotic_basis(
        rm,
        v_tail,
        energy,
        l,
        match_asymptotic,
        float(match_v_tol if match_v_tol is not None else 0.0),
        float(match_coulomb_tol),
        bool(match_allow_shift),
    )

    # For E <= V_infty in shifted asymptotics, do not perform least-squares
    # amplitude extraction against an oscillatory basis (ill-posed). Use the
    # same safe free fallback shape used by window-fallback logic.
    if meta_basis.get("kind") == "shifted_forbidden":
        k_use = float(k)
        u_free = k_use * r * spherical_jn(l, k_use * r)
        amp_target = np.sqrt(2.0 / np.pi) / np.sqrt(max(k_use, 1e-12))
        u_norm = amp_target * u_free
        return u_norm, 0.0, amp_target, {
            **meta,
            **meta_basis,
            "status": "forbidden_shifted_free",
            "k_use": k_use,
        }

    amp, delta, match_status = _solve_match_amplitude_phase(
        u,
        i0,
        i1,
        u_f,
        u_g,
    )
    if match_status != "ok":
        if str(match_fallback).lower() == "free":
            u_free, amp_target = _normalized_free_scattering_state(r, energy, l)
            return u_free, 0.0, amp_target, {
                **meta,
                **meta_basis,
                "status": f"{match_status}_free",
            }
        return u, 0.0, np.nan, {**meta, **meta_basis, "status": str(match_status)}

    k_use = float(meta_basis.get("k_use", k))
    amp_target = np.sqrt(2.0 / np.pi)/ np.sqrt(max(k_use, 1e-12))
    scale = amp_target / amp
    return u * scale, delta, amp_target, {
        **meta,
        **meta_basis,
        "status": "ok",
        "amp": float(amp),
        "amp_target": float(amp_target),
        "delta": float(delta),
    }


def _prepare_match_plan_for_energy(r: np.ndarray,
                                   v_eff: np.ndarray | None,
                                   energy: float,
                                   l_cap: int,
                                   match_fraction: float,
                                   match_slice: tuple[int, int] | None,
                                   match_r_cut: float | None,
                                   match_fraction_mode: str,
                                   match_width: float | None,
                                   match_kr_min: float | None,
                                   match_v_tol: float | None,
                                   match_min_points: int,
                                   match_asymptotic: str,
                                   match_coulomb_tol: float,
                                   match_allow_shift: bool) -> tuple[list[tuple[int, int]], list[bool], list[dict | None], dict | None]:
    """
    Precompute match-window and asymptotic-basis metadata for one energy.
    """
    match_slices: list[tuple[int, int]] = []
    fallback_flags: list[bool] = []
    basis_meta_list: list[dict | None] = []

    free_i0: list[int] = []
    free_i1: list[int] = []
    free_l_max = -1
    fallback_count = 0
    l_cap = int(l_cap)
    base_i0, base_i1 = _resolve_match_base_slice(
        r,
        match_slice,
        match_fraction,
        match_r_cut,
        match_width,
        match_fraction_mode,
    )
    min_points = max(int(match_min_points), 2)
    asym_mode = str(match_asymptotic).lower()

    # (1) Fast path: no l- or energy-dependent constraints. All channels use
    # the same tail slice, and only asymptotic-basis metadata can vary.
    if (
        (match_kr_min is None and match_v_tol is None)
        or (not np.isfinite(energy))
        or energy <= 0.0
    ):
        match_slice_l = (base_i0, base_i1)
        match_slices = [match_slice_l] * (l_cap + 1)
        fallback_flags = [False] * (l_cap + 1)

        basis_meta_shared = None
        rm = r[base_i0:base_i1]
        if rm.size >= 2:
            v_tail = v_eff[base_i0:base_i1] if v_eff is not None else None
            basis_meta_shared = _resolve_asymptotic_basis_meta(
                rm,
                v_tail,
                energy,
                match_asymptotic,
                float(match_v_tol if match_v_tol is not None else 0.0),
                float(match_coulomb_tol),
                bool(match_allow_shift),
            )
            if str(basis_meta_shared.get("kind", "free")).lower() == "free":
                free_l_max = int(l_cap)
                free_i0.append(int(base_i0))
                free_i1.append(int(base_i1))
        basis_meta_list = [basis_meta_shared] * (l_cap + 1)
    else:
        # (2) General path: preprocess the energy-fixed tail constraints once,
        # then update only the l-dependent kr threshold inside the loop.
        k = np.sqrt(2.0 * energy)
        rm_base = np.asarray(r[base_i0:base_i1], dtype=float)
        v_tail_base = np.asarray(v_eff[base_i0:base_i1], dtype=float) if v_eff is not None else None
        mask_base = np.ones(rm_base.size, dtype=bool)
        if match_v_tol is not None and v_tail_base is not None:
            mask_base &= np.abs(v_tail_base) <= float(match_v_tol)
        true_runs = _contiguous_true_runs(mask_base)
        auto_free_from_tol = (
            asym_mode == "auto"
            and match_v_tol is not None
            and v_tail_base is not None
        )
        free_meta_shared = None
        if auto_free_from_tol:
            free_meta_shared = {
                "kind": "free",
                "k_use": float(k),
                "eta": None,
                "v_shift": None,
                "coulomb_backend": None,
            }

        basis_meta_cache: dict[tuple[int, int], dict | None] = {}
        for l_val in range(l_cap + 1):
            j_kr = 0
            if match_kr_min is not None:
                kr_min = max(float(match_kr_min), float(l_val + 1))
                j_kr = int(np.searchsorted(rm_base, kr_min / k))

            slice_rel = None
            for run_start, run_end in true_runs[::-1]:
                start_eff = max(int(run_start), int(j_kr))
                if int(run_end) - start_eff >= min_points:
                    slice_rel = (start_eff, int(run_end))
                    break

            if slice_rel is None and match_kr_min is not None:
                # The kr threshold is a conditioning preference, not a
                # physical validity condition: the exact free/Coulomb basis
                # used below is defined at low kr.  Relax only kr when a
                # sufficiently long |V|-valid tail window exists.  Preserve
                # the free fallback for cases where even the potential-tail
                # criterion cannot be satisfied.
                for run_start, run_end in true_runs[::-1]:
                    if int(run_end) - int(run_start) >= min_points:
                        slice_rel = (int(run_start), int(run_end))
                        break

            if slice_rel is None:
                fallback_count += 1
                match_slice_l = (base_i0, base_i1)
                match_slices.append(match_slice_l)
                fallback_flags.append(True)
                basis_meta_list.append(None)
                continue

            i0 = int(base_i0 + slice_rel[0])
            i1 = int(base_i0 + slice_rel[1])
            match_slice_l = (i0, i1)
            match_slices.append(match_slice_l)
            fallback_flags.append(False)

            basis_meta = None
            if auto_free_from_tol:
                basis_meta = free_meta_shared
            else:
                slice_key = (int(slice_rel[0]), int(slice_rel[1]))
                basis_meta = basis_meta_cache.get(slice_key)
                if basis_meta is None:
                    rm = rm_base[slice_rel[0]:slice_rel[1]]
                    if rm.size >= 2:
                        v_tail = v_tail_base[slice_rel[0]:slice_rel[1]] if v_tail_base is not None else None
                        basis_meta = _resolve_asymptotic_basis_meta(
                            rm,
                            v_tail,
                            energy,
                            match_asymptotic,
                            float(match_v_tol if match_v_tol is not None else 0.0),
                            float(match_coulomb_tol),
                            bool(match_allow_shift),
                        )
                    basis_meta_cache[slice_key] = basis_meta
            if basis_meta is not None:
                if str(basis_meta.get("kind", "free")).lower() == "free":
                    free_l_max = max(free_l_max, int(l_val))
                    free_i0.append(int(i0))
                    free_i1.append(int(i1))
            basis_meta_list.append(basis_meta)

    free_basis_cache = None
    if free_l_max >= 0 and free_i0 and free_i1:
        i0_union = int(min(free_i0))
        i1_union = int(max(free_i1))
        z_union = np.sqrt(2.0 * energy) * r[i0_union:i1_union]
        j_tab, y_tab = _free_bessel_tables_numba(z_union, int(free_l_max))
        free_basis_cache = {
            "i0": i0_union,
            "z": z_union,
            "j_tab": j_tab,
            "y_tab": y_tab,
        }
    elif fallback_count > 0:
        # When every channel falls back to the free solution (common on the
        # first cold-start SCF iterate with the initial -Z/r guess), build the
        # exact free basis once on the full grid and let all fallback channels
        # reuse it instead of calling spherical_jn(l, k r) independently.
        z_full = np.sqrt(2.0 * energy) * r
        j_tab, y_tab = _free_bessel_tables_numba(z_full, int(l_cap))
        free_basis_cache = {
            "i0": 0,
            "z": z_full,
            "j_tab": j_tab,
            "y_tab": y_tab,
        }

    return match_slices, fallback_flags, basis_meta_list, free_basis_cache


def _match_scattering_u_preplanned(u: np.ndarray,
                                   r: np.ndarray,
                                   energy: float,
                                   l: int,
                                   match_slice: tuple[int, int],
                                   match_meta: dict,
                                   basis_meta: dict | None,
                                   match_fallback: str = "free",
                                   free_basis_cache: dict | None = None) -> tuple[np.ndarray, float, float, dict]:
    """
    Normalize u(r) using precomputed match-window and asymptotic metadata.
    """
    if energy <= 0.0:
        return u, 0.0, np.nan, {"status": "energy<=0"}

    k = np.sqrt(2.0 * energy)
    if not np.isfinite(k) or k <= 0.0:
        return u, 0.0, np.nan, {"status": "invalid_k"}

    meta = dict(match_meta)
    if meta.get("fallback") and str(match_fallback).lower() == "free":
        k_use = float(k)
        if (
            free_basis_cache is not None
            and int(free_basis_cache.get("i0", -1)) == 0
            and np.asarray(free_basis_cache.get("z", []), dtype=float).size == r.size
            and int(np.asarray(free_basis_cache.get("j_tab", np.empty((0, 0))), dtype=float).shape[0]) > int(l)
        ):
            z = np.asarray(free_basis_cache["z"], dtype=float)
            j_tab = np.asarray(free_basis_cache["j_tab"], dtype=float)
            u_free = z * j_tab[int(l)]
        else:
            u_free = k_use * r * spherical_jn(l, k_use * r)
        amp_target = np.sqrt(2.0 / np.pi) / np.sqrt(max(k_use, 1e-12))
        u_norm = amp_target * u_free
        return u_norm, 0.0, amp_target, {
            **meta,
            "status": "fallback_free",
            "kind": "free",
            "k_use": k_use,
        }

    i0, i1 = match_slice
    rm = r[i0:i1]
    if rm.size < 2:
        if str(match_fallback).lower() == "free":
            u_free, amp_target = _normalized_free_scattering_state(r, energy, l)
            return u_free, 0.0, amp_target, {**meta, "status": "short_window_free"}
        return u, 0.0, np.nan, {**meta, "status": "short_window"}
    if basis_meta is None:
        if str(match_fallback).lower() == "free":
            u_free, amp_target = _normalized_free_scattering_state(r, energy, l)
            return u_free, 0.0, amp_target, {**meta, "status": "missing_basis_free"}
        return u, 0.0, np.nan, {**meta, "status": "missing_basis_meta"}

    basis_meta_local = dict(basis_meta)
    kind = str(basis_meta_local.get("kind", "free")).lower()

    if kind == "free" and free_basis_cache is not None:
        base_i0 = int(free_basis_cache["i0"])
        off0 = int(i0 - base_i0)
        off1 = int(i1 - base_i0)
        z = np.asarray(free_basis_cache["z"][off0:off1], dtype=float)
        u_f = z * np.asarray(free_basis_cache["j_tab"][l, off0:off1], dtype=float)
        u_g = z * np.asarray(free_basis_cache["y_tab"][l, off0:off1], dtype=float)
    else:
        u_f, u_g = _build_asymptotic_basis_from_meta(
            rm,
            energy,
            l,
            basis_meta_local,
        )

    if kind == "shifted_forbidden":
        k_use = float(k)
        u_free = k_use * r * spherical_jn(l, k_use * r)
        amp_target = np.sqrt(2.0 / np.pi) / np.sqrt(max(k_use, 1e-12))
        u_norm = amp_target * u_free
        return u_norm, 0.0, amp_target, {
            **meta,
            **basis_meta_local,
            "status": "forbidden_shifted_free",
            "k_use": k_use,
        }

    amp, delta, match_status = _solve_match_amplitude_phase(
        u,
        i0,
        i1,
        u_f,
        u_g,
    )
    if match_status != "ok":
        if str(match_fallback).lower() == "free":
            u_free, amp_target = _normalized_free_scattering_state(r, energy, l)
            return u_free, 0.0, amp_target, {
                **meta,
                **basis_meta_local,
                "status": f"{match_status}_free",
            }
        return u, 0.0, np.nan, {**meta, **basis_meta_local, "status": str(match_status)}

    k_use = float(basis_meta_local.get("k_use", k))
    amp_target = np.sqrt(2.0 / np.pi) / np.sqrt(max(k_use, 1e-12))
    scale = amp_target / amp
    return u * scale, delta, amp_target, {
        **meta,
        **basis_meta_local,
        "status": "ok",
        "amp": float(amp),
        "amp_target": float(amp_target),
        "delta": float(delta),
    }


def _match_scattering_scale_preplanned(u: np.ndarray,
                                       r: np.ndarray,
                                       energy: float,
                                       l: int,
                                       match_slice: tuple[int, int],
                                       fallback_free: bool,
                                       basis_meta: dict | None,
                                       match_fallback: str = "free",
                                       free_basis_cache: dict | None = None) -> tuple[np.ndarray | None, float, float]:
    """
    Lightweight production fast path for matching.

    Returns
    -------
    u_override, delta, scale
        When ``u_override is None``, accumulate the raw propagated ``u`` with
        the returned scalar ``scale``. Rare fallback branches may return an
        already-normalized ``u_override`` with ``scale=1``.
    """
    if energy <= 0.0:
        return None, 0.0, 1.0

    k = np.sqrt(2.0 * energy)
    if not np.isfinite(k) or k <= 0.0:
        return None, 0.0, 1.0

    if bool(fallback_free) and str(match_fallback).lower() == "free":
        k_use = float(k)
        if (
            free_basis_cache is not None
            and int(free_basis_cache.get("i0", -1)) == 0
            and np.asarray(free_basis_cache.get("z", []), dtype=float).size == r.size
            and int(np.asarray(free_basis_cache.get("j_tab", np.empty((0, 0))), dtype=float).shape[0]) > int(l)
        ):
            z = np.asarray(free_basis_cache["z"], dtype=float)
            j_tab = np.asarray(free_basis_cache["j_tab"], dtype=float)
            u_free = z * j_tab[int(l)]
        else:
            u_free = k_use * r * spherical_jn(l, k_use * r)
        amp_target = np.sqrt(2.0 / np.pi) / np.sqrt(max(k_use, 1e-12))
        return amp_target * u_free, 0.0, 1.0

    i0, i1 = match_slice
    rm = r[i0:i1]
    if rm.size < 2 or basis_meta is None:
        if str(match_fallback).lower() == "free":
            u_free, _ = _normalized_free_scattering_state(r, energy, l)
            return u_free, 0.0, 1.0
        return None, 0.0, 1.0

    kind = str(basis_meta.get("kind", "free")).lower()
    if kind == "free" and free_basis_cache is not None:
        base_i0 = int(free_basis_cache["i0"])
        off0 = int(i0 - base_i0)
        off1 = int(i1 - base_i0)
        z = np.asarray(free_basis_cache["z"][off0:off1], dtype=float)
        j_l = np.asarray(free_basis_cache["j_tab"][l, off0:off1], dtype=float)
        y_l = np.asarray(free_basis_cache["y_tab"][l, off0:off1], dtype=float)
        status_code, amp, delta = _solve_match_amplitude_phase_free_numba(
            np.asarray(u, dtype=float),
            int(i0),
            z,
            j_l,
            y_l,
        )
        if status_code == 0:
            k_use = float(basis_meta.get("k_use", k))
            amp_target = np.sqrt(2.0 / np.pi) / np.sqrt(max(k_use, 1.0e-12))
            return None, float(delta), float(amp_target / amp)
        u_f = z * j_l
        u_g = z * y_l
    else:
        u_f, u_g = _build_asymptotic_basis_from_meta(rm, energy, l, basis_meta)

    if kind == "shifted_forbidden":
        k_use = float(k)
        u_free = k_use * r * spherical_jn(l, k_use * r)
        amp_target = np.sqrt(2.0 / np.pi) / np.sqrt(max(k_use, 1.0e-12))
        return amp_target * u_free, 0.0, 1.0

    amp, delta, match_status = _solve_match_amplitude_phase(
        u,
        i0,
        i1,
        u_f,
        u_g,
    )
    if match_status != "ok":
        if str(match_fallback).lower() == "free":
            u_free, _ = _normalized_free_scattering_state(r, energy, l)
            return u_free, 0.0, 1.0
        return None, 0.0, 1.0

    k_use = float(basis_meta.get("k_use", k))
    amp_target = np.sqrt(2.0 / np.pi) / np.sqrt(max(k_use, 1.0e-12))
    return None, float(delta), float(amp_target / amp)


def _compute_l_cap(energy: float,
                   l_max: int,
                   r: np.ndarray,
                   l_pad: int,
                   match_slice: tuple[int, int] | None,
                   match_r_cut: float | None,
                   match_fraction: float,
                   match_fraction_mode: str,
                   match_width: float | None,
                   match_min_points: int,
                   match_kr_min: float | None,
                   match_v_tol: float | None,
                   v_eff: np.ndarray | None,
                   strategy: str) -> int:
    """
    Compute an energy-dependent l cutoff to avoid unstable tail matching.

    Strategies
    ----------
    "match" (default):
        Use the end of the matching window r_m_end and a safety margin
        Δr ≈ match_min_points * dr to define
        l_cap(E) = floor(k * (r_m_end - Δr) - 1). This skips large l
        whose oscillatory region lies beyond the match window.
        If match_r_cut is provided, r_m_end defaults to match_r_cut + match_width.
    "rmax":
        l_cap(E) = ceil(k * Rmax + l_pad) (legacy behavior).
    "none":
        Use the global l_max with no energy-dependent truncation.
    """
    if energy <= 0.0 or r.size == 0:
        return 0

    k = np.sqrt(2.0 * energy)
    if not np.isfinite(k) or k <= 0.0:
        return 0

    strategy = str(strategy).lower()
    if strategy in ("none", "lmax", "full"):
        l_cap = int(l_max)
    elif strategy in ("rmax", "global"):
        l_cap = int(np.ceil(k * r[-1] + l_pad))
    else:
        # Tail-window-based cutoff (default).
        if match_slice is not None:
            i0, i1 = int(match_slice[0]), int(match_slice[1])
        elif match_r_cut is not None:
            i0 = int(np.searchsorted(r, float(match_r_cut)))
            if match_width is not None:
                i1 = int(np.searchsorted(r, float(match_r_cut + match_width)))
            else:
                i1 = int(r.size)
        else:
            mode = str(match_fraction_mode).lower()
            if mode in ("r", "radius", "physical"):
                r_start = float((1.0 - match_fraction) * r[-1])
                i0 = int(np.searchsorted(r, r_start))
            else:
                i0 = int(max(1, (1.0 - match_fraction) * r.size))
            i1 = int(r.size)

        i0 = max(1, min(i0, r.size - 1))
        i1 = max(i0 + 1, min(i1, r.size))

        # If a tail potential tolerance is provided, shift i0 to the
        # first point in the tail where |V_eff| is sufficiently small.
        if match_v_tol is not None and v_eff is not None:
            mask = np.abs(v_eff[i0:i1]) <= float(match_v_tol)
            if np.any(mask):
                i0 = i0 + int(np.argmax(mask))

        r_end = float(r[i1 - 1])
        r_end = min(r_end, float(r[-1]))

        # Estimate a safety margin Δr from the local spacing.
        dr_local = np.median(np.diff(r[i0:i1])) if (i1 - i0) >= 2 else np.median(np.diff(r))
        dr_local = float(dr_local) if np.isfinite(dr_local) else float(np.median(np.diff(r)))
        delta_r = max(int(match_min_points), 2) * dr_local
        r_eff = max(float(r[i0]), r_end - delta_r)

        if match_kr_min is not None:
            r_eff = max(r_eff, float(match_kr_min) / k)
        r_eff = min(r_eff, float(r[-1]))

        # Require k r_eff >= l + 1 ⇒ l_cap = floor(k r_eff - 1).
        l_cap = int(np.floor(k * r_eff - 1.0))

    l_cap = max(0, min(int(l_cap), int(l_max)))
    return l_cap


def _scattering_density_and_phase(v_eff: np.ndarray,
                                  r: np.ndarray,
                                  mu: float,
                                  temperature: float,
                                  energy: float,
                                  l_max: int,
                                  grid_kind: str,
                                  grid_step: float,
                                  l_pad: int,
                                  match_fraction: float,
                                  match_slice: tuple[int, int] | None,
                                  match_r_cut: float | None,
                                  match_fraction_mode: str,
                                  match_width: float | None,
                                  match_kr_min: float | None,
                                  match_v_tol: float | None,
                                  match_min_points: int,
                                  match_asymptotic: str,
                                  match_coulomb_tol: float,
                                  match_allow_shift: bool,
                                  match_fallback: str,
                                  prop_rescale_limit: float | None,
                                  apply_occ: bool = True,
                                  l_cap_strategy: str = "match",
                                  numerov_geom: dict[str, np.ndarray] | None = None,
                                  perf_accum: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuum density and phase shifts at a single energy.

    Returns
    -------
    n_r : ndarray
        Energy-resolved density contribution at this energy.
    delta_vec : ndarray
        Phase shifts δ_l for l=0..l_max.

    Notes
    -----
    - Numerov propagation per l-channel.
    - Asymptotic matching to extract phase shifts and normalization.
    - If apply_occ=True, includes Fermi-Dirac occupancy.
    - l is truncated at an energy-dependent l_cap, controlled by
      l_cap_strategy ("match" default). In the default mode, r_m is taken
      as the start of the tail-matching window and l_cap(E) = floor(k r_m - 1).
    """
    numerov_geom = numerov_geom if numerov_geom is not None else _prepare_numerov_geometry(r, v_eff)
    r_eval = np.asarray(numerov_geom["r"], dtype=float)
    v_eval = np.asarray(numerov_geom["v_eff"], dtype=float)
    t_eval = time.perf_counter() if perf_accum is not None else 0.0
    if energy <= 0.0:
        return np.zeros_like(r_eval, dtype=float), np.zeros(l_max + 1, dtype=float)
    if grid_kind != "sqrt":
        raise ValueError("Continuum scattering Numerov supports only grid_kind='sqrt'.")

    # FD occupation for this energy (optional).
    occ = 1.0
    if apply_occ:
        occ = fermi_dirac(np.array([energy], dtype=float), mu, temperature)[0]

    # (1) Determine the actual l workload and precompute the tail-matching
    # plan for this energy before entering the per-l propagation loop.
    t_stage = time.perf_counter() if perf_accum is not None else 0.0
    l_cap = _compute_l_cap(
        energy,
        l_max,
        r_eval,
        l_pad,
        match_slice,
        match_r_cut,
        match_fraction,
        match_fraction_mode,
        match_width,
        match_min_points,
        match_kr_min,
        match_v_tol,
        v_eval,
        l_cap_strategy,
    )
    match_slices, fallback_flags, basis_meta_list, free_basis_cache = _prepare_match_plan_for_energy(
        r_eval,
        v_eval,
        energy,
        l_cap,
        match_fraction,
        match_slice,
        match_r_cut,
        match_fraction_mode,
        match_width,
        match_kr_min,
        match_v_tol,
        match_min_points,
        match_asymptotic,
        match_coulomb_tol,
        match_allow_shift,
    )
    if perf_accum is not None:
        perf_accum["plan_s"] = float(perf_accum.get("plan_s", 0.0) + (time.perf_counter() - t_stage))
    r_quarter = np.asarray(numerov_geom["r_quarter"], dtype=float)
    inv_r = np.asarray(numerov_geom["inv_r"], dtype=float)
    inv_r2 = np.asarray(numerov_geom["inv_r2"], dtype=float)
    w_base = np.asarray(numerov_geom["r8"], dtype=float) * float(energy) + np.asarray(numerov_geom["v_term"], dtype=float)
    origin_charge = float(np.asarray(numerov_geom.get("origin_charge", 0.0)))

    # (2) Propagate the partial waves for this energy, then match and
    # accumulate them one channel at a time.
    sum_l = np.zeros_like(r_eval, dtype=float)
    delta_vec = np.zeros(l_max + 1, dtype=float)
    rescale_limit = float(prop_rescale_limit) if prop_rescale_limit is not None else 1e6
    u_batch = None
    if l_cap >= 2:
        l_vals = np.arange(l_cap + 1, dtype=np.int64)
        t_stage = time.perf_counter() if perf_accum is not None else 0.0
        u_batch = _numerov_propagate_sqrt_wbase_batch_numba(
            r_eval,
            r_quarter,
            inv_r,
            w_base,
            l_vals,
            grid_step,
            rescale_limit=rescale_limit,
            origin_charge=origin_charge,
        )
        if perf_accum is not None:
            perf_accum["propagate_s"] = float(perf_accum.get("propagate_s", 0.0) + (time.perf_counter() - t_stage))
    for l in range(l_cap + 1):
        if u_batch is None:
            t_stage = time.perf_counter() if perf_accum is not None else 0.0
            u = _numerov_propagate_sqrt_wbase_numba(
                r_eval,
                r_quarter,
                inv_r,
                w_base,
                l,
                grid_step,
                rescale_limit=rescale_limit,
                origin_charge=origin_charge,
            )
            if perf_accum is not None:
                perf_accum["propagate_s"] = float(perf_accum.get("propagate_s", 0.0) + (time.perf_counter() - t_stage))
        else:
            u = u_batch[l]

        # Asymptotic matching → normalized u and phase shift.
        t_stage = time.perf_counter() if perf_accum is not None else 0.0
        u_override, delta, scale = _match_scattering_scale_preplanned(
            u,
            r_eval,
            energy,
            l,
            match_slices[l],
            fallback_flags[l],
            basis_meta_list[l],
            match_fallback=match_fallback,
            free_basis_cache=free_basis_cache,
        )
        if perf_accum is not None:
            perf_accum["match_s"] = float(perf_accum.get("match_s", 0.0) + (time.perf_counter() - t_stage))
        delta_vec[l] = delta
        # Add l-channel contribution.
        t_stage = time.perf_counter() if perf_accum is not None else 0.0
        factor = (2.0 * (2 * l + 1)) / (4.0 * np.pi)
        if u_override is None:
            _accumulate_scattering_density_inplace(
                sum_l,
                u,
                inv_r2,
                factor * scale * scale,
            )
        else:
            _accumulate_scattering_density_inplace(
                sum_l,
                u_override,
                inv_r2,
                factor,
            )
        if perf_accum is not None:
            perf_accum["accumulate_s"] = float(perf_accum.get("accumulate_s", 0.0) + (time.perf_counter() - t_stage))

    if perf_accum is not None:
        perf_accum["eval_total_s"] = float(perf_accum.get("eval_total_s", 0.0) + (time.perf_counter() - t_eval))

    return occ * sum_l, delta_vec


def transport_cross_section_from_deltas(k: float, delta_vec: np.ndarray) -> float:
    """
    Compute the momentum-transport cross section from phase shifts (Eq. C4).

    Parameters
    ----------
    k : float
        Wave number (Bohr^-1).
    delta_vec : ndarray
        Phase shifts for l=0..l_max (radians).

    Returns
    -------
    float
        Transport cross section sigma_tr (Bohr^2).
    """
    k = float(k)
    if k <= 0.0:
        return 0.0
    delta_vec = np.asarray(delta_vec, dtype=float)
    if delta_vec.size < 2:
        return 0.0
    diffs = delta_vec[1:] - delta_vec[:-1]
    l_vals = np.arange(0, diffs.size, dtype=float)
    sigma_tr = (4.0 * np.pi / (k ** 2)) * np.sum((l_vals + 1.0) * np.sin(diffs) ** 2)
    return float(sigma_tr)


def gamma_from_phase_shift_cache(energy_cache: dict[float, tuple[np.ndarray, np.ndarray]],
                                 mu: float,
                                 temperature: float,
                                 n_i: float,
                                 n0: float,
                                 n0_floor: float = 0.0) -> float:
    """
    Estimate the bound-state broadening gamma from scattering phase shifts.

    Notes
    -----
    Uses the Starrett2013 Appendix C approach:
      tau(k) = 1 / (n_i * v * sigma_tr(k)), v = k
      sigma_dc = (1 / (3 pi^2)) * integral dk k^4 (-dg/dE) tau(k)
      gamma = 1 / tau, with tau = sigma_dc / n0
    """
    n0_floor = max(float(n0_floor), 0.0)
    n0_eff = max(float(n0), n0_floor)
    if n_i <= 0.0 or n0_eff <= 0.0:
        return 0.0
    if not energy_cache:
        return 0.0

    energies = []
    deltas = []
    for energy, entry in energy_cache.items():
        if energy <= 0.0:
            continue
        _, delta_vec = entry
        energies.append(float(energy))
        deltas.append(np.asarray(delta_vec, dtype=float))

    if len(energies) < 2:
        return 0.0

    order = np.argsort(energies)
    e_arr = np.asarray(energies, dtype=float)[order]
    delta_arr = [deltas[i] for i in order]
    k_arr = np.sqrt(2.0 * e_arr)

    g = fermi_dirac(e_arr, mu, temperature)
    dgde = g * (1.0 - g) / max(float(temperature), 1e-12)

    tau_k = np.zeros_like(k_arr)
    for i, k_val in enumerate(k_arr):
        sigma_tr = transport_cross_section_from_deltas(k_val, delta_arr[i])
        if sigma_tr <= 0.0 or k_val <= 0.0:
            tau_k[i] = 0.0
        else:
            # 1/tau = n_i * v * sigma_tr, with v = k in atomic units.
            tau_k[i] = 1.0 / (n_i * k_val * sigma_tr)

    integrand = (k_arr ** 4) * dgde * tau_k
    sigma_dc = (1.0 / (3.0 * np.pi ** 2)) * _trapz(integrand, k_arr)

    if sigma_dc <= 0.0:
        return 0.0
    tau = sigma_dc / n0_eff
    if tau <= 0.0:
        return 0.0
    return float(1.0 / tau)


def _scattering_density_at_energy(v_eff: np.ndarray,
                                  r: np.ndarray,
                                  mu: float,
                                  temperature: float,
                                  energy: float,
                                  l_max: int,
                                  grid_kind: str,
                                  grid_step: float,
                                  l_pad: int,
                                  match_fraction: float,
                                  match_slice: tuple[int, int] | None,
                                  match_r_cut: float | None,
                                  match_fraction_mode: str,
                                  match_width: float | None,
                                  match_kr_min: float | None,
                                  match_v_tol: float | None,
                                  match_min_points: int,
                                  match_asymptotic: str,
                                  match_coulomb_tol: float,
                                  match_allow_shift: bool,
                                  match_fallback: str,
                                  prop_rescale_limit: float | None,
                                  apply_occ: bool = True,
                                  l_cap_strategy: str = "match",
                                  numerov_geom: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """
    Continuum density contribution at a single energy.

    Thin wrapper around _scattering_density_and_phase; returns only n_e(r).
    """
    n_e, _ = _scattering_density_and_phase(
        v_eff,
        r,
        mu,
        temperature,
        energy,
        l_max,
        grid_kind,
        grid_step,
        l_pad,
        match_fraction,
        match_slice,
        match_r_cut,
        match_fraction_mode,
        match_width,
        match_kr_min,
        match_v_tol,
        match_min_points,
        match_asymptotic,
        match_coulomb_tol,
        match_allow_shift,
        match_fallback,
        prop_rescale_limit,
        apply_occ=apply_occ,
        l_cap_strategy=l_cap_strategy,
        numerov_geom=numerov_geom,
    )
    return n_e


# Globals for multiprocessing continuum evaluation.
_SCATTER_MP = {}


def _init_scatter_worker(v_eff: np.ndarray,
                         r: np.ndarray,
                         mu: float,
                         temperature: float,
                         l_max: int,
                         grid_kind: str,
                         grid_step: float,
                         l_pad: int,
                         match_fraction: float,
                         match_slice: tuple[int, int] | None,
                         match_r_cut: float | None,
                         match_fraction_mode: str,
                         match_width: float | None,
                         match_kr_min: float | None,
                         match_v_tol: float | None,
                         match_min_points: int,
                         match_asymptotic: str,
                         match_coulomb_tol: float,
                         match_allow_shift: bool,
                         match_fallback: str,
                         prop_rescale_limit: float | None,
                         apply_occ: bool = True,
                         l_cap_strategy: str = "match") -> None:
    numerov_geom = _prepare_numerov_geometry(r, v_eff)
    _SCATTER_MP["v_eff"] = np.asarray(numerov_geom["v_eff"], dtype=float)
    _SCATTER_MP["r"] = np.asarray(numerov_geom["r"], dtype=float)
    _SCATTER_MP["numerov_geom"] = numerov_geom
    _SCATTER_MP["mu"] = mu
    _SCATTER_MP["temperature"] = temperature
    _SCATTER_MP["l_max"] = l_max
    _SCATTER_MP["grid_kind"] = grid_kind
    _SCATTER_MP["grid_step"] = grid_step
    _SCATTER_MP["l_pad"] = l_pad
    _SCATTER_MP["match_fraction"] = match_fraction
    _SCATTER_MP["match_slice"] = match_slice
    _SCATTER_MP["match_r_cut"] = match_r_cut
    _SCATTER_MP["match_fraction_mode"] = match_fraction_mode
    _SCATTER_MP["match_width"] = match_width
    _SCATTER_MP["match_kr_min"] = match_kr_min
    _SCATTER_MP["match_v_tol"] = match_v_tol
    _SCATTER_MP["match_min_points"] = match_min_points
    _SCATTER_MP["match_asymptotic"] = match_asymptotic
    _SCATTER_MP["match_coulomb_tol"] = match_coulomb_tol
    _SCATTER_MP["match_allow_shift"] = match_allow_shift
    _SCATTER_MP["match_fallback"] = match_fallback
    _SCATTER_MP["prop_rescale_limit"] = prop_rescale_limit
    _SCATTER_MP["apply_occ"] = apply_occ
    _SCATTER_MP["l_cap_strategy"] = l_cap_strategy


def _scatter_worker(e_val: float) -> tuple[float, np.ndarray, np.ndarray]:
    n_e, delta_vec = _scattering_density_and_phase(
        _SCATTER_MP["v_eff"],
        _SCATTER_MP["r"],
        _SCATTER_MP["mu"],
        _SCATTER_MP["temperature"],
        float(e_val),
        _SCATTER_MP["l_max"],
        _SCATTER_MP["grid_kind"],
        _SCATTER_MP["grid_step"],
        _SCATTER_MP["l_pad"],
        _SCATTER_MP["match_fraction"],
        _SCATTER_MP["match_slice"],
        _SCATTER_MP["match_r_cut"],
        _SCATTER_MP.get("match_fraction_mode", "r"),
        _SCATTER_MP.get("match_width", None),
        _SCATTER_MP["match_kr_min"],
        _SCATTER_MP["match_v_tol"],
        _SCATTER_MP["match_min_points"],
        _SCATTER_MP["match_asymptotic"],
        _SCATTER_MP["match_coulomb_tol"],
        _SCATTER_MP["match_allow_shift"],
        _SCATTER_MP["match_fallback"],
        _SCATTER_MP["prop_rescale_limit"],
        apply_occ=_SCATTER_MP.get("apply_occ", True),
        l_cap_strategy=_SCATTER_MP.get("l_cap_strategy", "match"),
        numerov_geom=_SCATTER_MP.get("numerov_geom", None),
    )
    return float(e_val), n_e, delta_vec


def _adaptive_shard_worker(
    task: tuple[int, float, float, dict]
) -> tuple[int, np.ndarray, dict, list[tuple[float, np.ndarray, np.ndarray]] | None]:
    """
    Worker for coarse-grained adaptive sharding over disjoint energy intervals.

    Each worker integrates one independent [e0, e1] sub-interval and returns
    its continuum contribution. Intervals only share endpoints, so summing all
    shard contributions reconstructs the full [E_min, E_max] integral.
    """
    shard_idx, e0, e1, base_kwargs = task
    kwargs = dict(base_kwargs)
    # Optional export: when the caller requests basis reuse, each shard returns
    # its local (E -> (n_E(r), delta_l(E))) samples so the parent process can
    # rebuild a global cache/basis table.
    collect_cache_samples = bool(kwargs.pop("_collect_cache_samples", False))
    kwargs["e_min"] = float(e0)
    kwargs["e_max"] = float(e1)
    # Force serial inside each shard to avoid nested multiprocessing pools.
    kwargs["n_jobs"] = None
    local_cache: dict[float, tuple[np.ndarray, np.ndarray]] | None = {} if collect_cache_samples else None
    kwargs["energy_cache"] = local_cache
    kwargs["adaptive_parallel_mode"] = "batch"
    kwargs["adaptive_shards"] = None
    t0 = time.perf_counter()
    n_r, meta = continuum_density_scattering_adaptive(**kwargs)
    meta = dict(meta)
    meta["wall_s"] = float(time.perf_counter() - t0)
    samples = None
    if collect_cache_samples and local_cache:
        # Sort by energy so parent-side merge is deterministic and reproducible.
        samples = [
            (float(e), np.asarray(vals[0], dtype=float), np.asarray(vals[1], dtype=float))
            for e, vals in sorted(local_cache.items(), key=lambda kv: float(kv[0]))
        ]
    return int(shard_idx), n_r, meta, samples


def continuum_density_scattering(v_eff: np.ndarray,
                                 r: np.ndarray,
                                 mu: float,
                                 temperature: float,
                                 e_grid: np.ndarray,
                                 l_max: int,
                                 grid_kind: str = "sqrt",
                                 grid_step: float | None = None,
                                 l_pad: int = 2,
                                 match_fraction: float = 0.2,
                                 match_slice: tuple[int, int] | None = None,
                                 match_r_cut: float | None = None,
                                 match_fraction_mode: str = "r",
                                 match_width: float | None = None,
                                 match_kr_min: float | None = 4.0,
                                 match_v_tol: float | None = 1e-4,
                                 match_min_points: int = 12,
                                 match_asymptotic: str = "auto",
                                 match_coulomb_tol: float = 0.1,
                                 match_allow_shift: bool = True,
                                 match_fallback: str = "free",
                                 prop_rescale_limit: float | None = 1e6,
                                 l_cap_strategy: str = "match",
                                 energy_cache: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
                                 n_jobs: int | None = None,
                                 apply_occ: bool = True) -> np.ndarray:
    """
    Continuum density using Numerov scattering solutions in V_eff(r).

    Inputs
    ------
    v_eff, r : ndarray
        Effective potential and radial grid.
    mu, temperature : float
        Chemical potential and temperature (Ha).
    e_grid : ndarray
        Energy grid for the A3 integral.
    l_max : int
        Partial-wave cutoff (l=0..l_max).

    Notes
    -----
    energy_cache can be supplied to reuse (n_e, delta) evaluations across
    repeated calls with the same V_eff and energy grid.
    l_cap_strategy controls the energy-dependent partial-wave cutoff
    ("match" default, see _compute_l_cap).
    """
    r = np.asarray(r)
    v_eff = np.asarray(v_eff)
    e_grid = np.asarray(e_grid)
    if grid_kind != "sqrt":
        raise ValueError("continuum_density_scattering supports only grid_kind='sqrt'.")
    if grid_step is None:
        grid_step = float(np.sqrt(r[1]) - np.sqrt(r[0]))

    # Allocate per-energy density contributions.
    n_e_r = np.zeros((e_grid.size, r.size), dtype=float)
    cache = energy_cache if energy_cache is not None else {}
    # Parallel branch: distribute energies across processes.
    if n_jobs is not None and int(n_jobs) > 1:
        if cache:
            warnings.warn("energy_cache ignored when n_jobs>1 for continuum scattering.")
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=int(n_jobs),
            initializer=_init_scatter_worker,
            initargs=(
                v_eff,
                r,
                mu,
                temperature,
                l_max,
                grid_kind,
                grid_step,
                l_pad,
                match_fraction,
                match_slice,
                match_r_cut,
                match_fraction_mode,
                match_width,
                match_kr_min,
                match_v_tol,
                match_min_points,
                match_asymptotic,
                match_coulomb_tol,
                match_allow_shift,
                match_fallback,
                prop_rescale_limit,
                apply_occ,
                l_cap_strategy,
            ),
        ) as pool:
            results = pool.map(_scatter_worker, [float(e) for e in e_grid])
        # Collect per-energy densities.
        for i, (e_val, n_e, delta_vec) in enumerate(results):
            n_e_r[i] = n_e
            cache[e_val] = (n_e, delta_vec)
    else:
        # Serial branch with optional cache reuse.
        for i, e in enumerate(e_grid):
            e_val = float(e)
            if e_val in cache:
                cached = cache[e_val]
                n_e_r[i] = cached[0] if isinstance(cached, tuple) else cached
                continue
            n_e, delta_vec = _scattering_density_and_phase(
                v_eff,
                r,
                mu,
                temperature,
                e_val,
                l_max,
                grid_kind,
                grid_step,
                l_pad,
                match_fraction,
                match_slice,
                match_r_cut,
                match_fraction_mode,
                match_width,
                match_kr_min,
                match_v_tol,
                match_min_points,
                match_asymptotic,
                match_coulomb_tol,
                match_allow_shift,
                match_fallback,
                prop_rescale_limit,
                apply_occ=apply_occ,
                l_cap_strategy=l_cap_strategy,
            )
            cache[e_val] = (n_e, delta_vec)
            n_e_r[i] = n_e

    # Energy integration (A3) to get n_cont(r).
    n_r = _trapz(n_e_r, e_grid, axis=0)
    return n_r


def continuum_density_scattering_adaptive(v_eff: np.ndarray,
                                          r: np.ndarray,
                                          mu: float,
                                          temperature: float,
                                          e_min: float,
                                          e_max: float,
                                          l_max: int,
                                          grid_kind: str = "sqrt",
                                          grid_step: float | None = None,
                                          l_pad: int = 2,
                                          match_fraction: float = 0.2,
                                          match_slice: tuple[int, int] | None = None,
                                          match_r_cut: float | None = None,
                                          match_fraction_mode: str = "r",
                                          match_width: float | None = None,
                                          match_kr_min: float | None = 4.0,
                                          match_v_tol: float | None = 1e-4,
                                          match_min_points: int = 12,
                                          match_asymptotic: str = "auto",
                                          match_coulomb_tol: float = 0.1,
                                          match_allow_shift: bool = True,
                                          match_fallback: str = "free",
                                          prop_rescale_limit: float | None = 1e6,
                                          l_cap_strategy: str = "match",
                                          e_tol: float = 1e-3,
                                          e_max_depth: int = 10,
                                          e_min_width: float = 1e-4,
                                          n_e_base: int = 8,
                                          e_base_grid: str = "linear",
                                          near_zero_log_grid: bool = True,
                                          near_zero_log_points_per_decade: int = 4,
                                          near_zero_log_max_nodes: int = 24,
                                          near_zero_log_max_energy: float | None = 1.0e-2,
                                          resonance_tol: float | None = None,
                                          resonance_r_fractions: tuple[float, ...] | None = (0.25, 0.5, 0.75),
                                          resonance_floor: float = 1e-8,
                                          delta_tol: float | None = np.pi / 2.0,
                                          delta_mode: str = "max",
                                          adaptive_mode: str = "simpson",
                                          bisection_max_depth: int | None = None,
                                          resonance_window_factor: float = 12.0,
                                          resonance_window_min: float | None = None,
                                          resonance_window_max: float | None = None,
                                          resonance_max_windows: int | None = None,
                                          energy_cache: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
                                          n_jobs: int | None = None,
                                          adaptive_parallel_mode: str = "batch",
                                          adaptive_shards: int | None = None,
                                          adaptive_shard_policy: str = "egrid",
                                          apply_occ: bool = True,
                                          collect_perf: bool = False,
                                          resonance_theta_l_min: int = 1,
                                          resonance_theta_probe_count: int = 1,
                                          resonance_theta_scan_depth: int = 3,
                                          resonance_theta_scout_max_extra_nodes: int | None = 128,
                                          resonance_theta_root_tol: float | None = None,
                                          resonance_theta_sharpness_min: float = 2.0,
                                          resonance_theta_max_roots: int | None = None,
                                          resonance_theta_refine_depth: int | None = None) -> tuple[np.ndarray, dict]:
    """
    Adaptive energy integration for scattering continuum using Simpson refinement.

    Notes
    -----
    adaptive_mode selects the refinement strategy:
    - "simpson": recursive Simpson refinement using error + resonance/delta flags.
    - "bisection": use phase-shift bisection to locate resonance windows, then
      apply local Simpson refinement only inside those windows.
    - "phase-root" (alias "theta-scout"): explicitly scout the
      normalized regular matching coefficient
      ``T_l=cos(delta_l)`` for ``l>=resonance_theta_l_min``.  Bracketed
      zeros are located with Brent's method and symmetric local Simpson panels
      are carved around them.  This is a non-relativistic phase-root guard
      inspired by, but not mathematically identical to, the relativistic
      supplementary resonance detector of Wilson et al., JQSRT 99, 658-679
      (2006), Sec. 6.  The default "simpson" path is unchanged.
      The base/probe nodes are supplemented by a bounded dyadic scan.  Its
      reported maximum spacing is a resolution diagnostic, not a guarantee
      for features narrower than that spacing.
    e_base_grid controls how the initial base nodes are distributed:
    - "linear": uniform in energy E.
    - "sqrt": uniform in sqrt(E), producing denser low-energy sampling.
    near_zero_log_grid adds a few logarithmic anchors between e_min and the
    first ordinary base node.  It protects l=0 threshold quadrature without
    classifying an s-wave threshold crossing as a shape resonance.

    energy_cache can be supplied to reuse (n_e, delta) evaluations across
    repeated calls with the same V_eff and energy bounds.
    l_cap_strategy controls the energy-dependent partial-wave cutoff
    ("match" default, see _compute_l_cap).
    match_fraction_mode controls whether match_fraction is interpreted in
    index-space ("index", default) or physical radius ("r").
    n_jobs enables parallel evaluation of energy points in the adaptive
    refinement loops (batching interval midpoints per iteration).
    adaptive_parallel_mode controls adaptive parallel policy when n_jobs>1:
    - "batch": current midpoint-batch parallel refinement (default).
    - "shard": split [E_min, E_max] into independent sub-intervals and run
      one adaptive solve per shard in parallel.
      Shard boundaries follow e_base_grid ("sqrt" => denser low-E shards).
    adaptive_shard_policy controls shard-boundary placement when
    adaptive_parallel_mode="shard":
    - "egrid": use the base-energy grid directly
    - "cost": balance shards by approximate continuum work using l_cap(E)
    """
    r = np.asarray(r)
    v_eff = np.asarray(v_eff)
    if grid_kind != "sqrt":
        raise ValueError("continuum_density_scattering_adaptive supports only grid_kind='sqrt'.")
    if grid_step is None:
        grid_step = float(np.sqrt(r[1]) - np.sqrt(r[0]))

    adaptive_mode = str(adaptive_mode).lower().strip()
    use_bisection = adaptive_mode in ("bisection", "bisection_simpson", "bisection-local")
    use_theta_detector = adaptive_mode in ("phase-root", "theta-scout", "theta-local")
    delta_tol_local = None if (use_bisection or use_theta_detector) else delta_tol

    theta_l_min = max(int(resonance_theta_l_min), 0)
    theta_probe_count = max(int(resonance_theta_probe_count), 0)
    theta_scan_depth = max(int(resonance_theta_scan_depth), 0)
    theta_scout_max_extra_nodes = (
        None
        if resonance_theta_scout_max_extra_nodes is None
        else max(int(resonance_theta_scout_max_extra_nodes), 0)
    )
    theta_sharpness_min = max(float(resonance_theta_sharpness_min), 0.0)
    theta_root_tol = resonance_theta_root_tol
    theta_refine_depth = resonance_theta_refine_depth
    near_zero_enabled = bool(near_zero_log_grid)
    near_zero_points_per_decade = max(int(near_zero_log_points_per_decade), 0)
    near_zero_max_nodes = max(int(near_zero_log_max_nodes), 0)
    near_zero_max_energy = (
        None
        if near_zero_log_max_energy is None
        else max(float(near_zero_log_max_energy), 0.0)
    )

    cache = energy_cache if energy_cache is not None else {}
    cache_init = len(cache)
    max_depth = 0
    resonance_hits = 0
    delta_hits = 0
    theta_candidates = 0
    theta_rejected = 0
    theta_root_evals = 0
    theta_roots: list[dict[str, float | int]] = []
    theta_root_clusters: list[dict] = []
    theta_scout_base_node_count = 0
    theta_scout_node_count = 0
    theta_scout_extra_node_count = 0
    theta_scout_completed_depth = 0
    theta_scout_budget_exhausted = False
    theta_scout_min_spacing: float | None = None
    theta_scout_max_spacing: float | None = None
    near_zero_anchors: set[float] = set()
    cache_hits = 0
    n_eval_new = 0
    use_parallel = n_jobs is not None and int(n_jobs) > 1
    if use_parallel:
        n_jobs = int(n_jobs)
    t_wall = time.perf_counter() if bool(collect_perf) else 0.0
    perf_accum = _init_scatter_perf_accum() if bool(collect_perf) and not use_parallel else None
    pool = None
    # The Wilson-inspired phase-root mode refines explicitly carved resonance
    # windows while retaining adaptive integration in the smooth gaps,
    # so it can safely use a smaller local minimum width/deeper recursion than
    # the ordinary global Simpson path.
    refine_min_width = float(e_min_width)
    refine_depth_limit = int(e_max_depth)
    parallel_mode_requested = str(adaptive_parallel_mode).lower().strip()
    parallel_mode = parallel_mode_requested
    if parallel_mode not in ("batch", "shard"):
        raise ValueError(
            f"adaptive_parallel_mode must be 'batch' or 'shard', got '{adaptive_parallel_mode}'."
        )
    theta_shard_mode_forced_batch = bool(use_theta_detector and parallel_mode == "shard")
    if theta_shard_mode_forced_batch:
        # A root at a shard edge cannot be given a symmetric local panel, and
        # independent workers cannot cluster coincident/nearby roots across
        # that edge.  Keep parallel energy evaluation, but build the scout and
        # resonance windows globally in the parent solve.
        warnings.warn(
            "phase-root resonance scouting is incompatible with independent "
            "energy shards; forcing adaptive_parallel_mode='batch' so roots "
            "and symmetric windows are resolved on the full energy interval.",
            RuntimeWarning,
        )
        parallel_mode = "batch"
    shard_policy = str(adaptive_shard_policy).lower().strip()
    if shard_policy not in ("egrid", "cost"):
        raise ValueError(
            f"adaptive_shard_policy must be 'egrid' or 'cost', got '{adaptive_shard_policy}'."
        )

    def _build_base_nodes(e_lo: float, e_hi: float, n_base: int, mode: str) -> np.ndarray:
        """
        Build initial adaptive nodes on [e_lo, e_hi].

        This only affects the starting mesh; subsequent refinement still follows
        the same error/resonance/delta criteria.
        """
        n_use = max(int(n_base), 2)
        mode_l = str(mode).lower().strip()
        if mode_l in ("linear", "lin"):
            return np.linspace(e_lo, e_hi, n_use)
        if mode_l in ("sqrt", "root", "sqrt_e"):
            s_lo = np.sqrt(max(float(e_lo), 0.0))
            s_hi = np.sqrt(max(float(e_hi), 0.0))
            s_nodes = np.linspace(s_lo, s_hi, n_use)
            e_nodes = s_nodes * s_nodes
            e_nodes[0] = float(e_lo)
            e_nodes[-1] = float(e_hi)
            return e_nodes
        raise ValueError(
            f"Unsupported e_base_grid='{mode}'. Expected 'linear' or 'sqrt'."
        )

    def _build_integration_nodes(e_lo: float, e_hi: float, n_base: int, mode: str) -> np.ndarray:
        """Build base nodes and resolve the otherwise-wide threshold panel.

        The centrifugal-barrier resonance scout intentionally starts at l=1.
        Near-threshold s-wave structure is instead exposed to the ordinary
        error-controlled quadrature by inserting a small logarithmic mesh in
        the first energy panel.  See Starrett and Saumon, *High Energy Density
        Physics* (2014), Appendix B, and their cited Wilson et al. algorithm
        for the need for adaptive continuum energy meshes.  This logarithmic
        guard is a numerical implementation detail of that requirement, not
        an additional physical resonance model.
        """
        base = _build_base_nodes(e_lo, e_hi, n_base, mode)
        if (
            not near_zero_enabled
            or near_zero_points_per_decade <= 0
            or near_zero_max_nodes <= 0
            or base.size < 2
            or e_lo <= 0.0
        ):
            return base

        first = float(base[1])
        upper = first
        if near_zero_max_energy is not None:
            if float(e_lo) >= near_zero_max_energy:
                return base
            upper = min(upper, near_zero_max_energy)
        ratio = upper / float(e_lo)
        if (not np.isfinite(ratio)) or ratio <= 1.0 + 16.0 * np.finfo(float).eps:
            return base

        decades = max(float(np.log10(ratio)), 0.0)
        n_log_segments = min(
            near_zero_max_nodes,
            max(1, int(np.ceil(near_zero_points_per_decade * decades))),
        )
        log_nodes = np.geomspace(float(e_lo), upper, n_log_segments + 1)[1:]
        merged = np.unique(np.concatenate((base, log_nodes))).astype(float)
        base_set = set(float(val) for val in base)
        for val in merged:
            val_f = float(val)
            if val_f not in base_set:
                near_zero_anchors.add(val_f)
        return merged

    def _estimate_energy_cost(e_val: float) -> float:
        """
        Cheap per-energy work estimate for shard balancing.

        The dominant cost is the partial-wave loop, so use the current
        energy-dependent l_cap(E) estimate as the primary proxy.
        """
        l_cap_est = _compute_l_cap(
            float(e_val),
            l_max,
            r,
            l_pad,
            match_slice,
            match_r_cut,
            match_fraction,
            match_fraction_mode,
            match_width,
            match_min_points,
            match_kr_min,
            match_v_tol,
            v_eff,
            l_cap_strategy,
        )
        return float(max(int(l_cap_est) + 1, 1))

    def _build_shard_edges(e_lo: float, e_hi: float, n_shards: int, mode: str, policy: str) -> np.ndarray:
        """
        Build shard boundaries on [e_lo, e_hi].

        "egrid" preserves the old behavior. "cost" uses a cheap l_cap(E)-based
        workload model so high-energy shards become narrower.
        """
        if str(policy).lower().strip() == "egrid":
            edges_local = _build_base_nodes(e_lo, e_hi, n_shards + 1, mode).astype(float)
        else:
            n_probe = max(257, 64 * int(n_shards) + 1)
            probe = _build_base_nodes(e_lo, e_hi, n_probe, mode).astype(float)
            probe[0] = float(e_lo)
            probe[-1] = float(e_hi)
            weights = np.array([_estimate_energy_cost(e_val) for e_val in probe], dtype=float)
            seg_w = 0.5 * (weights[:-1] + weights[1:]) * np.diff(probe)
            cum = np.concatenate(([0.0], np.cumsum(seg_w)))
            total = float(cum[-1]) if cum.size > 0 else 0.0
            if (not np.isfinite(total)) or total <= 0.0:
                edges_local = _build_base_nodes(e_lo, e_hi, n_shards + 1, mode).astype(float)
            else:
                targets = np.linspace(0.0, total, int(n_shards) + 1, dtype=float)
                edges_local = np.interp(targets, cum, probe)
                edges_local[0] = float(e_lo)
                edges_local[-1] = float(e_hi)
        if np.any(np.diff(edges_local) <= 0.0):
            edges_local = np.linspace(e_lo, e_hi, int(n_shards) + 1, dtype=float)
        return edges_local

    probe_idx = None
    if resonance_tol is not None and resonance_tol > 0.0:
        if resonance_r_fractions is None:
            resonance_r_fractions = (0.25, 0.5, 0.75)
        idxs = []
        for frac in resonance_r_fractions:
            idx = int(round(frac * (r.size - 1)))
            idxs.append(min(max(idx, 0), r.size - 1))
        probe_idx = np.unique(np.array(idxs, dtype=int))

    def eval_energy(e: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate scattering density and phase shifts at energy e with caching.
        """
        nonlocal cache_hits
        nonlocal n_eval_new
        if e in cache:
            cache_hits += 1
            return cache[e]
        n_e, delta_vec = _scattering_density_and_phase(
            v_eff,
            r,
            mu,
            temperature,
            e,
            l_max,
            grid_kind,
            grid_step,
            l_pad,
            match_fraction,
            match_slice,
            match_r_cut,
            match_fraction_mode,
            match_width,
            match_kr_min,
            match_v_tol,
            match_min_points,
            match_asymptotic,
            match_coulomb_tol,
            match_allow_shift,
            match_fallback,
            prop_rescale_limit,
            apply_occ=apply_occ,
            l_cap_strategy=l_cap_strategy,
            perf_accum=perf_accum,
        )
        cache[e] = (n_e, delta_vec)
        n_eval_new += 1
        return n_e, delta_vec

    def eval_energy_batch(energies: list[float] | np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Evaluate a batch of energies, optionally in parallel, with caching.
        """
        nonlocal cache_hits
        nonlocal n_eval_new
        e_list = [float(e) for e in energies]
        if not e_list:
            return []

        # Count cache hits by unique new evaluations.
        cache_before = set(cache.keys())
        missing = [e for e in e_list if e not in cache_before]
        missing_unique = sorted(set(missing))

        if missing_unique:
            if pool is None:
                # Serial fallback.
                for e_val in missing_unique:
                    n_e, delta_vec = _scattering_density_and_phase(
                        v_eff,
                        r,
                        mu,
                        temperature,
                        e_val,
                        l_max,
                        grid_kind,
                        grid_step,
                        l_pad,
                        match_fraction,
                        match_slice,
                        match_r_cut,
                        match_fraction_mode,
                        match_width,
                        match_kr_min,
                        match_v_tol,
                        match_min_points,
                        match_asymptotic,
                        match_coulomb_tol,
                        match_allow_shift,
                        match_fallback,
                        prop_rescale_limit,
                        apply_occ=apply_occ,
                        l_cap_strategy=l_cap_strategy,
                        perf_accum=perf_accum,
                    )
                    cache[e_val] = (n_e, delta_vec)
                n_eval_new += len(missing_unique)
            else:
                results = pool.map(_scatter_worker, missing_unique)
                for e_val, n_e, delta_vec in results:
                    cache[e_val] = (n_e, delta_vec)
                n_eval_new += len(missing_unique)

        cache_hits += max(len(e_list) - len(missing_unique), 0)
        return [cache[e] for e in e_list]

    def eval_energy_single(e: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper to evaluate a single energy using the batch path if parallel.
        """
        if use_parallel:
            return eval_energy_batch([e])[0]
        return eval_energy(e)

    def delta_metric(d0: np.ndarray, d1: np.ndarray) -> float:
        """
        Compute a phase-shift change metric for resonance detection.
        """
        if delta_mode == "sum":
            return float(abs(np.sum(d1) - np.sum(d0)))
        if delta_mode == "weighted":
            weights = 2.0 * np.arange(d0.size) + 1.0
            return float(abs(np.sum(weights * d1) - np.sum(weights * d0)))
        return float(np.max(np.abs(d1 - d0)))

    def active_l_cap(e_val: float) -> int:
        """Return the channels that are genuinely evaluated at one energy."""
        return _compute_l_cap(
            float(e_val),
            l_max,
            r,
            l_pad,
            match_slice,
            match_r_cut,
            match_fraction,
            match_fraction_mode,
            match_width,
            match_min_points,
            match_kr_min,
            match_v_tol,
            v_eff,
            l_cap_strategy,
        )

    def theta_value(e_val: float, l_val: int) -> float:
        """Evaluate one normalised regular matching coefficient."""
        nonlocal theta_root_evals
        _, d_val = eval_energy_single(float(e_val))
        theta_root_evals += 1
        return float(phase_root_resonance_scout(d_val)[int(l_val)])

    def locate_theta_roots(
        scout_nodes: np.ndarray,
        scout_deltas: list[np.ndarray],
    ) -> list[dict[str, float | int]]:
        """Locate bracketed normalized regular-coefficient roots.

        The regular/irregular fit coefficients are smooth even when the
        energy-normalised interior density is too narrow for a quadrature
        estimator to notice.  Looking for their regular-coefficient zero is
        therefore an independent guard against false quadrature convergence,
        following Wilson et al. (2006), Sec. 6.
        """
        nonlocal theta_candidates
        nonlocal theta_rejected

        nodes = np.asarray(scout_nodes, dtype=float)
        if nodes.size < 2:
            return []
        theta_rows = [phase_root_resonance_scout(d) for d in scout_deltas]
        roots_local: list[dict[str, float | int]] = []
        root_tol_local = float(theta_root_tol)

        for idx in range(nodes.size - 1):
            ea = float(nodes[idx])
            eb = float(nodes[idx + 1])
            if eb <= ea:
                continue
            # Do not interpret the artificial delta=0 entries above an
            # energy-dependent l cap as matching coefficients.  Requiring the
            # channel at both ends also avoids roots created by a moving cap.
            cap = min(active_l_cap(ea), active_l_cap(eb))
            if cap < theta_l_min:
                continue
            for l_val in range(theta_l_min, cap + 1):
                ta = float(theta_rows[idx][l_val])
                tb = float(theta_rows[idx + 1][l_val])
                if not (np.isfinite(ta) and np.isfinite(tb)):
                    continue
                if ta == 0.0:
                    e_root = ea
                elif tb == 0.0:
                    e_root = eb
                elif ta * tb < 0.0:
                    theta_candidates += 1
                    try:
                        e_root = float(
                            brentq(
                                lambda e_trial: theta_value(e_trial, l_val),
                                ea,
                                eb,
                                xtol=root_tol_local,
                                rtol=4.0 * np.finfo(float).eps,
                                maxiter=100,
                            )
                        )
                    except (ValueError, RuntimeError):
                        theta_rejected += 1
                        continue
                else:
                    continue

                # Estimate the local energy scale from Theta'(E_r).  For a
                # Breit-Wigner-like shape resonance, 1/|Theta'| is its HWHM
                # to leading order.  A finite probe that overestimates an
                # ultra-narrow width is safe: it merely carves a wider panel.
                panel_width = eb - ea
                h_probe = max(8.0 * root_tol_local, 0.05 * panel_width)
                slope = np.nan
                previous_slope = np.nan
                # A single panel-scale secant can overestimate an ultra-narrow
                # resonance width by orders of magnitude.  Halve the probe
                # until the local derivative stabilizes.  Broad crossings
                # converge after one comparison; narrow ones spend a handful
                # of cheap scalar phase evaluations but greatly reduce the
                # subsequent density-quadrature window.
                for _ in range(18):
                    e_left = max(ea, e_root - h_probe)
                    e_right = min(eb, e_root + h_probe)
                    if e_right <= e_left:
                        break
                    t_left = theta_value(e_left, l_val)
                    t_right = theta_value(e_right, l_val)
                    slope = abs((t_right - t_left) / (e_right - e_left))
                    if np.isfinite(previous_slope) and np.isfinite(slope):
                        rel_change = abs(slope - previous_slope) / max(
                            slope,
                            np.finfo(float).tiny,
                        )
                        if rel_change <= 0.03:
                            break
                    if h_probe <= 16.0 * root_tol_local:
                        break
                    previous_slope = slope
                    h_probe = max(0.5 * h_probe, 8.0 * root_tol_local)
                sharpness = slope * panel_width
                if (not np.isfinite(slope)) or sharpness <= theta_sharpness_min:
                    theta_rejected += 1
                    continue
                width_est = 1.0 / max(slope, np.finfo(float).tiny)
                roots_local.append(
                    {
                        "energy": float(e_root),
                        "l": int(l_val),
                        "theta_slope": float(slope),
                        "theta_sharpness": float(sharpness),
                        "width_est": float(width_est),
                        "bracket_lo": float(ea),
                        "bracket_hi": float(eb),
                    }
                )

        # A root on a scout-node boundary is seen from both adjacent panels.
        # Keep the sharper duplicate deterministically.
        roots_local.sort(key=lambda item: (int(item["l"]), float(item["energy"])))
        deduped: list[dict[str, float | int]] = []
        for item in roots_local:
            if (
                deduped
                and int(item["l"]) == int(deduped[-1]["l"])
                and abs(float(item["energy"]) - float(deduped[-1]["energy"]))
                <= 4.0 * root_tol_local
            ):
                if float(item["theta_slope"]) > float(deduped[-1]["theta_slope"]):
                    deduped[-1] = item
                continue
            deduped.append(item)

        if resonance_theta_max_roots is not None and len(deduped) > int(resonance_theta_max_roots):
            selected = sorted(
                deduped,
                key=lambda item: float(item["theta_slope"]),
                reverse=True,
            )[: max(int(resonance_theta_max_roots), 0)]
            theta_rejected += len(deduped) - len(selected)
            deduped = sorted(selected, key=lambda item: float(item["energy"]))
        return deduped

    def integrate_interval(e0: float,
                           e1: float,
                           n0: np.ndarray,
                           n1: np.ndarray,
                           d0: np.ndarray,
                           d1: np.ndarray,
                           depth: int) -> np.ndarray:
        """
        Recursively integrate n_e(r, e) over [e0, e1] using Simpson refinement.
        """
        nonlocal max_depth
        nonlocal resonance_hits
        nonlocal delta_hits
        max_depth = max(max_depth, depth)
        em = 0.5 * (e0 + e1)
        nm, dm = eval_energy_single(em)

        trap = 0.5 * (n0 + n1) * (e1 - e0)
        simp = (e1 - e0) * (n0 + 4.0 * nm + n1) / 6.0

        diff = simp - trap
        err = np.linalg.norm(diff) / (np.linalg.norm(simp) + 1e-12)

        resonance_flag = False
        if probe_idx is not None:
            n0p = n0[probe_idx]
            n1p = n1[probe_idx]
            nmp = nm[probe_idx]
            curvature = np.max(np.abs(n0p - 2.0 * nmp + n1p) / (np.abs(nmp) + resonance_floor))
            if curvature > resonance_tol:
                resonance_flag = True
                resonance_hits += 1

        delta_flag = False
        if delta_tol_local is not None and np.isfinite(delta_tol_local):
            delta_diff = delta_metric(d0, d1)
            if delta_diff > float(delta_tol_local):
                delta_flag = True
                delta_hits += 1

        if (
            (err > e_tol or resonance_flag or delta_flag)
            and depth < refine_depth_limit
            and (e1 - e0) > refine_min_width
        ):
            left = integrate_interval(e0, em, n0, nm, d0, dm, depth + 1)
            right = integrate_interval(em, e1, nm, n1, dm, d1, depth + 1)
            return left + right
        return simp

    def integrate_intervals_parallel(intervals: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]
                                     ) -> np.ndarray:
        """
        Parallel-friendly adaptive refinement over a list of intervals.

        The algorithm processes intervals in batches, evaluates all midpoints
        at once (optionally in parallel), and refines/accepts each interval.
        """
        nonlocal max_depth
        nonlocal resonance_hits
        nonlocal delta_hits

        n_r_local = np.zeros_like(r, dtype=float)
        stack = list(intervals)
        batch_size = max(int(n_jobs or 1) * 4, 8)

        while stack:
            batch = stack[:batch_size]
            stack = stack[batch_size:]
            mids = [0.5 * (e0 + e1) for e0, e1, *_ in batch]
            mid_vals = eval_energy_batch(mids)

            for (e0, e1, n0, n1, d0, d1, depth), (nm, dm) in zip(batch, mid_vals):
                max_depth = max(max_depth, depth)
                em = 0.5 * (e0 + e1)
                trap = 0.5 * (n0 + n1) * (e1 - e0)
                simp = (e1 - e0) * (n0 + 4.0 * nm + n1) / 6.0

                diff = simp - trap
                err = np.linalg.norm(diff) / (np.linalg.norm(simp) + 1e-12)

                resonance_flag = False
                if probe_idx is not None:
                    n0p = n0[probe_idx]
                    n1p = n1[probe_idx]
                    nmp = nm[probe_idx]
                    curvature = np.max(np.abs(n0p - 2.0 * nmp + n1p) / (np.abs(nmp) + resonance_floor))
                    if curvature > resonance_tol:
                        resonance_flag = True
                        resonance_hits += 1

                delta_flag = False
                if delta_tol_local is not None and np.isfinite(delta_tol_local):
                    delta_diff = delta_metric(d0, d1)
                    if delta_diff > float(delta_tol_local):
                        delta_flag = True
                        delta_hits += 1

                if (
                    (err > e_tol or resonance_flag or delta_flag)
                    and depth < refine_depth_limit
                    and (e1 - e0) > refine_min_width
                ):
                    stack.append((e0, em, n0, nm, d0, dm, depth + 1))
                    stack.append((em, e1, nm, n1, dm, d1, depth + 1))
                else:
                    n_r_local += simp
        return n_r_local

    def simpson_segment(e0: float,
                        e1: float,
                        n0: np.ndarray,
                        n1: np.ndarray) -> np.ndarray:
        """
        Compute a single Simpson estimate on [e0, e1] without refinement.
        """
        em = 0.5 * (e0 + e1)
        nm, _ = eval_energy_single(em)
        return (e1 - e0) * (n0 + 4.0 * nm + n1) / 6.0

    def split_interval_by_windows(e0: float,
                                  e1: float,
                                  windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        Split [e0, e1] at resonance window boundaries.
        """
        cuts = [float(e0), float(e1)]
        for w0, w1 in windows:
            if e0 < w0 < e1:
                cuts.append(float(w0))
            if e0 < w1 < e1:
                cuts.append(float(w1))
        cuts = sorted(set(cuts))
        return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    def segment_in_window(e0: float, e1: float, windows: list[tuple[float, float]]) -> bool:
        """
        Return True if the segment midpoint lies inside any resonance window.
        """
        if not windows:
            return False
        mid = 0.5 * (e0 + e1)
        for w0, w1 in windows:
            if w0 <= mid <= w1:
                return True
        return False

    def bisect_resonance_interval(e0: float,
                                  e1: float,
                                  d0: np.ndarray,
                                  d1: np.ndarray,
                                  max_depth_local: int) -> tuple[float, float, list[float]]:
        """
        Single-sided bisection that follows the larger phase-shift jump.
        """
        a = float(e0)
        b = float(e1)
        da = np.array(d0, copy=False)
        db = np.array(d1, copy=False)
        samples = [a, b]
        for _ in range(int(max_depth_local)):
            if (b - a) <= e_min_width:
                break
            m = 0.5 * (a + b)
            _, dm = eval_energy_single(m)
            samples.append(float(m))
            if delta_metric(da, dm) >= delta_metric(dm, db):
                b = m
                db = dm
            else:
                a = m
                da = dm
        return a, b, samples

    e0 = max(float(e_min), 1e-12)
    e1 = float(e_max)
    if theta_root_tol is None:
        # Root location, unlike the density quadrature, is a smooth scalar
        # problem.  Resolve it more tightly than the ordinary panel width so a
        # symmetric panel can be centred on a sub-grid resonance.
        theta_root_tol = max(
            1.0e-12,
            min(1.0e-8, 1.0e-2 * float(e_min_width)),
        )
    else:
        theta_root_tol = max(float(theta_root_tol), 4.0 * np.finfo(float).eps)
    if use_theta_detector:
        refine_min_width = min(
            float(e_min_width),
            max(4.0 * float(theta_root_tol), 16.0 * np.finfo(float).eps),
        )
        refine_depth_limit = max(
            int(e_max_depth),
            int(theta_refine_depth) if theta_refine_depth is not None else int(e_max_depth) + 8,
        )

    # Coarse-grained parallel mode: split energy domain into disjoint shards
    # and solve each shard independently in its own process.
    if use_parallel and parallel_mode == "shard":
        if e1 <= e0:
            meta = {
                "n_eval": 0,
                "n_cache_init": 0,
                "n_cache_hits": 0,
                "n_cache_total": 0,
                "max_depth": 0,
                "e_min": e0,
                "e_max": e1,
                "n_jobs": int(n_jobs),
                "n_base": int(n_e_base),
                "e_base_grid": str(e_base_grid),
                "adaptive_mode": adaptive_mode,
                "adaptive_parallel_mode": "shard",
                "adaptive_shards": 0,
                "adaptive_shard_policy": str(shard_policy),
                "apply_occ": bool(apply_occ),
            }
            if bool(collect_perf):
                meta.update(_scatter_perf_meta(None, 0.0))
            return np.zeros_like(r, dtype=float), meta

        shard_count = int(adaptive_shards) if adaptive_shards is not None else int(n_jobs)
        shard_count = max(1, shard_count)
        if shard_count > 1:
            # When caller provides an external cache (basis reuse path), ask each
            # shard worker to return its local E-samples and merge them here.
            # This enables adaptive_reuse_basis together with shard parallelism.
            collect_cache_samples = energy_cache is not None

            # `n_e_base` is defined as a global base-node budget over [E_min, E_max].
            # If each shard used the full n_e_base independently, total initial nodes
            # would scale as O(shard_count * n_e_base), which over-refines and slows
            # runs dramatically at large shard counts. Distribute base nodes per shard
            # so global coverage stays approximately constant.
            n_e_base_total = max(int(n_e_base), 2)
            n_e_base_shard = max(2, int(np.ceil((n_e_base_total - 1) / float(shard_count))) + 1)

            edges = _build_shard_edges(e0, e1, shard_count, e_base_grid, shard_policy)
            base_kwargs = {
                "v_eff": v_eff,
                "r": r,
                "mu": mu,
                "temperature": temperature,
                "l_max": l_max,
                "grid_kind": grid_kind,
                "grid_step": grid_step,
                "l_pad": l_pad,
                "match_fraction": match_fraction,
                "match_slice": match_slice,
                "match_r_cut": match_r_cut,
                "match_fraction_mode": match_fraction_mode,
                "match_width": match_width,
                "match_kr_min": match_kr_min,
                "match_v_tol": match_v_tol,
                "match_min_points": match_min_points,
                "match_asymptotic": match_asymptotic,
                "match_coulomb_tol": match_coulomb_tol,
                "match_allow_shift": match_allow_shift,
                "match_fallback": match_fallback,
                "prop_rescale_limit": prop_rescale_limit,
                "l_cap_strategy": l_cap_strategy,
                "e_tol": e_tol,
                "e_max_depth": e_max_depth,
                "e_min_width": e_min_width,
                "n_e_base": n_e_base_shard,
                "e_base_grid": e_base_grid,
                "near_zero_log_grid": near_zero_enabled,
                "near_zero_log_points_per_decade": near_zero_points_per_decade,
                "near_zero_log_max_nodes": near_zero_max_nodes,
                "near_zero_log_max_energy": near_zero_max_energy,
                "resonance_tol": resonance_tol,
                "resonance_r_fractions": resonance_r_fractions,
                "resonance_floor": resonance_floor,
                "delta_tol": delta_tol,
                "delta_mode": delta_mode,
                "adaptive_mode": adaptive_mode,
                "bisection_max_depth": bisection_max_depth,
                "resonance_window_factor": resonance_window_factor,
                "resonance_window_min": resonance_window_min,
                "resonance_window_max": resonance_window_max,
                "resonance_max_windows": resonance_max_windows,
                "resonance_theta_l_min": theta_l_min,
                "resonance_theta_probe_count": theta_probe_count,
                "resonance_theta_scan_depth": theta_scan_depth,
                "resonance_theta_root_tol": theta_root_tol,
                "resonance_theta_sharpness_min": theta_sharpness_min,
                "resonance_theta_max_roots": resonance_theta_max_roots,
                "resonance_theta_refine_depth": theta_refine_depth,
                "energy_cache": None,
                "n_jobs": None,
                "adaptive_parallel_mode": "batch",
                "adaptive_shards": None,
                "adaptive_shard_policy": "egrid",
                "apply_occ": apply_occ,
                "_collect_cache_samples": bool(collect_cache_samples),
            }
            tasks = []
            for i in range(shard_count):
                shard_kwargs = dict(base_kwargs)
                if theta_scout_max_extra_nodes is None:
                    shard_budget = None
                else:
                    # Preserve a global extra-node bound rather than silently
                    # multiplying the configured budget by the shard count.
                    quotient, remainder = divmod(
                        int(theta_scout_max_extra_nodes),
                        int(shard_count),
                    )
                    shard_budget = quotient + (1 if i < remainder else 0)
                shard_kwargs["resonance_theta_scout_max_extra_nodes"] = shard_budget
                tasks.append(
                    (i, float(edges[i]), float(edges[i + 1]), shard_kwargs)
                )

            ctx = mp.get_context("fork")
            with ctx.Pool(processes=int(n_jobs)) as shard_pool:
                shard_results = shard_pool.map(_adaptive_shard_worker, tasks)

            shard_results = sorted(shard_results, key=lambda t: int(t[0]))
            n_r = np.zeros_like(r, dtype=float)
            n_eval_sum = 0
            cache_hits_sum = 0
            cache_total_sum = 0
            max_depth_agg = 0
            resonance_hits_sum = 0
            delta_hits_sum = 0
            n_windows_sum = 0
            bisect_intervals_sum = 0
            bisect_samples_sum = 0
            bisect_depth_max = 0
            theta_candidates_sum = 0
            theta_rejected_sum = 0
            theta_root_evals_sum = 0
            theta_roots_agg: list[dict[str, float | int]] = []
            theta_root_clusters_agg: list[dict] = []
            theta_scout_base_nodes_sum = 0
            theta_scout_nodes_sum = 0
            theta_scout_extra_nodes_sum = 0
            theta_scout_completed_depth_min: int | None = None
            theta_scout_budget_exhausted_any = False
            theta_scout_min_spacing_agg = np.inf
            theta_scout_max_spacing_agg = 0.0
            near_zero_anchors_agg: set[float] = set()
            shard_meta = []
            merged_cache_count = 0
            skipped_cache_count = 0

            for idx, n_part, m_part, samples in shard_results:
                n_r += np.asarray(n_part, dtype=float)
                n_eval_sum += int(m_part.get("n_eval", 0))
                cache_hits_sum += int(m_part.get("n_cache_hits", 0))
                cache_total_sum += int(m_part.get("n_cache_total", 0))
                max_depth_agg = max(max_depth_agg, int(m_part.get("max_depth", 0)))
                resonance_hits_sum += int(m_part.get("resonance_hits", 0))
                delta_hits_sum += int(m_part.get("delta_hits", 0))
                n_windows_sum += int(m_part.get("n_windows", 0))
                bisect_intervals_sum += int(m_part.get("bisection_intervals", 0))
                bisect_samples_sum += int(m_part.get("bisection_samples", 0))
                bisect_depth_max = max(bisect_depth_max, int(m_part.get("bisection_max_depth", 0)))
                theta_candidates_sum += int(m_part.get("theta_candidates", 0))
                theta_rejected_sum += int(m_part.get("theta_rejected", 0))
                theta_root_evals_sum += int(m_part.get("theta_root_evals", 0))
                for item in m_part.get("theta_roots", []):
                    item_copy = dict(item)
                    item_copy["shard"] = int(idx)
                    theta_roots_agg.append(item_copy)
                for cluster in m_part.get("theta_root_clusters", []):
                    cluster_copy = dict(cluster)
                    cluster_copy["shard"] = int(idx)
                    theta_root_clusters_agg.append(cluster_copy)
                theta_scout_base_nodes_sum += int(m_part.get("theta_scout_base_node_count", 0))
                theta_scout_nodes_sum += int(m_part.get("theta_scout_node_count", 0))
                theta_scout_extra_nodes_sum += int(m_part.get("theta_scout_extra_node_count", 0))
                completed_part = int(m_part.get("theta_scout_completed_depth", 0))
                theta_scout_completed_depth_min = (
                    completed_part
                    if theta_scout_completed_depth_min is None
                    else min(theta_scout_completed_depth_min, completed_part)
                )
                theta_scout_budget_exhausted_any = bool(
                    theta_scout_budget_exhausted_any
                    or m_part.get("theta_scout_budget_exhausted", False)
                )
                min_spacing_part = m_part.get("theta_scout_min_spacing", None)
                max_spacing_part = m_part.get("theta_scout_max_spacing", None)
                if min_spacing_part is not None and np.isfinite(float(min_spacing_part)):
                    theta_scout_min_spacing_agg = min(
                        theta_scout_min_spacing_agg,
                        float(min_spacing_part),
                    )
                if max_spacing_part is not None and np.isfinite(float(max_spacing_part)):
                    theta_scout_max_spacing_agg = max(
                        theta_scout_max_spacing_agg,
                        float(max_spacing_part),
                    )
                near_zero_anchors_agg.update(
                    float(val) for val in m_part.get("near_zero_log_anchors", [])
                )

                # Merge shard-local basis samples into caller cache. This is
                # required for inner-mu basis reuse where caller builds
                # cont_basis from energy_cache after this adaptive pass.
                if collect_cache_samples and samples is not None and energy_cache is not None:
                    for e_val, n_e_val, d_val in samples:
                        if e_val in energy_cache:
                            skipped_cache_count += 1
                            continue
                        energy_cache[e_val] = (n_e_val, d_val)
                        merged_cache_count += 1

                shard_meta.append(
                    {
                        "idx": int(idx),
                        "e_min": float(m_part.get("e_min", np.nan)),
                        "e_max": float(m_part.get("e_max", np.nan)),
                        "n_eval": int(m_part.get("n_eval", 0)),
                        "wall_s": float(m_part.get("wall_s", np.nan)),
                        "n_windows": int(m_part.get("n_windows", 0)),
                        "max_depth": int(m_part.get("max_depth", 0)),
                    }
                )

            cache_total_final = len(energy_cache) if energy_cache is not None else int(cache_total_sum)
            meta = {
                "n_eval": int(n_eval_sum),
                "n_cache_init": int(cache_init),
                "n_cache_hits": int(cache_hits_sum),
                "n_cache_total": int(cache_total_final),
                "max_depth": int(max_depth_agg),
                "e_min": float(e0),
                "e_max": float(e1),
                "n_jobs": int(n_jobs),
                "n_base": int(n_e_base_total),
                "n_base_per_shard": int(n_e_base_shard),
                "e_base_grid": str(e_base_grid),
                "resonance_hits": int(resonance_hits_sum),
                "delta_hits": int(delta_hits_sum),
                "delta_tol": float(delta_tol) if delta_tol is not None else None,
                "delta_mode": str(delta_mode),
                "adaptive_mode": adaptive_mode,
                "n_windows": int(n_windows_sum),
                "resonance_windows": [],
                "bisection_fallback": None,
                "max_delta_metric": None,
                "bisection_intervals": int(bisect_intervals_sum),
                "bisection_samples": int(bisect_samples_sum),
                "bisection_max_depth": int(bisect_depth_max),
                "theta_detector_enabled": adaptive_mode in ("phase-root", "theta-scout", "theta-local"),
                "theta_fallback": bool(
                    n_windows_sum == 0
                    and adaptive_mode in ("phase-root", "theta-scout", "theta-local")
                ),
                "theta_candidates": int(theta_candidates_sum),
                "theta_rejected": int(theta_rejected_sum),
                "theta_root_evals": int(theta_root_evals_sum),
                "theta_root_tol": float(theta_root_tol),
                "theta_refine_min_width": float(refine_min_width),
                "theta_refine_depth": int(refine_depth_limit),
                "theta_roots": sorted(theta_roots_agg, key=lambda item: float(item.get("energy", 0.0))),
                "theta_root_clusters": sorted(
                    theta_root_clusters_agg,
                    key=lambda item: (int(item.get("shard", 0)), float(item.get("energy", 0.0))),
                ),
                "theta_scout_requested_depth": int(theta_scan_depth),
                "theta_scout_completed_depth": int(theta_scout_completed_depth_min or 0),
                "theta_scout_base_node_count": int(theta_scout_base_nodes_sum),
                "theta_scout_node_count": int(theta_scout_nodes_sum),
                "theta_scout_extra_node_count": int(theta_scout_extra_nodes_sum),
                "theta_scout_max_extra_nodes": (
                    None
                    if theta_scout_max_extra_nodes is None
                    else int(theta_scout_max_extra_nodes)
                ),
                "theta_scout_budget_exhausted": bool(theta_scout_budget_exhausted_any),
                "theta_scout_min_spacing": (
                    float(theta_scout_min_spacing_agg)
                    if np.isfinite(theta_scout_min_spacing_agg)
                    else None
                ),
                "theta_scout_max_spacing": (
                    float(theta_scout_max_spacing_agg)
                    if theta_scout_max_spacing_agg > 0.0
                    else None
                ),
                "theta_scout_limitation": "finite_mesh_no_arbitrary_subgrid_guarantee",
                "near_zero_log_grid": bool(near_zero_enabled),
                "near_zero_log_anchor_count": int(len(near_zero_anchors_agg)),
                "near_zero_log_anchors": sorted(near_zero_anchors_agg),
                "apply_occ": bool(apply_occ),
                "adaptive_parallel_mode": "shard",
                "adaptive_parallel_mode_requested": str(parallel_mode_requested),
                "theta_shard_mode_forced_batch": bool(theta_shard_mode_forced_batch),
                "adaptive_shards": int(shard_count),
                "adaptive_shard_policy": str(shard_policy),
                "shard_meta": shard_meta,
                "shard_cache_collect": bool(collect_cache_samples),
                "shard_cache_merged": int(merged_cache_count),
                "shard_cache_skipped": int(skipped_cache_count),
            }
            if bool(collect_perf):
                meta.update(_scatter_perf_meta(None, 0.0))
            return n_r, meta

    windows: list[tuple[float, float]] = []
    bisection_samples = 0
    bisection_intervals = 0
    bisection_depth_max = 0
    max_metric = 0.0

    def build_theta_scout_nodes(base_nodes: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return a bounded multiresolution phase-root scout mesh.

        ``probe_count`` is retained for backward compatibility.  Dyadic
        levels then fill each interval in a nested order, so increasing the
        requested depth never moves an existing scout point.  The extra-node
        budget excludes mandatory base/log anchors and is consumed uniformly
        across the full energy range at each level.

        This is a deterministic resolution guard, not a proof that a panel is
        root-free.  In particular, an even number of roots can remain hidden
        inside any final gap.  Wilson et al. (JQSRT 99, 658-679, 2006, Sec. 6)
        instead interpolate their smoother boundary ``a(E), b(E)`` functions
        and analytic relativistic ``Theta(E)``; the present code does not yet
        implement that exact construction.
        """
        base = np.asarray(sorted(set(float(val) for val in base_nodes)), dtype=float)
        nodes = set(float(val) for val in base)
        max_extra = theta_scout_max_extra_nodes
        extra_used = 0
        budget_exhausted = False

        def add_candidates(candidates: list[float]) -> bool:
            """Add one nested level; return whether the whole level fit."""
            nonlocal extra_used
            nonlocal budget_exhausted

            pending = np.asarray(
                sorted(set(float(val) for val in candidates if float(val) not in nodes)),
                dtype=float,
            )
            if pending.size == 0:
                return True
            if max_extra is None:
                selected = pending
            else:
                remaining = max(int(max_extra) - int(extra_used), 0)
                if remaining <= 0:
                    budget_exhausted = True
                    return False
                if pending.size <= remaining:
                    selected = pending
                else:
                    # Spread a partial level over the whole domain instead of
                    # preferentially resolving only the lowest-energy panels.
                    idx = np.floor(
                        (np.arange(remaining, dtype=float) + 0.5)
                        * float(pending.size)
                        / float(remaining)
                    ).astype(int)
                    idx = np.clip(idx, 0, pending.size - 1)
                    selected = pending[idx]
                    budget_exhausted = True
            for val in selected:
                nodes.add(float(val))
            extra_used += int(selected.size)
            return bool(selected.size == pending.size)

        # Preserve the old equally-spaced probe semantics.  Common values
        # (one midpoint, or quarter points) are automatically reused by the
        # nested dyadic scan below and therefore consume the budget only once.
        probe_candidates: list[float] = []
        if theta_probe_count > 0:
            for idx in range(base.size - 1):
                ea = float(base[idx])
                eb = float(base[idx + 1])
                for i_probe in range(1, theta_probe_count + 1):
                    frac = i_probe / float(theta_probe_count + 1)
                    probe_candidates.append(ea + frac * (eb - ea))
        add_candidates(probe_candidates)

        completed_depth = 0
        for depth in range(1, theta_scan_depth + 1):
            denominator = 1 << depth
            level_candidates: list[float] = []
            for idx in range(base.size - 1):
                ea = float(base[idx])
                eb = float(base[idx + 1])
                width = eb - ea
                for numerator in range(1, denominator, 2):
                    level_candidates.append(ea + (numerator / denominator) * width)
            if add_candidates(level_candidates):
                completed_depth = depth

        scout = np.asarray(sorted(nodes), dtype=float)
        spacing = np.diff(scout)
        return scout, {
            "base_node_count": int(base.size),
            "node_count": int(scout.size),
            "extra_node_count": int(extra_used),
            "requested_depth": int(theta_scan_depth),
            "completed_depth": int(completed_depth),
            "max_extra_nodes": (
                None if theta_scout_max_extra_nodes is None else int(theta_scout_max_extra_nodes)
            ),
            "budget_exhausted": bool(budget_exhausted),
            "min_spacing": float(np.min(spacing)) if spacing.size else None,
            "max_spacing": float(np.max(spacing)) if spacing.size else None,
        }

    def _do_integration() -> np.ndarray:
        nonlocal use_bisection
        nonlocal use_theta_detector
        nonlocal windows
        nonlocal theta_roots
        nonlocal theta_root_clusters
        nonlocal bisection_samples
        nonlocal bisection_intervals
        nonlocal bisection_depth_max
        nonlocal max_metric
        nonlocal resonance_window_min
        nonlocal resonance_window_max
        nonlocal n_e_base
        nonlocal theta_scout_base_node_count
        nonlocal theta_scout_node_count
        nonlocal theta_scout_extra_node_count
        nonlocal theta_scout_completed_depth
        nonlocal theta_scout_budget_exhausted
        nonlocal theta_scout_min_spacing
        nonlocal theta_scout_max_spacing

        if use_theta_detector:
            # Scout the matching-coefficient root function on base/log nodes,
            # legacy probes, and a bounded nested dyadic mesh.  These nodes are
            # independent of density curvature, so isolated sign-changing
            # roots cannot hide merely by missing Simpson midpoints.
            e_base = _build_integration_nodes(e0, e1, int(n_e_base), e_base_grid)
            scout_nodes, scout_meta = build_theta_scout_nodes(e_base)
            theta_scout_base_node_count = int(scout_meta["base_node_count"])
            theta_scout_node_count = int(scout_meta["node_count"])
            theta_scout_extra_node_count = int(scout_meta["extra_node_count"])
            theta_scout_completed_depth = int(scout_meta["completed_depth"])
            theta_scout_budget_exhausted = bool(scout_meta["budget_exhausted"])
            theta_scout_min_spacing = scout_meta["min_spacing"]
            theta_scout_max_spacing = scout_meta["max_spacing"]
            scout_vals = eval_energy_batch(scout_nodes.tolist())
            scout_deltas = [d_val for _, d_val in scout_vals]
            theta_roots = locate_theta_roots(scout_nodes, scout_deltas)

            if resonance_window_min is None:
                resonance_window_min = max(6.0 * refine_min_width, 32.0 * float(theta_root_tol))
            if resonance_window_max is None:
                resonance_window_max = 0.2 * (e1 - e0)

            if theta_roots:
                roots_by_energy = sorted(theta_roots, key=lambda item: float(item["energy"]))
                # Roots in different l channels may be exactly coincident.
                # Treating each as a separate neighbour would shrink both
                # symmetric windows to zero.  Cluster only within the Brent
                # location tolerance; genuinely distinct nearby roots retain
                # separate centred panels.
                cluster_tol = max(
                    8.0 * float(theta_root_tol),
                    32.0 * np.finfo(float).eps * max(abs(e0), abs(e1), 1.0),
                )
                member_clusters: list[list[dict]] = []
                for item in roots_by_energy:
                    if (
                        member_clusters
                        and abs(
                            float(item["energy"])
                            - float(member_clusters[-1][-1]["energy"])
                        )
                        <= cluster_tol
                    ):
                        member_clusters[-1].append(item)
                    else:
                        member_clusters.append([item])

                theta_root_clusters = []
                for cluster_id, members in enumerate(member_clusters):
                    energies = np.asarray(
                        [float(item["energy"]) for item in members],
                        dtype=float,
                    )
                    center = float(np.mean(energies))
                    channels = sorted(set(int(item["l"]) for item in members))
                    width_est = max(
                        max(float(item["width_est"]) for item in members),
                        float(theta_root_tol),
                    )
                    cluster = {
                        "cluster_id": int(cluster_id),
                        "energy": center,
                        "energy_min": float(np.min(energies)),
                        "energy_max": float(np.max(energies)),
                        "channels": channels,
                        "root_count": int(len(members)),
                        "width_est": float(width_est),
                    }
                    theta_root_clusters.append(cluster)
                    for item in members:
                        item["cluster_id"] = int(cluster_id)
                        item["cluster_energy"] = center
                        item["cluster_channels"] = channels

                # Construct disjoint panels centred on each root cluster.  A
                # centred first Simpson evaluation is essential: otherwise an
                # ultra-narrow peak could again fall between all evaluations.
                theta_windows: list[tuple[float, float]] = []
                for i_root, cluster in enumerate(theta_root_clusters):
                    center = float(cluster["energy"])
                    width_est = float(cluster["width_est"])
                    half = min(
                        max(float(resonance_window_factor) * width_est, float(resonance_window_min)),
                        float(resonance_window_max),
                    )
                    left_limit = e0
                    right_limit = e1
                    if i_root > 0:
                        left_limit = 0.5 * (
                            float(theta_root_clusters[i_root - 1]["energy"]) + center
                        )
                    if i_root + 1 < len(theta_root_clusters):
                        right_limit = 0.5 * (
                            center + float(theta_root_clusters[i_root + 1]["energy"])
                        )
                    # Symmetry is kept even next to an energy boundary or a
                    # neighbouring root by shrinking, never shifting, a panel.
                    half = min(half, center - left_limit, right_limit - center)
                    if half <= 4.0 * np.finfo(float).eps * max(abs(center), 1.0):
                        continue
                    low = float(center - half)
                    high = float(center + half)
                    theta_windows.append((low, high))
                    cluster["window_lo"] = low
                    cluster["window_hi"] = high
                    for item in roots_by_energy:
                        if int(item.get("cluster_id", -1)) == int(cluster["cluster_id"]):
                            item["window_lo"] = low
                            item["window_hi"] = high

                windows = theta_windows
                if windows:
                    # Keep base resolution in the gaps, but deliberately drop
                    # base nodes inside a resonance panel so that the panel is
                    # not split away from its known centre.
                    cuts = [float(e0), float(e1)]
                    for val in e_base:
                        e_val = float(val)
                        if not any(w0 < e_val < w1 for w0, w1 in windows):
                            cuts.append(e_val)
                    for w0, w1 in windows:
                        cuts.extend([float(w0), float(w1)])
                    cuts = sorted(set(cuts))
                    coarse_segments: list[tuple[float, float]] = []
                    for i_cut in range(len(cuts) - 1):
                        sa = float(cuts[i_cut])
                        sb = float(cuts[i_cut + 1])
                        if sb <= sa or segment_in_window(sa, sb, windows):
                            continue
                        coarse_segments.append((sa, sb))

                    endpoints: list[float] = []
                    for sa, sb in coarse_segments + windows:
                        endpoints.extend([sa, sb])
                    # Prime the exact centres as well.  They are then reused by
                    # the first adaptive Simpson evaluation of each window.
                    endpoints.extend(
                        float(cluster["energy"]) for cluster in theta_root_clusters
                    )
                    endpoints.extend(float(item["energy"]) for item in roots_by_energy)
                    eval_energy_batch(endpoints)

                    # Apply the same error-controlled Simpson rule both inside
                    # and outside the carved panels.  The known resonance
                    # centre is the midpoint of each window, so it cannot be
                    # skipped; retaining adaptivity in the gaps prevents the
                    # scout mode from degrading otherwise-smooth continuum
                    # quadrature relative to the ordinary Simpson path.
                    n_r_local = np.zeros_like(r, dtype=float)
                    intervals = []
                    for sa, sb in coarse_segments + windows:
                        na, da = cache[sa]
                        nb, db = cache[sb]
                        intervals.append((sa, sb, na, nb, da, db, 0))
                    if use_parallel:
                        n_r_local += integrate_intervals_parallel(intervals)
                    else:
                        for sa, sb, na, nb, da, db, depth in intervals:
                            n_r_local += integrate_interval(sa, sb, na, nb, da, db, depth)
                    return n_r_local

            # No root (or no usable centred window): retain the safe global
            # Simpson fallback rather than returning a coarse integral.
            use_theta_detector = False

        if use_bisection:
            e_nodes = _build_integration_nodes(e0, e1, int(n_e_base), e_base_grid)
            # Precompute delta on base nodes (batch for parallel).
            if use_parallel:
                node_vals = eval_energy_batch(e_nodes.tolist())
                d_nodes = [d for _, d in node_vals]
            else:
                d_nodes = []
                for e in e_nodes:
                    _, d = eval_energy(float(e))
                    d_nodes.append(d)

            delta_threshold = float(delta_tol) if delta_tol is not None else np.inf
            max_depth_local = int(bisection_max_depth) if bisection_max_depth is not None else int(e_max_depth)
            if resonance_window_min is None:
                resonance_window_min = 6.0 * e_min_width
            if resonance_window_max is None:
                resonance_window_max = 0.2 * (e1 - e0)

            candidates = []
            max_metric = 0.0
            for i in range(len(e_nodes) - 1):
                metric = delta_metric(d_nodes[i], d_nodes[i + 1])
                max_metric = max(max_metric, metric)
                if metric > delta_threshold:
                    candidates.append((float(e_nodes[i]), float(e_nodes[i + 1]), d_nodes[i], d_nodes[i + 1], metric))

            # Weak-resonance fallback: no interval exceeded the delta threshold.
            fallback_to_simpson = len(candidates) == 0

            raw_windows = []
            if not fallback_to_simpson:
                for ea, eb, da, db, metric in candidates:
                    bisection_intervals += 1
                    a, b, samples = bisect_resonance_interval(ea, eb, da, db, max_depth_local)
                    bisection_samples += len(samples)
                    bisection_depth_max = max(bisection_depth_max, len(samples) - 2)
                    center = 0.5 * (a + b)
                    width = max(b - a, e_min_width)
                    half = min(max(resonance_window_factor * width, resonance_window_min), resonance_window_max)
                    low = max(e0, center - half)
                    high = min(e1, center + half)
                    raw_windows.append((float(low), float(high), float(metric)))

            if not fallback_to_simpson:
                raw_windows.sort(key=lambda w: w[0])
                merged = []
                for w0, w1, strength in raw_windows:
                    if merged and w0 <= merged[-1][1]:
                        prev = merged[-1]
                        merged[-1] = (prev[0], max(prev[1], w1), max(prev[2], strength))
                    else:
                        merged.append((w0, w1, strength))

                if resonance_max_windows is not None and len(merged) > int(resonance_max_windows):
                    merged = sorted(merged, key=lambda w: w[2], reverse=True)[: int(resonance_max_windows)]
                    merged.sort(key=lambda w: w[0])

                windows = [(w0, w1) for w0, w1, _ in merged]

                segments_refine: list[tuple[float, float]] = []
                segments_coarse: list[tuple[float, float]] = []
                for i in range(len(e_nodes) - 1):
                    ea = float(e_nodes[i])
                    eb = float(e_nodes[i + 1])
                    segments = split_interval_by_windows(ea, eb, windows)
                    for sa, sb in segments:
                        if segment_in_window(sa, sb, windows):
                            segments_refine.append((sa, sb))
                        else:
                            segments_coarse.append((sa, sb))

                # Prime cache with all segment endpoints.
                endpoints = []
                for sa, sb in segments_refine + segments_coarse:
                    endpoints.extend([sa, sb])
                if endpoints:
                    eval_energy_batch(endpoints)

                n_r_local = np.zeros_like(r, dtype=float)

                # Coarse segments: single Simpson step with batched midpoints.
                if segments_coarse:
                    mids = [0.5 * (sa + sb) for sa, sb in segments_coarse]
                    mid_vals = eval_energy_batch(mids)
                    for (sa, sb), (nm, _) in zip(segments_coarse, mid_vals):
                        na, _ = cache[sa]
                        nb, _ = cache[sb]
                        n_r_local += (sb - sa) * (na + 4.0 * nm + nb) / 6.0

                # Refinement segments: adaptive refinement with batched midpoints.
                if segments_refine:
                    intervals = []
                    for sa, sb in segments_refine:
                        na, da = cache[sa]
                        nb, db = cache[sb]
                        intervals.append((sa, sb, na, nb, da, db, 0))
                    n_r_local += integrate_intervals_parallel(intervals)

                return n_r_local

            if fallback_to_simpson:
                # Fall back to global Simpson refinement when no resonance window is detected.
                use_bisection = False

        if not use_bisection:
            e_nodes = _build_integration_nodes(e0, e1, int(n_e_base), e_base_grid)
            node_vals = eval_energy_batch(e_nodes.tolist()) if use_parallel else [eval_energy(float(e)) for e in e_nodes]
            n_nodes = [n for n, _ in node_vals]
            d_nodes = [d for _, d in node_vals]

            intervals = []
            for i in range(len(e_nodes) - 1):
                intervals.append((float(e_nodes[i]), float(e_nodes[i + 1]), n_nodes[i], n_nodes[i + 1],
                                  d_nodes[i], d_nodes[i + 1], 0))

            if use_parallel:
                return integrate_intervals_parallel(intervals)

            n_r_local = np.zeros_like(r, dtype=float)
            for e0_i, e1_i, n0_i, n1_i, d0_i, d1_i, depth in intervals:
                n_r_local += integrate_interval(e0_i, e1_i, n0_i, n1_i, d0_i, d1_i, depth)
            return n_r_local

        return np.zeros_like(r, dtype=float)

    if use_parallel:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=int(n_jobs),
            initializer=_init_scatter_worker,
            initargs=(
                v_eff,
                r,
                mu,
                temperature,
                l_max,
                grid_kind,
                grid_step,
                l_pad,
                match_fraction,
                match_slice,
                match_r_cut,
                match_fraction_mode,
                match_width,
                match_kr_min,
                match_v_tol,
                match_min_points,
                match_asymptotic,
                match_coulomb_tol,
                match_allow_shift,
                match_fallback,
                prop_rescale_limit,
                apply_occ,
                l_cap_strategy,
            ),
        ) as pool:
            n_r = _do_integration()
    else:
        n_r = _do_integration()

    meta = {
        "n_eval": n_eval_new,
        "n_cache_init": cache_init,
        "n_cache_hits": cache_hits,
        "n_cache_total": len(cache),
        "max_depth": max_depth,
        "e_min": e0,
        "e_max": e1,
        "n_jobs": int(n_jobs) if use_parallel else 1,
        "n_base": n_e_base,
        "e_base_grid": str(e_base_grid),
        "resonance_hits": resonance_hits,
        "delta_hits": delta_hits,
        "delta_tol": float(delta_tol) if delta_tol is not None else None,
        "delta_mode": str(delta_mode),
        "adaptive_mode": adaptive_mode,
        "adaptive_parallel_mode": "batch",
        "adaptive_parallel_mode_requested": str(parallel_mode_requested),
        "theta_shard_mode_forced_batch": bool(theta_shard_mode_forced_batch),
        "adaptive_shards": int(adaptive_shards) if adaptive_shards is not None else (int(n_jobs) if use_parallel else 1),
        "adaptive_shard_policy": str(shard_policy),
        "n_windows": len(windows),
        "resonance_windows": [(float(w0), float(w1)) for w0, w1 in windows],
        "bisection_fallback": bool(not use_bisection and adaptive_mode in ("bisection", "bisection_simpson", "bisection-local")),
        "max_delta_metric": float(max_metric) if use_bisection else None,
        "bisection_intervals": bisection_intervals,
        "bisection_samples": bisection_samples,
        "bisection_max_depth": bisection_depth_max,
        "theta_detector_enabled": adaptive_mode in ("phase-root", "theta-scout", "theta-local"),
        "theta_fallback": bool(
            not windows
            and adaptive_mode in ("phase-root", "theta-scout", "theta-local")
        ),
        "theta_candidates": int(theta_candidates),
        "theta_rejected": int(theta_rejected),
        "theta_root_evals": int(theta_root_evals),
        "theta_root_tol": float(theta_root_tol),
        "theta_refine_min_width": float(refine_min_width),
        "theta_refine_depth": int(refine_depth_limit),
        "theta_roots": [dict(item) for item in theta_roots],
        "theta_root_clusters": [dict(item) for item in theta_root_clusters],
        "theta_scout_requested_depth": int(theta_scan_depth),
        "theta_scout_completed_depth": int(theta_scout_completed_depth),
        "theta_scout_base_node_count": int(theta_scout_base_node_count),
        "theta_scout_node_count": int(theta_scout_node_count),
        "theta_scout_extra_node_count": int(theta_scout_extra_node_count),
        "theta_scout_max_extra_nodes": (
            None
            if theta_scout_max_extra_nodes is None
            else int(theta_scout_max_extra_nodes)
        ),
        "theta_scout_budget_exhausted": bool(theta_scout_budget_exhausted),
        "theta_scout_min_spacing": (
            None if theta_scout_min_spacing is None else float(theta_scout_min_spacing)
        ),
        "theta_scout_max_spacing": (
            None if theta_scout_max_spacing is None else float(theta_scout_max_spacing)
        ),
        "theta_scout_limitation": "finite_mesh_no_arbitrary_subgrid_guarantee",
        "near_zero_log_grid": bool(near_zero_enabled),
        "near_zero_log_anchor_count": int(len(near_zero_anchors)),
        "near_zero_log_anchors": sorted(near_zero_anchors),
        "apply_occ": bool(apply_occ),
    }
    if bool(collect_perf):
        wall_s = time.perf_counter() - t_wall
        meta.update(_scatter_perf_meta(perf_accum, wall_s))
    return n_r, meta


class QuantumContinuumFree(ContinuumModel):
    """
    Quantum continuum prototype using free-electron wavefunctions.

    Notes
    -----
    This is a baseline model; for realistic AA/PAMD, replace the free-electron
    wavefunctions with scattering solutions in V_eff(r).
    """

    def density(self,
                r: np.ndarray,
                mu: float,
                temperature: float,
                params: Dict[str, float] | None = None) -> np.ndarray:
        params = params or {}
        l_max = int(params.get("l_max", 6))
        e_max = float(params.get("e_max", max(mu + 10.0 * temperature, 5.0 * temperature)))
        n_e = int(params.get("n_e", 200))
        e_grid = np.linspace(0.0, e_max, n_e)
        return continuum_density_free(r, mu, temperature, e_grid, l_max)


class QuantumContinuumScattering(ContinuumModel):
    """
    Quantum continuum model using Numerov scattering solutions in V_eff(r).

    Implements Starrett2014 Eq. (A3) by integrating scattering-state
    densities over energy and summing partial waves.
    """

    def density(self,
                r: np.ndarray,
                mu: float,
                temperature: float,
                params: Dict[str, float] | None = None) -> np.ndarray:
        """
        Compute n_cont(r) using Numerov scattering solutions.

        Numbered workflow
        -----------------
        1) Parse grid / matching / integration parameters.
        2) Compute A3 continuum density with either:
           - adaptive energy integration, or
           - fixed linear energy grid.
        3) If `tail_match=True` and target includes `"cont"`, apply B3 splice:
           - choose `R_cut` (direct / auto / scan),
           - fit `(A, B, delta)` near `R_cut`,
           - replace `n_cont(r >= R_cut)` by analytic B3 tail.
        4) Return the final continuum profile.

        Required params
        ---------------
        v_eff : ndarray
            Effective potential on the same grid as r.

        Key options
        -----------
        energy_mode       : "adaptive" (default) or "linear".
        e_min, e_max, n_e : energy bounds / samples for A3 integral.
        l_max, l_pad      : partial-wave sum parameters.
        l_cap_strategy    : "match" (default), "rmax", or "none".
        match_*           : asymptotic matching controls.
        n_jobs            : parallelize across energies (linear) or midpoint batches (adaptive).
        """
        params = params or {}
        v_eff = params.get("v_eff", None)
        if v_eff is None:
            raise ValueError("params['v_eff'] is required for scattering continuum.")

        # (1) Grid step for Numerov propagation.
        # Sqrt-grid is fixed in the production path, so derive dxi directly
        # from r instead of exposing grid-kind controls in user params.
        grid_kind = "sqrt"
        grid_step = float(np.sqrt(r[1]) - np.sqrt(r[0]))

        # (2) Continuum integration settings.
        l_max = int(params.get("l_max", 6))
        l_pad = int(params.get("l_pad", 2))
        e_max = float(params.get("e_max", max(mu + 10.0 * temperature, 5.0 * temperature)))
        e_min = float(params.get("e_min", 1e-6))
        e_min = max(e_min, 1e-12)
        n_e = int(params.get("n_e", 200)) # energy grid points number
        match_fraction = float(params.get("match_fraction", 0.2))
        match_slice = params.get("match_slice", None)
        match_r_cut = params.get("match_r_cut", None)
        match_fraction_mode = str(params.get("match_fraction_mode", "r"))
        match_width = params.get("match_width", None)
        if match_width is not None:
            match_width = float(match_width)
        match_kr_min = params.get("match_kr_min", 4.0)
        match_v_tol = params.get("match_v_tol", 1e-4)
        match_min_points = int(params.get("match_min_points", 12))
        match_asymptotic = str(params.get("match_asymptotic", "auto"))
        match_coulomb_tol = float(params.get("match_coulomb_tol", 0.1))
        match_allow_shift = bool(params.get("match_allow_shift", True))
        match_fallback = str(params.get("match_fallback", "free"))
        prop_rescale_limit = params.get("prop_rescale_limit", 1e6)
        l_cap_strategy = str(params.get("l_cap_strategy", "match"))
        energy_cache = params.get("energy_cache", None)
        n_jobs = params.get("n_jobs", None)
        if n_jobs is not None:
            n_jobs = int(n_jobs)

        # Warn if l_max is too small for k_max * R_max.
        if r.size > 0 and l_cap_strategy in ("rmax", "global"):
            k_max = np.sqrt(max(2.0 * e_max, 0.0))
            l_rec = int(np.ceil(k_max * r[-1]))
            if l_max < l_rec:
                warnings.warn(
                    f"l_max={l_max} < ceil(k_max*R_max)={l_rec}; "
                    "continuum density may be underestimated. "
                    "Consider increasing l_max or reducing e_max.",
                    RuntimeWarning,
                )

        # Build matching window for asymptotic normalization.
        if match_slice is None and match_r_cut is not None:
            idx_cut = int(np.searchsorted(r, float(match_r_cut)))
            if 0 < idx_cut < r.size - 1:
                if match_width is not None:
                    idx_end = int(np.searchsorted(r, float(match_r_cut + match_width)))
                else:
                    idx_end = r.size
                match_slice = (idx_cut, min(max(idx_end, idx_cut + 1), r.size))

        # (3) B3 tail settings (optional splice for the continuum tail only).
        tail_match = bool(params.get("tail_match", False))
        tail_match_target = str(params.get("tail_match_target", "cont")).lower()
        tail_auto = bool(params.get("tail_auto", False))
        tail_strategy = params.get("tail_strategy", None)
        tail_scan = False
        if tail_strategy is not None:
            tail_strategy = str(tail_strategy).lower()
            if tail_strategy in ("auto", "scan"):
                tail_auto = True
            tail_scan = tail_strategy == "scan"
        tail_auto_r_fraction = float(params.get("tail_auto_r_fraction", 0.7))
        tail_auto_min_points = int(params.get("tail_auto_min_points", 12))
        tail_auto_rel_tol = params.get("tail_auto_rel_tol", None)
        if tail_auto_rel_tol is not None:
            tail_auto_rel_tol = float(tail_auto_rel_tol)
        tail_auto_abs_tol = params.get("tail_auto_abs_tol", None)
        if tail_auto_abs_tol is not None:
            tail_auto_abs_tol = float(tail_auto_abs_tol)
        tail_auto_v_tol = params.get("tail_auto_v_tol", match_v_tol)
        if tail_auto_v_tol is not None:
            tail_auto_v_tol = float(tail_auto_v_tol)
        tail_auto_fallback = str(params.get("tail_auto_fallback", "fraction"))
        tail_scan_r_cuts = params.get("tail_scan_r_cuts", None)
        if tail_scan_r_cuts is not None and not isinstance(tail_scan_r_cuts, (list, tuple, np.ndarray)):
            tail_scan_r_cuts = [tail_scan_r_cuts]
        if tail_scan_r_cuts is not None:
            tail_scan_r_cuts = [float(val) for val in tail_scan_r_cuts]
        tail_scan_r_fractions = params.get("tail_scan_r_fractions", (0.6, 0.7, 0.8))
        if tail_scan_r_fractions is not None and not isinstance(tail_scan_r_fractions, (list, tuple, np.ndarray)):
            tail_scan_r_fractions = [tail_scan_r_fractions]
        if tail_scan_r_fractions is not None:
            tail_scan_r_fractions = [float(val) for val in tail_scan_r_fractions]
        tail_scan_charge_tol_rel = float(params.get("tail_scan_charge_tol_rel", 1e-3))
        tail_scan_charge_tol_abs = float(params.get("tail_scan_charge_tol_abs", 1e-6))
        tail_scan_delta_tol = params.get("tail_scan_delta_tol", None)
        if tail_scan_delta_tol is not None:
            tail_scan_delta_tol = float(tail_scan_delta_tol)
        tail_scan_param_tol_rel = params.get("tail_scan_param_tol_rel", None)
        if tail_scan_param_tol_rel is not None:
            tail_scan_param_tol_rel = float(tail_scan_param_tol_rel)
        tail_r_cut = params.get("tail_r_cut", None)
        if tail_r_cut is not None:
            tail_r_cut = float(tail_r_cut)
        tail_cut_anchor = None
        if tail_match:
            if tail_r_cut is None:
                tail_cut_anchor = float(tail_auto_r_fraction * r[-1])
            else:
                tail_cut_anchor = float(tail_r_cut)
            idx_cut = int(np.searchsorted(r, tail_cut_anchor))
            if match_slice is None and 0 < idx_cut < r.size - 1:
                match_slice = (idx_cut, r.size)

        # Energy integration strategy (adaptive vs linear grid).
        energy_mode = str(params.get("energy_mode", "adaptive"))
        if energy_mode == "adaptive":
            e_tol = float(params.get("e_tol", 1e-3))
            e_max_depth = int(params.get("e_max_depth", 10))
            e_min_width = float(params.get("e_min_width", 1e-4))
            n_e_base = int(params.get("n_e_base", 8))
            e_base_grid = str(params.get("e_base_grid", "linear"))
            adaptive_parallel_mode = str(params.get("adaptive_parallel_mode", "batch"))
            adaptive_shards = params.get("adaptive_shards", None)
            if adaptive_shards is not None:
                adaptive_shards = int(adaptive_shards)
            resonance_tol = params.get("resonance_tol", None)
            resonance_r_fractions = params.get("resonance_r_fractions", (0.25, 0.5, 0.75))
            resonance_floor = float(params.get("resonance_floor", 1e-8))
            adaptive_mode = str(params.get("adaptive_mode", "bisection"))
            # Bisection defaults: favor total phase shifts and tighter windows.
            if adaptive_mode == "bisection":
                delta_mode_default = "sum"
                delta_tol_default = np.pi
                bisection_depth_default = 10
                window_factor_default = 6.0
            else:
                delta_mode_default = "max"
                delta_tol_default = np.pi / 2.0
                bisection_depth_default = None
                window_factor_default = 12.0

            bisection_max_depth = params.get("bisection_max_depth", bisection_depth_default)
            if bisection_max_depth is not None:
                bisection_max_depth = int(bisection_max_depth)

            resonance_window_factor = float(params.get("resonance_window_factor", window_factor_default))
            resonance_window_min = params.get("resonance_window_min", None)
            if resonance_window_min is not None:
                resonance_window_min = float(resonance_window_min)
            resonance_window_max = params.get("resonance_window_max", None)
            if resonance_window_max is not None:
                resonance_window_max = float(resonance_window_max)
            resonance_max_windows = params.get("resonance_max_windows", None)
            if resonance_max_windows is not None:
                resonance_max_windows = int(resonance_max_windows)
            resonance_theta_root_tol = params.get("resonance_theta_root_tol", None)
            if resonance_theta_root_tol is not None:
                resonance_theta_root_tol = float(resonance_theta_root_tol)
            resonance_theta_max_roots = params.get("resonance_theta_max_roots", None)
            if resonance_theta_max_roots is not None:
                resonance_theta_max_roots = int(resonance_theta_max_roots)
            resonance_theta_refine_depth = params.get("resonance_theta_refine_depth", None)
            if resonance_theta_refine_depth is not None:
                resonance_theta_refine_depth = int(resonance_theta_refine_depth)
            # Adaptive integration (refine around resonances/phase jumps).
            n_r, _ = continuum_density_scattering_adaptive(
                v_eff,
                r,
                mu,
                temperature,
                e_min,
                e_max,
                l_max,
                grid_kind,
                grid_step,
                l_pad=l_pad,
                match_fraction=match_fraction,
                match_slice=match_slice,
                match_r_cut=match_r_cut,
                match_fraction_mode=match_fraction_mode,
                match_width=match_width,
                match_kr_min=match_kr_min,
                match_v_tol=match_v_tol,
                match_min_points=match_min_points,
                match_asymptotic=match_asymptotic,
                match_coulomb_tol=match_coulomb_tol,
                match_allow_shift=match_allow_shift,
                match_fallback=match_fallback,
                prop_rescale_limit=prop_rescale_limit,
                l_cap_strategy=l_cap_strategy,
                e_tol=e_tol,
                e_max_depth=e_max_depth,
                e_min_width=e_min_width,
                n_e_base=n_e_base,
                e_base_grid=e_base_grid,
                near_zero_log_grid=bool(params.get("near_zero_log_grid", True)),
                near_zero_log_points_per_decade=int(
                    params.get("near_zero_log_points_per_decade", 4)
                ),
                near_zero_log_max_nodes=int(params.get("near_zero_log_max_nodes", 24)),
                near_zero_log_max_energy=params.get("near_zero_log_max_energy", 1.0e-2),
                resonance_tol=resonance_tol,
                resonance_r_fractions=resonance_r_fractions,
                resonance_floor=resonance_floor,
                delta_tol=float(params.get("delta_tol", delta_tol_default)),
                delta_mode=str(params.get("delta_mode", delta_mode_default)),
                adaptive_mode=adaptive_mode,
                bisection_max_depth=bisection_max_depth,
                resonance_window_factor=resonance_window_factor,
                resonance_window_min=resonance_window_min,
                resonance_window_max=resonance_window_max,
                resonance_max_windows=resonance_max_windows,
                resonance_theta_l_min=int(params.get("resonance_theta_l_min", 1)),
                resonance_theta_probe_count=int(params.get("resonance_theta_probe_count", 1)),
                resonance_theta_scan_depth=int(params.get("resonance_theta_scan_depth", 3)),
                resonance_theta_scout_max_extra_nodes=params.get(
                    "resonance_theta_scout_max_extra_nodes",
                    128,
                ),
                resonance_theta_root_tol=resonance_theta_root_tol,
                resonance_theta_sharpness_min=float(params.get("resonance_theta_sharpness_min", 2.0)),
                resonance_theta_max_roots=resonance_theta_max_roots,
                resonance_theta_refine_depth=resonance_theta_refine_depth,
                energy_cache=energy_cache,
                n_jobs=n_jobs,
                adaptive_parallel_mode=adaptive_parallel_mode,
                adaptive_shards=adaptive_shards,
                adaptive_shard_policy=str(params.get("adaptive_shard_policy", "egrid")),
            )
        else:
            # Fixed energy grid integration.
            e_grid = np.linspace(e_min, e_max, n_e)
            n_r = continuum_density_scattering(
                v_eff,
                r,
                mu,
                temperature,
                e_grid,
                l_max,
                grid_kind,
                grid_step,
                l_pad=l_pad,
                match_fraction=match_fraction,
                match_slice=match_slice,
                match_r_cut=match_r_cut,
                match_fraction_mode=match_fraction_mode,
                match_width=match_width,
                match_kr_min=match_kr_min,
                match_v_tol=match_v_tol,
                match_min_points=match_min_points,
                match_asymptotic=match_asymptotic,
                match_coulomb_tol=match_coulomb_tol,
                match_allow_shift=match_allow_shift,
                match_fallback=match_fallback,
                prop_rescale_limit=prop_rescale_limit,
                l_cap_strategy=l_cap_strategy,
                energy_cache=energy_cache,
                n_jobs=n_jobs,
            )

        if tail_match and tail_match_target in ("cont", "both"):
            from .ideal import ideal_unbound_density
            from .tail import (
                apply_tail_match,
                select_tail_cut,
                scan_tail_match,
                select_tail_cut_converged,
            )

            # (3.1) Tail reference state:
            # n0, mu_id are the asymptotic reference quantities in B3.
            n0 = float(params.get("tail_n0", ideal_unbound_density(mu, temperature)))
            mu_id = float(params.get("tail_mu_id", mu))
            fit_points = int(params.get("tail_fit_points", 16))
            tail_fit_window_mode = str(
                params.get("tail_fit_window_mode", "local")
            ).strip().lower()
            if tail_fit_window_mode == "auto" and tail_match_target == "cont":
                tail_fit_window_mode = "local"
            if tail_fit_window_mode not in ("auto", "physical", "local"):
                raise ValueError(
                    "tail_fit_window_mode must be 'auto', 'physical', or 'local'."
                )
            tail_r_fit_max = (
                params.get("tail_r_fit_max", None)
                if tail_fit_window_mode in ("auto", "physical")
                else None
            )
            blend_points = int(params.get("tail_blend_points", 0))
            tail_fallback_on_error = bool(params.get("tail_fallback_on_error", True))
            r_cut = float(tail_cut_anchor if tail_cut_anchor is not None else 0.7 * r[-1])
            idx_cut = int(np.searchsorted(r, r_cut))
            if tail_scan:
                # (3.2) Multi-cut scan: fit candidate R_cut values and select first converged one.
                if tail_scan_r_cuts is None:
                    if tail_scan_r_fractions is None:
                        r_cuts = [r_cut]
                    else:
                        r_cuts = [float(frac * r[-1]) for frac in tail_scan_r_fractions]
                else:
                    r_cuts = list(tail_scan_r_cuts)
                r_cuts = [rc for rc in r_cuts if rc >= r_cut and rc < float(r[-1])]
                if not r_cuts:
                    r_cuts = [r_cut]
                results, _ = scan_tail_match(
                    r,
                    n_r,
                    n0,
                    mu_id,
                    temperature,
                    r_cuts=r_cuts,
                    fit_points=fit_points,
                    r_fit_max=tail_r_fit_max,
                    fit_window_mode=tail_fit_window_mode,
                    blend_points=blend_points,
                )
                idx_scan, _ = select_tail_cut_converged(
                    results,
                    charge_tol_rel=tail_scan_charge_tol_rel,
                    charge_tol_abs=tail_scan_charge_tol_abs,
                    delta_tol=tail_scan_delta_tol,
                    param_tol_rel=tail_scan_param_tol_rel,
                )
                if idx_scan is not None:
                    idx_cut = idx_scan
                else:
                    # Fallback policy if scan does not converge.
                    fallback = tail_auto_fallback.lower()
                    if fallback == "skip":
                        warnings.warn(
                            "Tail scan did not converge; skipping tail match.",
                            RuntimeWarning,
                        )
                        return n_r
                    if fallback == "last" and results:
                        idx_cut = int(results[-1]["idx_cut"])
            elif tail_auto:
                # (3.3) Auto-cut mode: pick first tail-like window meeting n/V criteria.
                idx_auto, _ = select_tail_cut(
                    r,
                    n_r,
                    n0,
                    v_eff=v_eff,
                    r_min=r_cut,
                    r_fraction=tail_auto_r_fraction,
                    rel_tol=tail_auto_rel_tol,
                    abs_tol=tail_auto_abs_tol,
                    v_tol=tail_auto_v_tol,
                    min_points=tail_auto_min_points,
                )
                if idx_auto is not None:
                    idx_cut = idx_auto
                elif tail_auto_fallback.lower() == "skip":
                    return n_r
            if 0 < idx_cut < r.size - 1:
                # (3.4) Final B3 splice: keep A3 inside, replace tail outside.
                # When requested, failure falls back to the unmodified A3 density
                # so production workflows can compare A3 vs B3 without losing the
                # previously validated pure-KS path.
                try:
                    n_r, _ = apply_tail_match(
                        r,
                        n_r,
                        n0,
                        mu_id,
                        temperature,
                        idx_cut,
                        fit_points=fit_points,
                        r_fit_max=tail_r_fit_max,
                        local_fit_width=params.get("tail_local_fit_width", None),
                        fit_window_mode=tail_fit_window_mode,
                        blend_points=blend_points,
                    )
                except Exception as exc:
                    if not tail_fallback_on_error:
                        raise
                    warnings.warn(
                        f"B3 tail match failed at r_cut={float(r[idx_cut]):.6f}; "
                        f"keeping original A3 density. reason={exc}",
                        RuntimeWarning,
                    )

        return n_r
