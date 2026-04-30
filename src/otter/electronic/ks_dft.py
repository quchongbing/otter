"""
otter/electronic/ks_dft.py

Purpose
-------
Provide a minimal quantum Kohn-Sham AA solver (full/external) for IS models.

Methods
-------
- Solve bound states with sparse Numerov on a bound-state grid.
- Compute continuum density with a selectable continuum model.
- Assemble V_eff using Starrett2014 Eqs. (4) and (7) with g_II(r).
- Interpolate bound density onto the continuum grid for SCF updates.
- Iterate with linear mixing; optionally enforce mu neutrality.
- Optional Poisson-Helmholtz screening stabilizes non-neutral iterations.
- Define n_ion from bound states with cutoff and M(e) weights, then set
  n_scr = n_full - n_ext - n_ion (Starrett2014/2013).

Equations
---------
Bound density (A2):
  n_b(r) = sum_{n,l} g_n * [2(2l+1)/(4*pi)] * |R_{n,l}(r)|^2
Continuum density (A3):
  n_c(r) = integral dE g_k * sum_l [2(2l+1)/(4*pi)] * |y_{k,l}(r)/r|^2
Effective potentials (Starrett2014 Eqs. 4,7):
  V_full = -Z/r + Hartree[n_full - n0*g_II] + V_xc[n_full] - V_xc[n0]
  V_ext  =        Hartree[n_ext  - n0*g_II] + V_xc[n_ext]  - V_xc[n0]

References
----------
- C. E. Starrett & D. Saumon (2014), Appendix A/B.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import math
import time
import warnings
from typing import Dict, Any, Tuple
import numpy as np
from numba import njit

from otter.numerics.grids import create_sqrt_grid
from otter.numerics.interpolation import interp_to_grid
from otter.electronic.continuum.scattering import (
    fermi_dirac,
    QuantumContinuumScattering,
    continuum_density_scattering_basis,
    continuum_density_scattering_adaptive,
    gamma_from_phase_shift_cache,
)
from otter.electronic.continuum.ideal import IdealContinuum, ideal_unbound_density
from otter.electronic.continuum.hybrid import QuantumContinuumHybrid
from otter.ionic.correlation import IonSphereStepModel, ion_sphere_radius_from_density
from otter.electronic.solvers.bound import solve_bound_states_sparse_numerov
from otter.electronic.potential import effective_potential_full, effective_potential_external
from otter.electronic.densities import (
    _bound_density,
    _electron_count,
    _electron_count_full,
    _ion_cutoff,
    _ion_cutoff_starrett,
    _ion_density,
    _ion_level_weight,
)


_CONT_E_MAX_RETRY_STEP_HA = 2.0
_CONT_E_MAX_RETRY_MAX_TRIES = 32
_CONT_E_MAX_RETRY_CHARGE_REL_TOL = 1.0e-2


def _trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    """
    Trapezoidal quadrature weights for a 1D monotone grid.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    w = np.zeros_like(x)
    if n < 2:
        return w
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    if n > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


@njit(cache=True, fastmath=True)
def _weighted_energy_sum_numba(basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Contract an (n_energy, n_r) basis against energy weights without a 2D temporary.
    """
    n_e, n_r = basis.shape
    out = np.zeros(n_r, dtype=np.float64)
    for i in range(n_e):
        wi = weights[i]
        for j in range(n_r):
            out[j] += basis[i, j] * wi
    return out


def _resolve_iteration_continuum_e_max(params: Dict[str, Any],
                                       *,
                                       mu_ref: float,
                                       temperature: float) -> float:
    """
    Resolve the continuum energy ceiling used for one SCF iteration.

    Parameters
    ----------
    params : dict
        Continuum parameter dictionary. The optional dynamic policy is
        controlled by:

        - ``e_max_mode``: ``"fixed"`` or ``"prev_mu_fd"``
        - ``e_max_occ_tol``: target FD occupation threshold
        - ``e_max_floor``: safety floor applied after the FD inversion

    mu_ref : float
        Reference chemical potential for the current SCF iteration in Ha. In
        the stage-2 inner-mu solve this is the previous SCF iterate.
    temperature : float
        Electron temperature in Ha.

    Returns
    -------
    float
        Energy ceiling `e_max` in Ha for the current iteration.

    Notes
    -----
    For the dynamic ``prev_mu_fd`` mode we choose `e_max` so that the
    Fermi-Dirac occupation at the previous-SCF chemical potential is below a
    target threshold:

      f_FD(E_max; mu_ref, T) = eps

    which gives

      E_max = mu_ref + T ln((1-eps)/eps).

    The returned value is floored by both ``e_min`` and the optional
    ``e_max_floor`` safety bound.
    """
    e_min = max(float(params.get("e_min", 1.0e-6)), 1.0e-12)
    mode = str(params.get("e_max_mode", "fixed")).lower().strip()

    if mode in ("fixed", "off", "none"):
        return float(max(e_min, float(params.get("e_max", e_min))))
    if mode != "prev_mu_fd":
        raise ValueError("continuum e_max_mode must be 'fixed' or 'prev_mu_fd'.")

    temp = max(float(temperature), 1.0e-12)
    occ_tol = float(params.get("e_max_occ_tol", 1.0e-5))
    occ_tol = min(max(occ_tol, 1.0e-12), 0.49)
    fd_span = temp * math.log((1.0 - occ_tol) / occ_tol)
    e_target = float(mu_ref) + fd_span
    e_floor = max(e_min, float(params.get("e_max_floor", e_min)))
    return float(max(e_floor, e_target))


def _resolve_iteration_continuum_l_max(
    params: Dict[str, Any],
    *,
    e_max: float,
    r_eval_max: float,
) -> int:
    """
    Resolve the partial-wave ceiling used for one continuum basis build.

    Parameters
    ----------
    params : dict
        Continuum parameter dictionary. Uses ``l_pad`` and the optional
        ``l_max_ceiling`` safety cap when present.
    e_max : float
        Continuum energy ceiling in Ha used on the current SCF iteration.
    r_eval_max : float
        Right edge of the numerical continuum solve box in Bohr.

    Returns
    -------
    int
        Integer partial-wave ceiling for the current basis build.

    Notes
    -----
    The heuristic follows the usual oscillatory-box estimate

      l_max ~ ceil(k_max * r_eval_max + l_pad),  k_max = sqrt(2 E_max)

    and is recomputed every SCF iteration so a dynamic stage-2 ``e_max`` does
    not silently reuse a stale partial-wave ceiling from stage 1.
    """
    r_use = max(float(r_eval_max), 1.0e-12)
    e_use = max(float(e_max), 0.0)
    l_pad = max(int(params.get("l_pad", 2)), 0)
    l_est = int(math.ceil(math.sqrt(2.0 * e_use) * r_use + float(l_pad)))
    l_est = max(l_est, 0)
    l_cap = params.get("l_max_ceiling", None)
    if l_cap is None:
        return l_est
    try:
        return max(0, min(l_est, int(l_cap)))
    except Exception:
        return l_est

@dataclass
class KSDTFConfig:
    """
    Configuration for a minimal KS-DFT AA solver (IS mode).

    Notes
    -----
    - Bound and continuum both use sqrt in the validated path.
    - mu_mode controls fixed-mu vs neutrality search in the IS volume.
    - ph_kappa enables Poisson-Helmholtz screening during SCF.
    - compute_external controls whether the external continuum is evaluated.
    - shift_v_eff_tail enforces V_eff -> 0 by removing a constant tail offset.
    - bound_energy_cut_mode controls the continuum edge used for bound states
      ("zero", "v_ws", "fixed", or "auto").
    - full_v_eff_outer_decay applies an experimental exponential taper to the
      full effective potential outside a chosen multiple of R_ws. This is a
      non-Starrett diagnostic knob meant to suppress weak outer wells.
    - n0_mode selects how n0 is defined ("ideal", "tail", "window", or "fixed").
      "ideal" uses the free-electron gas at (mu,T), "tail" uses the outer
      continuum-density tail value, "window" uses a user-selected radial
      window average of continuum density, and "fixed" uses n0_fixed.
    - n0_tail_direct forces n0 = tail estimate each iteration (no mixing or ideal init).
    - n0_window_* controls interior-window n0 updates when n0_mode="window".
    - charge_tol (relative) can be used to require physical neutrality
      alongside the potential-mixing convergence check.
    - ion_cut_mode selects the n_ion cutoff: "starrett" (Eq. 80) or
      "smoothstep" (legacy).
    - ion_cut_c sets the Starrett cutoff width parameter c (dimensionless).
    - ion_cut_width sets the smoothstep cutoff width (fraction of R_ws).
    - ion_bound_gamma sets the fixed bound-state broadening (Ha) used in M(e).
    - ion_gamma_mode controls how gamma is obtained ("fixed" or "scattering") for M(e).
    - ion_gamma_scale multiplies the scattering-derived gamma only. It is a
      diagnostic/regularization knob for testing stronger plasma broadening
      without changing the fixed-gamma path.
    - ion_ws_weight_min applies the same WS localization filter to n_ion.
    - bound_occ_mode controls whether M(e) weights apply to n_bound.
    - mixing_scheme selects the SCF mixer ("linear" or "eyert").
    - mixing_m sets the history size M for Eyert mixing.
    - mixing_w0 is the Eyert stabilizer (w0^2 in Eq. 63).
    - n_jobs is kept as a backward-compatible bound-solver argument, but bound
      states are now always solved serially.
    - neutrality_mode selects the mu neutrality constraint:
      "ws" for Q(R_ws)=Z, "pa" for Q_pa=Z, "auto" switches by compute_external.
    - mu_zbar_min sets a minimum Zbar during mu search to avoid atomic branches
      (only meaningful when compute_external is True).
    - mu_solver selects the mu root finder: "bracket" (bisection),
      "brent" (bracketing + Brent), or "secant".
    - mu_verbose controls printing of per-iteration mu solve diagnostics.
    - mu_bounds_strict prevents mu bracket expansion beyond mu_bounds.
    - zbar_tol adds a convergence criterion on Delta Zbar / Zbar.
    - store_scf_snapshots_last stores the last N SCF snapshots (n_full/v_full).
    - dn_tol and dv_tol enable integral-based SCF convergence criteria
      using Delta n and Delta v (see atomec-style definitions).
    """
    Z: int
    temperature: float
    mu: float
    r_ws: float | None = None
    n_i: float | None = None
    # Preferred user-facing radial controls:
    # - rmax: explicit simulation-domain radius (Bohr)
    # - rmin: inner radius used to construct the sqrt grid (Bohr)
    # - n_points: number of radial points (shared by bound and continuum)
    rmax: float | None = None
    rmax_mult: float | None = 10.0
    n_points: int = 800
    # Grid type is fixed to sqrt in the validated solver path.
    # Keep rmax_mult as a legacy fallback: if rmax is None, use
    #   rmax = rmax_mult * R_ws
    rmin: float = 1e-5
    l_list: np.ndarray = field(default_factory=lambda: np.arange(0, 4))
    n_states: int = 6
    n_states_by_l: np.ndarray | None = None
    n_jobs: int | None = None
    boundary: str = "dirichlet"
    xc_model: str = "dirac"
    continuum_model: str = "scattering"
    continuum_params: Dict[str, Any] = field(default_factory=dict)
    compute_external: bool = True
    shift_v_eff_tail: bool = False
    v_tail_fraction: float = 0.001
    v_tail_mode: str = "median"
    full_v_eff_outer_decay: bool = False
    full_v_eff_outer_decay_start_rws: float = 1.0
    full_v_eff_outer_decay_length_rws: float = 0.5
    bound_energy_cut_mode: str = "zero"
    bound_energy_cut: float | None = None
    bound_occ_mode: str = "fd"
    # SCF mixer (linear or Eyert)
    # - mixing_scheme selects the SCF mixer ("eyert" or "linear").
    # - mixing_m sets the Eyert history size M (typical 4–8; smaller is safer).
    # - mixing_w0 is the Eyert stabilizer w0 (Eq. 63). Larger => more damping.
    #   Typical range: 1e-5–1e-3. If oscillatory, increase w0; if slow, decrease.
    mix: float = 0.2
    mixing_scheme: str = "eyert"
    mixing_m: int = 5
    mixing_w0: float = 1e-4
    max_iter: int = 40
    tol: float = 1e-4
    mu_mode: str = "fixed"
    # mu_strategy: "inner" (solve mu inside each SCF step) or "outer" (mu loop wraps SCF)
    mu_strategy: str = "inner"
    mu_bounds: tuple[float, float] = (-200.0, 200.0)
    mu_solver: str = "brent"
    mu_bounds_strict: bool = False
    mu_tol: float = 1e-3
    mu_max_iter: int = 12
    mu_scf_iter: int | None = None
    mu_bracket_step: float = 2.0
    mu_bracket_max_iter: int = 6
    mu_n0_floor: float = 1e-10
    mu_zbar_min: float | None = None
    v_full_init: np.ndarray | None = None
    v_ext_init: np.ndarray | None = None
    ph_kappa: float = 0.0
    ph_kappa_iters: int | None = None
    n0_mode: str = "ideal"
    n0_fixed: float | None = None
    n0_tail_fraction: float = 0.3
    n0_tail_mode: str = "mean"
    n0_tail_mix: float = 1.0
    n0_tail_direct: bool = False
    n0_window_lo_frac: float = 0.75
    n0_window_hi_frac: float = 0.80
    n0_window_mode: str = "mean"
    n0_window_mix: float = 0.2
    n0_window_direct: bool = False
    ion_cut_mode: str = "starrett"
    ion_cut_c: float = 0.05
    ion_cut_width: float = 0.2
    ion_bound_gamma: float = 0.05
    ion_gamma_mode: str = "fixed"
    ion_gamma_scale: float = 1.0
    ion_ws_weight_min: float = 0.0
    neutrality_mode: str = "auto"
    charge_tol: float | None = None
    zbar_tol: float | None = None
    verbose: bool = False
    mu_verbose: bool = False
    print_every: int = 1
    store_scf_snapshots_last: int | None = None
    dn_tol: float | None = None
    dv_tol: float | None = None
    # Per-iteration SCF performance diagnostics.
    perf_diag: bool = False
    perf_print_every: int = 1
    # Optional sub-switches for perf printouts:
    # - perf_show_stage: print aggregated stage timing line ("perf[s]").
    # - perf_show_basis: print adaptive-basis metadata line ("perf[basis]").
    #   (Used by inner-mu SCF where basis precompute metadata is available.)
    perf_show_stage: bool = True
    perf_show_basis: bool = True
    store_final_bound_debug: bool = False

def _resolve_r_ws(n_i: float | None, r_ws: float | None) -> float:
    if r_ws is not None:
        return float(r_ws)
    if n_i is None:
        raise ValueError("Either r_ws or n_i must be provided.")
    return ion_sphere_radius_from_density(float(n_i))

def _build_grid_pair(config: KSDTFConfig) -> Tuple[np.ndarray, float, str, np.ndarray, float, str]:
    r_ws = _resolve_r_ws(config.n_i, config.r_ws)
    if config.rmax is not None:
        rmax = float(config.rmax)
    else:
        if config.rmax_mult is None:
            raise ValueError("Either rmax or rmax_mult must be provided.")
        rmax = float(config.rmax_mult) * r_ws

    # Sqrt grid is the only supported production path for both bound and
    # continuum channels in this solver.
    kind_bound = "sqrt"
    kind_cont = "sqrt"
    # Use a single user-facing point count for both bound and continuum.
    # This keeps the interface minimal while preserving full control over
    # (rmin, rmax, n_points).
    grid_bound = create_sqrt_grid(rmax=rmax, N=config.n_points, rmin=config.rmin)
    grid_cont = create_sqrt_grid(rmax=rmax, N=config.n_points, rmin=config.rmin)
    r_bound, step_bound = grid_bound.r, grid_bound.dxi
    r_cont, step_cont = grid_cont.r, grid_cont.dxi

    return r_cont, step_cont, kind_cont, r_bound, step_bound, kind_bound


def _split_continuum_params_for_full_ext(base_params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split continuum controls into full/external parameter dictionaries.

    Conventions
    -----------
    - Keys without prefix apply to both full and external by default.
    - Keys prefixed with ``ext_`` override only the external branch.
      Example: ``ext_match_v_tol=None`` disables the |V_eff| tail-window
      filter for the external scattering solver while keeping full unchanged.

    Returns
    -------
    (full_params, ext_params)
        Two dictionaries ready for per-branch updates (v_eff, grid steps, cache).
    """
    full_params = {k: v for k, v in base_params.items() if not str(k).startswith("ext_")}
    ext_params = dict(full_params)
    for key, val in base_params.items():
        key_str = str(key)
        if not key_str.startswith("ext_"):
            continue
        core = key_str[4:]
        if core:
            ext_params[core] = val
    return full_params, ext_params


def _apply_external_energy_floor(params_ext: Dict[str, Any],
                                 r: np.ndarray,
                                 v_ext: np.ndarray) -> Dict[str, Any]:
    """
    Raise external continuum e_min to the asymptotic potential floor when needed.

    Why
    ---
    In finite boxes, the external potential can remain positive in the tail
    during intermediate SCF iterations. If E < V_tail, low-energy channels are
    non-oscillatory at the matching edge; forcing an oscillatory fit can create
    unstable continuum normalization.

    Policy
    ------
    - Applied only when external matching does *not* use a |V| tail filter
      (match_v_tol is None).
    - e_min <- max(e_min, median(V_tail) + margin), where V_tail is sampled
      over the outer tail fraction of the radial box.
    - This is an external-branch guard; full-branch settings are unchanged.
    """
    out = dict(params_ext)
    if out.get("match_v_tol", 1e-4) is not None:
        return out
    if r.size == 0:
        return out
    e_min = float(out.get("e_min", 1e-4))
    e_max = float(out.get("e_max", e_min + 1.0))
    tail_frac = float(out.get("energy_floor_tail_frac", 0.9))
    margin = float(out.get("energy_floor_margin", 1e-4))
    tail_frac = min(max(tail_frac, 0.0), 0.999999)
    r_cut = tail_frac * float(r[-1])
    mask = r >= r_cut
    if not np.any(mask):
        return out
    v_tail = float(np.median(v_ext[mask]))
    if np.isfinite(v_tail):
        e_min_new = max(e_min, v_tail + margin)
        # Keep a valid integration interval when the external tail floor rises
        # above the user-provided e_max.
        if e_max <= e_min_new:
            e_span = max(1.0, 0.1 * max(abs(e_min_new), 1.0))
            e_max = e_min_new + e_span
        out["e_min"] = e_min_new
        out["e_max"] = e_max
    return out


def _continuum_prefix_length(r_full: np.ndarray, solve_rmax: float | None) -> int:
    """
    Return how many points of the full sqrt grid should be used for the
    numerical continuum/A3 solve.

    The bound solver and the final SCF/output grid remain on the full domain.
    Only the continuum can be truncated, and only when a smaller solve radius
    is explicitly requested.
    """
    if solve_rmax is None:
        return int(r_full.size)
    solve_rmax = float(solve_rmax)
    if solve_rmax >= float(r_full[-1]):
        return int(r_full.size)
    idx_end = int(np.searchsorted(r_full, solve_rmax, side="right"))
    return max(3, min(idx_end, int(r_full.size)))


def _rebuild_continuum_on_full_grid(
    r_full: np.ndarray,
    n_eval: np.ndarray,
    *,
    idx_eval_end: int,
    params: Dict[str, Any],
    n0: float,
    mu_id: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Lift a continuum profile from the truncated A3 domain back to the full box.

    When the continuum was solved only on `r <= R_dft_max`, the outer part of
    the physical box has no direct A3 data. In the B3 workflow we seed the
    full-domain profile with `n0`, copy the computed A3 data into the inner
    prefix, then let `apply_tail_match` replace `r >= r_cut` by the analytic
    B3 continuation.

    If B3 is not active, we leave the outer region at `n0`. This keeps the
    helper total well-defined, but callers should only request a shortened
    continuum domain when B3 tail replacement is enabled.
    """
    n_eval = np.asarray(n_eval, dtype=float)
    if idx_eval_end >= r_full.size:
        full_eval = n_eval.copy()
        return full_eval, full_eval.copy(), {"applied": False, "reason": "full_grid"}

    n_out = np.full_like(r_full, float(n0), dtype=float)
    n_out[:idx_eval_end] = n_eval
    n_pre_tail = np.asarray(n_out, dtype=float).copy()

    tail_match = bool(params.get("tail_match", False))
    tail_target = str(params.get("tail_match_target", "cont")).lower()
    if not (tail_match and tail_target in ("cont", "both")):
        return n_out, n_pre_tail, {"applied": False, "reason": "tail_disabled"}

    tail_r_cut = params.get("tail_r_cut", None)
    if tail_r_cut is None:
        tail_r_cut = 0.7 * float(r_full[-1])
    idx_cut = int(np.searchsorted(r_full, float(tail_r_cut)))
    if idx_cut <= 0 or idx_cut >= idx_eval_end - 2:
        return n_out, n_pre_tail, {"applied": False, "reason": "invalid_r_cut"}

    fit_points = int(params.get("tail_fit_points", 16))
    tail_r_fit_max = params.get("tail_r_fit_max", None)
    if tail_r_fit_max is not None:
        idx_fit_max = int(np.searchsorted(r_full, float(tail_r_fit_max), side="right"))
        idx_fit_max = min(idx_fit_max, int(idx_eval_end))
        fit_points = max(3, min(fit_points, idx_fit_max - idx_cut))
    else:
        fit_points = max(3, min(fit_points, int(idx_eval_end) - idx_cut))
    if fit_points < 3:
        return n_out, n_pre_tail, {"applied": False, "reason": "insufficient_fit_points"}

    from otter.electronic.continuum.tail import apply_tail_match

    tail_n0_fixed = params.get("tail_n0_fixed", None)
    tail_mu_fixed = params.get("tail_mu_id_fixed", None)
    tail_n0 = float(n0 if tail_n0_fixed is None else tail_n0_fixed)
    tail_mu_id = float(mu_id if tail_mu_fixed is None else tail_mu_fixed)
    tail_blend = int(params.get("tail_blend_points", 0))
    tail_model = str(params.get("tail_model", "auto"))
    tail_auto_rel_improve_tol = float(params.get("tail_auto_rel_improve_tol", 0.15))
    tail_auto_signal_rel_tol = float(params.get("tail_auto_signal_rel_tol", 2.0e-5))
    tail_fallback = bool(params.get("tail_fallback_on_error", True))
    try:
        n_out, tail_meta = apply_tail_match(
            r_full,
            n_out,
            tail_n0,
            tail_mu_id,
            temperature,
            idx_cut,
            fit_points=fit_points,
            blend_points=tail_blend,
            model=tail_model,
            auto_rel_improve_tol=tail_auto_rel_improve_tol,
            auto_signal_rel_tol=tail_auto_signal_rel_tol,
        )
        tail_meta = {
            **dict(tail_meta),
            "applied": True,
            "tail_r_cut_bohr": float(r_full[idx_cut]),
            "solve_rmax_bohr": float(r_full[min(max(int(idx_eval_end) - 1, 0), r_full.size - 1)]),
        }
    except Exception as exc:
        if not tail_fallback:
            raise
        warnings.warn(
            f"B3 continuum tail match failed on truncated continuum domain at "
            f"r_cut={float(r_full[idx_cut]):.6f}; keeping raw A3 prefix and n0 tail. "
            f"reason={exc}",
            RuntimeWarning,
        )
        tail_meta = {"applied": False, "reason": str(exc)}
    return n_out, n_pre_tail, dict(tail_meta)

def _tail_shift_value(r: np.ndarray, v: np.ndarray, frac: float, mode: str = "median") -> float:
    """
    Estimate a constant tail shift from the outer frac of the radial grid.

    Parameters
    ----------
    r : ndarray
        Radial grid.
    v : ndarray
        Potential values on r.
    frac : float
        Fraction of the outer grid used for the tail estimate (0 < frac < 1).
    mode : str
        "median" (robust default), "mean" (average over tail window),
        or "last" (use the last grid point only).
    """
    frac = float(frac)
    mode = str(mode).lower()
    if mode == "last":
        return float(v[-1])
    if frac <= 0.0 or frac >= 1.0:
        return float(v[-1])
    r_cut = (1.0 - frac) * float(r[-1])
    mask = r >= r_cut
    if not np.any(mask):
        return float(v[-1])
    if mode == "mean":
        return float(np.mean(v[mask]))
    return float(np.median(v[mask]))


def _apply_outer_v_eff_decay(
    r: np.ndarray,
    v: np.ndarray,
    *,
    r_ws: float,
    enabled: bool,
    start_rws: float,
    decay_length_rws: float,
) -> np.ndarray:
    """
    Apply an experimental exponential taper to V_eff outside the WS sphere.

    Notes
    -----
    This is intentionally a diagnostic knob, not part of the validated
    Starrett-style potential definition. The taper is unity for
    ``r <= start_rws * R_ws`` and decays as

      exp(-(r - r0) / lambda),   lambda = decay_length_rws * R_ws

    outside that radius.
    """
    if not enabled:
        return v

    r_ws_use = max(float(r_ws), 1.0e-12)
    r0 = max(float(start_rws), 0.0) * r_ws_use
    lam = float(decay_length_rws) * r_ws_use
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("full_v_eff_outer_decay_length_rws must be positive.")

    out = np.asarray(v, dtype=float).copy()
    mask = np.asarray(r, dtype=float) > r0
    if np.any(mask):
        out[mask] *= np.exp(-(np.asarray(r, dtype=float)[mask] - r0) / lam)
    return out


def _window_stat_value(r: np.ndarray,
                       v: np.ndarray,
                       lo_frac: float,
                       hi_frac: float,
                       mode: str = "mean") -> float:
    """
    Estimate a representative value from a finite radial window.

    Parameters
    ----------
    r : ndarray
        Radial grid.
    v : ndarray
        Quantity sampled on r.
    lo_frac, hi_frac : float
        Window bounds as fractions of r_max.
    mode : str
        "mean" (default), "median", or "last" (value near upper bound).
    """
    if r.size == 0:
        return float("nan")

    lo = float(lo_frac)
    hi = float(hi_frac)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return float(v[-1])
    lo = min(max(lo, 0.0), 1.0)
    hi = min(max(hi, 0.0), 1.0)
    if hi < lo:
        lo, hi = hi, lo
    if hi <= lo:
        return float(v[-1])

    r_lo = lo * float(r[-1])
    r_hi = hi * float(r[-1])
    mask = (r >= r_lo) & (r <= r_hi)
    if not np.any(mask):
        return float(v[-1])

    mode = str(mode).lower().strip()
    vals = v[mask]
    if mode == "last":
        return float(vals[-1])
    if mode == "median":
        return float(np.median(vals))
    return float(np.mean(vals))


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    """
    Cubic smoothstep on [0, 1].
    """
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _close_continuum_source_to_n0(r: np.ndarray,
                                  n_cont: np.ndarray,
                                  n0: float,
                                  r_trust: float,
                                  blend_width: float) -> np.ndarray:
    """
    Build a potential-source continuum density that is unchanged in the trusted
    region and smoothly tends to n0 outside.

    Notes
    -----
    This is a source regularization for V_eff assembly only. It does not modify
    the reported continuum density arrays used for diagnostics.
    """
    n_src = np.asarray(n_cont, dtype=float).copy()
    if n_src.size == 0:
        return n_src
    r = np.asarray(r, dtype=float)
    r_trust = float(r_trust)
    w = max(float(blend_width), 0.0)
    r_trust = min(max(r_trust, float(r[0])), float(r[-1]))
    if w <= 0.0:
        n_src[r >= r_trust] = float(n0)
        return n_src

    r_end = min(r_trust + w, float(r[-1]))
    if r_end <= r_trust:
        n_src[r >= r_trust] = float(n0)
        return n_src

    mask_hi = r >= r_end
    n_src[mask_hi] = float(n0)

    mask_blend = (r >= r_trust) & (r < r_end)
    if np.any(mask_blend):
        t = (r[mask_blend] - r_trust) / max(r_end - r_trust, 1e-14)
        s = _smoothstep01(t)
        n_src[mask_blend] = (1.0 - s) * n_src[mask_blend] + s * float(n0)

    return n_src


def _enforce_source_charge_closure(r: np.ndarray,
                                   n_bound: np.ndarray,
                                   n_cont_source: np.ndarray,
                                   n0: float,
                                   g_ii: np.ndarray,
                                   Z: float,
                                   r_trust: float) -> tuple[np.ndarray, dict]:
    """
    Enforce global source charge closure by adding a uniform correction only
    in the outer (r > r_trust) region.

    This keeps the trusted inner region unchanged while forcing
      4*pi*int r^2 [n_bound + n_cont_source - n0*g_ii] dr = Z.
    """
    r = np.asarray(r, dtype=float)
    n_bound = np.asarray(n_bound, dtype=float)
    n_cont_source = np.asarray(n_cont_source, dtype=float)
    g_ii = np.asarray(g_ii, dtype=float)

    out = n_cont_source.copy()
    if r.size == 0:
        return out, {"applied": False, "reason": "empty_grid"}

    r_trust = float(r_trust)
    r_trust = min(max(r_trust, float(r[0])), float(r[-1]))
    mask = r > r_trust
    if not np.any(mask):
        return out, {"applied": False, "reason": "empty_outer_mask", "r_trust": r_trust}

    rho0 = n_bound + out - float(n0) * g_ii
    q0 = float(4.0 * np.pi * _trapz((r ** 2) * rho0, r))
    weight = np.where(mask, r ** 2, 0.0)
    denom = float(4.0 * np.pi * _trapz(weight, r))
    if not np.isfinite(denom) or denom <= 1e-14:
        return out, {"applied": False, "reason": "small_denom", "q_before": q0, "r_trust": r_trust}

    delta_n = float((float(Z) - q0) / denom)
    out[mask] = out[mask] + delta_n

    rho1 = n_bound + out - float(n0) * g_ii
    q1 = float(4.0 * np.pi * _trapz((r ** 2) * rho1, r))
    return out, {
        "applied": True,
        "r_trust": r_trust,
        "q_before": q0,
        "q_after": q1,
        "delta_n": delta_n,
    }


def _gauge_aligned_map_error(r: np.ndarray,
                             v_map: np.ndarray,
                             v_iter: np.ndarray,
                             frac: float,
                             mode: str) -> float:
    """
    Gauge-invariant SCF map residual for potentials.

    In inner-mu SCF, V_eff has a near-constant gauge freedom (mu shifts can
    compensate a constant V shift). To avoid a false non-convergence plateau,
    remove the tail-mean constant from (v_map - v_iter) before measuring error.
    """
    delta_v = np.asarray(v_map, dtype=float) - np.asarray(v_iter, dtype=float)
    delta_shift = _tail_shift_value(r, delta_v, frac, mode)
    delta_v_aligned = delta_v - float(delta_shift)
    err_num = _trapz(np.abs(delta_v_aligned), r)
    err_den = _trapz(np.abs(v_iter), r)
    return float(err_num / max(err_den, 1e-12))

def _select_continuum_model(name: str):
    name = name.lower().strip()
    if name in ("scattering", "quantum"):
        return QuantumContinuumScattering()
    if name == "ideal":
        return IdealContinuum()
    if name == "hybrid":
        return QuantumContinuumHybrid()
    raise ValueError(f"Unknown continuum_model: {name}")

def _scf_fixed_mu(config: KSDTFConfig,
                  mu: float,
                  v_full_init: np.ndarray | None = None,
                  v_ext_init: np.ndarray | None = None,
                  max_iter: int | None = None) -> Dict[str, Any]:
    """
    Fixed-μ SCF loop for the AA full/external systems.

    Inputs
    ------
    config : KSDTFConfig
        Global AA/IS configuration and numerical settings.
    mu : float
        Fixed chemical potential (Ha).
    v_full_init, v_ext_init : ndarray or None
        Optional initial guesses for V_eff (full/external) on the continuum grid.
    max_iter : int or None
        Override maximum SCF iterations.

    Outputs
    -------
    dict
        SCF results including densities, potentials, μ, and history.

    Algorithm (major steps)
    -----------------------
    1) Build grids and the ion-correlation profile `g_ii(r)`.
    2) Initialize n0 (ideal/tail/fixed).
    3) Per SCF step: bound solve → continuum density → n_full assembly
       → V_eff construction → mixing → convergence checks.
    """
    r_cont, step_cont, kind_cont, r_bound, step_bound, kind_bound = _build_grid_pair(config)
    r_ws = _resolve_r_ws(config.n_i, config.r_ws)
    if config.n_i is None:
        n_i_val = 3.0 / (4.0 * np.pi * r_ws ** 3)
    else:
        n_i_val = float(config.n_i)
    compute_external = bool(config.compute_external)

    if v_full_init is not None and np.asarray(v_full_init).shape != r_cont.shape:
        raise ValueError("v_full_init must match the continuum grid shape.")
    if v_ext_init is not None and np.asarray(v_ext_init).shape != r_cont.shape:
        raise ValueError("v_ext_init must match the continuum grid shape.")

    # Ion-correlation profile:
    # - default IS path: step model
    # - SC path: tabulated OZ/HNC profile interpolated to the AA grid
    g_ii = IonSphereStepModel(r_ws=r_ws).g_ii(r_cont)

    # n0 initialization (ideal/tail/window/fixed).
    # Tail mode semantics:
    # 1) Bootstrap from ideal n0 before any n_full exists.
    # 2) After n_full is assembled each SCF iteration, refresh n0 from the
    #    outer tail estimator (direct or mixed update).
    n0_mode = str(config.n0_mode).lower().strip()
    if n0_mode == "ideal":
        n0 = ideal_unbound_density(mu, config.temperature)
    elif n0_mode == "fixed":
        if config.n0_fixed is None:
            raise ValueError("n0_mode='fixed' requires n0_fixed to be set.")
        n0 = float(config.n0_fixed)
    elif n0_mode == "tail":
        # Initialize from ideal; update from tail after n_full is available.
        n0 = ideal_unbound_density(mu, config.temperature)
    elif n0_mode == "window":
        # Initialize from ideal; update from interior window after n_full is available.
        n0 = ideal_unbound_density(mu, config.temperature)
    else:
        raise ValueError("n0_mode must be 'ideal', 'fixed', 'tail', or 'window'.")

    # Initial potential guesses.
    v_full = (-float(config.Z) / r_cont) if v_full_init is None else v_full_init.copy()
    v_ext = np.zeros_like(r_cont) if v_ext_init is None else v_ext_init.copy()

    # Continuum model selection.
    continuum = _select_continuum_model(config.continuum_model)

    bound_cut_mode = str(config.bound_energy_cut_mode).lower().strip()
    if bound_cut_mode == "auto":
        bound_cut_mode = "v_ws" if not compute_external else "zero"
    if bound_cut_mode not in ("zero", "v_ws", "v_rmax", "v_frac", "fixed"):
        raise ValueError(
            "bound_energy_cut_mode must be 'zero', 'v_ws', 'v_rmax', 'v_frac', 'fixed', or 'auto'."
        )
    # Optional tail shift (explicit only): do not couple to bound_cut_mode.
    shift_tail = bool(config.shift_v_eff_tail)
    if config.shift_v_eff_tail and config.verbose:
        print(
            "  [SCF] Enabling V_eff tail shift to align zero-energy cutoff "
            f"(mode={config.v_tail_mode}, frac={config.v_tail_fraction})."
        )
    outer_decay = bool(config.full_v_eff_outer_decay)
    if outer_decay and config.verbose:
        print(
            "  [SCF] Enabling experimental full V_eff outer decay "
            f"(start={float(config.full_v_eff_outer_decay_start_rws):.3f}*R_ws, "
            f"lambda={float(config.full_v_eff_outer_decay_length_rws):.3f}*R_ws)."
        )
    bound_ws_mode = None

    history = []
    scf_converged = False
    n_bound = np.zeros_like(r_cont)
    n_cont = np.zeros_like(r_cont)
    n_cont_pre_tail = np.zeros_like(r_cont)
    n_full_pre_tail = np.zeros_like(r_cont)
    n_cont_dft_raw = np.full_like(r_cont, np.nan)
    n_cont_source = np.zeros_like(r_cont)
    n_full = np.zeros_like(r_cont)
    n_full_source = np.zeros_like(r_cont)
    source_closure_meta = {"applied": False}
    n_cont_tail_meta: dict[str, Any] = {"applied": False}
    n_ext = np.zeros_like(r_cont)
    n_ion = np.zeros_like(r_cont)
    n_pa = np.zeros_like(r_cont)
    charge_ws = 0.0
    zbar = 0.0
    neutrality_mode = str(config.neutrality_mode).lower().strip()
    if neutrality_mode not in ("auto", "ws", "pa"):
        raise ValueError("neutrality_mode must be 'auto', 'ws', or 'pa'.")
    if neutrality_mode == "auto":
        neutrality_mode = "pa" if compute_external else "ws"

    n_full_prev = None
    v_full_prev = None
    n_ext_prev = None
    zbar_prev = None
    mixing_scheme = str(config.mixing_scheme).lower().strip()
    if mixing_scheme not in ("linear", "eyert"):
        raise ValueError("mixing_scheme must be 'linear' or 'eyert'.")
    mixing_m = max(int(config.mixing_m), 1)
    mixing_w0 = float(config.mixing_w0)
    mix = float(config.mix)
    x_prev = None
    f_prev = None
    dx_hist = deque(maxlen=mixing_m)
    df_hist = deque(maxlen=mixing_m)
    scf_snapshots = None
    if config.store_scf_snapshots_last is not None and int(config.store_scf_snapshots_last) > 0:
        scf_snapshots = deque(maxlen=int(config.store_scf_snapshots_last))

    scf_iters = max_iter if max_iter is not None else config.max_iter

    if shift_tail:
        v_full = v_full - _tail_shift_value(
            r_cont,
            v_full,
            config.v_tail_fraction,
            config.v_tail_mode,
        )
        if compute_external:
            v_ext = v_ext - _tail_shift_value(
                r_cont,
                v_ext,
                config.v_tail_fraction,
                config.v_tail_mode,
            )
    if outer_decay:
        v_full = _apply_outer_v_eff_decay(
            r_cont,
            v_full,
            r_ws=r_ws,
            enabled=True,
            start_rws=float(config.full_v_eff_outer_decay_start_rws),
            decay_length_rws=float(config.full_v_eff_outer_decay_length_rws),
        )

    if config.verbose:
        print(f"  [SCF] mu={mu:.6f} Ha, n0={n0:.6e}")
        if compute_external:
            print(
                "  iter | dn_rel | dv_rel | Q_full | Q_bound | Q_cont | "
                "Zbar | charge_rel | dZbar"
            )
        else:
            print(
                "  iter | dn_rel | dv_rel | Q_full | Q_bound | Q_cont | "
                "Zbar | charge_rel"
            )

    # Optional per-iteration wall-clock diagnostics.
    # The timers are stage-level (not line-level) and intended to identify
    # dominant SCF costs quickly when continuum settings are changed.
    # Optional per-iteration wall-clock diagnostics for inner-mu SCF.
    # Keys:
    # - basis_build: continuum basis precompute for current V_eff.
    # - bound_solve: sparse bound-state eigen solve.
    # - mu_solve: neutrality root solve using the frozen basis.
    # - assembly/potential/mix/metrics: post-mu SCF map/update stages.
    perf_on = bool(config.perf_diag)
    perf_every = max(int(config.perf_print_every), 1)
    perf_show_stage = bool(getattr(config, "perf_show_stage", True))

    def _fmt_perf_int(value: Any) -> str:
        """
        Safe formatter for optional integer-like perf counters.

        Adaptive metadata can be missing in some execution paths, which leaves
        NaN placeholders in `perf`. Converting NaN with int(...) raises, so we
        print "nan" in that case.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "nan"
        if not np.isfinite(v):
            return "nan"
        return str(int(v))

    for it in range(int(scf_iters)):
        perf = {}
        t_iter = time.perf_counter() if perf_on else 0.0
        t_stage = t_iter
        basis_meta = None

        # (1) Choose bound/continuum partition energy.
        if bound_cut_mode == "fixed":
            energy_cut = float(config.bound_energy_cut) if config.bound_energy_cut is not None else 0.0
        elif bound_cut_mode == "v_ws":
            energy_cut = float(np.interp(r_ws, r_cont, v_full))
        elif bound_cut_mode == "v_rmax":
            energy_cut = float(v_full[-1])
        elif bound_cut_mode == "v_frac":
            r_frac = float(config.bound_energy_cut) if config.bound_energy_cut is not None else 0.7
            r_frac = min(max(r_frac, 0.0), 1.0)
            energy_cut = float(np.interp(r_frac * float(r_cont[-1]), r_cont, v_full))
        else:
            energy_cut = 0.0

        # (2) Map V_eff to the bound grid if grids differ.
        if r_bound.shape == r_cont.shape and np.allclose(r_bound, r_cont):
            v_full_bound = v_full
        else:
            v_full_bound = interp_to_grid(r_cont, v_full, r_bound)

        # (3) Prepare continuum parameters and caches.
        cont_params_full, cont_params_ext = _split_continuum_params_for_full_ext(config.continuum_params)
        cont_params = cont_params_full
        cont_params["v_eff"] = v_full
        cont_params_ext["v_eff"] = v_ext
        cont_e_max_iter = _resolve_iteration_continuum_e_max(
            cont_params,
            mu_ref=float(mu),
            temperature=float(config.temperature),
        )
        cont_params["e_max"] = float(cont_e_max_iter)
        if compute_external:
            # Keep external scattering well-posed when no |V| tail filter is used.
            cont_params_ext = _apply_external_energy_floor(cont_params_ext, r_cont, v_ext)
            cont_params_ext["v_eff"] = v_ext
        ion_gamma_mode = str(config.ion_gamma_mode).lower().strip()
        if ion_gamma_mode not in ("fixed", "scattering"):
            raise ValueError("ion_gamma_mode must be 'fixed' or 'scattering'.")
        ion_gamma_scale = float(getattr(config, "ion_gamma_scale", 1.0))
        if ion_gamma_scale <= 0.0:
            raise ValueError("ion_gamma_scale must be positive.")
        energy_cache_full = {} if ion_gamma_mode == "scattering" else None
        energy_cache_ext = {} if (ion_gamma_mode == "scattering" and compute_external) else None
        if energy_cache_full is not None:
            cont_params["energy_cache"] = energy_cache_full
        if energy_cache_ext is not None:
            cont_params_ext["energy_cache"] = energy_cache_ext

        # Optional continuum-only numerical cutoff:
        # - bound states still use the full box
        # - final densities/potentials still live on the full box
        # - only the expensive continuum/A3 solve can be truncated
        cont_solve_end = _continuum_prefix_length(r_cont, cont_params.get("solve_rmax", None))
        r_cont_solve = r_cont[:cont_solve_end]
        v_full_solve = v_full[:cont_solve_end]
        cont_solve_end_ext = _continuum_prefix_length(r_cont, cont_params_ext.get("solve_rmax", None))
        r_cont_solve_ext = r_cont[:cont_solve_end_ext]
        v_ext_solve = v_ext[:cont_solve_end_ext]
        cont_params["l_max"] = _resolve_iteration_continuum_l_max(
            cont_params,
            e_max=float(cont_e_max_iter),
            r_eval_max=float(r_cont_solve[-1]),
        )
        if compute_external:
            cont_params_ext["l_max"] = _resolve_iteration_continuum_l_max(
                cont_params_ext,
                e_max=float(cont_params_ext.get("e_max", cont_e_max_iter)),
                r_eval_max=float(r_cont_solve_ext[-1]),
            )

        # (4) Precompute scattering basis once per SCF step (no FD occupancy).
        # This is the key acceleration for inner-mu: V_eff is fixed within the
        # current SCF iteration, so the expensive scattering solve can be reused
        # while only FD occupations change during mu root finding.
        cont_basis = None
        e_grid = None
        e_weights = None
        if isinstance(continuum, QuantumContinuumScattering):
            energy_mode = str(cont_params.get("energy_mode", "linear")).lower()
            if energy_mode == "linear":
                e_min = float(cont_params.get("e_min", 1e-6))
                e_max = float(cont_params.get("e_max", max(mu + 10.0 * config.temperature, 5.0 * config.temperature)))
                e_min = max(e_min, 1e-12)
                n_e = int(cont_params.get("n_e", 200))
                e_grid = np.linspace(e_min, e_max, n_e)
                match_slice = cont_params.get("match_slice", None)
                match_r_cut = cont_params.get("match_r_cut", None)
                match_width = cont_params.get("match_width", None)
                if match_width is not None:
                    match_width = float(match_width)
                if match_slice is None and match_r_cut is not None:
                    idx_cut = int(np.searchsorted(r_cont_solve, float(match_r_cut)))
                    if 0 < idx_cut < r_cont_solve.size - 1:
                        if match_width is not None:
                            idx_end = int(np.searchsorted(r_cont_solve, float(match_r_cut + match_width)))
                            match_slice = (idx_cut, min(max(idx_end, idx_cut + 1), r_cont_solve.size))
                        else:
                            match_slice = (idx_cut, r_cont_solve.size)
                cont_basis = continuum_density_scattering_basis(
                    v_full,
                    r_cont,
                    e_grid,
                    int(cont_params.get("l_max", 6)),
                    kind_cont,
                    step_cont,
                    l_pad=int(cont_params.get("l_pad", 2)),
                    l_cap_strategy=str(cont_params.get("l_cap_strategy", "match")),
                    match_fraction=float(cont_params.get("match_fraction", 0.2)),
                    match_slice=match_slice,
                    match_r_cut=match_r_cut,
                    match_fraction_mode=str(cont_params.get("match_fraction_mode", "r")),
                    match_kr_min=cont_params.get("match_kr_min", 4.0),
                    match_v_tol=cont_params.get("match_v_tol", 1e-4),
                    match_min_points=int(cont_params.get("match_min_points", 12)),
                    match_asymptotic=str(cont_params.get("match_asymptotic", "auto")),
                    match_coulomb_tol=float(cont_params.get("match_coulomb_tol", 0.1)),
                    match_allow_shift=bool(cont_params.get("match_allow_shift", True)),
                    match_fallback=str(cont_params.get("match_fallback", "free")),
                    prop_rescale_limit=cont_params.get("prop_rescale_limit", 1e6),
                    energy_cache=energy_cache_full,
                    n_jobs=cont_params.get("n_jobs", None),
                )
        if perf_on:
            # fixed-mu path: basis build + continuum integration are the heavy parts.
            perf["prep_basis"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()

        # (5) Continuum density from scattering/ideal model (Eq. A3).
        n_cont = continuum.density(r_cont, mu, config.temperature, params=cont_params)
        if ion_gamma_mode == "scattering":
            ion_gamma = ion_gamma_scale * gamma_from_phase_shift_cache(
                energy_cache_full,
                mu,
                config.temperature,
                n_i_val,
                n0,
                n0_floor=config.mu_n0_floor,
            )
        else:
            ion_gamma = float(config.ion_bound_gamma)
        if perf_on:
            perf["continuum"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()

        # (6) Bound states via sparse Numerov (Eq. A2).
        vals, vecs = solve_bound_states_sparse_numerov(
            v_full_bound,
            r_bound,
            step_bound,
            np.asarray(config.l_list),
            n_states=(
                np.asarray(config.n_states_by_l, dtype=int)
                if config.n_states_by_l is not None
                else int(config.n_states)
            ),
            boundary=config.boundary,
            n_jobs=config.n_jobs,
        )
        n_bound_bound_raw = _bound_density(
            r_bound,
            vals,
            vecs,
            config.l_list,
            mu,
            config.temperature,
            energy_cut=energy_cut,
            occ_mode=config.bound_occ_mode,
            gamma=ion_gamma,
            r_ws=r_ws,
            ws_weight_min=0.0,
        )
        # (7) Interpolate bound density to continuum grid if needed.
        if r_bound.shape == r_cont.shape and np.allclose(r_bound, r_cont):
            n_bound = n_bound_bound_raw
        else:
            n_bound = interp_to_grid(r_bound, n_bound_bound_raw, r_cont)

        # (8) Ion density for Zbar (Starrett cutoff + M(e)).
        cut_width = float(config.ion_cut_width) * float(r_ws)
        ion_cut_mode = str(config.ion_cut_mode).lower().strip()
        ion_cut_c = float(config.ion_cut_c)
        ion_cut = _ion_cutoff(r_bound, r_ws, cut_width, mode=ion_cut_mode, c=ion_cut_c)
        n_ion_bound = _ion_density(
            r_bound,
            vals,
            vecs,
            config.l_list,
            mu,
            config.temperature,
            energy_cut=energy_cut,
            gamma=ion_gamma,
            cutoff=ion_cut,
            r_ws=r_ws,
            ws_weight_min=config.ion_ws_weight_min,
        )
        if r_bound.shape == r_cont.shape and np.allclose(r_bound, r_cont):
            n_ion = n_ion_bound
        else:
            n_ion = interp_to_grid(r_bound, n_ion_bound, r_cont)
        # (9) Full density assembly.
        n_full = n_bound + n_cont
        if perf_on:
            # Includes bound solve + ion-density assembly on this iteration.
            perf["bound_ion"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()

        # (10) Update n0 from tail/window if requested.
        # Use continuum density n_cont (not n_full) to avoid bound-tail leakage
        # when weakly bound states extend outside the WS sphere.
        if n0_mode == "tail":
            n0_tail = _tail_shift_value(
                r_cont,
                n_cont,
                config.n0_tail_fraction,
                config.n0_tail_mode,
            )
            if config.n0_tail_direct:
                n0 = float(n0_tail)
            else:
                mix_n0 = float(config.n0_tail_mix)
                n0 = mix_n0 * n0_tail + (1.0 - mix_n0) * n0
        elif n0_mode == "window":
            n0_win = _window_stat_value(
                r_cont,
                n_cont,
                config.n0_window_lo_frac,
                config.n0_window_hi_frac,
                config.n0_window_mode,
            )
            if config.n0_window_direct:
                n0 = float(n0_win)
            else:
                mix_n0 = float(config.n0_window_mix)
                n0 = mix_n0 * n0_win + (1.0 - mix_n0) * n0

        if v_full_prev is None:
            v_full_prev = v_full.copy()

        # (11) External system (if enabled).
        if compute_external:
            n_ext = continuum.density(
                r_cont,
                mu,
                config.temperature,
                params={
                    **cont_params_ext,
                },
            )
        else:
            n_ext = np.zeros_like(r_cont)

        # (12) Tail matching for n_full/n_ext if enabled.
        tail_match = bool(cont_params.get("tail_match", False))
        tail_match_target = str(cont_params.get("tail_match_target", "cont")).lower()
        if tail_match and tail_match_target in ("full", "both"):
            from otter.electronic.continuum.tail import apply_tail_match

            tail_n0_fixed = cont_params.get("tail_n0_fixed", None)
            tail_mu_fixed = cont_params.get("tail_mu_id_fixed", None)
            tail_n0 = float(n0 if tail_n0_fixed is None else tail_n0_fixed)
            tail_mu_id = float(mu if tail_mu_fixed is None else tail_mu_fixed)
            fit_points = int(cont_params.get("tail_fit_points", 16))
            blend_points = int(cont_params.get("tail_blend_points", 0))
            tail_model = str(cont_params.get("tail_model", "auto"))
            tail_auto_rel_improve_tol = float(cont_params.get("tail_auto_rel_improve_tol", 0.15))
            tail_auto_signal_rel_tol = float(cont_params.get("tail_auto_signal_rel_tol", 2.0e-5))
            tail_fallback_on_error = bool(cont_params.get("tail_fallback_on_error", True))
            tail_r_cut = cont_params.get("tail_r_cut", None)
            if tail_r_cut is None:
                tail_r_cut = float(cont_params.get("tail_auto_r_fraction", 0.7)) * float(r_cont[-1])
            tail_r_cut = float(tail_r_cut)
            idx_cut = int(np.searchsorted(r_cont, tail_r_cut))
            if 0 < idx_cut < r_cont.size - 1:
                try:
                    n_full, _ = apply_tail_match(
                        r_cont,
                        n_full,
                        tail_n0,
                        tail_mu_id,
                        config.temperature,
                        idx_cut,
                        fit_points=fit_points,
                        blend_points=blend_points,
                        model=tail_model,
                        auto_rel_improve_tol=tail_auto_rel_improve_tol,
                        auto_signal_rel_tol=tail_auto_signal_rel_tol,
                    )
                    if compute_external:
                        n_ext, _ = apply_tail_match(
                            r_cont,
                            n_ext,
                            tail_n0,
                            tail_mu_id,
                            config.temperature,
                            idx_cut,
                            fit_points=fit_points,
                            blend_points=blend_points,
                            model=tail_model,
                            auto_rel_improve_tol=tail_auto_rel_improve_tol,
                            auto_signal_rel_tol=tail_auto_signal_rel_tol,
                        )
                except Exception as exc:
                    if not tail_fallback_on_error:
                        raise
                    warnings.warn(
                        f"B3 tail match failed in fixed-mu SCF at r_cut={tail_r_cut:.6f}; "
                        f"keeping original density. reason={exc}",
                        RuntimeWarning,
                    )
                # n_cont remains the A3 continuum density (no back-substitution).

        # (14) External-tail closure (optional).
        #
        # For the external system (Eq. 7), finite-box continuum matching can be
        # unreliable in the far tail. When enabled, use the same trusted-region
        # policy as full-source closure: keep n_ext unchanged for r < r_trust and
        # smoothly blend to n0 outside. This modifies n_ext itself so downstream
        # diagnostics (n_pa, n_scr, Q_pa) use the stabilized external density.
        use_ext_source_closure = bool(cont_params_ext.get("source_closure", False))
        if str(cont_params_ext.get("tail_mode", "off")).strip().lower() == "in_scf":
            use_ext_source_closure = use_ext_source_closure and bool(
                cont_params_ext.get("source_closure_when_b3", False)
            )
        if compute_external and use_ext_source_closure:
            r_trust_ext = cont_params_ext.get("source_r_trust", None)
            if r_trust_ext is None:
                r_trust_frac_ext = float(cont_params_ext.get("source_r_trust_frac", 0.7))
                r_trust_ext = r_trust_frac_ext * float(r_cont[-1])
            blend_w_ext = cont_params_ext.get("source_blend_width", None)
            if blend_w_ext is None:
                blend_frac_ext = float(cont_params_ext.get("source_blend_frac", 0.05))
                blend_w_ext = blend_frac_ext * float(r_cont[-1])
            n_ext = _close_continuum_source_to_n0(
                r_cont,
                n_ext,
                float(n0),
                float(r_trust_ext),
                float(blend_w_ext),
            )

        # (15) Derived densities (PAMD definitions).
        n_pa = n_full - n_ext
        n_scr_iter = n_pa - n_ion
        # (16) Charge diagnostics / neutrality checks.
        charge_ws = _electron_count(r_cont, n_full, r_ws)
        charge_bound = _electron_count(r_cont, n_bound, r_ws)
        charge_ion = _electron_count(r_cont, n_ion, r_ws)
        charge_cont = _electron_count(r_cont, n_cont, r_ws)
        charge_ext = _electron_count(r_cont, n_ext, r_ws)
        charge_pa = _electron_count(r_cont, n_pa, r_ws)
        charge_pa_full = _electron_count_full(r_cont, n_pa)
        charge_scr = _electron_count(r_cont, n_scr_iter, r_ws)
        # Keep a meaningful Zbar in both modes:
        # - full+external: Zbar = integral of n_scr inside WS
        # - full only:     Zbar = Z - integral of n_ion inside WS
        zbar = charge_scr if compute_external else (float(config.Z) - float(charge_ion))
        charge_neutral = charge_pa_full if neutrality_mode == "pa" else charge_ws
        charge_rel = abs(charge_neutral - config.Z) / max(float(config.Z), 1e-12)
        if not compute_external:
            dzbar_rel = np.nan
        elif zbar_prev is None:
            dzbar_rel = np.nan
        else:
            denom = max(abs(zbar), 1e-12)
            dzbar_rel = abs(zbar - zbar_prev) / denom
        if perf_on:
            perf["assembly_diag"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()

        # Tail diagnostics removed from SCF history/printouts.

        # (17) Build effective potentials (Eq. 4 / Eq. 7), optionally screened.
        # Optional source-closure regularization for pure-A3 tails:
        # keep n_cont in trusted region and smoothly blend to n0 outside.
        n_cont_source = n_cont
        n_full_source = n_full
        use_full_source_closure = bool(cont_params.get("source_closure", False))
        if str(cont_params.get("tail_mode", "off")).strip().lower() == "in_scf":
            use_full_source_closure = use_full_source_closure and bool(
                cont_params.get("source_closure_when_b3", True)
            )
        if use_full_source_closure:
            r_trust = cont_params.get("source_r_trust", None)
            if r_trust is None:
                r_trust_frac = float(cont_params.get("source_r_trust_frac", 0.7))
                r_trust = r_trust_frac * float(r_cont[-1])
            blend_w = cont_params.get("source_blend_width", None)
            if blend_w is None:
                blend_frac = float(cont_params.get("source_blend_frac", 0.05))
                blend_w = blend_frac * float(r_cont[-1])
            n_cont_source = _close_continuum_source_to_n0(
                r_cont,
                n_cont,
                float(n0),
                float(r_trust),
                float(blend_w),
            )
            if bool(cont_params.get("source_charge_closure", False)):
                n_cont_source, source_closure_meta = _enforce_source_charge_closure(
                    r_cont,
                    n_bound,
                    n_cont_source,
                    float(n0),
                    g_ii,
                    float(config.Z),
                    float(r_trust),
                )
            else:
                source_closure_meta = {
                    "applied": False,
                    "reason": "disabled",
                    "r_trust": float(r_trust),
                }
            n_full_source = n_bound + n_cont_source

        kappa_eff = config.ph_kappa
        if config.ph_kappa_iters is not None and it >= int(config.ph_kappa_iters):
            kappa_eff = 0.0
        v_full_new = effective_potential_full(
            r_cont,
            n_full_source,
            n0,
            g_ii,
            config.Z,
            xc_model=config.xc_model,
            kappa=kappa_eff,
        )
        if shift_tail:
            v_shift_full = _tail_shift_value(
                r_cont,
                v_full_new,
                config.v_tail_fraction,
                config.v_tail_mode,
            )
            v_full_new = v_full_new - v_shift_full
        if outer_decay:
            v_full_new = _apply_outer_v_eff_decay(
                r_cont,
                v_full_new,
                r_ws=r_ws,
                enabled=True,
                start_rws=float(config.full_v_eff_outer_decay_start_rws),
                decay_length_rws=float(config.full_v_eff_outer_decay_length_rws),
            )
        if compute_external:
            v_ext_new = effective_potential_external(
                r_cont,
                n_ext,
                n0,
                g_ii,
                xc_model=config.xc_model,
                kappa=kappa_eff,
            )
            if shift_tail:
                v_shift_ext = _tail_shift_value(
                    r_cont,
                    v_ext_new,
                    config.v_tail_fraction,
                    config.v_tail_mode,
                )
                v_ext_new = v_ext_new - v_shift_ext
        else:
            v_ext_new = v_ext
        if perf_on:
            perf["potential"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()

        # (17) SCF mixing (linear or Eyert) on V_eff.
        if mixing_scheme == "linear":
            v_full = mix * v_full_new + (1.0 - mix) * v_full
        else:
            # Eyert SCF (Eq. 59-63): work with x = V_eff * r / Z
            x_in = v_full * r_cont / float(config.Z)
            x_out = v_full_new * r_cont / float(config.Z)
            f_now = x_out - x_in

            if x_prev is not None and f_prev is not None:
                dx_hist.append(x_in - x_prev)
                df_hist.append(f_now - f_prev)

            if dx_hist:
                hist_len = len(dx_hist)
                a_mat = np.zeros((hist_len, hist_len), dtype=float)
                b_vec = np.zeros(hist_len, dtype=float)
                for i in range(hist_len):
                    for j in range(hist_len):
                        prod = dx_hist[i] * 0.0
                        prod = df_hist[i] * df_hist[j]
                        a_mat[i, j] = _trapz(prod, r_cont)
                        if i == j:
                            a_mat[i, j] += mixing_w0 ** 2
                    b_vec[i] = _trapz(df_hist[i] * f_now, r_cont)
                try:
                    w_vec = np.linalg.solve(a_mat, b_vec)
                except np.linalg.LinAlgError:
                    w_vec = None
                if w_vec is not None:
                    corr = np.zeros_like(x_in)
                    for i in range(hist_len):
                        corr = corr + w_vec[i] * (dx_hist[i] + mix * df_hist[i])
                    x_next = x_in + mix * f_now - corr
                else:
                    x_next = x_in + mix * f_now
            else:
                x_next = x_in + mix * f_now

            v_full = np.where(r_cont > 0.0, x_next * float(config.Z) / r_cont, v_full_new)
            x_prev = x_in
            f_prev = f_now
        if shift_tail:
            v_full = v_full - _tail_shift_value(
                r_cont,
                v_full,
                config.v_tail_fraction,
                config.v_tail_mode,
            )
        if compute_external:
            v_ext = mix * v_ext_new + (1.0 - mix) * v_ext
            if shift_tail:
                v_ext = v_ext - _tail_shift_value(
                    r_cont,
                    v_ext,
                    config.v_tail_fraction,
                    config.v_tail_mode,
                )
        if perf_on:
            perf["mix"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()

        if scf_snapshots is not None:
            scf_snapshots.append({
                "iter": int(it),
                "n_full": n_full.copy(),
                "v_full": v_full.copy(),
            })

        # (18) Convergence metrics (dn_rel, dv_rel).
        dn_rel = np.nan
        dv_rel = np.nan
        if n_full_prev is not None:
            dn_num = _trapz(np.abs(n_full - n_full_prev), r_cont)
            dn_den = _trapz(np.abs(n_full_prev), r_cont)
            dn_rel = float(dn_num / max(dn_den, 1e-12))
        if v_full_prev is not None:
            dv_num = _trapz(np.abs(v_full - v_full_prev), r_cont)
            dv_den = _trapz(np.abs(v_full_prev), r_cont)
            dv_rel = float(dv_num / max(dv_den, 1e-12))
        # Gauge-aligned map residual (ignore constant potential offset mode).
        err_full = _gauge_aligned_map_error(
            r_cont,
            v_full_new,
            v_full,
            config.v_tail_fraction,
            config.v_tail_mode,
        )
        err_ext = 0.0
        if compute_external:
            err_ext = _gauge_aligned_map_error(
                r_cont,
                v_ext_new,
                v_ext,
                config.v_tail_fraction,
                config.v_tail_mode,
            )
        err = max(err_full, err_ext)
        if perf_on:
            perf["metrics"] = time.perf_counter() - t_stage
            perf["total"] = time.perf_counter() - t_iter

        history.append({
            "iter": it,
            "err": float(err),
            "charge_ws": float(charge_ws),
            "charge_neutral": float(charge_neutral),
            "neutrality_mode": neutrality_mode,
            "charge_rel": float(charge_rel),
            "zbar": float(zbar),
            "dzbar_rel": float(dzbar_rel) if np.isfinite(dzbar_rel) else np.nan,
            "ion_gamma": float(ion_gamma),
            "dn_full": np.nan,
            "dn_rel": dn_rel,
            "dv_rel": dv_rel,
            "charge_bound": float(charge_bound),
            "charge_ion": float(charge_ion),
            "charge_cont": float(charge_cont),
            "charge_ext": float(charge_ext),
            "charge_pa": float(charge_pa),
            "charge_pa_full": float(charge_pa_full),
            "charge_scr": float(charge_scr),
            "perf": perf if perf_on else None,
        })
        if config.verbose and (it % max(int(config.print_every), 1) == 0):
            if compute_external:
                print(
                    f"  {it:3d} | {dn_rel:7.3e} | {dv_rel:7.3e} | "
                    f"{charge_ws:7.3f} | {charge_bound:7.3f} | {charge_cont:7.3f} | "
                    f"{zbar:7.3f} | {charge_rel:9.3e} | {dzbar_rel:7.3e}"
                )
            else:
                print(
                    f"  {it:3d} | {dn_rel:7.3e} | {dv_rel:7.3e} | "
                    f"{charge_ws:7.3f} | {charge_bound:7.3f} | {charge_cont:7.3f} | "
                    f"{zbar:7.3f} | {charge_rel:9.3e}"
                )
            print(f"      cont_e_max={float(cont_e_max_iter):.6f} Ha")
            if perf_on and perf_show_stage and (it % perf_every == 0):
                print(
                    "      perf[s]: "
                    f"prep_basis={perf.get('prep_basis', 0.0):.3f}, "
                    f"continuum={perf.get('continuum', 0.0):.3f}, "
                    f"bound_ion={perf.get('bound_ion', 0.0):.3f}, "
                    f"assembly={perf.get('assembly_diag', 0.0):.3f}, "
                    f"potential={perf.get('potential', 0.0):.3f}, "
                    f"mix={perf.get('mix', 0.0):.3f}, "
                    f"metrics={perf.get('metrics', 0.0):.3f}, "
                    f"total={perf.get('total', 0.0):.3f}"
                )
        dn_ok = False
        if config.dn_tol is not None:
            dn_ok = np.isfinite(dn_rel) and dn_rel < config.dn_tol
        dv_ok = False
        if config.dv_tol is not None:
            dv_ok = np.isfinite(dv_rel) and dv_rel < config.dv_tol
        # Keep at least one unscreened update after the PH (Piron) stage.
        in_ph_stage = (
            config.ph_kappa_iters is not None
            and it < int(config.ph_kappa_iters)
        )
        # Also require the self-consistency map residual to be small.
        # (config.tol was previously unused.)
        err_ok = np.isfinite(err) and (err < float(config.tol))
        if (not in_ph_stage) and dn_ok and dv_ok and err_ok:
            scf_converged = True
            break

        n_full_prev = n_full.copy()
        v_full_prev = v_full.copy()
        n_ext_prev = n_ext.copy()
        zbar_prev = zbar

    # Final refresh on the returned mixed potential.
    #
    # The SCF loop evaluates densities on the pre-mix potential and then mixes
    # V_eff. Without one last refresh, the returned ``v_full`` and
    # ``n_bound/n_ion/n_full`` correspond to slightly different fixed-point
    # iterates, which is especially visible in post-SCF WS charge diagnostics.
    final_state = _run_iteration_attempt(
        it=int(len(history)),
        mu_guess_val=float(mu),
        n0_current=float(n0),
        v_full_current=v_full,
        v_ext_current=v_ext,
        zbar_prev_current=zbar_prev,
        stage_e_max_floor=float(cont_e_max_stage_floor),
    )
    mu = float(final_state["mu"])
    n0 = float(final_state["n0"])
    n_full = np.asarray(final_state["n_full"], dtype=float)
    n_full_pre_tail = np.asarray(final_state["n_full_pre_tail"], dtype=float)
    n_bound = np.asarray(final_state["n_bound"], dtype=float)
    n_cont = np.asarray(final_state["n_cont"], dtype=float)
    n_cont_pre_tail = np.asarray(final_state["n_cont_pre_tail"], dtype=float)
    n_cont_dft_raw = np.asarray(final_state["n_cont_dft_raw"], dtype=float)
    n_cont_tail_meta = dict(final_state["n_cont_tail_meta"])
    n_ext = np.asarray(final_state["n_ext"], dtype=float)
    n_ion = np.asarray(final_state["n_ion"], dtype=float)
    n_pa = np.asarray(final_state["n_pa"], dtype=float)
    charge_ws = float(final_state["charge_ws"])
    charge_bound = float(final_state["charge_bound"])
    charge_ion = float(final_state["charge_ion"])
    charge_cont = float(final_state["charge_cont"])
    charge_ext = float(final_state["charge_ext"])
    charge_pa = float(final_state["charge_pa"])
    charge_pa_full = float(final_state["charge_pa_full"])
    charge_scr = float(final_state["charge_scr"])
    charge_neutral = float(final_state["charge_neutral"])
    charge_rel = float(final_state["charge_rel"])
    zbar = float(final_state["zbar"])
    dzbar_rel = float(final_state["dzbar_rel"])
    cont_e_max_iter = float(final_state["cont_e_max_iter"])
    cont_params = dict(final_state["cont_params"])
    cont_params_ext = dict(final_state["cont_params_ext"])
    energy_cache_full_last = final_state["energy_cache_full"]

    n_scr = n_pa - n_ion

    cont_phase_energy = None
    cont_phase_shift = None
    if energy_cache_full is not None and len(energy_cache_full) >= 2:
        e_sorted = np.array(sorted(float(e) for e in energy_cache_full.keys()), dtype=float)
        delta_rows = [np.asarray(energy_cache_full[float(e)][1], dtype=float) for e in e_sorted]
        if delta_rows:
            cont_phase_energy = e_sorted
            cont_phase_shift = np.vstack(delta_rows)

    result = {
        "Z": float(config.Z),
        "r": r_cont,
        "r_bound": r_bound,
        "r_cont": r_cont,
        "g_ii": g_ii,
        "n0": float(n0),
        "n_bound": n_bound,
        "n_ion": n_ion,
        "n_cont": n_cont,
        "n_cont_pre_tail": n_cont_pre_tail,
        "n_cont_dft_raw": n_cont_dft_raw,
        "n_cont_tail_meta": n_cont_tail_meta,
        "n_cont_source": n_cont_source,
        "n_full": n_full,
        "n_full_pre_tail": n_full_pre_tail,
        "n_full_source": n_full_source,
        "source_closure_meta": source_closure_meta,
        "n_ext": n_ext,
        "n_pa": n_pa,
        "n_scr": n_scr,
        "v_full": v_full,
        "v_ext": v_ext,
        "history": history,
        "converged": bool(scf_converged),
        "iters": int(len(history)),
        "r_ws": float(r_ws),
        "mu": float(mu),
        "charge_ws": float(charge_ws),
        "charge_neutral": float(charge_neutral),
        "neutrality_mode": neutrality_mode,
        "zbar": float(zbar),
    }
    if cont_phase_energy is not None and cont_phase_shift is not None:
        result["cont_phase_energy_ha"] = cont_phase_energy
        result["cont_phase_shift_rad"] = cont_phase_shift
    if config.store_final_bound_debug:
        result["debug_bound_eigvals"] = np.asarray(vals, dtype=float)
        result["debug_bound_eigvecs"] = np.asarray(vecs, dtype=float)
        result["debug_energy_cut"] = float(energy_cut)
        result["debug_ion_gamma"] = float(ion_gamma)
    if scf_snapshots is not None:
        result["scf_snapshots"] = list(scf_snapshots)
    return result

def _scf_neutral_inner(config: KSDTFConfig,
                       v_full_init: np.ndarray | None = None,
                       v_ext_init: np.ndarray | None = None,
                       max_iter: int | None = None) -> Dict[str, Any]:
    """
    SCF loop with inner μ solve for each fixed V_eff.

    This performs **one KS solve per SCF step**, then searches μ using the
    precomputed bound states and continuum basis (if available).

    Numbered workflow
    -----------------
    1) Build/update bound-continuum partition (energy_cut) and map V_eff to bound grid.
    2) Precompute continuum basis for current V_eff (when enabled).
    3) Solve bound eigenproblem once for this SCF step.
    4) Inner μ root solve with reused basis: assemble n_full(mu) and enforce neutrality.
    5) Update n0 (ideal/tail/window/fixed modes).
    6) Build n_ion and optional n_ext, then rebuild V_eff from densities.
    7) Apply source-closure regularization (optional, potential-source only).
    8) Mix potentials (linear/Eyert), record diagnostics, test convergence.
    """
    r_cont, step_cont, kind_cont, r_bound, step_bound, kind_bound = _build_grid_pair(config)
    r_ws = _resolve_r_ws(config.n_i, config.r_ws)
    if config.n_i is None:
        n_i_val = 3.0 / (4.0 * np.pi * r_ws ** 3)
    else:
        n_i_val = float(config.n_i)
    compute_external = bool(config.compute_external)

    g_ii = IonSphereStepModel(r_ws=r_ws).g_ii(r_cont)

    mu = float(config.mu)
    n0_mode = str(config.n0_mode).lower().strip()
    if n0_mode == "ideal":
        n0 = ideal_unbound_density(mu, config.temperature)
    elif n0_mode == "fixed":
        if config.n0_fixed is None:
            raise ValueError("n0_mode='fixed' requires n0_fixed to be set.")
        n0 = float(config.n0_fixed)
    elif n0_mode == "tail":
        # Bootstrap from ideal before inner-mu has produced the first n_full.
        n0 = ideal_unbound_density(mu, config.temperature)
    elif n0_mode == "window":
        # Bootstrap from ideal before inner-mu has produced the first n_full.
        n0 = ideal_unbound_density(mu, config.temperature)
    else:
        raise ValueError("n0_mode must be 'ideal', 'fixed', 'tail', or 'window'.")

    v_full = (-float(config.Z) / r_cont) if v_full_init is None else v_full_init.copy()
    v_ext = np.zeros_like(r_cont) if v_ext_init is None else v_ext_init.copy()

    continuum = _select_continuum_model(config.continuum_model)

    bound_cut_mode = str(config.bound_energy_cut_mode).lower().strip()
    if bound_cut_mode == "auto":
        bound_cut_mode = "v_ws" if not compute_external else "zero"
    if bound_cut_mode not in ("zero", "v_ws", "v_rmax", "v_frac", "fixed"):
        raise ValueError(
            "bound_energy_cut_mode must be 'zero', 'v_ws', 'v_rmax', 'v_frac', 'fixed', or 'auto'."
        )
    # Optional tail shift (explicit only): do not couple to bound_cut_mode.
    shift_tail = bool(config.shift_v_eff_tail)
    if config.shift_v_eff_tail and config.verbose:
        print(
            "  [SCF] Enabling V_eff tail shift to align zero-energy cutoff "
            f"(mode={config.v_tail_mode}, frac={config.v_tail_fraction})."
        )
    outer_decay = bool(config.full_v_eff_outer_decay)
    if outer_decay and config.verbose:
        print(
            "  [SCF] Enabling experimental full V_eff outer decay "
            f"(start={float(config.full_v_eff_outer_decay_start_rws):.3f}*R_ws, "
            f"lambda={float(config.full_v_eff_outer_decay_length_rws):.3f}*R_ws)."
        )

    history = []
    scf_converged = False
    n_bound = np.zeros_like(r_cont)
    n_cont = np.zeros_like(r_cont)
    n_cont_pre_tail = np.zeros_like(r_cont)
    n_full_pre_tail = np.zeros_like(r_cont)
    n_cont_dft_raw = np.full_like(r_cont, np.nan)
    n_cont_source = np.zeros_like(r_cont)
    n_full = np.zeros_like(r_cont)
    n_full_source = np.zeros_like(r_cont)
    source_closure_meta = {"applied": False}
    n_cont_tail_meta: dict[str, Any] = {"applied": False}
    n_ext = np.zeros_like(r_cont)
    n_ion = np.zeros_like(r_cont)
    n_pa = np.zeros_like(r_cont)
    charge_ws = 0.0
    zbar = 0.0
    neutrality_mode = str(config.neutrality_mode).lower().strip()
    if neutrality_mode not in ("auto", "ws", "pa"):
        raise ValueError("neutrality_mode must be 'auto', 'ws', or 'pa'.")
    if neutrality_mode == "auto":
        neutrality_mode = "pa" if compute_external else "ws"

    n_full_prev = None
    v_full_prev = None
    n_ext_prev = None
    zbar_prev = None
    mixing_scheme = str(config.mixing_scheme).lower().strip()
    if mixing_scheme not in ("linear", "eyert"):
        raise ValueError("mixing_scheme must be 'linear' or 'eyert'.")
    mixing_m = max(int(config.mixing_m), 1)
    mixing_w0 = float(config.mixing_w0)
    mix = float(config.mix)
    x_prev = None
    f_prev = None
    dx_hist = deque(maxlen=mixing_m)
    df_hist = deque(maxlen=mixing_m)
    scf_snapshots = None
    if config.store_scf_snapshots_last is not None and int(config.store_scf_snapshots_last) > 0:
        scf_snapshots = deque(maxlen=int(config.store_scf_snapshots_last))

    scf_iters = max_iter if max_iter is not None else config.max_iter

    if shift_tail:
        v_full = v_full - _tail_shift_value(
            r_cont,
            v_full,
            config.v_tail_fraction,
            config.v_tail_mode,
        )
        if compute_external:
            v_ext = v_ext - _tail_shift_value(
                r_cont,
                v_ext,
                config.v_tail_fraction,
                config.v_tail_mode,
            )
    if outer_decay:
        v_full = _apply_outer_v_eff_decay(
            r_cont,
            v_full,
            r_ws=r_ws,
            enabled=True,
            start_rws=float(config.full_v_eff_outer_decay_start_rws),
            decay_length_rws=float(config.full_v_eff_outer_decay_length_rws),
        )

    if config.verbose:
        print(f"  [SCF] mu={mu:.6f} Ha, n0={n0:.6e}")
        if compute_external:
            print(
                "  iter | dn_rel | dv_rel | Q_full | Q_bound | Q_cont | "
                "Zbar | charge_rel | dZbar"
            )
        else:
            print(
                "  iter | dn_rel | dv_rel | Q_full | Q_bound | Q_cont | "
                "Zbar | charge_rel"
            )

    mu_solver = str(config.mu_solver).lower().strip()
    if mu_solver not in ("bracket", "brent", "secant"):
        raise ValueError("mu_solver must be 'bracket', 'brent', or 'secant'.")
    mu_verbose = bool(getattr(config, "mu_verbose", False))
    mu_lo, mu_hi = map(float, config.mu_bounds)
    if mu_lo >= mu_hi:
        raise ValueError("mu_bounds must be (mu_min, mu_max) with mu_min < mu_max.")
    mu_bounds_strict = bool(config.mu_bounds_strict)
    mu_min_bound = float(mu_lo)
    mu_max_bound = float(mu_hi)

    perf_on = bool(config.perf_diag)
    perf_every = max(int(config.perf_print_every), 1)
    perf_show_stage = bool(getattr(config, "perf_show_stage", True))
    perf_show_basis = bool(getattr(config, "perf_show_basis", True))

    def _fmt_perf_int(value: Any) -> str:
        """
        Safe formatter for optional integer-like perf counters.

        Adaptive metadata can be missing in some execution paths, which leaves
        NaN placeholders in `perf`. Converting NaN with int(...) raises, so we
        print "nan" in that case.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "nan"
        if not np.isfinite(v):
            return "nan"
        return str(int(v))

    def _merge_perf_sum(accum: dict[str, float], update: dict[str, Any]) -> None:
        """
        Add finite numeric perf counters from one attempt into an accumulator.
        """
        for key, value in update.items():
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            accum[key] = float(accum.get(key, 0.0) + val)

    def _run_iteration_attempt(
        *,
        it: int,
        mu_guess_val: float,
        n0_current: float,
        v_full_current: np.ndarray,
        v_ext_current: np.ndarray,
        zbar_prev_current: float | None,
        stage_e_max_floor: float,
    ) -> dict[str, Any]:
        """
        Run the expensive pre-mix part of one SCF iteration once.

        This includes the continuum basis build, bound solve, inner-μ solve,
        and density assembly. If the resulting neutrality miss is too large,
        the caller can raise ``stage_e_max_floor`` and repeat the same SCF
        iteration without advancing the outer SCF state.
        """
        perf_local: dict[str, float] = {}
        t_iter_local = time.perf_counter() if perf_on else 0.0
        t_stage_local = t_iter_local

        if bound_cut_mode == "fixed":
            energy_cut = float(config.bound_energy_cut) if config.bound_energy_cut is not None else 0.0
        elif bound_cut_mode == "v_ws":
            energy_cut = float(np.interp(r_ws, r_cont, v_full_current))
        elif bound_cut_mode == "v_rmax":
            energy_cut = float(v_full_current[-1])
        elif bound_cut_mode == "v_frac":
            r_frac = float(config.bound_energy_cut) if config.bound_energy_cut is not None else 0.7
            r_frac = min(max(r_frac, 0.0), 1.0)
            energy_cut = float(np.interp(r_frac * float(r_cont[-1]), r_cont, v_full_current))
        else:
            energy_cut = 0.0

        if r_bound.shape == r_cont.shape and np.allclose(r_bound, r_cont):
            v_full_bound = v_full_current
        else:
            v_full_bound = interp_to_grid(r_cont, v_full_current, r_bound)

        cont_params_full, cont_params_ext = _split_continuum_params_for_full_ext(config.continuum_params)
        cont_params = cont_params_full
        cont_params["v_eff"] = v_full_current
        cont_params_ext["v_eff"] = v_ext_current
        cont_e_max_iter = max(
            float(stage_e_max_floor),
            _resolve_iteration_continuum_e_max(
                cont_params,
                mu_ref=float(mu_guess_val),
                temperature=float(config.temperature),
            ),
        )
        cont_params["e_max"] = float(cont_e_max_iter)
        if compute_external:
            cont_params_ext = _apply_external_energy_floor(cont_params_ext, r_cont, v_ext_current)
            cont_params_ext["v_eff"] = v_ext_current

        ion_gamma_mode = str(config.ion_gamma_mode).lower().strip()
        if ion_gamma_mode not in ("fixed", "scattering"):
            raise ValueError("ion_gamma_mode must be 'fixed' or 'scattering'.")
        ion_gamma_scale = float(getattr(config, "ion_gamma_scale", 1.0))
        if ion_gamma_scale <= 0.0:
            raise ValueError("ion_gamma_scale must be positive.")
        energy_cache_full = {} if ion_gamma_mode == "scattering" else None
        energy_cache_ext = {} if (ion_gamma_mode == "scattering" and compute_external) else None
        if energy_cache_full is not None:
            cont_params["energy_cache"] = energy_cache_full
        if energy_cache_ext is not None:
            cont_params_ext["energy_cache"] = energy_cache_ext

        cont_solve_end = _continuum_prefix_length(r_cont, cont_params.get("solve_rmax", None))
        r_cont_solve = r_cont[:cont_solve_end]
        v_full_solve = v_full_current[:cont_solve_end]
        cont_solve_end_ext = _continuum_prefix_length(r_cont, cont_params_ext.get("solve_rmax", None))
        r_cont_solve_ext = r_cont[:cont_solve_end_ext]
        v_ext_solve = v_ext_current[:cont_solve_end_ext]
        cont_params["l_max"] = _resolve_iteration_continuum_l_max(
            cont_params,
            e_max=float(cont_e_max_iter),
            r_eval_max=float(r_cont_solve[-1]),
        )
        if compute_external:
            cont_params_ext["l_max"] = _resolve_iteration_continuum_l_max(
                cont_params_ext,
                e_max=float(cont_params_ext.get("e_max", cont_e_max_iter)),
                r_eval_max=float(r_cont_solve_ext[-1]),
            )

        cont_basis = None
        e_grid = None
        e_weights = None
        cont_basis_ext = None
        e_grid_ext = None
        e_weights_ext = None
        basis_meta = None
        basis_meta_ext = None
        if isinstance(continuum, QuantumContinuumScattering):
            energy_mode = str(cont_params.get("energy_mode", "linear")).lower()
            if energy_mode == "linear":
                e_min = float(cont_params.get("e_min", 1e-6))
                e_max = float(cont_params.get("e_max", max(mu_guess_val + 10.0 * config.temperature, 5.0 * config.temperature)))
                e_min = max(e_min, 1e-12)
                n_e = int(cont_params.get("n_e", 200))
                e_grid = np.linspace(e_min, e_max, n_e)
                match_slice = cont_params.get("match_slice", None)
                match_r_cut = cont_params.get("match_r_cut", None)
                match_width = cont_params.get("match_width", None)
                if match_width is not None:
                    match_width = float(match_width)
                if match_slice is None and match_r_cut is not None:
                    idx_cut = int(np.searchsorted(r_cont_solve, float(match_r_cut)))
                    if 0 < idx_cut < r_cont_solve.size - 1:
                        if match_width is not None:
                            idx_end = int(np.searchsorted(r_cont_solve, float(match_r_cut + match_width)))
                            match_slice = (idx_cut, min(max(idx_end, idx_cut + 1), r_cont_solve.size))
                        else:
                            match_slice = (idx_cut, r_cont_solve.size)
                energy_cache_basis = energy_cache_full
                n_jobs_basis = cont_params.get("n_jobs", None)
                if n_jobs_basis is not None and int(n_jobs_basis) > 1:
                    energy_cache_basis = None
                cont_basis_result = continuum_density_scattering_basis(
                    v_full_solve,
                    r_cont_solve,
                    e_grid,
                    int(cont_params.get("l_max", 6)),
                    kind_cont,
                    step_cont,
                    l_pad=int(cont_params.get("l_pad", 2)),
                    l_cap_strategy=str(cont_params.get("l_cap_strategy", "match")),
                    match_fraction=float(cont_params.get("match_fraction", 0.2)),
                    match_slice=match_slice,
                    match_r_cut=match_r_cut,
                    match_fraction_mode=str(cont_params.get("match_fraction_mode", "r")),
                    match_kr_min=cont_params.get("match_kr_min", 4.0),
                    match_v_tol=cont_params.get("match_v_tol", 1e-4),
                    match_min_points=int(cont_params.get("match_min_points", 12)),
                    match_asymptotic=str(cont_params.get("match_asymptotic", "auto")),
                    match_coulomb_tol=float(cont_params.get("match_coulomb_tol", 0.1)),
                    match_allow_shift=bool(cont_params.get("match_allow_shift", True)),
                    match_fallback=str(cont_params.get("match_fallback", "free")),
                    prop_rescale_limit=cont_params.get("prop_rescale_limit", 1e6),
                    energy_cache=energy_cache_basis,
                    n_jobs=n_jobs_basis,
                    return_meta=bool(perf_on),
                )
                if perf_on:
                    cont_basis, basis_meta_basis = cont_basis_result
                else:
                    cont_basis = cont_basis_result
                    basis_meta_basis = {}
                if compute_external:
                    energy_cache_basis_ext = energy_cache_ext
                    n_jobs_basis_ext = cont_params_ext.get("n_jobs", n_jobs_basis)
                    if n_jobs_basis_ext is not None and int(n_jobs_basis_ext) > 1:
                        energy_cache_basis_ext = None
                    e_min_ext = float(cont_params_ext.get("e_min", e_min))
                    e_max_ext = float(cont_params_ext.get("e_max", e_max))
                    e_min_ext = max(e_min_ext, 1e-12)
                    n_e_ext = int(cont_params_ext.get("n_e", n_e))
                    e_grid_ext = np.linspace(e_min_ext, e_max_ext, n_e_ext)
                    e_weights_ext = _trapz_weights(e_grid_ext)
                    cont_basis_ext_result = continuum_density_scattering_basis(
                        v_ext_solve,
                        r_cont_solve_ext,
                        e_grid_ext,
                        int(cont_params_ext.get("l_max", 6)),
                        kind_cont,
                        step_cont,
                        l_pad=int(cont_params_ext.get("l_pad", 2)),
                        l_cap_strategy=str(cont_params_ext.get("l_cap_strategy", "match")),
                        match_fraction=float(cont_params_ext.get("match_fraction", 0.2)),
                        match_slice=match_slice,
                        match_r_cut=match_r_cut,
                        match_fraction_mode=str(cont_params_ext.get("match_fraction_mode", "r")),
                        match_kr_min=cont_params_ext.get("match_kr_min", 4.0),
                        match_v_tol=cont_params_ext.get("match_v_tol", 1e-4),
                        match_min_points=int(cont_params_ext.get("match_min_points", 12)),
                        match_asymptotic=str(cont_params_ext.get("match_asymptotic", "auto")),
                        match_coulomb_tol=float(cont_params_ext.get("match_coulomb_tol", 0.1)),
                        match_allow_shift=bool(cont_params_ext.get("match_allow_shift", True)),
                        match_fallback=str(cont_params_ext.get("match_fallback", "free")),
                        prop_rescale_limit=cont_params_ext.get("prop_rescale_limit", 1e6),
                        energy_cache=energy_cache_basis_ext,
                        n_jobs=n_jobs_basis_ext,
                        return_meta=bool(perf_on),
                    )
                    if perf_on:
                        cont_basis_ext, basis_meta_ext_basis = cont_basis_ext_result
                    else:
                        cont_basis_ext = cont_basis_ext_result
                        basis_meta_ext_basis = {}
                    basis_meta_ext = {
                        "energy_mode": "linear",
                        "n_e_basis": int(e_grid_ext.size),
                        "n_cache": int(len(energy_cache_basis_ext)) if energy_cache_basis_ext is not None else 0,
                    }
                    basis_meta_ext.update(dict(basis_meta_ext_basis))
                basis_meta = {
                    "energy_mode": "linear",
                    "n_e_basis": int(e_grid.size),
                    "n_cache": int(len(energy_cache_basis)) if energy_cache_basis is not None else 0,
                }
                basis_meta.update(dict(basis_meta_basis))
            elif energy_mode == "adaptive" and bool(cont_params.get("adaptive_reuse_basis", True)):
                e_min = float(cont_params.get("e_min", 1e-6))
                e_max = float(cont_params.get("e_max", max(mu_guess_val + 10.0 * config.temperature, 5.0 * config.temperature)))
                e_min = max(e_min, 1e-12)
                match_slice = cont_params.get("match_slice", None)
                match_r_cut = cont_params.get("match_r_cut", None)
                match_width = cont_params.get("match_width", None)
                if match_width is not None:
                    match_width = float(match_width)
                if match_slice is None and match_r_cut is not None:
                    idx_cut = int(np.searchsorted(r_cont, float(match_r_cut)))
                    if 0 < idx_cut < r_cont.size - 1:
                        if match_width is not None:
                            idx_end = int(np.searchsorted(r_cont, float(match_r_cut + match_width)))
                            match_slice = (idx_cut, min(max(idx_end, idx_cut + 1), r_cont.size))
                        else:
                            match_slice = (idx_cut, r_cont.size)
                energy_cache_basis = energy_cache_full if energy_cache_full is not None else {}
                n_jobs_basis = cont_params.get("n_jobs", None)
                adaptive_mode_basis = str(cont_params.get("adaptive_mode", "bisection"))
                adaptive_parallel_mode_basis = str(cont_params.get("adaptive_parallel_mode", "batch")).lower().strip()
                adaptive_shards_basis = cont_params.get("adaptive_shards", None)
                if adaptive_mode_basis == "bisection":
                    delta_tol_basis = float(cont_params.get("delta_tol", np.pi))
                    delta_mode_basis = str(cont_params.get("delta_mode", "sum"))
                else:
                    delta_tol_basis = float(cont_params.get("delta_tol", np.pi / 2.0))
                    delta_mode_basis = str(cont_params.get("delta_mode", "max"))
                _, basis_meta = continuum_density_scattering_adaptive(
                    v_full_solve,
                    r_cont_solve,
                    mu_guess_val,
                    config.temperature,
                    e_min,
                    e_max,
                    int(cont_params.get("l_max", 6)),
                    kind_cont,
                    step_cont,
                    l_pad=int(cont_params.get("l_pad", 2)),
                    match_fraction=float(cont_params.get("match_fraction", 0.2)),
                    match_slice=match_slice,
                    match_r_cut=match_r_cut,
                    match_fraction_mode=str(cont_params.get("match_fraction_mode", "r")),
                    match_width=match_width,
                    match_kr_min=cont_params.get("match_kr_min", 4.0),
                    match_v_tol=cont_params.get("match_v_tol", 1e-4),
                    match_min_points=int(cont_params.get("match_min_points", 12)),
                    match_asymptotic=str(cont_params.get("match_asymptotic", "auto")),
                    match_coulomb_tol=float(cont_params.get("match_coulomb_tol", 0.1)),
                    match_allow_shift=bool(cont_params.get("match_allow_shift", True)),
                    match_fallback=str(cont_params.get("match_fallback", "free")),
                    prop_rescale_limit=cont_params.get("prop_rescale_limit", 1e6),
                    l_cap_strategy=str(cont_params.get("l_cap_strategy", "match")),
                    e_tol=float(cont_params.get("e_tol", 1e-3)),
                    e_max_depth=int(cont_params.get("e_max_depth", 10)),
                    e_min_width=float(cont_params.get("e_min_width", 1e-4)),
                    n_e_base=int(cont_params.get("n_e_base", 8)),
                    e_base_grid=str(cont_params.get("e_base_grid", "linear")),
                    resonance_tol=cont_params.get("resonance_tol", None),
                    resonance_r_fractions=cont_params.get("resonance_r_fractions", (0.25, 0.5, 0.75)),
                    resonance_floor=float(cont_params.get("resonance_floor", 1e-8)),
                    delta_tol=delta_tol_basis,
                    delta_mode=delta_mode_basis,
                    adaptive_mode=adaptive_mode_basis,
                    bisection_max_depth=cont_params.get("bisection_max_depth", None),
                    resonance_window_factor=float(cont_params.get("resonance_window_factor", 12.0)),
                    resonance_window_min=cont_params.get("resonance_window_min", None),
                    resonance_window_max=cont_params.get("resonance_window_max", None),
                    resonance_max_windows=cont_params.get("resonance_max_windows", None),
                    energy_cache=energy_cache_basis,
                    n_jobs=n_jobs_basis,
                    adaptive_parallel_mode=adaptive_parallel_mode_basis,
                    adaptive_shards=adaptive_shards_basis,
                    apply_occ=False,
                    collect_perf=bool(perf_on),
                )
                if len(energy_cache_basis) >= 2:
                    e_grid = np.array(sorted(float(e) for e in energy_cache_basis.keys()), dtype=float)
                    cont_basis = np.vstack([np.asarray(energy_cache_basis[float(e)][0], dtype=float) for e in e_grid])
                    e_weights = _trapz_weights(e_grid)
                    if basis_meta is None:
                        basis_meta = {}
                    basis_meta["n_e_basis"] = int(e_grid.size)

                if compute_external:
                    energy_cache_basis_ext = energy_cache_ext if energy_cache_ext is not None else {}
                    e_min_ext = float(cont_params_ext.get("e_min", e_min))
                    e_max_ext = float(cont_params_ext.get("e_max", e_max))
                    e_min_ext = max(e_min_ext, 1e-12)
                    n_jobs_basis_ext = cont_params_ext.get("n_jobs", n_jobs_basis)
                    adaptive_mode_basis_ext = str(cont_params_ext.get("adaptive_mode", adaptive_mode_basis))
                    adaptive_parallel_mode_basis_ext = str(cont_params_ext.get("adaptive_parallel_mode", adaptive_parallel_mode_basis)).lower().strip()
                    adaptive_shards_basis_ext = cont_params_ext.get("adaptive_shards", adaptive_shards_basis)
                    if adaptive_mode_basis_ext == "bisection":
                        delta_tol_basis_ext = float(cont_params_ext.get("delta_tol", np.pi))
                        delta_mode_basis_ext = str(cont_params_ext.get("delta_mode", "sum"))
                    else:
                        delta_tol_basis_ext = float(cont_params_ext.get("delta_tol", np.pi / 2.0))
                        delta_mode_basis_ext = str(cont_params_ext.get("delta_mode", "max"))
                    _, basis_meta_ext = continuum_density_scattering_adaptive(
                        v_ext_solve,
                        r_cont_solve_ext,
                        mu_guess_val,
                        config.temperature,
                        e_min_ext,
                        e_max_ext,
                        int(cont_params_ext.get("l_max", 6)),
                        kind_cont,
                        step_cont,
                        l_pad=int(cont_params_ext.get("l_pad", 2)),
                        match_fraction=float(cont_params_ext.get("match_fraction", 0.2)),
                        match_slice=match_slice,
                        match_r_cut=match_r_cut,
                        match_fraction_mode=str(cont_params_ext.get("match_fraction_mode", "r")),
                        match_width=match_width,
                        match_kr_min=cont_params_ext.get("match_kr_min", 4.0),
                        match_v_tol=cont_params_ext.get("match_v_tol", 1e-4),
                        match_min_points=int(cont_params_ext.get("match_min_points", 12)),
                        match_asymptotic=str(cont_params_ext.get("match_asymptotic", "auto")),
                        match_coulomb_tol=float(cont_params_ext.get("match_coulomb_tol", 0.1)),
                        match_allow_shift=bool(cont_params_ext.get("match_allow_shift", True)),
                        match_fallback=str(cont_params_ext.get("match_fallback", "free")),
                        prop_rescale_limit=cont_params_ext.get("prop_rescale_limit", 1e6),
                        l_cap_strategy=str(cont_params_ext.get("l_cap_strategy", "match")),
                        e_tol=float(cont_params_ext.get("e_tol", 1e-3)),
                        e_max_depth=int(cont_params_ext.get("e_max_depth", 10)),
                        e_min_width=float(cont_params_ext.get("e_min_width", 1e-4)),
                        n_e_base=int(cont_params_ext.get("n_e_base", 8)),
                        e_base_grid=str(cont_params_ext.get("e_base_grid", "linear")),
                        resonance_tol=cont_params_ext.get("resonance_tol", None),
                        resonance_r_fractions=cont_params_ext.get("resonance_r_fractions", (0.25, 0.5, 0.75)),
                        resonance_floor=float(cont_params_ext.get("resonance_floor", 1e-8)),
                        delta_tol=delta_tol_basis_ext,
                        delta_mode=delta_mode_basis_ext,
                        adaptive_mode=adaptive_mode_basis_ext,
                        bisection_max_depth=cont_params_ext.get("bisection_max_depth", None),
                        resonance_window_factor=float(cont_params_ext.get("resonance_window_factor", 12.0)),
                        resonance_window_min=cont_params_ext.get("resonance_window_min", None),
                        resonance_window_max=cont_params_ext.get("resonance_window_max", None),
                        resonance_max_windows=cont_params_ext.get("resonance_max_windows", None),
                        energy_cache=energy_cache_basis_ext,
                        n_jobs=n_jobs_basis_ext,
                        adaptive_parallel_mode=adaptive_parallel_mode_basis_ext,
                        adaptive_shards=adaptive_shards_basis_ext,
                        apply_occ=False,
                        collect_perf=bool(perf_on),
                    )
                    if len(energy_cache_basis_ext) >= 2:
                        e_grid_ext = np.array(sorted(float(e) for e in energy_cache_basis_ext.keys()), dtype=float)
                        cont_basis_ext = np.vstack(
                            [np.asarray(energy_cache_basis_ext[float(e)][0], dtype=float) for e in e_grid_ext]
                        )
                        e_weights_ext = _trapz_weights(e_grid_ext)
                        if basis_meta_ext is None:
                            basis_meta_ext = {}
                        basis_meta_ext["n_e_basis"] = int(e_grid_ext.size)
        if perf_on:
            perf_local["basis_build"] = time.perf_counter() - t_stage_local
            t_stage_local = time.perf_counter()

        if cont_basis is not None and e_grid is not None and e_weights is None:
            e_weights = _trapz_weights(np.asarray(e_grid, dtype=float))
        if cont_basis_ext is not None and e_grid_ext is not None and e_weights_ext is None:
            e_weights_ext = _trapz_weights(np.asarray(e_grid_ext, dtype=float))

        vals, vecs = solve_bound_states_sparse_numerov(
            v_full_bound,
            r_bound,
            step_bound,
            np.asarray(config.l_list),
            n_states=(
                np.asarray(config.n_states_by_l, dtype=int)
                if config.n_states_by_l is not None
                else int(config.n_states)
            ),
            boundary=config.boundary,
            n_jobs=config.n_jobs,
        )
        if perf_on:
            perf_local["bound_solve"] = time.perf_counter() - t_stage_local
            perf_local["precompute"] = perf_local["basis_build"] + perf_local["bound_solve"]
            if isinstance(basis_meta, dict):
                perf_local["basis_n_eval"] = float(basis_meta.get("n_eval", np.nan))
                perf_local["basis_n_cache_hits"] = float(basis_meta.get("n_cache_hits", np.nan))
                perf_local["basis_n_e"] = float(basis_meta.get("n_e_basis", np.nan))
                perf_local["basis_n_base_per_shard"] = float(basis_meta.get("n_base_per_shard", np.nan))
                perf_local["basis_shard_cache_merged"] = float(basis_meta.get("shard_cache_merged", np.nan))
                perf_local["basis_n_windows"] = float(basis_meta.get("n_windows", np.nan))
                perf_local["basis_eval_total"] = float(basis_meta.get("perf_eval_total_s", np.nan))
                perf_local["basis_plan"] = float(basis_meta.get("perf_plan_s", np.nan))
                perf_local["basis_propagate"] = float(basis_meta.get("perf_propagate_s", np.nan))
                perf_local["basis_match"] = float(basis_meta.get("perf_match_s", np.nan))
                perf_local["basis_accumulate"] = float(basis_meta.get("perf_accumulate_s", np.nan))
                perf_local["basis_control"] = float(basis_meta.get("perf_control_s", np.nan))
            if isinstance(basis_meta_ext, dict):
                perf_local["basis_ext_n_eval"] = float(basis_meta_ext.get("n_eval", np.nan))
                perf_local["basis_ext_n_cache_hits"] = float(basis_meta_ext.get("n_cache_hits", np.nan))
                perf_local["basis_ext_n_e"] = float(basis_meta_ext.get("n_e_basis", np.nan))
                perf_local["basis_ext_eval_total"] = float(basis_meta_ext.get("perf_eval_total_s", np.nan))
                perf_local["basis_ext_plan"] = float(basis_meta_ext.get("perf_plan_s", np.nan))
                perf_local["basis_ext_propagate"] = float(basis_meta_ext.get("perf_propagate_s", np.nan))
                perf_local["basis_ext_match"] = float(basis_meta_ext.get("perf_match_s", np.nan))
                perf_local["basis_ext_accumulate"] = float(basis_meta_ext.get("perf_accumulate_s", np.nan))
                perf_local["basis_ext_control"] = float(basis_meta_ext.get("perf_control_s", np.nan))
            t_stage_local = time.perf_counter()

        def _eval_mu(mu_val: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, dict[str, Any]]:
            if n0_mode == "ideal":
                n0_eval = float(ideal_unbound_density(mu_val, config.temperature))
            elif n0_mode == "fixed":
                n0_eval = float(config.n0_fixed) if config.n0_fixed is not None else float(n0_current)
            else:
                n0_eval = float(n0_current)

            tail_match = bool(cont_params.get("tail_match", False))
            tail_match_target = str(cont_params.get("tail_match_target", "cont")).lower()
            apply_cont_tail = (
                tail_match
                and tail_match_target in ("cont", "both")
                and cont_solve_end >= r_cont.size
            )
            if cont_basis is not None and e_grid is not None:
                occ = fermi_dirac(e_grid, mu_val, config.temperature)
                weights_cont = occ * e_weights if e_weights is not None else occ * _trapz_weights(e_grid)
                n_cont_eval = _weighted_energy_sum_numba(
                    np.asarray(cont_basis, dtype=float),
                    np.asarray(weights_cont, dtype=float),
                )
                n_cont_val, n_cont_pre_tail_val, tail_meta_local = _rebuild_continuum_on_full_grid(
                    r_cont,
                    n_cont_eval,
                    idx_eval_end=cont_solve_end,
                    params=cont_params,
                    n0=n0_eval,
                    mu_id=mu_val,
                    temperature=config.temperature,
                )
            else:
                cont_params_eval = dict(cont_params)
                cont_params_eval["tail_n0"] = float(n0_eval)
                cont_params_eval["tail_mu_id"] = float(mu_val)
                cont_params_eval["v_eff"] = v_full_solve
                n_cont_eval = continuum.density(
                    r_cont_solve,
                    mu_val,
                    config.temperature,
                    params=cont_params_eval,
                )
                n_cont_val, n_cont_pre_tail_val, tail_meta_local = _rebuild_continuum_on_full_grid(
                    r_cont,
                    n_cont_eval,
                    idx_eval_end=cont_solve_end,
                    params=cont_params_eval,
                    n0=n0_eval,
                    mu_id=mu_val,
                    temperature=config.temperature,
                )

            n_cont_pre_tail_val = np.asarray(n_cont_pre_tail_val, dtype=float).copy()
            n_cont_dft_raw_val = np.full_like(r_cont, np.nan, dtype=float)
            raw_len = min(int(np.asarray(n_cont_eval, dtype=float).size), int(r_cont.size))
            if raw_len > 0:
                n_cont_dft_raw_val[:raw_len] = np.asarray(n_cont_eval, dtype=float)[:raw_len]

            if apply_cont_tail and cont_basis is not None and e_grid is not None:
                from otter.electronic.continuum.tail import apply_tail_match

                tail_n0_fixed = cont_params.get("tail_n0_fixed", None)
                tail_mu_fixed = cont_params.get("tail_mu_id_fixed", None)
                tail_n0 = float(n0_eval if tail_n0_fixed is None else tail_n0_fixed)
                tail_mu_id = float(mu_val if tail_mu_fixed is None else tail_mu_fixed)
                fit_points = int(cont_params.get("tail_fit_points", 16))
                blend_points = int(cont_params.get("tail_blend_points", 0))
                tail_model = str(cont_params.get("tail_model", "auto"))
                tail_auto_rel_improve_tol = float(cont_params.get("tail_auto_rel_improve_tol", 0.15))
                tail_auto_signal_rel_tol = float(cont_params.get("tail_auto_signal_rel_tol", 2.0e-5))
                tail_fallback_on_error = bool(cont_params.get("tail_fallback_on_error", True))
                tail_r_cut = cont_params.get("tail_r_cut", None)
                if tail_r_cut is None:
                    tail_r_cut = float(cont_params.get("tail_auto_r_fraction", 0.7)) * float(r_cont[-1])
                idx_cut = int(np.searchsorted(r_cont, float(tail_r_cut)))
                if 0 < idx_cut < r_cont.size - 1:
                    try:
                        n_cont_val, tail_meta_local = apply_tail_match(
                            r_cont,
                            n_cont_val,
                            tail_n0,
                            tail_mu_id,
                            config.temperature,
                            idx_cut,
                            fit_points=fit_points,
                            blend_points=blend_points,
                            model=tail_model,
                            auto_rel_improve_tol=tail_auto_rel_improve_tol,
                            auto_signal_rel_tol=tail_auto_signal_rel_tol,
                        )
                        tail_meta_local = {
                            **dict(tail_meta_local),
                            "applied": True,
                            "tail_r_cut_bohr": float(r_cont[idx_cut]),
                        }
                    except Exception as exc:
                        if not tail_fallback_on_error:
                            raise
                        warnings.warn(
                            f"B3 continuum tail match failed during mu solve at "
                            f"r_cut={float(r_cont[idx_cut]):.6f}; keeping original A3 density. "
                            f"reason={exc}",
                            RuntimeWarning,
                        )
                        tail_meta_local = {
                            "applied": False,
                            "reason": str(exc),
                        }
            if ion_gamma_mode == "scattering":
                ion_gamma_val = ion_gamma_scale * gamma_from_phase_shift_cache(
                    energy_cache_full,
                    mu_val,
                    config.temperature,
                    n_i_val,
                    n0_eval,
                    n0_floor=config.mu_n0_floor,
                )
            else:
                ion_gamma_val = float(config.ion_bound_gamma)
            n_bound_bound_raw = _bound_density(
                r_bound,
                vals,
                vecs,
                config.l_list,
                mu_val,
                config.temperature,
                energy_cut=energy_cut,
                occ_mode=config.bound_occ_mode,
                gamma=ion_gamma_val,
                r_ws=r_ws,
                ws_weight_min=0.0,
            )
            if r_bound.shape == r_cont.shape and np.allclose(r_bound, r_cont):
                n_bound_val = n_bound_bound_raw
            else:
                n_bound_val = interp_to_grid(r_bound, n_bound_bound_raw, r_cont)
            n_full_val = n_bound_val + n_cont_val
            charge_ws_val = _electron_count(r_cont, n_full_val, r_ws)
            eval_diag = {
                "n_cont_pre_tail": np.asarray(n_cont_pre_tail_val, dtype=float),
                "n_full_pre_tail": np.asarray(n_bound_val + n_cont_pre_tail_val, dtype=float),
                "n_cont_dft_raw": np.asarray(n_cont_dft_raw_val, dtype=float),
                "n_cont_tail_meta": dict(tail_meta_local),
            }
            return n_full_val, n_bound_val, n_cont_val, charge_ws_val, ion_gamma_val, eval_diag

        def _solve_mu_for_potential(
            mu_guess_inner: float,
        ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any], dict[str, Any]]:
            z_val = float(config.Z)
            mu_eval_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, float, float, dict[str, Any]]] = {}
            mu_iter_count = 0

            def _eval_mu_cached(mu_val: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, dict[str, Any]]:
                mu_key = float(mu_val)
                cached = mu_eval_cache.get(mu_key)
                if cached is None:
                    cached = _eval_mu(mu_key)
                    mu_eval_cache[mu_key] = cached
                return cached

            def _package_mu_solution(
                mu_out: float,
                n_full_out: np.ndarray,
                n_bound_out: np.ndarray,
                n_cont_out: np.ndarray,
                ion_gamma_out: float,
                eval_diag_out: dict[str, Any],
            ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any], dict[str, Any]]:
                return (
                    float(mu_out),
                    n_full_out,
                    n_bound_out,
                    n_cont_out,
                    float(ion_gamma_out),
                    eval_diag_out,
                    {
                        "mu_n_eval": int(len(mu_eval_cache)),
                        "mu_n_iter": int(mu_iter_count),
                    },
                )

            mu_lo_eval = float(mu_lo)
            mu_hi_eval = float(mu_hi)
            n_full_lo, _, _, charge_lo, _, _ = _eval_mu_cached(mu_lo_eval)
            n_full_hi, _, _, charge_hi, _, _ = _eval_mu_cached(mu_hi_eval)
            f_lo = charge_lo - z_val
            f_hi = charge_hi - z_val

            if mu_verbose:
                print(f"[mu bracket] mu_lo={mu_lo_eval:.3f}, charge_err={f_lo:.3e}")
                print(f"[mu bracket] mu_hi={mu_hi_eval:.3f}, charge_err={f_hi:.3e}")

            if f_lo * f_hi > 0.0:
                if mu_bounds_strict:
                    raise ValueError("mu_bounds do not bracket neutrality within strict bounds.")
                step = float(config.mu_bracket_step)
                max_iter = int(config.mu_bracket_max_iter)
                for _ in range(max_iter):
                    mu_lo_eval = max(mu_min_bound, mu_lo_eval - step)
                    mu_hi_eval = min(mu_max_bound, mu_hi_eval + step)
                    n_full_lo, _, _, charge_lo, _, _ = _eval_mu_cached(mu_lo_eval)
                    n_full_hi, _, _, charge_hi, _, _ = _eval_mu_cached(mu_hi_eval)
                    f_lo = charge_lo - z_val
                    f_hi = charge_hi - z_val
                    if mu_verbose:
                        print(f"[mu expand] mu_lo={mu_lo_eval:.3f}, charge_err={f_lo:.3e}")
                        print(f"[mu expand] mu_hi={mu_hi_eval:.3f}, charge_err={f_hi:.3e}")
                    if f_lo * f_hi <= 0.0:
                        break
                    if mu_lo_eval <= mu_min_bound and mu_hi_eval >= mu_max_bound:
                        break
                    step *= 2.0

            if f_lo * f_hi > 0.0:
                n_scan = 21
                mu_scan = np.linspace(mu_min_bound, mu_max_bound, n_scan)
                f_scan = []
                for mu_s in mu_scan:
                    _, _, _, charge_s, _, _ = _eval_mu_cached(float(mu_s))
                    f_scan.append(charge_s - z_val)
                f_scan = np.array(f_scan, dtype=float)
                idx = np.where(np.sign(f_scan[:-1]) != np.sign(f_scan[1:]))[0]
                if idx.size > 0:
                    i0 = int(idx[0])
                    mu_lo_eval = float(mu_scan[i0])
                    mu_hi_eval = float(mu_scan[i0 + 1])
                    f_lo = float(f_scan[i0])
                    f_hi = float(f_scan[i0 + 1])
                    if mu_verbose:
                        print(f"[mu scan] bracketing in [{mu_lo_eval:.3f}, {mu_hi_eval:.3f}]")
                else:
                    i_best = int(np.argmin(np.abs(f_scan)))
                    mu_best = float(mu_scan[i_best])
                    if mu_verbose:
                        print("  [mu] No bracket found; using best |charge_err| from scan.")
                    n_full_best, n_bound_best, n_cont_best, _, ion_gamma_best, eval_diag_best = _eval_mu_cached(mu_best)
                    return _package_mu_solution(
                        mu_best,
                        n_full_best,
                        n_bound_best,
                        n_cont_best,
                        ion_gamma_best,
                        eval_diag_best,
                    )

            if mu_solver == "secant":
                mu0 = float(mu_guess_inner)
                mu1 = mu0 + float(config.mu_bracket_step)
                _, _, _, charge0_raw, _, _ = _eval_mu_cached(mu0)
                charge0 = charge0_raw - z_val
                for _ in range(int(config.mu_max_iter)):
                    mu_iter_count += 1
                    n_full_val, _, _, charge1, _, _ = _eval_mu_cached(mu1)
                    charge1 = charge1 - z_val
                    if mu_verbose:
                        print(f"[mu iter] mu={mu1:.4f}, charge_err={charge1:.3e}")
                    if abs(charge1) < config.mu_tol:
                        break
                    denom = (charge1 - charge0) if abs(charge1 - charge0) > 1e-14 else 1e-14
                    mu_new = mu1 - charge1 * (mu1 - mu0) / denom
                    mu0, mu1, charge0 = mu1, mu_new, charge1
                n_full_val, n_bound_val, n_cont_val, _, ion_gamma_val, eval_diag_val = _eval_mu_cached(mu1)
                return _package_mu_solution(
                    mu1,
                    n_full_val,
                    n_bound_val,
                    n_cont_val,
                    ion_gamma_val,
                    eval_diag_val,
                )

            if mu_solver == "brent":
                from scipy.optimize import brentq

                def f(mu_val: float) -> float:
                    nonlocal mu_iter_count
                    mu_iter_count += 1
                    _, _, _, charge_val, _, _ = _eval_mu_cached(mu_val)
                    charge = charge_val - z_val
                    if mu_verbose:
                        print(f"[mu iter] mu={mu_val:.4f}, charge_err={charge:.3e}")
                    return charge

                try:
                    mu_root = brentq(
                        f,
                        mu_lo_eval,
                        mu_hi_eval,
                        xtol=float(config.mu_tol),
                        rtol=1e-10,
                        maxiter=int(config.mu_max_iter),
                    )
                    n_full_val, n_bound_val, n_cont_val, _, ion_gamma_val, eval_diag_val = _eval_mu_cached(mu_root)
                    return _package_mu_solution(
                        float(mu_root),
                        n_full_val,
                        n_bound_val,
                        n_cont_val,
                        ion_gamma_val,
                        eval_diag_val,
                    )
                except RuntimeError as exc:
                    if mu_verbose:
                        print(f"[mu brent] failed ({exc}); falling back to bisection.")

            mu_left, mu_right = mu_lo_eval, mu_hi_eval
            f_left = (charge_lo - z_val)
            for _ in range(int(config.mu_max_iter)):
                mu_iter_count += 1
                mu_mid = 0.5 * (mu_left + mu_right)
                n_full_mid, n_bound_mid, n_cont_mid, charge_mid, ion_gamma_val, eval_diag_mid = _eval_mu_cached(mu_mid)
                charge_mid = charge_mid - z_val
                if mu_verbose:
                    print(f"[mu iter] mu={mu_mid:.4f}, charge_err={charge_mid:.3e}")
                if abs(charge_mid) < config.mu_tol:
                    return _package_mu_solution(
                        mu_mid,
                        n_full_mid,
                        n_bound_mid,
                        n_cont_mid,
                        ion_gamma_val,
                        eval_diag_mid,
                    )
                if f_left * charge_mid < 0.0:
                    mu_right = mu_mid
                else:
                    mu_left = mu_mid
                    f_left = charge_mid
            return _package_mu_solution(
                mu_mid,
                n_full_mid,
                n_bound_mid,
                n_cont_mid,
                ion_gamma_val,
                eval_diag_mid,
            )

        mu_val, n_full_val, n_bound_val, n_cont_val, ion_gamma, eval_diag, mu_diag = _solve_mu_for_potential(mu_guess_val)
        n_cont_pre_tail = np.asarray(eval_diag["n_cont_pre_tail"], dtype=float)
        n_full_pre_tail = np.asarray(eval_diag["n_full_pre_tail"], dtype=float)
        n_cont_dft_raw = np.asarray(eval_diag["n_cont_dft_raw"], dtype=float)
        n_cont_tail_meta = dict(eval_diag["n_cont_tail_meta"])
        if perf_on:
            perf_local["mu_solve"] = time.perf_counter() - t_stage_local
            t_stage_local = time.perf_counter()

        n0_next = float(n0_current)
        if n0_mode == "ideal":
            n0_next = float(ideal_unbound_density(mu_val, config.temperature))
        elif n0_mode == "fixed":
            n0_next = float(config.n0_fixed) if config.n0_fixed is not None else float(n0_next)

        if n0_mode == "tail":
            n0_tail = _tail_shift_value(
                r_cont,
                n_cont_val,
                config.n0_tail_fraction,
                config.n0_tail_mode,
            )
            if config.n0_tail_direct:
                n0_next = float(n0_tail)
            else:
                mix_n0 = float(config.n0_tail_mix)
                n0_next = mix_n0 * n0_tail + (1.0 - mix_n0) * n0_next
        elif n0_mode == "window":
            n0_win = _window_stat_value(
                r_cont,
                n_cont_val,
                config.n0_window_lo_frac,
                config.n0_window_hi_frac,
                config.n0_window_mode,
            )
            if config.n0_window_direct:
                n0_next = float(n0_win)
            else:
                mix_n0 = float(config.n0_window_mix)
                n0_next = mix_n0 * n0_win + (1.0 - mix_n0) * n0_next

        cut_width = float(config.ion_cut_width) * float(r_ws)
        ion_cut_mode = str(config.ion_cut_mode).lower().strip()
        ion_cut_c = float(config.ion_cut_c)
        ion_cut = _ion_cutoff(r_bound, r_ws, cut_width, mode=ion_cut_mode, c=ion_cut_c)
        n_ion_bound = _ion_density(
            r_bound,
            vals,
            vecs,
            config.l_list,
            mu_val,
            config.temperature,
            energy_cut=energy_cut,
            gamma=ion_gamma,
            cutoff=ion_cut,
            r_ws=r_ws,
            ws_weight_min=config.ion_ws_weight_min,
        )
        if r_bound.shape == r_cont.shape and np.allclose(r_bound, r_cont):
            n_ion_val = n_ion_bound
        else:
            n_ion_val = interp_to_grid(r_bound, n_ion_bound, r_cont)

        if compute_external:
            if cont_basis_ext is not None and e_grid_ext is not None:
                occ_ext = fermi_dirac(e_grid_ext, mu_val, config.temperature)
                weights_ext = (
                    occ_ext * e_weights_ext
                    if e_weights_ext is not None
                    else occ_ext * _trapz_weights(e_grid_ext)
                )
                n_ext_val = _weighted_energy_sum_numba(
                    np.asarray(cont_basis_ext, dtype=float),
                    np.asarray(weights_ext, dtype=float),
                )
            else:
                n_ext_val = continuum.density(
                    r_cont,
                    mu_val,
                    config.temperature,
                    params={**cont_params_ext},
                )
        else:
            n_ext_val = np.zeros_like(r_cont)

        tail_match = bool(cont_params.get("tail_match", False))
        tail_match_target = str(cont_params.get("tail_match_target", "cont")).lower()
        if tail_match and tail_match_target in ("full", "both"):
            from otter.electronic.continuum.tail import apply_tail_match

            tail_n0 = float(cont_params.get("tail_n0", n0_next))
            tail_mu_id = float(cont_params.get("tail_mu_id", mu_val))
            fit_points = int(cont_params.get("tail_fit_points", 16))
            blend_points = int(cont_params.get("tail_blend_points", 0))
            tail_model = str(cont_params.get("tail_model", "auto"))
            tail_auto_rel_improve_tol = float(cont_params.get("tail_auto_rel_improve_tol", 0.15))
            tail_auto_signal_rel_tol = float(cont_params.get("tail_auto_signal_rel_tol", 2.0e-5))
            tail_fallback_on_error = bool(cont_params.get("tail_fallback_on_error", True))
            tail_r_cut = cont_params.get("tail_r_cut", None)
            if tail_r_cut is None:
                tail_r_cut = float(cont_params.get("tail_auto_r_fraction", 0.7)) * float(r_cont[-1])
            tail_r_cut = float(tail_r_cut)
            idx_cut = int(np.searchsorted(r_cont, tail_r_cut))
            if 0 < idx_cut < r_cont.size - 1:
                try:
                    n_full_val, _ = apply_tail_match(
                        r_cont,
                        n_full_val,
                        tail_n0,
                        tail_mu_id,
                        config.temperature,
                        idx_cut,
                        fit_points=fit_points,
                        blend_points=blend_points,
                        model=tail_model,
                        auto_rel_improve_tol=tail_auto_rel_improve_tol,
                        auto_signal_rel_tol=tail_auto_signal_rel_tol,
                    )
                    if compute_external:
                        n_ext_val, _ = apply_tail_match(
                            r_cont,
                            n_ext_val,
                            tail_n0,
                            tail_mu_id,
                            config.temperature,
                            idx_cut,
                            fit_points=fit_points,
                            blend_points=blend_points,
                            model=tail_model,
                            auto_rel_improve_tol=tail_auto_rel_improve_tol,
                            auto_signal_rel_tol=tail_auto_signal_rel_tol,
                        )
                except Exception as exc:
                    if not tail_fallback_on_error:
                        raise
                    warnings.warn(
                        f"B3 tail match failed in inner-mu SCF at r_cut={tail_r_cut:.6f}; "
                        f"keeping original density. reason={exc}",
                        RuntimeWarning,
                    )

        use_ext_source_closure = bool(cont_params_ext.get("source_closure", False))
        if str(cont_params_ext.get("tail_mode", "off")).strip().lower() == "in_scf":
            use_ext_source_closure = use_ext_source_closure and bool(
                cont_params_ext.get("source_closure_when_b3", False)
            )
        if compute_external and use_ext_source_closure:
            r_trust_ext = cont_params_ext.get("source_r_trust", None)
            if r_trust_ext is None:
                r_trust_frac_ext = float(cont_params_ext.get("source_r_trust_frac", 0.7))
                r_trust_ext = r_trust_frac_ext * float(r_cont[-1])
            blend_w_ext = cont_params_ext.get("source_blend_width", None)
            if blend_w_ext is None:
                blend_frac_ext = float(cont_params_ext.get("source_blend_frac", 0.05))
                blend_w_ext = blend_frac_ext * float(r_cont[-1])
            n_ext_val = _close_continuum_source_to_n0(
                r_cont,
                n_ext_val,
                float(n0_next),
                float(r_trust_ext),
                float(blend_w_ext),
            )

        n_pa_val = n_full_val - n_ext_val
        n_scr_iter = n_pa_val - n_ion_val
        charge_ws_val = _electron_count(r_cont, n_full_val, r_ws)
        charge_bound_val = _electron_count(r_cont, n_bound_val, r_ws)
        charge_ion_val = _electron_count(r_cont, n_ion_val, r_ws)
        charge_cont_val = _electron_count(r_cont, n_cont_val, r_ws)
        charge_ext_val = _electron_count(r_cont, n_ext_val, r_ws)
        charge_pa_val = _electron_count(r_cont, n_pa_val, r_ws)
        charge_pa_full_val = _electron_count_full(r_cont, n_pa_val)
        charge_scr_val = _electron_count(r_cont, n_scr_iter, r_ws)
        zbar_val = charge_scr_val if compute_external else (float(config.Z) - float(charge_ion_val))
        charge_neutral_val = charge_pa_full_val if neutrality_mode == "pa" else charge_ws_val
        charge_rel_val = abs(charge_neutral_val - config.Z) / max(float(config.Z), 1e-12)
        if not compute_external:
            dzbar_rel_val = np.nan
        elif zbar_prev_current is None:
            dzbar_rel_val = np.nan
        else:
            denom = max(abs(zbar_val), 1e-12)
            dzbar_rel_val = abs(zbar_val - zbar_prev_current) / denom
        if perf_on:
            perf_local["assembly_diag"] = time.perf_counter() - t_stage_local

        return {
            "perf": perf_local,
            "cont_params": cont_params,
            "cont_params_ext": cont_params_ext,
            "cont_e_max_iter": float(cont_e_max_iter),
            "mu": float(mu_val),
            "mu_n_eval": int(mu_diag.get("mu_n_eval", 0)),
            "mu_n_iter": int(mu_diag.get("mu_n_iter", 0)),
            "n0": float(n0_next),
            "n_full": n_full_val,
            "n_full_pre_tail": np.asarray(eval_diag["n_full_pre_tail"], dtype=float),
            "n_bound": n_bound_val,
            "n_cont": n_cont_val,
            "n_cont_pre_tail": np.asarray(eval_diag["n_cont_pre_tail"], dtype=float),
            "n_cont_dft_raw": np.asarray(eval_diag["n_cont_dft_raw"], dtype=float),
            "n_cont_tail_meta": dict(eval_diag["n_cont_tail_meta"]),
            "n_ext": n_ext_val,
            "n_ion": n_ion_val,
            "n_pa": n_pa_val,
            "n_scr_iter": n_scr_iter,
            "charge_ws": float(charge_ws_val),
            "charge_bound": float(charge_bound_val),
            "charge_ion": float(charge_ion_val),
            "charge_cont": float(charge_cont_val),
            "charge_ext": float(charge_ext_val),
            "charge_pa": float(charge_pa_val),
            "charge_pa_full": float(charge_pa_full_val),
            "charge_scr": float(charge_scr_val),
            "charge_neutral": float(charge_neutral_val),
            "charge_rel": float(charge_rel_val),
            "zbar": float(zbar_val),
            "dzbar_rel": float(dzbar_rel_val) if np.isfinite(dzbar_rel_val) else np.nan,
            "ion_gamma": float(ion_gamma),
            "energy_cache_full": energy_cache_full,
            "energy_cache_ext": energy_cache_ext,
            "debug_bound_eigvals": (np.asarray(vals, dtype=float) if config.store_final_bound_debug else None),
            "debug_bound_eigvecs": (np.asarray(vecs, dtype=float) if config.store_final_bound_debug else None),
            "debug_energy_cut": (float(energy_cut) if config.store_final_bound_debug else np.nan),
            "debug_ion_gamma": (float(ion_gamma) if config.store_final_bound_debug else np.nan),
        }

    stage_e_max_mode = str(config.continuum_params.get("e_max_mode", "fixed")).lower().strip()
    e_min_stage = max(float(config.continuum_params.get("e_min", 1.0e-6)), 1.0e-12)
    if stage_e_max_mode in ("fixed", "off", "none"):
        cont_e_max_stage_floor = max(e_min_stage, float(config.continuum_params.get("e_max", e_min_stage)))
    else:
        cont_e_max_stage_floor = max(
            e_min_stage,
            float(config.continuum_params.get("e_max_floor", e_min_stage)),
        )
    energy_cache_full_last = None

    for it in range(int(scf_iters)):
        perf = {}
        attempt_state = None
        iter_e_max_floor = float(cont_e_max_stage_floor)
        for retry_idx in range(_CONT_E_MAX_RETRY_MAX_TRIES):
            attempt_state = _run_iteration_attempt(
                it=it,
                mu_guess_val=float(mu),
                n0_current=float(n0),
                v_full_current=v_full,
                v_ext_current=v_ext,
                zbar_prev_current=zbar_prev,
                stage_e_max_floor=float(iter_e_max_floor),
            )
            _merge_perf_sum(perf, attempt_state["perf"])
            charge_rel_try = float(attempt_state["charge_rel"])
            if charge_rel_try <= _CONT_E_MAX_RETRY_CHARGE_REL_TOL:
                break
            if retry_idx == _CONT_E_MAX_RETRY_MAX_TRIES - 1:
                if config.verbose:
                    print(
                        "      [e_max-retry] "
                        f"iter={it:3d}, charge_rel={charge_rel_try:.3e}; "
                        "reached retry limit and accepting this SCF iteration"
                    )
                break
            next_e_max = float(attempt_state["cont_e_max_iter"]) + _CONT_E_MAX_RETRY_STEP_HA
            if config.verbose:
                print(
                    "      [e_max-retry] "
                    f"iter={it:3d}, charge_rel={charge_rel_try:.3e}; "
                    f"retrying same SCF iteration with cont_e_max={next_e_max:.6f} Ha"
                )
            iter_e_max_floor = float(next_e_max)
        if attempt_state is None:
            raise RuntimeError("SCF iteration did not produce an attempt state.")

        cont_e_max_stage_floor = max(cont_e_max_stage_floor, float(attempt_state["cont_e_max_iter"]))
        mu = float(attempt_state["mu"])
        n0 = float(attempt_state["n0"])
        n_full = np.asarray(attempt_state["n_full"], dtype=float)
        n_full_pre_tail = np.asarray(attempt_state["n_full_pre_tail"], dtype=float)
        n_bound = np.asarray(attempt_state["n_bound"], dtype=float)
        n_cont = np.asarray(attempt_state["n_cont"], dtype=float)
        n_cont_pre_tail = np.asarray(attempt_state["n_cont_pre_tail"], dtype=float)
        n_cont_dft_raw = np.asarray(attempt_state["n_cont_dft_raw"], dtype=float)
        n_cont_tail_meta = dict(attempt_state["n_cont_tail_meta"])
        n_ext = np.asarray(attempt_state["n_ext"], dtype=float)
        n_ion = np.asarray(attempt_state["n_ion"], dtype=float)
        n_pa = np.asarray(attempt_state["n_pa"], dtype=float)
        charge_ws = float(attempt_state["charge_ws"])
        charge_bound = float(attempt_state["charge_bound"])
        charge_ion = float(attempt_state["charge_ion"])
        charge_cont = float(attempt_state["charge_cont"])
        charge_ext = float(attempt_state["charge_ext"])
        charge_pa = float(attempt_state["charge_pa"])
        charge_pa_full = float(attempt_state["charge_pa_full"])
        charge_scr = float(attempt_state["charge_scr"])
        charge_neutral = float(attempt_state["charge_neutral"])
        charge_rel = float(attempt_state["charge_rel"])
        zbar = float(attempt_state["zbar"])
        dzbar_rel = float(attempt_state["dzbar_rel"])
        cont_e_max_iter = float(attempt_state["cont_e_max_iter"])
        cont_params = dict(attempt_state["cont_params"])
        cont_params_ext = dict(attempt_state["cont_params_ext"])
        energy_cache_full_last = attempt_state["energy_cache_full"]

        if v_full_prev is None:
            v_full_prev = v_full.copy()

        t_stage = time.perf_counter() if perf_on else 0.0

        # (15) Build new effective potentials from current densities.
        # n_full_source is the density used only for potential assembly.
        # Diagnostics still keep the raw n_full/n_cont unchanged.
        n_cont_source = n_cont
        n_full_source = n_full
        use_full_source_closure = bool(cont_params.get("source_closure", False))
        if str(cont_params.get("tail_mode", "off")).strip().lower() == "in_scf":
            use_full_source_closure = use_full_source_closure and bool(
                cont_params.get("source_closure_when_b3", True)
            )
        if use_full_source_closure:
            r_trust = cont_params.get("source_r_trust", None)
            if r_trust is None:
                r_trust_frac = float(cont_params.get("source_r_trust_frac", 0.7))
                r_trust = r_trust_frac * float(r_cont[-1])
            blend_w = cont_params.get("source_blend_width", None)
            if blend_w is None:
                blend_frac = float(cont_params.get("source_blend_frac", 0.05))
                blend_w = blend_frac * float(r_cont[-1])
            # Close outer continuum tail smoothly to n0 beyond r_trust.
            n_cont_source = _close_continuum_source_to_n0(
                r_cont,
                n_cont,
                float(n0),
                float(r_trust),
                float(blend_w),
            )
            if bool(cont_params.get("source_charge_closure", False)):
                # Optional global charge fix on the outer region only.
                n_cont_source, source_closure_meta = _enforce_source_charge_closure(
                    r_cont,
                    n_bound,
                    n_cont_source,
                    float(n0),
                    g_ii,
                    float(config.Z),
                    float(r_trust),
                )
            else:
                source_closure_meta = {
                    "applied": False,
                    "reason": "disabled",
                    "r_trust": float(r_trust),
                }
            n_full_source = n_bound + n_cont_source

        kappa_eff = config.ph_kappa
        if config.ph_kappa_iters is not None and it >= int(config.ph_kappa_iters):
            kappa_eff = 0.0
        v_full_new = effective_potential_full(
            r_cont,
            n_full_source,
            n0,
            g_ii,
            config.Z,
            xc_model=config.xc_model,
            kappa=kappa_eff,
        )
        if shift_tail:
            v_shift_full = _tail_shift_value(
                r_cont,
                v_full_new,
                config.v_tail_fraction,
                config.v_tail_mode,
            )
            v_full_new = v_full_new - v_shift_full
        if outer_decay:
            v_full_new = _apply_outer_v_eff_decay(
                r_cont,
                v_full_new,
                r_ws=r_ws,
                enabled=True,
                start_rws=float(config.full_v_eff_outer_decay_start_rws),
                decay_length_rws=float(config.full_v_eff_outer_decay_length_rws),
            )
        if compute_external:
            v_ext_new = effective_potential_external(
                r_cont,
                n_ext,
                n0,
                g_ii,
                xc_model=config.xc_model,
                kappa=kappa_eff,
            )
            if shift_tail:
                v_shift_ext = _tail_shift_value(
                    r_cont,
                    v_ext_new,
                    config.v_tail_fraction,
                    config.v_tail_mode,
                )
                v_ext_new = v_ext_new - v_shift_ext
        else:
            v_ext_new = v_ext
        if perf_on:
            perf["potential"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()

        # (15) Mix V_eff (linear or Eyert) to stabilize SCF.
        if mixing_scheme == "linear":
            v_full = mix * v_full_new + (1.0 - mix) * v_full
        else:
            # Eyert SCF (Eq. 59-63): work with x = V_eff * r / Z
            x_in = v_full * r_cont / float(config.Z)
            x_out = v_full_new * r_cont / float(config.Z)
            f_now = x_out - x_in

            if x_prev is not None and f_prev is not None:
                dx_hist.append(x_in - x_prev)
                df_hist.append(f_now - f_prev)

            if dx_hist:
                hist_len = len(dx_hist)
                a_mat = np.zeros((hist_len, hist_len), dtype=float)
                b_vec = np.zeros(hist_len, dtype=float)
                for i in range(hist_len):
                    for j in range(hist_len):
                        prod = df_hist[i] * df_hist[j]
                        a_mat[i, j] = _trapz(prod, r_cont)
                        if i == j:
                            a_mat[i, j] += mixing_w0 ** 2
                    b_vec[i] = _trapz(df_hist[i] * f_now, r_cont)
                try:
                    w_vec = np.linalg.solve(a_mat, b_vec)
                except np.linalg.LinAlgError:
                    w_vec = None
                if w_vec is not None:
                    corr = np.zeros_like(x_in)
                    for i in range(hist_len):
                        corr = corr + w_vec[i] * (dx_hist[i] + mix * df_hist[i])
                    x_next = x_in + mix * f_now - corr
                else:
                    x_next = x_in + mix * f_now
            else:
                x_next = x_in + mix * f_now

            v_full = np.where(r_cont > 0.0, x_next * float(config.Z) / r_cont, v_full_new)
            x_prev = x_in
            f_prev = f_now
        if shift_tail:
            v_full = v_full - _tail_shift_value(
                r_cont,
                v_full,
                config.v_tail_fraction,
                config.v_tail_mode,
            )
        if compute_external:
            v_ext = mix * v_ext_new + (1.0 - mix) * v_ext
            if shift_tail:
                v_ext = v_ext - _tail_shift_value(
                    r_cont,
                    v_ext,
                    config.v_tail_fraction,
                    config.v_tail_mode,
                )
        if perf_on:
            perf["mix"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()

        # (16) Optionally store last-N SCF snapshots for diagnostics.
        if scf_snapshots is not None:
            scf_snapshots.append({
                "iter": int(it),
                "n_full": n_full.copy(),
                "v_full": v_full.copy(),
            })

        # (17) Convergence metrics (relative Δn and Δv).
        dn_rel = np.nan
        dv_rel = np.nan
        if n_full_prev is not None:
            dn_num = _trapz(np.abs(n_full - n_full_prev), r_cont)
            dn_den = _trapz(np.abs(n_full_prev), r_cont)
            dn_rel = float(dn_num / max(dn_den, 1e-12))
        if v_full_prev is not None:
            dv_num = _trapz(np.abs(v_full - v_full_prev), r_cont)
            dv_den = _trapz(np.abs(v_full_prev), r_cont)
            dv_rel = float(dv_num / max(dv_den, 1e-12))
        # Gauge-aligned map residual (ignore constant potential offset mode).
        err_full = _gauge_aligned_map_error(
            r_cont,
            v_full_new,
            v_full,
            config.v_tail_fraction,
            config.v_tail_mode,
        )
        err_ext = 0.0
        if compute_external:
            err_ext = _gauge_aligned_map_error(
                r_cont,
                v_ext_new,
                v_ext,
                config.v_tail_fraction,
                config.v_tail_mode,
            )
        err = max(err_full, err_ext)
        if perf_on:
            perf["metrics"] = time.perf_counter() - t_stage
            perf["total"] = sum(
                float(perf.get(key, 0.0))
                for key in (
                    "basis_build",
                    "bound_solve",
                    "mu_solve",
                    "assembly_diag",
                    "potential",
                    "mix",
                    "metrics",
                )
            )

        # (18) Optional iteration printout.
        if config.verbose and (it % config.print_every == 0):
            if compute_external:
                print(
                    f"  {it:3d} | {dn_rel:7.3e} | {dv_rel:7.3e} | "
                    f"{charge_ws:7.3f} | {charge_bound:7.3f} | {charge_cont:7.3f} | "
                    f"{zbar:7.3f} | {charge_rel:9.3e} | {dzbar_rel:7.3e}"
                )
                print(f"      cont_e_max={float(cont_e_max_iter):.6f} Ha")
                if perf_on and (it % perf_every == 0):
                    if perf_show_stage:
                        print(
                            "      perf[s]: "
                            f"basis={perf.get('basis_build', 0.0):.3f}, "
                            f"bound={perf.get('bound_solve', 0.0):.3f}, "
                            f"precompute={perf.get('precompute', 0.0):.3f}, "
                            f"mu_solve={perf.get('mu_solve', 0.0):.3f}, "
                            f"assembly={perf.get('assembly_diag', 0.0):.3f}, "
                            f"potential={perf.get('potential', 0.0):.3f}, "
                            f"mix={perf.get('mix', 0.0):.3f}, "
                            f"metrics={perf.get('metrics', 0.0):.3f}, "
                            f"total={perf.get('total', 0.0):.3f}"
                        )
                    if perf_show_basis and np.isfinite(perf.get("basis_n_eval", np.nan)):
                        print(
                            "      perf[basis]: "
                            f"n_eval={_fmt_perf_int(perf.get('basis_n_eval', np.nan))}, "
                            f"n_cache_hits={_fmt_perf_int(perf.get('basis_n_cache_hits', np.nan))}, "
                            f"n_e_basis={_fmt_perf_int(perf.get('basis_n_e', np.nan))}, "
                            f"n_base_per_shard={_fmt_perf_int(perf.get('basis_n_base_per_shard', np.nan))}, "
                            f"shard_cache_merged={_fmt_perf_int(perf.get('basis_shard_cache_merged', np.nan))}, "
                            f"n_windows={_fmt_perf_int(perf.get('basis_n_windows', np.nan))}"
                        )
                    if perf_show_basis and np.isfinite(perf.get("basis_ext_n_eval", np.nan)):
                        print(
                            "      perf[basis_ext]: "
                            f"n_eval={_fmt_perf_int(perf.get('basis_ext_n_eval', np.nan))}, "
                            f"n_cache_hits={_fmt_perf_int(perf.get('basis_ext_n_cache_hits', np.nan))}, "
                            f"n_e_basis={_fmt_perf_int(perf.get('basis_ext_n_e', np.nan))}"
                        )
            else:
                print(
                    f"  {it:3d} | {dn_rel:7.3e} | {dv_rel:7.3e} | "
                    f"{charge_ws:7.3f} | {charge_bound:7.3f} | {charge_cont:7.3f} | "
                    f"{zbar:7.3f} | {charge_rel:9.3e}"
                )
                print(f"      cont_e_max={float(cont_e_max_iter):.6f} Ha")
                if perf_on and (it % perf_every == 0):
                    if perf_show_stage:
                        print(
                            "      perf[s]: "
                            f"basis={perf.get('basis_build', 0.0):.3f}, "
                            f"bound={perf.get('bound_solve', 0.0):.3f}, "
                            f"precompute={perf.get('precompute', 0.0):.3f}, "
                            f"mu_solve={perf.get('mu_solve', 0.0):.3f}, "
                            f"assembly={perf.get('assembly_diag', 0.0):.3f}, "
                            f"potential={perf.get('potential', 0.0):.3f}, "
                            f"mix={perf.get('mix', 0.0):.3f}, "
                            f"metrics={perf.get('metrics', 0.0):.3f}, "
                            f"total={perf.get('total', 0.0):.3f}"
                        )
                    if perf_show_basis and np.isfinite(perf.get("basis_n_eval", np.nan)):
                        print(
                            "      perf[basis]: "
                            f"n_eval={_fmt_perf_int(perf.get('basis_n_eval', np.nan))}, "
                            f"n_cache_hits={_fmt_perf_int(perf.get('basis_n_cache_hits', np.nan))}, "
                            f"n_e_basis={_fmt_perf_int(perf.get('basis_n_e', np.nan))}, "
                            f"n_base_per_shard={_fmt_perf_int(perf.get('basis_n_base_per_shard', np.nan))}, "
                            f"shard_cache_merged={_fmt_perf_int(perf.get('basis_shard_cache_merged', np.nan))}, "
                            f"n_windows={_fmt_perf_int(perf.get('basis_n_windows', np.nan))}"
                        )
                    if perf_show_basis and np.isfinite(perf.get("basis_ext_n_eval", np.nan)):
                        print(
                            "      perf[basis_ext]: "
                            f"n_eval={_fmt_perf_int(perf.get('basis_ext_n_eval', np.nan))}, "
                            f"n_cache_hits={_fmt_perf_int(perf.get('basis_ext_n_cache_hits', np.nan))}, "
                            f"n_e_basis={_fmt_perf_int(perf.get('basis_ext_n_e', np.nan))}"
                        )
                # Tail diagnostic: monitor continuum level at B3 cut relative to n0.
                tail_r_cut_diag = cont_params.get("tail_r_cut", None)
                if tail_r_cut_diag is not None and n_cont.size == r_cont.size:
                    idx_tail = int(np.searchsorted(r_cont, float(tail_r_cut_diag)))
                    idx_tail = max(0, min(idx_tail, r_cont.size - 1))
                    delta_tail = float(n_cont[idx_tail] - n0)
                    print(
                        f"      tail_diag: r_cut={float(tail_r_cut_diag):.3f}, "
                        f"n_cont(r_cut)-n0={delta_tail:.3e}"
                    )
                tail_model_req = str(n_cont_tail_meta.get("model_requested", "")).strip().lower()
                if tail_model_req == "auto":
                    tail_model_sel = str(n_cont_tail_meta.get("model_selected", "n/a"))
                    fit_rel_improve = n_cont_tail_meta.get("fit_rel_improve_full", np.nan)
                    fit_signal_max = n_cont_tail_meta.get("fit_signal_max", np.nan)
                    fit_signal_threshold = n_cont_tail_meta.get("fit_signal_threshold", np.nan)
                    print(
                        "      tail_auto: "
                        f"selected={tail_model_sel}, "
                        f"rel_improve={float(fit_rel_improve):.3e}, "
                        f"signal={float(fit_signal_max):.3e}, "
                        f"threshold={float(fit_signal_threshold):.3e}"
                    )

        # (19) Store SCF history snapshot.
        history.append({
            "iter": int(it),
            "mu": float(mu),
            "mu_n_eval": int(attempt_state.get("mu_n_eval", 0)),
            "mu_n_iter": int(attempt_state.get("mu_n_iter", 0)),
            "cont_e_max": float(cont_e_max_iter),
            "dn_rel": dn_rel,
            "dv_rel": dv_rel,
            "err": float(err),
            "charge_ws": float(charge_ws),
            "charge_bound": float(charge_bound),
            "charge_ion": float(charge_ion),
            "charge_cont": float(charge_cont),
            "charge_rel": float(charge_rel),
            "zbar": float(zbar) if compute_external else np.nan,
            "dzbar": float(dzbar_rel) if compute_external else np.nan,
            "ion_gamma": float(attempt_state.get("ion_gamma", np.nan)),
            "n_cont_tail_model_requested": str(n_cont_tail_meta.get("model_requested", "")),
            "n_cont_tail_model_selected": str(n_cont_tail_meta.get("model_selected", "")),
            "perf": perf if perf_on else None,
        })

        # (20) Convergence decision (both Δn and Δv below tolerances).
        dn_ok = False
        if config.dn_tol is not None:
            dn_ok = np.isfinite(dn_rel) and dn_rel < config.dn_tol
        dv_ok = False
        if config.dv_tol is not None:
            dv_ok = np.isfinite(dv_rel) and dv_rel < config.dv_tol
        err_ok = np.isfinite(err) and (err < float(config.tol))
        # Keep at least one unscreened update after the PH (Piron) stage.
        in_ph_stage = (
            config.ph_kappa_iters is not None
            and it < int(config.ph_kappa_iters)
        )
        if (not in_ph_stage) and dn_ok and dv_ok and err_ok:
            scf_converged = True
            break

        n_full_prev = n_full.copy()
        v_full_prev = v_full.copy()
        n_ext_prev = n_ext.copy()
        zbar_prev = zbar

    # Final refresh on the returned mixed potential.
    #
    # The SCF loop evaluates densities on the pre-mix potential and then mixes
    # V_eff. Without one last refresh, the returned ``v_full`` and
    # ``n_bound/n_ion/n_full`` correspond to slightly different fixed-point
    # iterates, which is especially visible in post-SCF WS charge diagnostics.
    final_state = _run_iteration_attempt(
        it=int(len(history)),
        mu_guess_val=float(mu),
        n0_current=float(n0),
        v_full_current=v_full,
        v_ext_current=v_ext,
        zbar_prev_current=zbar_prev,
        stage_e_max_floor=float(cont_e_max_stage_floor),
    )
    mu = float(final_state["mu"])
    n0 = float(final_state["n0"])
    n_full = np.asarray(final_state["n_full"], dtype=float)
    n_full_pre_tail = np.asarray(final_state["n_full_pre_tail"], dtype=float)
    n_bound = np.asarray(final_state["n_bound"], dtype=float)
    n_cont = np.asarray(final_state["n_cont"], dtype=float)
    n_cont_pre_tail = np.asarray(final_state["n_cont_pre_tail"], dtype=float)
    n_cont_dft_raw = np.asarray(final_state["n_cont_dft_raw"], dtype=float)
    n_cont_tail_meta = dict(final_state["n_cont_tail_meta"])
    n_ext = np.asarray(final_state["n_ext"], dtype=float)
    n_ion = np.asarray(final_state["n_ion"], dtype=float)
    n_pa = np.asarray(final_state["n_pa"], dtype=float)
    charge_ws = float(final_state["charge_ws"])
    charge_bound = float(final_state["charge_bound"])
    charge_ion = float(final_state["charge_ion"])
    charge_cont = float(final_state["charge_cont"])
    charge_ext = float(final_state["charge_ext"])
    charge_pa = float(final_state["charge_pa"])
    charge_pa_full = float(final_state["charge_pa_full"])
    charge_scr = float(final_state["charge_scr"])
    charge_neutral = float(final_state["charge_neutral"])
    charge_rel = float(final_state["charge_rel"])
    zbar = float(final_state["zbar"])
    dzbar_rel = float(final_state["dzbar_rel"])
    cont_e_max_iter = float(final_state["cont_e_max_iter"])
    cont_params = dict(final_state["cont_params"])
    cont_params_ext = dict(final_state["cont_params_ext"])
    energy_cache_full_last = final_state["energy_cache_full"]

    n_scr = n_pa - n_ion
    cont_phase_energy = None
    cont_phase_shift = None
    if energy_cache_full_last is not None and len(energy_cache_full_last) >= 2:
        e_sorted = np.array(sorted(float(e) for e in energy_cache_full_last.keys()), dtype=float)
        delta_rows = [np.asarray(energy_cache_full_last[float(e)][1], dtype=float) for e in e_sorted]
        if delta_rows:
            cont_phase_energy = e_sorted
            cont_phase_shift = np.vstack(delta_rows)
    result = {
        "Z": float(config.Z),
        "r": r_cont,
        "r_bound": r_bound,
        "r_cont": r_cont,
        "g_ii": g_ii,
        "n0": float(n0),
        "n_bound": n_bound,
        "n_ion": n_ion,
        "n_cont": n_cont,
        "n_cont_pre_tail": n_cont_pre_tail,
        "n_cont_dft_raw": n_cont_dft_raw,
        "n_cont_tail_meta": n_cont_tail_meta,
        "n_cont_source": n_cont_source,
        "n_full": n_full,
        "n_full_pre_tail": n_full_pre_tail,
        "n_full_source": n_full_source,
        "source_closure_meta": source_closure_meta,
        "n_ext": n_ext,
        "n_pa": n_pa,
        "n_scr": n_scr,
        "v_full": v_full,
        "v_ext": v_ext,
        "history": history,
        "converged": bool(scf_converged),
        "iters": int(len(history)),
        "r_ws": float(r_ws),
        "mu": float(mu),
        "charge_ws": float(charge_ws),
        "charge_neutral": float(charge_neutral),
        "neutrality_mode": neutrality_mode,
        "zbar": float(zbar),
        "ion_gamma": float(final_state.get("ion_gamma", np.nan)),
    }
    if cont_phase_energy is not None and cont_phase_shift is not None:
        result["cont_phase_energy_ha"] = cont_phase_energy
        result["cont_phase_shift_rad"] = cont_phase_shift
    if config.store_final_bound_debug:
        result["debug_bound_eigvals"] = np.asarray(final_state["debug_bound_eigvals"], dtype=float)
        result["debug_bound_eigvecs"] = np.asarray(final_state["debug_bound_eigvecs"], dtype=float)
        result["debug_energy_cut"] = float(final_state["debug_energy_cut"])
        result["debug_ion_gamma"] = float(final_state["debug_ion_gamma"])
    if scf_snapshots is not None:
        result["scf_snapshots"] = list(scf_snapshots)
    return result

def solve_ks_dft_is(config: KSDTFConfig) -> Dict[str, Any]:
    """
    Solve a minimal IS quantum AA problem with fixed or neutralized mu.

    Returns
    -------
    dict
        Dictionary containing grids, densities, potentials, and diagnostics.
    """
    mu_mode = config.mu_mode.lower().strip()
    if mu_mode not in ("fixed", "neutral"):
        raise ValueError("mu_mode must be 'fixed' or 'neutral'.")

    if mu_mode == "fixed":
        # Respect optional restart potentials provided in config.
        return _scf_fixed_mu(
            config,
            config.mu,
            v_full_init=config.v_full_init,
            v_ext_init=config.v_ext_init,
        )
    mu_strategy = str(getattr(config, "mu_strategy", "inner")).lower().strip()
    if mu_strategy not in ("inner", "outer"):
        raise ValueError("mu_strategy must be 'inner' or 'outer'.")
    if mu_strategy == "inner":
        # Inner-mu path also supports restart from supplied potentials.
        return _scf_neutral_inner(
            config,
            v_full_init=config.v_full_init,
            v_ext_init=config.v_ext_init,
        )

    mu_solver = str(config.mu_solver).lower().strip()
    if mu_solver not in ("bracket", "brent", "secant"):
        raise ValueError("mu_solver must be 'bracket', 'brent', or 'secant'.")

    mu_lo, mu_hi = config.mu_bounds
    mu_lo = float(mu_lo)
    mu_hi = float(mu_hi)
    if mu_lo >= mu_hi:
        raise ValueError("mu_bounds must be (mu_min, mu_max) with mu_min < mu_max.")
    mu_bounds_strict = bool(config.mu_bounds_strict)
    mu_min_bound = float(mu_lo)
    mu_max_bound = float(mu_hi)

    v_full = None if config.v_full_init is None else np.asarray(config.v_full_init, dtype=float).copy()
    v_ext = None if config.v_ext_init is None else np.asarray(config.v_ext_init, dtype=float).copy()
    mu_scf_iter = config.mu_scf_iter
    if mu_scf_iter is None or mu_scf_iter <= 0:
        mu_scf_iter = int(config.max_iter)
    mu_zbar_min = config.mu_zbar_min
    if mu_zbar_min is not None:
        mu_zbar_min = float(mu_zbar_min)
        if mu_zbar_min < 0.0:
            raise ValueError("mu_zbar_min must be non-negative.")

    def _eval_mu(mu_val: float,
                 v_full_init: np.ndarray | None,
                 v_ext_init: np.ndarray | None) -> tuple[Dict[str, Any], float]:
        result = _scf_fixed_mu(config, mu_val, v_full_init=v_full_init, v_ext_init=v_ext_init, max_iter=mu_scf_iter)
        charge_err = result.get("charge_neutral", result["charge_ws"]) - config.Z
        return result, float(charge_err)

    def _zbar_ok(result: Dict[str, Any]) -> bool:
        if mu_zbar_min is None:
            return True
        return float(result.get("zbar", 0.0)) >= mu_zbar_min

    def _f_ws(result: Dict[str, Any]) -> float:
        return float(result["charge_ws"]) - config.Z

    if mu_solver == "secant":
        mu_min_bound = float(mu_lo)
        mu_max_bound = float(mu_hi)
        mu0 = float(config.mu)
        if mu0 <= mu_min_bound or mu0 >= mu_max_bound:
            raise ValueError("mu (initial guess) must lie within mu_bounds for secant solver.")
        step = float(config.mu_bracket_step)
        mu1 = mu0 + step
        if mu1 <= mu_min_bound or mu1 >= mu_max_bound:
            mu1 = mu0 - step
        if mu1 <= mu_min_bound or mu1 >= mu_max_bound:
            raise ValueError("mu_bounds too tight for secant step; widen mu_bounds or reduce mu_bracket_step.")

        result0, charge0 = _eval_mu(mu0, v_full, v_ext)
        v_full = result0["v_full"]
        v_ext = result0["v_ext"]
        if not _zbar_ok(result0):
            raise ValueError("mu guess yields zbar below mu_zbar_min; lower mu_zbar_min or adjust mu.")

        result1, charge1 = _eval_mu(mu1, v_full, v_ext)
        v_full = result1["v_full"]
        v_ext = result1["v_ext"]
        if not _zbar_ok(result1):
            raise ValueError("secant initial step yields zbar below mu_zbar_min; reduce mu_bracket_step or lower mu_zbar_min.")

        mu_history = []
        for _ in range(int(config.mu_max_iter)):
            mu_history.append({
                "mu": float(mu1),
                "charge_ws": float(result1["charge_ws"]),
                "charge_neutral": float(result1.get("charge_neutral", result1["charge_ws"])),
                "charge_err": float(charge1),
                "f_ws": _f_ws(result1),
            })
            if config.verbose:
                if result1.get("neutrality_mode", "ws") == "ws":
                    print(f"[mu iter] mu={mu1:.4f}, charge_err={charge1:.3e}")
                else:
                    f_ws = _f_ws(result1)
                    print(f"[mu iter] mu={mu1:.4f}, charge_err={charge1:.3e}, f_ws={f_ws:.3e}")
            if abs(charge1) < config.mu_tol:
                result1["mu_history"] = mu_history
                return result1

            denom = charge1 - charge0
            if abs(denom) < 1e-12:
                step_dir = -np.sign(charge1) if charge1 != 0.0 else 1.0
                step_size = max(abs(mu1 - mu0) * 0.5, 0.1)
                mu_new = mu1 + step_dir * step_size
            else:
                mu_new = mu1 - charge1 * (mu1 - mu0) / denom

            max_step = max(abs(mu1 - mu0), abs(config.mu_bracket_step))
            delta = mu_new - mu1
            if abs(delta) > max_step:
                if config.verbose:
                    print("  [mu] Secant step limited to stay near initial guess.")
                mu_new = mu1 + np.sign(delta) * max_step

            if not np.isfinite(mu_new):
                raise ValueError("secant update produced non-finite mu; adjust mu_bounds or mu_bracket_step.")
            if mu_new <= mu_min_bound or mu_new >= mu_max_bound:
                if config.verbose:
                    print("  [mu] Secant step hit mu_bounds; clamping.")
                eps = 1e-6 * (mu_max_bound - mu_min_bound)
                mu_new = min(max(mu_new, mu_min_bound + eps), mu_max_bound - eps)

            result_new, charge_new = _eval_mu(mu_new, v_full, v_ext)
            v_full = result_new["v_full"]
            v_ext = result_new["v_ext"]
            if not _zbar_ok(result_new):
                raise ValueError("secant iter yielded zbar below mu_zbar_min; lower mu_zbar_min or adjust mu_bounds.")

            mu0, charge0 = mu1, charge1
            mu1, charge1 = mu_new, charge_new
            result1 = result_new

        result1["mu_history"] = mu_history
        return result1

    result_lo, charge_lo = _eval_mu(mu_lo, v_full, v_ext)
    v_full = result_lo["v_full"]
    v_ext = result_lo["v_ext"]

    result_hi, charge_hi = _eval_mu(mu_hi, v_full, v_ext)
    v_full = result_hi["v_full"]
    v_ext = result_hi["v_ext"]

    if mu_zbar_min is not None:
        step = float(config.mu_bracket_step)
        max_iter = int(config.mu_bracket_max_iter)
        for _ in range(max_iter + 1):
            if _zbar_ok(result_lo):
                break
            mu_lo += step
            if mu_bounds_strict and mu_lo >= mu_max_bound:
                raise ValueError("mu_bounds_strict prevents reaching zbar_min; increase mu_hi.")
            if mu_lo >= mu_hi:
                raise ValueError("mu_bounds do not reach zbar_min; increase mu_hi or lower mu_zbar_min.")
            result_lo, charge_lo = _eval_mu(mu_lo, v_full, v_ext)
            v_full = result_lo["v_full"]
            v_ext = result_lo["v_ext"]
        if not _zbar_ok(result_lo):
            raise ValueError("mu_bounds do not reach zbar_min; increase mu_hi or lower mu_zbar_min.")
        if not _zbar_ok(result_hi):
            if mu_solver == "brent":
                raise ValueError("mu_hi yields zbar below mu_zbar_min; increase mu_hi or lower mu_zbar_min.")
            if config.verbose:
                print("  [mu] mu_hi yields Zbar below mu_zbar_min; continuing bracket.")

    if config.verbose:
        if result_lo.get("neutrality_mode", "ws") == "ws":
            print(f"[mu bracket] mu_lo={mu_lo:.3f}, charge_err={charge_lo:.3e}")
            print(f"[mu bracket] mu_hi={mu_hi:.3f}, charge_err={charge_hi:.3e}")
        else:
            print(f"[mu bracket] mu_lo={mu_lo:.3f}, charge_err={charge_lo:.3e}, f_ws={_f_ws(result_lo):.3e}")
            print(f"[mu bracket] mu_hi={mu_hi:.3f}, charge_err={charge_hi:.3e}, f_ws={_f_ws(result_hi):.3e}")

    if charge_lo == 0.0:
        return result_lo
    if charge_hi == 0.0:
        return result_hi

    if charge_lo * charge_hi > 0.0:
        if mu_bounds_strict:
            raise ValueError("mu_bounds do not bracket neutrality within strict bounds.")
        step = float(config.mu_bracket_step)
        max_iter = int(config.mu_bracket_max_iter)
        n0_floor = float(config.mu_n0_floor)
        blocked_by_zbar = False
        last_valid = (mu_lo, result_lo, charge_lo)
        for _ in range(max_iter):
            if charge_lo > 0.0 and charge_hi > 0.0:
                mu_lo -= step
                result_lo, charge_lo = _eval_mu(mu_lo, v_full, v_ext)
                v_full = result_lo["v_full"]
                v_ext = result_lo["v_ext"]
                n0_lo = ideal_unbound_density(mu_lo, config.temperature)
                if not _zbar_ok(result_lo):
                    mu_lo, result_lo, charge_lo = last_valid
                    blocked_by_zbar = True
                    break
                last_valid = (mu_lo, result_lo, charge_lo)
                if config.verbose:
                    if result_lo.get("neutrality_mode", "ws") == "ws":
                        print(f"[mu expand] mu_lo={mu_lo:.3f}, charge_err={charge_lo:.3e}")
                    else:
                        print(f"[mu expand] mu_lo={mu_lo:.3f}, charge_err={charge_lo:.3e}, f_ws={_f_ws(result_lo):.3e}")
                if n0_lo < n0_floor and charge_lo > 0.0:
                    break
            elif charge_lo < 0.0 and charge_hi < 0.0:
                mu_hi += step
                result_hi, charge_hi = _eval_mu(mu_hi, v_full, v_ext)
                v_full = result_hi["v_full"]
                v_ext = result_hi["v_ext"]
                if not _zbar_ok(result_hi):
                    blocked_by_zbar = True
                    break
                if config.verbose:
                    if result_hi.get("neutrality_mode", "ws") == "ws":
                        print(f"[mu expand] mu_hi={mu_hi:.3f}, charge_err={charge_hi:.3e}")
                    else:
                        print(f"[mu expand] mu_hi={mu_hi:.3f}, charge_err={charge_hi:.3e}, f_ws={_f_ws(result_hi):.3e}")
            else:
                break
            if charge_lo * charge_hi <= 0.0:
                break
            step *= 1.5

    if charge_lo * charge_hi > 0.0:
        if mu_zbar_min is not None and blocked_by_zbar:
            raise ValueError("mu_bounds do not bracket neutrality within zbar_min constraint.")
        raise ValueError("mu_bounds do not bracket neutrality after expansion.")

    if mu_solver == "brent":
        from scipy.optimize import brentq

        mu_history = []
        cache: Dict[float, tuple[Dict[str, Any], float]] = {}
        cache[float(mu_lo)] = (result_lo, charge_lo)
        cache[float(mu_hi)] = (result_hi, charge_hi)

        def f(mu_val: float) -> float:
            nonlocal v_full, v_ext
            mu_key = float(mu_val)
            if mu_key in cache:
                result, charge = cache[mu_key]
                v_full = result["v_full"]
                v_ext = result["v_ext"]
            else:
                result, charge = _eval_mu(mu_key, v_full, v_ext)
                v_full = result["v_full"]
                v_ext = result["v_ext"]
                cache[mu_key] = (result, charge)
            if not _zbar_ok(result):
                raise ValueError("brent iter yielded zbar below mu_zbar_min; adjust mu_bounds or mu_zbar_min.")
            mu_history.append({
                "mu": float(mu_key),
                "charge_ws": float(result["charge_ws"]),
                "charge_neutral": float(result.get("charge_neutral", result["charge_ws"])),
                "charge_err": float(charge),
                "f_ws": _f_ws(result),
            })
            if config.verbose:
                if result.get("neutrality_mode", "ws") == "ws":
                    print(f"[mu iter] mu={mu_key:.4f}, charge_err={charge:.3e}")
                else:
                    f_ws = _f_ws(result)
                    print(f"[mu iter] mu={mu_key:.4f}, charge_err={charge:.3e}, f_ws={f_ws:.3e}")
            return float(charge)

        mu_root = brentq(
            f,
            mu_lo,
            mu_hi,
            xtol=1e-6,
            rtol=1e-6,
            maxiter=int(config.mu_max_iter),
        )
        result_root, charge_root = cache.get(float(mu_root), _eval_mu(mu_root, v_full, v_ext))
        if abs(charge_root) > config.mu_tol and config.verbose:
            print(f"  [mu] Brent root charge_err={charge_root:.3e} exceeds mu_tol={config.mu_tol:.3e}")
        result_root["mu_history"] = mu_history
        return result_root

    mu_history = []
    result_mid = None
    for _ in range(int(config.mu_max_iter)):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        result_mid = _scf_fixed_mu(config, mu_mid, v_full_init=v_full, v_ext_init=v_ext, max_iter=mu_scf_iter)
        charge_mid = result_mid.get("charge_neutral", result_mid["charge_ws"]) - config.Z
        if not _zbar_ok(result_mid):
            raise ValueError("mu bisection entered zbar below mu_zbar_min; lower mu_zbar_min or adjust mu_bracket_step.")

        mu_history.append({
            "mu": float(mu_mid),
            "charge_ws": float(result_mid["charge_ws"]),
            "charge_neutral": float(result_mid.get("charge_neutral", result_mid["charge_ws"])),
            "charge_err": float(charge_mid),
            "f_ws": _f_ws(result_mid),
        })

        if config.verbose:
            if result_mid.get("neutrality_mode", "ws") == "ws":
                print(f"[mu iter] mu={mu_mid:.4f}, charge_err={charge_mid:.3e}")
            else:
                f_ws = _f_ws(result_mid)
                print(f"[mu iter] mu={mu_mid:.4f}, charge_err={charge_mid:.3e}, f_ws={f_ws:.3e}")

        v_full = result_mid["v_full"]
        v_ext = result_mid["v_ext"]

        if abs(charge_mid) < config.mu_tol:
            break

        if charge_lo * charge_mid < 0.0:
            mu_hi = mu_mid
            charge_hi = charge_mid
        else:
            mu_lo = mu_mid
            charge_lo = charge_mid

    if result_mid is None:
        return result_hi

    result_mid["mu_history"] = mu_history
    return result_mid
