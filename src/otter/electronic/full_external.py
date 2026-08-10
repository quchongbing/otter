"""
otter/electronic/full_external.py

User-facing runner for the electronic-structure workflow:

    full SCF (ion-sphere neutral) -> external SCF (fixed mu, fixed n0)

The full/external construction, pseudoatom density, Appendix-A continuum,
and Appendix-B far-field continuation follow C. E. Starrett and D. Saumon,
:cite:`StarrettSaumon2014`, High Energy Density Physics 10, 35--42 (2014),
DOI 10.1016/j.hedp.2013.12.001.  Root bracketing, warm starts, charge-closure
guards, threshold-state reliability tests, and fail-closed handoff rules are
Otter numerical methods; they not be attributed to that paper. The
pressure-ionization M(e) partition and cutoff are based on
:cite:`StarrettSaumon2013`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any

import numpy as np

from otter.electronic.ks_dft import (
    KSDTFConfig,
    _apply_charge_constrained_b3_tail,
    _apply_external_energy_floor,
    _continuum_prefix_length,
    _electron_count,
    _enforce_source_charge_closure,
    _rebuild_continuum_on_full_grid,
    _resolve_iteration_continuum_l_max,
    _source_electron_charge_target,
    _split_continuum_params_for_full_ext,
    _ion_level_weight,
    solve_ks_dft_is,
)
from otter.electronic.continuum import scattering as qmod
from otter.electronic.continuum.ideal import ideal_unbound_density
from otter.electronic.continuum.tail import apply_tail_match, tail_parameters
from otter.electronic.continuum.scattering import fermi_dirac
from otter.numerics.grids import create_sqrt_grid
from otter.numerics.constants import EV_TO_HA
from otter.data.elements import element as element_info
from otter.data.helpers import (
    ion_density_bohr3,
    mu_guess_from_density,
    trapz_integral,
)
from otter.io import save_full_external_data
from otter.ionic.correlation import IonSphereStepModel, ion_sphere_radius_from_density
from otter.electronic.potential import spherical_hartree_potential
from otter.electronic.potential import (
    _ion_sphere_background_hartree,
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
from otter.electronic.solvers.bound import solve_bound_states_sparse_numerov


def _default_b3_geometry_from_density(rho_g_cc: float) -> tuple[float, float, float]:
    """
    Return the current default A3+B3 geometry for a given mass density.

    The public default is currently density-independent. The validated
    production geometry is the same for low-, moderate-, and high-density
    states unless the user overrides it explicitly.
    """
    rho = float(rho_g_cc)
    return 7.0, 5.0, 4.0


def _resolve_outer_geometry(cfg: "FullExternalConfig", *, r_ws: float) -> dict[str, float | None]:
    """
    Resolve the actual AA / A3 / B3 outer radii used for one state.

    Parameters
    ----------
    cfg
        High-level full/external configuration.
    r_ws
        Wigner-Seitz radius in Bohr.

    Returns
    -------
    dict[str, float | None]
        Physical radii in Bohr together with the corresponding effective
        multipliers after any absolute high-density floors are applied.

    Notes
    -----
    The validated public defaults are still expressed in units of ``R_ws```.
    For dense / small-``R_ws`` states we keep the *physical* WS radius intact,
    but define one numerical geometry scale

      r_geom = max(r_ws, geometry_r_ws_floor_bohr)
      if geometry_r_ws_cap_bohr is not None:
          r_geom = min(r_geom, geometry_r_ws_cap_bohr)

    and apply the usual public multipliers to ``r_geom`` instead of to the
    smaller physical ``R_ws``. This keeps all outer radii and widths
    proportional to one another while avoiding the over-short geometry that can
    truncate the screening cloud in dense, fully ionized states.
    """
    r_ws_val = float(r_ws)
    if r_ws_val <= 0.0:
        raise ValueError("r_ws must be positive.")
    r_geom = max(r_ws_val, float(cfg.geometry_r_ws_floor_bohr))
    if cfg.geometry_r_ws_cap_bohr is not None:
        r_geom = min(r_geom, float(cfg.geometry_r_ws_cap_bohr))

    rmax = float(cfg.rmax_mult) * r_geom
    out: dict[str, float | None] = {
        "rmax": float(rmax),
        "rmax_eff_mult": float(rmax / r_ws_val),
        "r_geom": float(r_geom),
        "solve_rmax": None,
        "solve_rmax_eff_mult": None,
        "r_fit_max": None,
        "r_fit_max_eff_mult": None,
        "r_cut": None,
        "r_cut_eff_mult": None,
    }

    if cfg.cont_rmax_mult is not None:
        solve_rmax = float(cfg.cont_rmax_mult) * r_geom
        solve_rmax = min(float(solve_rmax), float(rmax))
        out["solve_rmax"] = float(solve_rmax)
        out["solve_rmax_eff_mult"] = float(solve_rmax / r_ws_val)

    if cfg.b3_r_fit_max_mult is not None:
        fit_max = float(cfg.b3_r_fit_max_mult) * r_geom
        fit_upper = float(out["solve_rmax"]) if out["solve_rmax"] is not None else float(rmax)
        fit_max = min(float(fit_max), fit_upper)
        out["r_fit_max"] = float(fit_max)
        out["r_fit_max_eff_mult"] = float(fit_max / r_ws_val)

    if cfg.b3_r_cut_mult is not None:
        cut_val = float(cfg.b3_r_cut_mult) * r_geom
        cut_upper = (
            float(out["r_fit_max"])
            if out["r_fit_max"] is not None
            else (float(out["solve_rmax"]) if out["solve_rmax"] is not None else float(rmax))
        )
        cut_val = min(float(cut_val), max(1.0e-8, cut_upper - 1.0e-8))
        out["r_cut"] = float(cut_val)
        out["r_cut_eff_mult"] = float(cut_val / r_ws_val)

    return out


def _uses_decoupled_ion_sphere_grid(
    cfg: "FullExternalConfig",
    *,
    r_ws: float,
    rmax: float,
) -> bool:
    """Return whether a floor/cap decouples the AA box from physical R_ws."""
    physical_rmax = float(cfg.rmax_mult) * float(r_ws)
    return not np.isclose(float(rmax), physical_rmax, rtol=1.0e-12, atol=1.0e-12)


def _uses_analytic_ion_sphere_background(
    cfg: "FullExternalConfig",
    *,
    r_ws: float,
    rmax: float,
) -> bool:
    """Return whether the default sharp background needs analytic integration."""
    return cfg.g_ii_override is None and _uses_decoupled_ion_sphere_grid(
        cfg,
        r_ws=float(r_ws),
        rmax=float(rmax),
    )


def _compute_veff_asymptotic_diagnostics(
    *,
    r: np.ndarray,
    v_eff: np.ndarray,
    n_full: np.ndarray,
    n0: float,
    g_ii: np.ndarray,
    z_nuc: float,
    r_ws: float,
    shallowest_bound_energy_ha: float = np.nan,
    analytic_ion_sphere_background: bool = False,
    absolute_tail_tolerance_ha: float = 1.0e-3,
) -> dict[str, Any]:
    """Return conservative large-radius diagnostics for the final full potential.

    ``n_full`` is the density actually used for potential assembly (the
    low-level ``n_full_source`` when source closure is active), rather than
    necessarily the separately retained output-density profile.

    The diagnostic deliberately does *not* require ``V_eff(r)`` to be monotone:
    a physical finite-temperature screening tail may cross zero because of
    Friedel oscillations.  Instead it records the signed extrema and maximum
    absolute amplitude in the outer 20% of the actual SCF box, the boundary
    value, and the net charge of the source in Starrett & Saumon's Eq. (4).

    When a negative-energy threshold state is present, potential amplitudes are
    also reported relative to its binding energy.  The resulting status is
    diagnostic only and never changes the density, potential, bound spectrum,
    or ``threshold_state_status``.  A ``marginal``/``unresolved`` value means
    that the finite SCF box is not demonstrably asymptotic on the energy scale
    needed to classify that shallow state.

    References
    ----------
    C. E. Starrett and D. Saumon, *High Energy Density Physics* **10**,
    35--42 (2014), Eq. (4) and Appendix B,
    doi:10.1016/j.hedp.2013.12.001.  Charge neutrality removes the Coulomb
    ``1/r`` tail, while the remaining screened potential need not approach
    zero monotonically.
    """
    r_arr = np.asarray(r, dtype=float)
    v_arr = np.asarray(v_eff, dtype=float)
    n_arr = np.asarray(n_full, dtype=float)
    g_arr = np.asarray(g_ii, dtype=float)
    if (
        r_arr.ndim != 1
        or r_arr.size < 2
        or v_arr.shape != r_arr.shape
        or n_arr.shape != r_arr.shape
        or g_arr.shape != r_arr.shape
    ):
        raise ValueError("r, v_eff, n_full, and g_ii must be aligned 1D arrays.")
    if (
        not np.all(np.isfinite(r_arr))
        or not np.all(np.isfinite(v_arr))
        or not np.all(np.isfinite(n_arr))
        or not np.all(np.isfinite(g_arr))
        or np.any(np.diff(r_arr) <= 0.0)
    ):
        raise ValueError("V_eff asymptotic diagnostics require finite increasing data.")
    n0_val = float(n0)
    z_val = float(z_nuc)
    r_ws_val = float(r_ws)
    if not np.isfinite(n0_val) or n0_val < 0.0:
        raise ValueError("n0 must be finite and non-negative.")
    if not np.isfinite(z_val) or z_val <= 0.0:
        raise ValueError("z_nuc must be finite and positive.")
    if not np.isfinite(r_ws_val) or r_ws_val <= 0.0:
        raise ValueError("r_ws must be finite and positive.")
    abs_tail_tol = float(absolute_tail_tolerance_ha)
    if not np.isfinite(abs_tail_tol) or abs_tail_tol <= 0.0:
        raise ValueError("absolute_tail_tolerance_ha must be finite and positive.")

    outer_start = float(r_arr[0] + 0.8 * (r_arr[-1] - r_arr[0]))
    outer_mask = r_arr >= outer_start
    v_outer = v_arr[outer_mask]
    outer_min = float(np.min(v_outer))
    outer_max = float(np.max(v_outer))
    outer_max_abs = float(np.max(np.abs(v_outer)))
    boundary = float(v_arr[-1])

    beyond_5rws = r_arr >= 5.0 * r_ws_val
    if np.any(beyond_5rws):
        v_beyond_5rws = v_arr[beyond_5rws]
        beyond_5rws_min = float(np.min(v_beyond_5rws))
        beyond_5rws_max = float(np.max(v_beyond_5rws))
        beyond_5rws_max_abs = float(np.max(np.abs(v_beyond_5rws)))
    else:
        beyond_5rws_min = np.nan
        beyond_5rws_max = np.nan
        beyond_5rws_max_abs = np.nan

    # Count crossings only as descriptive evidence of an oscillatory tail.
    # It is intentionally absent from the status criteria.
    outer_sign = np.signbit(v_outer)
    outer_zero_crossings = int(np.count_nonzero(outer_sign[1:] != outer_sign[:-1]))

    if bool(analytic_ion_sphere_background):
        # The actual potential used an analytic -n0*Theta(r-Rws) source.
        # Integrate (n_full-n0) over the box and restore the excluded sphere
        # analytically, avoiding a grid-dependent step-function charge.
        q_source = 4.0 * np.pi * trapz_integral(
            (r_arr**2) * (n_arr - n0_val), r_arr
        )
        sphere_radius = min(r_ws_val, float(r_arr[-1]))
        q_source += 4.0 * np.pi * n0_val * sphere_radius**3 / 3.0
        source_mode = "analytic-ion-sphere"
    else:
        q_source = 4.0 * np.pi * trapz_integral(
            (r_arr**2) * (n_arr - n0_val * g_arr), r_arr
        )
        source_mode = "sampled-gii"
    source_residual = float(q_source - z_val)
    source_residual_rel = float(abs(source_residual) / max(abs(z_val), 1.0e-300))
    coulomb_edge_estimate = float(source_residual / float(r_arr[-1]))

    shallow_energy = float(shallowest_bound_energy_ha)
    has_shallow_binding = bool(np.isfinite(shallow_energy) and shallow_energy < 0.0)
    binding = float(abs(shallow_energy)) if has_shallow_binding else np.nan
    if has_shallow_binding:
        boundary_ratio = float(abs(boundary) / binding)
        outer_ratio = float(outer_max_abs / binding)
        beyond_5rws_ratio = (
            float(beyond_5rws_max_abs / binding)
            if np.isfinite(beyond_5rws_max_abs)
            else np.nan
        )
    else:
        boundary_ratio = np.nan
        outer_ratio = np.nan
        beyond_5rws_ratio = np.nan

    severe_reasons: list[str] = []
    marginal_reasons: list[str] = []
    if source_residual_rel > 1.0e-1:
        severe_reasons.append("source-charge-residual-above-10-percent")
    elif source_residual_rel > 1.0e-2:
        marginal_reasons.append("source-charge-residual-above-1-percent")
    # A binding-energy ratio is unavailable when no negative-energy level is
    # returned.  Retain an absolute guard in that case (and as a backstop in
    # all cases) so a large constant or long-range potential cannot be labelled
    # "resolved" merely because the source charge integrates correctly.
    if abs(boundary) > 10.0 * abs_tail_tol:
        severe_reasons.append("boundary-potential-above-absolute-severe-tolerance")
    elif abs(boundary) > abs_tail_tol:
        marginal_reasons.append("boundary-potential-above-absolute-tolerance")
    if outer_max_abs > 10.0 * abs_tail_tol:
        severe_reasons.append("outer-potential-above-absolute-severe-tolerance")
    elif outer_max_abs > abs_tail_tol:
        marginal_reasons.append("outer-potential-above-absolute-tolerance")
    if has_shallow_binding:
        if boundary_ratio > 1.0:
            severe_reasons.append("boundary-potential-exceeds-binding")
        elif boundary_ratio > 1.0e-1:
            marginal_reasons.append("boundary-potential-above-10-percent-of-binding")
        if outer_ratio > 1.0:
            severe_reasons.append("outer-potential-exceeds-binding")
        elif outer_ratio > 1.0e-1:
            marginal_reasons.append("outer-potential-above-10-percent-of-binding")

    if severe_reasons:
        status = "unresolved"
        status_reason = ";".join(severe_reasons + marginal_reasons)
    elif marginal_reasons:
        status = "marginal"
        status_reason = ";".join(marginal_reasons)
    else:
        status = "resolved"
        status_reason = "outer-amplitude-and-source-charge-within-diagnostic-tolerances"

    return {
        "veff_asymptotic_boundary_ha": boundary,
        "veff_asymptotic_outer_start_bohr": outer_start,
        "veff_asymptotic_outer_min_ha": outer_min,
        "veff_asymptotic_outer_max_ha": outer_max,
        "veff_asymptotic_outer_max_abs_ha": outer_max_abs,
        "veff_asymptotic_outer_zero_crossings": outer_zero_crossings,
        "veff_asymptotic_beyond_5rws_min_ha": beyond_5rws_min,
        "veff_asymptotic_beyond_5rws_max_ha": beyond_5rws_max,
        "veff_asymptotic_beyond_5rws_max_abs_ha": beyond_5rws_max_abs,
        "veff_asymptotic_shallow_binding_ha": binding,
        "veff_asymptotic_boundary_to_binding_ratio": boundary_ratio,
        "veff_asymptotic_outer_to_binding_ratio": outer_ratio,
        "veff_asymptotic_beyond_5rws_to_binding_ratio": beyond_5rws_ratio,
        "veff_asymptotic_source_charge_e": float(q_source),
        "veff_asymptotic_source_charge_residual_e": source_residual,
        "veff_asymptotic_source_charge_residual_rel": source_residual_rel,
        "veff_asymptotic_coulomb_edge_estimate_ha": coulomb_edge_estimate,
        "veff_asymptotic_boundary_minus_coulomb_estimate_ha": float(
            boundary - coulomb_edge_estimate
        ),
        "veff_asymptotic_source_charge_mode": source_mode,
        "veff_asymptotic_absolute_tail_tolerance_ha": abs_tail_tol,
        "veff_asymptotic_status": status,
        "veff_asymptotic_status_reason": status_reason,
        "veff_asymptotic_severe_tail": bool(severe_reasons),
        "veff_asymptotic_marginal_tail": bool(
            marginal_reasons and not severe_reasons
        ),
        "veff_asymptotic_friedel_aware": True,
        "veff_asymptotic_monotonicity_tested": False,
    }


def _ion_density_from_r_ws(r_ws: float) -> float:
    """Return ion number density (Bohr^-3) implied by one WS radius."""
    r_ws = float(r_ws)
    if r_ws <= 0.0:
        raise ValueError("r_ws must be positive.")
    return float(3.0 / (4.0 * np.pi * r_ws**3))


def _target_radial_grid(*, rmax: float, n_points: int) -> np.ndarray:
    """Return the radial grid used by the low-level full/external solver."""
    return np.asarray(create_sqrt_grid(rmax=float(rmax), N=int(n_points)).r, dtype=float)


def _resample_initial_potential(
    *,
    values: np.ndarray | None,
    source_r: np.ndarray | None,
    target_r: np.ndarray,
    name: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Validate and map an optional initial potential onto the target AA grid.

    `values` without `source_r` preserves the historical contract: the array
    must already live on the target grid. When `source_r` is supplied, linear
    interpolation in physical radius is used, enabling safe continuation
    between nearby density/temperature states whose radial grids differ.
    """
    info: dict[str, Any] = {
        "used": False,
        "interpolated": False,
        "source_r_min": np.nan,
        "source_r_max": np.nan,
    }
    if values is None:
        return None, info

    target = np.asarray(target_r, dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")

    if source_r is None:
        if arr.shape != target.shape:
            raise ValueError(f"{name} must match the target grid shape when {name}_r is not supplied.")
        info["used"] = True
        return arr.copy(), info

    r_src = np.asarray(source_r, dtype=float)
    if r_src.ndim != 1 or r_src.shape != arr.shape:
        raise ValueError(f"{name}_r must be a 1D array with the same shape as {name}.")
    if r_src.size < 2:
        raise ValueError(f"{name}_r must contain at least two grid points.")
    if np.any(np.diff(r_src) <= 0.0):
        raise ValueError(f"{name}_r must be strictly increasing.")

    mapped = np.interp(target, r_src, arr, left=float(arr[0]), right=float(arr[-1]))
    info.update({
        "used": True,
        "interpolated": True,
        "source_r_min": float(r_src[0]),
        "source_r_max": float(r_src[-1]),
    })
    return np.asarray(mapped, dtype=float), info


_AUFBAU_ORBITALS: tuple[tuple[int, int], ...] = (
    (1, 0),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (4, 0),
    (3, 2),
    (4, 1),
    (5, 0),
    (4, 2),
    (5, 1),
    (6, 0),
    (4, 3),
    (5, 2),
    (6, 1),
    (7, 0),
    (5, 3),
    (6, 2),
    (7, 1),
)


def _auto_bound_basis_from_z(
    z_nuc: int,
    *,
    n_pad: int,
    l_pad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an element-aware default bound basis from neutral-shell filling.

    Parameters
    ----------
    z_nuc : int
        Nuclear charge.
    n_pad : int
        Extra radial states added to each occupied-l channel.
    l_pad : int
        Extra angular channels included above the highest occupied l.

    Returns
    -------
    l_list, n_states_by_l : ndarray, ndarray
        Auto-selected l channels and per-l radial-state caps.

    Notes
    -----
    The selection follows the standard aufbau filling order up to 7p, which is
    sufficient for Z <= 118. For each occupied angular momentum l we count the
    number of occupied subshells n and then add a small buffer ``n_pad`` so the
    bound solver also covers the first unoccupied state of that l channel. We
    additionally include ``l_pad`` extra angular channels above the highest
    occupied l; higher-l channels are then pruned cheaply by the sparse bound
    solver as soon as their lowest state becomes non-bound.
    """
    remaining = max(int(z_nuc), 0)
    occupied_shell_count: dict[int, int] = {}
    l_occ_max = 0
    for n_val, l_val in _AUFBAU_ORBITALS:
        if remaining <= 0:
            break
        cap = 2 * (2 * int(l_val) + 1)
        occ = min(cap, remaining)
        if occ > 0:
            occupied_shell_count[int(l_val)] = occupied_shell_count.get(int(l_val), 0) + 1
            l_occ_max = max(l_occ_max, int(l_val))
            remaining -= occ
    l_hi = max(l_occ_max + max(int(l_pad), 0), 0)
    l_list = np.arange(0, l_hi + 1, dtype=int)
    n_states_by_l = np.asarray(
        [
            max(occupied_shell_count.get(int(l_val), 0) + max(int(n_pad), 0), 1)
            for l_val in l_list
        ],
        dtype=int,
    )
    return l_list, n_states_by_l


def _summarize_history_perf(history: list[dict[str, Any]] | None) -> dict[str, Any]:
    """
    Aggregate per-iteration SCF timing metadata from a solver history.

    The low-level KS solver stores a ``perf`` dictionary on each history entry
    when ``perf_diag=True``. This helper keeps that information easy to inspect
    from the higher-level ``solve_full_only`` / ``solve_full_then_external``
    APIs by returning compact mean/max/sum summaries.
    """
    hist = list(history or [])
    perf_entries = [entry.get("perf") for entry in hist if isinstance(entry.get("perf"), dict)]
    if not perf_entries:
        return {"n_iter": len(hist), "n_perf": 0}

    basis_meta_keys = {
        "basis_n_eval",
        "basis_n_cache_hits",
        "basis_n_e",
        "basis_n_base_per_shard",
        "basis_shard_cache_merged",
        "basis_n_windows",
        "basis_ext_n_eval",
        "basis_ext_n_cache_hits",
        "basis_ext_n_e",
    }
    timing_keys: set[str] = set()
    meta_keys: set[str] = set()
    for perf in perf_entries:
        for key, value in perf.items():
            if not np.isfinite(float(value)):
                continue
            if key in basis_meta_keys:
                meta_keys.add(key)
            else:
                timing_keys.add(key)

    def _stats(keys: set[str]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for key in sorted(keys):
            vals = np.asarray(
                [float(perf[key]) for perf in perf_entries if key in perf and np.isfinite(float(perf[key]))],
                dtype=float,
            )
            if vals.size == 0:
                continue
            out[key] = {
                "sum": float(np.sum(vals)),
                "mean": float(np.mean(vals)),
                "max": float(np.max(vals)),
                "min": float(np.min(vals)),
            }
        return out

    return {
        "n_iter": len(hist),
        "n_perf": len(perf_entries),
        "timing_s": _stats(timing_keys),
        "basis_meta": _stats(meta_keys),
    }


def _format_perf_summary_line(summary: dict[str, Any], *, field: str, label: str) -> str | None:
    values = summary.get(field, {})
    if not isinstance(values, dict) or not values:
        return None
    parts = []
    for key, stats in values.items():
        if not isinstance(stats, dict):
            continue
        val = stats.get(label)
        if val is None or not np.isfinite(float(val)):
            continue
        parts.append(f"{key}={float(val):.3f}")
    if not parts:
        return None
    return ", ".join(parts)


def _resolve_species_geometry(
    cfg: "FullExternalConfig",
    *,
    atomic_weight: float,
) -> tuple[float, float]:
    """
    Resolve the per-ion density and WS radius used by one AA solve.

    The standard single-species path derives both from the mass density. Binary
    and future mixture solvers instead provide one of the explicit overrides so
    the AA sphere volume is controlled directly by the mixture volume closure.
    """
    r_ws_override = cfg.r_ws_override_bohr
    n_i_override = cfg.n_i_override_bohr3
    if r_ws_override is not None:
        r_ws = float(r_ws_override)
        n_i = float(n_i_override) if n_i_override is not None else _ion_density_from_r_ws(r_ws)
        if n_i_override is not None:
            r_ws_from_n = ion_sphere_radius_from_density(n_i)
            rel = abs(r_ws_from_n - r_ws) / max(abs(r_ws), 1e-14)
            if rel > 1e-8:
                raise ValueError("r_ws_override_bohr and n_i_override_bohr3 are inconsistent.")
        return n_i, r_ws
    if n_i_override is not None:
        n_i = float(n_i_override)
        if n_i <= 0.0:
            raise ValueError("n_i_override_bohr3 must be positive.")
        return n_i, ion_sphere_radius_from_density(n_i)
    n_i = ion_density_bohr3(float(cfg.rho_g_cc), atomic_weight)
    return n_i, ion_sphere_radius_from_density(n_i)


@dataclass
class FullExternalConfig(CitationMixin):
    """
    Input configuration for full->external electronic-structure solve.

    Minimal required inputs
    -----------------------
    - element: atomic number (e.g. 13) or symbol (e.g. "Al")
    - temperature_ev: electron temperature in eV
    - rho_g_cc: mass density in g/cc
    """

    # ----- Minimal required state -----
    element: int | str
    # Element selector: atomic number (int, e.g. 13) or symbol (str, e.g. "Al").
    temperature_ev: float
    # Electron temperature Te in eV (converted internally to Hartree).
    rho_g_cc: float
    # Mass density in g/cc.
    electronic_model: str = "qm"
    # Electronic-structure approximation. ``"qm"`` uses the orbital KS-AA
    # solver; ``"tf"`` uses the finite-temperature Thomas--Fermi full/external
    # construction of Starrett and Saumon, HEDP 10, 35--42 (2014), Eqs. (2)--(11).
    xc_model: str = "dirac"
    # Exchange-correlation model: dirac, pbe, an LDA alias, or libxc:<names>.
    gga_core_mode: str = "finite"
    # ``finite`` regularizes the nuclear GGA core; ``strict`` keeps raw GGA.
    gga_core_zr: float = 0.05
    # Dimensionless transition radius Z*r_c for the finite GGA core.
    r_ws_override_bohr: float | None = None
    # Optional explicit WS radius. When set, the AA sphere size is no longer
    # derived from rho_g_cc; this is the validated hook used by mixture solvers.
    n_i_override_bohr3: float | None = None
    # Optional explicit ion density matching r_ws_override_bohr. If r_ws is
    # provided alone we reconstruct n_i from the sphere volume.
    v_full_init: np.ndarray | None = None
    # Optional initial guess for the full effective potential on the target
    # radial grid. This is mainly used by higher-level continuation workflows
    # such as mixture AA polishing.
    v_full_init_r: np.ndarray | None = None
    # Optional source radial grid for v_full_init. When supplied, v_full_init is
    # interpolated onto the target grid, which is the safe path for density or
    # temperature scans where r_max changes between neighboring states.
    v_ext_init: np.ndarray | None = None
    # Optional initial guess for the external effective potential on the target
    # radial grid.
    v_ext_init_r: np.ndarray | None = None
    # Optional source radial grid for v_ext_init. Kept for API symmetry and
    # future external-branch continuation workflows.
    v_corr_full: np.ndarray | None = None
    # Experimental additive full-system correlation potential on the target
    # grid, or on v_corr_full_r when provided.  This is intended for SC/QTCP
    # feedback diagnostics, e.g. Starrett2014 Eq.(19), and is off by default.
    v_corr_full_r: np.ndarray | None = None
    # Source radial grid for v_corr_full.
    v_corr_ext: np.ndarray | None = None
    # Experimental additive external-system correlation potential.
    v_corr_ext_r: np.ndarray | None = None
    # Source radial grid for v_corr_ext.
    g_ii_override: np.ndarray | None = None
    # Optional ion-background profile for experimental AA/TCP feedback tests.
    # If None, use the IS ion-sphere step g_II(r)=theta(r-R_ws) from
    # Starrett--Saumon (2014), Eq. (8).
    g_ii_override_r: np.ndarray | None = None
    # Source radial grid for g_ii_override. If None, g_ii_override must already
    # live on the target AA grid.
    continuation_stage2_from_init: bool = False
    # If True and v_full_init is supplied, stage-2 starts from the interpolated
    # continuation potential instead of the stage-1 output. This is useful for
    # smooth parameter scans where the previous converged V_eff is already a
    # better stage-2 initial guess than the cheaper stage-1 preconditioner.
    continuation_mu_init: float | None = None
    # Optional chemical-potential guess paired with the continuation potential.
    # When continuation_stage2_from_init=True, stage-2 uses this as its initial
    # mu guess instead of the stage-1 mu.
    continuation_scf_mixing_scheme: str | None = "eyert"
    # Full-SCF mixer used only for direct stage-2 continuation starts. None
    # means reuse the ordinary cold-start scf_mixing_scheme. The default keeps
    # the validated Eyert mixer but permits a warm-start-specific history/mix.
    continuation_scf_mixing_m: int | None = 6
    # Eyert history length for direct stage-2 continuation starts. A converged
    # neighboring V_eff is already close to the fixed point, so a slightly
    # longer history than the conservative cold default can reduce iterations.
    continuation_scf_mix: float | None = 0.3
    # Full-SCF mix parameter for direct stage-2 continuation starts. This is
    # deliberately applied only when a continuation V_eff initializes stage-2;
    # cold-start and stage-1 remain on the conservative scf_mix default.

    # ----- Radial domain + SCF staging -----
    rmax_mult: float = 15.0
    # Radial box size: r_max = rmax_mult * R_ws.
    tf_scf_mix: float = 0.20
    # Eyert/Anderson base mixing used only by the Thomas--Fermi backend.
    tf_scf_mixing_m: int = 6
    tf_scf_mixing_w0: float = 5.0e-4
    tf_scf_max_iter: int = 400
    tf_scf_tol: float = 5.0e-5
    tf_polish_tol: float = 1.0e-10
    tf_polish_max_iter: int = 80
    tf_mu_tol: float = 1.0e-10
    tf_quadrature_order: int = 160
    bound_rmax_mult: float | None = None
    # Experimental bound-only box extension.  The Starrett--Saumon (2014)
    # Appendix-A calculation uses one AA domain (typically 15 R_WS); it does
    # not prescribe a separate 40 R_WS bound box.  When this option is set,
    # Otter extends the converged SCF potential by an artificial zero tail, so
    # the result is a numerical sensitivity diagnostic rather than additional
    # physical plasma volume.  The production default therefore uses the same
    # domain for bound, continuum, and SCF calculations.
    bound_zero_tail_refine: bool = False
    # Optional low-l threshold check.  With no bound_rmax_mult it replaces a
    # shallow Dirichlet-box orbital by an analytic exterior-matched pole
    # normalized over all space, but only when the common physical SCF
    # boundary is already asymptotic on the binding-energy scale.  With an
    # explicitly requested bound-only zero extension it is instead a numerical
    # sensitivity diagnostic.  Neither path is a production default.
    # Starrett et al. (CPC 235, 50--62, 2019), Eqs. (21)-(22), give the
    # non-relativistic negative-energy exterior matching and normalization.
    bound_zero_tail_min_binding_ha: float = 1.0e-8
    bound_zero_tail_max_binding_ha: float = 1.0e-3
    bound_zero_tail_scan_points: int = 24
    bound_zero_tail_l_max: int = 0
    bound_zero_tail_edge_rel_tol: float = 0.25
    geometry_r_ws_floor_bohr: float = 0.8
    # Numerical geometry scale floor (Bohr). When R_ws is smaller than this
    # value, all outer-geometry multipliers are applied to this floor instead
    # of to the physical R_ws.
    geometry_r_ws_cap_bohr: float | None = None
    # Optional numerical geometry scale cap (Bohr). When set, all outer-
    # geometry multipliers are applied to min(max(R_ws, floor), cap). This is
    # intended as an optional speed/box-length limiter for very low-density
    # states; keep None unless a shorter outer box is explicitly desired. For
    # exploratory low-density scans, values around 5 Bohr are a reasonable
    # user-side starting point to test.
    n_points: int = 2**12
    # Number of radial grid points (sqrt grid used internally).
    mu_bounds: tuple[float, float] = (-200.0, 200.0)
    # Brent bracket for inner mu-neutral solver (Ha).
    full_fixed_mu_ha: float | None = None
    # Optional fixed full-branch chemical potential [Ha]. This is disabled by
    # default. It is intended for Starrett SC/QTCP diagnostics, where the
    # outer feedback loop keeps the ion-sphere chemical potential fixed instead
    # of re-solving neutrality after each ion-electron correlation update.
    mu_guess_zbar: float = 2.0
    # Initial mu guess from ideal free-electron estimate with this Zbar.
    stage1_max_iter: int = 5
    # Stage-1 full SCF iteration cap (robust precondition stage).
    stage2_max_iter: int = 107
    # Stage-2 full SCF iteration cap (refined production stage).
    stage2_local_mu_bounds: bool = False
    # If True, stage-2 mu bracket is centered around stage-1 mu.
    stage2_mu_half_width: float = 4.0
    # Half-width of local stage-2 mu bracket (Ha).
    run_mode: str = "full+ext"
    # "full" -> run only full SCF; "full+ext" -> run full then fixed-mu external SCF.

    # ----- Bound-state basis -----
    bound_basis_mode: str = "auto"
    # Bound basis policy:
    #   "auto"   -> infer l_list / per-l radial caps from the element
    #   "manual" -> use user-supplied n_states / l_list
    n_states: int = 3
    # Manual bound solver principal-state cap. In auto mode this is used only
    # when the user overrides the historical default.
    l_list: np.ndarray | None = None
    # Explicit l channels for bound solve. In auto mode, setting this switches
    # the bound basis back to a manual override.
    bound_auto_n_pad: int = 1
    # Extra radial states added to each occupied l channel in auto mode.
    bound_auto_l_pad: int = 1
    # Extra angular channels above the highest occupied l in auto mode.

    # ----- Continuum energy integration -----
    cont_energy_mode: str = "adaptive"
    # "adaptive" (recommended) or "linear".
    cont_adaptive_mode_stage1: str = "bisection"
    # Adaptive mode for stage-1 full solve.
    cont_adaptive_mode_stage2: str = "simpson"
    # Adaptive mode for stage-2 full solve. Keep the published/validated
    # workflow on global Simpson refinement by default; phase-shift-localized
    # bisection remains available as an optional experimental mode.
    cont_n_e_base: int = 100
    # Number of initial energy grid points before adaptive refinement.
    cont_e_tol: float = 2e-3
    # Adaptive refinement tolerance in energy integration.
    cont_e_max_depth: int = 10
    # Max adaptive recursion depth.
    cont_dE_min: float = 1e-2
    # Minimum allowed energy interval width before stopping refinement (Ha).
    cont_e_base_grid: str = "sqrt"
    # Initial energy-grid distribution ("sqrt" gives denser low-E sampling).
    cont_e_min: float = 1e-6
    # Minimum continuum energy (Ha).  The lower floor, together with the
    # logarithmic anchors below, resolves the s-wave spectral-weight transfer
    # when a very shallow bound state pressure-ionizes.  Wilson et al. (2006),
    # Appendix A.4, shows why this threshold limit must remain continuous.
    cont_near_zero_log_grid: bool = True
    # Insert logarithmic energy anchors between cont_e_min and the first
    # ordinary sqrt-grid node.  This is a threshold quadrature guard, not an
    # l=0 shape-resonance classifier (ordinary s waves do not form those).
    cont_near_zero_log_points_per_decade: int = 4
    # Number of logarithmic threshold anchors requested per energy decade.
    cont_near_zero_log_max_nodes: int = 24
    # Hard cap on the added low-energy nodes per adaptive integration domain.
    cont_near_zero_log_max_energy: float | None = 1.0e-2
    # Do not add threshold anchors above this energy [Ha].
    cont_resonance_theta_l_min: int = 1
    # Lowest partial wave included in the optional phase-root resonance scout.
    # Keep l=0 excluded: an s-wave threshold state is handled by the low-energy
    # mesh above and does not form an ordinary centrifugal shape resonance.
    cont_resonance_theta_probe_count: int = 1
    # Interior phase probes inserted in every base-energy panel when
    # cont_adaptive_mode_stage{1,2}="phase-root".
    cont_resonance_theta_scan_depth: int = 3
    # Nested dyadic scout depth for off-anchor matching-coefficient roots.
    # This gives a measurable finite resolution but cannot guarantee an
    # arbitrary ultra-narrow even root pair; see Wilson et al., JQSRT 99,
    # 658-679 (2006), Sec. 6, for the more complete a(E), b(E), Theta(E) idea.
    cont_resonance_theta_scout_max_extra_nodes: int | None = 128
    # Global cap on extra scout nodes beyond mandatory base/log anchors.  The
    # budget is shared across energy shards (phase-root currently forces the
    # safer global batch mode so symmetric root windows cross no shard edge).
    cont_resonance_theta_root_tol: float | None = None
    # Brent tolerance [Ha] for a bracketed normalized matching-coefficient root.
    cont_resonance_theta_sharpness_min: float = 2.0
    # Reject broad phase crossings whose dimensionless panel-scale slope is
    # below this value; they remain covered by the ordinary quadrature.
    cont_resonance_theta_max_roots: int | None = None
    # Optional hard cap on the sharpest roots retained per integration domain.
    cont_resonance_theta_refine_depth: int | None = None
    # Optional recursion-depth override inside explicitly located resonance
    # panels. None adds the continuum solver's bounded default allowance.
    cont_e_max: float | None = None
    # Fixed continuum energy ceiling (Ha). If left unset, stage 1 defaults to
    # max(10 * T_e[Ha], 5 Ha).
    cont_stage2_e_max_mode: str = "prev_mu_fd"
    # Stage-2 continuum ceiling policy:
    #   "fixed"      -> keep cont_e_max on every SCF iteration
    #   "prev_mu_fd" -> resolve e_max from the previous SCF mu using the FD
    #                   tail criterion, then apply cont_stage2_e_max_floor
    cont_stage2_e_max_occ_tol: float = 1.0e-5
    # Target FD occupation used by the stage-2 auto e_max estimate.
    cont_stage2_e_max_floor: float = 7.0
    # Safety floor (Ha) applied after the stage-2 FD inversion.
    cont_n_e_linear: int = 300
    # Number of energy points when cont_energy_mode="linear".
    cont_n_jobs: int = 1
    # Worker count for continuum calculations. Keep the single-AA default at 1
    # so higher-level parameter scans and mixture solvers can control parallel
    # work explicitly at the outer level without accidental nested oversubscription.
    cont_parallel_mode: str = "shard"
    # Adaptive parallelization strategy ("shard" validated path).
    cont_shards: int | None = 32
    # Explicit number of continuum energy shards. Using more shards than n_jobs
    # improves load balance because workers can pick up multiple smaller energy
    # intervals instead of being pinned to one large static interval.
    cont_shard_policy: str = "egrid"
    # Shard partition policy when cont_parallel_mode="shard":
    #   "egrid" -> shard boundaries follow the base-energy grid policy
    #   "cost"  -> shard boundaries approximately equalize continuum workload
    #              using the current l_cap(E) estimate.
    cont_adaptive_reuse_basis: bool = True
    # Reuse adaptive basis caches when available.
    cont_l_cap_strategy: str = "match"
    # l-cap policy for continuum partial-wave summation. "match" is the
    # validated default: the actual per-energy l_cap(E) is set by the A3
    # matching window rather than by the full simulation box.
    cont_rmax_mult: float | None = None
    # Optional continuum-only numerical radius in units of R_ws.
    # If None, the continuum is solved on the full box (same as r_max).
    # If set, only the continuum/A3 solve is truncated to
    #   R_dft_max = cont_rmax_mult * R_ws
    # while the bound solve and final output grid still use the full r_max.

    # ----- Continuum match-window controls -----
    cont_match_fraction: float = 0.30
    # Match-window length as a fraction of r_max.
    cont_match_kr_min: float = 3.0
    # Minimum k*r required for oscillatory matching.
    cont_match_v_tol: float = 1e-1
    # |V_eff| threshold for free-wave asymptotic matching.
    cont_match_min_points: int = 16
    # Minimum points required in match window.
    cont_match_r_cut_frac: float = 0.85
    # Right-edge cutoff of match window as fraction of r_max.
    cont_match_width_frac: float = 0.15
    # Match-window width as fraction of r_max.

    # ----- Optional B3 tail replacement -----
    b3_tail_stage1_mode: str = "in_scf"
    # B3 usage in stage-1 full SCF:
    #   "off"    -> pure A3/KS
    #   "post"   -> apply B3 once after stage-1 converges
    #   "in_scf" -> apply B3 during each SCF iteration
    b3_tail_stage2_mode: str = "in_scf"
    # Same choices as b3_tail_stage1_mode, but for stage-2.
    b3_tail_target: str = "cont"
    # Tail replacement target:
    #   "cont" -> replace n_cont and then rebuild n_full (historical default)
    #   "full" -> fit n_full itself, as stated in Starrett & Saumon (2014),
    #             Appendix B; important when a diffuse threshold orbital
    #             extends beyond the handoff radius
    #   "both" -> apply both operations for diagnostics
    # The literal full-density path remains opt-in until its pressure-
    # ionization common-mu regression is converged across the validation grid.
    b3_tail_fit_points: int = 20
    # Number of fit samples used by the B3 least-squares handoff.
    b3_tail_local_fit_width_mult: float | None = 0.064
    # Physical width of the local B3 matching stencil in units of R_ws.
    # A fixed point count alone changes the fitted radial interval when
    # n_points changes.  None retains that legacy, grid-dependent behavior.
    b3_tail_fit_window_mode: str = "local"
    # Placement of those fit samples:
    #   "auto"     -> for Appendix-B "full"/"both", span the physical window
    #                 only while its right edge retains >=10% of the Friedel
    #                 envelope; retain the validated local stencil for "cont"
    #   "physical" -> distribute samples across [r_cut, r_fit_max]
    #   "local"    -> distribute samples over the short physical stencil
    #                 after r_cut (or consecutive grid points when
    #                 b3_tail_local_fit_width_mult=None)
    b3_tail_blend_points: int = 10
    # Number of smoothing samples around the A3 -> B3 splice.
    b3_tail_model: str = "full"
    # Tail model choice:
    #   "full"   -> always use the full B3 oscillatory ansatz (default)
    #   "a_only" -> force the monotone Yukawa/TF term only
    #   "auto"   -> diagnostic selector: keep the full B3 term only when it
    #               improves the fit enough
    b3_tail_auto_rel_improve_tol: float = 0.2
    # In auto mode, require at least this relative RMS improvement before the
    # oscillatory B3 term is accepted over the simpler A-only tail.
    b3_tail_auto_signal_rel_tol: float = 5e-5
    # In auto mode, also require the fit-window signal `|n(r)-n0|` to exceed
    # this fraction of `n0`; otherwise the oscillatory term is suppressed.
    b3_r_fit_max_mult: float | None = None
    # Right edge of the B3 fit window in units of R_ws:
    #   r_fit_max = b3_r_fit_max_mult * R_ws
    # If None, fall back to the historical trusted radius
    #   r_fit_max = source_r_trust_frac * r_max
    b3_r_cut_mult: float | None = None
    # Direct B3 handoff radius in units of R_ws:
    #   r_cut = b3_r_cut_mult * R_ws
    # If set, this takes precedence over b3_cut_width and legacy r_cut aliases.
    b3_cut_width: float | None = None
    # B3 handoff width in units of R_ws:
    #   r_cut = r_fit_max - b3_cut_width * R_ws
    # This keeps the user-facing parameter independent of the absolute box size.
    b3_fallback_on_error: bool = True
    # If True, keep the original A3 density when B3 fit/splice fails.

    # Legacy B3 aliases kept for backward compatibility with older tests/scripts.
    cont_tail_match: bool = False
    # Legacy alias for "enable B3 during SCF" on all stages.
    cont_tail_match_target: str = "cont"
    # Legacy alias for b3_tail_target.
    cont_tail_fit_points: int = 20
    # Legacy alias for b3_tail_fit_points.
    cont_tail_blend_points: int = 10
    # Legacy alias for b3_tail_blend_points.
    cont_tail_r_cut_frac: float | None = None
    # Legacy alias for explicit r_cut as a fraction of r_max.
    cont_tail_cut_width: float | None = None
    # Legacy alias for an absolute B3 handoff width in Bohr.
    cont_tail_fallback_on_error: bool = True
    # Legacy alias for b3_fallback_on_error.

    # ----- External-branch stability overrides -----
    ext_match_v_tol: float | None = 1e-1
    # External-branch |V_eff| threshold for asymptotic matching.
    # Set to None to disable the |V| filter explicitly.
    ext_b3_tail_mode: str = "in_scf"
    # External B3 usage:
    #   "off"    -> aligned A3/source-closure external branch
    #   "in_scf" -> truncated A3 solve + B3 rebuild during each ext iteration
    ext_b3_tail_model: str | None = None
    # Optional external-only override for the B3 tail model:
    #   None     -> reuse b3_tail_model
    #   "full"   -> force the full oscillatory B3 ansatz
    #   "a_only" -> force the monotone Yukawa/TF tail only
    #   "auto"   -> let the external branch choose automatically
    ext_energy_floor_tail_frac: float = 0.90
    # Tail fraction used to estimate external energy floor when ext_match_v_tol=None.
    ext_energy_floor_margin: float = 1e-3
    # Safety margin added above estimated tail floor (Ha).

    # ----- Source-closure controls -----
    source_closure: bool = True
    # Enable trusted-region closure for source densities.
    source_r_trust_frac: float = 0.75
    # Trusted radius fraction of the continuum numerical radius R_dft_max
    # (or r_max when no continuum truncation is used).
    source_blend_frac: float = 0.03
    # Blend-zone width fraction of the continuum numerical radius R_dft_max
    # (or r_max when no continuum truncation is used).
    source_charge_closure: bool = True
    # Enforce integrated source charge closure in outer region.
    b3_source_charge_constraint: bool = False
    # When B3 is active in-SCF, impose the finite-box source-charge sum rule
    # directly on its linear (A,C,D) coefficients:
    #   integral[n_full - n0*g_II] = Z,
    #   integral[n_ext  - n0*g_II] = 0.
    # This is an experimental replacement for the historical uniform
    # outer-density correction. It is deliberately opt-in until its fitted
    # amplitudes and residuals are validated across the benchmark grid.
    b3_charge_constraint_fit_rms_ratio_max: float | None = 10.0
    # Reject an exact-charge B3 candidate when its local fit RMS is more than
    # this factor above the corresponding unconstrained B3 fit.  Ten is a
    # deliberately permissive guard against catastrophic extrapolation, not a
    # model-selection preference.  Set None to disable this acceptance check.
    b3_charge_constraint_profile_delta_rel_max: float | None = 10.0
    # Reject when the spherical L2 norm of the constrained-minus-unconstrained
    # tail exceeds this factor times the unconstrained B3 response norm.  This
    # is invariant to coefficient units and basis rescaling.  Set None to
    # disable this acceptance check.
    # Finite/non-negative density and post-Hermite charge closure are always
    # required even when both optional quality guards are disabled.
    full_b3_use_source_closure: bool | None = None
    # Full-branch source-closure policy while B3 is used in-SCF:
    #   None/auto -> disable it for b3_tail_target="full"/"both", because B3
    #                already supplies the matched total density; retain the
    #                historical closure for b3_tail_target="cont"
    #   True      -> force the legacy source closure
    #   False     -> use the matched B3 density directly
    ext_b3_use_source_closure: bool | None = None
    # External-branch source-closure policy when ext B3 is active:
    #   None/auto -> enable only after the rebuilt ext tail is already close
    #                enough to n0 near the trusted radius
    #   True      -> always apply source_closure to ext B3
    #   False     -> never apply source_closure to ext B3
    ext_source_closure_auto_rel_tol: float = 1.0e-5
    # Relative `|n_ext-n0|/max(|n0|, eps)` threshold used by the automatic
    # external source-closure policy.

    # ----- n0 closure overrides -----
    n0_mode_override: str | None = None
    # Optional override for the low-level KS `n0_mode`. When None, the current
    # validated policy is used:
    #   - "ideal"  for B3-in-SCF stages
    #   - "window" for pure-A3 stages
    # This is primarily a diagnostic hook used to compare whether differences
    # between pure A3 and A3+B3 come from the B3 handoff itself or from the
    # accompanying choice of uniform background closure.
    n0_fixed_override: float | None = None
    # Optional explicit `n0` used only when `n0_mode_override="fixed"`.

    # ----- Optional pseudoatom/screening tail diagnostic -----
    screening_tail_repair_mode: str = "off"
    # Optional downstream diagnostic evaluated after the final full+ext solve:
    #   "off"            -> do not construct a fitted screening tail
    #   "constrained_b3" -> report n_scr_repaired / n_pa_repaired from a
    #                       B3/Friedel fit constrained to Z-Q_ion(all)
    # The canonical n_scr=n_full-n_ext-n_ion and n_pa=n_full-n_ext profiles
    # are never overwritten.  A separately fitted screening tail is not an
    # electronic SCF solution and must not be passed to QOZ as though it were.
    screening_tail_repair_rel_tol: float = 5.0e-2
    # Only apply the repair when |Q_scr(box)-Zbar|/Zbar exceeds this threshold.
    screening_tail_charge_weight: float = 1.0e3
    # Relative weight of the integrated-charge constraint in the constrained
    # least-squares tail fit.
    screening_tail_r_cut_mult: float | None = None
    # Optional tail-fit handoff radius in units of R_ws. If None, reuse the
    # resolved B3 r_cut for the final full/external stage.
    screening_tail_r_fit_max_mult: float | None = None
    # Optional right edge of the screening-tail fit window in units of R_ws.
    # If None, reuse the resolved B3 r_fit_max for the final stage.

    # ----- Full SCF mixer -----
    scf_mix: float = 0.15
    # Base linear/Eyert mix parameter for full SCF map.
    scf_tol: float = 1e-3
    # Residual tolerance for full SCF map.
    scf_dn_tol: float = 1e-5
    # Density change tolerance for full SCF convergence.
    scf_dv_tol: float = 1e-5
    # Potential change tolerance for full SCF convergence.
    scf_mixing_scheme: str = "eyert"
    # "eyert" (recommended) or "linear".
    scf_mixing_m: int = 5
    # Eyert history size M.
    scf_mixing_w0: float = 5e-4
    # Eyert regularization weight.
    ph_kappa: float = 0.0
    # Poisson-Helmholtz preconditioner [Bohr^-1]. A positive value screens
    # the Hartree source only during the first ``ph_kappa_iters`` updates of
    # each SCF stage. It is a convergence aid, not part of the final physical
    # model: an accepted fixed point must include an ordinary kappa=0 Poisson
    # update. The default preserves the historical unscreened workflow.
    ph_kappa_iters: int = 0
    # Number of initial updates that use ``ph_kappa``. Stage-1, stage-2, and
    # the optional external fixed-mu SCF each count their own iterations.
    # Zero disables Poisson-Helmholtz preconditioning.
    full_v_eff_outer_decay: bool = False
    # Experimental non-Starrett taper: multiply full V_eff by an exponential
    # factor for r > full_v_eff_outer_decay_start_rws * R_ws.
    full_v_eff_outer_decay_start_rws: float = 1.0
    # Start radius of the outer V_eff taper in units of R_ws.
    full_v_eff_outer_decay_length_rws: float = 0.5
    # Exponential decay length of the taper in units of R_ws.
    veff_asymptotic_abs_tol_ha: float = 1.0e-3
    # Diagnostic (non-mutating) absolute tolerance for the final potential in
    # the outer 20% of the AA box. Values above 10 times this tolerance are
    # marked unresolved even when no shallow bound level supplies an energy
    # scale. 1e-3 Ha is about 0.027 eV.

    # ----- External fixed-mu SCF mixer -----
    ext_scf_enabled: bool = True
    # Run external fixed-(mu,n0) SCF after full SCF.
    ext_scf_max_iter: int = 120
    # Max external fixed-mu SCF iterations.
    ext_scf_mix: float = 0.25
    # Base mixing factor in external SCF. With Eyert mixing this slightly
    # larger fixed value converges faster than the older 0.2 default.
    ext_scf_dn_tol: float = 1e-4
    # Density tolerance for external SCF convergence.
    ext_scf_dv_tol: float = 1e-4
    # Potential tolerance for external SCF convergence.
    ext_scf_adaptive_mix: bool = False
    # Adapt external mixing strength based on iteration stability. Disabled by
    # default for the external Eyert path because the scalar shrink/expand
    # logic tends to over-damp late iterations and increase total ext iters.
    ext_mixing_scheme: str = "eyert"
    # "eyert" (recommended) or "linear" for external SCF.
    ext_mixing_m: int = 5
    # External Eyert history size M.
    ext_mixing_w0: float = 5e-4
    # External Eyert regularization.

    # ----- Bound partition controls -----
    bound_energy_cut_mode: str = "v_frac"
    # Bound/continuum split mode ("v_frac" is the validated default:'zero', 'v_ws', 'fixed').
    bound_energy_cut: float = 0.70
    # When mode="v_frac", use V_eff(r=bound_energy_cut*r_max) as threshold.
    bound_occ_mode: str = "fd"
    # Occupation used for n_bound and therefore n_full in the SCF map.
    # Starrett2013 applies the pressure-ionization weight M(e) to n_ion
    # [Eq. (79)], not to the total bound density in n_full. Keep "fd" as the
    # production default; "fd_m" is retained as a diagnostic/legacy option.
    ion_cut_mode: str = "starrett"
    # Radial cutoff used when assembling n_ion from bound states.
    ion_cut_c: float = 0.05
    # Width parameter c in the Starrett2013 f_cut(r) definition.
    ion_bound_gamma: float = 0.05
    # Fixed M(e) broadening width gamma (Ha) used when ion_gamma_mode="fixed".
    ion_gamma_mode: str = "scattering"
    # Bound-state broadening source for M(e): "scattering" or "fixed".
    ion_gamma_scale: float = 1.0
    # Multiplier for the scattering-derived gamma. The fixed-gamma path ignores
    # this knob so ion_bound_gamma remains an absolute width.
    ion_ws_weight_min: float = 0.0
    # Optional hard WS-localization filter applied when assembling n_ion.
    # Starrett2013 already localizes n_ion continuously through f_cut(r) and
    # M(E), so the public default keeps this extra binary filter disabled.
    bound_table_n_jobs: int = 1
    # Parallel workers used only for post-SCF bound tables/DOS construction.
    # Keep 1 by default to avoid multiprocessing re-entry issues in ad-hoc scripts.

    # ----- Runtime diagnostics -----
    neutrality_mode: str = "ws"
    # Neutrality target for full SCF ("ws" recommended).
    perf_diag: bool = False
    # Enable per-iteration performance diagnostics from low-level solver.
    perf_show_stage: bool = True
    # Print stage timing aggregate line.
    perf_show_basis: bool = True
    # Print basis metadata line.
    store_final_bound_debug: bool = False
    # If True, keep the final bound eigenpairs and ion-partition controls in
    # the in-memory result for exact post-SCF n_ion decomposition diagnostics.
    verbose: bool = False
    # Backward-compatible alias for show_scf_progress.
    print_every: int = 1
    # Log stride when verbose=True.
    show_scf_progress: bool = False
    # If True, print per-iteration SCF progress (full and external loops).

    # ----- Output controls -----
    save_data: bool = False
    # Save output dataset to disk.
    save_output_dir: str | Path = "outputs"
    # Target directory for output file.
    save_suffix: str = ""
    # Optional user suffix appended to default filename.

    def __post_init__(self) -> None:
        """
        Fill density-dependent default A3+B3 geometry when the user leaves it unset.

        The public default workflow now uses A3+B3 on both the full and
        external branches. To keep the minimal user input small, the continuum
        numerical radius and B3 handoff radii are inferred from density unless
        the user overrides them explicitly.
        """
        electronic_model = str(self.electronic_model).strip().lower().replace("-", "_")
        if electronic_model in {"ks", "ks_aa", "quantum"}:
            electronic_model = "qm"
        elif electronic_model in {"thomas_fermi", "thomasfermi"}:
            electronic_model = "tf"
        if electronic_model not in {"qm", "tf"}:
            raise ValueError("electronic_model must be 'qm' or 'tf'.")
        self.electronic_model = electronic_model
        self.gga_core_mode = str(self.gga_core_mode).strip().lower()
        if self.gga_core_mode not in {"finite", "strict"}:
            raise ValueError("gga_core_mode must be 'finite' or 'strict'.")
        if (
            not np.isfinite(float(self.gga_core_zr))
            or float(self.gga_core_zr) <= 0.0
        ):
            raise ValueError("gga_core_zr must be finite and positive.")
        if not (0.0 < float(self.tf_scf_mix) <= 1.0):
            raise ValueError("tf_scf_mix must lie in (0, 1].")
        if int(self.tf_scf_mixing_m) < 1:
            raise ValueError("tf_scf_mixing_m must be at least 1.")
        if float(self.tf_scf_mixing_w0) <= 0.0:
            raise ValueError("tf_scf_mixing_w0 must be positive.")
        if int(self.tf_scf_max_iter) < 1:
            raise ValueError("tf_scf_max_iter must be at least 1.")
        if int(self.tf_polish_max_iter) < 1:
            raise ValueError("tf_polish_max_iter must be at least 1.")
        if (
            float(self.tf_scf_tol) <= 0.0
            or float(self.tf_polish_tol) <= 0.0
            or float(self.tf_mu_tol) <= 0.0
        ):
            raise ValueError(
                "tf_scf_tol, tf_polish_tol, and tf_mu_tol must be positive."
            )
        if int(self.tf_quadrature_order) < 16:
            raise ValueError("tf_quadrature_order must be at least 16.")

        if self.cont_e_max is None:
            te_ha = max(float(self.temperature_ev) * EV_TO_HA, 1.0e-12)
            # Stage 1 needs a conservative floor. Pure 10*T works at moderate
            # and high temperatures, but it becomes too small for low-Te cold
            # starts and can break the inner-mu neutrality solve before the
            # continuum branch is established.
            self.cont_e_max = max(float(self.cont_e_min), 10.0 * te_ha, 5.0)

        if not np.isfinite(float(self.rmax_mult)) or float(self.rmax_mult) <= 0.0:
            raise ValueError("rmax_mult must be finite and positive.")
        if not np.isfinite(float(self.cont_e_min)) or float(self.cont_e_min) <= 0.0:
            raise ValueError("cont_e_min must be finite and positive.")
        if int(self.cont_near_zero_log_points_per_decade) < 0:
            raise ValueError("cont_near_zero_log_points_per_decade must be non-negative.")
        if int(self.cont_near_zero_log_max_nodes) < 0:
            raise ValueError("cont_near_zero_log_max_nodes must be non-negative.")
        if self.cont_near_zero_log_max_energy is not None:
            near_zero_max = float(self.cont_near_zero_log_max_energy)
            if not np.isfinite(near_zero_max) or near_zero_max <= 0.0:
                raise ValueError(
                    "cont_near_zero_log_max_energy must be finite and positive when set."
                )
        if int(self.cont_resonance_theta_l_min) < 0:
            raise ValueError("cont_resonance_theta_l_min must be non-negative.")
        if int(self.cont_resonance_theta_probe_count) < 0:
            raise ValueError("cont_resonance_theta_probe_count must be non-negative.")
        if int(self.cont_resonance_theta_scan_depth) < 0:
            raise ValueError("cont_resonance_theta_scan_depth must be non-negative.")
        if (
            self.cont_resonance_theta_scout_max_extra_nodes is not None
            and int(self.cont_resonance_theta_scout_max_extra_nodes) < 0
        ):
            raise ValueError(
                "cont_resonance_theta_scout_max_extra_nodes must be non-negative when set."
            )
        if self.cont_resonance_theta_root_tol is not None:
            root_tol = float(self.cont_resonance_theta_root_tol)
            if not np.isfinite(root_tol) or root_tol <= 0.0:
                raise ValueError(
                    "cont_resonance_theta_root_tol must be finite and positive when set."
                )
        sharpness = float(self.cont_resonance_theta_sharpness_min)
        if not np.isfinite(sharpness) or sharpness < 0.0:
            raise ValueError(
                "cont_resonance_theta_sharpness_min must be finite and non-negative."
            )
        if (
            self.cont_resonance_theta_max_roots is not None
            and int(self.cont_resonance_theta_max_roots) < 0
        ):
            raise ValueError("cont_resonance_theta_max_roots must be non-negative when set.")
        if (
            self.cont_resonance_theta_refine_depth is not None
            and int(self.cont_resonance_theta_refine_depth) < 0
        ):
            raise ValueError(
                "cont_resonance_theta_refine_depth must be non-negative when set."
            )
        if float(self.geometry_r_ws_floor_bohr) <= 0.0:
            raise ValueError("geometry_r_ws_floor_bohr must be positive.")
        if self.bound_rmax_mult is not None:
            bound_rmax_mult = float(self.bound_rmax_mult)
            if not np.isfinite(bound_rmax_mult) or bound_rmax_mult < float(self.rmax_mult):
                raise ValueError(
                    "bound_rmax_mult must be finite and >= rmax_mult when set."
                )
        if not (
            0.0 < float(self.bound_zero_tail_min_binding_ha)
            < float(self.bound_zero_tail_max_binding_ha)
        ):
            raise ValueError(
                "Require 0 < bound_zero_tail_min_binding_ha < "
                "bound_zero_tail_max_binding_ha."
            )
        if int(self.bound_zero_tail_scan_points) < 3:
            raise ValueError("bound_zero_tail_scan_points must be at least three.")
        if int(self.bound_zero_tail_l_max) < 0:
            raise ValueError("bound_zero_tail_l_max must be non-negative.")
        if not np.isfinite(float(self.bound_zero_tail_edge_rel_tol)) or float(
            self.bound_zero_tail_edge_rel_tol
        ) <= 0.0:
            raise ValueError("bound_zero_tail_edge_rel_tol must be finite and positive.")
        ion_gamma_mode = str(self.ion_gamma_mode).lower().strip()
        if ion_gamma_mode not in ("fixed", "scattering"):
            raise ValueError("ion_gamma_mode must be 'fixed' or 'scattering'.")
        self.ion_gamma_mode = ion_gamma_mode
        bound_occ_mode = str(self.bound_occ_mode).lower().strip()
        if bound_occ_mode not in ("fd", "fd_m"):
            raise ValueError("bound_occ_mode must be 'fd' or 'fd_m'.")
        self.bound_occ_mode = bound_occ_mode
        if float(self.ion_gamma_scale) <= 0.0:
            raise ValueError("ion_gamma_scale must be positive.")
        ion_cut_mode = str(self.ion_cut_mode).lower().strip()
        if ion_cut_mode not in ("starrett", "none"):
            raise ValueError("ion_cut_mode must be 'starrett' or 'none'.")
        self.ion_cut_mode = ion_cut_mode
        if float(self.ion_cut_c) < 0.0:
            raise ValueError("ion_cut_c must be non-negative.")
        if float(self.ion_ws_weight_min) < 0.0 or float(self.ion_ws_weight_min) > 1.0:
            raise ValueError("ion_ws_weight_min must be in [0, 1].")
        if float(self.full_v_eff_outer_decay_start_rws) < 0.0:
            raise ValueError("full_v_eff_outer_decay_start_rws must be non-negative.")
        if float(self.full_v_eff_outer_decay_length_rws) <= 0.0:
            raise ValueError("full_v_eff_outer_decay_length_rws must be positive.")
        if (
            not np.isfinite(float(self.veff_asymptotic_abs_tol_ha))
            or float(self.veff_asymptotic_abs_tol_ha) <= 0.0
        ):
            raise ValueError("veff_asymptotic_abs_tol_ha must be finite and positive.")
        if not np.isfinite(float(self.ph_kappa)) or float(self.ph_kappa) < 0.0:
            raise ValueError("ph_kappa must be finite and non-negative.")
        if int(self.ph_kappa_iters) < 0:
            raise ValueError("ph_kappa_iters must be non-negative.")
        if float(self.ph_kappa) > 0.0 and int(self.ph_kappa_iters) == 0:
            raise ValueError("ph_kappa_iters must be positive when ph_kappa > 0.")
        if float(self.ph_kappa) == 0.0 and int(self.ph_kappa_iters) > 0:
            raise ValueError("ph_kappa must be positive when ph_kappa_iters > 0.")
        if self.continuation_scf_mixing_scheme is not None:
            cont_scheme = str(self.continuation_scf_mixing_scheme).strip().lower()
            if cont_scheme not in ("linear", "eyert"):
                raise ValueError("continuation_scf_mixing_scheme must be 'linear', 'eyert', or None.")
            self.continuation_scf_mixing_scheme = cont_scheme
        if self.continuation_scf_mixing_m is not None and int(self.continuation_scf_mixing_m) < 1:
            raise ValueError("continuation_scf_mixing_m must be >= 1 when set.")
        if self.continuation_scf_mix is not None and float(self.continuation_scf_mix) <= 0.0:
            raise ValueError("continuation_scf_mix must be positive when set.")
        if (
            self.geometry_r_ws_cap_bohr is not None
            and float(self.geometry_r_ws_cap_bohr) <= 0.0
        ):
            raise ValueError("geometry_r_ws_cap_bohr must be positive when set.")
        if (
            self.geometry_r_ws_cap_bohr is not None
            and float(self.geometry_r_ws_cap_bohr) < float(self.geometry_r_ws_floor_bohr)
        ):
            raise ValueError("Require geometry_r_ws_cap_bohr >= geometry_r_ws_floor_bohr.")

        use_b3_full = str(self.b3_tail_stage1_mode).strip().lower() in ("post", "in_scf") or str(
            self.b3_tail_stage2_mode
        ).strip().lower() in ("post", "in_scf")
        use_b3_ext = str(self.ext_b3_tail_mode).strip().lower() == "in_scf"
        if not (use_b3_full or use_b3_ext):
            return

        cont_rmax_mult, b3_r_fit_max_mult, b3_r_cut_mult = _default_b3_geometry_from_density(self.rho_g_cc)
        if self.cont_rmax_mult is None:
            self.cont_rmax_mult = float(cont_rmax_mult)
        if self.b3_r_fit_max_mult is None:
            self.b3_r_fit_max_mult = float(b3_r_fit_max_mult)
        if (
            self.b3_r_cut_mult is None
            and self.b3_cut_width is None
            and self.cont_tail_r_cut_frac is None
        ):
            self.b3_r_cut_mult = float(b3_r_cut_mult)

        if (
            self.b3_r_cut_mult is not None
            and self.b3_r_fit_max_mult is not None
            and float(self.b3_r_cut_mult) >= float(self.b3_r_fit_max_mult)
        ):
            raise ValueError("Require b3_r_cut_mult < b3_r_fit_max_mult.")
        tail_model = str(self.b3_tail_model).strip().lower()
        if tail_model not in ("auto", "full", "a_only"):
            raise ValueError("b3_tail_model must be 'auto', 'full', or 'a_only'.")
        self.b3_tail_model = tail_model
        fit_window_mode = str(self.b3_tail_fit_window_mode).strip().lower()
        if fit_window_mode not in ("auto", "physical", "local"):
            raise ValueError(
                "b3_tail_fit_window_mode must be 'auto', 'physical', or 'local'."
            )
        self.b3_tail_fit_window_mode = fit_window_mode
        if self.b3_tail_local_fit_width_mult is not None:
            local_width_mult = float(self.b3_tail_local_fit_width_mult)
            if not np.isfinite(local_width_mult) or local_width_mult <= 0.0:
                raise ValueError(
                    "b3_tail_local_fit_width_mult must be finite and positive, or None."
                )
            self.b3_tail_local_fit_width_mult = local_width_mult
        if self.ext_b3_tail_model is not None:
            ext_tail_model = str(self.ext_b3_tail_model).strip().lower()
            if ext_tail_model not in ("auto", "full", "a_only"):
                raise ValueError("ext_b3_tail_model must be 'auto', 'full', or 'a_only'.")
            self.ext_b3_tail_model = ext_tail_model
        if isinstance(self.full_b3_use_source_closure, str):
            source_mode = str(self.full_b3_use_source_closure).strip().lower()
            if source_mode == "auto":
                self.full_b3_use_source_closure = None
            elif source_mode in ("true", "on", "yes", "1"):
                self.full_b3_use_source_closure = True
            elif source_mode in ("false", "off", "no", "0"):
                self.full_b3_use_source_closure = False
            else:
                raise ValueError(
                    "full_b3_use_source_closure must be True, False, None, or 'auto'."
                )
        elif self.full_b3_use_source_closure is not None:
            self.full_b3_use_source_closure = bool(self.full_b3_use_source_closure)
        if isinstance(self.ext_b3_use_source_closure, str):
            source_mode = str(self.ext_b3_use_source_closure).strip().lower()
            if source_mode == "auto":
                self.ext_b3_use_source_closure = None
            elif source_mode in ("true", "on", "yes", "1"):
                self.ext_b3_use_source_closure = True
            elif source_mode in ("false", "off", "no", "0"):
                self.ext_b3_use_source_closure = False
            else:
                raise ValueError(
                    "ext_b3_use_source_closure must be True, False, None, or 'auto'."
                )
        elif self.ext_b3_use_source_closure is not None:
            self.ext_b3_use_source_closure = bool(self.ext_b3_use_source_closure)
        if float(self.b3_tail_auto_rel_improve_tol) < 0.0:
            raise ValueError("b3_tail_auto_rel_improve_tol must be non-negative.")
        if float(self.b3_tail_auto_signal_rel_tol) < 0.0:
            raise ValueError("b3_tail_auto_signal_rel_tol must be non-negative.")
        if self.b3_charge_constraint_fit_rms_ratio_max is not None:
            fit_ratio_max = float(self.b3_charge_constraint_fit_rms_ratio_max)
            if not np.isfinite(fit_ratio_max) or fit_ratio_max <= 0.0:
                raise ValueError(
                    "b3_charge_constraint_fit_rms_ratio_max must be finite and "
                    "positive when set."
                )
        if self.b3_charge_constraint_profile_delta_rel_max is not None:
            profile_delta_rel_max = float(
                self.b3_charge_constraint_profile_delta_rel_max
            )
            if (
                not np.isfinite(profile_delta_rel_max)
                or profile_delta_rel_max <= 0.0
            ):
                raise ValueError(
                    "b3_charge_constraint_profile_delta_rel_max must be finite "
                    "and positive when set."
                )
        if float(self.ext_source_closure_auto_rel_tol) < 0.0:
            raise ValueError("ext_source_closure_auto_rel_tol must be non-negative.")

    def _use_manual_bound_basis(self) -> bool:
        """
        Return True when the user has explicitly overridden the bound basis.

        Notes
        -----
        The public default is now ``bound_basis_mode="auto"``. To preserve the
        historical manual path without adding more boilerplate for users, any
        explicit ``l_list`` or non-default ``n_states`` is treated as a manual
        override even when the mode is left at ``"auto"``.
        """
        mode = str(self.bound_basis_mode).strip().lower()
        if mode == "manual":
            return True
        if mode != "auto":
            raise ValueError("bound_basis_mode must be 'auto' or 'manual'.")
        return (self.l_list is not None) or (int(self.n_states) != 3)

    def resolved_bound_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the resolved bound basis as ``(l_list, n_states_by_l)``.

        In auto mode the default basis is inferred from the neutral-shell
        filling of the selected element. In manual mode, or whenever the user
        explicitly supplies ``l_list`` / a non-default ``n_states``, we fall
        back to the historical rectangular basis.
        """
        if not self._use_manual_bound_basis():
            z_nuc = int(element_info(self.element).z)
            return _auto_bound_basis_from_z(
                z_nuc,
                n_pad=int(self.bound_auto_n_pad),
                l_pad=int(self.bound_auto_l_pad),
            )

        n_states = max(int(self.n_states), 1)
        if self.l_list is None:
            l_arr = np.arange(0, max(n_states - 1, 1), dtype=int)
        else:
            l_arr = np.asarray(self.l_list, dtype=int)
        n_states_by_l = np.full(l_arr.shape, n_states, dtype=int)
        return l_arr, n_states_by_l

    def resolved_l_list(self) -> np.ndarray:
        """Return resolved angular-momentum channels for the bound solver."""
        l_arr, _ = self.resolved_bound_basis()
        return np.asarray(l_arr, dtype=int)

    def resolved_n_states_by_l(self) -> np.ndarray:
        """Return resolved per-l radial-state caps for the bound solver."""
        _, n_states_by_l = self.resolved_bound_basis()
        return np.asarray(n_states_by_l, dtype=int)

    def resolved_n_states(self) -> int:
        """Return the rectangular bound-state width used in saved arrays."""
        n_states_by_l = self.resolved_n_states_by_l()
        if n_states_by_l.size == 0:
            return 1
        return int(np.max(n_states_by_l))

    @property
    def citation_keys(self) -> tuple[str, ...]:
        """Primary papers for the full/external calculation and XC model."""
        return (
            "StarrettSaumon2014",
            "PillaiGoglioWalker2012",
            "WilsonEtAl2006",
            "StarrettEtAl2019",
            *citation_keys_for_xc_model(self.xc_model),
        )


def _ws_charge(
    r: np.ndarray,
    density: np.ndarray,
    r_ws: float,
    *,
    interpolate_boundary: bool = False,
) -> float:
    """
    Integrate one radial density inside the WS sphere.

    Keeping this in one helper avoids inconsistencies between terminal
    diagnostics, metadata, and saved output files.
    """
    r = np.asarray(r, dtype=float)
    density = np.asarray(density, dtype=float)
    if bool(interpolate_boundary):
        return _electron_count(r, density, float(r_ws), interpolate_boundary=True)
    mask = r <= float(r_ws)
    if not np.any(mask):
        return 0.0
    r_ws_slice = r[mask]
    n_ws_slice = density[mask]
    return float(
        4.0
        * np.pi
        * trapz_integral((r_ws_slice**2) * n_ws_slice, r_ws_slice)
    )


def _tail_shift_value(r: np.ndarray, v: np.ndarray, frac: float = 0.05) -> float:
    """Estimate a constant tail gauge shift using the outer radial fraction."""
    frac = min(max(float(frac), 1e-6), 0.5)
    r_cut = (1.0 - frac) * float(r[-1])
    mask = r >= r_cut
    if not np.any(mask):
        return float(v[-1])
    return float(np.median(v[mask]))


def _close_to_n0(
    r: np.ndarray,
    n: np.ndarray,
    n0: float,
    r_trust: float,
    blend_width: float,
) -> np.ndarray:
    """
    Keep n(r) untouched in the trusted region and blend to n0 outside.
    """
    out = np.asarray(n, dtype=float).copy()
    r = np.asarray(r, dtype=float)
    r_trust = min(max(float(r_trust), float(r[0])), float(r[-1]))
    width = max(float(blend_width), 0.0)
    if width <= 0.0:
        out[r >= r_trust] = float(n0)
        return out
    r_end = min(r_trust + width, float(r[-1]))
    if r_end <= r_trust:
        out[r >= r_trust] = float(n0)
        return out
    mask_hi = r >= r_end
    out[mask_hi] = float(n0)
    mask_blend = (r >= r_trust) & (r < r_end)
    if np.any(mask_blend):
        t = (r[mask_blend] - r_trust) / max(r_end - r_trust, 1e-14)
        s = t * t * (3.0 - 2.0 * t)  # smoothstep
        out[mask_blend] = (1.0 - s) * out[mask_blend] + s * float(n0)
    return out


def _repair_screening_density_tail(
    r: np.ndarray,
    n_scr: np.ndarray,
    *,
    zbar: float,
    mu_id: float,
    temperature: float,
    r_cut: float,
    r_fit_max: float | None,
    charge_weight: float,
) -> tuple[np.ndarray, dict[str, float | bool | str | None]]:
    """
    Refit the screening-density tail on a constrained B3/Friedel basis.

    Parameters
    ----------
    r
        Radial grid in Bohr.
    n_scr
        Screening density on `r` in Bohr^-3.
    zbar
        Target partition charge.  The production full/external caller uses
        ``Z - integral(n_ion)`` over the complete pseudoatom box, consistent
        with Starrett--Saumon Eqs. (73), (75), and (76).
    mu_id
        Chemical potential used for the B3 tail parameters in Ha.
    temperature
        Electron temperature in Ha.
    r_cut
        Left edge of the tail replacement region in Bohr.
    r_fit_max
        Right edge of the fit window in Bohr. If None, use the remainder of
        the available grid.
    charge_weight
        Relative weight of the integrated-charge constraint.

    Returns
    -------
    n_scr_out, meta
        Repaired screening density and fit diagnostics.

    Notes
    -----
    The raw `n_scr = n_full - n_ext - n_ion` profile can be noisy in the outer
    region because it subtracts two large continuum densities that were
    reconstructed independently. This helper keeps the inner profile untouched
    and replaces only the tail by a direct fit of `n_scr` itself to the same
    linear-response basis used in the B3 continuum model:

      n_scr(r) = A exp(-k_TF r)/r
               + B exp(-2 b0 r)/r^3 sin(2 a0 r + delta)

    We fit the equivalent linear `(A, C, D)` basis on `[r_cut, r_fit_max]`
    and add one extra least-squares row that enforces

      4*pi*int r^2 n_scr(r) dr = Zbar.
    """
    r_arr = np.asarray(r, dtype=float)
    n_arr = np.asarray(n_scr, dtype=float)
    if r_arr.shape != n_arr.shape:
        raise ValueError("r and n_scr must have the same shape.")

    idx_cut = int(np.searchsorted(r_arr, float(r_cut)))
    if idx_cut <= 1 or idx_cut >= r_arr.size - 2:
        raise ValueError("screening tail repair requires an interior r_cut.")

    idx_fit_max = int(r_arr.size)
    if r_fit_max is not None:
        idx_fit_max = int(np.searchsorted(r_arr, float(r_fit_max), side="right"))
        idx_fit_max = min(max(idx_fit_max, idx_cut + 3), int(r_arr.size))
    if idx_fit_max - idx_cut < 3:
        raise ValueError("screening tail repair needs at least three fit points.")

    r_fit = r_arr[idx_cut:idx_fit_max]
    y_fit = n_arr[idx_cut:idx_fit_max]
    a0, b0, k_tf = tail_parameters(mu_id, temperature)

    # (1) Build the linearized Friedel basis on the trusted fit window.
    f0 = np.exp(-k_tf * r_fit) / r_fit
    decay_fit = np.exp(-2.0 * b0 * r_fit) / (r_fit**3)
    f1 = decay_fit * np.sin(2.0 * a0 * r_fit)
    f2 = decay_fit * np.cos(2.0 * a0 * r_fit)
    basis_fit = np.column_stack((f0, f1, f2))

    # (2) Add one global charge-closure equation for the replaced tail.
    r_tail = r_arr[idx_cut:]
    q_inner = 4.0 * np.pi * trapz_integral(
        (r_arr[:idx_cut] ** 2) * n_arr[:idx_cut],
        r_arr[:idx_cut],
    )
    target_tail_charge = float(zbar) - float(q_inner)

    f0_tail = np.exp(-k_tf * r_tail) / r_tail
    decay_tail = np.exp(-2.0 * b0 * r_tail) / (r_tail**3)
    f1_tail = decay_tail * np.sin(2.0 * a0 * r_tail)
    f2_tail = decay_tail * np.cos(2.0 * a0 * r_tail)
    basis_tail = np.column_stack((f0_tail, f1_tail, f2_tail))
    charge_row = np.asarray(
        [
            4.0
            * np.pi
            * trapz_integral((r_tail**2) * basis_tail[:, col], r_tail)
            for col in range(3)
        ],
        dtype=float,
    )

    weight = max(float(charge_weight), 0.0)
    if weight > 0.0:
        basis_aug = np.vstack((basis_fit, weight * charge_row[np.newaxis, :]))
        y_aug = np.concatenate((y_fit, np.asarray([weight * target_tail_charge], dtype=float)))
    else:
        basis_aug = basis_fit
        y_aug = y_fit

    coeffs, _, _, _ = np.linalg.lstsq(basis_aug, y_aug, rcond=None)

    # (3) Keep the inner profile unchanged and replace only the outer tail.
    n_out = np.asarray(n_arr, dtype=float).copy()
    n_out[idx_cut:] = basis_tail @ coeffs

    q_raw = 4.0 * np.pi * trapz_integral((r_arr**2) * n_arr, r_arr)
    q_new = 4.0 * np.pi * trapz_integral((r_arr**2) * n_out, r_arr)
    meta: dict[str, float | bool | str | None] = {
        "mode": "constrained_b3",
        "applied": True,
        "r_cut": float(r_arr[idx_cut]),
        "r_fit_max": float(r_arr[idx_fit_max - 1]),
        "fit_points": int(idx_fit_max - idx_cut),
        "charge_weight": float(weight),
        "q_scr_raw": float(q_raw),
        "q_scr_repaired": float(q_new),
        "q_scr_target": float(zbar),
        "q_scr_rel_raw": abs(float(q_raw) - float(zbar)) / max(abs(float(zbar)), 1.0e-12),
        "q_scr_rel_repaired": abs(float(q_new) - float(zbar)) / max(abs(float(zbar)), 1.0e-12),
        "A": float(coeffs[0]),
        "C": float(coeffs[1]),
        "D": float(coeffs[2]),
        "a0": float(a0),
        "b0": float(b0),
        "k_tf": float(k_tf),
    }
    return n_out, meta


def _normalize_b3_stage_mode(mode: str | None, *, legacy_tail_match: bool = False) -> str:
    """
    Normalize the user-facing B3 mode for one SCF stage.

    The new API uses explicit names that mention B3 directly. We still accept
    the historical boolean `cont_tail_match` as a compatibility alias.
    """
    mode_norm = str(mode or "").strip().lower()
    if mode_norm in ("", "none", "false"):
        mode_norm = "in_scf" if legacy_tail_match else "off"
    if mode_norm not in ("off", "post", "in_scf"):
        raise ValueError("B3 stage mode must be 'off', 'post', or 'in_scf'.")
    return mode_norm


def _resolve_b3_tail_controls(
    cfg: FullExternalConfig,
    *,
    r_ws: float,
    rmax: float,
    stage_mode: str,
) -> dict[str, Any]:
    """
    Convert all high-level B3 controls into physical radii in Bohr.

    User inputs are expressed relative to R_ws so the same settings scale
    sensibly across different materials and densities.
    """
    mode = _normalize_b3_stage_mode(stage_mode, legacy_tail_match=bool(cfg.cont_tail_match))
    target = str(cfg.b3_tail_target or cfg.cont_tail_match_target).strip().lower()
    if target not in ("cont", "full", "both"):
        raise ValueError("b3_tail_target must be 'cont', 'full', or 'both'.")
    fit_window_mode = str(cfg.b3_tail_fit_window_mode).strip().lower()
    if fit_window_mode == "auto" and target == "cont":
        fit_window_mode = "local"

    geometry = _resolve_outer_geometry(cfg, r_ws=r_ws)
    rmax_eff = float(geometry["rmax"])

    solve_rmax = None
    if mode == "in_scf" and cfg.cont_rmax_mult is not None:
        solve_rmax = float(geometry["solve_rmax"]) if geometry["solve_rmax"] is not None else float(rmax_eff)

    r_fit_max = None
    if cfg.b3_r_fit_max_mult is not None:
        r_fit_max = float(geometry["r_fit_max"]) if geometry["r_fit_max"] is not None else None
    elif cfg.source_r_trust_frac is not None:
        fit_ref = float(solve_rmax) if solve_rmax is not None else float(rmax_eff)
        r_fit_max = float(cfg.source_r_trust_frac) * fit_ref
    if solve_rmax is not None and r_fit_max is not None:
        r_fit_max = min(float(r_fit_max), float(solve_rmax))
    elif r_fit_max is not None:
        r_fit_max = min(float(r_fit_max), float(rmax_eff))

    r_cut = None
    if cfg.b3_r_cut_mult is not None:
        r_cut = float(geometry["r_cut"]) if geometry["r_cut"] is not None else None
    elif cfg.cont_tail_r_cut_frac is not None:
        r_cut = float(cfg.cont_tail_r_cut_frac) * float(rmax_eff)
    else:
        cut_width = None
        if cfg.b3_cut_width is not None:
            cut_width = float(cfg.b3_cut_width) * float(geometry["r_geom"])
        elif cfg.cont_tail_cut_width is not None:
            cut_width = float(cfg.cont_tail_cut_width)
        if cut_width is not None:
            base = float(r_fit_max) if r_fit_max is not None else float(cfg.source_r_trust_frac) * float(rmax_eff)
            r_cut = max(1e-8, base - cut_width)

    if r_cut is not None:
        cut_upper = float(r_fit_max) if r_fit_max is not None else (float(solve_rmax) if solve_rmax is not None else float(rmax_eff))
        r_cut = min(float(r_cut), max(1.0e-8, cut_upper - 1.0e-8))

    return {
        "mode": mode,
        "target": target,
        "solve_rmax": solve_rmax,
        "r_fit_max": r_fit_max,
        "r_cut": r_cut,
        "fit_points": int(cfg.b3_tail_fit_points or cfg.cont_tail_fit_points),
        "local_fit_width": (
            None
            if cfg.b3_tail_local_fit_width_mult is None
            else float(cfg.b3_tail_local_fit_width_mult) * float(r_ws)
        ),
        "fit_window_mode": str(fit_window_mode),
        "blend_points": int(cfg.b3_tail_blend_points or cfg.cont_tail_blend_points),
        "model": str(cfg.b3_tail_model),
        "auto_rel_improve_tol": float(cfg.b3_tail_auto_rel_improve_tol),
        "auto_signal_rel_tol": float(cfg.b3_tail_auto_signal_rel_tol),
        "fallback_on_error": bool(cfg.b3_fallback_on_error and cfg.cont_tail_fallback_on_error),
    }


def _apply_b3_post_to_full_result(
    result: dict[str, Any],
    cfg: FullExternalConfig,
    *,
    r_ws: float,
    rmax: float,
    stage_mode: str,
) -> dict[str, Any]:
    """
    Replace only the final full-continuum tail by B3 after SCF convergence.

    This keeps the pure-A3 SCF fixed point intact while letting us test whether
    the B3 tail itself is physically sensible.  The returned postprocessed
    density is not a new SCF fixed point: ``v_full`` remains the potential that
    generated the A3 orbitals.  We therefore expose the potential rebuilt from
    the post-B3 density separately as ``v_post_from_density`` rather than
    labelling it ``v_scf``.

    Notes
    -----
    Starrett & Saumon (2014), Appendix B, use Eq. (B3) inside their iterative
    density construction and require convergence with respect to the matching
    radius.  This ``post`` mode is intentionally a tail-sensitivity diagnostic;
    ``in_scf`` is the self-consistent B3 path.
    """
    controls = _resolve_b3_tail_controls(cfg, r_ws=r_ws, rmax=rmax, stage_mode=stage_mode)
    if controls["mode"] != "post":
        return result
    if controls["target"] not in ("cont", "both"):
        raise ValueError("Post-B3 currently supports b3_tail_target='cont' or 'both' only.")
    if controls["r_cut"] is None:
        return result

    r = np.asarray(result["r"], dtype=float)
    n_cont = np.asarray(result["n_cont"], dtype=float)
    n_bound = np.asarray(result["n_bound"], dtype=float)
    mu = float(result["mu"])
    temperature_ha = float(cfg.temperature_ev) * EV_TO_HA
    n0 = float(result["n0"])
    idx_cut = int(np.searchsorted(r, float(controls["r_cut"])))
    if idx_cut <= 0 or idx_cut >= r.size - 2:
        return result

    fit_points = int(controls["fit_points"])
    if controls["r_fit_max"] is not None:
        idx_fit_max = int(np.searchsorted(r, float(controls["r_fit_max"]), side="right"))
        if idx_fit_max - idx_cut < 3:
            return result
    if fit_points < 3:
        return result

    try:
        n_cont_b3, tail_meta = apply_tail_match(
            r,
            n_cont,
            n0,
            mu,
            temperature_ha,
            idx_cut,
            fit_points=fit_points,
            r_fit_max=(
                controls["r_fit_max"]
                if str(controls["fit_window_mode"]) in ("auto", "physical")
                else None
            ),
            local_fit_width=controls["local_fit_width"],
            fit_window_mode=str(controls["fit_window_mode"]),
            blend_points=int(controls["blend_points"]),
            model=str(controls["model"]),
            auto_rel_improve_tol=float(controls["auto_rel_improve_tol"]),
            auto_signal_rel_tol=float(controls["auto_signal_rel_tol"]),
        )
    except Exception:
        if not bool(controls["fallback_on_error"]):
            raise
        return result

    out = dict(result)
    out["n_cont"] = np.asarray(n_cont_b3, dtype=float)
    out["n_full"] = np.asarray(n_bound + n_cont_b3, dtype=float)
    out["v_full_fixed_point"] = np.asarray(result["v_full"], dtype=float).copy()
    out["v_post_from_density"] = effective_potential_full(
        r,
        np.asarray(out["n_full"], dtype=float),
        n0,
        np.asarray(out["g_ii"], dtype=float),
        float(out["Z"]),
        xc_model=cfg.xc_model,
        kappa=0.0,
        ion_sphere_radius=(
            float(r_ws)
            if _uses_analytic_ion_sphere_background(
                cfg, r_ws=float(r_ws), rmax=float(rmax)
            )
            else None
        ),
        gga_core_mode=cfg.gga_core_mode,
        gga_core_zr=cfg.gga_core_zr,
    )
    out["b3_tail_meta"] = {
        **dict(tail_meta),
        "self_consistent": False,
        "diagnostic_only": True,
        "background_n0_mode": str(_resolve_n0_mode(cfg, b3_tail_mode="post")),
    }
    out["b3_post_self_consistent"] = False
    out["b3_tail_r_cut"] = float(controls["r_cut"])
    out["b3_tail_r_fit_max"] = None if controls["r_fit_max"] is None else float(controls["r_fit_max"])
    return out


def _build_continuum_params(
    cfg: FullExternalConfig,
    *,
    l_max: int,
    r_ws: float,
    rmax: float,
    adaptive_mode: str,
    b3_stage_mode: str,
    e_max_mode: str,
    for_external: bool = False,
) -> dict[str, Any]:
    """
    Build one continuum control dictionary shared by full/external branches.
    """
    b3 = _resolve_b3_tail_controls(cfg, r_ws=r_ws, rmax=rmax, stage_mode=b3_stage_mode)
    solve_rmax = float(b3["solve_rmax"]) if b3["solve_rmax"] is not None else float(rmax)
    e_max_value = max(float(cfg.cont_e_max), float(cfg.cont_e_min))
    l_pad = 2
    l_max_value = min(int(np.ceil(np.sqrt(2.0 * e_max_value) * solve_rmax + float(l_pad))), 150)

    tail_model = str(b3["model"])
    if for_external and cfg.ext_b3_tail_model is not None:
        tail_model = str(cfg.ext_b3_tail_model).strip().lower()

    use_full_b3_source_closure = _resolve_full_b3_source_closure_policy(
        setting=cfg.full_b3_use_source_closure,
        tail_target=str(b3["target"]),
    )
    params: dict[str, Any] = {
        "l_max": int(max(l_max_value, 0)),
        "l_max_ceiling": 150,
        "l_pad": int(l_pad),
        "e_min": float(cfg.cont_e_min),
        "near_zero_log_grid": bool(cfg.cont_near_zero_log_grid),
        "near_zero_log_points_per_decade": int(
            cfg.cont_near_zero_log_points_per_decade
        ),
        "near_zero_log_max_nodes": int(cfg.cont_near_zero_log_max_nodes),
        "near_zero_log_max_energy": (
            None
            if cfg.cont_near_zero_log_max_energy is None
            else float(cfg.cont_near_zero_log_max_energy)
        ),
        "resonance_theta_l_min": int(cfg.cont_resonance_theta_l_min),
        "resonance_theta_probe_count": int(cfg.cont_resonance_theta_probe_count),
        "resonance_theta_scan_depth": int(cfg.cont_resonance_theta_scan_depth),
        "resonance_theta_scout_max_extra_nodes": (
            None
            if cfg.cont_resonance_theta_scout_max_extra_nodes is None
            else int(cfg.cont_resonance_theta_scout_max_extra_nodes)
        ),
        "resonance_theta_root_tol": (
            None
            if cfg.cont_resonance_theta_root_tol is None
            else float(cfg.cont_resonance_theta_root_tol)
        ),
        "resonance_theta_sharpness_min": float(
            cfg.cont_resonance_theta_sharpness_min
        ),
        "resonance_theta_max_roots": (
            None
            if cfg.cont_resonance_theta_max_roots is None
            else int(cfg.cont_resonance_theta_max_roots)
        ),
        "resonance_theta_refine_depth": (
            None
            if cfg.cont_resonance_theta_refine_depth is None
            else int(cfg.cont_resonance_theta_refine_depth)
        ),
        "e_max": float(e_max_value),
        "e_max_mode": str(e_max_mode),
        "e_max_occ_tol": float(cfg.cont_stage2_e_max_occ_tol),
        "e_max_floor": float(cfg.cont_stage2_e_max_floor),
        "n_e": int(cfg.cont_n_e_linear),
        "n_jobs": int(cfg.cont_n_jobs),
        "energy_mode": str(cfg.cont_energy_mode),
        "l_cap_strategy": str(cfg.cont_l_cap_strategy),
        "match_fraction": float(cfg.cont_match_fraction),
        "match_fraction_mode": "r",
        # Match-window geometry must track the numerical continuum solve box,
        # not the outer postprocessed/full box.
        "match_r_cut": float(cfg.cont_match_r_cut_frac) * solve_rmax,
        "match_width": float(cfg.cont_match_width_frac) * solve_rmax,
        "match_kr_min": float(cfg.cont_match_kr_min),
        "match_v_tol": float(cfg.cont_match_v_tol),
        "match_min_points": int(cfg.cont_match_min_points),
        "match_asymptotic": "auto",
        "ext_match_v_tol": cfg.ext_match_v_tol,
        "ext_energy_floor_tail_frac": float(cfg.ext_energy_floor_tail_frac),
        "ext_energy_floor_margin": float(cfg.ext_energy_floor_margin),
        "tail_mode": str(b3["mode"]),
        "tail_match": bool(b3["mode"] == "in_scf"),
        "tail_match_target": str(b3["target"]),
        "tail_fit_points": int(b3["fit_points"]),
        "tail_local_fit_width": b3["local_fit_width"],
        "tail_fit_window_mode": str(b3["fit_window_mode"]),
        "tail_blend_points": int(b3["blend_points"]),
        "tail_model": str(tail_model),
        "tail_auto_rel_improve_tol": float(b3["auto_rel_improve_tol"]),
        "tail_auto_signal_rel_tol": float(b3["auto_signal_rel_tol"]),
        "tail_fallback_on_error": bool(b3["fallback_on_error"]),
        "source_closure": bool(cfg.source_closure),
        "source_closure_when_b3": bool(use_full_b3_source_closure),
        "source_r_trust": float(cfg.source_r_trust_frac) * solve_rmax,
        "source_blend_width": float(cfg.source_blend_frac) * solve_rmax,
        "source_r_trust_frac": float(cfg.source_r_trust_frac),
        "source_blend_frac": float(cfg.source_blend_frac),
        "source_charge_closure": bool(cfg.source_charge_closure),
        "b3_source_charge_constraint": bool(cfg.b3_source_charge_constraint),
        "b3_charge_constraint_fit_rms_ratio_max": (
            None
            if cfg.b3_charge_constraint_fit_rms_ratio_max is None
            else float(cfg.b3_charge_constraint_fit_rms_ratio_max)
        ),
        "b3_charge_constraint_profile_delta_rel_max": (
            None
            if cfg.b3_charge_constraint_profile_delta_rel_max is None
            else float(cfg.b3_charge_constraint_profile_delta_rel_max)
        ),
        "ext_source_closure_when_b3": cfg.ext_b3_use_source_closure,
        "ext_source_closure_auto_rel_tol": float(cfg.ext_source_closure_auto_rel_tol),
        "analytic_ion_sphere_background": bool(
            _uses_analytic_ion_sphere_background(cfg, r_ws=float(r_ws), rmax=float(rmax))
        ),
    }
    if b3["r_cut"] is not None:
        params["tail_r_cut"] = float(b3["r_cut"])
    if b3["r_fit_max"] is not None:
        params["tail_r_fit_max"] = float(b3["r_fit_max"])
    if b3["solve_rmax"] is not None:
        params["solve_rmax"] = float(b3["solve_rmax"])
    if str(cfg.cont_energy_mode).lower() == "adaptive":
        params["adaptive_mode"] = str(adaptive_mode)
        params["n_e_base"] = int(cfg.cont_n_e_base)
        params["e_base_grid"] = str(cfg.cont_e_base_grid)
        params["e_tol"] = float(cfg.cont_e_tol)
        params["e_max_depth"] = int(cfg.cont_e_max_depth)
        params["e_min_width"] = float(cfg.cont_dE_min)
        params["adaptive_parallel_mode"] = str(cfg.cont_parallel_mode)
        params["adaptive_shards"] = cfg.cont_shards
        params["adaptive_shard_policy"] = str(cfg.cont_shard_policy)
        params["adaptive_reuse_basis"] = bool(cfg.cont_adaptive_reuse_basis)
    return params


def _resolve_full_b3_source_closure_policy(
    *,
    setting: bool | None,
    tail_target: str,
) -> bool:
    """
    Resolve source closure for the central/full B3 branch.

    Starrett & Saumon (2014), Appendix B, fit Eq. (B3) separately to the total
    full and external densities.  When the selected target is ``full`` (or
    ``both``), applying the historical continuum source closure afterwards
    first forces ``n_cont`` to ``n0`` and then re-adds ``n_bound``.  A diffuse
    bound tail therefore makes the potential source approach
    ``n0 + n_bound`` rather than the already matched ``n_full -> n0``.

    Auto mode consequently disables this extra closure for full-density B3,
    while preserving it for the legacy continuum-target path.
    """
    if setting is not None:
        return bool(setting)
    target = str(tail_target).strip().lower()
    return target not in ("full", "both")


def _resolve_ext_source_closure_policy(
    *,
    setting: bool | None,
    r: np.ndarray,
    n_ext_candidate: np.ndarray,
    n0: float,
    solve_rmax: float | None,
    tail_r_fit_max: float | None,
    r_trust: float,
    blend_width: float,
    rel_tol: float,
) -> tuple[bool, dict[str, Any]]:
    """
    Decide whether external source-closure should be applied on this iteration.

    Notes
    -----
    The external-B3 branch previously always applied source_closure when the
    flag was enabled. For light species with short continuum boxes this could
    flatten `n_ext` well before the rebuilt B3 tail had naturally relaxed to
    `n0`, which then fed directly into `V_ext`, `n_scr`, and the downstream
    HNC kernel. In auto mode, only enable source_closure once the candidate
    rebuilt `n_ext` is already sufficiently close to `n0` near the trusted
    radius and has also flattened across the pre-trust tail segment. This
    avoids switching on source_closure while the B3 tail is still visibly
    relaxing toward `n0`.
    """
    if setting is True:
        return True, {"mode": "forced_on"}
    if setting is False:
        return False, {"mode": "forced_off"}

    r_arr = np.asarray(r, dtype=float)
    n_arr = np.asarray(n_ext_candidate, dtype=float)
    if r_arr.size == 0 or n_arr.size != r_arr.size:
        return False, {"mode": "auto", "reason": "invalid_grid"}

    r_trust_val = min(max(float(r_trust), float(r_arr[0])), float(r_arr[-1]))
    r_hi = min(r_trust_val + max(float(blend_width), 0.0), float(r_arr[-1]))
    if solve_rmax is not None:
        r_hi = min(float(r_hi), float(solve_rmax))
    if r_hi <= r_trust_val:
        r_hi = min(
            float(solve_rmax) if solve_rmax is not None else float(r_arr[-1]),
            float(r_arr[-1]),
        )

    mask = (r_arr >= r_trust_val) & (r_arr <= r_hi)
    if not np.any(mask):
        return False, {"mode": "auto", "reason": "empty_probe_window"}

    scale = max(abs(float(n0)), 1.0e-12)
    rel_dev = float(np.max(np.abs(n_arr[mask] - float(n0))) / scale)
    pre_lo = r_arr[0] if tail_r_fit_max is None else float(tail_r_fit_max)
    pre_lo = min(max(float(pre_lo), float(r_arr[0])), float(r_trust_val))
    pre_mask = (r_arr >= pre_lo) & (r_arr <= r_trust_val)
    if int(np.count_nonzero(pre_mask)) < 2:
        pre_mask = mask

    pre_vals = n_arr[pre_mask]
    pre_rel_dev = float(np.max(np.abs(pre_vals - float(n0))) / scale)
    pre_rel_span = float((np.max(pre_vals) - np.min(pre_vals)) / scale)
    pre_rel_trend = float(abs(float(pre_vals[-1]) - float(pre_vals[0])) / scale)
    tol = max(float(rel_tol), 0.0)
    apply = bool(
        rel_dev <= tol
        and pre_rel_dev <= tol
        and pre_rel_span <= tol
    )
    return apply, {
        "mode": "auto",
        "rel_dev_max": float(rel_dev),
        "pretrust_rel_dev_max": float(pre_rel_dev),
        "pretrust_rel_span": float(pre_rel_span),
        "pretrust_rel_trend": float(pre_rel_trend),
        "rel_tol": float(rel_tol),
        "pretrust_r_lo": float(r_arr[pre_mask][0]),
        "pretrust_r_hi": float(r_arr[pre_mask][-1]),
        "probe_r_lo": float(r_trust_val),
        "probe_r_hi": float(r_hi),
    }


def _resolve_n0_mode(cfg: FullExternalConfig, *, b3_tail_mode: str) -> str:
    """
    Resolve the low-level KS `n0_mode` for one AA stage.

    Parameters
    ----------
    cfg
        High-level full/external configuration.
    b3_tail_mode
        Resolved B3 stage mode (`"off"`, `"post"`, or `"in_scf"`).

    Returns
    -------
    str
        Low-level `KSDTFConfig.n0_mode`.

    Notes
    -----
    The validated default policy is:

    - `ideal` when B3 participates inside SCF
    - `window` for the pure-A3 baseline

    `cfg.n0_mode_override` exists as a diagnostic hook so the user can test
    whether an observed A3/B3 difference is driven by the B3 handoff geometry
    itself or by the associated choice of uniform background closure.
    """
    override = cfg.n0_mode_override
    if override is None:
        return "ideal" if str(b3_tail_mode).strip().lower() == "in_scf" else "window"

    mode = str(override).strip().lower()
    if mode not in ("ideal", "window", "tail", "fixed"):
        raise ValueError(
            "n0_mode_override must be one of 'ideal', 'window', 'tail', or 'fixed'."
        )
    if mode == "fixed" and cfg.n0_fixed_override is None:
        raise ValueError(
            "n0_mode_override='fixed' requires n0_fixed_override to be set."
        )
    return mode


def _build_ks_config(
    cfg: FullExternalConfig,
    *,
    z_nuc: int,
    temperature_ha: float,
    n_i: float,
    r_ws: float,
    rmax: float,
    mu_guess: float,
    mu_bounds: tuple[float, float],
    max_iter: int,
    cont_params: dict[str, Any],
    compute_external: bool,
    v_full_init: np.ndarray | None = None,
    v_ext_init: np.ndarray | None = None,
    v_corr_full: np.ndarray | None = None,
    v_corr_ext: np.ndarray | None = None,
    scf_mix_override: float | None = None,
    scf_mixing_scheme_override: str | None = None,
    scf_mixing_m_override: int | None = None,
) -> KSDTFConfig:
    """
    Translate high-level API config into low-level KSDTFConfig.
    """
    # B3 used inside SCF must share the same uniform background n0 as the
    # Starrett Eq.(4) potential assembly. In practice this means using the
    # ideal-gas n0(mu, Te), not the outer-window estimate that works for the
    # pure-A3 baseline. Otherwise the tail model can feed its own asymptotic
    # value back into the window estimator and flatten n_cont unphysically.
    b3_tail_mode = str(cont_params.get("tail_mode", "off")).strip().lower()
    n0_mode = _resolve_n0_mode(cfg, b3_tail_mode=b3_tail_mode)
    scf_mix = float(cfg.scf_mix if scf_mix_override is None else scf_mix_override)
    scf_mixing_scheme = str(
        cfg.scf_mixing_scheme
        if scf_mixing_scheme_override is None
        else scf_mixing_scheme_override
    )
    scf_mixing_m = int(
        cfg.scf_mixing_m if scf_mixing_m_override is None else scf_mixing_m_override
    )
    full_fixed_mu = None if compute_external else cfg.full_fixed_mu_ha
    mu_mode = "fixed" if full_fixed_mu is not None else "neutral"
    mu_value = float(full_fixed_mu) if full_fixed_mu is not None else float(mu_guess)

    return KSDTFConfig(
        Z=int(z_nuc),
        temperature=float(temperature_ha),
        mu=mu_value,
        mu_mode=mu_mode,
        mu_strategy="inner",
        mu_bounds=tuple(float(x) for x in mu_bounds),
        mu_solver="brent",
        mu_tol=1e-10,
        mu_max_iter=100,
        mu_verbose=False,
        n_i=float(n_i),
        r_ws=float(r_ws),
        rmax=float(rmax),
        rmax_mult=None,
        n_points=int(cfg.n_points),
        bound_rmax=(
            None
            if cfg.bound_rmax_mult is None
            else max(
                float(rmax),
                float(cfg.bound_rmax_mult) * (float(rmax) / float(cfg.rmax_mult)),
            )
        ),
        bound_zero_tail_refine=bool(cfg.bound_zero_tail_refine),
        bound_zero_tail_min_binding=float(cfg.bound_zero_tail_min_binding_ha),
        bound_zero_tail_max_binding=float(cfg.bound_zero_tail_max_binding_ha),
        bound_zero_tail_scan_points=int(cfg.bound_zero_tail_scan_points),
        bound_zero_tail_l_max=int(cfg.bound_zero_tail_l_max),
        bound_zero_tail_edge_rel_tol=float(cfg.bound_zero_tail_edge_rel_tol),
        l_list=cfg.resolved_l_list(),
        n_states=cfg.resolved_n_states(),
        n_states_by_l=cfg.resolved_n_states_by_l(),
        xc_model=str(cfg.xc_model),
        gga_core_mode=str(cfg.gga_core_mode),
        gga_core_zr=float(cfg.gga_core_zr),
        continuum_model="scattering",
        continuum_params=dict(cont_params),
        compute_external=bool(compute_external),
        mix=scf_mix,
        max_iter=int(max_iter),
        tol=float(cfg.scf_tol),
        # PH changes the initial long-range Hartree map.  Keep every
        # PH-enabled mixed iterate in the physical V(infinity)=0 gauge so the
        # later ordinary-Poisson fixed point cannot inherit a screened-tail
        # constant and manufacture a spurious threshold state.
        shift_v_eff_tail=bool(
            float(cfg.ph_kappa) > 0.0 and int(cfg.ph_kappa_iters) > 0
        ),
        full_v_eff_outer_decay=bool(cfg.full_v_eff_outer_decay),
        full_v_eff_outer_decay_start_rws=float(cfg.full_v_eff_outer_decay_start_rws),
        full_v_eff_outer_decay_length_rws=float(cfg.full_v_eff_outer_decay_length_rws),
        bound_energy_cut_mode=str(cfg.bound_energy_cut_mode),
        bound_energy_cut=float(cfg.bound_energy_cut),
        ph_kappa=(
            float(cfg.ph_kappa)
            if float(cfg.ph_kappa) > 0.0 and int(cfg.ph_kappa_iters) > 0
            else 0.0
        ),
        ph_kappa_iters=(
            int(cfg.ph_kappa_iters)
            if float(cfg.ph_kappa) > 0.0 and int(cfg.ph_kappa_iters) > 0
            else 0
        ),
        n0_mode=n0_mode,
        n0_fixed=(
            float(cfg.n0_fixed_override) if cfg.n0_fixed_override is not None else None
        ),
        n0_window_lo_frac=0.75,
        n0_window_hi_frac=0.80,
        n0_window_mode="mean",
        n0_window_direct=True,
        n0_window_mix=0.2,
        mixing_scheme=scf_mixing_scheme,
        mixing_m=scf_mixing_m,
        mixing_w0=float(cfg.scf_mixing_w0),
        neutrality_mode=str(cfg.neutrality_mode),
        charge_tol=None,
        zbar_tol=None,
        bound_occ_mode=str(cfg.bound_occ_mode),
        ion_cut_mode=str(cfg.ion_cut_mode),
        ion_cut_c=float(cfg.ion_cut_c),
        ion_bound_gamma=float(cfg.ion_bound_gamma),
        ion_gamma_mode=str(cfg.ion_gamma_mode),
        ion_gamma_scale=float(cfg.ion_gamma_scale),
        ion_ws_weight_min=float(cfg.ion_ws_weight_min),
        g_ii_override=(
            None
            if cfg.g_ii_override is None
            else np.asarray(cfg.g_ii_override, dtype=float)
        ),
        g_ii_override_r=(
            None
            if cfg.g_ii_override_r is None
            else np.asarray(cfg.g_ii_override_r, dtype=float)
        ),
        analytic_ion_sphere_background=bool(
            _uses_analytic_ion_sphere_background(
                cfg, r_ws=float(r_ws), rmax=float(rmax)
            )
        ),
        exact_ws_boundary_quadrature=bool(
            _uses_decoupled_ion_sphere_grid(cfg, r_ws=float(r_ws), rmax=float(rmax))
        ),
        verbose=bool(cfg.show_scf_progress or cfg.verbose),
        print_every=int(cfg.print_every),
        perf_diag=bool(cfg.perf_diag),
        perf_print_every=1,
        perf_show_stage=bool(cfg.perf_show_stage),
        perf_show_basis=bool(cfg.perf_show_basis),
        store_final_bound_debug=bool(cfg.store_final_bound_debug),
        dn_tol=float(cfg.scf_dn_tol),
        dv_tol=float(cfg.scf_dv_tol),
        v_full_init=v_full_init,
        v_ext_init=v_ext_init,
        v_corr_full=v_corr_full,
        v_corr_ext=v_corr_ext,
    )


def _bound_energy_cut_value(
    *,
    r: np.ndarray,
    v_full: np.ndarray,
    r_ws: float,
    mode: str,
    value: float,
) -> float:
    """
    Compute the bound/continuum split threshold used by bound tables.
    """
    mode_l = str(mode).lower().strip()
    if mode_l == "fixed":
        return float(value)
    if mode_l == "v_ws":
        return float(np.interp(float(r_ws), r, v_full))
    if mode_l == "v_rmax":
        return float(v_full[-1])
    if mode_l == "v_frac":
        r_frac = float(value)
        return float(np.interp(r_frac * float(r[-1]), r, v_full))
    # "zero" and "auto" fallback.
    return 0.0


def _final_ion_gamma_for_reporting(result: dict[str, Any], cfg: FullExternalConfig) -> float:
    """
    Resolve the final n_ion broadening width used by the converged SCF state.

    Parameters
    ----------
    result
        Full-SCF result dictionary returned by the low-level KS solver.
    cfg
        High-level full/external configuration.

    Returns
    -------
    float
        Final gamma in Hartree used for post-SCF `n_ion` diagnostics. When the
        runtime history does not expose the converged gamma, the configured
        value is used as a stable fallback.
    """
    history = result.get("history", [])
    if isinstance(history, list) and history:
        gamma_hist = history[-1].get("ion_gamma", np.nan)
        if np.isfinite(gamma_hist):
            return float(gamma_hist)
    gamma_result = result.get("ion_gamma", np.nan)
    if np.isfinite(gamma_result):
        return float(gamma_result)
    gamma_debug = result.get("debug_ion_gamma", np.nan)
    if np.isfinite(gamma_debug):
        return float(gamma_debug)
    return float(cfg.ion_bound_gamma)


def _build_bound_tables_and_dos(
    *,
    r: np.ndarray,
    v_full: np.ndarray,
    l_list: np.ndarray,
    n_states: int | np.ndarray,
    mu: float,
    temperature_ha: float,
    energy_cut: float,
    gamma: float,
    n_jobs: int,
    zero_tail_bound_meta: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """
    Build bound-level tables and DOS arrays for saving/inspection.

    Tables saved
    ------------
    - bound_energy_ha[l, n]
    - bound_fd[l, n]         = f_FD(E_nl)
    - bound_m[l, n]          = M(E_nl; gamma)
    - bound_fdm[l, n]        = f_FD * M
    - bound_occ_deg_fd[l, n] = 2(2l+1) f_FD
    - bound_occ_deg_fdm[l, n]= 2(2l+1) f_FD M
    """
    r = np.asarray(r, dtype=float)
    v_full = np.asarray(v_full, dtype=float)
    l_arr = np.asarray(l_list, dtype=int)
    if r.size < 2 or l_arr.size == 0:
        return {}

    n_states_arr = None if np.isscalar(n_states) else np.asarray(n_states, dtype=int)
    n_states_max = int(np.max(n_states_arr)) if n_states_arr is not None and n_states_arr.size else int(n_states)
    dxi = float(np.sqrt(r[1]) - np.sqrt(r[0]))
    vals, _ = solve_bound_states_sparse_numerov(
        v_full,
        r,
        dxi,
        l_arr,
        n_states=n_states,
        boundary="dirichlet",
        n_jobs=max(int(n_jobs), 1),
    )

    e_mat = np.asarray(vals, dtype=float)
    # The production SCF can replace a shallow Dirichlet-box orbital by a
    # zero-tail matched pole, including the case where the finite box misses
    # that pole entirely.  Reporting must use the same spectrum as n_bound;
    # blindly rerunning only the finite-box eigensolver here otherwise saves
    # ``inf`` and FD~0 for a state which contributed to the converged density.
    if isinstance(zero_tail_bound_meta, dict):
        for state in zero_tail_bound_meta.get("states", []):
            if not isinstance(state, dict):
                continue
            l_value = int(state.get("l", -1))
            state_index = int(state.get("state_index", -1))
            l_matches = np.flatnonzero(l_arr == l_value)
            if (
                l_matches.size != 1
                or state_index < 0
                or state_index >= e_mat.shape[1]
            ):
                continue
            matched_energy = float(state.get("matched_energy_ha", np.nan))
            if np.isfinite(matched_energy) and matched_energy < 0.0:
                e_mat[int(l_matches[0]), state_index] = matched_energy
    fd_mat = fermi_dirac(e_mat, float(mu), float(temperature_ha))
    m_mat = np.vectorize(_ion_level_weight, otypes=[float])(e_mat, float(gamma))
    fdm_mat = fd_mat * m_mat

    # Degeneracy-weighted occupation tables (useful for population accounting).
    deg_l = 2.0 * (2.0 * l_arr[:, None] + 1.0)
    occ_deg_fd = deg_l * fd_mat
    occ_deg_fdm = deg_l * fdm_mat

    # Simple DOS representation:
    # - bound: Gaussian peaks for E < energy_cut
    # - continuum: ideal free-electron DOS on E>0
    e_finite = e_mat[np.isfinite(e_mat)]
    e_bound = e_finite[e_finite < float(energy_cut)]
    e_min = float(np.min(e_bound) - 0.5) if e_bound.size else -1.0
    e_max = max(float(np.max(e_finite) + 0.5), 0.5) if e_finite.size else 0.5
    e_grid = np.linspace(e_min, e_max, 3000)
    sigma = 0.05
    dos_bound = np.zeros_like(e_grid)
    dos_bound_fd = np.zeros_like(e_grid)
    for li, lval in enumerate(l_arr):
        g_l = 2.0 * (2.0 * float(lval) + 1.0)
        for si in range(e_mat.shape[1]):
            e_nl = float(e_mat[li, si])
            if e_nl >= float(energy_cut):
                continue
            gauss = np.exp(-0.5 * ((e_grid - e_nl) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
            dos_bound += g_l * gauss
            dos_bound_fd += g_l * float(fd_mat[li, si]) * gauss

    e_pos = e_grid[e_grid > 0.0]
    dos_cont = np.zeros_like(e_pos)
    dos_cont_fd = np.zeros_like(e_pos)
    if e_pos.size:
        k = np.sqrt(2.0 * e_pos)
        dos_cont = k / (np.pi**2)
        dos_cont_fd = dos_cont * fermi_dirac(e_pos, float(mu), float(temperature_ha))

    # Store continuum DOS back on e_grid for uniform shape (fill 0 for E<=0).
    dos_cont_full = np.zeros_like(e_grid)
    dos_cont_fd_full = np.zeros_like(e_grid)
    if e_pos.size:
        mask = e_grid > 0.0
        dos_cont_full[mask] = dos_cont
        dos_cont_fd_full[mask] = dos_cont_fd

    # Use [l, n] layout consistently in saved files.
    return {
        "bound_l_list": l_arr.astype(float),
        "bound_n_index": np.arange(1, int(n_states_max) + 1, dtype=float),
        "bound_energy_ha": e_mat,
        "bound_fd": fd_mat,
        "bound_m": m_mat,
        "bound_fdm": fdm_mat,
        "bound_occ_deg_fd": occ_deg_fd,
        "bound_occ_deg_fdm": occ_deg_fdm,
        "dos_energy_ha": e_grid,
        "dos_bound": dos_bound,
        "dos_bound_fd": dos_bound_fd,
        "dos_cont_ideal": dos_cont_full,
        "dos_cont_ideal_fd": dos_cont_fd_full,
    }


def _build_scattering_continuum_dos(
    *,
    energy_ha: np.ndarray,
    phase_shift_rad: np.ndarray,
    r_ws: float,
    mu: float,
    temperature_ha: float,
) -> dict[str, np.ndarray]:
    """
    Build a continuum DOS estimate from scattering phase shifts.

    Notes
    -----
    We use the standard DOS correction proportional to d(delta_l)/dE and scale
    it by the WS volume so it can be compared against the ideal continuum DOS
    already plotted as a density-like quantity. The saved arrays are therefore
    best interpreted as practical diagnostics, not as a rigorously normalized
    many-body DOS observable.
    """
    e = np.asarray(energy_ha, dtype=float)
    delta = np.asarray(phase_shift_rad, dtype=float)
    if e.ndim != 1 or delta.ndim != 2 or e.size < 2 or delta.shape[0] != e.size:
        return {}

    order = np.argsort(e)
    e = e[order]
    delta = delta[order]
    mask = e > 0.0
    if np.count_nonzero(mask) < 2:
        return {}

    e_pos = e[mask]
    delta_pos = qmod.unwrap_scattering_phases(delta[mask], axis=0)
    ddelta_dE = np.gradient(delta_pos, e_pos, axis=0, edge_order=1)
    l_arr = np.arange(delta_pos.shape[1], dtype=float)
    degeneracy = 2.0 * (2.0 * l_arr + 1.0)
    v_ws = (4.0 / 3.0) * np.pi * float(r_ws) ** 3

    k = np.sqrt(2.0 * e_pos)
    dos_ideal = k / (np.pi**2)
    dos_correction = (1.0 / (np.pi * max(v_ws, 1e-14))) * np.sum(degeneracy[None, :] * ddelta_dE, axis=1)
    dos_scattering = dos_ideal + dos_correction
    fd = fermi_dirac(e_pos, float(mu), float(temperature_ha))

    return {
        "dos_cont_energy_ha": e_pos,
        "dos_cont_scattering": dos_scattering,
        "dos_cont_scattering_fd": dos_scattering * fd,
        "dos_cont_fd": fd,
    }


def _print_state_summary(
    *,
    symbol: str,
    z_nuc: int,
    temperature_ev: float,
    rho_g_cc: float,
    r_ws: float,
    rmax_mult: float,
    rmax_eff_mult: float,
    rmax: float,
    n_points: int,
    l_max: int,
    cont_l_max_ceiling: int | None,
    run_mode: str,
    cont_rmax_mult: float | None = None,
    cont_rmax_eff_mult: float | None = None,
    cont_solve_rmax: float | None = None,
    cont_l_cap_strategy: str = "match",
    cont_shards: int | None = None,
    cont_shard_policy: str = "egrid",
) -> None:
    """Print one-line and expanded run setup summary."""
    print(
        f"[full_external] element={symbol} (Z={z_nuc}), Te={temperature_ev:.3f} eV, "
        f"rho={rho_g_cc:.3f} g/cc, run_mode={run_mode}"
    )
    msg = (
        f"[full_external] R_ws={r_ws:.6f} Bohr, rmax_mult={rmax_mult:.3f}, "
        f"r_max={rmax:.6f} Bohr, n_points={n_points}, l_max_global={l_max}"
    )
    if abs(float(rmax_eff_mult) - float(rmax_mult)) > 1.0e-10:
        msg += f", rmax_eff_mult={float(rmax_eff_mult):.3f}"
    if cont_rmax_mult is not None:
        msg += (
            f", cont_rmax_mult={float(cont_rmax_mult):.3f}, "
            f"R_dft_max={float(cont_solve_rmax):.6f} Bohr"
        )
        if cont_rmax_eff_mult is not None and abs(float(cont_rmax_eff_mult) - float(cont_rmax_mult)) > 1.0e-10:
            msg += f", cont_rmax_eff_mult={float(cont_rmax_eff_mult):.3f}"
    if cont_l_max_ceiling is not None:
        msg += f", l_max_cont_ceiling={int(cont_l_max_ceiling)}"
    msg += f", l_cap_strategy={str(cont_l_cap_strategy)}"
    if cont_shards is not None:
        msg += f", cont_shards={int(cont_shards)}, shard_policy={str(cont_shard_policy)}"
    print(msg)


def _print_bound_table(title: str, matrix: np.ndarray, l_values: np.ndarray) -> None:
    """
    Pretty-print a bound-state matrix indexed by [l, n].
    """
    mat = np.asarray(matrix, dtype=float)
    l_arr = np.asarray(l_values, dtype=int)
    if mat.ndim != 2 or mat.shape[0] != l_arr.size:
        return
    n_states = mat.shape[1]
    col_labels = [f"n={i+1}" for i in range(n_states)]
    head = "     | " + " | ".join([f"{c:>10s}" for c in col_labels])
    sep = "-----+" + "+".join(["-" * 12 for _ in col_labels])
    print(f"\n{title}")
    print(head)
    print(sep)
    for i, l_val in enumerate(l_arr):
        row = " | ".join([f"{mat[i, j]:10.4f}" for j in range(n_states)])
        print(f" l={int(l_val):<2d}| {row}")


def _external_fixed_mu_scf(
    *,
    r: np.ndarray,
    mu: float,
    temperature_ha: float,
    n0: float,
    ext_params_base: dict[str, Any],
    mix: float,
    dn_tol: float,
    dv_tol: float,
    max_iter: int,
    adaptive_mix: bool,
    mixing_scheme: str,
    mixing_m: int,
    mixing_w0: float,
    ext_b3_tail_mode: str,
    verbose: bool,
    print_every: int,
    perf_diag: bool = False,
    perf_show_stage: bool = True,
    ph_kappa: float = 0.0,
    ph_kappa_iters: int = 0,
    nuclear_charge: float = 1.0,
    xc_model: str = "dirac",
    gga_core_mode: str = "finite",
    gga_core_zr: float = 0.05,
    g_ii: np.ndarray | None = None,
    v_corr_ext: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    External-only SCF loop with fixed (mu, n0).

    The map follows Eq.(7)-style external source build:
      n_ext -> V_ext -> mixed V_ext -> new n_ext
    """
    from otter.electronic.ks_dft import _select_continuum_model

    ext_params = dict(ext_params_base)
    scheme = str(mixing_scheme).strip().lower()
    if scheme not in ("linear", "eyert"):
        raise ValueError("mixing_scheme must be 'linear' or 'eyert'.")
    b3_mode = str(ext_b3_tail_mode or "off").strip().lower()
    if b3_mode not in ("off", "in_scf"):
        raise ValueError("ext_b3_tail_mode must be 'off' or 'in_scf'.")
    ph_kappa_use = float(ph_kappa)
    ph_iters_use = int(ph_kappa_iters)
    if not np.isfinite(ph_kappa_use) or ph_kappa_use < 0.0:
        raise ValueError("ph_kappa must be finite and non-negative.")
    if ph_iters_use < 0:
        raise ValueError("ph_kappa_iters must be non-negative.")
    if ph_kappa_use <= 0.0 or ph_iters_use <= 0:
        ph_kappa_use = 0.0
        ph_iters_use = 0
    use_b3_tail = (b3_mode == "in_scf")
    if use_b3_tail:
        # The external reference system has no nuclear potential and therefore
        # no discrete bound density: its total electron density is its
        # continuum density.  Even when the full AA branch follows Appendix B
        # by selecting ``b3_tail_target="full"``, the external-only SCF must
        # rebuild that density through the continuum B3 path.  Passing
        # ``"full"`` through to `_rebuild_continuum_on_full_grid` disables the
        # tail there and leaves an artificial step to n0 at `solve_rmax`.
        ext_params["tail_match_target"] = "cont"

    continuum = _select_continuum_model("scattering")
    r_ws = float(ext_params.get("r_ws", 0.0))
    default_ion_sphere_background = (
        g_ii is None and bool(ext_params.get("analytic_ion_sphere_background", False))
    )
    if g_ii is None:
        g_ii_use = IonSphereStepModel(r_ws=r_ws).g_ii(r)
    else:
        g_ii_use = np.asarray(g_ii, dtype=float)
        if g_ii_use.shape != np.asarray(r, dtype=float).shape:
            raise ValueError("external g_ii must match the radial grid.")
    n_bound_zero = np.zeros_like(r, dtype=float)
    use_source_closure_base = bool(ext_params.get("source_closure", False))

    n_ext = np.full_like(r, float(n0), dtype=float)
    n_ext_pre_tail = np.full_like(r, float(n0), dtype=float)
    v_ext = np.zeros_like(r, dtype=float)
    if v_corr_ext is not None and np.asarray(v_corr_ext).shape != np.asarray(r).shape:
        raise ValueError("external v_corr_ext must match the radial grid.")
    v_corr_use = (
        np.zeros_like(r, dtype=float)
        if v_corr_ext is None
        else np.asarray(v_corr_ext, dtype=float)
    )
    current_mix = float(mix)
    prev_err = np.inf
    prev_dv = np.inf

    # Eyert/Anderson history buffers.
    x_prev = None
    f_prev = None
    dx_hist = deque(maxlen=max(int(mixing_m), 1))
    df_hist = deque(maxlen=max(int(mixing_m), 1))
    w0 = max(float(mixing_w0), 1e-12)

    status = {
        "iters": int(max_iter),
        "err": np.nan,
        "converged": False,
    }
    history: list[dict[str, Any]] = []
    tail_meta_ext: dict[str, Any] = {
        "applied": False,
        "reason": "no_external_scf_iteration",
    }

    for it in range(int(max_iter)):
        in_ph_stage = bool(ph_kappa_use > 0.0 and it < ph_iters_use)
        kappa_eff = float(ph_kappa_use) if in_ph_stage else 0.0
        ph_mixer_reset = bool(
            ph_kappa_use > 0.0
            and ph_iters_use > 0
            and it == ph_iters_use
        )
        if ph_mixer_reset:
            # Eyert/Anderson secant vectors from the screened map do not
            # approximate the Jacobian of the final ordinary-Poisson map.
            dx_hist.clear()
            df_hist.clear()
            x_prev = None
            f_prev = None
            # The adaptive linear-mixing controller also compares consecutive
            # map errors.  Screened and unscreened residuals are not
            # commensurate, so do not let the first physical Poisson step
            # inherit a reduced mixing factor or stale trend information.
            current_mix = float(mix)
            prev_err = np.inf
            prev_dv = np.inf
        perf: dict[str, float] = {}
        t_iter = time.perf_counter() if perf_diag else 0.0
        t_stage = t_iter
        n_prev = n_ext.copy()
        v_prev = v_ext.copy()
        tail_meta_ext = {"applied": False, "reason": "tail_disabled"}
        b3_charge_constraint_applied = False
        b3_tail_active_this_iter = bool(use_b3_tail)
        use_source_closure = bool(use_source_closure_base)
        source_closure_meta: dict[str, Any] = {"mode": "enabled" if use_source_closure else "disabled"}

        # 1) Evaluate external continuum density at current V_ext.
        params_iter = dict(ext_params)
        params_iter["v_eff"] = v_ext
        params_iter = _apply_external_energy_floor(params_iter, r, v_ext)
        params_iter["v_eff"] = v_ext
        if b3_tail_active_this_iter:
            cont_solve_end = _continuum_prefix_length(r, params_iter.get("solve_rmax", None))
            r_eval = r[:cont_solve_end]
            v_eval = v_ext[:cont_solve_end]
            params_eval = dict(params_iter)
            params_eval["v_eff"] = v_eval
            params_eval["tail_match"] = False
            n_ext_eval = continuum.density(r_eval, float(mu), float(temperature_ha), params=params_eval)
            try:
                n_ext_raw, n_ext_pre_tail, tail_meta_ext = _rebuild_continuum_on_full_grid(
                    r,
                    n_ext_eval,
                    idx_eval_end=cont_solve_end,
                    params=params_iter,
                    n0=float(n0),
                    mu_id=float(mu),
                    temperature=float(temperature_ha),
                )
                # Candidate physical path: once B3 is enabled, use the rebuilt
                # external density directly as both the observable density and
                # the source entering V_ext unless the user explicitly asks to
                # keep source_closure active with B3.
                n_ext_new = np.asarray(n_ext_raw, dtype=float)
                if bool(params_iter.get("b3_source_charge_constraint", False)):
                    electron_target_ext = _source_electron_charge_target(
                        r,
                        float(n0),
                        g_ii_use,
                        0.0,
                        ion_sphere_radius=(
                            r_ws if default_ion_sphere_background else None
                        ),
                    )
                    n_ext_new, tail_meta_ext = _apply_charge_constrained_b3_tail(
                        r,
                        n_ext_pre_tail,
                        n0=float(n0),
                        mu_id=float(mu),
                        temperature=float(temperature_ha),
                        params=params_iter,
                        electron_charge_target=float(electron_target_ext),
                    )
                    tail_meta_ext = {
                        **dict(tail_meta_ext),
                        "target": "external",
                        "source_charge_target": 0.0,
                    }
                    b3_charge_constraint_applied = True
            except Exception as exc:
                tail_meta_ext = {
                    **dict(tail_meta_ext),
                    "applied": False,
                    "charge_constraint_requested": bool(
                        params_iter.get("b3_source_charge_constraint", False)
                    ),
                    "charge_constraint_applied": False,
                    "charge_constraint_accepted": False,
                    "charge_constraint_failure_reason": str(exc),
                    "failure_reason": str(exc),
                }
                if not bool(params_iter.get("tail_fallback_on_error", True)):
                    raise
                # Fall back for this iteration only.  A transient bad fit must
                # not silently and permanently disable B3 for the remainder
                # of the external SCF.
                b3_tail_active_this_iter = False
        if not b3_tail_active_this_iter:
            n_ext_raw = continuum.density(r, float(mu), float(temperature_ha), params=params_iter)
            n_ext_pre_tail = np.asarray(n_ext_raw, dtype=float)

            # 2) Trusted-region closure and (optional) source-charge closure.
            r_trust = float(
                params_iter.get(
                    "source_r_trust",
                    float(params_iter.get("source_r_trust_frac", 0.75)) * float(r[-1]),
                )
            )
            blend_w = float(
                params_iter.get(
                    "source_blend_width",
                    float(params_iter.get("source_blend_frac", 0.03)) * float(r[-1]),
                )
            )
            n_ext_new = _close_to_n0(r, n_ext_raw, float(n0), r_trust, blend_w)
            if use_source_closure and bool(params_iter.get("source_charge_closure", False)):
                n_ext_new, _ = _enforce_source_charge_closure(
                    r,
                    n_bound_zero,
                    n_ext_new,
                    float(n0),
                    g_ii_use,
                    0.0,
                    float(r_trust),
                    ion_sphere_radius=(r_ws if default_ion_sphere_background else None),
                )
                n_ext_new = np.maximum(n_ext_new, 0.0)
                source_closure_meta = {
                    **dict(source_closure_meta),
                    "charge_closure_applied": True,
                }
        else:
            r_trust = float(
                params_iter.get(
                    "source_r_trust",
                    float(params_iter.get("source_r_trust_frac", 0.75)) * float(r[-1]),
                )
            )
            blend_w = float(
                params_iter.get(
                    "source_blend_width",
                    float(params_iter.get("source_blend_frac", 0.03)) * float(r[-1]),
                )
            )
            if b3_tail_active_this_iter:
                use_source_closure, source_closure_meta = _resolve_ext_source_closure_policy(
                    setting=params_iter.get("ext_source_closure_when_b3", None),
                    r=r,
                    n_ext_candidate=n_ext_new,
                    n0=float(n0),
                    solve_rmax=params_iter.get("solve_rmax", None),
                    tail_r_fit_max=params_iter.get("tail_r_fit_max", None),
                    r_trust=float(r_trust),
                    blend_width=float(blend_w),
                    rel_tol=float(params_iter.get("ext_source_closure_auto_rel_tol", 1.0e-5)),
                )
                use_source_closure = bool(use_source_closure and use_source_closure_base)
            elif not use_source_closure:
                source_closure_meta = {"mode": "disabled"}

            if b3_charge_constraint_applied:
                use_source_closure = False
                source_closure_meta = {
                    "mode": "b3_exact_integral_constraint",
                    "applied": True,
                    "uniform_outer_correction_applied": False,
                    "source_charge_target": 0.0,
                }
            elif use_source_closure:
                n_ext_new = _close_to_n0(r, n_ext_new, float(n0), r_trust, blend_w)
                if bool(params_iter.get("source_charge_closure", False)):
                    n_ext_new, charge_meta = _enforce_source_charge_closure(
                        r,
                        n_bound_zero,
                        n_ext_new,
                        float(n0),
                        g_ii_use,
                        0.0,
                        float(r_trust),
                        ion_sphere_radius=(r_ws if default_ion_sphere_background else None),
                    )
                    source_closure_meta = {
                        **dict(source_closure_meta),
                        "charge_closure_applied": bool(charge_meta.get("applied", False)),
                        "charge_closure_delta_n": float(charge_meta.get("delta_n", 0.0)),
                    }
                    n_ext_new = np.maximum(n_ext_new, 0.0)
            else:
                source_closure_meta = {**dict(source_closure_meta), "applied": False}

        if perf_diag:
            perf["continuum"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()

        # 3) Rebuild V_ext and remove finite-box gauge.
        v_new = effective_potential_external(
            r,
            n_ext_new,
            float(n0),
            g_ii_use,
            xc_model=xc_model,
            kappa=kappa_eff,
            ion_sphere_radius=(r_ws if default_ion_sphere_background else None),
            nuclear_charge=float(nuclear_charge),
            gga_core_mode=gga_core_mode,
            gga_core_zr=gga_core_zr,
        )
        v_new = v_new + v_corr_use
        v_new = v_new - _tail_shift_value(r, v_new, frac=0.05)
        if perf_diag:
            perf["closure"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()

        # 4) Mix V_ext (linear or Eyert).
        if scheme == "linear":
            v_ext = current_mix * v_new + (1.0 - current_mix) * v_ext
        else:
            x_in = v_ext
            x_out = v_new
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
                        a_mat[i, j] = trapz_integral(
                            df_hist[i] * df_hist[j], r
                        )
                        if i == j:
                            a_mat[i, j] += w0**2
                    b_vec[i] = trapz_integral(df_hist[i] * f_now, r)
                try:
                    w_vec = np.linalg.solve(a_mat, b_vec)
                except np.linalg.LinAlgError:
                    w_vec = None
                if w_vec is not None:
                    corr = np.zeros_like(x_in)
                    for i in range(hist_len):
                        corr += w_vec[i] * (dx_hist[i] + current_mix * df_hist[i])
                    x_next = x_in + current_mix * f_now - corr
                else:
                    x_next = x_in + current_mix * f_now
            else:
                    x_next = x_in + current_mix * f_now

            v_ext = x_next
            x_prev = x_in.copy()
            f_prev = f_now.copy()

        if perf_diag:
            perf["potential"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()

        v_ext = v_ext - _tail_shift_value(r, v_ext, frac=0.05)
        n_ext = n_ext_new
        if perf_diag:
            perf["mix"] = time.perf_counter() - t_stage
            t_stage = time.perf_counter()

        err = float(np.max(np.abs(v_new - v_ext)) / max(np.max(np.abs(v_ext)), 1e-12))
        # Only V_ext is mixed in the external fixed-mu loop. The density is
        # rebuilt directly from the current potential, so dn_rel can plateau to
        # machine zero before the mixed potential has fully settled.
        dn_rel = trapz_integral(np.abs(n_ext - n_prev), r) / max(
            trapz_integral(np.abs(n_prev), r), 1e-12
        )
        dv_rel = trapz_integral(np.abs(v_ext - v_prev), r) / max(
            trapz_integral(np.abs(v_prev), r), 1e-12
        )
        if perf_diag:
            perf["metrics"] = time.perf_counter() - t_stage
            perf["total"] = time.perf_counter() - t_iter

        if verbose and (it % max(int(print_every), 1) == 0):
            print(
                f"  [ext] it={it:2d}  dn_rel={dn_rel:.3e}  dv_rel={dv_rel:.3e}  "
                f"err={err:.3e}  mix={current_mix:.3f}  scheme={scheme}  "
                f"kappa={kappa_eff:.3e} Bohr^-1  "
                f"cont_e_max={float(ext_params.get('e_max', np.nan)):.6f} Ha"
            )
            tail_model_req = str(tail_meta_ext.get("model_requested", "")).strip().lower()
            if tail_model_req == "auto":
                tail_model_sel = str(tail_meta_ext.get("model_selected", "n/a"))
                fit_rel_improve = tail_meta_ext.get("fit_rel_improve_full", np.nan)
                fit_signal_max = tail_meta_ext.get("fit_signal_max", np.nan)
                fit_signal_threshold = tail_meta_ext.get("fit_signal_threshold", np.nan)
                print(
                    "      tail_auto: "
                    f"selected={tail_model_sel}, "
                    f"rel_improve={float(fit_rel_improve):.3e}, "
                    f"signal={float(fit_signal_max):.3e}, "
                    f"threshold={float(fit_signal_threshold):.3e}"
                )
            if perf_diag and perf_show_stage:
                perf_line = ", ".join(
                    f"{key}={perf[key]:.3f}"
                    for key in ("continuum", "closure", "potential", "mix", "metrics", "total")
                    if key in perf and np.isfinite(float(perf[key]))
                )
                if perf_line:
                    print(f"      perf[s]: {perf_line}")

        history.append({
            "iter": int(it),
            "err": float(err),
            "dn_rel": float(dn_rel),
            "dv_rel": float(dv_rel),
            "cont_e_max": float(ext_params.get("e_max", np.nan)),
            "mix": float(current_mix),
            "scheme": str(scheme),
            "ph_kappa": float(kappa_eff),
            "in_ph_stage": bool(in_ph_stage),
            "ph_mixer_reset": bool(ph_mixer_reset),
            "n_ext_tail_model_requested": str(tail_meta_ext.get("model_requested", "")),
            "n_ext_tail_model_selected": str(tail_meta_ext.get("model_selected", "")),
            "n_ext_tail_charge_constraint_requested": bool(
                tail_meta_ext.get("charge_constraint_requested", False)
            ),
            "n_ext_tail_charge_constraint_applied": bool(
                tail_meta_ext.get("charge_constraint_applied", False)
            ),
            "n_ext_tail_charge_constraint_residual": float(
                tail_meta_ext.get("charge_constraint_residual", np.nan)
            ),
            "n_ext_tail_charge_constraint_fit_rms_ratio": float(
                tail_meta_ext.get("charge_constraint_fit_rms_ratio", np.nan)
            ),
            "n_ext_tail_charge_constraint_coeff_delta_rel": float(
                tail_meta_ext.get("charge_constraint_coeff_delta_rel", np.nan)
            ),
            "n_ext_tail_charge_constraint_profile_delta_rel": float(
                tail_meta_ext.get("charge_constraint_profile_delta_rel", np.nan)
            ),
            "n_ext_tail_charge_constraint_min_density": float(
                tail_meta_ext.get("charge_constraint_tail_min", np.nan)
            ),
            "n_ext_tail_charge_constraint_failure_reason": str(
                tail_meta_ext.get("charge_constraint_failure_reason", "")
            ),
            "ext_source_closure_mode": str(source_closure_meta.get("mode", "")),
            "ext_source_closure_applied": bool(source_closure_meta.get("applied", use_source_closure_base)),
            "ext_source_closure_rel_dev_max": float(source_closure_meta.get("rel_dev_max", np.nan)),
            "ext_source_closure_pretrust_rel_dev_max": float(
                source_closure_meta.get("pretrust_rel_dev_max", np.nan)
            ),
            "ext_source_closure_pretrust_rel_span": float(
                source_closure_meta.get("pretrust_rel_span", np.nan)
            ),
            "ext_source_closure_pretrust_rel_trend": float(
                source_closure_meta.get("pretrust_rel_trend", np.nan)
            ),
            "ext_source_closure_rel_tol": float(source_closure_meta.get("rel_tol", np.nan)),
            "perf": perf if perf_diag else None,
        })

        # Poisson-Helmholtz is only a preconditioner. Even if the screened map
        # appears converged, require at least one ordinary Poisson update
        # before accepting the external fixed point.
        b3_constraint_required = bool(
            use_b3_tail
            and params_iter.get("b3_source_charge_constraint", False)
        )
        b3_constraint_ok = bool(
            (not b3_constraint_required) or b3_charge_constraint_applied
        )
        if (
            (not in_ph_stage)
            and b3_constraint_ok
            and dn_rel < float(dn_tol)
            and dv_rel < float(dv_tol)
        ):
            status = {
                "iters": it + 1,
                "err": err,
                "converged": True,
                "history": history,
                "ph_kappa": float(ph_kappa_use),
                "ph_kappa_iters": int(ph_iters_use),
                "final_ph_kappa": float(kappa_eff),
                "tail_meta": dict(tail_meta_ext),
                "n_ext_pre_tail": np.asarray(n_ext_pre_tail, dtype=float),
            }
            return n_ext, v_ext, status

        if adaptive_mix:
            if (err > prev_err * 1.05) or (dv_rel > prev_dv * 1.25):
                current_mix = max(0.01, current_mix * 0.5)
            elif (err < prev_err * 0.75) and (dv_rel < prev_dv * 0.8):
                current_mix = min(float(mix), current_mix * 1.15)
        prev_err = err
        prev_dv = dv_rel

    status = {
        "iters": int(max_iter),
        "err": float(prev_err),
        "converged": False,
        "history": history,
        "ph_kappa": float(ph_kappa_use),
        "ph_kappa_iters": int(ph_iters_use),
        "final_ph_kappa": (
            float(history[-1]["ph_kappa"]) if history else 0.0
        ),
        "tail_meta": dict(tail_meta_ext),
        "n_ext_pre_tail": np.asarray(n_ext_pre_tail, dtype=float),
    }
    return n_ext, v_ext, status


def solve_full_then_external(cfg: FullExternalConfig) -> dict[str, Any]:
    """
    Run validated two-stage full SCF, then optional fixed-mu external SCF.

    Returns
    -------
    dict
        Unified result dictionary with:
        - solver outputs (densities, potentials, mu, history),
        - metadata ("meta"),
        - optional save paths ("saved_paths") when cfg.save_data=True.
    """
    if str(cfg.electronic_model).strip().lower() == "tf":
        # Import lazily so the mature orbital path keeps its historical import
        # graph and startup behavior.  The TF result implements the same
        # full/external pseudoatom payload consumed below by QOZ/HNC.
        from otter.electronic.thomas_fermi import (
            ThomasFermiConfig,
            solve_thomas_fermi_full_then_external,
        )

        return solve_thomas_fermi_full_then_external(
            ThomasFermiConfig(
                element=cfg.element,
                temperature_ev=float(cfg.temperature_ev),
                rho_g_cc=float(cfg.rho_g_cc),
                xc_model=str(cfg.xc_model),
                gga_core_mode=str(cfg.gga_core_mode),
                gga_core_zr=float(cfg.gga_core_zr),
                r_ws_override_bohr=cfg.r_ws_override_bohr,
                n_i_override_bohr3=cfg.n_i_override_bohr3,
                run_mode=str(cfg.run_mode),
                rmax_mult=float(cfg.rmax_mult),
                n_points=int(cfg.n_points),
                mix=float(cfg.tf_scf_mix),
                mixing_history=int(cfg.tf_scf_mixing_m),
                mixing_regularization=float(cfg.tf_scf_mixing_w0),
                max_iter=int(cfg.tf_scf_max_iter),
                tol=float(cfg.tf_scf_tol),
                polish_tol=float(cfg.tf_polish_tol),
                polish_max_iter=int(cfg.tf_polish_max_iter),
                mu_tol=float(cfg.tf_mu_tol),
                mu_bounds=(
                    float(cfg.mu_bounds[0]),
                    float(cfg.mu_bounds[1]),
                ),
                mu_guess_zbar=float(cfg.mu_guess_zbar),
                ion_cut_c=float(cfg.ion_cut_c),
                quadrature_order=int(cfg.tf_quadrature_order),
                show_progress=bool(cfg.show_scf_progress),
                verbose=bool(cfg.verbose),
                v_full_init=cfg.v_full_init,
                v_full_init_r=cfg.v_full_init_r,
                v_ext_init=cfg.v_ext_init,
                v_ext_init_r=cfg.v_ext_init_r,
                full_fixed_mu_ha=cfg.full_fixed_mu_ha,
                g_ii_override=cfg.g_ii_override,
                g_ii_override_r=cfg.g_ii_override_r,
                v_corr_full=cfg.v_corr_full,
                v_corr_full_r=cfg.v_corr_full_r,
                v_corr_ext=cfg.v_corr_ext,
                v_corr_ext_r=cfg.v_corr_ext_r,
                save_data=bool(cfg.save_data),
                save_output_dir=cfg.save_output_dir,
                save_suffix=str(cfg.save_suffix),
            )
        )

    elem = element_info(cfg.element)
    z_nuc = int(elem.z)
    symbol = str(elem.symbol)
    atomic_weight = float(elem.atomic_mass)

    temperature_ha = float(cfg.temperature_ev) * EV_TO_HA
    n_i, r_ws = _resolve_species_geometry(cfg, atomic_weight=atomic_weight)
    geometry = _resolve_outer_geometry(cfg, r_ws=float(r_ws))
    rmax = float(geometry["rmax"])
    r_target = _target_radial_grid(rmax=rmax, n_points=int(cfg.n_points))
    v_full_init, v_full_init_meta = _resample_initial_potential(
        values=cfg.v_full_init,
        source_r=cfg.v_full_init_r,
        target_r=r_target,
        name="v_full_init",
    )
    v_ext_init, v_ext_init_meta = _resample_initial_potential(
        values=cfg.v_ext_init,
        source_r=cfg.v_ext_init_r,
        target_r=r_target,
        name="v_ext_init",
    )
    v_corr_full, v_corr_full_meta = _resample_initial_potential(
        values=cfg.v_corr_full,
        source_r=cfg.v_corr_full_r,
        target_r=r_target,
        name="v_corr_full",
    )
    v_corr_ext, v_corr_ext_meta = _resample_initial_potential(
        values=cfg.v_corr_ext,
        source_r=cfg.v_corr_ext_r,
        target_r=r_target,
        name="v_corr_ext",
    )
    mu_guess = mu_guess_from_density(n_i, zbar=float(cfg.mu_guess_zbar))

    k_max = np.sqrt(2.0 * float(cfg.cont_e_max))
    l_max = min(int(np.ceil(k_max * rmax + 2.0)), 150)
    cont_solve_rmax = float(geometry["solve_rmax"]) if geometry["solve_rmax"] is not None else float(rmax)
    cont_l_max_ceiling = min(int(np.ceil(k_max * cont_solve_rmax + 2.0)), 150)
    if cfg.show_scf_progress or cfg.verbose:
        _print_state_summary(
            symbol=symbol,
            z_nuc=z_nuc,
            temperature_ev=float(cfg.temperature_ev),
            rho_g_cc=float(cfg.rho_g_cc),
            r_ws=float(r_ws),
            rmax_mult=float(cfg.rmax_mult),
            rmax_eff_mult=float(geometry["rmax_eff_mult"]),
            rmax=float(rmax),
            n_points=int(cfg.n_points),
            l_max=int(l_max),
            cont_l_max_ceiling=int(cont_l_max_ceiling),
            run_mode=str(cfg.run_mode),
            cont_rmax_mult=cfg.cont_rmax_mult,
            cont_rmax_eff_mult=(
                float(geometry["solve_rmax_eff_mult"]) if geometry["solve_rmax_eff_mult"] is not None else None
            ),
            cont_solve_rmax=float(cont_solve_rmax),
            cont_l_cap_strategy=str(cfg.cont_l_cap_strategy),
            cont_shards=cfg.cont_shards,
            cont_shard_policy=str(cfg.cont_shard_policy),
        )

    cont_stage1 = _build_continuum_params(
        cfg,
        l_max=l_max,
        r_ws=r_ws,
        rmax=rmax,
        adaptive_mode=str(cfg.cont_adaptive_mode_stage1),
        b3_stage_mode=str(cfg.b3_tail_stage1_mode),
        e_max_mode="fixed",
    )
    cont_stage2 = _build_continuum_params(
        cfg,
        l_max=l_max,
        r_ws=r_ws,
        rmax=rmax,
        adaptive_mode=str(cfg.cont_adaptive_mode_stage2),
        b3_stage_mode=str(cfg.b3_tail_stage2_mode),
        e_max_mode=str(cfg.cont_stage2_e_max_mode),
    )

    def _run_stage1_once(
        *,
        cont_params_stage1: dict[str, Any],
    ) -> dict[str, Any]:
        if cfg.show_scf_progress or cfg.verbose:
            e_max_mode = str(cont_params_stage1.get("e_max_mode", "fixed"))
            msg = f"[full_external] stage1: e_max_mode={e_max_mode}"
            if e_max_mode == "fixed":
                try:
                    msg += f", e_max={float(cont_params_stage1['e_max']):.6f} Ha"
                except Exception:
                    pass
            print(msg)
        cfg1_local = _build_ks_config(
            cfg,
            z_nuc=z_nuc,
            temperature_ha=temperature_ha,
            n_i=n_i,
            r_ws=r_ws,
            rmax=rmax,
            mu_guess=mu_guess,
            mu_bounds=cfg.mu_bounds,
            max_iter=cfg.stage1_max_iter,
            cont_params=cont_params_stage1,
            compute_external=False,
            v_full_init=None if v_full_init is None else v_full_init.copy(),
            v_ext_init=None if v_ext_init is None else v_ext_init.copy(),
            v_corr_full=None if v_corr_full is None else v_corr_full.copy(),
            v_corr_ext=None if v_corr_ext is None else v_corr_ext.copy(),
        )
        stage1_local = solve_ks_dft_is(cfg1_local)
        return _apply_b3_post_to_full_result(
            stage1_local,
            cfg,
            r_ws=float(r_ws),
            rmax=float(rmax),
            stage_mode=str(cfg.b3_tail_stage1_mode),
        )

    use_stage2_continuation_init = bool(cfg.continuation_stage2_from_init) and v_full_init is not None
    skip_stage1 = bool(use_stage2_continuation_init and int(cfg.stage1_max_iter) <= 0)
    if skip_stage1:
        if cfg.show_scf_progress or cfg.verbose:
            print("[full_external] stage1: skipped; using continuation V_eff directly for stage2")
        stage1 = {
            "history": [],
            "mu": (
                float(cfg.continuation_mu_init)
                if cfg.continuation_mu_init is not None
                else float(mu_guess)
            ),
            "converged": True,
            "v_full": np.asarray(v_full_init, dtype=float).copy(),
        }
    else:
        stage1 = _run_stage1_once(cont_params_stage1=cont_stage1)
    mu_stage1 = float(stage1["mu"])
    mu_stage2_guess = (
        float(cfg.continuation_mu_init)
        if use_stage2_continuation_init and cfg.continuation_mu_init is not None
        else mu_stage1
    )

    if cfg.stage2_local_mu_bounds:
        lo = max(float(cfg.mu_bounds[0]), mu_stage2_guess - float(cfg.stage2_mu_half_width))
        hi = min(float(cfg.mu_bounds[1]), mu_stage2_guess + float(cfg.stage2_mu_half_width))
        mu_bounds2 = (lo, hi) if lo < hi else cfg.mu_bounds
    else:
        mu_bounds2 = cfg.mu_bounds

    def _stage2_mixer_values() -> tuple[str, int, float]:
        if use_stage2_continuation_init:
            scheme = (
                str(cfg.continuation_scf_mixing_scheme)
                if cfg.continuation_scf_mixing_scheme is not None
                else str(cfg.scf_mixing_scheme)
            )
            hist_m = (
                int(cfg.continuation_scf_mixing_m)
                if cfg.continuation_scf_mixing_m is not None
                else int(cfg.scf_mixing_m)
            )
            mix_val = (
                float(cfg.continuation_scf_mix)
                if cfg.continuation_scf_mix is not None
                else float(cfg.scf_mix)
            )
            return scheme, hist_m, mix_val
        return str(cfg.scf_mixing_scheme), int(cfg.scf_mixing_m), float(cfg.scf_mix)

    def _run_stage2_once(
        *,
        mu_bounds_stage2: tuple[float, float],
        cont_params_stage2: dict[str, Any],
    ) -> dict[str, Any]:
        stage2_mixer_scheme, stage2_mixer_m, stage2_mixer_mix = _stage2_mixer_values()
        if cfg.show_scf_progress or cfg.verbose:
            e_max_mode = str(cont_params_stage2.get("e_max_mode", "fixed"))
            msg = (
                "[full_external] stage2: "
                f"mu_bounds=({float(mu_bounds_stage2[0]):.6f}, {float(mu_bounds_stage2[1]):.6f}) Ha, "
                f"e_max_mode={e_max_mode}"
            )
            if e_max_mode == "fixed":
                try:
                    msg += f", e_max={float(cont_params_stage2['e_max']):.6f} Ha"
                except Exception:
                    pass
            if use_stage2_continuation_init:
                msg += (
                    f", continuation_mixer={stage2_mixer_scheme}, "
                    f"m={int(stage2_mixer_m)}, mix={float(stage2_mixer_mix):.3f}"
                )
            print(msg)
        cfg2_local = _build_ks_config(
            cfg,
            z_nuc=z_nuc,
            temperature_ha=temperature_ha,
            n_i=n_i,
            r_ws=r_ws,
            rmax=rmax,
            mu_guess=mu_stage2_guess,
            mu_bounds=mu_bounds_stage2,
            max_iter=cfg.stage2_max_iter,
            cont_params=cont_params_stage2,
            compute_external=False,
            v_full_init=(
                np.asarray(v_full_init, dtype=float)
                if use_stage2_continuation_init
                else np.asarray(stage1["v_full"], dtype=float)
            ),
            v_corr_full=None if v_corr_full is None else np.asarray(v_corr_full, dtype=float),
            v_corr_ext=None if v_corr_ext is None else np.asarray(v_corr_ext, dtype=float),
            scf_mix_override=stage2_mixer_mix,
            scf_mixing_scheme_override=stage2_mixer_scheme,
            scf_mixing_m_override=stage2_mixer_m,
        )
        full_local = solve_ks_dft_is(cfg2_local)
        full_local = _apply_b3_post_to_full_result(
            full_local,
            cfg,
            r_ws=float(r_ws),
            rmax=float(rmax),
            stage_mode=str(cfg.b3_tail_stage2_mode),
        )
        return full_local

    full = _run_stage2_once(
        mu_bounds_stage2=mu_bounds2,
        cont_params_stage2=cont_stage2,
    )

    result = dict(full)
    result["stage1_history"] = list(stage1.get("history", []))
    result["stage1_mu"] = mu_stage1
    result["stage1_skipped"] = bool(skip_stage1)
    result["stage1_converged"] = bool(stage1.get("converged", False))
    result["stage1_iters"] = int(len(stage1.get("history", [])))
    result["stage2_converged"] = bool(full.get("converged", False))
    result["stage2_iters"] = int(len(full.get("history", [])))
    result["perf_summary_stage1"] = _summarize_history_perf(stage1.get("history", []))
    result["perf_summary_stage2"] = _summarize_history_perf(full.get("history", []))
    result["perf_summary_full"] = dict(result["perf_summary_stage2"])
    result["workflow"] = "full_then_ext"

    if cfg.perf_diag and (cfg.show_scf_progress or cfg.verbose):
        for stage_label, summary in (
            ("stage1", result["perf_summary_stage1"]),
            ("stage2", result["perf_summary_stage2"]),
        ):
            mean_line = _format_perf_summary_line(summary, field="timing_s", label="mean")
            max_line = _format_perf_summary_line(summary, field="timing_s", label="max")
            basis_line = _format_perf_summary_line(summary, field="basis_meta", label="mean")
            if mean_line:
                print(f"  [perf-summary:{stage_label}] mean[s]: {mean_line}")
            if max_line:
                print(f"  [perf-summary:{stage_label}] max[s]: {max_line}")
            if basis_line:
                print(f"  [perf-summary:{stage_label}] mean[basis]: {basis_line}")

    run_mode = str(cfg.run_mode).strip().lower()
    if run_mode not in ("full", "full+ext", "full_ext"):
        raise ValueError("run_mode must be 'full' or 'full+ext'.")
    do_external = (run_mode != "full") and bool(cfg.ext_scf_enabled)

    if do_external:
        # External-only SCF with fixed (mu,n0) from converged full solve.
        r_full = np.asarray(full["r"], dtype=float)
        mu_fix = float(full["mu"])
        n0_fix = float(full["n0"])
        stage2_hist = list(full.get("history", []))
        stage2_e_max_final = float(cfg.cont_e_max)
        if stage2_hist:
            e_last = stage2_hist[-1].get("cont_e_max", stage2_e_max_final)
            try:
                e_last = float(e_last)
            except Exception:
                e_last = float(cfg.cont_e_max)
            if np.isfinite(e_last):
                stage2_e_max_final = e_last
        # External branch reuses the stage-2 continuum integration settings,
        # but its B3 activation is still controlled independently from the
        # full solver so users can fall back to a pure-A3 external branch
        # for diagnostics if needed.
        cont_ext = _build_continuum_params(
            cfg,
            l_max=l_max,
            r_ws=r_ws,
            rmax=rmax,
            adaptive_mode=str(cfg.cont_adaptive_mode_stage2),
            b3_stage_mode=str(cfg.ext_b3_tail_mode),
            e_max_mode="fixed",
            for_external=True,
        )
        cont_ext["e_max"] = float(stage2_e_max_final)
        cont_ext["l_max"] = _resolve_iteration_continuum_l_max(
            cont_ext,
            e_max=float(stage2_e_max_final),
            r_eval_max=float(cont_ext.get("solve_rmax", r_full[-1])),
        )
        _, ext_params = _split_continuum_params_for_full_ext(cont_ext)
        ext_params["r_ws"] = float(full["r_ws"])
        n_ext_fix, v_ext_fix, ext_status = _external_fixed_mu_scf(
            r=r_full,
            g_ii=np.asarray(full["g_ii"], dtype=float),
            mu=mu_fix,
            temperature_ha=temperature_ha,
            n0=n0_fix,
            ext_params_base=ext_params,
            mix=float(cfg.ext_scf_mix),
            dn_tol=float(cfg.ext_scf_dn_tol),
            dv_tol=float(cfg.ext_scf_dv_tol),
            max_iter=int(cfg.ext_scf_max_iter),
            adaptive_mix=bool(cfg.ext_scf_adaptive_mix),
            mixing_scheme=str(cfg.ext_mixing_scheme),
            mixing_m=int(cfg.ext_mixing_m),
            mixing_w0=float(cfg.ext_mixing_w0),
            ext_b3_tail_mode=str(cfg.ext_b3_tail_mode),
            verbose=bool(cfg.show_scf_progress or cfg.verbose),
            print_every=int(cfg.print_every),
            perf_diag=bool(cfg.perf_diag),
            perf_show_stage=bool(cfg.perf_show_stage),
            ph_kappa=float(cfg.ph_kappa),
            ph_kappa_iters=int(cfg.ph_kappa_iters),
            nuclear_charge=float(z_nuc),
            xc_model=str(cfg.xc_model),
            gga_core_mode=str(cfg.gga_core_mode),
            gga_core_zr=float(cfg.gga_core_zr),
            v_corr_ext=None if v_corr_ext is None else np.asarray(v_corr_ext, dtype=float),
        )
        result["n_ext"] = np.asarray(n_ext_fix, dtype=float)
        result["n_ext_pre_tail"] = np.asarray(
            ext_status.get("n_ext_pre_tail", n_ext_fix),
            dtype=float,
        )
        result["v_ext"] = np.asarray(v_ext_fix, dtype=float)
        result["n_pa"] = np.asarray(result["n_full"], dtype=float) - result["n_ext"]
        result["n_scr"] = result["n_pa"] - np.asarray(result["n_ion"], dtype=float)
        result["ext_status"] = dict(ext_status)
        result["ext_history"] = list(ext_status.get("history", []))
        result["n_ext_tail_meta"] = dict(
            ext_status.get("tail_meta", {})
        )
        result["b3_charge_constraint_ext_applied"] = bool(
            result["n_ext_tail_meta"].get(
                "charge_constraint_applied", False
            )
        )
        result["perf_summary_ext"] = _summarize_history_perf(result["ext_history"])
        result["stage2_cont_e_max_final"] = float(stage2_e_max_final)
    else:
        result["ext_status"] = {"enabled": False}
        result["ext_history"] = []
        result["perf_summary_ext"] = {"n_iter": 0, "n_perf": 0}
        result["stage2_cont_e_max_final"] = float(cfg.cont_e_max)

    if cfg.perf_diag and (cfg.show_scf_progress or cfg.verbose) and do_external:
        summary = result.get("perf_summary_ext", {})
        mean_line = _format_perf_summary_line(summary, field="timing_s", label="mean")
        max_line = _format_perf_summary_line(summary, field="timing_s", label="max")
        if mean_line:
            print(f"  [perf-summary:ext] mean[s]: {mean_line}")
        if max_line:
            print(f"  [perf-summary:ext] max[s]: {max_line}")

    # Potential decomposition arrays requested by downstream storage.
    r = np.asarray(result["r"], dtype=float)
    n_full = np.asarray(result["n_full"], dtype=float)
    potential_source_key = (
        "n_full_source" if "n_full_source" in result else "n_full"
    )
    n_potential_source = np.asarray(result[potential_source_key], dtype=float)
    n0 = float(result["n0"])
    g_ii = np.asarray(result["g_ii"], dtype=float)
    n0_arr = np.full_like(r, n0)
    use_analytic_background = _uses_analytic_ion_sphere_background(
        cfg,
        r_ws=float(result["r_ws"]),
        rmax=float(r[-1]),
    )
    if use_analytic_background:
        v_h = spherical_hartree_potential(
            r, n_potential_source
        ) + _ion_sphere_background_hartree(
            r,
            n0=n0,
            r_ws=float(result["r_ws"]),
        )
    else:
        rho_source = n_potential_source - n0_arr * g_ii
        v_h = spherical_hartree_potential(r, rho_source)
    gga_core_radius = resolve_gga_core_radius(
        cfg.xc_model,
        nuclear_charge=float(z_nuc),
        mode=cfg.gga_core_mode,
        core_zr=cfg.gga_core_zr,
        r=r,
    )
    v_xc = xc_potential(
        n_potential_source,
        model=cfg.xc_model,
        r=r,
        gga_core_radius_bohr=gga_core_radius,
    ) - xc_potential(
        n0_arr,
        model=cfg.xc_model,
        r=r,
        gga_core_radius_bohr=gga_core_radius,
    )
    result["v_H"] = v_h
    result["v_xc"] = v_xc
    core_diagnostics = radial_core_diagnostics(
        r,
        n_potential_source,
        v_xc,
        nuclear_charge=float(z_nuc),
        core_radius_bohr=gga_core_radius,
    )
    result["gga_core_mode"] = str(cfg.gga_core_mode)
    result["gga_core_zr"] = float(cfg.gga_core_zr)
    result["gga_core_radius_bohr"] = (
        float(gga_core_radius) if gga_core_radius is not None else np.nan
    )
    result["gga_core_points"] = int(core_diagnostics["core_points"])
    result["gga_radial_operator"] = "sqrt-grid-discrete-adjoint-v1"
    result["density_cusp_rel_error"] = float(
        core_diagnostics["density_cusp_rel_error"]
    )
    result["v_xc_core_turn_count"] = int(
        core_diagnostics["potential_turn_count"]
    )
    result["v_xc_core_max_abs_ha"] = float(
        core_diagnostics["max_abs_potential_ha"]
    )
    # Keep Libxc software/version/functional references separate from Otter's
    # finite-core provenance.  See https://libxc.gitlab.io/#citing-libxc.
    result["xc_provenance"] = xc_provenance(cfg.xc_model)
    result["v_scf"] = np.asarray(result["v_full"], dtype=float)
    result["v_decomposition_source_density"] = str(potential_source_key)
    result["v_decomposition_source_provenance"] = str(
        result.get(
            "n_full_source_provenance",
            "reported_density" if potential_source_key == "n_full" else "unknown",
        )
    )
    result["n0_ideal"] = float(
        ideal_unbound_density(float(result["mu"]), float(temperature_ha))
    )

    energy_cut = _bound_energy_cut_value(
        r=r,
        v_full=np.asarray(result["v_full"], dtype=float),
        r_ws=float(result["r_ws"]),
        mode=str(cfg.bound_energy_cut_mode),
        value=float(cfg.bound_energy_cut),
    )
    final_ion_gamma = _final_ion_gamma_for_reporting(result, cfg)
    r_bound_diag = np.asarray(result.get("r_bound", r), dtype=float)
    v_full_diag = np.asarray(result["v_full"], dtype=float)
    if r_bound_diag.shape != r.shape or not np.allclose(r_bound_diag, r):
        v_full_diag = np.interp(r_bound_diag, r, v_full_diag, right=0.0)
    bound_diag = _build_bound_tables_and_dos(
        r=r_bound_diag,
        v_full=v_full_diag,
        l_list=cfg.resolved_l_list(),
        n_states=cfg.resolved_n_states_by_l(),
        mu=float(result["mu"]),
        temperature_ha=float(temperature_ha),
        energy_cut=float(energy_cut),
        gamma=float(final_ion_gamma),
        n_jobs=int(max(cfg.bound_table_n_jobs, 1)),
        zero_tail_bound_meta=result.get("zero_tail_bound_meta", None),
    )
    result.update(bound_diag)
    # Source closure may deliberately use a tail-corrected density only for
    # assembling V_eff while preserving the physical/output n_full profile.
    # The final-refresh branch stores a fixed-point source candidate associated
    # with the returned potential, rather than claiming that the last mixed
    # potential was generated from that density exactly.
    veff_source_density_key = (
        "n_full_source" if "n_full_source" in result else "n_full"
    )
    veff_asymptotic_diag = _compute_veff_asymptotic_diagnostics(
        r=r,
        v_eff=np.asarray(result["v_full"], dtype=float),
        n_full=np.asarray(result[veff_source_density_key], dtype=float),
        n0=n0,
        g_ii=g_ii,
        z_nuc=float(z_nuc),
        r_ws=float(result["r_ws"]),
        shallowest_bound_energy_ha=float(
            result.get("shallowest_bound_energy_ha", np.nan)
        ),
        analytic_ion_sphere_background=bool(use_analytic_background),
        absolute_tail_tolerance_ha=float(cfg.veff_asymptotic_abs_tol_ha),
    )
    veff_asymptotic_diag["veff_asymptotic_source_density"] = (
        veff_source_density_key
    )
    veff_asymptotic_diag["veff_asymptotic_source_provenance"] = str(
        result.get(
            "n_full_source_provenance",
            "reported_density"
            if veff_source_density_key == "n_full"
            else "unknown",
        )
    )
    result.update(veff_asymptotic_diag)
    result.update(
        _build_scattering_continuum_dos(
            energy_ha=np.asarray(result.get("cont_phase_energy_ha", []), dtype=float),
            phase_shift_rad=np.asarray(result.get("cont_phase_shift_rad", []), dtype=float),
            r_ws=float(result["r_ws"]),
            mu=float(result["mu"]),
            temperature_ha=float(temperature_ha),
        )
    )
    if cfg.show_scf_progress or cfg.verbose:
        l_vals = np.asarray(result.get("bound_l_list", []), dtype=int)
        _print_bound_table(
            "=== Bound energies (Ha) ===",
            np.asarray(result.get("bound_energy_ha", np.empty((0, 0))), dtype=float),
            l_vals,
        )
        _print_bound_table(
            "=== Bound FD occupancy f(E_nl) ===",
            np.asarray(result.get("bound_fd", np.empty((0, 0))), dtype=float),
            l_vals,
        )

    # Ionization-state and WS-integral diagnostics. These are useful both
    # for direct inspection and for lightweight post-processing of saved
    # NPZ files, so we compute them once here and store them explicitly.
    n_ion = np.asarray(result["n_ion"], dtype=float)
    exact_ws_boundary = _uses_decoupled_ion_sphere_grid(
        cfg,
        r_ws=float(result["r_ws"]),
        rmax=float(r[-1]),
    )
    q_full_ws = _ws_charge(
        r, np.asarray(result["n_full"], dtype=float), float(result["r_ws"]),
        interpolate_boundary=exact_ws_boundary,
    )
    n_free_ws_profile = np.asarray(
        result.get("n_free", result["n_cont"]),
        dtype=float,
    )
    ws_mask = r <= float(result["r_ws"])
    if not np.all(np.isfinite(n_free_ws_profile[ws_mask])):
        raise RuntimeError(
            "The direct Eq. (A3) continuum domain does not cover the WS sphere; "
            "q_cont_ws cannot be evaluated from positive-energy states."
        )
    q_cont_ws = _ws_charge(
        r, n_free_ws_profile, float(result["r_ws"]),
        interpolate_boundary=exact_ws_boundary,
    )
    q_ion_ws = _ws_charge(
        r, n_ion, float(result["r_ws"]),
        interpolate_boundary=exact_ws_boundary,
    )
    bound_q_ion_ws = np.asarray(
        result.get("bound_q_ion_ws", np.empty((0, 0))),
        dtype=float,
    )
    bound_q_ion_ws_sum = (
        float(np.nansum(bound_q_ion_ws)) if bound_q_ion_ws.size else np.nan
    )
    bound_q_ion_ws_closure_residual = (
        float(bound_q_ion_ws_sum - q_ion_ws)
        if np.isfinite(bound_q_ion_ws_sum)
        else np.nan
    )
    q_ext_ws = (
        _ws_charge(
            r, np.asarray(result["n_ext"], dtype=float), float(result["r_ws"]),
            interpolate_boundary=exact_ws_boundary,
        )
        if "n_ext" in result
        else np.nan
    )
    zbar_ws = float(z_nuc) - q_ion_ws
    q_ion_all = 4.0 * np.pi * trapz_integral((r**2) * n_ion, r)
    zbar_partition = float(z_nuc) - float(q_ion_all)
    result["q_full_ws"] = float(q_full_ws)
    result["q_cont_ws"] = float(q_cont_ws)
    result["q_cont_ws_definition"] = "integral-of-n_free-Eq-A3-within-Rws"
    result["q_ion_ws"] = q_ion_ws
    result["bound_q_ion_ws_sum"] = bound_q_ion_ws_sum
    result["bound_q_ion_ws_closure_residual"] = (
        bound_q_ion_ws_closure_residual
    )
    result["bound_q_ion_ws_definition"] = (
        "2(2l+1) f_FD(E_nl) M(E_nl) integral_0^Rws "
        "f_cut(r) |P_nl(r)|^2 dr"
    )
    result["q_ion_all"] = float(q_ion_all)
    result["q_ext_ws"] = float(q_ext_ws) if np.isfinite(q_ext_ws) else np.nan
    result["zbar_ws"] = float(zbar_ws)
    result["zbar_partition"] = float(zbar_partition)
    # Backward-compatible AA diagnostic.  The downstream QOZ workflow keeps
    # this separate from the TCP charge defined by integral(n_scr), following
    # Starrett & Saumon (2014), Eqs. (9), (10), and (15).
    result["zbar"] = zbar_ws

    full_controls = _resolve_b3_tail_controls(
        cfg,
        r_ws=float(r_ws),
        rmax=float(rmax),
        stage_mode=str(cfg.b3_tail_stage2_mode),
    )
    ext_controls = _resolve_b3_tail_controls(
        cfg,
        r_ws=float(r_ws),
        rmax=float(rmax),
        stage_mode=str(cfg.ext_b3_tail_mode),
    )

    screening_meta: dict[str, Any] = {"mode": str(cfg.screening_tail_repair_mode), "applied": False}
    if do_external:
        screening_mode = str(cfg.screening_tail_repair_mode or "off").strip().lower()
        if screening_mode not in ("off", "constrained_b3"):
            raise ValueError("screening_tail_repair_mode must be 'off' or 'constrained_b3'.")
        if screening_mode == "constrained_b3":
            n_scr_raw = np.asarray(result["n_scr"], dtype=float)
            q_scr_box_raw = (
                4.0 * np.pi * trapz_integral((r**2) * n_scr_raw, r)
            )
            q_scr_rel_raw = abs(float(q_scr_box_raw) - float(zbar_partition)) / max(
                abs(float(zbar_partition)), 1.0e-12
            )
            screening_meta = {
                "mode": "constrained_b3",
                "applied": False,
                "q_scr_raw": float(q_scr_box_raw),
                "q_scr_rel_raw": float(q_scr_rel_raw),
                "q_scr_target": float(zbar_partition),
                "q_scr_target_source": "Z_minus_Qion_all",
            }
            if q_scr_rel_raw > float(cfg.screening_tail_repair_rel_tol):
                repair_r_cut = None
                if cfg.screening_tail_r_cut_mult is not None:
                    repair_r_cut = float(cfg.screening_tail_r_cut_mult) * float(r_ws)
                elif full_controls["r_cut"] is not None:
                    repair_r_cut = float(full_controls["r_cut"])
                elif ext_controls["r_cut"] is not None:
                    repair_r_cut = float(ext_controls["r_cut"])

                repair_r_fit_max = None
                if cfg.screening_tail_r_fit_max_mult is not None:
                    repair_r_fit_max = float(cfg.screening_tail_r_fit_max_mult) * float(r_ws)
                elif full_controls["r_fit_max"] is not None:
                    repair_r_fit_max = float(full_controls["r_fit_max"])
                elif ext_controls["r_fit_max"] is not None:
                    repair_r_fit_max = float(ext_controls["r_fit_max"])

                if repair_r_cut is not None:
                    try:
                        n_scr_fixed, screening_meta = _repair_screening_density_tail(
                            r,
                            n_scr_raw,
                            zbar=float(zbar_partition),
                            mu_id=float(result["mu"]),
                            temperature=float(temperature_ha),
                            r_cut=float(repair_r_cut),
                            r_fit_max=None if repair_r_fit_max is None else float(repair_r_fit_max),
                            charge_weight=float(cfg.screening_tail_charge_weight),
                        )
                        # Keep the defining pseudoatom identities canonical
                        # (Starrett & Saumon 2014, Eqs. 9-10):
                        #
                        #   n_pa  = n_full - n_ext
                        #   n_scr = n_pa - n_ion.
                        #
                        # The constrained B3 fit is useful for diagnosing how
                        # much outer-tail charge would be needed, but replacing
                        # n_scr alone breaks those identities and can create a
                        # spurious low-k QOZ/HNC instability.  Store it under
                        # explicit diagnostic names instead.
                        result["n_scr_repaired"] = np.asarray(n_scr_fixed, dtype=float)
                        result["n_pa_repaired"] = (
                            np.asarray(n_scr_fixed, dtype=float) + n_ion
                        )
                        screening_meta["q_scr_target_source"] = "Z_minus_Qion_all"
                        screening_meta["diagnostic_only"] = True
                        screening_meta["canonical_profiles_preserved"] = True
                        screening_meta["used_for_qoz"] = False
                    except Exception as exc:
                        screening_meta = {
                            **screening_meta,
                            "applied": False,
                            "error": str(exc),
                        }
                else:
                    screening_meta = {
                        **screening_meta,
                        "applied": False,
                        "error": "missing r_cut",
                    }
    if "n_scr" in result:
        q_scr_all = 4.0 * np.pi * trapz_integral(
            (r**2) * np.asarray(result["n_scr"], dtype=float), r
        )
        result["q_scr_all"] = float(q_scr_all)
    if "n_pa" in result:
        q_pa_all = 4.0 * np.pi * trapz_integral(
            (r**2) * np.asarray(result["n_pa"], dtype=float), r
        )
        result["q_pa_all"] = float(q_pa_all)
        result["pa_charge_residual"] = float(q_pa_all - float(z_nuc))
    result["screening_tail_repair"] = dict(screening_meta)

    stage2_mixer_scheme, stage2_mixer_m, stage2_mixer_mix = _stage2_mixer_values()
    zero_tail_states = result.get("zero_tail_bound_meta", {}).get("states", [])
    zero_tail_state = dict(zero_tail_states[0]) if zero_tail_states else {}
    ph_enabled = bool(float(cfg.ph_kappa) > 0.0 and int(cfg.ph_kappa_iters) > 0)
    stage2_update_count = int(result.get("stage2_iters", 0))
    stage2_final_ph_kappa = float(
        result.get(
            "final_ph_kappa",
            (
                float(cfg.ph_kappa)
                if ph_enabled
                and stage2_update_count > 0
                and stage2_update_count <= int(cfg.ph_kappa_iters)
                else 0.0
            ),
        )
    )
    ext_final_ph_kappa = float(
        result.get("ext_status", {}).get("final_ph_kappa", 0.0)
    )
    final_ph_kappa = max(stage2_final_ph_kappa, ext_final_ph_kappa)
    if bool(result.get("stage2_converged", False)) and stage2_final_ph_kappa > 0.0:
        raise RuntimeError(
            "Internal error: stage-2 SCF was accepted before an unscreened "
            "Poisson update after Poisson-Helmholtz preconditioning."
        )
    if bool(result.get("ext_status", {}).get("converged", False)) and ext_final_ph_kappa > 0.0:
        raise RuntimeError(
            "Internal error: external SCF was accepted before an unscreened "
            "Poisson update after Poisson-Helmholtz preconditioning."
        )

    final_tail_target = str(full_controls["target"])
    if final_tail_target in ("full", "both"):
        final_charge_tail_meta = dict(result.get("n_full_tail_meta", {}))
    else:
        final_charge_tail_meta = dict(result.get("n_cont_tail_meta", {}))
    ext_charge_tail_meta = dict(
        result.get("ext_status", {}).get("tail_meta", {})
    )
    final_charge_constraint_requested = bool(
        cfg.b3_source_charge_constraint and str(full_controls["mode"]) == "in_scf"
    )
    ext_charge_constraint_requested = bool(
        do_external
        and cfg.b3_source_charge_constraint
        and str(ext_controls["mode"]) == "in_scf"
    )

    meta = {
        "element": symbol,
        "Z": int(z_nuc),
        "atomic_mass": atomic_weight,
        "rho_g_cc": float(cfg.rho_g_cc),
        "temperature_ev": float(cfg.temperature_ev),
        "temperature_ha": float(temperature_ha),
        "xc_model": str(cfg.xc_model),
        "xc_provenance": dict(result["xc_provenance"]),
        "gga_core_mode": str(cfg.gga_core_mode),
        "gga_core_zr": float(cfg.gga_core_zr),
        "gga_core_radius_bohr": float(
            result.get("gga_core_radius_bohr", np.nan)
        ),
        "gga_core_points": int(result.get("gga_core_points", 0)),
        "gga_radial_operator": str(
            result.get("gga_radial_operator", "not_applicable")
        ),
        "density_cusp_rel_error": float(
            result.get("density_cusp_rel_error", np.nan)
        ),
        "v_xc_core_turn_count": int(
            result.get("v_xc_core_turn_count", 0)
        ),
        "n_i_bohr3": float(n_i),
        "r_ws_bohr": float(r_ws),
        "ws_charge_quadrature": (
            "exact_boundary_linear" if exact_ws_boundary else "sampled_step_grid"
        ),
        "ion_sphere_background_quadrature": (
            "analytic_sharp_shell"
            if _uses_analytic_ion_sphere_background(
                cfg, r_ws=float(r_ws), rmax=float(rmax)
            )
            else (
                "sampled_step" if cfg.g_ii_override is None else "sampled_g_ii_override"
            )
        ),
        "r_geom_bohr": float(geometry["r_geom"]),
        "r_max_bohr": float(rmax),
        "r_max_eff_mult": float(geometry["rmax_eff_mult"]),
        "bound_r_max_bohr": float(np.asarray(result.get("r_bound", result["r"]))[-1]),
        "bound_rmax_mult": (
            float(cfg.bound_rmax_mult) if cfg.bound_rmax_mult is not None else np.nan
        ),
        "bound_zero_tail_refine": bool(cfg.bound_zero_tail_refine),
        "bound_zero_tail_min_binding_ha": float(cfg.bound_zero_tail_min_binding_ha),
        "bound_zero_tail_max_binding_ha": float(cfg.bound_zero_tail_max_binding_ha),
        "bound_zero_tail_scan_points": int(cfg.bound_zero_tail_scan_points),
        "bound_zero_tail_l_max": int(cfg.bound_zero_tail_l_max),
        "bound_zero_tail_edge_rel_tol": float(cfg.bound_zero_tail_edge_rel_tol),
        "bound_zero_tail_applied": bool(
            result.get("zero_tail_bound_meta", {}).get("applied", False)
        ),
        "geometry_r_ws_floor_bohr": float(cfg.geometry_r_ws_floor_bohr),
        "geometry_r_ws_cap_bohr": (
            float(cfg.geometry_r_ws_cap_bohr)
            if cfg.geometry_r_ws_cap_bohr is not None
            else np.nan
        ),
        "q_full_ws": float(q_full_ws),
        "q_cont_ws": float(q_cont_ws),
        "q_ext_ws": float(q_ext_ws) if np.isfinite(q_ext_ws) else np.nan,
        "n_points": int(cfg.n_points),
        "mu_bounds": tuple(float(x) for x in cfg.mu_bounds),
        "full_fixed_mu_ha": (
            float(cfg.full_fixed_mu_ha) if cfg.full_fixed_mu_ha is not None else np.nan
        ),
        "mu_stage1_ha": float(mu_stage1),
        "mu_final_ha": float(result["mu"]),
        "final_mu": float(result["mu"]),
        "n0_final_bohr3": float(result["n0"]),
        "cont_e_max": float(cfg.cont_e_max),
        "cont_stage2_e_max_mode": str(cfg.cont_stage2_e_max_mode),
        "cont_stage2_e_max_occ_tol": float(cfg.cont_stage2_e_max_occ_tol),
        "cont_stage2_e_max_floor": float(cfg.cont_stage2_e_max_floor),
        "cont_stage2_e_max_final": float(
            result.get("stage2_cont_e_max_final", cfg.cont_e_max)
        ),
        "cont_n_jobs": int(cfg.cont_n_jobs),
        "cont_n_e_base": int(cfg.cont_n_e_base),
        "cont_energy_mode": str(cfg.cont_energy_mode),
        "cont_adaptive_mode_stage1": str(cfg.cont_adaptive_mode_stage1),
        "cont_adaptive_mode_stage2": str(cfg.cont_adaptive_mode_stage2),
        "cont_e_base_grid": str(cfg.cont_e_base_grid),
        "cont_dE_min": float(cfg.cont_dE_min),
        "cont_e_min": float(cfg.cont_e_min),
        "cont_near_zero_log_grid": bool(cfg.cont_near_zero_log_grid),
        "cont_near_zero_log_points_per_decade": int(
            cfg.cont_near_zero_log_points_per_decade
        ),
        "cont_near_zero_log_max_nodes": int(cfg.cont_near_zero_log_max_nodes),
        "cont_near_zero_log_max_energy": (
            float(cfg.cont_near_zero_log_max_energy)
            if cfg.cont_near_zero_log_max_energy is not None
            else np.nan
        ),
        "cont_resonance_theta_l_min": int(cfg.cont_resonance_theta_l_min),
        "cont_resonance_theta_probe_count": int(cfg.cont_resonance_theta_probe_count),
        "cont_resonance_theta_scan_depth": int(cfg.cont_resonance_theta_scan_depth),
        "cont_resonance_theta_scout_max_extra_nodes": (
            int(cfg.cont_resonance_theta_scout_max_extra_nodes)
            if cfg.cont_resonance_theta_scout_max_extra_nodes is not None
            else -1
        ),
        "cont_resonance_theta_root_tol": (
            float(cfg.cont_resonance_theta_root_tol)
            if cfg.cont_resonance_theta_root_tol is not None
            else np.nan
        ),
        "cont_resonance_theta_sharpness_min": float(
            cfg.cont_resonance_theta_sharpness_min
        ),
        "cont_resonance_theta_max_roots": (
            int(cfg.cont_resonance_theta_max_roots)
            if cfg.cont_resonance_theta_max_roots is not None
            else -1
        ),
        "cont_resonance_theta_refine_depth": (
            int(cfg.cont_resonance_theta_refine_depth)
            if cfg.cont_resonance_theta_refine_depth is not None
            else -1
        ),
        "cont_rmax_mult": cfg.cont_rmax_mult,
        "cont_rmax_eff_mult": (
            float(geometry["solve_rmax_eff_mult"])
            if geometry["solve_rmax_eff_mult"] is not None
            else np.nan
        ),
        "cont_parallel_mode": str(cfg.cont_parallel_mode),
        "cont_shards": cfg.cont_shards,
        "full_r_dft_max_bohr": (
            float(full_controls["solve_rmax"])
            if full_controls["solve_rmax"] is not None
            else float(rmax)
        ),
        "full_match_r_cut_bohr": float(cont_stage2.get("match_r_cut", np.nan)),
        "full_match_width_bohr": float(cont_stage2.get("match_width", np.nan)),
        "full_match_hi_bohr": float(
            float(cont_stage2.get("match_r_cut", np.nan))
            + float(cont_stage2.get("match_width", np.nan))
        ),
        "full_match_kr_min": float(cont_stage2.get("match_kr_min", np.nan)),
        "full_cont_l_max": float(cont_stage2.get("l_max", np.nan)),
        "full_cont_l_max_ceiling": float(cont_stage2.get("l_max_ceiling", np.nan)),
        "full_r_fit_max_bohr": (
            float(full_controls["r_fit_max"])
            if full_controls["r_fit_max"] is not None
            else np.nan
        ),
        "full_r_cut_bohr": (
            float(full_controls["r_cut"])
            if full_controls["r_cut"] is not None
            else np.nan
        ),
        "b3_tail_stage1_mode": str(cfg.b3_tail_stage1_mode),
        "b3_tail_stage2_mode": str(full_controls["mode"]),
        "b3_tail_stage2_mode_raw": str(cfg.b3_tail_stage2_mode),
        "b3_tail_target": str(full_controls["target"]),
        "b3_tail_target_raw": str(cfg.b3_tail_target),
        "b3_tail_fit_points": int(cfg.b3_tail_fit_points),
        "b3_tail_local_fit_width_mult": (
            np.nan
            if cfg.b3_tail_local_fit_width_mult is None
            else float(cfg.b3_tail_local_fit_width_mult)
        ),
        "b3_tail_fit_window_mode": str(cfg.b3_tail_fit_window_mode),
        "b3_tail_fit_window_mode_resolved": str(full_controls["fit_window_mode"]),
        "b3_tail_blend_points": int(cfg.b3_tail_blend_points),
        "b3_tail_model": str(cfg.b3_tail_model),
        "b3_tail_auto_rel_improve_tol": float(cfg.b3_tail_auto_rel_improve_tol),
        "b3_tail_auto_signal_rel_tol": float(cfg.b3_tail_auto_signal_rel_tol),
        "b3_r_fit_max_mult": cfg.b3_r_fit_max_mult,
        "b3_r_fit_max_eff_mult": (
            float(geometry["r_fit_max_eff_mult"])
            if geometry["r_fit_max_eff_mult"] is not None
            else np.nan
        ),
        "b3_r_cut_mult": cfg.b3_r_cut_mult,
        "b3_r_cut_eff_mult": (
            float(geometry["r_cut_eff_mult"])
            if geometry["r_cut_eff_mult"] is not None
            else np.nan
        ),
        "b3_cut_width": cfg.b3_cut_width,
        "b3_fallback_on_error": bool(cfg.b3_fallback_on_error),
        "b3_charge_constraint_requested": bool(final_charge_constraint_requested),
        "b3_charge_constraint_applied": bool(
            final_charge_tail_meta.get("charge_constraint_applied", False)
        ),
        "b3_charge_constraint_residual": float(
            final_charge_tail_meta.get("charge_constraint_residual", np.nan)
        ),
        "b3_charge_constraint_fit_rms_ratio": float(
            final_charge_tail_meta.get("charge_constraint_fit_rms_ratio", np.nan)
        ),
        "b3_charge_constraint_coeff_delta_rel": float(
            final_charge_tail_meta.get("charge_constraint_coeff_delta_rel", np.nan)
        ),
        "b3_charge_constraint_profile_delta_rel": float(
            final_charge_tail_meta.get("charge_constraint_profile_delta_rel", np.nan)
        ),
        "b3_charge_constraint_min_density": float(
            final_charge_tail_meta.get("charge_constraint_tail_min", np.nan)
        ),
        "b3_charge_constraint_failure_reason": str(
            final_charge_tail_meta.get("charge_constraint_failure_reason", "")
        ),
        "b3_charge_constraint_fit_rms_ratio_max": (
            np.nan
            if cfg.b3_charge_constraint_fit_rms_ratio_max is None
            else float(cfg.b3_charge_constraint_fit_rms_ratio_max)
        ),
        "b3_charge_constraint_profile_delta_rel_max": (
            np.nan
            if cfg.b3_charge_constraint_profile_delta_rel_max is None
            else float(cfg.b3_charge_constraint_profile_delta_rel_max)
        ),
        "cont_tail_match": bool(cfg.cont_tail_match),
        "cont_tail_match_target": str(cfg.cont_tail_match_target),
        "cont_tail_fit_points": int(cfg.cont_tail_fit_points),
        "cont_tail_blend_points": int(cfg.cont_tail_blend_points),
        "cont_tail_r_cut_frac": cfg.cont_tail_r_cut_frac,
        "cont_tail_cut_width": cfg.cont_tail_cut_width,
        "cont_tail_fallback_on_error": bool(cfg.cont_tail_fallback_on_error),
        "source_closure": bool(cfg.source_closure),
        "source_r_trust_frac": float(cfg.source_r_trust_frac),
        "source_blend_frac": float(cfg.source_blend_frac),
        "source_charge_closure": bool(cfg.source_charge_closure),
        "full_b3_use_source_closure": (
            "auto"
            if cfg.full_b3_use_source_closure is None
            else bool(cfg.full_b3_use_source_closure)
        ),
        "full_b3_source_closure_resolved": bool(
            _resolve_full_b3_source_closure_policy(
                setting=cfg.full_b3_use_source_closure,
                tail_target=str(full_controls["target"]),
            )
        ),
        "ext_b3_use_source_closure": (
            "auto"
            if cfg.ext_b3_use_source_closure is None
            else bool(cfg.ext_b3_use_source_closure)
        ),
        "ext_source_closure_auto_rel_tol": float(cfg.ext_source_closure_auto_rel_tol),
        "n0_mode_override": (
            None if cfg.n0_mode_override is None else str(cfg.n0_mode_override)
        ),
        "n0_fixed_override": (
            float(cfg.n0_fixed_override)
            if cfg.n0_fixed_override is not None
            else np.nan
        ),
        "full_n0_mode": str(
            _resolve_n0_mode(cfg, b3_tail_mode=str(full_controls["mode"]))
        ),
        "run_mode": str(cfg.run_mode),
        "ext_enabled": bool(do_external),
        "ext_match_v_tol": cfg.ext_match_v_tol,
        "ext_b3_tail_mode": str(ext_controls["mode"]),
        "ext_b3_tail_mode_raw": str(cfg.ext_b3_tail_mode),
        "ext_b3_tail_model": (
            str(cfg.ext_b3_tail_model) if cfg.ext_b3_tail_model is not None else ""
        ),
        "ext_b3_charge_constraint_requested": bool(ext_charge_constraint_requested),
        "ext_b3_charge_constraint_applied": bool(
            ext_charge_tail_meta.get("charge_constraint_applied", False)
        ),
        "ext_b3_charge_constraint_residual": float(
            ext_charge_tail_meta.get("charge_constraint_residual", np.nan)
        ),
        "ext_b3_charge_constraint_fit_rms_ratio": float(
            ext_charge_tail_meta.get("charge_constraint_fit_rms_ratio", np.nan)
        ),
        "ext_b3_charge_constraint_coeff_delta_rel": float(
            ext_charge_tail_meta.get("charge_constraint_coeff_delta_rel", np.nan)
        ),
        "ext_b3_charge_constraint_profile_delta_rel": float(
            ext_charge_tail_meta.get("charge_constraint_profile_delta_rel", np.nan)
        ),
        "ext_b3_charge_constraint_min_density": float(
            ext_charge_tail_meta.get("charge_constraint_tail_min", np.nan)
        ),
        "ext_b3_charge_constraint_failure_reason": str(
            ext_charge_tail_meta.get("charge_constraint_failure_reason", "")
        ),
        "ext_n0_mode": str(
            _resolve_n0_mode(cfg, b3_tail_mode=str(ext_controls["mode"]))
        ),
        "ext_r_dft_max_bohr": (
            float(ext_controls["solve_rmax"])
            if ext_controls["solve_rmax"] is not None
            else float(rmax)
        ),
        "ext_r_fit_max_bohr": (
            float(ext_controls["r_fit_max"])
            if ext_controls["r_fit_max"] is not None
            else np.nan
        ),
        "ext_r_cut_bohr": (
            float(ext_controls["r_cut"])
            if ext_controls["r_cut"] is not None
            else np.nan
        ),
        "screening_tail_repair_mode": str(cfg.screening_tail_repair_mode),
        "screening_tail_repair_rel_tol": float(cfg.screening_tail_repair_rel_tol),
        "screening_tail_charge_weight": float(cfg.screening_tail_charge_weight),
        "screening_tail_r_cut_mult": (
            float(cfg.screening_tail_r_cut_mult)
            if cfg.screening_tail_r_cut_mult is not None
            else np.nan
        ),
        "screening_tail_r_fit_max_mult": (
            float(cfg.screening_tail_r_fit_max_mult)
            if cfg.screening_tail_r_fit_max_mult is not None
            else np.nan
        ),
        "ext_iters": int(result.get("ext_status", {}).get("iters", 0)),
        "ext_err": float(result.get("ext_status", {}).get("err", np.nan)),
        "ext_converged": bool(result.get("ext_status", {}).get("converged", False)),
        "stage1_iters": int(result.get("stage1_iters", 0)),
        "stage2_iters": int(result.get("stage2_iters", 0)),
        "stage1_skipped": bool(result.get("stage1_skipped", False)),
        "stage1_converged": bool(result.get("stage1_converged", False)),
        "stage2_converged": bool(result.get("stage2_converged", False)),
        "threshold_state_status": str(result.get("threshold_state_status", "none")),
        "threshold_state_localization": str(
            result.get("threshold_state_localization", "none")
        ),
        "threshold_state_representation": str(
            result.get("threshold_state_representation", "none")
        ),
        "threshold_spectral_representation_status": str(
            result.get("threshold_spectral_representation_status", "none")
        ),
        "threshold_tail_domain_status": str(
            result.get("threshold_tail_domain_status", "none")
        ),
        "shallowest_bound_energy_ha": float(
            result.get("shallowest_bound_energy_ha", np.nan)
        ),
        "bound_box_decay_metric": float(result.get("bound_box_decay_metric", np.nan)),
        "bound_box_decay_metric_from_rws": float(
            result.get("bound_box_decay_metric_from_rws", np.nan)
        ),
        "bound_asymptotic_start_bohr": float(
            result.get("bound_asymptotic_start_bohr", np.nan)
        ),
        "bound_zero_extension_edge_ratio": float(
            result.get("bound_zero_extension_edge_ratio", np.nan)
        ),
        "bound_zero_tail_exterior_probability": float(
            result.get("bound_zero_tail_exterior_probability", np.nan)
        ),
        "bound_zero_tail_finite_wall_energy_ha": float(
            zero_tail_state.get("finite_wall_energy_ha", np.nan)
        ),
        "bound_zero_tail_matched_energy_ha": float(
            zero_tail_state.get("matched_energy_ha", np.nan)
        ),
        "bound_zero_tail_action": str(zero_tail_state.get("action", "none")),
        "bound_zero_tail_attractive_edge_ratio": float(
            zero_tail_state.get("edge_relative_to_binding", np.nan)
        ),
        "bound_zero_tail_absolute_edge_ratio": float(
            zero_tail_state.get("edge_absolute_relative_to_binding", np.nan)
        ),
        "bound_tail_potential_ratio": float(
            result.get("bound_tail_potential_ratio", np.nan)
        ),
        "bound_probability_inside_rws": float(
            result.get("bound_probability_inside_rws", np.nan)
        ),
        "bound_probability_inside_continuum_rmax": float(
            result.get("bound_probability_inside_continuum_rmax", np.nan)
        ),
        "bound_mean_radius_over_rws": float(
            result.get("bound_mean_radius_over_rws", np.nan)
        ),
        "bound_rms_radius_over_rws": float(
            result.get("bound_rms_radius_over_rws", np.nan)
        ),
        "stage2_scf_mixing_scheme": str(stage2_mixer_scheme),
        "stage2_scf_mixing_m": int(stage2_mixer_m),
        "stage2_scf_mix": float(stage2_mixer_mix),
        "bound_energy_cut_mode": str(cfg.bound_energy_cut_mode),
        "bound_energy_cut_value": float(cfg.bound_energy_cut),
        "bound_energy_cut_ha": float(energy_cut),
        "bound_occ_mode": str(cfg.bound_occ_mode),
        "ph_kappa": float(cfg.ph_kappa) if ph_enabled else 0.0,
        "ph_kappa_iters": int(cfg.ph_kappa_iters) if ph_enabled else 0,
        "final_ph_kappa": float(final_ph_kappa),
        "stage2_final_ph_kappa": float(stage2_final_ph_kappa),
        "ext_final_ph_kappa": float(ext_final_ph_kappa),
        "v_decomposition_source_density": str(
            result.get("v_decomposition_source_density", "n_full")
        ),
        "v_decomposition_source_provenance": str(
            result.get("v_decomposition_source_provenance", "reported_density")
        ),
        "full_v_eff_outer_decay": bool(cfg.full_v_eff_outer_decay),
        "full_v_eff_outer_decay_start_rws": float(cfg.full_v_eff_outer_decay_start_rws),
        "full_v_eff_outer_decay_length_rws": float(
            cfg.full_v_eff_outer_decay_length_rws
        ),
        "ion_bound_gamma": float(cfg.ion_bound_gamma),
        "ion_gamma_mode": str(cfg.ion_gamma_mode),
        "ion_gamma_scale": float(cfg.ion_gamma_scale),
        "ion_gamma_final": float(final_ion_gamma),
        "g_ii_override_used": bool(cfg.g_ii_override is not None),
        "v_corr_full_used": bool(v_corr_full_meta["used"]),
        "v_corr_full_interpolated": bool(v_corr_full_meta["interpolated"]),
        "v_corr_ext_used": bool(v_corr_ext_meta["used"]),
        "v_corr_ext_interpolated": bool(v_corr_ext_meta["interpolated"]),
        "ion_cut_mode": str(cfg.ion_cut_mode),
        "ion_cut_c": float(cfg.ion_cut_c),
        "ion_ws_weight_min": float(cfg.ion_ws_weight_min),
        "q_ion_ws": float(q_ion_ws),
        "bound_q_ion_ws_sum": float(bound_q_ion_ws_sum),
        "bound_q_ion_ws_closure_residual": float(
            bound_q_ion_ws_closure_residual
        ),
        "q_ion_all": float(q_ion_all),
        "q_pa_all": float(result.get("q_pa_all", np.nan)),
        "q_scr_all": float(result.get("q_scr_all", np.nan)),
        "pa_charge_residual": float(result.get("pa_charge_residual", np.nan)),
        "zbar_ws": float(zbar_ws),
        "zbar_partition": float(zbar_partition),
        "zbar": float(zbar_ws),
        "continuation_v_full_init_used": bool(v_full_init_meta["used"]),
        "continuation_v_full_init_interpolated": bool(v_full_init_meta["interpolated"]),
        "continuation_v_full_init_source_r_min": float(
            v_full_init_meta["source_r_min"]
        ),
        "continuation_v_full_init_source_r_max": float(
            v_full_init_meta["source_r_max"]
        ),
        "continuation_v_ext_init_used": bool(v_ext_init_meta["used"]),
        "continuation_v_ext_init_interpolated": bool(v_ext_init_meta["interpolated"]),
        "continuation_v_ext_init_source_r_min": float(v_ext_init_meta["source_r_min"]),
        "continuation_v_ext_init_source_r_max": float(v_ext_init_meta["source_r_max"]),
        "continuation_stage2_from_init": bool(use_stage2_continuation_init),
        "continuation_mu_init": (
            float(cfg.continuation_mu_init)
            if cfg.continuation_mu_init is not None
            else np.nan
        ),
        "continuation_mu_stage2_guess": float(mu_stage2_guess),
        "continuation_scf_mixing_scheme": (
            str(cfg.continuation_scf_mixing_scheme)
            if cfg.continuation_scf_mixing_scheme is not None
            else ""
        ),
        "continuation_scf_mixing_m": (
            int(cfg.continuation_scf_mixing_m)
            if cfg.continuation_scf_mixing_m is not None
            else -1
        ),
        "continuation_scf_mix": (
            float(cfg.continuation_scf_mix)
            if cfg.continuation_scf_mix is not None
            else np.nan
        ),
    }
    # Keep every scalar V_eff tail diagnostic directly accessible from both
    # the ordinary result and saved metadata.  These fields are intentionally
    # diagnostic-only and do not override threshold_state_status.
    meta.update(veff_asymptotic_diag)
    result["meta"] = meta

    if cfg.save_data:
        paths = save_full_external_data(
            output_dir=cfg.save_output_dir,
            element_symbol=symbol,
            z=int(z_nuc),
            temperature_ev=float(cfg.temperature_ev),
            rho_g_cc=float(cfg.rho_g_cc),
            suffix=str(cfg.save_suffix),
            result=result,
            metadata=meta,
        )
        result["saved_paths"] = paths

    return result

def solve_full_only(cfg: FullExternalConfig) -> dict[str, Any]:
    """
    Run only the full AA workflow.

    This is a thin, explicit wrapper around `solve_full_then_external(...)`
    that forces `run_mode="full"`. It exists to keep user code and tests
    unambiguous when the external branch is intentionally out of scope.
    """
    cfg_full = FullExternalConfig(**{**cfg.__dict__, "run_mode": "full"})
    return solve_full_then_external(cfg_full)


def with_continuation_initial_guess(
    cfg: FullExternalConfig,
    previous_result: dict[str, Any] | None,
    *,
    enabled: bool = True,
    reuse_external: bool = False,
    stage2_from_init: bool = False,
) -> FullExternalConfig:
    """
    Return a config that reuses a previous converged potential as SCF input.

    The previous potential is stored with its source radial grid. The actual
    interpolation happens inside `solve_full_then_external(...)`, after the
    target geometry is known. This keeps the helper safe for scans where
    `R_ws`, `r_max`, or `n_points` changes between neighboring states.
    """
    if not bool(enabled) or previous_result is None:
        return replace(cfg)
    if "r" not in previous_result or "v_full" not in previous_result:
        return replace(cfg)

    r_src = np.asarray(previous_result["r"], dtype=float).copy()
    v_full_src = np.asarray(previous_result["v_full"], dtype=float).copy()
    updates: dict[str, Any] = {
        "v_full_init": v_full_src,
        "v_full_init_r": r_src,
        "continuation_stage2_from_init": bool(stage2_from_init),
    }
    if "mu" in previous_result:
        updates["continuation_mu_init"] = float(previous_result["mu"])
    if bool(reuse_external) and "v_ext" in previous_result:
        v_ext_src = np.asarray(previous_result["v_ext"], dtype=float).copy()
        if v_ext_src.shape == r_src.shape:
            updates["v_ext_init"] = v_ext_src
            updates["v_ext_init_r"] = r_src.copy()
    return replace(cfg, **updates)


@dataclass
class FullExternalContinuation:
    """
    Small stateful helper for rho/Te scans with optional warm-started SCF.

    Usage
    -----
    cache = FullExternalContinuation(enabled=True)
    for cfg in cfgs:
        result = cache.solve_full_only(cfg)

    Only converged stage-2 results are cached by default. This avoids seeding
    later states from a suspicious branch while preserving the user-visible
    cold-start behavior when `enabled=False`.
    """

    enabled: bool = True
    reuse_external: bool = False
    stage2_from_init: bool = False
    update_on_unconverged: bool = False
    previous_result: dict[str, Any] | None = None

    def prepare(self, cfg: FullExternalConfig) -> FullExternalConfig:
        return with_continuation_initial_guess(
            cfg,
            self.previous_result,
            enabled=bool(self.enabled),
            reuse_external=bool(self.reuse_external),
            stage2_from_init=bool(self.stage2_from_init),
        )

    def update(self, result: dict[str, Any]) -> None:
        converged = bool(result.get("stage2_converged", result.get("converged", False)))
        if bool(self.update_on_unconverged) or converged:
            self.previous_result = {
                "r": np.asarray(result["r"], dtype=float).copy(),
                "v_full": np.asarray(result["v_full"], dtype=float).copy(),
                "mu": float(result["mu"]),
            }
            if "v_ext" in result:
                self.previous_result["v_ext"] = np.asarray(result["v_ext"], dtype=float).copy()

    def solve_full_only(self, cfg: FullExternalConfig) -> dict[str, Any]:
        result = solve_full_only(self.prepare(cfg))
        self.update(result)
        return result

    def solve_full_then_external(self, cfg: FullExternalConfig) -> dict[str, Any]:
        result = solve_full_then_external(self.prepare(cfg))
        self.update(result)
        return result


def run_minimal(
    *,
    element: int | str,
    temperature_ev: float,
    rho_g_cc: float,
    electronic_model: str = "qm",
    xc_model: str = "dirac",
    gga_core_mode: str = "finite",
    gga_core_zr: float = 0.05,
    run_mode: str = "full+ext",
    show_scf_progress: bool = False,
    save_data: bool = False,
    save_output_dir: str | Path = "outputs",
    save_suffix: str = "",
) -> dict[str, Any]:
    """
    Convenience entry-point for the minimal-input workflow.
    """
    cfg = FullExternalConfig(
        element=element,
        temperature_ev=float(temperature_ev),
        rho_g_cc=float(rho_g_cc),
        electronic_model=str(electronic_model),
        xc_model=str(xc_model),
        gga_core_mode=str(gga_core_mode),
        gga_core_zr=float(gga_core_zr),
        run_mode=str(run_mode),
        show_scf_progress=bool(show_scf_progress),
        save_data=bool(save_data),
        save_output_dir=save_output_dir,
        save_suffix=str(save_suffix),
    )
    return solve_full_then_external(cfg)
