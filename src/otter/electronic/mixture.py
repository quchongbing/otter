"""
otter/electronic/mixture.py

General multicomponent AA electronic structure.

This module extends the current binary common-chemical-potential construction
to an arbitrary number of ionic species. The physical constraints follow
Starrett et al. (2014):

  (1) all average atoms share one electron chemical potential
  (2) the per-species ion-sphere volumes satisfy the mixture volume sum rule

The implementation keeps the AA work itself unchanged. The only new outer loop
solves for positive per-species volume weights and then reuses the existing
single-species `FullExternalConfig` solver for each component.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq, least_squares

from otter.numerics.grids import create_sqrt_grid
from otter.data.elements import element as element_info
from otter.data.helpers import ion_density_bohr3
from otter.io import save_mixture_data
from .full_external import (
    FullExternalConfig,
    _resolve_outer_geometry,
    solve_full_only,
    solve_full_then_external,
)


_FORBIDDEN_SPECIES_KEYS = {
    "element",
    "temperature_ev",
    "rho_g_cc",
    "run_mode",
    "ext_scf_enabled",
    "r_ws_override_bohr",
    "n_i_override_bohr3",
    "v_full_init",
    "v_ext_init",
}


def _validate_species_overrides(overrides: dict[str, Any], *, label: str) -> None:
    """Reject per-species overrides that would break the outer mixture closure."""
    invalid = sorted(_FORBIDDEN_SPECIES_KEYS.intersection(overrides))
    if invalid:
        raise ValueError(f"{label} overrides cannot set reserved keys: {', '.join(invalid)}")


def _mix_fractions(counts: list[float]) -> np.ndarray:
    """Convert stoichiometric counts into atomic fractions."""
    counts_arr = np.asarray(counts, dtype=float)
    if counts_arr.ndim != 1 or counts_arr.size < 2:
        raise ValueError("A mixture requires at least two species counts.")
    if np.any(counts_arr <= 0.0):
        raise ValueError("Stoichiometric counts must be positive.")
    return counts_arr / float(np.sum(counts_arr))


def _ws_radius_from_volume(volume_bohr3: float) -> float:
    """Return the ion-sphere radius associated with one per-ion volume."""
    volume_bohr3 = float(volume_bohr3)
    if volume_bohr3 <= 0.0:
        raise ValueError("Per-ion volume must be positive.")
    return float((3.0 * volume_bohr3 / (4.0 * np.pi)) ** (1.0 / 3.0))


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Map unconstrained logits to positive weights with unit sum."""
    logits_arr = np.asarray(logits, dtype=float)
    logits_shift = logits_arr - float(np.max(logits_arr))
    weights = np.exp(logits_shift)
    return weights / float(np.sum(weights))


def _weights_to_theta(weights: np.ndarray) -> np.ndarray:
    """
    Encode positive volume weights in the `(N-1)` softmax coordinates.

    Notes
    -----
    The last species is used as the reference, so only the relative weights
    enter the nonlinear solve.
    """
    weights_arr = np.asarray(weights, dtype=float)
    if weights_arr.ndim != 1 or weights_arr.size < 2:
        raise ValueError("weights must contain at least two entries.")
    if np.any(weights_arr <= 0.0):
        raise ValueError("weights must be strictly positive.")
    weights_arr = weights_arr / float(np.sum(weights_arr))
    ref = float(weights_arr[-1])
    return np.log(weights_arr[:-1] / ref)


def _theta_to_weights(theta: np.ndarray) -> np.ndarray:
    """
    Recover positive volume weights from the `(N-1)` solve coordinates.

    Parameters
    ----------
    theta
        Unconstrained parameter vector of length `N-1`.

    Returns
    -------
    np.ndarray
        Positive weights whose sum is unity.
    """
    theta_arr = np.asarray(theta, dtype=float)
    logits = np.concatenate([theta_arr, np.zeros(1, dtype=float)])
    return _stable_softmax(logits)


def _symbol_key(spec: int | str) -> str:
    """Resolve one user species identifier to the canonical element symbol."""
    return str(element_info(spec).symbol)


def _normalize_positive_weights(weights: np.ndarray) -> np.ndarray:
    """Project one positive weight vector back to a normalized simplex point."""
    weights_arr = np.asarray(weights, dtype=float)
    weights_arr = np.maximum(weights_arr, 1.0e-12)
    return weights_arr / float(np.sum(weights_arr))


def _volumes_to_weights(
    volumes_bohr3: np.ndarray,
    *,
    fractions: np.ndarray,
    vbar_bohr3: float,
) -> np.ndarray:
    """
    Convert per-species ion volumes back to normalized mixture weights.

    Parameters
    ----------
    volumes_bohr3
        Per-species ion volumes in Bohr^3.
    fractions
        Number fractions `x_i`.
    vbar_bohr3
        Mixture-averaged ion volume `Vbar`.

    Returns
    -------
    np.ndarray
        Positive weights `w_i` satisfying `sum_i w_i = 1`.

    Notes
    -----
    The Starrett mixture volume closure is

      V_i = w_i * Vbar / x_i

    so the inverse map is `w_i = x_i V_i / Vbar`. A normalization is kept as a
    numerical guard against tiny interpolation error in the surrogate solve.
    """
    volumes_arr = np.asarray(volumes_bohr3, dtype=float)
    fractions_arr = np.asarray(fractions, dtype=float)
    weights = fractions_arr * volumes_arr / max(float(vbar_bohr3), 1.0e-12)
    return _normalize_positive_weights(weights)


def _resample_full_potential_init(
    result: dict[str, Any] | None,
    *,
    cfg_species: FullExternalConfig,
    r_ws_bohr: float,
) -> np.ndarray | None:
    """
    Resample one cached full-SCF potential onto the target AA grid.

    Parameters
    ----------
    result
        Previously converged AA result containing `r` and `v_full`.
    cfg_species
        Target species configuration for the next solve.
    r_ws_bohr
        Target ion-sphere radius in Bohr.

    Returns
    -------
    np.ndarray or None
        Initial potential on the target sqrt grid, or None when no valid
        cached potential is available.
    """
    if result is None:
        return None
    v_key = "v_full" if "v_full" in result else ("v_scf" if "v_scf" in result else None)
    if "r" not in result or v_key is None:
        return None

    r_src = np.asarray(result["r"], dtype=float)
    v_src = np.asarray(result[v_key], dtype=float)
    if r_src.ndim != 1 or v_src.shape != r_src.shape or r_src.size < 2:
        return None

    geometry = _resolve_outer_geometry(cfg_species, r_ws=float(r_ws_bohr))
    r_target = create_sqrt_grid(rmax=float(geometry["rmax"]), N=int(cfg_species.n_points)).r
    return np.asarray(
        np.interp(r_target, r_src, v_src, left=float(v_src[0]), right=float(v_src[-1])),
        dtype=float,
    )


def _copy_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    """
    Copy one AA result payload for safe reuse from in-memory caches.

    Notes
    -----
    The mixture outer loop may reuse one previously solved species result at
    an identical ion-sphere radius. A shallow dict copy is not enough because
    many entries are mutable `ndarray` objects.
    """
    copied: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            copied[key] = np.asarray(value, dtype=value.dtype).copy()
        else:
            copied[key] = value
    return copied


def _project_monotone_decreasing(values: np.ndarray) -> np.ndarray:
    """
    Project one sampled sequence onto a monotone-decreasing trend.

    Parameters
    ----------
    values
        Sampled scalar data ordered on an increasing volume grid.

    Returns
    -------
    np.ndarray
        Monotone-decreasing least-squares fit of the input sequence.

    Notes
    -----
    The mixture AA closure expects `mu(V)` to decrease with growing ion-sphere
    volume. Small SCF or branch-switch noise can violate that numerically. A
    lightweight pool-adjacent-violators projection keeps the surrogate tables
    consistent with the physical trend before we invert them.
    """
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if vals.size == 0:
        return vals.copy()

    means: list[float] = []
    weights: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    for idx, val in enumerate(vals):
        means.append(float(val))
        weights.append(1)
        starts.append(idx)
        ends.append(idx)
        while len(means) >= 2 and means[-2] < means[-1]:
            w_sum = weights[-2] + weights[-1]
            mean_sum = (weights[-2] * means[-2] + weights[-1] * means[-1]) / float(w_sum)
            means[-2] = float(mean_sum)
            weights[-2] = int(w_sum)
            ends[-2] = ends[-1]
            means.pop()
            weights.pop()
            starts.pop()
            ends.pop()

    out = np.zeros_like(vals, dtype=float)
    for mean_val, start_idx, end_idx in zip(means, starts, ends):
        out[start_idx : end_idx + 1] = float(mean_val)
    return out


def _interp_volume_from_mu(samples: list[tuple[float, float] | tuple[float, float, float]], mu_target: float) -> float:
    """
    Interpolate one monotone `V(mu)` map from tabulated `mu(V)` samples.

    Parameters
    ----------
    samples
        List of `(volume_bohr3, mu_ha)` samples for one species.
    mu_target
        Common chemical-potential target in Hartree.

    Returns
    -------
    float
        Interpolated per-ion volume in Bohr^3.

    Notes
    -----
    For the AA cases studied so far `mu(V)` is monotone decreasing. We invert
    that relation with a piecewise-linear interpolation in `mu`, after sorting
    and removing near-duplicate `mu` samples.
    """
    if len(samples) < 2:
        raise ValueError("Need at least two samples to interpolate V(mu).")

    arr = np.asarray(samples, dtype=float)
    order_vol = np.argsort(arr[:, 0])
    vol_by_volume = arr[order_vol, 0]
    mu_by_volume = arr[order_vol, 1]

    vol_unique: list[float] = []
    mu_unique_by_volume: list[float] = []
    for vol_val, mu_val in zip(vol_by_volume, mu_by_volume):
        if not vol_unique or abs(float(vol_val) - float(vol_unique[-1])) > 1.0e-10:
            vol_unique.append(float(vol_val))
            mu_unique_by_volume.append(float(mu_val))
        else:
            mu_unique_by_volume[-1] = 0.5 * (float(mu_unique_by_volume[-1]) + float(mu_val))

    # Enforce the physically expected monotone decrease of mu(V) with growing
    # ion volume. This filters small branch-switch / SCF noise before the
    # surrogate inversion uses the samples.
    mu_fit = _project_monotone_decreasing(np.asarray(mu_unique_by_volume, dtype=float))
    order_mu = np.argsort(mu_fit)
    mu_sorted = mu_fit[order_mu]
    vol_sorted = np.asarray(vol_unique, dtype=float)[order_mu]

    mu_unique: list[float] = []
    vol_inv_unique: list[float] = []
    for mu_val, vol_val in zip(mu_sorted, vol_sorted):
        if not mu_unique or abs(float(mu_val) - float(mu_unique[-1])) > 1.0e-10:
            mu_unique.append(float(mu_val))
            vol_inv_unique.append(float(vol_val))
        else:
            vol_inv_unique[-1] = 0.5 * (float(vol_inv_unique[-1]) + float(vol_val))

    if len(mu_unique) < 2:
        raise ValueError("Degenerate mu samples do not define a usable V(mu) map.")
    return float(np.interp(float(mu_target), np.asarray(mu_unique), np.asarray(vol_inv_unique)))


def _surrogate_common_mu(
    species_samples: list[list[tuple[float, float] | tuple[float, float, float]]],
    *,
    fractions: np.ndarray,
    vbar_bohr3: float,
    xtol: float,
    rtol: float,
) -> tuple[float, np.ndarray]:
    """
    Solve the scalar common-`mu_e` closure from per-species `mu(V)` tables.

    Parameters
    ----------
    species_samples
        Per-species lists of `(volume_bohr3, mu_ha)` samples.
    fractions
        Number fractions `x_i`.
    vbar_bohr3
        Mixture-averaged ion volume `Vbar`.
    xtol, rtol
        Scalar Brent tolerances for the surrogate `mu` root.

    Returns
    -------
    mu_common_ha, volumes_bohr3
        Surrogate common chemical potential and the corresponding per-species
        ion volumes.

    Notes
    -----
    Starrett et al. (2014) note that mixture closure can be accelerated by
    tabulating `mu_e^i(V_i, T)`. This helper uses exactly that idea: invert the
    current piecewise-linear `V_i(mu)` tables and solve the remaining scalar
    equation

      sum_i x_i V_i(mu) = Vbar
    """
    mu_lows = []
    mu_highs = []
    for samples in species_samples:
        if len(samples) < 2:
            raise RuntimeError("Insufficient tabulated points for the common-mu surrogate.")
        mu_vals = np.asarray([row[1] for row in samples], dtype=float)
        mu_lows.append(float(np.min(mu_vals)))
        mu_highs.append(float(np.max(mu_vals)))

    mu_lo = max(mu_lows)
    mu_hi = min(mu_highs)
    if not np.isfinite(mu_lo) or not np.isfinite(mu_hi) or mu_lo >= mu_hi:
        raise RuntimeError("Per-species mu(V) tables do not yet overlap in mu.")

    def _volume_closure(mu_target: float) -> float:
        volumes_here = np.asarray(
            [_interp_volume_from_mu(samples, float(mu_target)) for samples in species_samples],
            dtype=float,
        )
        return float(np.sum(fractions * volumes_here) - float(vbar_bohr3))

    f_lo = _volume_closure(mu_lo)
    if abs(f_lo) <= max(float(xtol), 1.0e-12):
        mu_star = float(mu_lo)
    else:
        f_hi = _volume_closure(mu_hi)
        if abs(f_hi) <= max(float(xtol), 1.0e-12):
            mu_star = float(mu_hi)
        else:
            if np.sign(f_lo) == np.sign(f_hi):
                raise RuntimeError(
                    "The current surrogate tables do not bracket the mixture volume closure."
                )
            mu_star = float(
                brentq(
                    _volume_closure,
                    float(mu_lo),
                    float(mu_hi),
                    xtol=float(xtol),
                    rtol=float(rtol),
                    maxiter=64,
                )
            )
    volumes_star = np.asarray(
        [_interp_volume_from_mu(samples, float(mu_star)) for samples in species_samples],
        dtype=float,
    )
    return float(mu_star), volumes_star


def _local_theta_refine(
    evaluator: "_MixtureEvaluator",
    *,
    theta_init: np.ndarray,
    root_tol: float,
    max_nfev: int,
    half_width: float = 0.20,
) -> np.ndarray | None:
    """
    Apply one small local direct refinement around an existing mixture point.

    Parameters
    ----------
    evaluator
        Mixture residual evaluator.
    theta_init
        Starting softmax coordinates from the current best candidate.
    root_tol
        Local least-squares termination tolerance.
    max_nfev
        Maximum additional evaluator calls.
    half_width
        Symmetric trust-region half-width in each theta coordinate.

    Returns
    -------
    np.ndarray or None
        Refined theta vector, or None if the local solve fails.

    Notes
    -----
    The tabulated `mu(V)` surrogate is efficient, but on hard mixture stages it
    can stop with a candidate that is close enough for a local direct solve to
    finish the job. The refinement is intentionally bounded to keep the search
    on the same local branch and to avoid expensive global wandering.
    """
    theta0 = np.asarray(theta_init, dtype=float)
    lower = theta0 - float(half_width)
    upper = theta0 + float(half_width)
    tol = max(float(root_tol), 1.0e-8)
    try:
        opt = least_squares(
            evaluator.residual,
            theta0,
            bounds=(lower, upper),
            method="trf",
            xtol=tol,
            ftol=tol,
            gtol=tol,
            max_nfev=max(int(max_nfev), 4),
        )
    except Exception:
        return None
    if not np.all(np.isfinite(opt.x)):
        return None
    return np.asarray(opt.x, dtype=float)


def _scalar_theta_bracket_refine(
    evaluator: "_MixtureEvaluator",
    *,
    theta_center: np.ndarray,
    root_tol: float,
    max_nfev: int,
    half_widths: tuple[float, ...] = (0.01, 0.02, 0.04, 0.08, 0.16, 0.28),
) -> np.ndarray | None:
    """
    Refine one binary-mixture theta by bracketing a sign change in the direct residual.

    Notes
    -----
    For binary mixtures the common-`mu` closure is one-dimensional. On hard
    cases the direct residual can still show a local sign change near the
    physically correct branch even when a local least-squares polish stalls.
    A small direct `brentq` pass on the true AA residual is more robust in
    that situation than continuing with a broad surrogate reseed.
    """
    theta0 = np.asarray(theta_center, dtype=float).reshape(-1)
    if theta0.size != 1:
        return None

    xtol = max(float(root_tol), 1.0e-8)
    center = float(theta0[0])
    f_center = float(np.asarray(evaluator.residual(theta0), dtype=float).reshape(-1)[0])
    if not np.isfinite(f_center):
        return None
    if abs(f_center) <= xtol:
        return theta0.copy()

    def _f(theta_val: float) -> float:
        return float(
            np.asarray(evaluator.residual(np.asarray([float(theta_val)], dtype=float)), dtype=float).reshape(-1)[0]
        )

    for half_width in tuple(float(val) for val in half_widths):
        left = float(center - half_width)
        right = float(center + half_width)
        f_left = _f(left)
        if not np.isfinite(f_left):
            continue
        if abs(f_left) <= xtol:
            return np.asarray([left], dtype=float)
        if np.sign(f_left) != np.sign(f_center):
            try:
                theta_star = float(
                    brentq(
                        _f,
                        left,
                        center,
                        xtol=xtol,
                        rtol=xtol,
                        maxiter=max(1, int(max_nfev)),
                    )
                )
            except Exception:
                theta_star = np.nan
            if np.isfinite(theta_star):
                return np.asarray([theta_star], dtype=float)

        f_right = _f(right)
        if not np.isfinite(f_right):
            continue
        if abs(f_right) <= xtol:
            return np.asarray([right], dtype=float)
        if np.sign(f_center) != np.sign(f_right):
            try:
                theta_star = float(
                    brentq(
                        _f,
                        center,
                        right,
                        xtol=xtol,
                        rtol=xtol,
                        maxiter=max(1, int(max_nfev)),
                    )
                )
            except Exception:
                theta_star = np.nan
            if np.isfinite(theta_star):
                return np.asarray([theta_star], dtype=float)

        if np.sign(f_left) != np.sign(f_right):
            try:
                theta_star = float(
                    brentq(
                        _f,
                        left,
                        right,
                        xtol=xtol,
                        rtol=xtol,
                        maxiter=max(1, int(max_nfev)),
                    )
                )
            except Exception:
                theta_star = np.nan
            if np.isfinite(theta_star):
                return np.asarray([theta_star], dtype=float)
    return None


def _scalar_theta_scan_bracket_refine(
    evaluator: "_MixtureEvaluator",
    *,
    theta_min: float,
    theta_max: float,
    n_points: int,
    root_tol: float,
    max_nfev: int,
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    """
    Scan one binary-mixture theta interval and refine the best sign-change bracket.

    Notes
    -----
    This is the robust fallback for hard binary common-mu points. It evaluates
    the true AA residual on one small 1D grid, identifies adjacent sign-change
    intervals, and then applies `brentq` on the direct residual inside the best
    bracket. No surrogate continuity is assumed beyond the local bracket.
    """
    n_grid = max(int(n_points), 3)
    theta_grid = np.linspace(float(theta_min), float(theta_max), n_grid, dtype=float)
    f_grid = np.asarray(
        [
            float(
                np.asarray(evaluator.residual(np.asarray([float(theta)], dtype=float)), dtype=float).reshape(-1)[0]
            )
            for theta in theta_grid
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(f_grid)):
        return None, None

    xtol = max(float(root_tol), 1.0e-8)
    idx_exact = np.flatnonzero(np.abs(f_grid) <= xtol)
    if idx_exact.size > 0:
        return np.asarray([float(theta_grid[int(idx_exact[0])])], dtype=float), (
            float(theta_grid[int(idx_exact[0])]),
            float(theta_grid[int(idx_exact[0])]),
        )

    sign_change_idx = np.flatnonzero(np.signbit(f_grid[:-1]) != np.signbit(f_grid[1:]))
    if sign_change_idx.size == 0:
        return None, None

    idx_best = int(
        min(
            sign_change_idx,
            key=lambda idx: max(abs(float(f_grid[int(idx)])), abs(float(f_grid[int(idx) + 1]))),
        )
    )
    left = float(theta_grid[idx_best])
    right = float(theta_grid[idx_best + 1])

    def _f(theta_val: float) -> float:
        return float(
            np.asarray(evaluator.residual(np.asarray([float(theta_val)], dtype=float)), dtype=float).reshape(-1)[0]
        )

    try:
        theta_star = float(
            brentq(
                _f,
                left,
                right,
                xtol=xtol,
                rtol=xtol,
                maxiter=max(1, int(max_nfev)),
            )
        )
    except Exception:
        return None, (left, right)
    return np.asarray([theta_star], dtype=float), (left, right)


def _solve_species_from_config(cfg_species: FullExternalConfig) -> dict[str, Any]:
    """
    Run one species AA solve from a fully prepared `FullExternalConfig`.

    Notes
    -----
    This helper is defined at module scope so it can be used by a
    `ProcessPoolExecutor` during multicomponent mixture evaluations.
    """
    run_mode = str(cfg_species.run_mode).strip().lower()
    if run_mode == "full" or not bool(cfg_species.ext_scf_enabled):
        return solve_full_only(cfg_species)
    return solve_full_then_external(cfg_species)


def _initial_weight_guesses(
    *,
    fractions: np.ndarray,
    elements: list[Any],
    explicit: np.ndarray | None,
    local_only: bool = False,
) -> list[np.ndarray]:
    """
    Build a small set of positive initial guesses for the outer mixture solve.

    Notes
    -----
    The common-`mu_e` closure can be stiff for very asymmetric mixtures. A few
    physically motivated seeds are cheap compared with one failed outer solve:

    - `x_i`
    - equal weights
    - `x_i Z_i`
    - `x_i Z_i^2`
    - `x_i sqrt(Z_i)`
    - `x_i / Z_i`

    The last three bias the initial per-ion volumes away from the equal-volume
    point, which is often a poor starting guess when hydrogen and high-`Z`
    species coexist.

    When `local_only=True` and an explicit seed is available, only a small
    neighborhood around that seed is explored. This is intended for polish
    stages, where wide global reseeding is more likely to jump branch than to
    help continuation.
    """
    z_arr = np.asarray([float(elem.z) for elem in elements], dtype=float)
    guesses_raw = []
    if explicit is not None:
        explicit_arr = _normalize_positive_weights(np.asarray(explicit, dtype=float))
        guesses_raw.append(explicit_arr)
        theta_explicit = _weights_to_theta(explicit_arr)
        for idx in range(theta_explicit.size):
            delta_list = (-0.05, 0.05, -0.10, 0.10) if local_only else (-0.25, 0.25)
            for delta in delta_list:
                theta_try = theta_explicit.copy()
                theta_try[idx] = theta_try[idx] + float(delta)
                guesses_raw.append(_theta_to_weights(theta_try))
        if local_only:
            out_local: list[np.ndarray] = []
            seen_local: set[tuple[float, ...]] = set()
            for guess in guesses_raw:
                guess_norm = _normalize_positive_weights(guess)
                key = tuple(np.round(guess_norm, 10))
                if key in seen_local:
                    continue
                seen_local.add(key)
                out_local.append(guess_norm)
            return out_local
    guesses_raw.extend(
        [
            np.asarray(fractions, dtype=float) * z_arr,
            np.asarray(fractions, dtype=float) * z_arr * z_arr,
            np.asarray(fractions, dtype=float) * np.sqrt(z_arr),
            np.asarray(fractions, dtype=float),
            np.ones_like(fractions, dtype=float),
            np.asarray(fractions, dtype=float) / np.maximum(z_arr, 1.0),
        ]
    )

    out: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for guess in guesses_raw:
        guess_norm = _normalize_positive_weights(guess)
        key = tuple(np.round(guess_norm, 10))
        if key in seen:
            continue
        seen.add(key)
        out.append(guess_norm)
    return out


def _record_species_samples(
    species_samples: list[list[tuple[float, float, float]]],
    record: dict[str, Any],
) -> None:
    """
    Append one evaluated mixture point to the per-species `mu(V)` tables.

    Parameters
    ----------
    species_samples
        Mutable per-species sample lists storing
        `(volume_bohr3, mu_ha, record_quality)`.
    record
        One `_MixtureEvaluator.evaluate(...)` payload.
    """
    volumes = np.asarray(record["volumes_bohr3"], dtype=float)
    mu_vals = np.asarray(record["mu_ha"], dtype=float)
    residual = np.asarray(record["mu_residual_ha"], dtype=float)
    quality = float(np.max(np.abs(residual))) if residual.size > 0 else 0.0
    for idx in range(len(species_samples)):
        volume_here = float(volumes[idx])
        mu_here = float(mu_vals[idx])
        duplicate = False
        replace_at: int | None = None
        for jdx, (vol_old, mu_old, qual_old) in enumerate(species_samples[idx]):
            vol_scale = max(abs(float(vol_old)), abs(volume_here), 1.0)
            if abs(volume_here - float(vol_old)) / vol_scale <= 3.0e-3:
                duplicate = True
                if quality < float(qual_old):
                    replace_at = jdx
                break
            if abs(volume_here - float(vol_old)) <= 1.0e-10 and abs(mu_here - float(mu_old)) <= 1.0e-10:
                duplicate = True
                break
        if replace_at is not None:
            species_samples[idx][replace_at] = (volume_here, mu_here, quality)
        elif not duplicate:
            species_samples[idx].append((volume_here, mu_here, quality))


@dataclass
class MixtureConfig:
    """
    Common-`mu_e` AA solve for an arbitrary number of species.

    Parameters
    ----------
    species
        Sequence of distinct element symbols or atomic numbers.
    counts
        Stoichiometric counts. These are converted to number fractions
        internally via `x_i = count_i / sum(counts)`.
    temperature_ev
        Common electron temperature in eV.
    rho_g_cc
        Total mass density in g/cc.
    aa_overrides
        Per-species AA settings shared by all components.
    species_overrides
        Optional mapping from element symbol to extra AA overrides for that
        component only.

    Returns
    -------
    MixtureConfig
        Validated configuration object consumed by the mixture solvers.

    Notes
    -----
    The outer root solve does not change the underlying AA model. It only
    varies the positive volume weights `w_i` such that

      V_i = w_i * Vbar / x_i

    with `sum_i w_i = 1`, while enforcing `mu_e^i = mu_e` for all species.
    """

    species: list[int | str] | tuple[int | str, ...]
    counts: list[float] | tuple[float, ...]
    temperature_ev: float
    rho_g_cc: float

    aa_overrides: dict[str, Any] = field(default_factory=dict)
    root_aa_overrides: dict[str, Any] = field(default_factory=dict)
    species_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    mu_e_tol: float = 1e-4
    root_tol: float = 1e-4
    root_maxfev: int = 20
    cache_round_digits: int = 12
    show_progress: bool = False
    show_mu_progress: bool = False
    verbose: bool = False
    final_run_mode: str = "full+ext"
    volume_weights_init: list[float] | tuple[float, ...] | None = None
    species_parallel_jobs: int = 1
    species_parallel_backend: str = "process"

    save_data: bool = False
    save_output_dir: str | Path = "outputs"
    save_suffix: str = ""
    save_common_linear_grid: bool = True
    save_linear_n_points: int = 4096

    def __post_init__(self) -> None:
        species_list = list(self.species)
        counts_list = [float(v) for v in self.counts]
        if len(species_list) != len(counts_list):
            raise ValueError("species and counts must have the same length.")
        if len(species_list) < 2:
            raise ValueError("A mixture requires at least two species.")
        if float(self.rho_g_cc) <= 0.0:
            raise ValueError("rho_g_cc must be positive.")
        if float(self.temperature_ev) < 0.0:
            raise ValueError("temperature_ev must be non-negative.")
        if float(self.mu_e_tol) <= 0.0:
            raise ValueError("mu_e_tol must be positive.")
        if float(self.root_tol) <= 0.0:
            raise ValueError("root_tol must be positive.")
        if int(self.root_maxfev) < 4:
            raise ValueError("root_maxfev must be at least 4.")
        if int(self.species_parallel_jobs) < 1:
            raise ValueError("species_parallel_jobs must be at least 1.")
        if int(self.save_linear_n_points) < 16:
            raise ValueError("save_linear_n_points must be at least 16.")
        if str(self.final_run_mode).strip().lower() not in ("full", "full+ext", "full_ext"):
            raise ValueError("final_run_mode must be 'full' or 'full+ext'.")
        if str(self.species_parallel_backend).strip().lower() not in ("process", "thread"):
            raise ValueError("species_parallel_backend must be 'process' or 'thread'.")

        resolved_symbols = [_symbol_key(spec) for spec in species_list]
        if len(set(resolved_symbols)) != len(resolved_symbols):
            raise ValueError("Mixture species must be distinct.")

        _mix_fractions(counts_list)
        _validate_species_overrides(self.aa_overrides, label="Common AA")
        if len(self.root_aa_overrides) > 0:
            raise ValueError(
                "MixtureConfig.root_aa_overrides is no longer supported. "
                "The outer mixture closure now keeps one fixed AA setup; set "
                "AA controls such as n_points and cont_e_max directly in "
                "aa_overrides."
            )
        for symbol, overrides in self.species_overrides.items():
            if _symbol_key(symbol) not in resolved_symbols:
                raise ValueError(
                    f"species_overrides contains unknown species {symbol!r}; "
                    f"expected one of {resolved_symbols}."
                )
            _validate_species_overrides(dict(overrides), label=f"Species {symbol}")

        if self.volume_weights_init is not None:
            init = np.asarray(self.volume_weights_init, dtype=float)
            if init.shape != (len(species_list),):
                raise ValueError("volume_weights_init must have one entry per species.")
            if np.any(init <= 0.0):
                raise ValueError("volume_weights_init must be strictly positive.")


class _MixtureEvaluator:
    """
    Cached evaluator for the multicomponent common-`mu_e` residual.

    Notes
    -----
    Each function evaluation performs one AA solve per species at the current
    set of ion-sphere volumes. Because those solves are the expensive part,
    rounded-parameter caching is used to avoid accidental repeats.
    """

    def __init__(
        self,
        cfg: MixtureConfig,
        *,
        species_init_cache: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.elements = [element_info(spec) for spec in cfg.species]
        self.symbols = [str(elem.symbol) for elem in self.elements]
        self.counts = np.asarray(cfg.counts, dtype=float)
        self.x = _mix_fractions(self.counts.tolist())
        avg_atomic_mass = float(
            np.sum(self.x * np.asarray([float(elem.atomic_mass) for elem in self.elements], dtype=float))
        )
        self.n_mix = float(ion_density_bohr3(float(cfg.rho_g_cc), avg_atomic_mass))
        self.vbar_bohr3 = float(1.0 / self.n_mix)
        self.cache: dict[tuple[float, ...], dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []
        self._eval_counter: int = 0
        self._species_init_cache: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in self.symbols}
        self._species_result_cache: dict[tuple[str, float], dict[str, Any]] = {}
        self._species_result_cache_hits: int = 0
        self._species_result_cache_misses: int = 0
        if species_init_cache is not None:
            for symbol, entries in species_init_cache.items():
                key = str(symbol)
                if key not in self._species_init_cache:
                    continue
                self._species_init_cache[key] = [
                    {
                        "r_ws_bohr": float(one["r_ws_bohr"]),
                        "result": {
                            "r": np.asarray(one["result"]["r"], dtype=float).copy(),
                            "v_full": np.asarray(one["result"]["v_full"], dtype=float).copy(),
                        },
                    }
                    for one in entries
                ]
        self._executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None
        jobs = int(cfg.species_parallel_jobs)
        backend = str(cfg.species_parallel_backend).strip().lower()
        if jobs > 1:
            if backend == "thread":
                self._executor = ThreadPoolExecutor(max_workers=jobs)
            else:
                self._executor = ProcessPoolExecutor(max_workers=jobs)

    def _species_overrides(self, symbol: str) -> dict[str, Any]:
        return dict(self.cfg.species_overrides.get(symbol, {}))

    def _species_result_cache_key(self, *, symbol: str, r_ws_bohr: float) -> tuple[str, float]:
        """Build one rounded cache key for a species full-only solve."""
        return (
            str(symbol),
            float(np.round(float(r_ws_bohr), int(self.cfg.cache_round_digits))),
        )

    def _cached_species_result(
        self,
        *,
        symbol: str,
        r_ws_bohr: float,
    ) -> dict[str, Any] | None:
        """Return one cached species result at an identical rounded WS radius."""
        cache_key = self._species_result_cache_key(symbol=str(symbol), r_ws_bohr=float(r_ws_bohr))
        cached = self._species_result_cache.get(cache_key)
        if cached is None:
            return None
        self._species_result_cache_hits += 1
        return _copy_result_payload(cached)

    def _store_species_result(
        self,
        *,
        symbol: str,
        r_ws_bohr: float,
        result: dict[str, Any],
    ) -> None:
        """Store one solved species result in the exact-radius cache."""
        cache_key = self._species_result_cache_key(symbol=str(symbol), r_ws_bohr=float(r_ws_bohr))
        self._species_result_cache[cache_key] = _copy_result_payload(result)

    def _cached_species_init(
        self,
        *,
        symbol: str,
        cfg_species: FullExternalConfig,
        r_ws_bohr: float,
    ) -> np.ndarray | None:
        """Return the nearest cached full potential for one species, if any."""
        entries = self._species_init_cache.get(str(symbol), [])
        if not entries:
            return None
        entry = min(entries, key=lambda one: abs(np.log(max(one["r_ws_bohr"], 1.0e-12) / max(r_ws_bohr, 1.0e-12))))
        return _resample_full_potential_init(
            dict(entry["result"]),
            cfg_species=cfg_species,
            r_ws_bohr=float(r_ws_bohr),
        )

    def _update_species_init_cache(
        self,
        *,
        symbol: str,
        r_ws_bohr: float,
        result: dict[str, Any],
    ) -> None:
        """Store one converged full result for later nearby warm-start use."""
        entries = self._species_init_cache.setdefault(str(symbol), [])
        entries.append(
            {
                "r_ws_bohr": float(r_ws_bohr),
                "result": {
                    "r": np.asarray(result.get("r", []), dtype=float).copy(),
                    "v_full": np.asarray(result.get("v_full", []), dtype=float).copy(),
                },
            }
        )
        if len(entries) > 8:
            del entries[:-8]

    def close(self) -> None:
        """Release any optional species-level parallel executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _species_config(
        self,
        *,
        element_key: int | str,
        r_ws_bohr: float,
        n_i_bohr3: float,
        extra_overrides: dict[str, Any],
        run_mode: str,
        ext_enabled: bool,
    ) -> FullExternalConfig:
        kwargs: dict[str, Any] = {
            "element": element_key,
            "temperature_ev": float(self.cfg.temperature_ev),
            "rho_g_cc": float(self.cfg.rho_g_cc),
            "run_mode": str(run_mode),
            "ext_scf_enabled": bool(ext_enabled),
            "r_ws_override_bohr": float(r_ws_bohr),
            "n_i_override_bohr3": float(n_i_bohr3),
            "show_scf_progress": bool(self.cfg.show_progress),
            "verbose": bool(self.cfg.verbose),
        }
        kwargs.update(self.cfg.aa_overrides)
        kwargs.update(extra_overrides)
        cfg_species = FullExternalConfig(**kwargs)
        v_full_init = self._cached_species_init(
            symbol=str(element_info(element_key).symbol),
            cfg_species=cfg_species,
            r_ws_bohr=float(r_ws_bohr),
        )
        if v_full_init is not None:
            cfg_species.v_full_init = np.asarray(v_full_init, dtype=float)
        return cfg_species

    def evaluate(self, theta: np.ndarray) -> dict[str, Any]:
        theta_arr = np.asarray(theta, dtype=float)
        cache_key = tuple(np.round(theta_arr, int(self.cfg.cache_round_digits)))
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        weights = _theta_to_weights(theta_arr)
        volumes = weights * self.vbar_bohr3 / self.x
        n_i_species = 1.0 / volumes
        r_ws_species = np.asarray([_ws_radius_from_volume(v) for v in volumes], dtype=float)

        species_cfgs: list[FullExternalConfig] = []
        for idx, elem in enumerate(self.elements):
            cfg_species = self._species_config(
                element_key=elem.z,
                r_ws_bohr=float(r_ws_species[idx]),
                n_i_bohr3=float(n_i_species[idx]),
                extra_overrides=self._species_overrides(str(elem.symbol)),
                run_mode="full",
                ext_enabled=False,
            )
            species_cfgs.append(cfg_species)

        full_results: list[dict[str, Any] | None] = [None] * len(self.elements)
        miss_indices: list[int] = []
        miss_cfgs: list[FullExternalConfig] = []
        for idx, cfg_species in enumerate(species_cfgs):
            cached_species = self._cached_species_result(
                symbol=self.symbols[idx],
                r_ws_bohr=float(r_ws_species[idx]),
            )
            if cached_species is not None:
                full_results[idx] = cached_species
                continue
            miss_indices.append(idx)
            miss_cfgs.append(cfg_species)

        miss_results: list[dict[str, Any]]
        if len(miss_cfgs) == 0:
            miss_results = []
        elif self._executor is None:
            miss_results = [_solve_species_from_config(cfg_species) for cfg_species in miss_cfgs]
        else:
            miss_results = list(self._executor.map(_solve_species_from_config, miss_cfgs))

        self._species_result_cache_misses += len(miss_results)
        for idx, result_species in zip(miss_indices, miss_results, strict=True):
            full_results[idx] = result_species
            self._store_species_result(
                symbol=self.symbols[idx],
                r_ws_bohr=float(r_ws_species[idx]),
                result=dict(result_species),
            )
            self._update_species_init_cache(
                symbol=self.symbols[idx],
                r_ws_bohr=float(r_ws_species[idx]),
                result=dict(result_species),
            )

        full_results = [dict(result_species) for result_species in full_results if result_species is not None]

        mu_values = np.zeros(len(self.elements), dtype=float)
        for idx, result_species in enumerate(full_results):
            mu_values[idx] = float(result_species["mu"])

        residual = mu_values[:-1] - mu_values[-1]
        record = {
            "theta": theta_arr.copy(),
            "weights": weights.copy(),
            "volumes_bohr3": volumes.copy(),
            "n_i_bohr3": n_i_species.copy(),
            "r_ws_bohr": r_ws_species.copy(),
            "mu_ha": mu_values.copy(),
            "mu_residual_ha": residual.copy(),
            "results": full_results,
        }
        self.cache[cache_key] = record

        self._eval_counter += 1
        hist_row: dict[str, Any] = {
            "iter": int(self._eval_counter),
            "theta_norm": float(np.linalg.norm(theta_arr)),
            "mu_span_ha": float(np.max(mu_values) - np.min(mu_values)),
        }
        for idx, symbol in enumerate(self.symbols):
            result_species = full_results[idx]
            stage1_hist = list(result_species.get("stage1_history", []))
            stage2_hist = list(result_species.get("history", []))
            hist_row[f"weight_{symbol}"] = float(weights[idx])
            hist_row[f"volume_{symbol}_bohr3"] = float(volumes[idx])
            hist_row[f"r_ws_{symbol}_bohr"] = float(r_ws_species[idx])
            hist_row[f"mu_{symbol}_ha"] = float(mu_values[idx])
            hist_row[f"zbar_{symbol}"] = float(result_species.get("zbar", np.nan))
            hist_row[f"stage1_converged_{symbol}"] = bool(result_species.get("stage1_converged", False))
            hist_row[f"stage2_converged_{symbol}"] = bool(result_species.get("stage2_converged", False))
            hist_row[f"stage1_iters_{symbol}"] = int(result_species.get("stage1_iters", 0))
            hist_row[f"stage2_iters_{symbol}"] = int(result_species.get("stage2_iters", 0))
            hist_row[f"stage1_err_{symbol}"] = float(stage1_hist[-1].get("err", np.nan)) if stage1_hist else np.nan
            hist_row[f"stage2_err_{symbol}"] = float(stage2_hist[-1].get("err", np.nan)) if stage2_hist else np.nan
            if idx < len(self.symbols) - 1:
                hist_row[f"dmu_{symbol}_{self.symbols[-1]}_ha"] = float(residual[idx])
        self.history.append(hist_row)

        if self.cfg.show_mu_progress or self.cfg.verbose:
            theta_txt = "[" + ", ".join(f"{float(val):.6f}" for val in theta_arr) + "]"
            mu_txt = "  ".join(
                f"mu_{symbol}={float(mu_values[idx]):.6f} Ha" for idx, symbol in enumerate(self.symbols)
            )
            dmu_txt = "  ".join(
                f"dmu_{self.symbols[idx]}_{self.symbols[-1]}={float(residual[idx]):.3e} Ha"
                for idx in range(len(self.symbols) - 1)
            )
            rws_txt = "  ".join(
                f"Rws_{symbol}={float(r_ws_species[idx]):.6f} Bohr" for idx, symbol in enumerate(self.symbols)
            )
            print(
                "[mixture] "
                f"iter={int(self._eval_counter)}  "
                f"theta={theta_txt}  "
                f"theta_norm={float(np.linalg.norm(theta_arr)):.6f}  "
                f"mu_span={float(np.max(mu_values) - np.min(mu_values)):.3e} Ha  "
                f"{dmu_txt}  "
                f"{mu_txt}  "
                f"{rws_txt}"
            )
            for idx, symbol in enumerate(self.symbols):
                result_species = full_results[idx]
                stage1_hist = list(result_species.get("stage1_history", []))
                stage2_hist = list(result_species.get("history", []))
                stage1_err = float(stage1_hist[-1].get("err", np.nan)) if stage1_hist else np.nan
                stage2_err = float(stage2_hist[-1].get("err", np.nan)) if stage2_hist else np.nan
                print(
                    "[mixture] "
                    f"  {symbol}: "
                    f"stage1={bool(result_species.get('stage1_converged', False))} "
                    f"(iters={int(result_species.get('stage1_iters', 0))}, err={stage1_err:.3e})  "
                    f"stage2={bool(result_species.get('stage2_converged', False))} "
                    f"(iters={int(result_species.get('stage2_iters', 0))}, err={stage2_err:.3e})  "
                    f"Zbar={float(result_species.get('zbar', np.nan)):.6f}"
                )
        return record

    def residual(self, theta: np.ndarray) -> np.ndarray:
        """Return the common-chemical-potential mismatch vector."""
        return np.asarray(self.evaluate(theta)["mu_residual_ha"], dtype=float)


def _mixture_label_from_counts(symbols: list[str], counts: np.ndarray) -> str:
    """Build a compact mixture label such as `C1H2O1` for saved outputs."""
    tokens = []
    for symbol, count in zip(symbols, counts):
        if abs(float(count) - round(float(count))) < 1e-12:
            count_txt = str(int(round(float(count))))
        else:
            count_txt = f"{float(count):.6g}"
        tokens.append(f"{symbol}{count_txt}")
    return "".join(tokens)


def _assemble_species_payload(
    *,
    elements: list[Any],
    counts: np.ndarray,
    fractions: np.ndarray,
    volumes_bohr3: np.ndarray,
    r_ws_bohr: np.ndarray,
    mu_values: np.ndarray,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert aligned per-species arrays into the public mixture payload."""
    species_payload: list[dict[str, Any]] = []
    for idx, elem in enumerate(elements):
        species_payload.append(
            {
                "element": str(elem.symbol),
                "Z": int(elem.z),
                "atomic_mass": float(elem.atomic_mass),
                "count": float(counts[idx]),
                "x": float(fractions[idx]),
                "volume_bohr3": float(volumes_bohr3[idx]),
                "r_ws_bohr": float(r_ws_bohr[idx]),
                "mu_ha": float(mu_values[idx]),
                "result": results[idx],
            }
        )
    return species_payload


def solve_mixture_full_only(
    cfg: MixtureConfig,
    *,
    species_init_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """
    Solve the multicomponent common-`mu_e` AA problem.

    Parameters
    ----------
    cfg
        Mixture configuration describing the species list, stoichiometry, and
        AA settings.

    Returns
    -------
    dict
        One dictionary containing the converged common chemical potential, the
        resolved per-species ion-sphere volumes, and the final full-only AA
        result for each species.

    Notes
    -----
    The expensive AA work is still performed on the physical per-species
    ion-sphere volumes. Following the acceleration strategy discussed by
    Starrett et al. (2014), the outer closure is handled by tabulating
    `mu_e^i(V_i)` for each species, solving the remaining scalar common-`mu_e`
    volume constraint, and then refining that surrogate with real AA solves.
    """
    evaluator = _MixtureEvaluator(cfg, species_init_cache=species_init_cache)
    try:
        explicit_init = None
        if cfg.volume_weights_init is not None:
            explicit_init = _normalize_positive_weights(np.asarray(cfg.volume_weights_init, dtype=float))
        # An explicit warm-start should bias the outer closure locally even when
        # no species-potential cache is available. The AA potential cache and
        # the mixture-weight seed strategy are independent accelerators.
        prefer_local_seeds = explicit_init is not None

        seed_weights = _initial_weight_guesses(
            fractions=evaluator.x,
            elements=evaluator.elements,
            explicit=explicit_init,
            local_only=bool(prefer_local_seeds),
        )
        seed_keys_seen = {tuple(np.round(np.asarray(seed, dtype=float), 10)) for seed in seed_weights}
        n_species = len(evaluator.elements)
        if prefer_local_seeds:
            n_seed_target = len(seed_weights)
        else:
            n_seed_target = min(len(seed_weights), max(n_species + 1, 4))
        species_samples: list[list[tuple[float, float, float]]] = [[] for _ in range(n_species)]
        best_residual_max = np.inf
        best_final = None
        tol = float(cfg.mu_e_tol)
        surrogate_iters = 0
        seed_eval_count = 0
        root_method = "tabulated_mu_brentq"
        root_message = "Converged via tabulated mu(V) common-mu closure."
        bracket_found = False
        bracket_interval: tuple[float, float] | None = None
        reported_bracket_interval: tuple[float, float] | None = None
        is_binary = len(evaluator.elements) == 2
        primary_root_maxfev = min(int(cfg.root_maxfev), 10) if is_binary else int(cfg.root_maxfev)
        binary_brent_maxiter = 10
        observed_binary_points: list[tuple[float, float]] = []

        def _primary_method_budget_exhausted() -> bool:
            if is_binary:
                return int(len(evaluator.history)) >= 10
            return int(surrogate_iters) >= int(primary_root_maxfev)

        def _update_observed_binary_bracket() -> None:
            nonlocal bracket_found, bracket_interval, reported_bracket_interval
            if not is_binary or len(observed_binary_points) < 2:
                return
            points = sorted(observed_binary_points, key=lambda item: float(item[0]))
            bracket_candidates: list[tuple[float, float, float]] = []
            for (theta_l, dmu_l), (theta_r, dmu_r) in zip(points[:-1], points[1:]):
                if not (np.isfinite(dmu_l) and np.isfinite(dmu_r)):
                    continue
                if dmu_l == 0.0:
                    bracket_candidates.append((0.0, float(theta_l), float(theta_l)))
                    continue
                if dmu_r == 0.0:
                    bracket_candidates.append((0.0, float(theta_r), float(theta_r)))
                    continue
                if np.signbit(dmu_l) == np.signbit(dmu_r):
                    continue
                quality = max(abs(float(dmu_l)), abs(float(dmu_r)))
                bracket_candidates.append((float(quality), float(theta_l), float(theta_r)))
            if len(bracket_candidates) == 0:
                return
            _, theta_l_best, theta_r_best = min(bracket_candidates, key=lambda item: float(item[0]))
            bracket_found = True
            bracket_interval = (float(theta_l_best), float(theta_r_best))
            if (cfg.show_mu_progress or cfg.verbose) and reported_bracket_interval != bracket_interval:
                reported_bracket_interval = bracket_interval
                print(
                    "[mixture] "
                    f"observed binary sign-change bracket: theta in [{float(theta_l_best):.6f}, "
                    f"{float(theta_r_best):.6f}]"
                )

        def _refine_observed_binary_bracket() -> bool:
            nonlocal root_method, root_message
            if not (is_binary and bracket_found and bracket_interval is not None):
                return False

            theta_l = float(bracket_interval[0])
            theta_r = float(bracket_interval[1])
            remaining_budget = max(int(cfg.root_maxfev) - int(len(evaluator.history)), 0)
            if remaining_budget <= 0:
                root_method = "binary_observed_bracket_brentq_budget_exhausted"
                root_message = (
                    "The binary common-mu solve reached the configured total iteration budget before "
                    "the bracketed brent fallback could be completed."
                )
                return False

            def _f(theta_val: float) -> float:
                return float(
                    np.asarray(
                        evaluator.residual(np.asarray([float(theta_val)], dtype=float)),
                        dtype=float,
                    ).reshape(-1)[0]
                )

            try:
                theta_star = float(
                    brentq(
                        _f,
                        theta_l,
                        theta_r,
                        xtol=min(float(cfg.root_tol), 1.0e-6),
                        rtol=min(float(cfg.root_tol), 1.0e-6),
                        maxiter=min(int(binary_brent_maxiter), int(remaining_budget)),
                    )
                )
            except Exception:
                root_method = "binary_observed_bracket_brentq_failed"
                root_message = (
                    "The binary common-mu solve found one observed sign-change bracket, but brentq did not "
                    "reach a root within the remaining iteration budget."
                )
                return False
            root_method = "binary_observed_bracket_brentq"
            root_message = (
                "Converged via brentq on one sign-change interval observed directly during the primary binary solve."
            )
            _consider(evaluator.evaluate(np.asarray([theta_star], dtype=float)))
            return True

        def _consider(record: dict[str, Any]) -> float:
            nonlocal best_final, best_residual_max
            _record_species_samples(species_samples, record)
            residual_try = np.asarray(record["mu_residual_ha"], dtype=float)
            residual_max = float(np.max(np.abs(residual_try)))
            if is_binary and residual_try.size == 1:
                theta_scalar = float(np.asarray(record["theta"], dtype=float).reshape(-1)[0])
                dmu_scalar = float(residual_try[0])
                if np.isfinite(theta_scalar) and np.isfinite(dmu_scalar):
                    observed_binary_points.append((theta_scalar, dmu_scalar))
                    _update_observed_binary_bracket()
            if residual_max < best_residual_max:
                best_residual_max = residual_max
                best_final = record
            return residual_max

        # For N>2 we keep the broader legacy surrogate/local-refine strategy.
        # For binary mixtures we deliberately switch to one bracketed scalar
        # fallback after at most 10 conventional iterations instead of relying
        # on repeated local surrogate reseeds.
        if explicit_init is not None and not is_binary:
            explicit_theta = _weights_to_theta(explicit_init)
            seed_eval_count += 1
            _consider(evaluator.evaluate(explicit_theta))
            if best_residual_max > tol:
                for half_width, max_nfev in (
                    (0.04, 10),
                    (0.08, 14),
                    (0.16, 20),
                    (0.28, 28),
                ):
                    theta_center = (
                        np.asarray(best_final["theta"], dtype=float)
                        if best_final is not None
                        else explicit_theta
                    )
                    theta_refined = _local_theta_refine(
                        evaluator,
                        theta_init=theta_center,
                        root_tol=min(float(cfg.root_tol), 1.0e-6),
                        max_nfev=max_nfev,
                        half_width=float(half_width),
                    )
                    if theta_refined is None:
                        continue
                    residual_max = _consider(evaluator.evaluate(theta_refined))
                    if residual_max <= tol:
                        break
            if best_residual_max > tol:
                theta_bracketed = _scalar_theta_bracket_refine(
                    evaluator,
                    theta_center=(
                        np.asarray(best_final["theta"], dtype=float)
                        if best_final is not None
                        else explicit_theta
                    ),
                    root_tol=min(float(cfg.root_tol), 1.0e-6),
                    max_nfev=min(max(int(cfg.root_maxfev), 12), 32),
                )
                if theta_bracketed is not None:
                    _consider(evaluator.evaluate(theta_bracketed))
            if best_residual_max > tol:
                theta_scanned, bracket = _scalar_theta_scan_bracket_refine(
                    evaluator,
                    theta_min=float(explicit_theta[0]) - 0.05,
                    theta_max=float(explicit_theta[0]) + 0.05,
                    n_points=33,
                    root_tol=min(float(cfg.root_tol), 1.0e-6),
                    max_nfev=min(max(int(cfg.root_maxfev), 16), 64),
                )
                if bracket is not None:
                    bracket_found = True
                    bracket_interval = (float(bracket[0]), float(bracket[1]))
                    root_method = "binary_direct_scan_brentq"
                    root_message = (
                        "Converged via direct binary-theta bracket scan around the explicit warm-start."
                    )
                if theta_scanned is not None:
                    _consider(evaluator.evaluate(theta_scanned))

        def _run_surrogate_refine(max_iters: int) -> bool:
            nonlocal surrogate_iters, best_residual_max
            local_iters = 0
            ran_refine = False
            while best_residual_max > tol and local_iters < int(max_iters) and not _primary_method_budget_exhausted():
                try:
                    _, volumes_try = _surrogate_common_mu(
                        species_samples,
                        fractions=evaluator.x,
                        vbar_bohr3=float(evaluator.vbar_bohr3),
                        xtol=max(float(cfg.root_tol), 1.0e-8),
                        rtol=max(float(cfg.root_tol), 1.0e-8),
                    )
                except RuntimeError:
                    break

                surrogate_iters += 1
                local_iters += 1
                ran_refine = True
                weights_try = _volumes_to_weights(
                    volumes_try,
                    fractions=evaluator.x,
                    vbar_bohr3=float(evaluator.vbar_bohr3),
                )
                _consider(evaluator.evaluate(_weights_to_theta(weights_try)))
            return ran_refine

        for seed in seed_weights[:n_seed_target]:
            if _primary_method_budget_exhausted():
                break
            seed_eval_count += 1
            residual_max = _consider(evaluator.evaluate(_weights_to_theta(seed)))
            if residual_max <= tol:
                break

        seed_cursor = n_seed_target
        while best_residual_max > tol and not _primary_method_budget_exhausted():
            ran_refine = _run_surrogate_refine(int(primary_root_maxfev) - surrogate_iters)
            if best_residual_max <= tol or ran_refine:
                break
            if seed_cursor >= len(seed_weights):
                break
            seed_eval_count += 1
            _consider(evaluator.evaluate(_weights_to_theta(seed_weights[seed_cursor])))
            seed_cursor += 1

        if best_residual_max > tol and is_binary and bracket_found and bracket_interval is not None:
            _refine_observed_binary_bracket()

        if best_residual_max > tol and prefer_local_seeds and not is_binary:
            global_seed_weights = _initial_weight_guesses(
                fractions=evaluator.x,
                elements=evaluator.elements,
                explicit=explicit_init,
                local_only=False,
            )
            for seed in global_seed_weights:
                seed_key = tuple(np.round(np.asarray(seed, dtype=float), 10))
                if seed_key in seed_keys_seen:
                    continue
                seed_keys_seen.add(seed_key)
                seed_eval_count += 1
                residual_max = _consider(evaluator.evaluate(_weights_to_theta(seed)))
                if residual_max <= tol:
                    break

            if best_residual_max > tol:
                _run_surrogate_refine(max(4, min(int(primary_root_maxfev), 8)))

        if best_residual_max > tol and best_final is not None and not is_binary:
            theta_refined = _local_theta_refine(
                evaluator,
                theta_init=np.asarray(best_final["theta"], dtype=float),
                root_tol=float(cfg.root_tol),
                max_nfev=min(max(int(primary_root_maxfev), 8), 12),
            )
            if theta_refined is not None:
                _consider(evaluator.evaluate(theta_refined))
        if best_residual_max > tol and best_final is not None and not is_binary:
            theta_bracketed = _scalar_theta_bracket_refine(
                evaluator,
                theta_center=np.asarray(best_final["theta"], dtype=float),
                root_tol=min(float(cfg.root_tol), 1.0e-6),
                max_nfev=min(max(int(primary_root_maxfev), 12), 32),
            )
            if theta_bracketed is not None:
                _consider(evaluator.evaluate(theta_bracketed))
        if best_final is None:
            raise RuntimeError("Multicomponent common-mu solve did not produce any candidate.")

        final = best_final
        theta_star = np.asarray(final["theta"], dtype=float)
        residual = np.asarray(final["mu_residual_ha"], dtype=float)
        residual_max = float(np.max(np.abs(residual)))
        root_success = bool(residual_max <= tol)
        if not root_success:
            if is_binary and bracket_found and bracket_interval is not None:
                if root_method not in (
                    "binary_observed_bracket_brentq_budget_exhausted",
                    "binary_observed_bracket_brentq_failed",
                ):
                    root_method = "binary_observed_bracket_brentq_unconverged"
                    root_message = (
                        "The binary observed-bracket brentq fallback located a sign-change interval in the "
                        "true residual, but did not reach the requested tolerance within the configured "
                        "10-step brent phase. Returning the best available AA state for this temperature."
                    )
            elif bracket_found and bracket_interval is not None:
                root_method = "binary_direct_scan_brentq_bracket_only"
                root_message = (
                    "The binary common-mu scan located a sign-change bracket in the true residual, "
                    "but the requested tolerance was not reached. Returning the best available AA state."
                )
            else:
                root_message = (
                    "The common-mu solve did not reach the requested tolerance. "
                    "Returning the best available AA state instead of aborting."
                )
            if cfg.show_mu_progress or cfg.verbose:
                msg = (
                    "WARNING: multicomponent common-mu solve remained above tolerance: "
                    f"best max|dmu|={residual_max:.3e} Ha, tol={tol:.3e} Ha."
                )
                if bracket_found and bracket_interval is not None:
                    msg += (
                        " "
                        f"Detected one binary sign-change bracket in theta=[{float(bracket_interval[0]):.6f}, "
                        f"{float(bracket_interval[1]):.6f}]."
                    )
                print(msg)

        mu_values = np.asarray(final["mu_ha"], dtype=float)
        mu_common = float(np.mean(mu_values))
        species_payload = _assemble_species_payload(
            elements=evaluator.elements,
            counts=evaluator.counts,
            fractions=evaluator.x,
            volumes_bohr3=np.asarray(final["volumes_bohr3"], dtype=float),
            r_ws_bohr=np.asarray(final["r_ws_bohr"], dtype=float),
            mu_values=mu_values,
            results=list(final["results"]),
        )

        return {
            "mu_common_ha": float(mu_common),
            "theta": theta_star.copy(),
            "volume_weights": np.asarray(final["weights"], dtype=float).copy(),
            "vbar_bohr3": float(evaluator.vbar_bohr3),
            "n_mix_bohr3": float(evaluator.n_mix),
            "species": species_payload,
            "history": list(evaluator.history),
            "meta": {
                "species": [str(elem.symbol) for elem in evaluator.elements],
                "counts": [float(v) for v in evaluator.counts],
                "fractions": [float(v) for v in evaluator.x],
                "temperature_ev": float(cfg.temperature_ev),
                "rho_g_cc": float(cfg.rho_g_cc),
                "mu_common_ha": float(mu_common),
                "mu_residual_max_ha": float(residual_max),
                "volume_weights": [float(v) for v in final["weights"]],
                "vbar_bohr3": float(evaluator.vbar_bohr3),
                "root_success": bool(root_success),
                "root_message": str(root_message),
                "root_nfev": int(len(evaluator.history)),
                "root_method": str(root_method),
                "root_n_seed_evals": int(seed_eval_count),
                "root_n_refine": int(surrogate_iters),
                "root_bracket_found": bool(bracket_found),
                "root_bracket_interval_theta": (
                    None
                    if bracket_interval is None
                    else [float(bracket_interval[0]), float(bracket_interval[1])]
                ),
                "root_species_cache_hits": int(evaluator._species_result_cache_hits),
                "root_species_solves": int(evaluator._species_result_cache_misses),
                "species_parallel_jobs": int(cfg.species_parallel_jobs),
                "species_parallel_backend": str(cfg.species_parallel_backend),
            },
        }
    finally:
        evaluator.close()


def _mixture_full_only_payload(cfg: MixtureConfig) -> dict[str, Any]:
    """
    Resolve the full-only mixture closure with one fixed AA setup.

    Parameters
    ----------
    cfg
        Mixture configuration describing the species list and one shared AA
        configuration.

    Returns
    -------
    dict
        Converged full-only mixture result.
    """
    return solve_mixture_full_only(cfg)


def solve_mixture_full(cfg: MixtureConfig) -> dict[str, Any]:
    """
    Solve the physically relevant full-only multicomponent mixture-AA problem.

    Parameters
    ----------
    cfg
        Mixture configuration object.

    Returns
    -------
    dict
        Full-only mixture result.

    Notes
    -----
    The common-`mu_e` closure is defined entirely by the full-only AA solves.
    The external branch is not part of that closure and is therefore omitted
    here. This entry point exists to keep that physics distinction explicit in
    user code and tests.
    """
    mixture_full = _mixture_full_only_payload(cfg)
    result = {
        "mu_common_ha": float(mixture_full["mu_common_ha"]),
        "theta": np.asarray(mixture_full["theta"], dtype=float).copy(),
        "volume_weights": np.asarray(mixture_full["volume_weights"], dtype=float).copy(),
        "history": list(mixture_full["history"]),
        "species": [{**sp} for sp in mixture_full["species"]],
        "meta": {
            **dict(mixture_full["meta"]),
            "final_run_mode": "full",
            "save_common_linear_grid": bool(cfg.save_common_linear_grid),
            "save_linear_n_points": int(cfg.save_linear_n_points),
        },
    }
    if cfg.save_data:
        save_paths = save_mixture_data(
            output_dir=cfg.save_output_dir,
            mixture_label=_mixture_label_from_counts(
                [str(sp["element"]) for sp in result["species"]],
                np.asarray([float(sp["count"]) for sp in result["species"]], dtype=float),
            ),
            temperature_ev=float(cfg.temperature_ev),
            rho_g_cc=float(cfg.rho_g_cc),
            suffix=str(cfg.save_suffix),
            result=result,
            save_common_linear_grid=bool(cfg.save_common_linear_grid),
            linear_n_points=int(cfg.save_linear_n_points),
        )
        result["saved_paths"] = save_paths
    return result


def _final_species_config(
    cfg: MixtureConfig,
    *,
    element_key: int | str,
    r_ws_bohr: float,
    n_i_bohr3: float,
    extra_overrides: dict[str, Any],
    full_result_init: dict[str, Any] | None = None,
) -> FullExternalConfig:
    """Build one post-root `FullExternalConfig` for a mixture species."""
    run_mode = str(cfg.final_run_mode).strip().lower()
    species_kwargs: dict[str, Any] = {
        "element": element_key,
        "temperature_ev": float(cfg.temperature_ev),
        "rho_g_cc": float(cfg.rho_g_cc),
        "run_mode": "full+ext" if run_mode != "full" else "full",
        "ext_scf_enabled": bool(run_mode != "full"),
        "r_ws_override_bohr": float(r_ws_bohr),
        "n_i_override_bohr3": float(n_i_bohr3),
        "show_scf_progress": bool(cfg.show_progress),
        "verbose": bool(cfg.verbose),
        **cfg.aa_overrides,
        **extra_overrides,
    }
    if run_mode != "full":
        species_kwargs.setdefault("screening_tail_repair_mode", "constrained_b3")
        species_kwargs.setdefault("screening_tail_repair_rel_tol", 5.0e-2)
        species_kwargs.setdefault("screening_tail_charge_weight", 1.0e3)
    cfg_species = FullExternalConfig(
        **species_kwargs
    )
    v_full_init = _resample_full_potential_init(
        full_result_init,
        cfg_species=cfg_species,
        r_ws_bohr=float(r_ws_bohr),
    )
    if v_full_init is not None:
        cfg_species.v_full_init = np.asarray(v_full_init, dtype=float)
    return cfg_species


def solve_mixture_full_then_ext(cfg: MixtureConfig) -> dict[str, Any]:
    """
    Solve the multicomponent common-`mu_e` problem, then rerun each species.

    Parameters
    ----------
    cfg
        Mixture configuration object.

    Returns
    -------
    dict
        Mixture result with per-species final AA branches and optional saved
        dataset paths.
    """
    mixture_full = _mixture_full_only_payload(cfg)

    run_mode = str(cfg.final_run_mode).strip().lower()
    if run_mode == "full":
        final_results = [dict(sp["result"]) for sp in mixture_full["species"]]
    else:
        species_cfgs: list[FullExternalConfig] = []
        for sp in mixture_full["species"]:
            symbol = str(sp["element"])
            species_cfgs.append(
                _final_species_config(
                    cfg,
                    element_key=int(sp["Z"]),
                    r_ws_bohr=float(sp["r_ws_bohr"]),
                    n_i_bohr3=float(1.0 / float(sp["volume_bohr3"])),
                    extra_overrides=dict(cfg.species_overrides.get(symbol, {})),
                    full_result_init=dict(sp["result"]),
                )
            )

        jobs = int(cfg.species_parallel_jobs)
        backend = str(cfg.species_parallel_backend).strip().lower()
        if jobs <= 1:
            final_results = [_solve_species_from_config(cfg_species) for cfg_species in species_cfgs]
        elif backend == "thread":
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                final_results = list(executor.map(_solve_species_from_config, species_cfgs))
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                final_results = list(executor.map(_solve_species_from_config, species_cfgs))
    final_mu_values = np.asarray([float(result_species["mu"]) for result_species in final_results], dtype=float)
    final_mu_common = float(np.mean(final_mu_values))
    final_mu_residual = final_mu_values[:-1] - final_mu_values[-1]
    final_mu_residual_max = float(np.max(np.abs(final_mu_residual))) if final_mu_residual.size > 0 else 0.0

    final_species: list[dict[str, Any]] = []
    for sp, final_result in zip(mixture_full["species"], final_results, strict=True):
        final_species.append(
            {
                **{k: v for k, v in sp.items() if k != "result"},
                "full_only_result": sp["result"],
                "result": final_result,
            }
        )

    result = {
        "mu_common_ha": float(mixture_full["mu_common_ha"]),
        "theta": np.asarray(mixture_full["theta"], dtype=float).copy(),
        "volume_weights": np.asarray(mixture_full["volume_weights"], dtype=float).copy(),
        "history": list(mixture_full["history"]),
        "species": final_species,
        "meta": {
            **dict(mixture_full["meta"]),
            "final_run_mode": "full+ext" if str(cfg.final_run_mode).strip().lower() != "full" else "full",
            "save_common_linear_grid": bool(cfg.save_common_linear_grid),
            "save_linear_n_points": int(cfg.save_linear_n_points),
            "final_mu_common_ha": float(final_mu_common),
            "final_mu_residual_max_ha": float(final_mu_residual_max),
        },
    }
    if cfg.save_data:
        save_paths = save_mixture_data(
            output_dir=cfg.save_output_dir,
            mixture_label=_mixture_label_from_counts(
                [str(sp["element"]) for sp in final_species],
                np.asarray([float(sp["count"]) for sp in final_species], dtype=float),
            ),
            temperature_ev=float(cfg.temperature_ev),
            rho_g_cc=float(cfg.rho_g_cc),
            suffix=str(cfg.save_suffix),
            result=result,
            save_common_linear_grid=bool(cfg.save_common_linear_grid),
            linear_n_points=int(cfg.save_linear_n_points),
        )
        result["saved_paths"] = save_paths
    return result


__all__ = [
    "MixtureConfig",
    "solve_mixture_full",
    "solve_mixture_full_only",
    "solve_mixture_full_then_ext",
]
