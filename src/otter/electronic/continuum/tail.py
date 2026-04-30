"""
otter/electronic/continuum/tail.py

Purpose
-------
Provide analytic linear-response tails for continuum densities (Starrett 2014).

Methods
-------
- Compute Friedel parameters (a0, b0) and Thomas-Fermi k_TF from mu_id, T.
- Fit A, B, delta at R_cut and build analytic tail n(r) - n0.

Equations
---------
Tail form (Piron 2011, Eq. 39; Friedel decay):
  n(r) - n0 = A * exp(-k_TF r) / r
             + B * exp(-2 b0 r) / r^3 * sin(2 a0 r + delta)

Parameters (Piron 2011):
  a0 = sqrt( mu_id + sqrt(mu_id^2 + pi^2 T^2))
  b0 = sqrt(-mu_id + sqrt(mu_id^2 + pi^2 T^2))
  k_TF^2 = (2/pi) * sqrt(2 T) * I_{-1/2}(beta * mu_id)

References
----------
- C. E. Starrett & D. Saumon (2014), Appendix B.

B3 matching pipeline in this module
-----------------------------------
1) Compute tail physics parameters `(a0, b0, k_tf)` from `(mu_id, T)`.
2) At a chosen `R_cut` (index `idx_cut`), fit `n(r)-n0` on a short tail window
   to obtain `(A, B, delta)` in the B3 ansatz.
3) Reconstruct the analytic tail for `r >= R_cut`.
4) Optionally blend over a few points to avoid a visible splice kink.

Main entry points:
- `apply_tail_match`: single-cut B3 splice.
- `scan_tail_match` + `select_tail_cut_converged`: multi-cut selection workflow.
"""
from __future__ import annotations

from typing import Tuple, Dict
import numpy as np
from scipy.integrate import quad
from scipy.special import expit


def _finite_difference_slope(r: np.ndarray, y: np.ndarray, idx: int) -> float:
    """
    Estimate one local slope on a radial grid with the nearest stencil.

    Parameters
    ----------
    r
        Monotone radial grid in Bohr.
    y
        Scalar profile sampled on `r`.
    idx
        Index where the slope is estimated.

    Returns
    -------
    float
        Finite-difference estimate of `dy/dr` at `r[idx]`.
    """
    idx = int(idx)
    i0 = max(idx - 1, 0)
    i1 = min(idx + 1, y.size - 1)
    if i1 == i0:
        return 0.0
    return float((y[i1] - y[i0]) / (r[i1] - r[i0]))


def _hermite_bridge_segment(
    r_seg: np.ndarray,
    *,
    y_left: float,
    dy_left: float,
    y_right: float,
    dy_right: float,
) -> np.ndarray:
    """
    Build one cubic Hermite bridge that matches value and slope at both ends.

    Parameters
    ----------
    r_seg
        Radial segment where the bridge is evaluated (Bohr).
    y_left
        Function value at the left endpoint.
    dy_left
        Radial derivative at the left endpoint.
    y_right
        Function value at the right endpoint.
    dy_right
        Radial derivative at the right endpoint.

    Returns
    -------
    np.ndarray
        Smooth bridge profile sampled on `r_seg`.

    Notes
    -----
    The current B3 fit already matches the A3 tail values fairly well. The
    visible splice ripple is more strongly tied to a slope mismatch near
    `R_cut`, so this bridge enforces `C1` continuity across the blend window.
    """
    r_seg = np.asarray(r_seg, dtype=float)
    if r_seg.size == 0:
        return np.asarray([], dtype=float)
    if r_seg.size == 1:
        return np.asarray([float(y_left)], dtype=float)

    x0 = float(r_seg[0])
    x1 = float(r_seg[-1])
    length = max(x1 - x0, 1.0e-12)
    t = (r_seg - x0) / length

    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return (
        h00 * float(y_left)
        + h10 * length * float(dy_left)
        + h01 * float(y_right)
        + h11 * length * float(dy_right)
    )

def fermi_integral_mhalf(eta: float, n: int = 4000, t_max: float | None = None) -> float:
    """
    Computes the exact unnormalized Fermi-Dirac integral I_{-1/2}(eta).

    Matches the definition in Starrett & Saumon (2013, Eq. 25; 2014, Eq. B3 context):
        I_{-1/2}(eta) = ∫_0^∞ t^(-1/2) / (1 + exp(t - eta)) dt

    Modifications:
    1. Uses adaptive quadrature (scipy.quad) instead of trapezoidal rule for accuracy.
    2. Applies substitution t = u^2 to eliminate the singularity at t=0.
    3. Returns the raw integral (UNNORMALIZED), removing the 1/sqrt(pi) factor.

    Parameters
    ----------
    eta : float
        Scaled chemical potential (mu/T).
    n : int, optional
        Ignored (kept for backward compatibility).
    t_max : float, optional
        Ignored (quad handles infinite bounds automatically).

    Returns
    -------
    float
        The value of the integral I_{-1/2}(eta).
    """
    
    # Substitution: t = u^2  =>  dt = 2 * u * du
    # The term t^(-1/2) becomes 1/u.
    # Integrand becomes: (1/u) * [1 / (1 + exp(u^2 - eta))] * (2 * u)
    #                  = 2 / (1 + exp(u^2 - eta))
    # We use expit(x) = 1 / (1 + exp(-x)) for numerical stability.
    # Here, 1 / (1 + exp(u^2 - eta)) is equivalent to expit(eta - u^2).
    
    def integrand(u):
        return 2.0 * expit(eta - u**2)

    # Integrate from u=0 to u=infinity
    # limit=100 is usually sufficient for this smooth integrand
    val, _ = quad(integrand, 0, np.inf, epsabs=1e-8, epsrel=1e-8, limit=100)

    return float(val)

def tail_parameters(mu_id: float, temperature: float) -> Tuple[float, float, float]:
    """
    Compute a0, b0, k_TF for the analytic tail (Piron 2011(10.1103/PhysRevE.83.026403))--(40), (41)).
    """
    T = max(float(temperature), 1e-12)
    mu = float(mu_id)
    root = np.sqrt(mu * mu + (np.pi * T) ** 2)
    # Starrett 2014 B3: b0 uses +mu, a0 uses -mu, which is not correct
    b0 = np.sqrt(max(-mu + root, 0.0))
    a0 = np.sqrt(max(mu + root, 0.0))

    eta = mu / T
    I_mhalf = fermi_integral_mhalf(eta)
    k_tf_sq = (2.0 / np.pi) * np.sqrt(2.0 * T) * I_mhalf
    k_tf = np.sqrt(max(k_tf_sq, 0.0))
    return a0, b0, k_tf

def linear_response_tail(r: np.ndarray,
                         n0: float,
                         mu_id: float,
                         temperature: float,
                         A: float,
                         B: float,
                         delta: float) -> np.ndarray:
    """
    Evaluate the analytic tail n(r) for r >= R_cut.

    Formula (B3-style tail model)
    -----------------------------
    n(r) = n0
         + A * exp(-k_TF r) / r
         + B * exp(-2 b0 r) / r^3 * sin(2 a0 r + delta)
    """
    a0, b0, k_tf = tail_parameters(mu_id, temperature)
    r = np.asarray(r)
    term_screen = A * np.exp(-k_tf * r) / r
    term_friedel = B * np.exp(-2.0 * b0 * r) / (r ** 3) * np.sin(2.0 * a0 * r + delta)
    return n0 + term_screen + term_friedel


def _solve_tail_fit_system(
    basis: np.ndarray,
    y_fit: np.ndarray,
    *,
    r: np.ndarray,
    n_r: np.ndarray,
    idx_cut: int,
    r_fit: np.ndarray,
    deriv_row: np.ndarray,
    match_value_weight: float,
    match_slope_weight: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve one weighted tail least-squares system and return its fit residual.

    Parameters
    ----------
    basis
        Linear basis sampled on the fit window.
    y_fit
        Target values `n(r)-n0` on the same fit window.
    r
        Full radial grid in Bohr.
    n_r
        Full sampled profile.
    idx_cut
        Left endpoint of the fit window.
    r_fit
        Fit-window radial subgrid.
    deriv_row
        Basis derivative row evaluated at `r_fit[0]`.
    match_value_weight
        Weight applied to the value match at `r_cut`.
    match_slope_weight
        Weight applied to the slope match at `r_cut`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Fitted coefficients, reconstructed fit-window values, and RMS residual.
    """
    basis_aug = np.asarray(basis, dtype=float)
    y_aug = np.asarray(y_fit, dtype=float)

    dr_ref = float(np.mean(np.diff(r_fit))) if r_fit.size > 1 else max(float(r_fit[0]) * 1.0e-3, 1.0e-6)
    rows: list[np.ndarray] = []
    targets: list[float] = []
    if float(match_value_weight) > 0.0:
        rows.append(float(match_value_weight) * basis_aug[0])
        targets.append(float(match_value_weight) * float(y_fit[0]))
    if float(match_slope_weight) > 0.0:
        slope_cut = _finite_difference_slope(r, n_r, int(idx_cut))
        rows.append(float(match_slope_weight) * dr_ref * np.asarray(deriv_row, dtype=float))
        targets.append(float(match_slope_weight) * dr_ref * float(slope_cut))
    if rows:
        basis_aug = np.vstack((basis_aug, *rows))
        y_aug = np.concatenate((y_aug, np.asarray(targets, dtype=float)))

    coeffs, _, _, _ = np.linalg.lstsq(basis_aug, y_aug, rcond=None)
    y_pred = np.asarray(basis, dtype=float) @ np.asarray(coeffs, dtype=float)
    fit_rms = float(np.sqrt(np.mean((y_fit - y_pred) ** 2)))
    return np.asarray(coeffs, dtype=float), np.asarray(y_pred, dtype=float), fit_rms

def fit_tail_params(r: np.ndarray,
                    n_r: np.ndarray,
                    n0: float,
                    mu_id: float,
                    temperature: float,
                    idx_cut: int,
                    fit_points: int = 12,
                    match_value_weight: float = 2.0,
                    match_slope_weight: float = 6.0,
                    model: str = "auto",
                    auto_rel_improve_tol: float = 0.15,
                    auto_signal_rel_tol: float = 2.0e-5) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fit A, B, delta at R_cut using least-squares on a tail window.

    Steps
    -----
    1) Build fit window `[idx_cut, idx_cut + fit_points)`.
    2) Form `y_fit = n_r - n0`.
    3) Build linear basis:
       - `f0 = exp(-k_TF r)/r`
       - `f1 = decay * sin(2 a0 r)`
       - `f2 = decay * cos(2 a0 r)`
       where `decay = exp(-2 b0 r)/r^3`.
    4) Fit either the full B3 basis, the `A-only` Yukawa/TF basis, or both.
    5) In `auto` mode, keep the full B3 model only when it improves the
       fit-window RMS residual by at least `auto_rel_improve_tol` *and* the
       fit-window signal is large enough to justify an oscillatory term.
    6) Augment the fit with optional weighted value / slope constraints at
       `R_cut` so the local splice is less sensitive to a derivative mismatch.
    7) Solve least squares `y ≈ A*f0 + C*f1 + D*f2`.
    8) Convert `(C, D)` to `(B, delta)` via
       `B = sqrt(C^2 + D^2)`, `delta = atan2(D, C)`.
    """
    r = np.asarray(r)
    n_r = np.asarray(n_r)
    idx_cut = int(idx_cut)

    i0 = max(0, idx_cut)
    i1 = min(r.size, i0 + max(3, fit_points))
    r_fit = r[i0:i1]
    y_fit = n_r[i0:i1] - n0

    if r_fit.size < 3:
        raise ValueError("Not enough points to fit tail parameters.")

    model_req = str(model or "auto").strip().lower()
    if model_req not in ("auto", "full", "a_only"):
        raise ValueError("tail fit model must be 'auto', 'full', or 'a_only'.")

    a0, b0, k_tf = tail_parameters(mu_id, temperature)
    # Linearized tail basis so the fit remains a single lstsq solve.
    f0 = np.exp(-k_tf * r_fit) / r_fit
    decay = np.exp(-2.0 * b0 * r_fit) / (r_fit ** 3)
    f1 = decay * np.sin(2.0 * a0 * r_fit)
    f2 = decay * np.cos(2.0 * a0 * r_fit)
    d_f0 = f0[0] * (-(k_tf) - 1.0 / r_fit[0])
    decay0 = decay[0]
    d_decay0 = decay0 * (-(2.0 * b0) - 3.0 / r_fit[0])
    phase0 = 2.0 * a0 * r_fit[0]
    d_f1 = d_decay0 * np.sin(phase0) + decay0 * (2.0 * a0) * np.cos(phase0)
    d_f2 = d_decay0 * np.cos(phase0) - decay0 * (2.0 * a0) * np.sin(phase0)

    basis_a_only = np.column_stack((f0,))
    coeffs_a_only, y_pred_a_only, fit_rms_a_only = _solve_tail_fit_system(
        basis_a_only,
        y_fit,
        r=r,
        n_r=n_r,
        idx_cut=i0,
        r_fit=r_fit,
        deriv_row=np.array([d_f0], dtype=float),
        match_value_weight=match_value_weight,
        match_slope_weight=match_slope_weight,
    )
    A_a_only = float(coeffs_a_only[0])

    basis_full = np.column_stack((f0, f1, f2))
    coeffs_full, y_pred_full, fit_rms_full = _solve_tail_fit_system(
        basis_full,
        y_fit,
        r=r,
        n_r=n_r,
        idx_cut=i0,
        r_fit=r_fit,
        deriv_row=np.array([d_f0, d_f1, d_f2], dtype=float),
        match_value_weight=match_value_weight,
        match_slope_weight=match_slope_weight,
    )
    A_full, C_full, D_full = [float(x) for x in coeffs_full]
    B_full = float(np.hypot(C_full, D_full))
    delta_full = float(np.arctan2(D_full, C_full))

    rel_improve = max(float(fit_rms_a_only) - float(fit_rms_full), 0.0) / max(float(fit_rms_a_only), 1.0e-30)
    signal_max = float(np.max(np.abs(y_fit)))
    signal_threshold = max(1.0e-6, float(auto_signal_rel_tol) * max(abs(float(n0)), 1.0e-12)) # deside using A3+B3 tail matching or a_only (no friedel oscillation)
    signal_large_enough = bool(signal_max >= signal_threshold)
    if model_req == "a_only":
        A = float(A_a_only)
        B = 0.0
        delta = 0.0
        fit_rms = float(fit_rms_a_only)
        model_sel = "a_only"
    elif model_req == "full":
        A = float(A_full)
        B = float(B_full)
        delta = float(delta_full)
        fit_rms = float(fit_rms_full)
        model_sel = "full"
    else:
        use_full = bool(rel_improve >= float(auto_rel_improve_tol) and signal_large_enough)
        if use_full:
            A = float(A_full)
            B = float(B_full)
            delta = float(delta_full)
            fit_rms = float(fit_rms_full)
            model_sel = "full"
        else:
            A = float(A_a_only)
            B = 0.0
            delta = 0.0
            fit_rms = float(fit_rms_a_only)
            model_sel = "a_only"
    meta = {
        "A": float(A),
        "B": float(B),
        "delta": float(delta),
        "a0": float(a0),
        "b0": float(b0),
        "k_tf": float(k_tf),
        "idx_cut": int(idx_cut),
        "fit_points": int(r_fit.size),
        "fit_method": "linear",
        "fit_rms": float(fit_rms),
        "fit_rms_a_only": float(fit_rms_a_only),
        "fit_rms_full": float(fit_rms_full),
        "fit_rel_improve_full": float(rel_improve),
        "fit_signal_max": float(signal_max),
        "fit_signal_threshold": float(signal_threshold),
        "match_value_weight": float(match_value_weight),
        "match_slope_weight": float(match_slope_weight),
        "model_requested": str(model_req),
        "model_selected": str(model_sel),
        "auto_rel_improve_tol": float(auto_rel_improve_tol),
        "auto_signal_rel_tol": float(auto_signal_rel_tol),
        "full_B": float(B_full),
        "full_delta": float(delta_full),
        "a_only_A": float(A_a_only),
    }
    return np.array([A, B, delta], dtype=float), meta


def apply_tail_match(r: np.ndarray,
                     n_r: np.ndarray,
                     n0: float,
                     mu_id: float,
                     temperature: float,
                     idx_cut: int,
                     fit_points: int = 12,
                     blend_points: int = 0,
                     model: str = "auto",
                     auto_rel_improve_tol: float = 0.15,
                     auto_signal_rel_tol: float = 2.0e-5) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Replace n(r) for r >= R_cut with an analytic tail fit.

    Steps
    -----
    1) Fit `(A, B, delta)` near `R_cut` with `fit_tail_params`.
    2) Build analytic tail `n_tail(r)` for `r >= R_cut`.
    3) Hard splice: replace `n_r[idx_cut:]` by `n_tail`.
    4) Optional Hermite bridge over `blend_points` to smooth the splice.

    Notes
    -----
    `blend_points` applies a cubic Hermite bridge across
    `[idx_cut, idx_cut + blend_points)` so both the value and the local slope
    match at the start and end of the transition window.
    """
    # (1) Local parameter fit around R_cut.
    params, meta = fit_tail_params(
        r,
        n_r,
        n0,
        mu_id,
        temperature,
        idx_cut,
        fit_points=fit_points,
        model=model,
        auto_rel_improve_tol=auto_rel_improve_tol,
        auto_signal_rel_tol=auto_signal_rel_tol,
    )
    A, B, delta = params

    # (2) Build analytic tail and (3) splice into the output profile.
    n_out = np.array(n_r, copy=True)
    r_tail = r[idx_cut:]
    n_tail = linear_response_tail(r_tail, n0, mu_id, temperature, A, B, delta)
    n_out[idx_cut:] = n_tail

    # (4) Optional C1 bridge across the splice junction.
    blend_points = int(max(blend_points, 0))
    if blend_points > 0:
        b0 = int(idx_cut)
        b1 = min(int(idx_cut + blend_points), r.size)
        if b1 > b0 + 1:
            n_out[b0:b1] = _hermite_bridge_segment(
                r[b0:b1],
                y_left=float(n_r[b0]),
                dy_left=_finite_difference_slope(r, n_r, b0),
                y_right=float(n_out[b1 - 1]),
                dy_right=_finite_difference_slope(r, n_out, b1 - 1),
            )

    meta["blend_points"] = int(min(blend_points, max(r.size - idx_cut, 0)))
    meta["blend_mode"] = "hermite" if blend_points > 0 else "off"
    return n_out, meta


def select_tail_cut(r: np.ndarray,
                    n_r: np.ndarray,
                    n0: float,
                    *,
                    v_eff: np.ndarray | None = None,
                    r_min: float | None = None,
                    r_fraction: float = 0.7,
                    rel_tol: float | None = None,
                    abs_tol: float | None = None,
                    v_tol: float | None = None,
                    min_points: int = 12) -> Tuple[int | None, Dict[str, float]]:
    """
    Select an R_cut index for tail matching.

    Parameters
    ----------
    r
        Radial grid (Bohr).
    n_r
        Continuum density n_c(r) (a0^-3).
    n0
        Uniform reference density (a0^-3).
    v_eff
        Effective potential V_eff(r) (Hartree). Optional.
    r_min
        Minimum radius to consider for R_cut (Bohr). If None, use r_fraction * rmax.
    r_fraction
        Fraction of rmax used when r_min is None.
    rel_tol
        Relative tolerance for |n_r - n0| (optional).
    abs_tol
        Absolute tolerance for |n_r - n0| (optional).
    v_tol
        Potential tolerance for |V_eff| (optional).
    min_points
        Minimum contiguous points that must satisfy the criteria.

    Returns
    -------
    idx_cut
        Index of the selected R_cut, or None if no window satisfies criteria.
    meta
        Diagnostics for the selection process.
    """
    r = np.asarray(r, dtype=float)
    n_r = np.asarray(n_r, dtype=float)
    if r.shape != n_r.shape:
        raise ValueError("r and n_r must have the same shape.")

    if r.size == 0:
        return None, {
            "idx_cut": None,
            "r_cut": None,
            "r_min": None,
            "r_fraction": float(r_fraction),
            "rel_tol": float(rel_tol) if rel_tol is not None else None,
            "abs_tol": float(abs_tol) if abs_tol is not None else None,
            "n_tol": None,
            "v_tol": float(v_tol) if v_tol is not None else None,
            "min_points": int(min_points),
            "n_criteria": False,
            "v_criteria": False,
        }

    if r_min is None:
        r_min = float(r_fraction) * float(r[-1])
    r_min = float(r_min)
    i_start = int(np.searchsorted(r, r_min))

    min_points = max(int(min_points), 2)

    n_tol = None
    mask_n = np.ones_like(r, dtype=bool)
    if rel_tol is not None or abs_tol is not None:
        abs_tol_val = 0.0 if abs_tol is None else float(abs_tol)
        rel_tol_val = None if rel_tol is None else float(rel_tol)
        n_tol = abs_tol_val
        if rel_tol_val is not None:
            n_tol = max(n_tol, rel_tol_val * max(abs(n0), 0.0))
        mask_n = np.abs(n_r - n0) <= n_tol

    v_used = False
    mask_v = np.ones_like(r, dtype=bool)
    if v_tol is not None and v_eff is not None:
        v_eff = np.asarray(v_eff, dtype=float)
        if v_eff.shape != r.shape:
            raise ValueError("v_eff must have the same shape as r.")
        v_used = True
        mask_v = np.abs(v_eff) <= float(v_tol)

    mask = mask_n & mask_v
    mask[:i_start] = False

    idx_cut = None
    if i_start < r.size:
        for i in range(i_start, r.size - min_points + 1):
            if np.all(mask[i:i + min_points]):
                idx_cut = i
                break

    meta = {
        "idx_cut": int(idx_cut) if idx_cut is not None else None,
        "r_cut": float(r[idx_cut]) if idx_cut is not None else None,
        "r_min": float(r_min),
        "r_fraction": float(r_fraction),
        "rel_tol": float(rel_tol) if rel_tol is not None else None,
        "abs_tol": float(abs_tol) if abs_tol is not None else None,
        "n_tol": float(n_tol) if n_tol is not None else None,
        "v_tol": float(v_tol) if v_tol is not None else None,
        "min_points": int(min_points),
        "n_criteria": bool(n_tol is not None),
        "v_criteria": bool(v_used),
        "n_pass": int(np.sum(mask_n[i_start:])) if i_start < r.size else 0,
        "v_pass": int(np.sum(mask_v[i_start:])) if i_start < r.size else 0,
    }
    return idx_cut, meta


def _tail_charge(r: np.ndarray,
                 n_tail: np.ndarray,
                 n0: float,
                 idx_cut: int) -> float:
    """
    Integrate the tail screening charge for r >= R_cut.
    """
    r_tail = r[idx_cut:]
    integrand = 4.0 * np.pi * (n_tail[idx_cut:] - n0) * (r_tail ** 2)
    charge = np.trapezoid(integrand, r_tail)
    return float(charge)


def scan_tail_match(r: np.ndarray,
                    n_r: np.ndarray,
                    n0: float,
                    mu_id: float,
                    temperature: float,
                    *,
                    r_cuts: list[float] | None = None,
                    idx_cuts: list[int] | None = None,
                    fit_points: int = 12,
                    blend_points: int = 0) -> Tuple[list[Dict[str, float]], Dict[str, float]]:
    """
    Scan tail matching across multiple R_cut candidates.

    Parameters
    ----------
    r
        Radial grid (Bohr).
    n_r
        Continuum density n_c(r) (a0^-3).
    n0
        Uniform reference density (a0^-3).
    mu_id
        Ideal chemical potential (Hartree).
    temperature
        Temperature (Hartree).
    r_cuts
        Candidate R_cut values (Bohr).
    idx_cuts
        Candidate indices for R_cut. If provided, overrides r_cuts.
    fit_points
        Number of points used to fit the tail parameters.
    blend_points
        Number of points used for cosine blending near R_cut.

    Returns
    -------
    results
        List of per-cut metrics (A, B, delta, tail_charge, tail_mean_abs).
    meta
        Scan summary and bounds.
    """
    r = np.asarray(r, dtype=float)
    n_r = np.asarray(n_r, dtype=float)
    if r.shape != n_r.shape:
        raise ValueError("r and n_r must have the same shape.")

    if idx_cuts is None:
        if r_cuts is None:
            raise ValueError("Either r_cuts or idx_cuts must be provided.")
        idx_cuts = [int(np.searchsorted(r, float(rc))) for rc in r_cuts]

    idx_sorted = sorted(set(int(idx) for idx in idx_cuts))
    results: list[Dict[str, float]] = []
    for idx_cut in idx_sorted:
        if idx_cut <= 0 or idx_cut >= r.size - 1:
            continue
        n_tail, meta = apply_tail_match(
            r,
            n_r,
            n0,
            mu_id,
            temperature,
            idx_cut,
            fit_points=fit_points,
            blend_points=blend_points,
        )
        tail_charge = _tail_charge(r, n_tail, n0, idx_cut)
        total_charge = _tail_charge(r, n_tail, n0, 0)
        tail_mean_abs = float(np.mean(np.abs(n_tail[idx_cut:] - n_r[idx_cut:])))
        metrics = {
            "idx_cut": int(idx_cut),
            "r_cut": float(r[idx_cut]),
            "A": float(meta["A"]),
            "B": float(meta["B"]),
            "delta": float(meta["delta"]),
            "tail_charge": float(tail_charge),
            "total_charge": float(total_charge),
            "tail_mean_abs": float(tail_mean_abs),
        }
        results.append(metrics)

    meta = {
        "n_cuts": int(len(results)),
        "fit_points": int(fit_points),
        "blend_points": int(blend_points),
        "r_min": float(r[0]) if r.size > 0 else None,
        "r_max": float(r[-1]) if r.size > 0 else None,
    }
    return results, meta


def select_tail_cut_converged(scan_results: list[Dict[str, float]],
                              *,
                              charge_key: str = "total_charge",
                              charge_tol_rel: float = 1e-3,
                              charge_tol_abs: float = 1e-6,
                              delta_tol: float | None = None,
                              param_tol_rel: float | None = None) -> Tuple[int | None, Dict[str, float]]:
    """
    Choose the first R_cut where tail metrics stabilize.

    Parameters
    ----------
    scan_results
        Output of scan_tail_match (sorted by R_cut).
    charge_tol_rel
        Relative tolerance for tail_charge stabilization.
    charge_tol_abs
        Absolute tolerance for tail_charge stabilization.
    delta_tol
        Optional absolute tolerance for delta stabilization.
    param_tol_rel
        Optional relative tolerance for A/B stabilization.

    Returns
    -------
    idx_cut
        Selected index for R_cut (or None if not converged).
    meta
        Diagnostics for the selection criteria.
    """
    if len(scan_results) < 2:
        return None, {
            "idx_cut": None,
            "r_cut": None,
            "charge_key": str(charge_key),
            "charge_tol_rel": float(charge_tol_rel),
            "charge_tol_abs": float(charge_tol_abs),
            "delta_tol": float(delta_tol) if delta_tol is not None else None,
            "param_tol_rel": float(param_tol_rel) if param_tol_rel is not None else None,
            "n_pairs": int(max(len(scan_results) - 1, 0)),
        }

    def _param_ok(val: float, prev: float, tol_rel: float | None) -> bool:
        if tol_rel is None:
            return True
        scale = max(abs(val), abs(prev), 1e-12)
        return abs(val - prev) <= tol_rel * scale

    selected = None
    diffs = []
    for i in range(1, len(scan_results)):
        cur = scan_results[i]
        prev = scan_results[i - 1]
        if charge_key not in cur or charge_key not in prev:
            raise KeyError(f"scan_results missing '{charge_key}' for convergence check.")
        charge = float(cur[charge_key])
        charge_prev = float(prev[charge_key])
        charge_diff = abs(charge - charge_prev)
        charge_scale = max(abs(charge), abs(charge_prev), 1e-12)
        charge_ok = charge_diff <= max(charge_tol_abs, charge_tol_rel * charge_scale)

        delta_ok = True
        if delta_tol is not None:
            delta_ok = abs(float(cur["delta"]) - float(prev["delta"])) <= float(delta_tol)

        param_ok = True
        if param_tol_rel is not None:
            param_ok = (
                _param_ok(float(cur["A"]), float(prev["A"]), param_tol_rel)
                and _param_ok(float(cur["B"]), float(prev["B"]), param_tol_rel)
            )

        diffs.append(
            {
                "idx_prev": int(prev["idx_cut"]),
                "idx_cur": int(cur["idx_cut"]),
                "charge_diff": float(charge_diff),
                "charge_ok": bool(charge_ok),
                "delta_ok": bool(delta_ok),
                "param_ok": bool(param_ok),
            }
        )

        if charge_ok and delta_ok and param_ok:
            selected = cur
            break

    meta = {
        "idx_cut": int(selected["idx_cut"]) if selected is not None else None,
        "r_cut": float(selected["r_cut"]) if selected is not None else None,
        "charge_key": str(charge_key),
        "charge_tol_rel": float(charge_tol_rel),
        "charge_tol_abs": float(charge_tol_abs),
        "delta_tol": float(delta_tol) if delta_tol is not None else None,
        "param_tol_rel": float(param_tol_rel) if param_tol_rel is not None else None,
        "n_pairs": int(len(scan_results) - 1),
        "diffs": diffs,
    }
    return (int(selected["idx_cut"]) if selected is not None else None), meta
