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
Tail form (:cite:`PironBlenski2011`, Eq. 39; Friedel decay):
  n(r) - n0 = A * exp(-k_TF r) / r
             + B * exp(-2 b0 r) / r^3 * sin(2 a0 r + delta)

Parameters (:cite:`PironBlenski2011`):
  a0 = sqrt( mu_id + sqrt(mu_id^2 + pi^2 T^2))
  b0 = sqrt(-mu_id + sqrt(mu_id^2 + pi^2 T^2))
  k_TF^2 = (2/pi) * sqrt(2 T) * I_{-1/2}(beta * mu_id)

References
----------
- :cite:`StarrettSaumon2014`, Appendix B. The fit-window selection and splice
  blending controls are Otter implementation choices.

B3 matching pipeline in this module
-----------------------------------
1) Compute tail physics parameters `(a0, b0, k_tf)` from `(mu_id, T)`.
2) At a chosen `R_cut` (index `idx_cut`), fit `n(r)-n0` on a tail window to
   obtain `(A, B, delta)` in the B3 ansatz.  `fit_window_mode` chooses between
   the historical contiguous-point stencil, samples distributed across
   `[R_cut, r_fit_max]`, or a decay-aware automatic choice.
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


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Version-independent one-dimensional trapezoidal integral."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if y_arr.ndim != 1 or x_arr.ndim != 1 or y_arr.size != x_arr.size:
        raise ValueError("The tail trapezoidal integral requires equal 1D arrays.")
    if y_arr.size < 2:
        return 0.0
    return float(
        np.sum(0.5 * (y_arr[1:] + y_arr[:-1]) * np.diff(x_arr))
    )


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
    Compute a0, b0, k_TF for the analytic tail from
    :cite:`PironBlenski2011`, Eqs. (39)--(41).
    """
    T = max(float(temperature), 1e-12)
    mu = float(mu_id)
    root = np.sqrt(mu * mu + (np.pi * T) ** 2)
    # Use Piron et al. (2011), Eqs. (40)--(41), with ``a0`` as the
    # oscillation wavenumber and ``b0`` as the damping coefficient.  This
    # branch has the required degenerate limit a0 -> k_F and b0 -> 0.
    # Appendix B of Starrett--Saumon (2014) interchanges these labels in the
    # displayed definitions while retaining the same damped-Friedel ansatz.
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


def _linear_response_tail_basis(
    r: np.ndarray,
    mu_id: float,
    temperature: float,
) -> np.ndarray:
    """
    Return the linearized B3 basis associated with coefficients ``(A, C, D)``.

    Here ``C = B cos(delta)`` and ``D = B sin(delta)``, so the oscillatory
    contribution is ``C*f_sin + D*f_cos``.  Keeping this representation
    explicit lets the finite-box charge integral remain a linear constraint.
    """
    r_arr = np.asarray(r, dtype=float)
    a0, b0, k_tf = tail_parameters(mu_id, temperature)
    f0 = np.exp(-k_tf * r_arr) / r_arr
    decay = np.exp(-2.0 * b0 * r_arr) / (r_arr**3)
    f1 = decay * np.sin(2.0 * a0 * r_arr)
    f2 = decay * np.cos(2.0 * a0 * r_arr)
    return np.column_stack((f0, f1, f2))


def _splice_linear_tail(
    r: np.ndarray,
    n_r: np.ndarray,
    n0: float,
    mu_id: float,
    temperature: float,
    idx_cut: int,
    coeffs_acd: np.ndarray,
    blend_points: int,
) -> np.ndarray:
    """
    Splice one linearized B3 tail, including the optional Hermite bridge.

    The operation is affine in ``(A, C, D)``.  That property is used below to
    derive an exact finite-grid charge row which includes both the trapezoidal
    endpoint coupling and the bridge segment.
    """
    r_arr = np.asarray(r, dtype=float)
    n_arr = np.asarray(n_r, dtype=float)
    coeffs = np.asarray(coeffs_acd, dtype=float).reshape(-1)
    if coeffs.size != 3:
        raise ValueError("coeffs_acd must contain (A, C, D).")

    out = np.array(n_arr, copy=True)
    basis_tail = _linear_response_tail_basis(
        r_arr[int(idx_cut):],
        mu_id,
        temperature,
    )
    out[int(idx_cut):] = float(n0) + basis_tail @ coeffs

    n_blend = int(max(blend_points, 0))
    if n_blend > 0:
        b0 = int(idx_cut)
        b1 = min(int(idx_cut + n_blend), r_arr.size)
        if b1 > b0 + 1:
            out[b0:b1] = _hermite_bridge_segment(
                r_arr[b0:b1],
                y_left=float(n_arr[b0]),
                dy_left=_finite_difference_slope(r_arr, n_arr, b0),
                y_right=float(out[b1 - 1]),
                dy_right=_finite_difference_slope(r_arr, out, b1 - 1),
            )
    return out


def _splice_linear_tail_response(
    r: np.ndarray,
    mu_id: float,
    temperature: float,
    idx_cut: int,
    coeffs_acd: np.ndarray,
    blend_points: int,
) -> np.ndarray:
    """
    Return the linear response of the final splice to ``(A, C, D)``.

    Computing this response directly is important when a B3 basis column is
    many orders of magnitude below ``n0``.  Forming
    ``splice(unit)-splice(zero)`` would then lose the column to floating-point
    cancellation even though its integrated response can still be finite.
    """
    r_arr = np.asarray(r, dtype=float)
    coeffs = np.asarray(coeffs_acd, dtype=float).reshape(-1)
    if coeffs.size != 3:
        raise ValueError("coeffs_acd must contain (A, C, D).")

    idx_cut = int(idx_cut)
    response = np.zeros_like(r_arr)
    basis_tail = _linear_response_tail_basis(
        r_arr[idx_cut:],
        mu_id,
        temperature,
    )
    response[idx_cut:] = basis_tail @ coeffs

    n_blend = int(max(blend_points, 0))
    if n_blend > 0:
        b0 = idx_cut
        b1 = min(idx_cut + n_blend, r_arr.size)
        if b1 > b0 + 1:
            response[b0:b1] = _hermite_bridge_segment(
                r_arr[b0:b1],
                y_left=0.0,
                dy_left=0.0,
                y_right=float(response[b1 - 1]),
                dy_right=_finite_difference_slope(
                    r_arr, response, b1 - 1
                ),
            )
    return response


def _spherical_integral(r: np.ndarray, density: np.ndarray) -> float:
    """Return ``4*pi*integral r^2 density(r) dr`` on the supplied finite grid."""
    r_arr = np.asarray(r, dtype=float)
    density_arr = np.asarray(density, dtype=float)
    return float(4.0 * np.pi * _trapz((r_arr**2) * density_arr, r_arr))


def _spherical_l2_norm(r: np.ndarray, density: np.ndarray) -> float:
    """Return the radial three-dimensional L2 norm of one density profile."""
    r_arr = np.asarray(r, dtype=float)
    density_arr = np.asarray(density, dtype=float)
    norm_sq = 4.0 * np.pi * _trapz(
        (r_arr**2) * (density_arr**2),
        r_arr,
    )
    return float(np.sqrt(max(norm_sq, 0.0)))


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
    integral_row: np.ndarray | None = None,
    integral_target: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float | int]]:
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
    integral_row
        Optional row whose dot product with the fitted linear coefficients is
        an integrated tail charge.  When supplied together with
        ``integral_target``, this equation is imposed as an exact equality,
        rather than as a large-weight least-squares penalty.
    integral_target
        Right-hand side of the optional exact integral constraint.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float, dict[str, float | int]]
        Fitted coefficients, reconstructed fit-window values, RMS residual,
        and linear-system conditioning diagnostics.
    """
    basis_aug = np.asarray(basis, dtype=float)
    y_aug = np.asarray(y_fit, dtype=float)

    # Keep the derivative-row normalization tied to the underlying radial
    # grid, not to the spacing of possibly subsampled fit points.  Otherwise
    # selecting samples across a wider physical window would silently increase
    # the slope constraint by orders of magnitude.
    i_dr_hi = min(int(idx_cut) + 1, int(r.size) - 1)
    i_dr_lo = max(int(idx_cut), 0)
    dr_ref = (
        float(r[i_dr_hi] - r[i_dr_lo])
        if i_dr_hi > i_dr_lo
        else max(float(r_fit[0]) * 1.0e-3, 1.0e-6)
    )
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

    # The Friedel columns can be many orders of magnitude smaller than the
    # Yukawa/TF column at large R_cut.  Column scaling does not regularize the
    # physics; it only prevents the numerical rank decision from depending on
    # those arbitrary basis units.
    column_norm = np.linalg.norm(basis_aug, axis=0)
    column_scale = np.where(column_norm > 0.0, column_norm, 1.0)
    basis_scaled = basis_aug / column_scale[np.newaxis, :]
    _, singular_values, _ = np.linalg.svd(basis_scaled, full_matrices=False)
    rank = int(np.linalg.matrix_rank(basis_scaled))
    constraint_applied = integral_row is not None or integral_target is not None
    constraint_residual = float("nan")
    constraint_redundant = False
    if constraint_applied:
        if integral_row is None or integral_target is None:
            raise ValueError(
                "integral_row and integral_target must either both be supplied or both be None."
            )
        row = np.asarray(integral_row, dtype=float).reshape(-1)
        if row.size != basis_aug.shape[1]:
            raise ValueError(
                "integral_row length must match the number of fitted tail coefficients."
            )
        target = float(integral_target)
        if not np.all(np.isfinite(row)) or not np.isfinite(target):
            raise ValueError("The integral constraint must contain only finite values.")

        # If x_scaled = column_scale*x, the equality row*x=target becomes
        # (row/column_scale)*x_scaled=target.  Solve in the null space of this
        # one-row constraint so charge closure is satisfied to roundoff,
        # without introducing an arbitrary very large least-squares weight.
        row_scaled = row / column_scale
        row_norm_sq = float(np.dot(row_scaled, row_scaled))
        if row_norm_sq <= np.finfo(float).tiny:
            if abs(target) > 1.0e-12:
                raise ValueError(
                    "The B3 integral-constraint row is numerically zero "
                    "for a nonzero target."
                )
            # 0*x=0 adds no information.  Treat it as an explicitly recorded
            # redundant equality instead of rejecting a valid tail fit.
            coeffs_scaled, _, _, _ = np.linalg.lstsq(
                basis_scaled,
                y_aug,
                rcond=None,
            )
            constraint_redundant = True
            constraint_residual = float(-target)
        else:
            coeffs_particular = row_scaled * (target / row_norm_sq)
            _, _, vh_constraint = np.linalg.svd(
                row_scaled[np.newaxis, :],
                full_matrices=True,
            )
            null_basis = vh_constraint[1:, :].T
            if null_basis.shape[1] > 0:
                reduced_basis = basis_scaled @ null_basis
                reduced_target = y_aug - basis_scaled @ coeffs_particular
                reduced_coeffs, _, _, _ = np.linalg.lstsq(
                    reduced_basis,
                    reduced_target,
                    rcond=None,
                )
                coeffs_scaled = (
                    coeffs_particular + null_basis @ reduced_coeffs
                )
            else:
                coeffs_scaled = coeffs_particular
            constraint_residual = float(
                np.dot(row_scaled, coeffs_scaled) - target
            )
    else:
        coeffs_scaled, _, _, _ = np.linalg.lstsq(
            basis_scaled,
            y_aug,
            rcond=None,
        )
    coeffs = coeffs_scaled / column_scale
    y_pred = np.asarray(basis, dtype=float) @ np.asarray(coeffs, dtype=float)
    fit_rms = float(np.sqrt(np.mean((y_fit - y_pred) ** 2)))
    cond_scaled = _finite_singular_value_condition_number(singular_values)
    cond_raw = float(np.linalg.cond(basis_aug))
    diagnostics: dict[str, float | int] = {
        "rank": int(rank),
        "condition_number": float(cond_raw),
        "condition_number_scaled": float(cond_scaled),
        "column_scale_min": float(np.min(column_scale)),
        "column_scale_max": float(np.max(column_scale)),
        "integral_constraint_applied": bool(constraint_applied),
        "integral_constraint_redundant": bool(constraint_redundant),
        "integral_constraint_residual": float(constraint_residual),
    }
    return (
        np.asarray(coeffs, dtype=float),
        np.asarray(y_pred, dtype=float),
        fit_rms,
        diagnostics,
    )


def _finite_singular_value_condition_number(
    singular_values: np.ndarray,
) -> float:
    """Return a finite-precision condition number without divide overflow.

    An exactly or numerically singular tail-fit matrix legitimately has an
    infinite condition number.  Compare against the largest representable
    ratio before dividing so this diagnostic does not emit an unrelated
    ``RuntimeWarning`` during an otherwise valid fit.
    """
    values = np.asarray(singular_values, dtype=float)
    if values.size == 0:
        return float("inf")
    largest = float(values[0])
    smallest = float(values[-1])
    if not np.isfinite(largest) or not np.isfinite(smallest) or smallest <= 0.0:
        return float("inf")
    if smallest <= largest / np.finfo(float).max:
        return float("inf")
    return float(largest / smallest)


def _tail_fit_indices(
    r: np.ndarray,
    *,
    idx_cut: int,
    fit_points: int,
    r_fit_max: float | None,
) -> tuple[np.ndarray, str]:
    """
    Select B3 fit samples while preserving the physical fit-window semantics.

    With no explicit right edge, this returns the historical consecutive
    samples starting at ``idx_cut``.  With ``r_fit_max``, the requested number
    of samples is distributed approximately uniformly in radius across
    ``[r[idx_cut], r_fit_max]``.  This makes ``r_fit_max`` an actual fit-window
    boundary rather than merely a cap on a tiny local stencil.
    """
    r_arr = np.asarray(r, dtype=float)
    i0 = max(int(idx_cut), 0)
    n_requested = max(int(fit_points), 3)

    if r_fit_max is None:
        i1 = min(int(r_arr.size), i0 + n_requested)
        return np.arange(i0, i1, dtype=int), "contiguous"

    i1 = int(np.searchsorted(r_arr, float(r_fit_max), side="right"))
    i1 = min(max(i1, i0), int(r_arr.size))
    n_available = int(i1 - i0)
    if n_available < 3:
        raise ValueError(
            "B3 fit window needs at least three grid points between "
            f"r_cut={float(r_arr[i0]):.8g} and r_fit_max={float(r_fit_max):.8g}."
        )

    n_samples = min(n_requested, n_available)
    if n_samples == n_available:
        return np.arange(i0, i1, dtype=int), "physical_window_all"

    targets = np.linspace(float(r_arr[i0]), float(r_arr[i1 - 1]), n_samples)
    indices = np.searchsorted(r_arr, targets, side="left")
    indices = np.clip(indices, i0, i1 - 1)
    indices[0] = i0
    indices[-1] = i1 - 1
    indices = np.unique(indices.astype(int))
    if indices.size < 3:
        raise ValueError("B3 physical fit window produced fewer than three unique samples.")
    return indices, "physical_window_subsampled"

def fit_tail_params(r: np.ndarray,
                    n_r: np.ndarray,
                    n0: float,
                    mu_id: float,
                    temperature: float,
                    idx_cut: int,
                    fit_points: int = 12,
                    r_fit_max: float | None = None,
                    local_fit_width: float | None = None,
                    fit_window_mode: str = "local",
                    match_value_weight: float = 2.0,
                    match_slope_weight: float = 6.0,
                    model: str = "full",
                    auto_rel_improve_tol: float = 0.15,
                    auto_signal_rel_tol: float = 2.0e-5,
                    integral_constraint_row: np.ndarray | None = None,
                    integral_constraint_target: float | None = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fit A, B, delta at R_cut using least-squares on a tail window.

    Steps
    -----
    1) Build the fit samples:
       - in `local` mode with `local_fit_width`, distribute at most
         `fit_points` samples over the short physical stencil
         `[r_cut, r_cut + local_fit_width]`;
       - in legacy `local` mode without `local_fit_width`, use the contiguous
         index window `[idx_cut, idx_cut + fit_points)`;
       - in `physical` mode with `r_fit_max`, distribute at most `fit_points` samples across the
         physical window `[r[idx_cut], r_fit_max]`.
       - in `auto` mode, use the physical window only while the Friedel
         envelope at its right edge remains at least 10% of its value at
         `r_cut`; otherwise use the local stencil rather than fit numerical
         noise after the oscillatory tail has already decayed.
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
    9) When supplied, impose the finite-box integral equation
       `integral_constraint_row @ (A, C, D) = integral_constraint_target`
       exactly.  The A-only candidate uses the first entry of the same row.
    """
    r = np.asarray(r)
    n_r = np.asarray(n_r)
    idx_cut = int(idx_cut)

    i0 = max(0, idx_cut)
    a0, b0, k_tf = tail_parameters(mu_id, temperature)
    fit_window_mode_requested = str(fit_window_mode or "local").strip().lower()
    if fit_window_mode_requested not in ("auto", "physical", "local"):
        raise ValueError("fit_window_mode must be 'auto', 'physical', or 'local'.")
    fit_window_mode_resolved = str(fit_window_mode_requested)
    friedel_edge_ratio = float("nan")
    if r_fit_max is None:
        fit_window_mode_resolved = "local"
    elif fit_window_mode_requested == "auto":
        fit_span_requested = max(float(r_fit_max) - float(r[i0]), 0.0)
        friedel_edge_ratio = float(np.exp(-2.0 * float(b0) * fit_span_requested))
        fit_window_mode_resolved = (
            "physical" if friedel_edge_ratio >= 0.10 else "local"
        )
    local_width_active = (
        fit_window_mode_resolved == "local" and local_fit_width is not None
    )
    if fit_window_mode_resolved == "physical":
        r_fit_max_effective = float(r_fit_max)
    elif local_width_active:
        width = float(local_fit_width)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("local_fit_width must be finite and positive when supplied.")
        r_fit_max_effective = min(float(r[-1]), float(r[i0]) + width)
    else:
        r_fit_max_effective = None
    fit_indices, fit_sampling = _tail_fit_indices(
        r,
        idx_cut=i0,
        fit_points=fit_points,
        r_fit_max=r_fit_max_effective,
    )
    r_fit = r[fit_indices]
    y_fit = n_r[fit_indices] - n0

    if r_fit.size < 3:
        raise ValueError("Not enough points to fit tail parameters.")

    model_req = str(model or "auto").strip().lower()
    if model_req not in ("auto", "full", "a_only"):
        raise ValueError("tail fit model must be 'auto', 'full', or 'a_only'.")

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
    integral_row_full = (
        None
        if integral_constraint_row is None
        else np.asarray(integral_constraint_row, dtype=float).reshape(-1)
    )
    if integral_row_full is not None and integral_row_full.size != 3:
        raise ValueError("integral_constraint_row must have three entries for (A, C, D).")
    integral_target = (
        None if integral_constraint_target is None else float(integral_constraint_target)
    )
    a_only_constraint_feasible = True
    a_only_constraint_error = ""
    try:
        coeffs_a_only, y_pred_a_only, fit_rms_a_only, diag_a_only = (
            _solve_tail_fit_system(
                basis_a_only,
                y_fit,
                r=r,
                n_r=n_r,
                idx_cut=i0,
                r_fit=r_fit,
                deriv_row=np.array([d_f0], dtype=float),
                match_value_weight=match_value_weight,
                match_slope_weight=match_slope_weight,
                integral_row=(
                    None
                    if integral_row_full is None
                    else np.asarray(integral_row_full[:1], dtype=float)
                ),
                integral_target=integral_target,
            )
        )
    except ValueError as exc:
        if integral_row_full is None or model_req == "a_only":
            raise
        # The requested full model can remain well posed when the monotone
        # Yukawa column has underflowed but the Friedel columns have not.
        # Retain an unconstrained A-only fit solely as a comparison diagnostic;
        # it must not be selected by automatic model choice.
        a_only_constraint_feasible = False
        a_only_constraint_error = str(exc)
        coeffs_a_only, y_pred_a_only, fit_rms_a_only, diag_a_only = (
            _solve_tail_fit_system(
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
        )
    A_a_only = float(coeffs_a_only[0])

    basis_full = np.column_stack((f0, f1, f2))
    coeffs_full, y_pred_full, fit_rms_full, diag_full = _solve_tail_fit_system(
        basis_full,
        y_fit,
        r=r,
        n_r=n_r,
        idx_cut=i0,
        r_fit=r_fit,
        deriv_row=np.array([d_f0, d_f1, d_f2], dtype=float),
        match_value_weight=match_value_weight,
        match_slope_weight=match_slope_weight,
        integral_row=integral_row_full,
        integral_target=integral_target,
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
        use_full = bool(
            (not a_only_constraint_feasible)
            or (
                rel_improve >= float(auto_rel_improve_tol)
                and signal_large_enough
            )
        )
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
    diag_selected = diag_full if model_sel == "full" else diag_a_only
    meta = {
        "A": float(A),
        "B": float(B),
        "delta": float(delta),
        "a0": float(a0),
        "b0": float(b0),
        "k_tf": float(k_tf),
        "idx_cut": int(idx_cut),
        "fit_points": int(r_fit.size),
        "fit_method": "linear_column_scaled",
        "fit_sampling": str(fit_sampling),
        "fit_r_min": float(r_fit[0]),
        "fit_r_max": float(r_fit[-1]),
        "fit_r_span": float(r_fit[-1] - r_fit[0]),
        "fit_r_max_requested": None if r_fit_max is None else float(r_fit_max),
        "local_fit_width_requested": (
            None if local_fit_width is None else float(local_fit_width)
        ),
        "fit_window_mode_requested": str(fit_window_mode_requested),
        "fit_window_mode_resolved": str(fit_window_mode_resolved),
        "fit_window_friedel_edge_ratio": float(friedel_edge_ratio),
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
        "fit_rank": int(diag_full["rank"]),
        "fit_condition_number": float(diag_full["condition_number"]),
        "fit_condition_number_scaled": float(diag_full["condition_number_scaled"]),
        "fit_condition_number_a_only": float(diag_a_only["condition_number"]),
        "fit_condition_number_a_only_scaled": float(diag_a_only["condition_number_scaled"]),
        "a_only_integral_constraint_feasible": bool(
            a_only_constraint_feasible
        ),
        "a_only_integral_constraint_error": str(a_only_constraint_error),
        "integral_constraint_applied": bool(diag_selected["integral_constraint_applied"]),
        "integral_constraint_redundant": bool(
            diag_selected["integral_constraint_redundant"]
        ),
        "integral_constraint_target": (
            None if integral_target is None else float(integral_target)
        ),
        "integral_constraint_residual": float(diag_selected["integral_constraint_residual"]),
        "integral_constraint_residual_a_only": float(
            diag_a_only["integral_constraint_residual"]
        ),
    }
    return np.array([A, B, delta], dtype=float), meta


def apply_tail_match(r: np.ndarray,
                     n_r: np.ndarray,
                     n0: float,
                     mu_id: float,
                     temperature: float,
                     idx_cut: int,
                     fit_points: int = 12,
                     r_fit_max: float | None = None,
                     local_fit_width: float | None = None,
                     fit_window_mode: str = "local",
                     blend_points: int = 0,
                     model: str = "full",
                     auto_rel_improve_tol: float = 0.15,
                     auto_signal_rel_tol: float = 2.0e-5,
                     charge_target: float | None = None,
                     charge_constraint_fit_rms_ratio_max: float | None = 10.0,
                     charge_constraint_profile_delta_rel_max: float | None = 10.0,
                     ) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Replace n(r) for r >= R_cut with an analytic tail fit.

    Steps
    -----
    1) Fit `(A, B, delta)` from `R_cut` through `r_fit_max` (when supplied)
       with `fit_tail_params`.
    2) Build analytic tail `n_tail(r)` for `r >= R_cut`.
    3) Hard splice: replace `n_r[idx_cut:]` by `n_tail`.
    4) Optional Hermite bridge over `blend_points` to smooth the splice.
    5) If ``charge_target`` is supplied, constrain the linear B3 coefficients
       so the final, bridged finite-grid profile obeys
       ``4*pi*integral r^2 n(r) dr = charge_target`` exactly.
    6) Reject a constrained candidate when its post-Hermite density is
       non-finite/negative, its post-Hermite charge residual exceeds
       roundoff-scaled tolerance, or the constraint degrades the local fit or
       density profile beyond the optional acceptance thresholds.

    Notes
    -----
    `blend_points` applies a cubic Hermite bridge across
    `[idx_cut, idx_cut + blend_points)` so both the value and the local slope
    match at the start and end of the transition window.
    """
    r_arr = np.asarray(r, dtype=float)
    n_arr = np.asarray(n_r, dtype=float)
    idx_cut = int(idx_cut)
    blend_points = int(max(blend_points, 0))

    # The final splice (including the Hermite bridge) is affine in (A,C,D).
    # Build its constant profile and three unit responses to obtain the exact
    # charge row on this radial grid.  This avoids the old outer-shell
    # constant-density correction and does not rely on a tunable penalty.
    integral_row = None
    integral_target = None
    charge_base = float("nan")
    unconstrained_params = None
    unconstrained_meta = None
    if charge_target is not None:
        zero = np.zeros(3, dtype=float)
        base_profile = _splice_linear_tail(
            r_arr,
            n_arr,
            n0,
            mu_id,
            temperature,
            idx_cut,
            zero,
            blend_points,
        )
        charge_base = _spherical_integral(r_arr, base_profile)
        integral_row = np.empty(3, dtype=float)
        for col in range(3):
            unit = np.zeros(3, dtype=float)
            unit[col] = 1.0
            unit_response = _splice_linear_tail_response(
                r_arr,
                mu_id,
                temperature,
                idx_cut,
                unit,
                blend_points,
            )
            integral_row[col] = _spherical_integral(r_arr, unit_response)
        integral_target = float(charge_target) - float(charge_base)
        unconstrained_params, unconstrained_meta = fit_tail_params(
            r_arr,
            n_arr,
            n0,
            mu_id,
            temperature,
            idx_cut,
            fit_points=fit_points,
            r_fit_max=r_fit_max,
            local_fit_width=local_fit_width,
            fit_window_mode=fit_window_mode,
            model=model,
            auto_rel_improve_tol=auto_rel_improve_tol,
            auto_signal_rel_tol=auto_signal_rel_tol,
        )

    # Model selection must describe the *local numerical density*, not the
    # global finite-box charge correction.  In particular, an exact charge
    # equality can make the constrained A-only and full candidates have very
    # similar fit residuals even when the unconstrained A3 data clearly contain
    # a Friedel component.  Selecting ``a_only`` from those constrained
    # residuals erases the oscillatory term for a numerical, rather than a
    # physical, reason.  Resolve ``auto`` on the unconstrained local fit first,
    # then keep that basis while imposing the charge equality.
    #
    # Starrett & Saumon (2014), Appendix B, state that the B3 parameters are
    # obtained by matching the numerical density at R_cut; the finite-box
    # charge equality is an additional finite-box stabilization and must not
    # determine whether the B3 Friedel term exists.
    constrained_model = str(model)
    model_selection_basis = "requested_model"
    if (
        charge_target is not None
        and str(model or "auto").strip().lower() == "auto"
        and unconstrained_meta is not None
    ):
        constrained_model = str(
            unconstrained_meta.get("model_selected", "auto")
        )
        model_selection_basis = "unconstrained_local_fit"

    # (1) Local parameter fit around R_cut.
    params, meta = fit_tail_params(
        r_arr,
        n_arr,
        n0,
        mu_id,
        temperature,
        idx_cut,
        fit_points=fit_points,
        r_fit_max=r_fit_max,
        local_fit_width=local_fit_width,
        fit_window_mode=fit_window_mode,
        model=constrained_model,
        auto_rel_improve_tol=auto_rel_improve_tol,
        auto_signal_rel_tol=auto_signal_rel_tol,
        integral_constraint_row=integral_row,
        integral_constraint_target=integral_target,
    )
    constrained_fit_rel_improve = float(
        meta.get("fit_rel_improve_full", np.nan)
    )
    meta["model_requested"] = str(model or "auto").strip().lower()
    # ``fit_tail_params`` still performs the ordinary automatic choice when
    # no charge constraint is present.  Preserve its resolved value rather
    # than replacing it with the literal request string ``"auto"``.
    meta["model_selected"] = str(
        meta.get("model_selected", constrained_model)
    ).strip().lower()
    meta["model_selection_basis"] = str(model_selection_basis)
    meta["charge_constrained_model"] = str(constrained_model).strip().lower()
    meta["charge_constrained_fit_rel_improve_full"] = float(
        constrained_fit_rel_improve
    )
    if unconstrained_meta is not None:
        meta["model_selection_unconstrained_model"] = str(
            unconstrained_meta.get("model_selected", "")
        )
        meta["model_selection_unconstrained_fit_rel_improve_full"] = float(
            unconstrained_meta.get("fit_rel_improve_full", np.nan)
        )
        # Keep the long-standing field useful as the diagnostic explaining an
        # automatic decision.  The constrained comparison remains available
        # explicitly under ``charge_constrained_fit_rel_improve_full``.
        if str(model or "auto").strip().lower() == "auto":
            meta["fit_rel_improve_full"] = float(
                unconstrained_meta.get("fit_rel_improve_full", np.nan)
            )
    else:
        meta["model_selection_unconstrained_model"] = ""
        meta["model_selection_unconstrained_fit_rel_improve_full"] = float(
            "nan"
        )
    A, B, delta = params

    # (2-4) Build the analytic tail and optional C1 bridge.
    coeffs_acd = np.asarray(
        [float(A), float(B) * np.cos(float(delta)), float(B) * np.sin(float(delta))],
        dtype=float,
    )
    n_out = _splice_linear_tail(
        r_arr,
        n_arr,
        n0,
        mu_id,
        temperature,
        idx_cut,
        coeffs_acd,
        blend_points,
    )

    meta["blend_points"] = int(min(blend_points, max(r_arr.size - idx_cut, 0)))
    meta["blend_mode"] = "hermite" if blend_points > 0 else "off"
    meta["charge_constraint_applied"] = bool(charge_target is not None)
    meta["charge_constraint_target"] = (
        None if charge_target is None else float(charge_target)
    )
    meta["charge_constraint_base"] = float(charge_base)
    meta["charge_constraint_achieved"] = _spherical_integral(r_arr, n_out)
    meta["charge_constraint_residual"] = (
        float("nan")
        if charge_target is None
        else float(meta["charge_constraint_achieved"] - float(charge_target))
    )
    meta["charge_constraint_row_A"] = (
        float("nan") if integral_row is None else float(integral_row[0])
    )
    meta["charge_constraint_row_C"] = (
        float("nan") if integral_row is None else float(integral_row[1])
    )
    meta["charge_constraint_row_D"] = (
        float("nan") if integral_row is None else float(integral_row[2])
    )
    if unconstrained_params is not None and unconstrained_meta is not None:
        A0, B0, delta0 = [float(value) for value in unconstrained_params]
        coeffs_unconstrained = np.asarray(
            [A0, B0 * np.cos(delta0), B0 * np.sin(delta0)],
            dtype=float,
        )
        unconstrained_profile = _splice_linear_tail(
            r_arr,
            n_arr,
            n0,
            mu_id,
            temperature,
            idx_cut,
            coeffs_unconstrained,
            blend_points,
        )
        charge_unconstrained = _spherical_integral(r_arr, unconstrained_profile)
        coeff_delta = coeffs_acd - coeffs_unconstrained
        meta["charge_constraint_unconstrained_achieved"] = float(charge_unconstrained)
        meta["charge_constraint_unconstrained_residual"] = float(
            charge_unconstrained - float(charge_target)
        )
        meta["charge_constraint_unconstrained_fit_rms"] = float(
            unconstrained_meta["fit_rms"]
        )
        meta["charge_constraint_fit_rms_ratio"] = float(
            float(meta["fit_rms"]) / max(float(unconstrained_meta["fit_rms"]), 1.0e-30)
        )
        meta["charge_constraint_coeff_delta_norm"] = float(np.linalg.norm(coeff_delta))
        meta["charge_constraint_coeff_delta_rel"] = float(
            np.linalg.norm(coeff_delta)
            / max(float(np.linalg.norm(coeffs_unconstrained)), 1.0e-30)
        )
        profile_delta = np.asarray(
            n_out - unconstrained_profile,
            dtype=float,
        )
        unconstrained_response = np.asarray(
            unconstrained_profile - base_profile,
            dtype=float,
        )
        profile_delta_norm = _spherical_l2_norm(r_arr, profile_delta)
        unconstrained_response_norm = _spherical_l2_norm(
            r_arr, unconstrained_response
        )
        meta["charge_constraint_profile_delta_norm"] = float(
            profile_delta_norm
        )
        meta["charge_constraint_unconstrained_response_norm"] = float(
            unconstrained_response_norm
        )
        meta["charge_constraint_profile_delta_rel"] = float(
            profile_delta_norm / max(unconstrained_response_norm, 1.0e-30)
        )
        meta["charge_constraint_unconstrained_A"] = float(A0)
        meta["charge_constraint_unconstrained_B"] = float(B0)
        meta["charge_constraint_unconstrained_delta"] = float(delta0)
    else:
        meta["charge_constraint_unconstrained_achieved"] = float("nan")
        meta["charge_constraint_unconstrained_residual"] = float("nan")
        meta["charge_constraint_unconstrained_fit_rms"] = float("nan")
        meta["charge_constraint_fit_rms_ratio"] = float("nan")
        meta["charge_constraint_coeff_delta_norm"] = float("nan")
        meta["charge_constraint_coeff_delta_rel"] = float("nan")
        meta["charge_constraint_profile_delta_norm"] = float("nan")
        meta["charge_constraint_unconstrained_response_norm"] = float("nan")
        meta["charge_constraint_profile_delta_rel"] = float("nan")
        meta["charge_constraint_unconstrained_A"] = float("nan")
        meta["charge_constraint_unconstrained_B"] = float("nan")
        meta["charge_constraint_unconstrained_delta"] = float("nan")

    # A charge equality alone does not make an extrapolated density physical:
    # an ill-conditioned fit can satisfy the integral with a negative tail or
    # with coefficients far outside the local unconstrained B3 solution.
    # Perform this check *after* the Hermite bridge, because that is the actual
    # density subsequently used by Poisson and by the pseudoatom construction.
    meta["charge_constraint_acceptance_checked"] = bool(charge_target is not None)
    meta["charge_constraint_fit_rms_ratio_max"] = (
        None
        if charge_constraint_fit_rms_ratio_max is None
        else float(charge_constraint_fit_rms_ratio_max)
    )
    meta["charge_constraint_profile_delta_rel_max"] = (
        None
        if charge_constraint_profile_delta_rel_max is None
        else float(charge_constraint_profile_delta_rel_max)
    )
    if charge_target is not None:
        tail_values = np.asarray(n_out[idx_cut:], dtype=float)
        finite_tail = bool(np.all(np.isfinite(tail_values)))
        tail_min = float(np.min(tail_values)) if tail_values.size else float("nan")
        # Permit only roundoff-scale undershoot.  This tolerance scales with
        # the local/background density and is intentionally not a tunable way
        # to admit a genuinely negative analytic tail.
        negative_tol = max(
            1.0e-14,
            1.0e-10
            * max(
                abs(float(n0)),
                float(np.max(np.abs(tail_values))) if tail_values.size and finite_tail else 0.0,
            ),
        )
        meta["charge_constraint_tail_finite"] = bool(finite_tail)
        meta["charge_constraint_tail_min"] = float(tail_min)
        meta["charge_constraint_negative_tol"] = float(negative_tol)
        if not finite_tail:
            raise ValueError(
                "Charge-constrained B3 tail rejected: the post-Hermite tail "
                "contains a non-finite density."
            )
        if tail_min < -negative_tol:
            raise ValueError(
                "Charge-constrained B3 tail rejected: post-Hermite minimum "
                f"density {tail_min:.6e} is below -{negative_tol:.6e}."
            )
        if tail_min < 0.0:
            # Remove a roundoff-only negative value, then re-evaluate the
            # finite-grid charge residual below.
            n_out = np.asarray(n_out, dtype=float).copy()
            n_out[idx_cut:] = np.maximum(n_out[idx_cut:], 0.0)
            meta["charge_constraint_roundoff_clip_applied"] = True
            meta["charge_constraint_tail_min"] = float(
                np.min(n_out[idx_cut:])
            )
        else:
            meta["charge_constraint_roundoff_clip_applied"] = False

        achieved_post = _spherical_integral(r_arr, n_out)
        residual_post = float(achieved_post - float(charge_target))
        residual_tol = max(1.0e-10, 1.0e-10 * abs(float(charge_target)))
        meta["charge_constraint_achieved"] = float(achieved_post)
        meta["charge_constraint_residual"] = float(residual_post)
        meta["charge_constraint_residual_tol"] = float(residual_tol)
        if not np.isfinite(residual_post) or abs(residual_post) > residual_tol:
            raise ValueError(
                "Charge-constrained B3 tail rejected: post-Hermite charge "
                f"residual {residual_post:.6e} exceeds {residual_tol:.6e}."
            )

        fit_ratio = float(meta["charge_constraint_fit_rms_ratio"])
        profile_delta_rel = float(
            meta["charge_constraint_profile_delta_rel"]
        )
        if charge_constraint_fit_rms_ratio_max is not None:
            fit_ratio_max = float(charge_constraint_fit_rms_ratio_max)
            if not np.isfinite(fit_ratio_max) or fit_ratio_max <= 0.0:
                raise ValueError(
                    "charge_constraint_fit_rms_ratio_max must be finite and "
                    "positive, or None."
                )
            if not np.isfinite(fit_ratio) or fit_ratio > fit_ratio_max:
                raise ValueError(
                    "Charge-constrained B3 tail rejected: fit RMS ratio "
                    f"{fit_ratio:.6e} exceeds {fit_ratio_max:.6e}."
                )
        if charge_constraint_profile_delta_rel_max is not None:
            profile_delta_rel_max = float(
                charge_constraint_profile_delta_rel_max
            )
            if (
                not np.isfinite(profile_delta_rel_max)
                or profile_delta_rel_max <= 0.0
            ):
                raise ValueError(
                    "charge_constraint_profile_delta_rel_max must be finite and "
                    "positive, or None."
                )
            if (
                not np.isfinite(profile_delta_rel)
                or profile_delta_rel > profile_delta_rel_max
            ):
                raise ValueError(
                    "Charge-constrained B3 tail rejected: relative density-"
                    f"profile change {profile_delta_rel:.6e} exceeds "
                    f"{profile_delta_rel_max:.6e}."
                )
        meta["charge_constraint_accepted"] = True
    else:
        meta["charge_constraint_tail_finite"] = None
        meta["charge_constraint_tail_min"] = float("nan")
        meta["charge_constraint_negative_tol"] = float("nan")
        meta["charge_constraint_roundoff_clip_applied"] = False
        meta["charge_constraint_residual_tol"] = float("nan")
        meta["charge_constraint_accepted"] = False
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
    charge = _trapz(integrand, r_tail)
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
                    r_fit_max: float | None = None,
                    fit_window_mode: str = "local",
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
    r_fit_max
        Optional physical right edge shared by the candidate fit windows.
    fit_window_mode
        ``local``, ``physical``, or decay-aware ``auto`` sample placement.
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
            r_fit_max=r_fit_max,
            fit_window_mode=fit_window_mode,
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
        "r_fit_max": None if r_fit_max is None else float(r_fit_max),
        "fit_window_mode": str(fit_window_mode),
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
