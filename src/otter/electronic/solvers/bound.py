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
- M. Pillai et al., AJP 80, 1017 (2012).
- C. E. Starrett & D. Saumon (2014), Appendix A/B.
"""
from typing import Tuple, Optional
import warnings
import numpy as np

try:
    from scipy import sparse as sp
    from scipy.sparse.linalg import eigs
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    sp = None
    eigs = None


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
                              fit_stop: int = 10) -> np.ndarray:
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

    Returns
    -------
    ndarray
        Regularized copy of ``y``.

    Notes
    -----
    The physical regular solution obeys

    ``y(r) = r^(l + 1/2) [a0 + a1 r + O(r^2)]``.

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
                           sigma_guess: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
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
        y[:, i] = _regularize_origin_series(y[:, i], grid_r, l)
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
                                      n_jobs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
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
