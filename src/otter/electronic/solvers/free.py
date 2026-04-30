"""
otter.electronic.solvers.free

Free-state Numerov propagation kernels used by the quantum continuum solver.

The project requires numba, so these are the production kernels rather than
optional accelerators.
"""
from __future__ import annotations

import numpy as np
from numba import njit


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
    Psi[0] = r[0] ** power
    if Psi[0] < 1e-300:
        Psi[0] = 1e-300
    Psi[1] = r[1] ** power
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
    return {
        "r": r_safe,
        "v_eff": v_eff,
        "r_quarter": np.sqrt(np.sqrt(r_safe)),
        "inv_r": 1.0 / r_safe,
        "inv_r2": 1.0 / (r_safe * r_safe),
        "r8": 8.0 * r_safe,
        "v_term": -8.0 * r_safe * v_eff,
    }


@njit(cache=True, fastmath=True)
def _numerov_propagate_sqrt_wbase_inplace_numba(r: np.ndarray,
                                                r_quarter: np.ndarray,
                                                inv_r: np.ndarray,
                                                w_base: np.ndarray,
                                                l: int,
                                                dxi: float,
                                                out: np.ndarray,
                                                rescale_limit: float = 1e6) -> None:
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
    out[0] = r[0] ** power
    if out[0] < 1e-300:
        out[0] = 1e-300
    out[1] = r[1] ** power
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
                                        rescale_limit: float = 1e6) -> np.ndarray:
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
    )
    return out


@njit(cache=True, fastmath=True)
def _numerov_propagate_sqrt_wbase_batch_numba(r: np.ndarray,
                                              r_quarter: np.ndarray,
                                              inv_r: np.ndarray,
                                              w_base: np.ndarray,
                                              l_vals: np.ndarray,
                                              dxi: float,
                                              rescale_limit: float = 1e6) -> np.ndarray:
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
        psi[0, j] = r[0] ** power
        if psi[0, j] < 1e-300:
            psi[0, j] = 1e-300
        psi[1, j] = r[1] ** power
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
