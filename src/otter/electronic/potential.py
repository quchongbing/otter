"""
otter/electronic/potential.py

Purpose
-------
Assemble effective electron potentials for the AA full/external systems.

Methods
-------
- Combine nucleus term, Hartree term, and XC difference.
- Use g_II(r) to include the ion-background subtraction term.

Equations
---------
Full system (Starrett2014 Eq. 4):
  V_eff(r) = -Z/r + integral (n_full - n0 g_II) / |r-r'| + V_xc[n_full] - V_xc[n0]

External system (Starrett2014 Eq. 7):
  V_eff_ext(r) = integral (n_ext - n0 g_II) / |r-r'| + V_xc[n_ext] - V_xc[n0]

References
----------
- C. E. Starrett & D. Saumon (2014), Eqs. (4) and (7).
"""
from __future__ import annotations

import numpy as np

from otter.electronic.xc import lda_xc_potential


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral with values on one monotonic 1D grid."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("y and x must be 1D arrays.")
    if y.size != x.size:
        raise ValueError("y and x must have the same length.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")

    out = np.zeros_like(y, dtype=float)
    if y.size == 1:
        return out

    dx = np.diff(x)
    avg = 0.5 * (y[1:] + y[:-1])
    out[1:] = np.cumsum(avg * dx)
    return out


def spherical_hartree_potential(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Compute the spherical Hartree potential from a radial charge density.

    The expression is

        V_H(r) = 4 pi [ 1/r int_0^r rho(r') r'^2 dr'
                        + int_r^R rho(r') r' dr' ].
    """
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if r.ndim != 1 or rho.ndim != 1:
        raise ValueError("r and rho must be 1D arrays.")
    if r.size != rho.size:
        raise ValueError("r and rho must have the same length.")
    if np.any(r <= 0):
        raise ValueError("r must be > 0 for spherical potential.")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r must be strictly increasing.")

    int_f = _cumtrapz(rho * r**2, r)
    int_g = _cumtrapz(rho * r, r)
    return 4.0 * np.pi * (int_f / r + (int_g[-1] - int_g))


def spherical_hartree_potential_screened(
    r: np.ndarray,
    rho: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Compute a screened Poisson-Helmholtz Hartree potential."""
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if kappa <= 0.0:
        return spherical_hartree_potential(r, rho)

    if r.ndim != 1 or rho.ndim != 1:
        raise ValueError("r and rho must be 1D arrays.")
    if r.size != rho.size:
        raise ValueError("r and rho must have the same length.")
    if np.any(r <= 0):
        raise ValueError("r must be > 0 for spherical potential.")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r must be strictly increasing.")

    kr = kappa * r
    kr_clip = np.clip(kr, 0.0, 50.0)
    exp_pkr = np.exp(kr_clip)
    exp_mkr = np.exp(-kr)
    sinh_kr = 0.5 * (exp_pkr - exp_mkr)

    inner = _cumtrapz(rho * r * sinh_kr, r)
    outer = _cumtrapz(rho * r * exp_mkr, r)
    return (4.0 * np.pi / (kappa * r)) * (
        exp_mkr * inner + sinh_kr * (outer[-1] - outer)
    )


def _as_array_like(x: float | np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Broadcast a scalar or validate an array against ref.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full_like(ref, float(arr), dtype=float)
    if arr.shape != ref.shape:
        raise ValueError("Input array must match reference shape.")
    return arr


def effective_potential_full(r: np.ndarray,
                             n_full: np.ndarray,
                             n0: float | np.ndarray,
                             g_ii: np.ndarray,
                             Z: float,
                             xc_model: str = "dirac",
                             kappa: float = 0.0) -> np.ndarray:
    """
    Effective potential for the full AA system (with nucleus).

    Parameters
    ----------
    r : ndarray
        Radial grid (Bohr).
    n_full : ndarray
        Full electron density (Bohr^-3).
    n0 : float or ndarray
        Field-free electron density (Bohr^-3).
    g_ii : ndarray
        Ion-ion pair distribution function.
    Z : float
        Nuclear charge.
    xc_model : str
        XC model name for LDA potential.
    kappa : float
        Poisson-Helmholtz screening parameter (Bohr^-1).

    Returns
    -------
    ndarray
        Effective potential V_eff(r) (Ha).
    """
    r = np.asarray(r, dtype=float)
    n_full = np.asarray(n_full, dtype=float)
    g_ii = np.asarray(g_ii, dtype=float)

    if r.ndim != 1:
        raise ValueError("r must be 1D.")
    if n_full.shape != r.shape or g_ii.shape != r.shape:
        raise ValueError("n_full and g_ii must match r shape.")

    n0_arr = _as_array_like(n0, r)
    rho = n_full - n0_arr * g_ii

    if kappa > 0.0:
        v_h = spherical_hartree_potential_screened(r, rho, kappa)
    else:
        v_h = spherical_hartree_potential(r, rho)
    v_xc = lda_xc_potential(n_full, model=xc_model) - lda_xc_potential(n0_arr, model=xc_model)

    return -Z / r + v_h + v_xc


def effective_potential_external(r: np.ndarray,
                                 n_ext: np.ndarray,
                                 n0: float | np.ndarray,
                                 g_ii: np.ndarray,
                                 xc_model: str = "dirac",
                                 kappa: float = 0.0) -> np.ndarray:
    """
    Effective potential for the external AA system (no nucleus).

    Parameters
    ----------
    r : ndarray
        Radial grid (Bohr).
    n_ext : ndarray
        External electron density (Bohr^-3).
    n0 : float or ndarray
        Field-free electron density (Bohr^-3).
    g_ii : ndarray
        Ion-ion pair distribution function.
    xc_model : str
        XC model name for LDA potential.
    kappa : float
        Poisson-Helmholtz screening parameter (Bohr^-1).

    Returns
    -------
    ndarray
        Effective potential V_eff_ext(r) (Ha).
    """
    r = np.asarray(r, dtype=float)
    n_ext = np.asarray(n_ext, dtype=float)
    g_ii = np.asarray(g_ii, dtype=float)

    if r.ndim != 1:
        raise ValueError("r must be 1D.")
    if n_ext.shape != r.shape or g_ii.shape != r.shape:
        raise ValueError("n_ext and g_ii must match r shape.")

    n0_arr = _as_array_like(n0, r)
    rho = n_ext - n0_arr * g_ii

    if kappa > 0.0:
        v_h = spherical_hartree_potential_screened(r, rho, kappa)
    else:
        v_h = spherical_hartree_potential(r, rho)
    v_xc = lda_xc_potential(n_ext, model=xc_model) - lda_xc_potential(n0_arr, model=xc_model)

    return v_h + v_xc
