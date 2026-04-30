"""Bound-state density and ion-partition helpers for average-atom solves."""
from __future__ import annotations

import math

import numpy as np

from otter.electronic.continuum.scattering import fermi_dirac


def _trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def _bound_density(r: np.ndarray,
                   eigvals: np.ndarray,
                   eigvecs: np.ndarray,
                   l_list: np.ndarray,
                   mu: float,
                   temperature: float,
                   energy_cut: float = 0.0,
                   occ_mode: str = "fd",
                   gamma: float = 0.0,
                   r_ws: float | None = None,
                   ws_weight_min: float = 0.0) -> np.ndarray:
    """
    Compute bound density from eigenpairs using Eq. (A2).

    Parameters
    ----------
    r : ndarray
        Radial grid.
    eigvals : ndarray
        Eigenvalues with shape (n_l, n_states).
    eigvecs : ndarray
        Eigenvectors with shape (n_l, n_r, n_states), y = sqrt(r) * R.
    l_list : ndarray
        Angular momenta matching eigvals/eigvecs.
    mu : float
        Chemical potential (Ha).
    temperature : float
        Temperature (Ha).
    energy_cut : float
        Energy threshold for bound-state inclusion (default 0.0 Ha).
    occ_mode : str
        Occupation mode: "fd" or "fd_m" (FD times M(e) broadening weight).
    gamma : float
        Broadening FWHM (Ha) used when occ_mode="fd_m".
    r_ws : float or None
        Ion-sphere radius (Bohr). Required when ws_weight_min > 0.
    ws_weight_min : float
        Minimum WS localization weight to include a bound state.

    Returns
    -------
    ndarray
        Bound electron density on r.
    """
    r = np.asarray(r, dtype=float)
    n_bound = np.zeros_like(r)
    l_list = np.asarray(l_list, dtype=int)

    occ_mode = str(occ_mode).lower().strip()
    if occ_mode not in ("fd", "fd_m"):
        raise ValueError("occ_mode must be 'fd' or 'fd_m'.")
    ws_weight_min = float(ws_weight_min)
    if ws_weight_min < 0.0 or ws_weight_min > 1.0:
        raise ValueError("ws_weight_min must be in [0, 1].")
    if ws_weight_min > 0.0 and r_ws is None:
        raise ValueError("r_ws is required when ws_weight_min > 0.")
    r_safe = np.maximum(r, 1e-14)
    for l_idx, l_val in enumerate(l_list):
        for s_idx in range(eigvals.shape[1]):
            e_nl = float(eigvals[l_idx, s_idx])
            if e_nl >= energy_cut:
                continue
            occ = float(fermi_dirac(np.array([e_nl]), mu, temperature)[0])
            if occ_mode == "fd_m":
                occ *= _ion_level_weight(e_nl, gamma)
            if occ <= 0.0:
                continue
            y = eigvecs[l_idx, :, s_idx]
            R = y / np.sqrt(r_safe)
            factor = (2.0 * (2 * l_val + 1)) / (4.0 * np.pi)
            if ws_weight_min > 0.0:
                density = (np.abs(R) ** 2) * (r ** 2)
                denom = _trapz(density, r)
                if denom <= 0.0:
                    continue
                mask_ws = r <= float(r_ws)
                numer = _trapz(density[mask_ws], r[mask_ws])
                ws_weight = numer / denom
                if ws_weight < ws_weight_min:
                    continue
            n_bound += occ * factor * (np.abs(R) ** 2)

    return n_bound

def _ion_level_weight(energy: float, gamma: float) -> float:
    """
    Return M(e) weight for bound states using Starrett2013 Eq. (81).

    Parameters
    ----------
    energy : float
        Orbital energy (Ha).
    gamma : float
        Broadening FWHM (Ha). If <= 0, returns 1.0 for bound states.

    Returns
    -------
    float
        Weight M(e) in [0, 1].
    """
    if gamma <= 0.0:
        return 1.0
    arg = -2.0 * math.sqrt(math.log(2.0)) * energy / gamma
    m_val = math.erf(arg)
    if m_val < 0.0:
        return 0.0
    if m_val > 1.0:
        return 1.0
    return float(m_val)

def _ion_cutoff_starrett(r: np.ndarray, r_ws: float, c: float) -> np.ndarray:
    """
    Starrett2013 cutoff function f_cut(r) (Eq. 80).

    Parameters
    ----------
    r : ndarray
        Radial grid (Bohr).
    r_ws : float
        Ion-sphere radius (Bohr).
    c : float
        Dimensionless cutoff width parameter.

    Returns
    -------
    ndarray
        f_cut(r) in [0, 1].
    """
    r = np.asarray(r, dtype=float)
    c = float(c)
    if c <= 0.0:
        return (r <= r_ws).astype(float)
    numerator = 1.0 + math.exp(-1.0 / c)
    denom = 1.0 + np.exp((r - r_ws) / (c * r_ws))
    return numerator / denom

def _ion_density(r: np.ndarray,
                 eigvals: np.ndarray,
                 eigvecs: np.ndarray,
                 l_list: np.ndarray,
                 mu: float,
                 temperature: float,
                 energy_cut: float,
                 gamma: float,
                 cutoff: np.ndarray | None = None,
                 r_ws: float | None = None,
                 ws_weight_min: float = 0.0) -> np.ndarray:
    """
    Compute n_ion from bound eigenpairs with M(e) and cutoff.

    Parameters
    ----------
    r : ndarray
        Radial grid (Bohr).
    eigvals : ndarray
        Eigenvalues with shape (n_l, n_states).
    eigvecs : ndarray
        Eigenvectors with shape (n_l, n_r, n_states), y = sqrt(r) * R.
    l_list : ndarray
        Angular momenta matching eigvals/eigvecs.
    mu : float
        Chemical potential (Ha).
    temperature : float
        Temperature (Ha).
    energy_cut : float
        Energy threshold for bound-state inclusion.
    gamma : float
        Broadening FWHM (Ha) for M(e).
    cutoff : ndarray or None
        Radial cutoff function f_cut(r) in [0, 1].

    Returns
    -------
    ndarray
        Ion electron density n_ion(r).
    """
    r = np.asarray(r, dtype=float)
    n_ion = np.zeros_like(r)
    r_safe = np.maximum(r, 1e-14)
    l_list = np.asarray(l_list, dtype=int)
    ws_weight_min = float(ws_weight_min)
    if ws_weight_min < 0.0 or ws_weight_min > 1.0:
        raise ValueError("ion_ws_weight_min must be in [0, 1].")
    if ws_weight_min > 0.0 and r_ws is None:
        raise ValueError("r_ws is required when ion_ws_weight_min > 0.")

    for l_idx, l_val in enumerate(l_list):
        for s_idx in range(eigvals.shape[1]):
            e_nl = float(eigvals[l_idx, s_idx])
            if e_nl >= energy_cut:
                continue
            occ = float(fermi_dirac(np.array([e_nl]), mu, temperature)[0])
            if occ <= 0.0:
                continue
            m_val = _ion_level_weight(e_nl, gamma)
            if m_val <= 0.0:
                continue
            y = eigvecs[l_idx, :, s_idx]
            R = y / np.sqrt(r_safe)
            factor = (2.0 * (2 * l_val + 1)) / (4.0 * np.pi)
            if ws_weight_min > 0.0:
                density = (np.abs(R) ** 2) * (r ** 2)
                denom = _trapz(density, r)
                if denom <= 0.0:
                    continue
                mask_ws = r <= float(r_ws)
                numer = _trapz(density[mask_ws], r[mask_ws])
                ws_weight = numer / denom
                if ws_weight < ws_weight_min:
                    continue
            n_ion += (occ * m_val) * factor * (np.abs(R) ** 2)

    if cutoff is not None:
        n_ion = n_ion * cutoff

    return n_ion

def _electron_count(r: np.ndarray, n: np.ndarray, r_ws: float) -> float:
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)
    mask = r <= r_ws
    if not np.any(mask):
        return 0.0
    integrand = n[mask] * r[mask] ** 2
    return float(4.0 * np.pi * _trapz(integrand, r[mask]))

def _electron_count_full(r: np.ndarray, n: np.ndarray) -> float:
    """
    Integrate electron density over the full radial grid.

    Parameters
    ----------
    r : ndarray
        Radial grid (Bohr).
    n : ndarray
        Electron density (Bohr^-3).

    Returns
    -------
    float
        Total electron count on the finite grid.
    """
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)
    integrand = n * r ** 2
    return float(4.0 * np.pi * _trapz(integrand, r))

def _ion_cutoff(r: np.ndarray,
                r_ws: float,
                width: float,
                mode: str = "starrett",
                c: float = 0.05) -> np.ndarray:
    """
    Cutoff f_cut(r) for defining n_ion from bound states.

    Notes
    -----
    - mode="starrett": Eq. (80) with parameter c.
    - mode="smoothstep": C1 smoothstep from 1 (r<=R_ws) to 0 (r>=R_ws+width).
    - mode="none": returns 1 everywhere.
    """
    r = np.asarray(r, dtype=float)
    mode = str(mode).lower().strip()
    if mode == "none":
        return np.ones_like(r)
    if mode == "smoothstep":
        if width <= 0.0:
            return (r <= r_ws).astype(float)
        x = (r - r_ws) / width
        f = np.ones_like(r)
        mask = x >= 0.0
        if np.any(mask):
            x_clip = np.clip(x[mask], 0.0, 1.0)
            smooth = 1.0 - (3.0 * x_clip ** 2 - 2.0 * x_clip ** 3)
            f[mask] = smooth
        return f
    return _ion_cutoff_starrett(r, r_ws, c)
