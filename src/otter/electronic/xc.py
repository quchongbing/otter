"""
otter/electronic/xc.py

Purpose
-------
Provide exchange-correlation (XC) models for electron density functional theory.

Methods
-------
- Dirac exchange (LDA, spin-unpolarized).
- Placeholder for correlation models (future extension).

Equations
---------
Dirac exchange energy density:
  e_x(n) = - (3/4) * (3/pi)^(1/3) * n^(4/3)
Exchange potential:
  v_x(n) = d e_x / d n = - (3/pi)^(1/3) * n^(1/3)

References
----------
- P. A. M. Dirac (1930) exchange for uniform electron gas.
"""
import numpy as np


def dirac_exchange_energy_density(n: np.ndarray) -> np.ndarray:
    """
    Dirac exchange energy density e_x(n) for spin-unpolarized electrons.

    Parameters
    ----------
    n : ndarray
        Electron density (Bohr^-3).

    Returns
    -------
    ndarray
        Exchange energy density (Ha * Bohr^-3).
    """
    n = np.asarray(n, dtype=float)
    n = np.where(np.isfinite(n), n, 0.0)
    n = np.clip(n, 0.0, None)
    pref = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    return pref * np.power(n, 4.0 / 3.0)


def dirac_exchange_potential(n: np.ndarray) -> np.ndarray:
    """
    Dirac exchange potential v_x(n).

    Parameters
    ----------
    n : ndarray
        Electron density (Bohr^-3).

    Returns
    -------
    ndarray
        Exchange potential (Ha).
    """
    n = np.asarray(n, dtype=float)
    n = np.where(np.isfinite(n), n, 0.0)
    n = np.clip(n, 0.0, None)
    pref = - (3.0 / np.pi) ** (1.0 / 3.0)
    return pref * np.power(n, 1.0 / 3.0)


def lda_xc_potential(n: np.ndarray, model: str = "dirac") -> np.ndarray:
    """
    LDA XC potential dispatcher.

    Parameters
    ----------
    n : ndarray
        Electron density (Bohr^-3).
    model : str
        XC model name ("dirac" for exchange only).

    Returns
    -------
    ndarray
        XC potential (Ha).
    """
    model = model.lower()
    if model == "dirac":
        return dirac_exchange_potential(n)
    raise ValueError(f"Unknown XC model: {model}")
