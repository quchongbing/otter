"""
otter/ionic/correlation.py

Purpose
-------
Provide ion-ion correlation models g_II(r) for AA/PAMD workflows.

Methods
-------
- Ion-sphere (IS) step model: g_II(r) = Theta(r - R_ws).
- Interface placeholder for future self-consistent QOZ updates.

Equations
---------
Ion-sphere model:
  g_II(r) = 0 (r < R_ws), 1 (r >= R_ws)

Wigner-Seitz radius:
  R_ws = (3 / (4*pi*n_I^0))^(1/3)

References
----------
- C. E. Starrett & D. Saumon (2014), Eq. (1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


def ion_sphere_radius_from_density(n_i: float) -> float:
    """
    Compute the Wigner-Seitz radius from ion number density.

    Parameters
    ----------
    n_i : float
        Ion number density (Bohr^-3).

    Returns
    -------
    float
        Wigner-Seitz radius R_ws (Bohr).
    """
    if n_i <= 0.0:
        raise ValueError("n_i must be positive.")
    return (3.0 / (4.0 * np.pi * n_i)) ** (1.0 / 3.0)


class IonCorrelationModel:
    """
    Base interface for ion-ion correlation models g_II(r).
    """

    def g_ii(self, r: np.ndarray) -> np.ndarray:
        """
        Evaluate g_II(r).

        Parameters
        ----------
        r : ndarray
            Radial grid (Bohr).

        Returns
        -------
        ndarray
            Pair distribution g_II(r).
        """
        raise NotImplementedError("g_ii must be implemented by subclasses.")

    def update_from_screening(self,
                              r: np.ndarray,
                              n_scr: np.ndarray,
                              params: Dict[str, Any] | None = None) -> None:
        """
        Optional hook for self-consistent updates using screening density.

        Parameters
        ----------
        r : ndarray
            Radial grid (Bohr).
        n_scr : ndarray
            Screening density (Bohr^-3).
        params : dict
            Optional update parameters.
        """
        return None


@dataclass
class IonSphereStepModel(IonCorrelationModel):
    """
    Ion-sphere step model for g_II(r).

    Parameters
    ----------
    r_ws : float
        Wigner-Seitz radius (Bohr).
    """
    r_ws: float

    def g_ii(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        if np.any(r < 0.0):
            raise ValueError("r must be non-negative.")
        return np.where(r >= self.r_ws, 1.0, 0.0)
