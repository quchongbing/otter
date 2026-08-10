"""
otter/electronic/continuum/interface.py

Purpose
-------
Define interfaces for continuum (unbound) electron models.

Methods
-------
- Quantum continuum: energy integral + partial-wave sum (Starrett & Saumon 2014).
- Approximate models: ideal and bands/DOS.

Equations
---------
Continuum density (QM form):
  n_c(r) = ∫_0^∞ dε g_k Σ_l [ 2(2l+1)/(4π) ] | y_{k,l}(r) / r |^2
  (A3 is an energy integral, so y_{k,l} uses δ(E) normalization.)

References
----------
- :cite:`StarrettSaumon2014`, Eq. (A3). Energy quadrature and partial-wave
  truncation controls are Otter implementation choices.
"""
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class ContinuumModel(ABC):
    """
    Abstract interface for continuum electron models.
    """

    @abstractmethod
    def density(self,
                r: np.ndarray,
                mu: float,
                temperature: float,
                params: Dict[str, float] | None = None) -> np.ndarray:
        """
        Return continuum electron density on the grid.

        Parameters
        ----------
        r : ndarray
            Radial grid (Bohr).
        mu : float
            Chemical potential (Ha).
        temperature : float
            Temperature (Ha).
        params : dict or None
            Model-specific parameters.

        Returns
        -------
        ndarray
            Continuum density n_c(r).
        """
        raise NotImplementedError
