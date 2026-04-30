"""
otter/electronic/continuum/hybrid.py

Purpose
-------
Provide a hybrid continuum model: quantum scattering at low energies
plus an ideal-gas high-energy tail.

Methods
-------
- Compute low-energy continuum with Numerov scattering (A3).
- Add a uniform high-energy tail using the ideal free-electron DOS.
- Default n0_mode uses an integral-based density for consistency.

Equations
---------
Ideal DOS contribution (atomic units):
  n = (sqrt(2)/pi^2) * integral_{e0}^{e1} e^(1/2) f(e; mu, T) de

Hybrid density:
  n_c(r) = n_low(r; e in [e_min, e_cut]) + n_high (uniform)

References
----------
- C. E. Starrett & D. Saumon (2014), Appendix A/B.
"""
from __future__ import annotations

from typing import Dict
import numpy as np

from .interface import ContinuumModel
from .scattering import fermi_dirac, QuantumContinuumScattering
from .ideal import ideal_unbound_density


def _trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def ideal_density_range(mu: float,
                         temperature: float,
                         e_min: float,
                         e_max: float,
                         n_samples: int = 800) -> float:
    """
    Integrate the ideal free-electron density over a finite energy range.

    Parameters
    ----------
    mu : float
        Chemical potential (Ha).
    temperature : float
        Temperature (Ha).
    e_min : float
        Lower energy bound (Ha).
    e_max : float
        Upper energy bound (Ha).
    n_samples : int
        Number of energy samples.

    Returns
    -------
    float
        Density contribution over [e_min, e_max] (Bohr^-3).
    """
    if e_max <= e_min:
        return 0.0
    e_min = max(float(e_min), 0.0)
    e_max = float(e_max)
    n_samples = max(int(n_samples), 16)

    e_grid = np.linspace(e_min, e_max, n_samples)
    occ = fermi_dirac(e_grid, mu, temperature)
    integrand = np.sqrt(np.maximum(e_grid, 0.0)) * occ
    pref = np.sqrt(2.0) / (np.pi ** 2)
    return float(pref * _trapz(integrand, e_grid))


class QuantumContinuumHybrid(ContinuumModel):
    """
    Hybrid continuum: scattering at low energy + ideal high-energy tail.
    """

    def density(self,
                r: np.ndarray,
                mu: float,
                temperature: float,
                params: Dict[str, float] | None = None) -> np.ndarray:
        params = params or {}

        v_eff = params.get("v_eff", None)
        if v_eff is None:
            raise ValueError("params['v_eff'] is required for hybrid continuum.")

        e_min = float(params.get("e_min", 1e-6))
        e_cut = float(params.get("e_cut", max(mu + 4.0 * temperature, 1.0)))
        e_high_max = float(params.get("e_high_max", max(e_cut + 20.0 * temperature, mu + 30.0 * temperature)))
        n_e_low = int(params.get("n_e_low", 120))
        n_e_high = int(params.get("n_e_high", 400))

        n0_mode = str(params.get("n0_mode", "integral")).lower()
        if n0_mode not in ("ideal", "integral"):
            raise ValueError("n0_mode must be 'ideal' or 'integral'.")

        scatter = QuantumContinuumScattering()
        low_params = dict(params)
        low_params["e_max"] = e_cut
        low_params["n_e"] = n_e_low
        n_low = scatter.density(r, mu, temperature, params=low_params)

        if n0_mode == "ideal":
            n0_total = ideal_unbound_density(mu, temperature)
            n0_low = ideal_density_range(mu, temperature, e_min, e_cut, n_samples=n_e_high)
            n_high = max(n0_total - n0_low, 0.0)
        else:
            n0_total = ideal_density_range(mu, temperature, 0.0, e_high_max, n_samples=n_e_high)
            n_high = ideal_density_range(mu, temperature, e_cut, e_high_max, n_samples=n_e_high)

        n_hybrid = n_low + n_high
        return n_hybrid
