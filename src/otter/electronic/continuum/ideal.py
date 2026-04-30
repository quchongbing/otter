"""
otter/electronic/continuum/ideal.py

Purpose
-------
Provide an ideal-gas continuum electron model (uniform free electrons).

Methods
-------
- Exact: Fermi integral I_{1/2}(eta) (default).
- Approx: Sommerfeld (degenerate) + Maxwell-Boltzmann (classical) limits.
  (Exact method requires SciPy; otherwise use method="approx".)

Equations
---------
Exact (atomic units):
  n = (sqrt(2)/pi^2) * T^(3/2) * I_{1/2}(eta)
  I_{1/2}(eta) = ∫_0^∞ sqrt(x) / (1 + exp(x - eta)) dx
  eta = mu / T

Approx:
  Degenerate (mu > 0):
    n ≈ (1/3π^2) (2 mu)^(3/2) [1 + (π^2/8)(T/mu)^2]
  Classical (mu <= 0):
    n = 2 / λ^3 * exp(mu/T),  λ = sqrt(2π/T)

References
----------
- Standard Fermi gas relations (atomic units).
"""
from typing import Dict
import numpy as np
from .interface import ContinuumModel

try:
    from scipy.integrate import quad
    from scipy.special import expit
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def _fermi_integral_half_exact(eta: float, tol: float = 1e-10) -> float:
    """
    Exact I_{1/2}(eta) = ∫_0^∞ sqrt(x) / (1 + exp(x - eta)) dx
    
    Implementation:
    Uses substitution x = u^2 => dx = 2u du to remove infinite derivative at x=0.
    Integrand becomes: 2 * u^2 / (1 + exp(u^2 - eta))
    sigmod(x) = 1/(1+exp(-x))
    """
    def integrand(u: float) -> float:
        return 2.0 * (u**2) * expit(eta - u**2) # expit is sigmod func
    res, _ = quad(integrand, 0.0, np.inf, epsabs=tol, epsrel=tol, limit=100)
    return float(res)

def _ideal_unbound_density_exact(mu: float, temperature: float) -> float:
    """
    Exact ideal-gas density using I_{1/2}(eta).
    """
    T = max(float(temperature), 1e-12)
    eta = float(mu) / T
    pref = np.sqrt(2.0) / (np.pi ** 2)
    return pref * (T ** 1.5) * _fermi_integral_half_exact(eta)


def _ideal_unbound_density_approx(mu: float, temperature: float) -> float:
    """
    Fast approximate density: Sommerfeld (degenerate) or Maxwell-Boltzmann (classical).
    """
    T = max(float(temperature), 1e-12)
    mu = float(mu)

    pref = 1.0 / (3.0 * np.pi**2)
    mu_pos = max(mu, 1e-12)
    n_deg = pref * (2.0 * mu_pos) ** 1.5
    n_deg = n_deg * (1.0 + (np.pi**2 / 8.0) * (T / mu_pos) ** 2)

    lam = np.sqrt(2.0 * np.pi / T)
    n_cl = 2.0 * (lam ** -3) * np.exp(mu / T)

    return n_deg if mu > 0.0 else n_cl


def ideal_unbound_density(mu: float, temperature: float, method: str = "exact") -> float:
    """
    Uniform ideal electron density from (mu, T).

    Parameters
    ----------
    mu : float
        Chemical potential (Ha).
    temperature : float
        Temperature (Ha).
    method : {"exact", "approx"}
        "exact" uses Fermi integral I_{1/2}(eta).
        "approx" uses degenerate/classical limits.

    Returns
    -------
    float
        Uniform density (Bohr^-3).
    """
    method = str(method).lower()
    if method == "exact":
        return _ideal_unbound_density_exact(mu, temperature)
    if method == "approx":
        return _ideal_unbound_density_approx(mu, temperature)
    raise ValueError(f"Unknown method={method!r}")


class IdealContinuum(ContinuumModel):
    """
    Ideal (uniform) continuum model.
    """

    def density(self,
                r: np.ndarray,
                mu: float,
                temperature: float,
                params: Dict[str, float] | None = None) -> np.ndarray:
        method = "exact"
        if params is not None and "ideal_method" in params:
            method = str(params["ideal_method"])
        n0 = ideal_unbound_density(mu, temperature, method=method)
        return np.full_like(r, fill_value=n0, dtype=float)
