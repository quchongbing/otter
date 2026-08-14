"""
otter/data/helpers.py

Shared helpers for tests and example scripts.
"""
from __future__ import annotations

import math
import numpy as np

from otter.numerics.constants import CM_TO_BOHR
from otter.data.elements import atomic_weight as element_atomic_weight


def ion_density_bohr3(rho_g_cc: float, atomic_weight: float) -> float:
    """Return ion number density (Bohr^-3) from mass density and atomic weight."""
    if atomic_weight <= 0.0:
        raise ValueError("atomic_weight must be positive.")
    avogadro = 6.02214076e23
    n_i_cm3 = rho_g_cc / atomic_weight * avogadro
    return n_i_cm3 / (CM_TO_BOHR ** 3)


def resolve_atomic_weight(spec: float | int | str) -> float:
    """
    Resolve atomic weight from a direct mass, atomic number, or symbol.

    Rules
    -----
    - float -> treated as explicit atomic weight.
    - int   -> treated as atomic number Z.
    - str   -> symbol (e.g. \"Al\") or integer string (e.g. \"13\").
    """
    if isinstance(spec, bool):
        raise TypeError("Atomic weight spec must not be bool.")
    if isinstance(spec, float):
        if spec <= 0.0:
            raise ValueError("Atomic weight must be positive.")
        return float(spec)
    if isinstance(spec, int):
        return float(element_atomic_weight(spec))
    if isinstance(spec, str):
        s = spec.strip()
        if not s:
            raise ValueError("Atomic weight spec string cannot be empty.")
        if s.replace(".", "", 1).isdigit() and "." in s:
            val = float(s)
            if val <= 0.0:
                raise ValueError("Atomic weight must be positive.")
            return val
        return float(element_atomic_weight(s))
    raise TypeError("Atomic weight spec must be float, int(Z), or str(symbol/Z).")


def mu_guess_from_density(n_i: float, zbar: float = 3.0) -> float:
    """Return a free-electron mu guess from ion density and target Zbar."""
    n0 = zbar * n_i
    return 0.5 * (3.0 * np.pi**2 * n0) ** (2.0 / 3.0)


def grid_step_from_r(r: np.ndarray, grid_kind: str) -> float:
    """Return the grid step for the specified grid kind."""
    if r.size < 2:
        raise ValueError("Grid must have at least two points.")
    grid_kind = str(grid_kind).lower()
    if grid_kind == "log":
        return float(np.log(r[1] / r[0]))
    if grid_kind == "sqrt":
        return float(np.sqrt(r[1]) - np.sqrt(r[0]))
    if grid_kind == "linear":
        return float(r[1] - r[0])
    raise ValueError(f"Unsupported grid_kind: {grid_kind}")


def trapz_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Return a version-independent one-dimensional trapezoidal integral."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if y_arr.ndim != 1 or x_arr.ndim != 1 or y_arr.size != x_arr.size:
        raise ValueError("trapz_integral requires equal one-dimensional arrays.")
    if y_arr.size < 2:
        return 0.0
    return float(
        np.sum(0.5 * (y_arr[1:] + y_arr[:-1]) * np.diff(x_arr))
    )


def ion_level_weight(energy: float, gamma: float) -> float:
    """Return the M(e) weight from :cite:`StarrettSaumon2013`, Eq. (81)."""
    if gamma <= 0.0:
        return 1.0
    arg = -2.0 * math.sqrt(math.log(2.0)) * energy / gamma
    m_val = math.erf(arg)
    return float(min(max(m_val, 0.0), 1.0))


def electron_count(r: np.ndarray,
                   n: np.ndarray,
                   r_ws: float | None = None,
                   r_min: float | None = None) -> float:
    """Integrate electron density on r (optionally between r_min and r_ws)."""
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)
    mask = np.ones_like(r, dtype=bool)
    if r_min is not None:
        mask &= r >= float(r_min)
    if r_ws is not None:
        mask &= r <= float(r_ws)
    if not np.any(mask):
        return 0.0
    integrand = n[mask] * r[mask] ** 2
    if hasattr(np, "trapezoid"):
        integral = np.trapezoid(integrand, r[mask])
    else:
        integral = np.trapz(integrand, r[mask])
    return float(4.0 * np.pi * integral)


def tail_stats(r: np.ndarray, n_full: np.ndarray, n_ext: np.ndarray) -> dict:
    """Return tail means for full/ext/pa/scr densities."""
    tail_mask = r >= 0.7 * r[-1]
    if not np.any(tail_mask):
        tail_mask = slice(None)
    n_pa = n_full - n_ext
    n_scr = n_pa
    return {
        "n_full": float(np.mean(n_full[tail_mask])),
        "n_ext": float(np.mean(n_ext[tail_mask])),
        "n_pa": float(np.mean(n_pa[tail_mask])),
        "n_scr": float(np.mean(n_scr[tail_mask])),
    }
