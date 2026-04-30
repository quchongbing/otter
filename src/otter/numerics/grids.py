"""
otter/numerics/grids.py

Purpose
-------
Provide radial grids used by the AA / PAMD solvers.

Methods
-------
- Logarithmic grid: x = ln(r), uniform spacing in x.
- Sqrt grid: xi = sqrt(r), uniform spacing in xi (Starrett & Saumon 2014, App. B).

Equations
---------
Log grid:
  x_i = x_min + i * dx,   r_i = exp(x_i)
Sqrt grid:
  xi_i = i * dxi,         r_i = xi_i^2

References
----------
- C. E. Starrett & D. Saumon (2014), Appendix B.
"""
from typing import NamedTuple
import numpy as np


class LogGrid(NamedTuple):
    """
    Logarithmic radial grid container.

    Attributes
    ----------
    r : ndarray
        Radial points (Bohr).
    x : ndarray
        Logarithmic coordinate x = ln(r).
    dx : float
        Uniform spacing in x.
    N : int
        Number of points.
    rmin : float
        Minimum radius.
    rmax : float
        Maximum radius.
    """
    r: np.ndarray
    x: np.ndarray
    dx: float
    N: int
    rmin: float
    rmax: float


class SqrtGrid(NamedTuple):
    """
    Sqrt radial grid container.

    Attributes
    ----------
    r : ndarray
        Radial points (Bohr).
    xi : ndarray
        Coordinate xi = sqrt(r).
    dxi : float
        Uniform spacing in xi.
    N : int
        Number of points.
    rmin : float
        Minimum radius.
    rmax : float
        Maximum radius.
    """
    r: np.ndarray
    xi: np.ndarray
    dxi: float
    N: int
    rmin: float
    rmax: float


class LinearGrid(NamedTuple):
    """
    Linear radial grid container.

    Attributes
    ----------
    r : ndarray
        Radial points (Bohr).
    dr : float
        Uniform spacing in r.
    N : int
        Number of points.
    rmax : float
        Maximum radius.
    """
    r: np.ndarray
    dr: float
    N: int
    rmax: float


def create_log_grid(rmin: float, rmax: float, N: int) -> LogGrid:
    """
    Create a logarithmic radial grid.

    Parameters
    ----------
    rmin : float
        Minimum radius.
    rmax : float
        Maximum radius.
    N : int
        Number of points.

    Returns
    -------
    LogGrid
        Logarithmic grid structure.
    """
    if rmin <= 0:
        raise ValueError("rmin must be > 0 for log grid.")
    x_min = np.log(rmin)
    x_max = np.log(rmax)
    x = np.linspace(x_min, x_max, N)
    r = np.exp(x)
    dx = (x_max - x_min) / (N - 1)
    return LogGrid(r=r, x=x, dx=float(dx), N=int(N), rmin=float(rmin), rmax=float(rmax))


def create_sqrt_grid(rmax: float, N: int, rmin: float | None = None) -> SqrtGrid:
    """
    Create a sqrt-spaced radial grid (Starrett & Saumon 2014).

    Parameters
    ----------
    rmax : float
        Maximum radius.
    N : int
        Number of points.
    rmin : float or None
        Optional minimum radius. If None, use the legacy grid anchored at xi=0:
        xi_i = i * dxi, i=1..N. If provided, build a uniform-xi grid on
        [sqrt(rmin), sqrt(rmax)] with N points.

    Returns
    -------
    SqrtGrid
        Sqrt grid structure.
    """
    if rmax <= 0:
        raise ValueError("rmax must be > 0.")
    if N < 2:
        raise ValueError("N must be >= 2.")
    if rmin is None:
        dxi = np.sqrt(rmax) / N
        xi = dxi * np.arange(1, N + 1)
        rmin_eff = float((dxi * 1.0) ** 2)
    else:
        rmin_eff = float(rmin)
        if rmin_eff <= 0.0:
            raise ValueError("rmin must be > 0 for explicit sqrt grid range.")
        if rmin_eff >= float(rmax):
            raise ValueError("rmin must be < rmax.")
        xi_min = np.sqrt(rmin_eff)
        xi_max = np.sqrt(rmax)
        dxi = (xi_max - xi_min) / (N - 1)
        xi = xi_min + dxi * np.arange(N)
    r = xi**2
    return SqrtGrid(
        r=r,
        xi=xi,
        dxi=float(dxi),
        N=int(N),
        rmin=float(rmin_eff),
        rmax=float(rmax),
    )


def create_linear_grid(rmax: float, N: int) -> LinearGrid:
    """
    Create a linear radial grid.

    Parameters
    ----------
    rmax : float
        Maximum radius.
    N : int
        Number of points.

    Returns
    -------
    LinearGrid
        Linear grid structure.
    """
    if rmax <= 0:
        raise ValueError("rmax must be > 0.")
    dr = rmax / (N + 1)
    r = dr * np.arange(1, N + 1)
    return LinearGrid(r=r, dr=float(dr), N=int(N), rmax=float(rmax))
