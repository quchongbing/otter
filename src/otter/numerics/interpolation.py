"""
otter/numerics/interpolation.py

Purpose
-------
Interpolation utilities for mapping data between radial grids.

Methods
-------
- 1D linear interpolation in r.
- Convenience wrapper to map from log/sqrt grids to linear grids.

References
----------
- Designed for QOZ and g_ii(r) workflows requiring linear grids.
"""
from typing import Tuple
import numpy as np


def interp_to_grid(r_src: np.ndarray, f_src: np.ndarray, r_dst: np.ndarray) -> np.ndarray:
    """
    Interpolate scalar data f(r) from r_src to r_dst.

    Parameters
    ----------
    r_src : ndarray
        Source radial grid (monotonic increasing).
    f_src : ndarray
        Values on the source grid.
    r_dst : ndarray
        Target radial grid.

    Returns
    -------
    ndarray
        Interpolated values on r_dst.
    """
    r_src = np.asarray(r_src)
    f_src = np.asarray(f_src)
    r_dst = np.asarray(r_dst)

    if r_src.ndim != 1 or r_dst.ndim != 1:
        raise ValueError("r_src and r_dst must be 1D arrays.")
    if r_src.size != f_src.size:
        raise ValueError("r_src and f_src must have the same length.")
    if np.any(np.diff(r_src) <= 0):
        raise ValueError("r_src must be strictly increasing.")

    return np.interp(r_dst, r_src, f_src)


def map_to_linear_grid(r_src: np.ndarray,
                       f_src: np.ndarray,
                       rmax: float,
                       N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map values from an arbitrary grid onto a linear grid.

    Parameters
    ----------
    r_src : ndarray
        Source radial grid.
    f_src : ndarray
        Values on r_src.
    rmax : float
        Maximum radius for the linear grid.
    N : int
        Number of points for the linear grid.

    Returns
    -------
    r_lin : ndarray
        Linear grid points.
    f_lin : ndarray
        Interpolated values on r_lin.
    """
    dr = rmax / (N + 1)
    r_lin = dr * np.arange(1, N + 1)
    f_lin = interp_to_grid(r_src, f_src, r_lin)
    return r_lin, f_lin
