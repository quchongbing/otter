"""
otter/numerics/transforms.py

Radial Fourier transforms for the OZ/QOZ workflow.

Current policy
--------------
The production QOZ/HNC path uses only the strict DST-I lattice transform. We
do not keep alternative radial backends in this module because the OZ/HNC loop
applies the transform repeatedly and therefore benefits from a single, exact
discrete forward/inverse pair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dst


@dataclass(frozen=True)
class DSTLatticeTransform:
    """
    Strict DST-I lattice used by the QOZ/HNC solver.

    Grid definition
    ---------------
    The lattice follows the classic HNC convention

      r_i = i * dr,        i = 1 .. ng-1
      k_i = i * dk,        i = 1 .. ng-1
      dr * dk = pi / ng

    so the forward and inverse radial transforms are an exact discrete sine
    transform pair on this lattice.
    """

    r: np.ndarray
    k: np.ndarray
    dr: float
    dk: float
    ng: int


RadialTransform = DSTLatticeTransform


def precompute_dst_lattice_transform_like(
    r_like: np.ndarray,
    *,
    n_grid: int | None = None,
) -> DSTLatticeTransform:
    """
    Build a strict DST-I lattice from an existing approximately uniform `r` grid.

    Parameters
    ----------
    r_like
        Reference linear real-space grid. Only its spacing and right boundary
        are used.
    n_grid
        Optional DST size parameter `ng`. If omitted, it is inferred from the
        supplied spacing and box size.
    """
    r_like = np.asarray(r_like, dtype=np.float64)
    if r_like.ndim != 1 or r_like.size < 2:
        raise ValueError("r_like must be a 1D linear grid with at least two points.")

    dr = float(np.mean(np.diff(r_like)))
    if not np.allclose(
        np.diff(r_like),
        dr,
        rtol=1e-7,
        atol=max(1e-12, abs(dr) * 1e-7),
    ):
        raise ValueError("r_like must be uniformly spaced to build a DST lattice.")

    ng = int(np.round(float(r_like[-1]) / dr)) + 1 if n_grid is None else max(int(n_grid), 4)
    r = dr * np.arange(1, ng, dtype=np.float64)
    dk = np.pi / (dr * ng)
    k = dk * np.arange(1, ng, dtype=np.float64)
    return DSTLatticeTransform(r=r, k=k, dr=dr, dk=dk, ng=ng)


def dst_lattice_forward(
    f_r: np.ndarray,
    transform: DSTLatticeTransform,
) -> np.ndarray:
    """
    Forward radial Fourier-Bessel transform on the strict DST-I lattice.

      F(k_i) = 2*pi*dr/k_i * DST-I[r_i f(r_i)]

    The transform is applied along the last axis, so batched inputs with shape
    `(..., n_r)` are supported directly.
    """
    f_r = np.asarray(f_r, dtype=np.float64)
    if f_r.ndim == 0 or f_r.shape[-1] != transform.r.size:
        raise ValueError("f_r must have transform.r.size entries along its last axis.")
    work = transform.r * f_r
    return (2.0 * np.pi * transform.dr / transform.k) * dst(work, type=1, axis=-1)


def dst_lattice_zero_moment(
    f_r: np.ndarray,
    transform: DSTLatticeTransform,
) -> np.ndarray:
    r"""Return the strict DST-I lattice's discrete :math:`k\to0` moment.

    For :func:`dst_lattice_forward`, the discrete limit is

    .. math::

       F(0) = 4\pi\,\Delta r\sum_i r_i^2 f(r_i).

    This rectangle-rule moment—not an endpoint-weighted trapezoidal
    integral—is the zero-mode limit of the transform actually used by
    QOZ/HNC. Closing the screening charge on the same lattice avoids
    amplifying a tiny charge residual by the Coulomb ``1/k**2`` factor.
    """
    f_r = np.asarray(f_r, dtype=np.float64)
    if f_r.ndim == 0 or f_r.shape[-1] != transform.r.size:
        raise ValueError(
            "f_r must have transform.r.size entries along its last axis."
        )
    return 4.0 * np.pi * transform.dr * np.sum(
        (transform.r**2) * f_r,
        axis=-1,
    )


def dst_lattice_inverse(
    f_k: np.ndarray,
    transform: DSTLatticeTransform,
) -> np.ndarray:
    """
    Inverse radial Fourier-Bessel transform on the strict DST-I lattice.

      f(r_i) = dk/(4*pi^2*r_i) * DST-I[k_i F(k_i)]

    The transform is applied along the last axis, so batched inputs with shape
    `(..., n_k)` are supported directly.
    """
    f_k = np.asarray(f_k, dtype=np.float64)
    if f_k.ndim == 0 or f_k.shape[-1] != transform.k.size:
        raise ValueError("f_k must have transform.k.size entries along its last axis.")
    work = transform.k * f_k
    return transform.dk / (4.0 * np.pi**2 * transform.r) * dst(work, type=1, axis=-1)


def radial_forward(
    f_r: np.ndarray,
    transform: RadialTransform,
) -> np.ndarray:
    """
    Forward radial transform on the strict DST lattice.
    """
    return dst_lattice_forward(f_r, transform)


def radial_inverse(
    f_k: np.ndarray,
    transform: RadialTransform,
) -> np.ndarray:
    """
    Inverse radial transform on the strict DST lattice.
    """
    return dst_lattice_inverse(f_k, transform)
