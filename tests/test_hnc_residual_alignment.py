"""Regression tests for HNC residual/output iterate alignment."""

from __future__ import annotations

import numpy as np

from otter.numerics.grids import create_linear_grid
from otter.ionic import (
    hnc_solver,
    hnc_solver_multicomponent,
    precompute_dst_lattice_transform_like,
)
from otter.numerics.transforms import radial_forward, radial_inverse


def _one_component_output_residual(
    *,
    h_r: np.ndarray,
    c_r: np.ndarray,
    transform,
    n_i: float,
) -> float:
    """Re-evaluate ||F(N)-N|| at the returned scalar HNC iterate."""
    nodal = np.asarray(h_r) - np.asarray(c_r)
    c_k = radial_forward(np.asarray(c_r), transform)
    h_k = c_k / (1.0 - float(n_i) * c_k)
    nodal_map = radial_inverse(h_k - c_k, transform)
    return float(
        np.linalg.norm(nodal_map - nodal)
        / max(np.linalg.norm(nodal), np.linalg.norm(nodal_map), 1.0e-14)
    )


def test_scalar_hnc_residual_describes_returned_iterate() -> None:
    """The reported residual must belong to the arrays returned by the solver."""
    grid = create_linear_grid(rmax=16.0, N=128)
    transform = precompute_dst_lattice_transform_like(np.asarray(grid.r))
    r = np.asarray(transform.r)
    potential = 0.15 * np.exp(-r)
    n_i = 2.0e-3

    _, _, h_r, c_r, residuals = hnc_solver(
        r,
        transform.k,
        potential,
        transform,
        n_i,
        1.0,
        mix=0.25,
        max_iter=1,
        mixing_scheme="picard",
        enforce_h_tail_zero=False,
    )

    output_residual = _one_component_output_residual(
        h_r=h_r,
        c_r=c_r,
        transform=transform,
        n_i=n_i,
    )
    np.testing.assert_allclose(
        residuals[-1],
        output_residual,
        atol=1.0e-12,
        rtol=1.0e-10,
    )


def test_matrix_hnc_residual_describes_returned_iterate() -> None:
    """The same alignment invariant must hold for the matrix Anderson backend."""
    grid = create_linear_grid(rmax=16.0, N=128)
    transform = precompute_dst_lattice_transform_like(np.asarray(grid.r))
    r = np.asarray(transform.r)
    potential = np.zeros((2, 2, r.size), dtype=float)
    potential[0, 0] = 0.15 * np.exp(-r)
    potential[1, 1] = 0.08 * np.exp(-0.8 * r)
    potential[0, 1] = potential[1, 0] = 0.10 * np.exp(-0.9 * r)
    n_i = np.asarray([2.0e-3, 1.0e-3])

    _, _, h_r, c_r, residuals = hnc_solver_multicomponent(
        r,
        transform.k,
        potential,
        transform,
        n_i,
        1.0,
        mix=0.25,
        max_iter=1,
        mixing_scheme="picard",
        s_projection_mode="none",
        c_map_clip=0.0,
        enforce_h_tail_zero=False,
    )

    nodal = np.asarray(h_r) - np.asarray(c_r)
    c_k = radial_forward(np.asarray(c_r), transform)
    sqrt_n = np.sqrt(np.outer(n_i, n_i))
    eye = np.eye(2)
    c_tilde = np.moveaxis(c_k * sqrt_n[:, :, None], -1, 0)
    s_batch = np.linalg.solve(
        eye[None, :, :] - c_tilde,
        np.broadcast_to(eye, c_tilde.shape),
    )
    s_k = np.moveaxis(s_batch, 0, -1)
    h_k = (s_k - eye[:, :, None]) / sqrt_n[:, :, None]
    nodal_map = radial_inverse(h_k - c_k, transform)
    output_residual = float(
        np.linalg.norm(nodal_map - nodal)
        / max(np.linalg.norm(nodal), np.linalg.norm(nodal_map), 1.0e-14)
    )
    np.testing.assert_allclose(
        residuals[-1],
        output_residual,
        atol=1.0e-12,
        rtol=1.0e-10,
    )
