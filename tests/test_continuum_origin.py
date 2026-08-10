from __future__ import annotations

import numpy as np

from otter.electronic.continuum.scattering import (
    _numerov_propagate_sqrt_numba,
    _prepare_numerov_geometry,
)
from otter.numerics.grids import create_sqrt_grid


def test_continuum_numerov_seed_includes_coulomb_cusp() -> None:
    charge = 6.0
    grid = create_sqrt_grid(rmax=2.0, N=512, rmin=1.0e-5)
    r = grid.r
    potential = -charge / r

    radial = _numerov_propagate_sqrt_numba(
        r,
        potential,
        1.0,
        0,
        float(grid.dxi),
    )
    reduced = radial[:2] / r[:2]
    slope = (reduced[1] - reduced[0]) / (r[1] - r[0])
    intercept = reduced[0] - slope * r[0]

    np.testing.assert_allclose(slope / intercept, -charge, rtol=2.0e-4)
    geometry = _prepare_numerov_geometry(r, potential)
    assert float(np.asarray(geometry["origin_charge"])) == charge


def test_external_continuum_has_zero_origin_charge() -> None:
    grid = create_sqrt_grid(rmax=2.0, N=128, rmin=1.0e-5)
    geometry = _prepare_numerov_geometry(grid.r, np.zeros_like(grid.r))

    assert float(np.asarray(geometry["origin_charge"])) == 0.0
