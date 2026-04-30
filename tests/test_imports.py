from __future__ import annotations

import numpy as np

from otter import PlasmaWorkflowConfig
from otter.electronic.solvers.free import _prepare_numerov_geometry
from otter.electronic.xc import dirac_exchange_potential
from otter.ionic import QOZPotentialOptions
from otter.numerics import create_linear_grid


def test_public_imports() -> None:
    cfg = PlasmaWorkflowConfig(elements=["C"], temperature_ev=10.0, rho_g_cc=1.0)
    assert cfg.temperature_ev == 10.0
    assert isinstance(QOZPotentialOptions(), QOZPotentialOptions)


def test_core_helpers_are_callable() -> None:
    grid = create_linear_grid(2.0, 16)
    v = dirac_exchange_potential(np.full_like(grid.r, 0.1))
    geom = _prepare_numerov_geometry(grid.r, v)
    assert geom["r"].shape == grid.r.shape
    assert np.all(np.isfinite(v))
