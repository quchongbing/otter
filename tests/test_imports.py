from __future__ import annotations

from pathlib import Path
import tomllib

import numpy as np

from otter import PlasmaWorkflowConfig, __version__
from otter.electronic.solvers.free import _prepare_numerov_geometry
from otter.electronic.xc import dirac_exchange_potential
from otter.ionic import QOZPotentialOptions
from otter.numerics import create_linear_grid


def test_public_imports() -> None:
    cfg = PlasmaWorkflowConfig(elements=["C"], temperature_ev=10.0, rho_g_cc=1.0)
    assert cfg.temperature_ev == 10.0
    assert isinstance(QOZPotentialOptions(), QOZPotentialOptions)
    assert isinstance(__version__, str)
    assert __version__
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    assert __version__ == pyproject["project"]["version"]


def test_core_helpers_are_callable() -> None:
    grid = create_linear_grid(2.0, 16)
    v = dirac_exchange_potential(np.full_like(grid.r, 0.1))
    geom = _prepare_numerov_geometry(grid.r, v)
    assert geom["r"].shape == grid.r.shape
    assert np.all(np.isfinite(v))
