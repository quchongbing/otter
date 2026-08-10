from __future__ import annotations

import numpy as np

from otter.electronic.solvers.bound import _regularize_origin_series
from otter.numerics.grids import create_sqrt_grid


def _reduced_radial(y: np.ndarray, r: np.ndarray, l_value: int) -> np.ndarray:
    return np.asarray(y, dtype=float) / r ** (float(l_value) + 0.5)


def test_coulomb_origin_repair_restores_bound_state_cusp() -> None:
    charge = 6.0
    l_value = 0
    r = create_sqrt_grid(rmax=1.0, N=1024, rmin=1.0e-5).r
    exact_reduced = np.exp(-charge * r)
    y = r ** (l_value + 0.5) * exact_reduced
    y_bad = y.copy()
    y_bad[:36] *= 1.0 + 0.025 * np.sin(np.arange(36) * np.pi / 2.0)

    repaired = _regularize_origin_series(
        y_bad,
        r,
        float(l_value),
        nuclear_charge=charge,
        core_zr=0.05,
    )
    reduced = _reduced_radial(repaired, r, l_value)
    slope = (reduced[1] - reduced[0]) / (r[1] - r[0])
    intercept = reduced[0] - slope * r[0]

    np.testing.assert_allclose(slope / intercept, -charge, rtol=2.0e-4)
    join = (r >= 0.05 / charge) & (r <= 1.2 * 0.05 / charge)
    assert np.max(np.abs(repaired[join] - y_bad[join])) < 1.0e-12


def test_legacy_origin_repair_remains_available_without_charge() -> None:
    r = create_sqrt_grid(rmax=1.0, N=128, rmin=1.0e-5).r
    y = np.sqrt(r) * np.exp(-r)
    repaired = _regularize_origin_series(y, r, 0.0)

    assert repaired.shape == y.shape
    assert np.all(np.isfinite(repaired))
