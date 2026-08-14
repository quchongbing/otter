"""Determinism regressions for the sparse bound-state eigensolver."""

from __future__ import annotations

import numpy as np

from otter.electronic.solvers import bound


def test_sparse_bound_solver_supplies_reproducible_arpack_start(monkeypatch) -> None:
    starts: list[np.ndarray] = []

    def fake_eigs(matrix, *, k, M, sigma, which, tol, v0):
        starts.append(np.asarray(v0, dtype=float).copy())
        n = matrix.shape[0]
        index = np.arange(1, n + 1, dtype=float)
        vectors = np.column_stack(
            [np.sin((mode + 1) * np.pi * index / (n + 1)) for mode in range(k)]
        )
        return np.linspace(-1.0, -0.1, k), vectors

    monkeypatch.setattr(bound, "eigs", fake_eigs)
    x = np.linspace(0.04, 1.2, 40)
    r = x * x
    potential = -2.0 * np.exp(-r)

    np.random.seed(1)
    bound._solve_single_l_sparse(potential, r, x[1] - x[0], 0, 2)
    np.random.seed(987654)
    bound._solve_single_l_sparse(potential, r, x[1] - x[0], 0, 2)

    assert len(starts) == 2
    assert np.array_equal(starts[0], starts[1])
    assert np.all(np.isfinite(starts[0]))
    np.testing.assert_allclose(np.linalg.norm(starts[0]), 1.0)
