"""Regression tests for the physical B3 fit-window semantics."""

from __future__ import annotations

import numpy as np

from otter.electronic.continuum.tail import (
    fit_tail_params,
    linear_response_tail,
)


N0 = 0.023
MU = 0.11
TEMPERATURE = 2.0 / 27.211386245988
A_REF = -0.31
B_REF = 18.0
DELTA_REF = -0.7


def _sqrt_grid(rmax: float, n: int) -> np.ndarray:
    xi = np.linspace(np.sqrt(1.0e-4), np.sqrt(float(rmax)), int(n))
    return xi * xi


def _fit_on_grid(n_grid: int) -> tuple[np.ndarray, dict[str, object]]:
    r = _sqrt_grid(30.0, n_grid)
    density = linear_response_tail(
        r,
        N0,
        MU,
        TEMPERATURE,
        A_REF,
        B_REF,
        DELTA_REF,
    )
    idx_cut = int(np.searchsorted(r, 8.0))
    params, meta = fit_tail_params(
        r,
        density,
        N0,
        MU,
        TEMPERATURE,
        idx_cut,
        fit_points=20,
        r_fit_max=12.0,
        fit_window_mode="physical",
        match_value_weight=0.0,
        match_slope_weight=0.0,
        model="full",
    )
    return params, meta


def test_r_fit_max_spans_physical_window() -> None:
    params, meta = _fit_on_grid(4096)

    assert meta["fit_sampling"] == "physical_window_subsampled"
    assert meta["fit_points"] == 20
    assert float(meta["fit_r_min"]) >= 8.0
    assert float(meta["fit_r_max"]) <= 12.0
    assert float(meta["fit_r_max"]) > 11.95
    assert float(meta["fit_r_span"]) > 3.9
    assert int(meta["fit_rank"]) == 3
    assert np.allclose(params, np.array([A_REF, B_REF, DELTA_REF]), rtol=2.0e-8, atol=2.0e-8)


def test_physical_window_fit_is_grid_resolution_invariant() -> None:
    params_coarse, meta_coarse = _fit_on_grid(2048)
    params_fine, meta_fine = _fit_on_grid(8192)

    assert np.allclose(params_coarse, params_fine, rtol=3.0e-8, atol=3.0e-8)
    assert abs(float(meta_coarse["fit_r_span"]) - float(meta_fine["fit_r_span"])) < 2.0e-2


def test_auto_window_uses_friedel_decay_guard() -> None:
    r = _sqrt_grid(30.0, 4096)
    density = linear_response_tail(
        r,
        N0,
        MU,
        TEMPERATURE,
        A_REF,
        B_REF,
        DELTA_REF,
    )
    idx_cut = int(np.searchsorted(r, 8.0))

    _, low_t = fit_tail_params(
        r,
        density,
        N0,
        MU,
        TEMPERATURE,
        idx_cut,
        fit_points=20,
        r_fit_max=11.0,
        fit_window_mode="auto",
        model="full",
    )
    _, high_t = fit_tail_params(
        r,
        density,
        N0,
        0.52,
        8.617333262145 / 27.211386245988,
        idx_cut,
        fit_points=20,
        r_fit_max=11.0,
        fit_window_mode="auto",
        model="full",
    )

    assert low_t["fit_window_mode_resolved"] == "physical"
    assert high_t["fit_window_mode_resolved"] == "local"
    assert float(low_t["fit_window_friedel_edge_ratio"]) >= 0.10
    assert float(high_t["fit_window_friedel_edge_ratio"]) < 0.10


def test_omitting_r_fit_max_preserves_contiguous_legacy_window() -> None:
    r = _sqrt_grid(30.0, 4096)
    density = linear_response_tail(
        r,
        N0,
        MU,
        TEMPERATURE,
        A_REF,
        B_REF,
        DELTA_REF,
    )
    idx_cut = int(np.searchsorted(r, 8.0))
    _, meta = fit_tail_params(
        r,
        density,
        N0,
        MU,
        TEMPERATURE,
        idx_cut,
        fit_points=20,
        match_value_weight=0.0,
        match_slope_weight=0.0,
        model="full",
    )

    assert meta["fit_sampling"] == "contiguous"
    assert meta["fit_points"] == 20
    assert float(meta["fit_r_span"]) < 0.2


def test_local_physical_stencil_is_grid_resolution_invariant() -> None:
    results = []
    for n_grid in (2048, 8192):
        r = _sqrt_grid(30.0, n_grid)
        density = linear_response_tail(
            r,
            N0,
            MU,
            TEMPERATURE,
            A_REF,
            B_REF,
            DELTA_REF,
        )
        idx_cut = int(np.searchsorted(r, 8.0))
        params, meta = fit_tail_params(
            r,
            density,
            N0,
            MU,
            TEMPERATURE,
            idx_cut,
            fit_points=20,
            local_fit_width=0.25,
            fit_window_mode="local",
            match_value_weight=0.0,
            match_slope_weight=0.0,
            model="full",
        )
        results.append((params, meta))

    assert np.allclose(results[0][0], results[1][0], rtol=3.0e-8, atol=3.0e-8)
    for _, meta in results:
        assert meta["fit_window_mode_resolved"] == "local"
        assert meta["fit_sampling"].startswith("physical_window")
        assert 0.22 < float(meta["fit_r_span"]) <= 0.25


def test_default_local_width_reproduces_default_4096_point_stencil() -> None:
    """0.064 R_ws preserves the validated 20-node default handoff."""
    r = _sqrt_grid(15.0, 4096)
    density = linear_response_tail(
        r,
        N0,
        MU,
        TEMPERATURE,
        A_REF,
        B_REF,
        DELTA_REF,
    )
    idx_cut = int(np.searchsorted(r, 3.0))
    legacy, legacy_meta = fit_tail_params(
        r,
        density,
        N0,
        MU,
        TEMPERATURE,
        idx_cut,
        fit_points=20,
        fit_window_mode="local",
        match_value_weight=0.0,
        match_slope_weight=0.0,
        model="full",
    )
    physical, physical_meta = fit_tail_params(
        r,
        density,
        N0,
        MU,
        TEMPERATURE,
        idx_cut,
        fit_points=20,
        local_fit_width=0.064,
        fit_window_mode="local",
        match_value_weight=0.0,
        match_slope_weight=0.0,
        model="full",
    )

    assert legacy_meta["fit_points"] == physical_meta["fit_points"] == 20
    assert legacy_meta["fit_r_min"] == physical_meta["fit_r_min"]
    assert legacy_meta["fit_r_max"] == physical_meta["fit_r_max"]
    assert legacy_meta["fit_r_span"] == physical_meta["fit_r_span"]
    assert np.array_equal(legacy, physical)
