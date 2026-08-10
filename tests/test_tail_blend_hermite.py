"""
tests/test_tail_blend_hermite.py

Purpose
-------
Keep the Hermite splice bridge under regression control.
"""
from __future__ import annotations

import numpy as np

from otter.electronic.continuum.tail import (
    _finite_difference_slope,
    _hermite_bridge_segment,
    fit_tail_params,
    linear_response_tail,
)


def test_hermite_bridge_matches_endpoint_values_and_slopes() -> None:
    """The Hermite bridge should reproduce the prescribed endpoint data."""
    r_seg = np.linspace(0.0, 1.0, 101)
    y_left = 1.25
    dy_left = -0.5
    y_right = 0.75
    dy_right = 0.2

    y = _hermite_bridge_segment(
        r_seg,
        y_left=y_left,
        dy_left=dy_left,
        y_right=y_right,
        dy_right=dy_right,
    )

    assert np.isclose(y[0], y_left)
    assert np.isclose(y[-1], y_right)
    assert np.isclose(_finite_difference_slope(r_seg, y, 0), dy_left, atol=0.02)
    assert np.isclose(_finite_difference_slope(r_seg, y, y.size - 1), dy_right, atol=0.02)


def test_fit_tail_params_recovers_exact_linear_tail() -> None:
    """Auto mode should keep the full B3 model on a truly oscillatory tail."""
    r = np.linspace(1.0, 6.0, 256)
    n0 = 0.05
    mu_id = 0.8
    temperature = 0.2
    A_true = -0.03
    B_true = 0.08
    delta_true = 0.4
    n_r = linear_response_tail(r, n0, mu_id, temperature, A_true, B_true, delta_true)

    params, meta = fit_tail_params(
        r,
        n_r,
        n0,
        mu_id,
        temperature,
        idx_cut=32,
        fit_points=40,
        model="auto",
    )
    A_fit, B_fit, delta_fit = [float(x) for x in params]

    phase_err = np.arctan2(np.sin(delta_fit - delta_true), np.cos(delta_fit - delta_true))
    assert np.isclose(A_fit, A_true, rtol=2.0e-4, atol=1.0e-6)
    assert np.isclose(B_fit, B_true, rtol=2.0e-4, atol=1.0e-6)
    assert abs(float(phase_err)) < 2.0e-4
    assert meta["match_slope_weight"] > 0.0
    assert meta["model_selected"] == "full"


def test_fit_tail_params_auto_uses_a_only_on_non_oscillatory_tail() -> None:
    """Auto mode should suppress the oscillatory term when it does not help."""
    r = np.linspace(1.0, 6.0, 256)
    n0 = 0.05
    mu_id = 0.8
    temperature = 0.2
    A_true = -0.03
    n_r = linear_response_tail(r, n0, mu_id, temperature, A_true, 0.0, 0.0)

    params, meta = fit_tail_params(
        r,
        n_r,
        n0,
        mu_id,
        temperature,
        idx_cut=32,
        fit_points=40,
        model="auto",
    )
    A_fit, B_fit, delta_fit = [float(x) for x in params]

    assert np.isclose(A_fit, A_true, rtol=2.0e-4, atol=1.0e-6)
    assert abs(B_fit) < 1.0e-12
    assert abs(delta_fit) < 1.0e-12
    assert meta["model_selected"] == "a_only"
    assert meta["fit_rel_improve_full"] < meta["auto_rel_improve_tol"]


def test_fit_tail_params_auto_suppresses_tiny_oscillatory_signal() -> None:
    """Auto mode should suppress tiny oscillatory fits even if full B3 can overfit them."""
    r = np.linspace(1.0, 6.0, 256)
    n0 = 0.05
    mu_id = 0.8
    temperature = 0.2
    n_r = linear_response_tail(r, n0, mu_id, temperature, -5.0e-7, 2.0e-7, 0.4)

    params, meta = fit_tail_params(
        r,
        n_r,
        n0,
        mu_id,
        temperature,
        idx_cut=32,
        fit_points=40,
        model="auto",
        auto_rel_improve_tol=0.05,
        auto_signal_rel_tol=2.0e-5,
    )

    assert meta["fit_signal_max"] < meta["fit_signal_threshold"]
    assert meta["model_selected"] == "a_only"
    assert abs(float(params[1])) < 1.0e-12

if __name__ == "__main__":
    test_hermite_bridge_matches_endpoint_values_and_slopes()
    test_fit_tail_params_recovers_exact_linear_tail()
    test_fit_tail_params_auto_uses_a_only_on_non_oscillatory_tail()
    test_fit_tail_params_auto_suppresses_tiny_oscillatory_signal()
    print("test_tail_blend_hermite: ok")
