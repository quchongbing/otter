"""Regressions for the opt-in bound-box sensitivity diagnostic."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from otter.electronic.ks_dft import KSDTFConfig, _build_grid_pair
from otter.electronic.full_external import FullExternalConfig
from otter.electronic.solvers.bound import solve_bound_states_sparse_numerov


def _square_well_s_energy(depth: float, radius: float) -> float:
    """Return the least-bound l=0 energy for a spherical square well."""
    q_max = np.sqrt(2.0 * depth)

    def mismatch(kappa: float) -> float:
        q = np.sqrt(2.0 * depth - kappa * kappa)
        return float(q / np.tan(q * radius) + kappa)

    kappa = brentq(mismatch, 1.0e-12, q_max - 1.0e-12)
    return float(-0.5 * kappa * kappa)


def test_bound_grid_extension_preserves_sqrt_spacing() -> None:
    cfg = KSDTFConfig(
        Z=1,
        temperature=0.2,
        mu=0.0,
        r_ws=2.0,
        rmax=10.0,
        rmax_mult=None,
        n_points=900,
        bound_rmax=40.0,
    )
    r_cont, dxi_cont, _, r_bound, dxi_bound, _ = _build_grid_pair(cfg)

    assert np.isclose(r_cont[-1], 10.0)
    assert np.isclose(r_bound[-1], 40.0)
    assert r_bound.size > r_cont.size
    assert abs(dxi_bound - dxi_cont) / dxi_cont < 1.0e-3


def test_extended_box_recovers_a_shallow_square_well_state() -> None:
    radius = 2.0
    depth = 0.36
    expected = _square_well_s_energy(depth, radius)

    cfg = KSDTFConfig(
        Z=1,
        temperature=0.2,
        mu=0.0,
        r_ws=2.0,
        rmax=10.0,
        rmax_mult=None,
        n_points=900,
        bound_rmax=40.0,
    )
    r_cont, _, _, r_bound, dxi_bound, _ = _build_grid_pair(cfg)
    v_cont = np.where(r_cont < radius, -depth, 0.0)
    v_bound = np.interp(r_bound, r_cont, v_cont, right=0.0)
    values_short, _ = solve_bound_states_sparse_numerov(
        v_cont,
        r_cont,
        float(np.sqrt(r_cont[1]) - np.sqrt(r_cont[0])),
        np.asarray([0]),
        n_states=2,
    )
    values, vectors = solve_bound_states_sparse_numerov(
        v_bound,
        r_bound,
        dxi_bound,
        np.asarray([0]),
        n_states=2,
    )

    energy = float(values[0, 0])
    radial = np.asarray(vectors[0, :, 0], dtype=float)
    norm = float(np.trapezoid(radial * radial * r_bound, r_bound))

    # The short Dirichlet box pushes this physical shallow state above zero;
    # the public bound wrapper therefore discards it as a continuum box state.
    assert not np.isfinite(values_short[0, 0])
    assert energy < 0.0
    assert abs(energy - expected) < 2.0e-4
    assert abs(norm - 1.0) < 2.0e-4


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan, 5.0])
def test_invalid_low_level_bound_box_is_rejected(value: float) -> None:
    cfg = KSDTFConfig(
        Z=1,
        temperature=0.2,
        mu=0.0,
        r_ws=2.0,
        rmax=10.0,
        rmax_mult=None,
        n_points=200,
        bound_rmax=value,
    )
    with pytest.raises(ValueError, match="bound_rmax"):
        _build_grid_pair(cfg)


def test_full_external_bound_box_controls_are_validated() -> None:
    cfg = FullExternalConfig(
        element="H",
        temperature_ev=10.0,
        rho_g_cc=1.0,
        rmax_mult=15.0,
        bound_rmax_mult=40.0,
        save_data=False,
    )
    assert cfg.bound_rmax_mult == pytest.approx(40.0)

    for invalid in (14.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="bound_rmax_mult"):
            FullExternalConfig(
                element="H",
                temperature_ev=10.0,
                rho_g_cc=1.0,
                rmax_mult=15.0,
                bound_rmax_mult=invalid,
                save_data=False,
            )


def test_production_defaults_use_the_common_aa_domain() -> None:
    """A separate zero-filled bound box must remain an explicit diagnostic."""
    cfg = FullExternalConfig(
        element="H",
        temperature_ev=10.0,
        rho_g_cc=1.0,
        save_data=False,
    )

    assert cfg.rmax_mult == pytest.approx(15.0)
    assert cfg.bound_rmax_mult is None
    assert cfg.bound_zero_tail_refine is False

    low_level = KSDTFConfig(
        Z=1,
        temperature=0.2,
        mu=0.0,
        r_ws=2.0,
        rmax=30.0,
        rmax_mult=None,
        n_points=300,
    )
    r_cont, dxi_cont, kind_cont, r_bound, dxi_bound, kind_bound = (
        _build_grid_pair(low_level)
    )

    assert kind_bound == kind_cont == "sqrt"
    assert r_bound.shape == r_cont.shape
    assert np.array_equal(r_bound, r_cont)
    assert dxi_bound == pytest.approx(dxi_cont)
