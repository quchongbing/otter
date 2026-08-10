"""Threshold-bound-state diagnostics and analytic exterior matching.

The AA bound solver stores ``y=sqrt(r) R``.  Consequently its probability
measure is ``y**2 * r dr``; these regressions deliberately use shapes for
which omitting the factor of ``r`` would give a qualitatively wrong answer.

References
----------
Starrett et al., Computer Physics Communications 235, 50--62 (2019),
Eqs. 21--22.
Wilson et al., JQSRT 99, 658 (2006), Appendix A.4.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from otter.electronic.ks_dft import (
    _annotate_zero_tail_bound_diagnostics,
    _bound_diagnostic_result_fields,
    _refine_shallow_bound_states_zero_tail,
    bound_state_reliability_diagnostics,
)
from otter.numerics.grids import create_sqrt_grid
from otter.electronic.solvers.bound import (
    find_shallowest_zero_tail_bound_state,
    find_zero_tail_bound_energy,
    solve_zero_tail_bound_state,
)


def _one_state(y: np.ndarray, energy: float = -0.02):
    values = np.asarray([[energy]], dtype=float)
    vectors = np.asarray(y, dtype=float)[None, :, None]
    return values, vectors


def test_localization_uses_y_squared_times_r_measure() -> None:
    r = np.linspace(0.01, 40.0, 4001)
    # y^2*r = 1, so probability is uniform in radius.  The incorrect y^2 dr
    # measure would instead put most of this synthetic state near the origin.
    values, vectors = _one_state(1.0 / np.sqrt(r))
    r_pot = np.linspace(0.01, 15.0, 1501)
    potential = np.where(r_pot < 4.0, -0.01, 0.0)

    diagnostics = bound_state_reliability_diagnostics(
        r,
        values,
        vectors,
        np.asarray([0]),
        r_ws=2.0,
        potential_r=r_pot,
        potential=potential,
    )
    state = diagnostics["shallowest"]

    assert state is not None
    assert state["probability_inside_rws"] == pytest.approx(2.0 / 40.0, abs=5e-4)
    assert state["mean_radius_bohr"] == pytest.approx(20.0, abs=0.03)
    assert state["localization"] == "diffuse"


def test_asymptotic_start_must_be_sustained_to_scf_boundary() -> None:
    r = np.linspace(0.01, 40.0, 4001)
    values, vectors = _one_state(np.exp(-0.25 * r) / np.sqrt(r))
    r_pot = np.linspace(0.01, 15.0, 1501)
    potential = np.zeros_like(r_pot)
    potential[r_pot < 3.0] = -0.02
    # A late tail excursion is five times the 10%-of-binding criterion.  The
    # sustained r_asym must occur after it, not at the earlier zero crossing.
    spike = int(np.argmin(np.abs(r_pot - 12.0)))
    potential[spike] = 0.01

    state = bound_state_reliability_diagnostics(
        r,
        values,
        vectors,
        np.asarray([0]),
        r_ws=2.0,
        potential_r=r_pot,
        potential=potential,
    )["shallowest"]

    assert state is not None
    assert bool(state["asymptotic_region_found"])
    assert float(state["asymptotic_start_bohr"]) > float(r_pot[spike])
    expected = np.sqrt(0.04) * (r[-1] - state["asymptotic_start_bohr"])
    assert state["bound_box_decay_metric"] == pytest.approx(expected)


def test_tail_and_zero_extension_ratios_flag_unresolved_level() -> None:
    r = np.linspace(0.01, 40.0, 4001)
    values, vectors = _one_state(np.exp(-0.1 * r) / np.sqrt(r), energy=-2.0e-5)
    r_pot = np.linspace(0.01, 15.0, 1501)
    potential = np.zeros_like(r_pot)
    potential[r_pot < 3.0] = -0.1
    potential[r_pot >= 12.0] = 8.0e-5
    potential[-1] = 6.0e-6

    state = bound_state_reliability_diagnostics(
        r,
        values,
        vectors,
        np.asarray([0]),
        r_ws=2.0,
        potential_r=r_pot,
        potential=potential,
    )["shallowest"]

    assert state is not None
    assert state["potential_outer_to_binding_ratio"] == pytest.approx(4.0)
    assert state["potential_edge_to_binding_ratio"] == pytest.approx(0.3)
    assert not bool(state["asymptotic_region_found"])
    assert state["numerical_status"] == "unresolved"


def test_state_above_configured_continuum_edge_does_not_block_aa() -> None:
    """An excluded finite-box state is not part of the bound representation."""
    r = np.linspace(0.01, 40.0, 4001)
    values, vectors = _one_state(
        np.exp(-0.02 * r) / np.sqrt(r),
        energy=-5.0e-6,
    )
    r_pot = np.linspace(0.01, 15.0, 1501)
    potential = np.full_like(r_pot, -2.0e-4)

    diagnostics = bound_state_reliability_diagnostics(
        r,
        values,
        vectors,
        np.asarray([0]),
        r_ws=2.0,
        potential_r=r_pot,
        potential=potential,
        energy_cut=-1.0e-4,
    )

    assert diagnostics["shallowest"] is None
    assert diagnostics["n_states_below_energy_cut"] == 0
    assert diagnostics["energy_cut_ha"] == pytest.approx(-1.0e-4)
    flattened = _bound_diagnostic_result_fields(diagnostics)
    assert flattened["threshold_state_status"] == "none"


def test_threshold_diagnostic_is_invariant_to_common_energy_gauge() -> None:
    r = np.linspace(0.01, 40.0, 4001)
    values, vectors = _one_state(
        np.exp(-0.25 * r) / np.sqrt(r),
        energy=-0.02,
    )
    r_pot = np.linspace(0.01, 15.0, 1501)
    potential = np.where(r_pot < 3.0, -0.02, 0.0)

    reference = bound_state_reliability_diagnostics(
        r,
        values,
        vectors,
        np.asarray([0]),
        r_ws=2.0,
        potential_r=r_pot,
        potential=potential,
        energy_cut=0.0,
    )["shallowest"]
    offset = 0.37
    shifted = bound_state_reliability_diagnostics(
        r,
        values + offset,
        vectors,
        np.asarray([0]),
        r_ws=2.0,
        potential_r=r_pot,
        potential=potential + offset,
        energy_cut=offset,
    )["shallowest"]

    assert reference is not None
    assert shifted is not None
    assert shifted["binding_below_continuum_edge_ha"] == pytest.approx(
        reference["binding_below_continuum_edge_ha"]
    )
    assert shifted["bound_box_decay_metric"] == pytest.approx(
        reference["bound_box_decay_metric"]
    )
    assert shifted["potential_outer_to_binding_ratio"] == pytest.approx(
        reference["potential_outer_to_binding_ratio"]
    )
    assert shifted["numerical_status"] == reference["numerical_status"]


def _shallow_square_well_energy(depth: float, radius: float) -> float:
    q_max = np.sqrt(2.0 * depth)

    def mismatch(kappa: float) -> float:
        q = np.sqrt(2.0 * depth - kappa * kappa)
        return float(q / np.tan(q * radius) + kappa)

    kappa = brentq(mismatch, 1.0e-12, q_max - 1.0e-12)
    return float(-0.5 * kappa * kappa)


def test_zero_tail_matching_recovers_infinite_domain_square_well() -> None:
    radius = 2.0
    depth = 0.36
    expected = _shallow_square_well_energy(depth, radius)
    grid = create_sqrt_grid(rmax=20.0, N=2000)
    r = np.asarray(grid.r, dtype=float)
    potential = np.where(r < radius, -depth, 0.0)

    energy = find_zero_tail_bound_energy(
        potential,
        r,
        float(grid.dxi),
        0,
        (1.1 * expected, 0.9 * expected),
        match_index=int(np.searchsorted(r, 3.0)),
    )

    assert energy == pytest.approx(expected, abs=5.0e-5)


def test_zero_tail_orbital_has_all_space_normalization() -> None:
    radius = 2.0
    depth = 0.36
    expected = _shallow_square_well_energy(depth, radius)
    grid = create_sqrt_grid(rmax=20.0, N=2000)
    r = np.asarray(grid.r, dtype=float)
    potential = np.where(r < radius, -depth, 0.0)

    energy, y, meta = solve_zero_tail_bound_state(
        potential,
        r,
        float(grid.dxi),
        0,
        (1.1 * expected, 0.9 * expected),
        match_index=int(np.searchsorted(r, 3.0)),
    )

    interior_probability = float(np.trapezoid(y * y * r, r))
    assert energy == pytest.approx(expected, abs=5.0e-5)
    assert interior_probability == pytest.approx(meta["interior_probability"], abs=2.0e-8)
    assert meta["interior_probability"] + meta["exterior_probability"] == pytest.approx(1.0)
    assert 0.0 < meta["exterior_probability"] < 1.0

    # In the zero-potential s-wave tail P=sqrt(r)*y has logarithmic
    # derivative -kappa.  Check it away from the numerical endpoint.
    p_reduced = np.sqrt(r) * y
    tail_index = int(np.searchsorted(r, 15.0))
    dlogp = np.gradient(np.log(np.abs(p_reduced[tail_index:])), r[tail_index:])
    assert float(np.median(dlogp)) == pytest.approx(
        -float(meta["kappa_bohr_inv"]), abs=3.0e-4
    )


def test_shallow_zero_tail_scout_returns_nearest_threshold_root() -> None:
    radius = 2.0
    depth = 0.36
    expected = _shallow_square_well_energy(depth, radius)
    grid = create_sqrt_grid(rmax=20.0, N=2000)
    r = np.asarray(grid.r, dtype=float)
    potential = np.where(r < radius, -depth, 0.0)

    state = find_shallowest_zero_tail_bound_state(
        potential,
        r,
        float(grid.dxi),
        0,
        min_binding=1.0e-4,
        max_binding=2.0e-2,
        n_scan=64,
        match_index=int(np.searchsorted(r, 3.0)),
    )

    assert state is not None
    energy, _, meta = state
    assert energy == pytest.approx(expected, abs=7.0e-5)
    assert meta["exterior_probability"] > 0.0


def test_zero_tail_matching_rejects_nonasymptotic_potential_edge() -> None:
    grid = create_sqrt_grid(rmax=20.0, N=800)
    r = np.asarray(grid.r, dtype=float)
    potential = np.full_like(r, -1.0e-2)

    with pytest.raises(ValueError, match="not negligible"):
        find_zero_tail_bound_energy(
            potential,
            r,
            float(grid.dxi),
            0,
            (-0.02, -0.01),
        )


def test_ks_zero_tail_refinement_adds_pole_missed_by_finite_box() -> None:
    radius = 2.0
    depth = 0.36
    expected = _shallow_square_well_energy(depth, radius)
    grid = create_sqrt_grid(rmax=20.0, N=2000)
    r_bound = np.asarray(grid.r, dtype=float)
    potential_r = r_bound[r_bound <= 12.0]
    potential = np.where(potential_r < radius, -depth, 0.0)
    v_bound = np.interp(r_bound, potential_r, potential, right=0.0)
    values = np.asarray([[np.inf]])
    vectors = np.zeros((1, r_bound.size, 1), dtype=float)

    refined_values, refined_vectors, meta = _refine_shallow_bound_states_zero_tail(
        r_bound,
        float(grid.dxi),
        values,
        vectors,
        np.asarray([0]),
        v_bound,
        potential_r=potential_r,
        potential=potential,
        enabled=True,
        min_binding=1.0e-4,
        max_binding=2.0e-2,
        scan_points=48,
        l_max=0,
        edge_rel_tol=0.25,
    )

    assert meta["applied"] is True
    assert meta["states"][0]["action"] == "added_pole_missed_by_finite_wall"
    assert refined_values[0, 0] == pytest.approx(expected, abs=7.0e-5)
    assert 0.0 < np.trapezoid(refined_vectors[0, :, 0] ** 2 * r_bound, r_bound) < 1.0

    diagnostics = bound_state_reliability_diagnostics(
        r_bound,
        refined_values,
        refined_vectors,
        np.asarray([0]),
        r_ws=2.0,
        potential_r=potential_r,
        potential=potential,
    )
    diagnostics = _annotate_zero_tail_bound_diagnostics(diagnostics, meta)
    assert diagnostics["shallowest_status"] == "resolved"
    assert diagnostics["shallowest"]["representation"] == "zero_tail_matched"
    assert diagnostics["shallowest"]["zero_tail_exterior_probability"] > 0.0
    assert diagnostics["shallowest"]["localization"] == "diffuse"
    assert "numeric_box_conditional_localization" in diagnostics["shallowest"]
    flattened = _bound_diagnostic_result_fields(diagnostics)
    assert flattened["bound_probability_inside_rws"] == pytest.approx(
        diagnostics["shallowest"]["all_space_probability_inside_rws"]
    )
    assert flattened["bound_probability_inside_rws"] < diagnostics["shallowest"][
        "probability_inside_rws"
    ]
    assert flattened["bound_mean_radius_over_rws"] == pytest.approx(
        diagnostics["shallowest"]["all_space_mean_radius_bohr"] / 2.0
    )
    assert flattened["threshold_spectral_representation_status"] == "resolved"
    assert flattened["threshold_tail_domain_status"] == "resolved"


def test_ks_zero_tail_refinement_matches_at_physical_scf_boundary() -> None:
    """A resolved Robin boundary must not require a bound-only extension."""
    radius = 2.0
    depth = 0.36
    expected = _shallow_square_well_energy(depth, radius)
    grid = create_sqrt_grid(rmax=20.0, N=2000)
    r = np.asarray(grid.r, dtype=float)
    potential = np.where(r < radius, -depth, 0.0)

    values, vectors, meta = _refine_shallow_bound_states_zero_tail(
        r,
        float(grid.dxi),
        np.asarray([[np.inf]]),
        np.zeros((1, r.size, 1), dtype=float),
        np.asarray([0]),
        potential,
        potential_r=r,
        potential=potential,
        enabled=True,
        min_binding=1.0e-4,
        max_binding=2.0e-2,
        scan_points=48,
        l_max=0,
        edge_rel_tol=0.1,
    )

    assert meta["applied"] is True
    assert meta["matching_mode"] == "direct_physical_boundary"
    assert meta["states"][0]["matching_mode"] == "direct_physical_boundary"
    assert values[0, 0] == pytest.approx(expected, abs=7.0e-5)
    assert 0.0 < np.trapezoid(vectors[0, :, 0] ** 2 * r, r) < 1.0


def test_ks_zero_tail_refinement_rejects_large_scf_edge() -> None:
    grid = create_sqrt_grid(rmax=20.0, N=1200)
    r_bound = np.asarray(grid.r, dtype=float)
    potential_r = r_bound[r_bound <= 12.0]
    potential = np.where(potential_r < 2.0, -0.36, -0.02)
    v_bound = np.interp(r_bound, potential_r, potential, right=0.0)

    values, vectors, meta = _refine_shallow_bound_states_zero_tail(
        r_bound,
        float(grid.dxi),
        np.asarray([[np.inf]]),
        np.zeros((1, r_bound.size, 1), dtype=float),
        np.asarray([0]),
        v_bound,
        potential_r=potential_r,
        potential=potential,
        enabled=True,
        min_binding=1.0e-4,
        max_binding=5.0e-2,
        scan_points=32,
        l_max=0,
        edge_rel_tol=0.25,
    )

    assert meta["applied"] is False
    assert meta["reason"] in {
        "scf_potential_has_attractive_outer_tail",
        "no_shallow_pole",
    }
    assert not np.isfinite(values[0, 0])
    assert np.all(vectors == 0.0)
    diagnostics = bound_state_reliability_diagnostics(
        r_bound,
        values,
        vectors,
        np.asarray([0]),
        r_ws=2.0,
        potential_r=potential_r,
        potential=potential,
    )
    diagnostics = _annotate_zero_tail_bound_diagnostics(diagnostics, meta)
    flattened = _bound_diagnostic_result_fields(diagnostics)
    if meta.get("rejected_states"):
        assert flattened["threshold_state_status"] == "unresolved"
        assert flattened["threshold_state_representation"] == (
            "zero_tail_candidate_rejected"
        )
        assert flattened["threshold_tail_domain_status"] == "unresolved"


@pytest.mark.parametrize(
    ("outer_min", "outer_abs_ratio", "expected"),
    [
        (-2.0e-4, 2.0, "unresolved"),
        (+2.0e-4, 2.0, "marginal"),
    ],
)
def test_zero_tail_annotation_distinguishes_attractive_and_repulsive_tails(
    outer_min: float,
    outer_abs_ratio: float,
    expected: str,
) -> None:
    """Only an attractive remote tail can invalidate the matched pole."""
    energy = -1.0e-4
    state = {
        "l": 0,
        "state_index": 0,
        "r_ws_bohr": 2.0,
        "energy_ha": energy,
        "mean_radius_bohr": 3.0,
        "rms_radius_bohr": 4.0,
        "probability_inside_rws": 0.4,
        "probability_inside_2rws": 0.7,
        "probability_inside_5rws": 0.95,
        "probability_inside_potential_box": 0.98,
        "potential_outer_min_ha": outer_min,
        "potential_outer_to_binding_ratio": outer_abs_ratio,
        "potential_beyond_5rws_min_ha": outer_min,
        "potential_beyond_5rws_to_binding_ratio": outer_abs_ratio,
        "potential_edge_to_binding_ratio": 0.05,
        "bound_box_decay_metric": 1.0,
        "bound_box_decay_metric_from_rws": 1.0,
        "asymptotic_start_bohr": np.nan,
        "asymptotic_start_over_rws": np.nan,
        "asymptotic_region_found": False,
        "numerical_status": "unresolved",
        "localization": "diffuse",
        "reasons": ["finite_box"],
    }
    matched = {
        "l": 0,
        "state_index": 0,
        "finite_wall_energy_ha": -8.0e-5,
        "matched_energy_ha": energy,
        "interior_probability": 0.8,
        "exterior_probability": 0.2,
        "all_space_mean_radius_bohr": 6.0,
        "all_space_rms_radius_bohr": 8.0,
        "edge_relative_to_binding": max(0.0, -outer_min) / abs(energy),
        "edge_absolute_relative_to_binding": 0.05,
    }

    annotated = _annotate_zero_tail_bound_diagnostics(
        {"states": [state], "shallowest": state, "shallowest_status": "unresolved"},
        {"applied": True, "states": [matched], "reason": "matched"},
    )
    shallow = annotated["shallowest"]
    flattened = _bound_diagnostic_result_fields(annotated)

    assert shallow["spectral_representation_status"] == "resolved"
    assert shallow["tail_domain_status"] == expected
    assert shallow["numerical_status"] == expected
    assert flattened["threshold_spectral_representation_status"] == "resolved"
    assert flattened["threshold_tail_domain_status"] == expected
    assert flattened["threshold_state_status"] == expected
    if expected == "unresolved":
        assert shallow["outer_attractive_to_binding_ratio"] == pytest.approx(2.0)
        assert "attractive_outer_scf_tail_exceeds_binding_energy" in shallow["reasons"]
    else:
        assert shallow["outer_attractive_to_binding_ratio"] == pytest.approx(0.0)
        assert "outer_scf_tail_not_small_relative_to_binding" in shallow["reasons"]


def test_direct_boundary_annotation_uses_physical_matching_gauge() -> None:
    """A local density-partition edge must not invalidate a matched pole."""
    state = {
        "l": 0,
        "state_index": 0,
        "r_ws_bohr": 2.0,
        "energy_ha": -1.0e-3,
        "mean_radius_bohr": 3.0,
        "rms_radius_bohr": 4.0,
        "probability_inside_rws": 0.4,
        "probability_inside_2rws": 0.7,
        "probability_inside_5rws": 0.95,
        "probability_inside_potential_box": 0.98,
        # These ratios are formed relative to a nonzero local E_cut and can
        # therefore be large even though the physical V_eff tends to zero.
        "potential_outer_min_ha": -2.0e-4,
        "potential_outer_to_binding_ratio": 0.2,
        "potential_beyond_5rws_min_ha": -2.0e-4,
        "potential_beyond_5rws_to_binding_ratio": 0.2,
        "potential_edge_to_binding_ratio": 0.2,
        "bound_box_decay_metric": 1.0,
        "bound_box_decay_metric_from_rws": 1.0,
        "asymptotic_start_bohr": np.nan,
        "asymptotic_start_over_rws": np.nan,
        "asymptotic_region_found": False,
        "numerical_status": "marginal",
        "localization": "diffuse",
        "reasons": ["partition_edge_offset"],
    }
    matched = {
        "l": 0,
        "state_index": 0,
        "matching_mode": "direct_physical_boundary",
        "finite_wall_energy_ha": -8.0e-4,
        "matched_energy_ha": -1.0e-3,
        "interior_probability": 0.9,
        "exterior_probability": 0.1,
        "all_space_mean_radius_bohr": 4.0,
        "all_space_rms_radius_bohr": 5.0,
        "edge_relative_to_binding": 0.02,
        "edge_absolute_relative_to_binding": 0.03,
    }

    annotated = _annotate_zero_tail_bound_diagnostics(
        {"states": [state], "shallowest": state, "shallowest_status": "marginal"},
        {
            "applied": True,
            "matching_mode": "direct_physical_boundary",
            "states": [matched],
            "reason": "matched",
        },
    )
    shallow = annotated["shallowest"]
    flattened = _bound_diagnostic_result_fields(annotated)

    assert shallow["tail_diagnostic_basis"] == (
        "physical_scf_boundary_matching_guard"
    )
    assert shallow["tail_domain_status"] == "resolved"
    assert shallow["numerical_status"] == "resolved"
    assert flattened["threshold_state_status"] == "resolved"


def test_unmatched_shallow_finite_box_state_is_fail_safe_unresolved() -> None:
    grid = create_sqrt_grid(rmax=20.0, N=600)
    r_bound = np.asarray(grid.r, dtype=float)
    potential_r = r_bound[r_bound <= 12.0]
    potential = np.zeros_like(potential_r)
    v_bound = np.zeros_like(r_bound)
    # Deliberately inconsistent synthetic input: a free Hamiltonian has no
    # negative pole, but the finite-box payload claims one.  The refinement
    # must retain it for diagnosis while preventing root/cache/QOZ reuse.
    values_in = np.asarray([[-1.0e-5]])
    vectors_in = np.zeros((1, r_bound.size, 1), dtype=float)
    vectors_in[0, :, 0] = np.exp(-0.02 * r_bound) / np.sqrt(r_bound)

    values, vectors, meta = _refine_shallow_bound_states_zero_tail(
        r_bound,
        float(grid.dxi),
        values_in,
        vectors_in,
        np.asarray([0]),
        v_bound,
        potential_r=potential_r,
        potential=potential,
        enabled=True,
        min_binding=1.0e-8,
        max_binding=1.0e-3,
        scan_points=16,
        l_max=0,
        edge_rel_tol=0.25,
    )

    assert meta["applied"] is False
    assert meta["reason"] == "finite_box_shallow_pole_not_matched"
    assert values[0, 0] == values_in[0, 0]
    diagnostics = bound_state_reliability_diagnostics(
        r_bound,
        values,
        vectors,
        np.asarray([0]),
        r_ws=2.0,
        potential_r=potential_r,
        potential=potential,
    )
    diagnostics = _annotate_zero_tail_bound_diagnostics(diagnostics, meta)
    flattened = _bound_diagnostic_result_fields(diagnostics)
    assert flattened["threshold_state_status"] == "unresolved"
    assert flattened["threshold_state_representation"] == (
        "finite_box_shallow_pole_not_matched"
    )
