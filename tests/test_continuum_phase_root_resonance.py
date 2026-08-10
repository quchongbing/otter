"""Regression tests for threshold and sub-grid continuum energy features.

The phase-root scout is deliberately tested with a controlled Breit-Wigner
line shape.  That isolates energy-mesh behavior from radial Numerov and tail
matching errors while retaining the phase/density relation of a shape
resonance.  It is inspired by the supplementary resonance treatment in
Wilson et al., JQSRT 99, 658-679 (2006), but does not claim to reproduce their
relativistic Theta construction.  The Breit-Wigner phase leaves a bracketed
sign change outside the narrow peak; this regression therefore does not claim
that a finite scout mesh catches every possible rise-and-fall phase feature.
"""
from __future__ import annotations

import numpy as np
import pytest

import otter.electronic.continuum.scattering as quantum
from otter.numerics.grids import create_sqrt_grid


def _run_synthetic(monkeypatch: pytest.MonkeyPatch, evaluator, **overrides):
    grid = create_sqrt_grid(rmax=2.0, N=48)

    def _fake_scattering(*args, **kwargs):
        energy = float(args[4])
        l_max = int(args[5])
        density, phases = evaluator(energy, l_max)
        return np.full_like(grid.r, float(density)), np.asarray(phases, dtype=float)

    monkeypatch.setattr(quantum, "_scattering_density_and_phase", _fake_scattering)
    options = {
        "v_eff": np.zeros_like(grid.r),
        "r": grid.r,
        "mu": 0.0,
        "temperature": 1.0,
        "e_min": 0.1,
        "e_max": 1.1,
        "l_max": 2,
        "grid_kind": "sqrt",
        "grid_step": grid.dxi,
        "l_cap_strategy": "none",
        "e_tol": 1.0e-3,
        "e_max_depth": 8,
        "e_min_width": 1.0e-3,
        "n_e_base": 5,
        "e_base_grid": "linear",
        "delta_tol": None,
        "resonance_tol": None,
        "near_zero_log_grid": False,
        "apply_occ": False,
        "energy_cache": {},
    }
    options.update(overrides)
    return quantum.continuum_density_scattering_adaptive(**options)


def test_phase_root_scout_integrates_subgrid_l1_resonance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A narrow l=1 peak between all base nodes agrees with a dense reference."""
    e_root = 0.531234
    gamma = 1.0e-5
    strength = 0.4

    def evaluator(energy: float, l_max: int):
        lorentzian = (gamma / np.pi) / ((energy - e_root) ** 2 + gamma**2)
        phases = np.zeros(l_max + 1)
        # cos(delta_1) changes sign smoothly at the Breit-Wigner centre.
        phases[1] = np.arctan2(gamma, e_root - energy)
        return 1.0 + strength * lorentzian, phases

    plain, plain_meta = _run_synthetic(
        monkeypatch,
        evaluator,
        adaptive_mode="simpson",
    )
    caught, caught_meta = _run_synthetic(
        monkeypatch,
        evaluator,
        adaptive_mode="phase-root",
        resonance_theta_root_tol=1.0e-10,
        resonance_theta_refine_depth=22,
    )

    dense_e = np.linspace(0.1, 1.1, 400_001)
    dense_y = 1.0 + strength * (gamma / np.pi) / (
        (dense_e - e_root) ** 2 + gamma**2
    )
    reference = float(np.trapezoid(dense_y, dense_e))
    plain_value = float(plain[0])
    caught_value = float(caught[0])

    assert abs(plain_value - reference) / reference > 0.1
    assert abs(caught_value - reference) / reference < 3.0e-3
    assert int(caught_meta["theta_candidates"]) >= 1
    assert len(caught_meta["theta_roots"]) == 1
    assert int(caught_meta["theta_roots"][0]["l"]) == 1
    assert abs(float(caught_meta["theta_roots"][0]["energy"]) - e_root) < 1.0e-8
    assert int(caught_meta["n_eval"]) > int(plain_meta["n_eval"])


def test_phase_root_scout_rejects_broad_crossing_and_excludes_l0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broad phase crossings and s-wave threshold crossings are not shape peaks."""
    e_root = 0.531234

    def broad_l1(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        phases[1] = 0.5 * np.pi + 0.25 * (energy - e_root)
        return 1.0, phases

    _, broad_meta = _run_synthetic(
        monkeypatch,
        broad_l1,
        adaptive_mode="phase-root",
    )
    assert broad_meta["theta_roots"] == []
    assert bool(broad_meta["theta_fallback"])
    assert int(broad_meta["theta_rejected"]) >= 1

    def narrow_l0(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        phases[0] = np.arctan2(1.0e-5, e_root - energy)
        return 1.0, phases

    _, s_meta = _run_synthetic(
        monkeypatch,
        narrow_l0,
        adaptive_mode="phase-root",
    )
    assert s_meta["theta_roots"] == []
    assert int(s_meta["theta_candidates"]) == 0


def test_multiresolution_scout_resolves_off_anchor_even_root_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two roots hidden between coarse scouts require an explicit scan scale.

    This is the adversarial case an isolated Breit-Wigner regression does not
    exercise: the phase advances by two pi, so ``cos(delta)`` has the same
    sign on both sides of the pair.  A finite scout still has no arbitrary
    sub-grid guarantee, but the nested depth and achieved spacing make the
    resolved scale measurable rather than accidental.
    """
    roots = (0.514, 0.526)
    gamma = 1.0e-5
    strength_each = 0.25

    def evaluator(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        phases[1] = sum(np.arctan2(gamma, root - energy) for root in roots)
        density = 1.0
        for root in roots:
            density += strength_each * (gamma / np.pi) / (
                (energy - root) ** 2 + gamma**2
            )
        return density, phases

    common = {
        "adaptive_mode": "phase-root",
        "resonance_theta_root_tol": 1.0e-10,
        "resonance_theta_refine_depth": 22,
        "resonance_theta_scout_max_extra_nodes": 80,
    }
    coarse, coarse_meta = _run_synthetic(
        monkeypatch,
        evaluator,
        resonance_theta_scan_depth=3,
        **common,
    )
    caught, caught_meta = _run_synthetic(
        monkeypatch,
        evaluator,
        resonance_theta_scan_depth=4,
        **common,
    )

    e_lo, e_hi = 0.1, 1.1
    line_area = sum(
        (
            np.arctan((e_hi - root) / gamma)
            - np.arctan((e_lo - root) / gamma)
        )
        / np.pi
        for root in roots
    )
    reference = (e_hi - e_lo) + strength_each * line_area

    assert abs(float(coarse[0]) - reference) / reference > 0.1
    assert coarse_meta["theta_roots"] == []
    assert bool(coarse_meta["theta_fallback"])
    assert abs(float(caught[0]) - reference) / reference < 4.0e-3
    assert len(caught_meta["theta_roots"]) == 2
    assert int(caught_meta["theta_scout_completed_depth"]) == 4
    assert int(caught_meta["theta_scout_extra_node_count"]) <= 80
    assert not bool(caught_meta["theta_scout_budget_exhausted"])
    assert float(caught_meta["theta_scout_min_spacing"]) > 0.0
    assert float(caught_meta["theta_scout_max_spacing"]) <= 0.25 / 16.0 + 1.0e-14
    assert caught_meta["theta_scout_limitation"] == (
        "finite_mesh_no_arbitrary_subgrid_guarantee"
    )

    _, limited_meta = _run_synthetic(
        monkeypatch,
        evaluator,
        resonance_theta_scan_depth=6,
        resonance_theta_scout_max_extra_nodes=20,
        resonance_theta_root_tol=1.0e-10,
        resonance_theta_refine_depth=22,
        adaptive_mode="phase-root",
    )
    assert int(limited_meta["theta_scout_extra_node_count"]) <= 20
    assert bool(limited_meta["theta_scout_budget_exhausted"])
    assert int(limited_meta["theta_scout_completed_depth"]) < 6


def test_coincident_multichannel_roots_share_one_usable_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coincident roots in different l channels must not collapse the panel."""
    e_root = 0.531234
    gamma = 1.0e-5
    strength = 0.4

    def evaluator(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        phase = np.arctan2(gamma, e_root - energy)
        phases[1] = phase
        phases[2] = phase
        lorentzian = (gamma / np.pi) / ((energy - e_root) ** 2 + gamma**2)
        return 1.0 + strength * lorentzian, phases

    caught, meta = _run_synthetic(
        monkeypatch,
        evaluator,
        adaptive_mode="phase-root",
        resonance_theta_root_tol=1.0e-10,
        resonance_theta_refine_depth=22,
        resonance_theta_scan_depth=1,
    )
    e_lo, e_hi = 0.1, 1.1
    line_area = (
        np.arctan((e_hi - e_root) / gamma)
        - np.arctan((e_lo - e_root) / gamma)
    ) / np.pi
    reference = (e_hi - e_lo) + strength * line_area

    assert abs(float(caught[0]) - reference) / reference < 4.0e-3
    assert len(meta["theta_roots"]) == 2
    assert len(meta["theta_root_clusters"]) == 1
    cluster = meta["theta_root_clusters"][0]
    assert cluster["channels"] == [1, 2]
    assert int(cluster["root_count"]) == 2
    assert int(meta["n_windows"]) == 1
    assert not bool(meta["theta_fallback"])
    assert {int(item["cluster_id"]) for item in meta["theta_roots"]} == {0}


def test_phase_root_shard_request_is_forced_to_global_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent energy shards cannot safely own boundary-centred roots."""
    e_root = 0.6  # exactly on the requested two-shard boundary
    gamma = 2.0e-5

    def evaluator(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        phases[1] = np.arctan2(gamma, e_root - energy)
        lorentzian = (gamma / np.pi) / ((energy - e_root) ** 2 + gamma**2)
        return 1.0 + 0.2 * lorentzian, phases

    with pytest.warns(RuntimeWarning, match="forcing adaptive_parallel_mode='batch'"):
        _, meta = _run_synthetic(
            monkeypatch,
            evaluator,
            adaptive_mode="phase-root",
            n_jobs=2,
            adaptive_parallel_mode="shard",
            adaptive_shards=2,
            resonance_theta_root_tol=1.0e-10,
            resonance_theta_refine_depth=22,
        )

    assert meta["adaptive_parallel_mode_requested"] == "shard"
    assert meta["adaptive_parallel_mode"] == "batch"
    assert bool(meta["theta_shard_mode_forced_batch"])
    assert len(meta["theta_roots"]) == 1
    assert int(meta["n_windows"]) == 1


def test_near_zero_log_nodes_recover_s_wave_threshold_integral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Log anchors expose a threshold feature hidden inside the first sqrt panel."""
    e_root = 1.0e-5
    sigma = 1.0e-7
    strength = 0.4

    def threshold_feature(energy: float, l_max: int):
        phases = np.zeros(l_max + 1)
        density = 1.0 + strength * np.exp(-((energy - e_root) / sigma) ** 2) / (
            np.sqrt(np.pi) * sigma
        )
        return density, phases

    common = {
        "e_min": 1.0e-6,
        "e_max": 1.0,
        "l_max": 0,
        "n_e_base": 6,
        "e_base_grid": "sqrt",
        "e_tol": 1.0e-3,
        "e_min_width": 1.0e-8,
        "e_max_depth": 14,
        "adaptive_mode": "simpson",
    }
    missed, missed_meta = _run_synthetic(
        monkeypatch,
        threshold_feature,
        near_zero_log_grid=False,
        **common,
    )
    resolved, resolved_meta = _run_synthetic(
        monkeypatch,
        threshold_feature,
        near_zero_log_grid=True,
        near_zero_log_points_per_decade=4,
        near_zero_log_max_nodes=24,
        near_zero_log_max_energy=1.0e-2,
        **common,
    )

    # The Gaussian is over 90 widths from the lower limit and effectively all
    # of its normalized area lies inside the integration domain.
    reference = (1.0 - 1.0e-6) + strength
    assert abs(float(missed[0]) - reference) / reference > 0.1
    assert abs(float(resolved[0]) - reference) / reference < 3.0e-3
    assert 1 <= int(resolved_meta["near_zero_log_anchor_count"]) <= 24
    assert resolved_meta["near_zero_log_anchors"][0] > 1.0e-6
    # Resolving an actual 1e-7-Ha feature necessarily triggers deep local
    # refinement.  In a smooth threshold channel, the guard itself costs only
    # endpoints and one Simpson midpoint per added panel.
    def flat_threshold(energy: float, l_max: int):
        return 1.0, np.zeros(l_max + 1)

    _, flat_off = _run_synthetic(
        monkeypatch,
        flat_threshold,
        near_zero_log_grid=False,
        **common,
    )
    _, flat_on = _run_synthetic(
        monkeypatch,
        flat_threshold,
        near_zero_log_grid=True,
        near_zero_log_points_per_decade=4,
        near_zero_log_max_nodes=24,
        near_zero_log_max_energy=1.0e-2,
        **common,
    )
    extra_flat = int(flat_on["n_eval"]) - int(flat_off["n_eval"])
    assert extra_flat <= 2 * int(flat_on["near_zero_log_anchor_count"]) + 2
