"""
tests/test_full_external_partition_controls.py

Purpose
-------
Lock in the high-level bound-partition controls used by `solve_full_then_external`.

These checks are intentionally lightweight: they verify config plumbing and
diagnostic reporting without running a full SCF solve.
"""
from __future__ import annotations

import numpy as np

from otter.electronic.ks_dft import (
    KSDTFConfig,
    _bound_density,
    _bound_orbital_density_tables,
    _bound_ion_charge_table,
    _electron_count,
    _ion_density,
)
from otter.electronic.full_external import (
    FullExternalConfig,
    _auto_bound_basis_from_z,
    _bound_basis_saturation,
    _bound_energy_cut_value,
    _build_bound_tables_and_dos,
    _build_continuum_params,
    _build_ks_config,
    _final_ion_gamma_for_reporting,
)
from otter.electronic.thomas_fermi import ThomasFermiConfig


def test_average_atom_radial_defaults_are_4096_points() -> None:
    """Every public AA backend must default to the production radial grid."""
    assert KSDTFConfig(Z=6, temperature=1.0, mu=0.0).n_points == 2**12
    assert FullExternalConfig(
        element="C",
        temperature_ev=100.0,
        rho_g_cc=2.0,
    ).n_points == 2**12


def test_auto_bound_basis_covers_neutral_shells_through_z118() -> None:
    """The automatic basis must cover every occupied neutral-atom subshell."""
    expected = {
        13: ([0, 1, 2], [4, 3, 1]),
        26: ([0, 1, 2, 3], [5, 3, 2, 1]),
        74: ([0, 1, 2, 3, 4], [7, 5, 4, 2, 1]),
        118: ([0, 1, 2, 3, 4], [8, 7, 5, 3, 1]),
    }
    for z_nuc, (l_expected, caps_expected) in expected.items():
        l_values, caps = _auto_bound_basis_from_z(z_nuc, n_pad=1, l_pad=1)
        np.testing.assert_array_equal(l_values, l_expected)
        np.testing.assert_array_equal(caps, caps_expected)


def test_bound_basis_saturation_detects_radial_and_angular_truncation() -> None:
    energies = np.asarray(
        [
            [-3.0, -0.4, np.inf],
            [-1.0, np.inf, np.inf],
            [-0.1, np.inf, np.inf],
        ]
    )
    diagnostics = _bound_basis_saturation(
        energies,
        np.asarray([0, 1, 2]),
        np.asarray([2, 2, 1]),
    )
    assert diagnostics == {
        "saturated": True,
        "radial_saturated_l": [0, 2],
        "angular_saturated": True,
    }

    complete = _bound_basis_saturation(
        np.asarray([[-3.0, np.inf], [-0.5, np.inf], [np.inf, np.inf]]),
        np.asarray([0, 1, 2]),
        np.asarray([2, 2, 1]),
    )
    assert complete == {
        "saturated": False,
        "radial_saturated_l": [],
        "angular_saturated": False,
    }
    assert ThomasFermiConfig(
        element="C",
        temperature_ev=100.0,
        rho_g_cc=2.0,
    ).n_points == 2**12


def test_zero_and_local_edge_partition_differ_only_for_shallow_levels() -> None:
    """A negative local edge excludes only levels between it and zero.

    The production bound solver has already discarded non-negative
    eigenvalues.  This test isolates the additional density filter applied by
    ``bound_energy_cut_mode`` without running an expensive SCF calculation.
    """
    r = np.linspace(1.0e-4, 8.0, 801)
    energies = np.asarray(((-0.20, -5.0e-4),), dtype=float)
    vectors = np.zeros((1, r.size, 2), dtype=float)
    vectors[0, :, 0] = np.sqrt(r) * np.exp(-r)
    vectors[0, :, 1] = np.sqrt(r) * np.exp(-0.15 * r)
    angular = np.asarray((0,), dtype=int)
    common = {
        "r": r,
        "eigvals": energies,
        "eigvecs": vectors,
        "l_list": angular,
        "mu": 0.05,
        "temperature": 0.4,
        "occ_mode": "fd",
    }

    potential = -2.0e-3 * (1.0 - r / r[-1])
    local_edge = _bound_energy_cut_value(
        r=r,
        v_full=potential,
        r_ws=2.0,
        mode="v_frac",
        value=0.70,
    )
    n_zero = _bound_density(energy_cut=0.0, **common)
    n_local = _bound_density(energy_cut=local_edge, **common)
    n_deep = _bound_density(
        energy_cut=0.0,
        **{**common, "eigvals": energies[:, :1], "eigvecs": vectors[:, :, :1]},
    )

    np.testing.assert_allclose(n_local, n_deep, rtol=2.0e-14, atol=2.0e-14)
    assert np.all(n_zero >= n_local)
    assert float(np.max(n_zero - n_local)) > 0.0

    assert _bound_energy_cut_value(
        r=r,
        v_full=potential,
        r_ws=2.0,
        mode="zero",
        value=0.0,
    ) == 0.0
    assert local_edge == np.interp(0.70 * r[-1], r, potential)


def test_shell_ion_charges_close_total_ion_density_and_degeneracy() -> None:
    """Shell Qion uses the same FD, M(E), cutoff, and 2(2l+1) factors."""
    r = np.linspace(1.0e-4, 4.0, 401)
    values = np.asarray(((-0.2,), (-0.2,)), dtype=float)
    vectors = np.empty((2, r.size, 1), dtype=float)
    vectors[:, :, 0] = np.exp(-r)[None, :]
    angular = np.asarray((0, 1), dtype=int)
    cutoff = 1.0 / (1.0 + np.exp((r - 2.0) / 0.1))
    common = {
        "mu": 0.1,
        "temperature": 0.5,
        "energy_cut": 0.0,
        "gamma": 0.3,
        "r_ws": 2.0,
        "ws_weight_min": 0.0,
    }
    density = _ion_density(
        r,
        values,
        vectors,
        angular,
        cutoff=cutoff,
        **common,
    )
    charges = _bound_ion_charge_table(
        r,
        values,
        vectors,
        angular,
        cutoff=cutoff,
        r_count=r,
        interpolate_boundary=True,
        **common,
    )

    np.testing.assert_allclose(
        np.nansum(charges),
        _electron_count(r, density, 2.0, interpolate_boundary=True),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(charges[1, 0] / charges[0, 0], 3.0)


def test_orbital_density_tables_close_bound_and_ion_densities() -> None:
    """Per-level exports must sum to the densities used by the AA model."""
    r = np.linspace(1.0e-4, 4.0, 401)
    energies = np.asarray(((-0.8, -0.1), (-0.3, np.inf)), dtype=float)
    vectors = np.zeros((2, r.size, 2), dtype=float)
    vectors[0, :, 0] = np.sqrt(r) * np.exp(-r)
    vectors[0, :, 1] = np.sqrt(r) * np.exp(-0.4 * r)
    vectors[1, :, 0] = np.sqrt(r) * r * np.exp(-0.7 * r)
    angular = np.asarray((0, 1), dtype=int)
    cutoff = 1.0 / (1.0 + np.exp((r - 2.0) / 0.1))
    common = {
        "mu": 0.1,
        "temperature": 0.5,
        "energy_cut": 0.0,
        "gamma": 0.2,
        "r_ws": 2.0,
        "ws_weight_min": 0.0,
    }
    per_bound, per_ion = _bound_orbital_density_tables(
        r,
        energies,
        vectors,
        angular,
        bound_occ_mode="fd",
        cutoff=cutoff,
        r_target=r,
        **common,
    )
    expected_bound = _bound_density(
        r,
        energies,
        vectors,
        angular,
        occ_mode="fd",
        **{key: common[key] for key in (
            "mu", "temperature", "energy_cut", "gamma", "r_ws", "ws_weight_min"
        )},
    )
    expected_ion = _ion_density(
        r,
        energies,
        vectors,
        angular,
        cutoff=cutoff,
        **common,
    )
    np.testing.assert_allclose(np.sum(per_bound, axis=(0, 1)), expected_bound)
    np.testing.assert_allclose(np.sum(per_ion, axis=(0, 1)), expected_ion)


def test_bound_table_overlays_zero_tail_scf_spectrum(monkeypatch) -> None:
    """Saved levels must be the same matched poles used by ``n_bound``."""

    def finite_box_misses_state(v_eff, grid_r, grid_dx, l_list, **kwargs):
        n_l = int(np.asarray(l_list).size)
        values = np.full((n_l, 2), np.inf, dtype=float)
        vectors = np.zeros((n_l, np.asarray(grid_r).size, 2), dtype=float)
        return values, vectors

    monkeypatch.setattr(
        "otter.electronic.full_external.solve_bound_states_sparse_numerov",
        finite_box_misses_state,
    )
    x = np.linspace(np.sqrt(1.0e-5), np.sqrt(20.0), 300)
    r = x * x
    matched_energy = -2.5e-7
    tables = _build_bound_tables_and_dos(
        r=r,
        v_full=np.zeros_like(r),
        l_list=np.asarray([0, 1]),
        n_states=np.asarray([2, 1]),
        mu=-0.02,
        temperature_ha=0.3,
        energy_cut=0.0,
        gamma=0.05,
        n_jobs=1,
        zero_tail_bound_meta={
            "applied": True,
            "states": [
                {
                    "l": 0,
                    "state_index": 0,
                    "matched_energy_ha": matched_energy,
                }
            ],
        },
    )

    assert tables["bound_energy_ha"][0, 0] == matched_energy
    assert np.isfinite(tables["bound_fd"][0, 0])
    assert 0.0 < tables["bound_fd"][0, 0] < 1.0
    assert 0.0 < tables["bound_m"][0, 0] < 1.0
    assert tables["bound_fdm"][0, 0] == (
        tables["bound_fd"][0, 0] * tables["bound_m"][0, 0]
    )
    assert float(np.max(tables["dos_bound"])) > 0.0


def test_partition_controls_are_plumbed_to_ks_config() -> None:
    """High-level partition controls should reach the low-level KS config."""
    cfg = FullExternalConfig(
        element="C",
        temperature_ev=100.0,
        rho_g_cc=2.0,
        ion_cut_mode="none",
        ion_cut_c=0.02,
        ion_bound_gamma=0.12,
        ion_gamma_mode="fixed",
        ion_gamma_scale=2.5,
        ion_ws_weight_min=0.9,
        cont_e_min=1.0e-7,
        cont_near_zero_log_grid=True,
        cont_near_zero_log_points_per_decade=5,
        cont_near_zero_log_max_nodes=17,
        cont_near_zero_log_max_energy=2.0e-3,
        cont_resonance_theta_l_min=2,
        cont_resonance_theta_probe_count=3,
        cont_resonance_theta_scan_depth=5,
        cont_resonance_theta_scout_max_extra_nodes=61,
        cont_resonance_theta_root_tol=2.0e-9,
        cont_resonance_theta_sharpness_min=4.0,
        cont_resonance_theta_max_roots=7,
        cont_resonance_theta_refine_depth=18,
        bound_zero_tail_refine=True,
        bound_zero_tail_min_binding_ha=2.0e-9,
        bound_zero_tail_max_binding_ha=3.0e-3,
        bound_zero_tail_scan_points=31,
        bound_zero_tail_l_max=1,
        bound_zero_tail_edge_rel_tol=0.4,
        show_scf_progress=False,
        verbose=False,
    )
    cont_params = _build_continuum_params(
        cfg,
        l_max=8,
        r_ws=2.0,
        rmax=20.0,
        adaptive_mode=str(cfg.cont_adaptive_mode_stage2),
        b3_stage_mode="off",
        e_max_mode=str(cfg.cont_stage2_e_max_mode),
    )
    ks_cfg = _build_ks_config(
        cfg,
        z_nuc=6,
        temperature_ha=10.0 / 27.211386245988,
        n_i=0.01,
        r_ws=2.0,
        rmax=20.0,
        mu_guess=0.1,
        mu_bounds=(-5.0, 5.0),
        max_iter=4,
        cont_params=cont_params,
        compute_external=True,
    )

    assert str(ks_cfg.ion_cut_mode) == "none"
    assert abs(float(ks_cfg.ion_cut_c) - 0.02) < 1.0e-14
    assert abs(float(ks_cfg.ion_bound_gamma) - 0.12) < 1.0e-14
    assert str(ks_cfg.ion_gamma_mode) == "fixed"
    assert abs(float(ks_cfg.ion_gamma_scale) - 2.5) < 1.0e-14
    assert abs(float(ks_cfg.ion_ws_weight_min) - 0.9) < 1.0e-14
    assert float(ks_cfg.continuum_params["e_min"]) == 1.0e-7
    assert bool(ks_cfg.continuum_params["near_zero_log_grid"])
    assert int(ks_cfg.continuum_params["near_zero_log_points_per_decade"]) == 5
    assert int(ks_cfg.continuum_params["near_zero_log_max_nodes"]) == 17
    assert float(ks_cfg.continuum_params["near_zero_log_max_energy"]) == 2.0e-3
    assert int(ks_cfg.continuum_params["resonance_theta_l_min"]) == 2
    assert int(ks_cfg.continuum_params["resonance_theta_probe_count"]) == 3
    assert int(ks_cfg.continuum_params["resonance_theta_scan_depth"]) == 5
    assert int(
        ks_cfg.continuum_params["resonance_theta_scout_max_extra_nodes"]
    ) == 61
    assert float(ks_cfg.continuum_params["resonance_theta_root_tol"]) == 2.0e-9
    assert float(ks_cfg.continuum_params["resonance_theta_sharpness_min"]) == 4.0
    assert int(ks_cfg.continuum_params["resonance_theta_max_roots"]) == 7
    assert int(ks_cfg.continuum_params["resonance_theta_refine_depth"]) == 18
    assert bool(ks_cfg.bound_zero_tail_refine)
    assert float(ks_cfg.bound_zero_tail_min_binding) == 2.0e-9
    assert float(ks_cfg.bound_zero_tail_max_binding) == 3.0e-3
    assert int(ks_cfg.bound_zero_tail_scan_points) == 31
    assert int(ks_cfg.bound_zero_tail_l_max) == 1
    assert float(ks_cfg.bound_zero_tail_edge_rel_tol) == 0.4


def test_final_ion_gamma_for_reporting_prefers_history_value() -> None:
    """Post-SCF diagnostics should report the actual runtime ion gamma."""
    cfg = FullExternalConfig(
        element="C",
        temperature_ev=10.0,
        rho_g_cc=2.0,
        ion_bound_gamma=0.05,
        show_scf_progress=False,
        verbose=False,
    )
    result = {"history": [{"ion_gamma": 0.173}]}
    gamma = _final_ion_gamma_for_reporting(result, cfg)
    assert abs(float(gamma) - 0.173) < 1.0e-14


def test_partition_control_validation_rejects_invalid_values() -> None:
    """Invalid public partition settings should fail early."""
    try:
        FullExternalConfig(
            element="C",
            temperature_ev=10.0,
            rho_g_cc=2.0,
            ion_gamma_mode="bad",
            show_scf_progress=False,
            verbose=False,
        )
    except ValueError as exc:
        assert str(exc) == "ion_gamma_mode must be 'fixed' or 'scattering'."
    else:
        raise AssertionError("Expected invalid ion_gamma_mode to raise ValueError.")

    try:
        FullExternalConfig(
            element="C",
            temperature_ev=10.0,
            rho_g_cc=2.0,
            ion_gamma_scale=0.0,
            show_scf_progress=False,
            verbose=False,
        )
    except ValueError as exc:
        assert str(exc) == "ion_gamma_scale must be positive."
    else:
        raise AssertionError("Expected invalid ion_gamma_scale to raise ValueError.")

    try:
        FullExternalConfig(
            element="C",
            temperature_ev=10.0,
            rho_g_cc=2.0,
            ion_ws_weight_min=1.5,
            show_scf_progress=False,
            verbose=False,
        )
    except ValueError as exc:
        assert str(exc) == "ion_ws_weight_min must be in [0, 1]."
    else:
        raise AssertionError("Expected invalid ion_ws_weight_min to raise ValueError.")

    try:
        FullExternalConfig(
            element="C",
            temperature_ev=10.0,
            rho_g_cc=2.0,
            cont_resonance_theta_scan_depth=-1,
            show_scf_progress=False,
            verbose=False,
        )
    except ValueError as exc:
        assert str(exc) == "cont_resonance_theta_scan_depth must be non-negative."
    else:
        raise AssertionError("Expected negative resonance scan depth to raise ValueError.")

    try:
        FullExternalConfig(
            element="C",
            temperature_ev=10.0,
            rho_g_cc=2.0,
            cont_resonance_theta_scout_max_extra_nodes=-1,
            show_scf_progress=False,
            verbose=False,
        )
    except ValueError as exc:
        assert str(exc) == (
            "cont_resonance_theta_scout_max_extra_nodes must be non-negative when set."
        )
    else:
        raise AssertionError("Expected negative resonance scout budget to raise ValueError.")


if __name__ == "__main__":
    test_partition_controls_are_plumbed_to_ks_config()
    test_final_ion_gamma_for_reporting_prefers_history_value()
    test_partition_control_validation_rejects_invalid_values()
    print("test_full_external_partition_controls: ok")
