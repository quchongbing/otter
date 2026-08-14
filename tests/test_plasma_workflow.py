"""
Lightweight tests for the unified composition-driven plasma workflow.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import otter.workflows as wf


def test_parse_formula_composition_aggregates_repeated_symbols() -> None:
    """Structural formulas should collapse to one distinct-species composition."""
    symbols, counts = wf.parse_formula_composition("CH3COOH")
    assert symbols == ["C", "H", "O"]
    assert counts == [2.0, 4.0, 2.0]


def test_parse_formula_composition_supports_single_and_multi_digit_counts() -> None:
    """Compact formulas with omitted and explicit counts should both work."""
    assert wf.parse_formula_composition("Al") == (["Al"], [1.0])
    assert wf.parse_formula_composition("C10H8O4") == (["C", "H", "O"], [10.0, 8.0, 4.0])


def test_resolve_plasma_composition_supports_explicit_counts() -> None:
    """The preferred explicit composition format should round-trip directly."""
    symbols, counts = wf.resolve_plasma_composition(
        formula=None,
        elements=["C", "H"],
        counts=[1.0, 2.0],
        number_fraction=None,
    )
    assert symbols == ["C", "H"]
    assert counts == [1.0, 2.0]


def test_resolve_plasma_composition_allows_single_species_without_counts() -> None:
    """One species should default to one count when explicit counts are omitted."""
    symbols, counts = wf.resolve_plasma_composition(
        formula=None,
        elements=["Fe"],
        counts=None,
        number_fraction=None,
    )
    assert symbols == ["Fe"]
    assert counts == [1.0]


def test_workflow_defaults_to_partition_charge_and_strict_hnc_output_closure() -> None:
    cfg = wf.PlasmaWorkflowConfig(
        elements=["H"],
        temperature_ev=10.0,
        rho_g_cc=1.0,
    )
    assert cfg.qoz_zbar_mode == "pseudoatom_partition"
    assert cfg.qoz_response_lfc_model == "chabrier1990"
    assert cfg.electronic_model == "qm"
    assert cfg.show_progress is True
    assert cfg.debug is False
    assert cfg.qoz_renormalize_nscr_to_zbar is True
    assert cfg.hnc_enforce_nodal_tail_zero is False
    assert cfg.hnc_closure_transform_tol is None
    assert wf._species_parallel_jobs_default(cfg, 2) == 2
    cfg.species_parallel_jobs = 1
    assert wf._species_parallel_jobs_default(cfg, 2) == 1


def test_workflow_config_solve_method_uses_high_level_entry_point(monkeypatch) -> None:
    cfg = wf.PlasmaWorkflowConfig(
        elements=["H"],
        temperature_ev=10.0,
        rho_g_cc=1.0,
    )
    expected = {"ok": True}
    monkeypatch.setattr(wf, "solve_plasma_workflow", lambda value: expected if value is cfg else None)
    assert cfg.solve() is expected


def test_workflow_rejects_nonpositive_hnc_closure_transform_tolerance() -> None:
    with pytest.raises(
        ValueError,
        match="hnc_closure_transform_tol must be positive",
    ):
        wf.PlasmaWorkflowConfig(
            elements=["H"],
            temperature_ev=10.0,
            rho_g_cc=1.0,
            hnc_closure_transform_tol=0.0,
        )


def test_solve_plasma_workflow_dispatches_single_species_electronic() -> None:
    """A one-species explicit composition should use the single-species AA runner."""
    old_full = wf.solve_full_then_external
    old_mix = wf.solve_mixture_full_then_ext
    seen: dict[str, object] = {}

    def _fake_full(cfg_single):
        seen["cfg"] = cfg_single
        return {
            "mu": 0.25,
            "r_ws": 1.5,
            "r": np.linspace(0.1, 4.0, 16),
            "n_scr": np.exp(-np.linspace(0.1, 4.0, 16)),
            "zbar": 3.0,
        }

    def _fake_mix(_cfg_mix):
        raise AssertionError("Mixture solver should not run for a single species.")

    try:
        wf.solve_full_then_external = _fake_full
        wf.solve_mixture_full_then_ext = _fake_mix
        result = wf.solve_plasma_workflow(
            wf.PlasmaWorkflowConfig(
                elements=["Al"],
                temperature_ev=10.0,
                rho_g_cc=2.7,
            )
        )
    finally:
        wf.solve_full_then_external = old_full
        wf.solve_mixture_full_then_ext = old_mix

    assert result["electronic"]["kind"] == "single_species"
    assert result["otter_version"] == wf.__version__
    assert result["configuration"]["electronic_model"] == "qm"
    assert result["configuration"]["rho_g_cc"] == pytest.approx(2.7)
    assert "StarrettSaumon2014" in result["citation_keys"]
    assert str(seen["cfg"].element) == "Al"
    assert str(seen["cfg"].electronic_model) == "qm"
    assert seen["cfg"].show_scf_progress is True
    assert seen["cfg"].show_summary is True
    assert seen["cfg"].debug is False


def test_workflow_compact_report_and_quiet_mode(monkeypatch, capsys) -> None:
    def _fake_full(_cfg_single):
        r = np.linspace(0.1, 4.0, 16)
        return {
            "mu": 0.25,
            "r_ws": 1.5,
            "r": r,
            "n_scr": np.exp(-r),
            "zbar": 3.0,
        }

    monkeypatch.setattr(wf, "solve_full_then_external", _fake_full)
    wf.solve_plasma_workflow(
        wf.PlasmaWorkflowConfig(
            elements=["Al"],
            temperature_ev=10.0,
            rho_g_cc=2.7,
        )
    )
    report = capsys.readouterr().out
    assert "Otter " in report
    assert "[run] Al | rho=2.7 g/cm^3 | Te=10 eV" in report
    assert "QM electronic structure" in report
    assert "Te=10 eV" in report
    assert "[done] total time=" in report

    wf.solve_plasma_workflow(
        wf.PlasmaWorkflowConfig(
            elements=["Al"],
            temperature_ev=10.0,
            rho_g_cc=2.7,
            show_progress=False,
        )
    )
    assert capsys.readouterr().out == ""


def test_solve_plasma_workflow_dispatches_multicomponent_electronic() -> None:
    """A multi-species explicit composition should use the generic mixture runner."""
    old_full = wf.solve_full_then_external
    old_mix = wf.solve_mixture_full_then_ext
    seen: dict[str, object] = {}

    def _fake_full(_cfg_single):
        raise AssertionError("Single-species solver should not run for a mixture.")

    def _fake_mix(cfg_mix):
        seen["cfg"] = cfg_mix
        return {
            "meta": {
                "mu_common_ha": 0.5,
                "mu_residual_max_ha": 1.0e-6,
                "species_parallel_jobs": int(cfg_mix.species_parallel_jobs),
                "species_parallel_backend": str(cfg_mix.species_parallel_backend),
                "vbar_bohr3": 12.0,
            },
            "species": [
                {
                    "element": "C",
                    "count": 1.0,
                    "x": 1.0 / 3.0,
                    "volume_bohr3": 18.0,
                    "r_ws_bohr": 1.62,
                    "mu_ha": 0.5,
                    "result": {
                        "r": np.linspace(0.1, 4.0, 16),
                        "n_scr": np.exp(-np.linspace(0.1, 4.0, 16)),
                        "zbar": 4.0,
                        "mu": 0.5,
                    },
                },
                {
                    "element": "H",
                    "count": 2.0,
                    "x": 2.0 / 3.0,
                    "volume_bohr3": 9.0,
                    "r_ws_bohr": 1.29,
                    "mu_ha": 0.5,
                    "result": {
                        "r": np.linspace(0.1, 4.0, 16),
                        "n_scr": np.exp(-np.linspace(0.1, 4.0, 16)),
                        "zbar": 1.0,
                        "mu": 0.5,
                    },
                },
            ],
        }

    try:
        wf.solve_full_then_external = _fake_full
        wf.solve_mixture_full_then_ext = _fake_mix
        result = wf.solve_plasma_workflow(
            wf.PlasmaWorkflowConfig(
                elements=["C", "H"],
                counts=[1.0, 2.0],
                temperature_ev=10.0,
                rho_g_cc=5.0,
            )
        )
    finally:
        wf.solve_full_then_external = old_full
        wf.solve_mixture_full_then_ext = old_mix

    assert result["electronic"]["kind"] == "mixture"
    assert list(seen["cfg"].species) == ["C", "H"]
    assert list(seen["cfg"].counts) == [1.0, 2.0]
    assert seen["cfg"].aa_overrides["electronic_model"] == "qm"
    assert int(seen["cfg"].species_parallel_jobs) == 2


def test_solve_plasma_workflow_runs_one_component_ion_structure_when_ti_is_given() -> None:
    """Providing Ti for one species should continue to one-component HNC."""
    old_full = wf.solve_full_then_external
    old_build = wf.build_effective_vii_from_nscr
    old_hnc = wf.hnc_solver
    seen: dict[str, object] = {}

    def _fake_full(_cfg_single):
        r = np.linspace(0.1, 4.0, 32)
        return {
            "mu": 0.25,
            "r_ws": 1.5,
            "r": r,
            "n_ion": np.zeros_like(r),
            "n_scr": np.exp(-r),
            "zbar": 3.0,
            "zbar_partition": 3.0,
        }

    def _fake_build(**kwargs):
        seen["options"] = kwargs.get("options", None)
        r = np.asarray(kwargs["r"], dtype=float)
        k = np.asarray(kwargs["k"], dtype=float)
        return SimpleNamespace(
            vii_r=np.exp(-r),
            vii_k=np.exp(-k),
            n_scr_k=np.exp(-0.5 * k),
            chi_ee_k=-1.0 / (1.0 + k * k),
            chi0_k=-0.8 / (1.0 + 0.5 * k * k),
            gee_k=0.25 * (1.0 - np.exp(-0.5 * k)),
        )

    def _fake_hnc(r, k, v_ii_r, transform, n_i, temperature_ha, **kwargs):
        seen["hnc_tail_shift"] = kwargs.get("enforce_h_tail_zero")
        return (
            np.ones_like(r),
            np.ones_like(k),
            np.zeros_like(r),
            np.zeros_like(r),
            [1.0e-3, 1.0e-5],
        )

    try:
        wf.solve_full_then_external = _fake_full
        wf.build_effective_vii_from_nscr = _fake_build
        wf.hnc_solver = _fake_hnc
        result = wf.solve_plasma_workflow(
            wf.PlasmaWorkflowConfig(
                elements=["Al"],
                temperature_ev=10.0,
                rho_g_cc=2.7,
                ion_temperature_ev=10.0,
            )
        )
    finally:
        wf.solve_full_then_external = old_full
        wf.build_effective_vii_from_nscr = old_build
        wf.hnc_solver = old_hnc

    assert result["ion"] is not None
    assert result["ion"]["kind"] == "one_component"
    assert result["ion"]["gii_r"].ndim == 1
    assert result["ion"]["sii_k"].ndim == 1
    assert result["ion"]["gij_r"].shape[0:2] == (1, 1)
    assert result["ion"]["sij_k"].shape[0:2] == (1, 1)
    assert result["ion"]["v_ie_k"].ndim == 1
    assert result["ion"]["v_ie_r"].shape == result["ion"]["r"].shape
    assert result["ion"]["v_ee_r"].shape == result["ion"]["r"].shape
    assert result["ion"]["c_ie_r"].shape == result["ion"]["r"].shape
    assert result["ion"]["c_ee_r"].shape == result["ion"]["r"].shape
    assert result["ion"]["n_ion_r"].shape == result["ion"]["r"].shape
    assert result["ion"]["f_k"].shape == result["ion"]["k"].shape
    np.testing.assert_array_equal(result["ion"]["f_k"], result["ion"]["n_ion_k"])
    np.testing.assert_array_equal(result["ion"]["q_k"], result["ion"]["n_scr_k"])
    np.testing.assert_array_equal(result["ion"]["gee_k"], result["ion"]["g_ee_k"])
    np.testing.assert_array_equal(result["ion"]["v_ei_k"], result["ion"]["v_ie_k"])
    np.testing.assert_allclose(
        result["ion"]["c_ie_k"],
        -result["ion"]["v_ie_k"] / (10.0 * wf.EV_TO_HA),
    )
    np.testing.assert_allclose(
        result["ion"]["c_ee_k"],
        -result["ion"]["v_ee_k"] / (10.0 * wf.EV_TO_HA),
    )
    assert float(seen["options"].high_k_taper_start_frac) == 0.9
    assert result["ion"]["qoz_response_chi0_model"] == "lindhard_fd"
    assert result["ion"]["qoz_response_lfc_model"] == "chabrier1990"
    assert seen["hnc_tail_shift"] is False
    assert result["ion"]["hnc_enforce_nodal_tail_zero"] is False
    assert result["ion"]["hnc_converged"] is True
    assert float(result["ion"]["hnc_best_residual"]) == pytest.approx(1.0e-5)
    assert float(result["ion"]["hnc_output_residual"]) == pytest.approx(1.0e-5)
    assert float(result["ion"]["closure_transform_max_abs"]) < 1.0e-14
    assert float(result["ion"]["closure_transform_tol"]) == pytest.approx(
        1.0e-3
    )
    assert result["ion"]["hnc_fallback_used"] is False
    assert result["ion"]["hnc_solver_path"] == "direct_anderson"


def test_one_component_ion_structure_rejects_unconverged_hnc_map() -> None:
    """A frozen diagnostic map with negative S(k) must not be returned as HNC."""
    old_build = wf.build_effective_vii_from_nscr
    old_hnc = wf.hnc_solver

    r_native = np.linspace(0.01, 6.0, 96)
    species_entry = {
        "element": "C",
        "Z": 6,
        "volume_bohr3": 20.0,
        "result": {
            "r": r_native,
            "n_ion": np.zeros_like(r_native),
            "n_scr": np.exp(-r_native),
            "zbar": 2.0,
            "zbar_partition": 2.0,
        },
    }

    def _fake_build(**kwargs):
        r = np.asarray(kwargs["r"], dtype=float)
        k = np.asarray(kwargs["k"], dtype=float)
        return SimpleNamespace(
            vii_r=np.exp(-r),
            vii_k=np.exp(-k),
            n_scr_k=np.exp(-k),
            chi_ee_k=-np.ones_like(k),
            chi0_k=-np.ones_like(k),
            gee_k=np.zeros_like(k),
        )

    def _frozen_hnc(r, k, *_args, **_kwargs):
        return (
            np.exp(-r),
            -np.ones_like(k),
            np.zeros_like(r),
            np.zeros_like(r),
            [1.0, 1.0, 1.0],
        )

    cfg = wf.PlasmaWorkflowConfig(
        elements=["C"],
        temperature_ev=10.0,
        ion_temperature_ev=10.0,
        rho_g_cc=1.0,
        qoz_linear_n_points=128,
        hnc_require_converged=True,
        hnc_fallback_mixing_scheme=None,
    )
    try:
        wf.build_effective_vii_from_nscr = _fake_build
        wf.hnc_solver = _frozen_hnc
        with pytest.raises(RuntimeError, match="One-component OZ/HNC did not reach"):
            wf._one_component_ion_structure(cfg, species_entry=species_entry)

        cfg.hnc_require_converged = False
        diagnostic = wf._one_component_ion_structure(
            cfg,
            species_entry=species_entry,
        )
    finally:
        wf.build_effective_vii_from_nscr = old_build
        wf.hnc_solver = old_hnc

    assert diagnostic["hnc_converged"] is False
    assert diagnostic["hnc_best_residual"] == pytest.approx(1.0)
    assert diagnostic["hnc_output_residual"] == pytest.approx(1.0)
    assert diagnostic["hnc_s_min"] < 0.0


def test_one_component_ion_structure_falls_back_to_strict_continuation() -> None:
    """A failed cold start should use the configured raw continuation path."""
    old_build = wf.build_effective_vii_from_nscr
    old_hnc = wf.hnc_solver
    old_continuation = wf.hnc_solver_multicomponent_continuation
    seen: dict[str, object] = {}

    r_native = np.linspace(0.01, 6.0, 96)
    species_entry = {
        "element": "Al",
        "Z": 13,
        "volume_bohr3": 20.0,
        "result": {
            "r": r_native,
            "n_ion": np.zeros_like(r_native),
            "n_scr": np.exp(-r_native),
            "zbar": 3.0,
            "zbar_partition": 3.0,
        },
    }

    def _fake_build(**kwargs):
        r = np.asarray(kwargs["r"], dtype=float)
        k = np.asarray(kwargs["k"], dtype=float)
        return SimpleNamespace(
            vii_r=np.exp(-r),
            vii_k=np.exp(-k),
            n_scr_k=np.exp(-k),
            chi_ee_k=-np.ones_like(k),
            chi0_k=-np.ones_like(k),
            gee_k=np.zeros_like(k),
        )

    def _failed_direct(r, k, *_args, **_kwargs):
        return (
            np.exp(-r),
            -np.ones_like(k),
            np.zeros_like(r),
            np.zeros_like(r),
            [1.0],
        )

    def _physical_continuation(r, k, v_ij_r, _transform, n_i, _temperature, **kwargs):
        seen["v_shape"] = np.asarray(v_ij_r).shape
        seen["n_i"] = np.asarray(n_i).copy()
        seen["potential_scales"] = tuple(kwargs["potential_scales"])
        seen["mixing_scheme"] = kwargs["mixing_scheme"]
        seen["projection"] = kwargs["s_projection_mode"]
        ones_r = np.ones((1, 1, r.size), dtype=float)
        ones_k = np.ones((1, 1, k.size), dtype=float)
        zeros_r = np.zeros((1, 1, r.size), dtype=float)
        return (
            ones_r,
            ones_k,
            zeros_r,
            zeros_r,
            [5.0e-6],
            [
                {
                    "potential_scale": 1.0,
                    "converged": True,
                    "mixing_scheme": "newton_krylov",
                }
            ],
        )

    cfg = wf.PlasmaWorkflowConfig(
        elements=["Al"],
        temperature_ev=30.0,
        ion_temperature_ev=1.0,
        rho_g_cc=2.7,
        qoz_linear_n_points=128,
        hnc_require_converged=True,
    )
    try:
        wf.build_effective_vii_from_nscr = _fake_build
        wf.hnc_solver = _failed_direct
        wf.hnc_solver_multicomponent_continuation = _physical_continuation
        result = wf._one_component_ion_structure(
            cfg,
            species_entry=species_entry,
        )
    finally:
        wf.build_effective_vii_from_nscr = old_build
        wf.hnc_solver = old_hnc
        wf.hnc_solver_multicomponent_continuation = old_continuation

    assert result["hnc_converged"] is True
    assert result["hnc_primary_converged"] is False
    assert result["hnc_fallback_used"] is True
    assert result["hnc_solver_path"] == (
        "direct_anderson->continuation_newton_krylov"
    )
    assert result["hnc_best_residual"] == pytest.approx(5.0e-6)
    assert result["hnc_s_min"] == pytest.approx(1.0)
    assert result["closure_transform_max_abs"] < 1.0e-14
    assert tuple(seen["v_shape"])[:2] == (1, 1)
    assert np.asarray(seen["n_i"]).shape == (1,)
    assert seen["potential_scales"] == cfg.hnc_potential_scales
    assert seen["mixing_scheme"] == "newton_krylov"
    assert seen["projection"] == "none"


def test_solve_plasma_workflow_runs_multicomponent_ion_structure_when_ti_is_given() -> None:
    """Providing Ti for a mixture should continue to multicomponent HNC."""
    old_mix = wf.solve_mixture_full_then_ext
    old_build = wf.build_effective_vij_from_nscr
    old_hnc = wf.hnc_solver_multicomponent_continuation
    seen: dict[str, object] = {}

    def _fake_mix(_cfg_mix):
        r = np.linspace(0.1, 4.0, 32)
        return {
            "meta": {
                "mu_common_ha": 0.5,
                "mu_residual_max_ha": 1.0e-6,
                "species_parallel_jobs": 2,
                "species_parallel_backend": "thread",
                "vbar_bohr3": 12.0,
            },
            "species": [
                {
                    "element": "C",
                    "Z": 6,
                    "count": 1.0,
                    "x": 1.0 / 3.0,
                    "volume_bohr3": 18.0,
                    "r_ws_bohr": 1.62,
                    "mu_ha": 0.5,
                    "result": {
                        "r": r,
                        "n_ion": np.zeros_like(r),
                        "n_scr": np.exp(-r),
                        "zbar": 4.0,
                        "zbar_partition": 4.0,
                        "mu": 0.5,
                    },
                },
                {
                    "element": "H",
                    "Z": 1,
                    "count": 2.0,
                    "x": 2.0 / 3.0,
                    "volume_bohr3": 9.0,
                    "r_ws_bohr": 1.29,
                    "mu_ha": 0.5,
                    "result": {
                        "r": r,
                        "n_ion": np.zeros_like(r),
                        "n_scr": 0.5 * np.exp(-r),
                        "zbar": 1.0,
                        "zbar_partition": 1.0,
                        "mu": 0.5,
                    },
                },
            ],
        }

    def _fake_build(**kwargs):
        seen["options"] = kwargs.get("options", None)
        r = np.asarray(kwargs["r"], dtype=float)
        k = np.asarray(kwargs["k"], dtype=float)
        assert np.allclose(kwargs["n_i"], np.asarray([1.0 / 36.0, 1.0 / 18.0], dtype=float))
        vij_r = np.zeros((2, 2, r.size), dtype=float)
        vij_k = np.zeros((2, 2, k.size), dtype=float)
        n_scr_k = np.zeros((2, k.size), dtype=float)
        n_scr_k[0] = np.exp(-0.5 * k)
        n_scr_k[1] = 0.5 * np.exp(-0.5 * k)
        return SimpleNamespace(
            vij_r=vij_r,
            vij_k=vij_k,
            n_scr_k=n_scr_k,
            chi_ee_k=-1.0 / (1.0 + k * k),
            chi0_k=-0.8 / (1.0 + 0.5 * k * k),
            gee_k=0.25 * (1.0 - np.exp(-0.5 * k)),
        )

    def _fake_hnc(r, k, v_ij_r, transform, n_i, temperature_ha, **kwargs):
        seen["hnc_tail_shift"] = kwargs.get("enforce_h_tail_zero")
        n_species = int(np.asarray(v_ij_r).shape[0])
        g_r = np.ones((n_species, n_species, r.size), dtype=float)
        s_k = np.repeat(np.eye(n_species, dtype=float)[:, :, None], k.size, axis=2)
        h_r = np.zeros_like(g_r)
        c_r = np.zeros_like(g_r)
        return g_r, s_k, h_r, c_r, [1.0e-3, 1.0e-5], [{"potential_scale": 1.0, "res_final": 1.0e-5}]

    try:
        wf.solve_mixture_full_then_ext = _fake_mix
        wf.build_effective_vij_from_nscr = _fake_build
        wf.hnc_solver_multicomponent_continuation = _fake_hnc
        result = wf.solve_plasma_workflow(
            wf.PlasmaWorkflowConfig(
                elements=["C", "H"],
                counts=[1.0, 2.0],
                temperature_ev=10.0,
                rho_g_cc=5.0,
                ion_temperature_ev=10.0,
            )
        )
    finally:
        wf.solve_mixture_full_then_ext = old_mix
        wf.build_effective_vij_from_nscr = old_build
        wf.hnc_solver_multicomponent_continuation = old_hnc

    assert result["ion"] is not None
    assert result["ion"]["kind"] == "multicomponent"
    assert result["ion"]["gij_r"].ndim == 3
    assert result["ion"]["sij_k"].ndim == 3
    assert result["ion"]["v_ie_k"].shape[0] == 2
    assert result["ion"]["c_ie_k"].shape[0] == 2
    assert result["ion"]["v_ee_k"].ndim == 1
    assert result["ion"]["c_ee_k"].ndim == 1
    assert result["ion"]["v_ie_r"].shape == (2, result["ion"]["r"].size)
    assert result["ion"]["c_ie_r"].shape == (2, result["ion"]["r"].size)
    assert result["ion"]["v_ee_r"].shape == result["ion"]["r"].shape
    assert result["ion"]["c_ee_r"].shape == result["ion"]["r"].shape
    assert result["ion"]["n_ion_r"].shape == (2, result["ion"]["r"].size)
    assert result["ion"]["f_k"].shape == (2, result["ion"]["k"].size)
    np.testing.assert_array_equal(result["ion"]["f_k"], result["ion"]["n_ion_k"])
    np.testing.assert_array_equal(result["ion"]["q_k"], result["ion"]["n_scr_k"])
    np.testing.assert_array_equal(result["ion"]["gee_k"], result["ion"]["g_ee_k"])
    assert float(seen["options"].high_k_taper_start_frac) == 0.9
    assert seen["hnc_tail_shift"] is False
    assert result["ion"]["hnc_enforce_nodal_tail_zero"] is False
    assert float(result["ion"]["closure_transform_max_abs"]) < 1.0e-14


def test_continue_plasma_workflow_from_electronic_result_reuses_saved_electronic_payload() -> None:
    """A cached electronic payload should be reusable without rerunning AA."""
    old_build = wf.build_effective_vij_from_nscr
    old_hnc = wf.hnc_solver_multicomponent_continuation
    seen: dict[str, object] = {}

    electronic_result = {
        "meta": {
            "mu_common_ha": 0.5,
            "mu_residual_max_ha": 1.0e-6,
            "species_parallel_jobs": 2,
            "species_parallel_backend": "thread",
            "vbar_bohr3": 12.0,
        },
        "species": [
            {
                "element": "C",
                "Z": 6,
                "count": 1.0,
                "x": 1.0 / 3.0,
                "volume_bohr3": 18.0,
                "r_ws_bohr": 1.62,
                "mu_ha": 0.5,
                "result": {
                    "r": np.linspace(0.1, 4.0, 16),
                    "n_ion": np.zeros(16),
                    "n_scr": np.exp(-np.linspace(0.1, 4.0, 16)),
                    "zbar": 4.0,
                    "zbar_partition": 4.0,
                    "mu": 0.5,
                },
            },
            {
                "element": "H",
                "Z": 1,
                "count": 2.0,
                "x": 2.0 / 3.0,
                "volume_bohr3": 9.0,
                "r_ws_bohr": 1.29,
                "mu_ha": 0.5,
                "result": {
                    "r": np.linspace(0.1, 4.0, 16),
                    "n_ion": np.zeros(16),
                    "n_scr": 0.5 * np.exp(-np.linspace(0.1, 4.0, 16)),
                    "zbar": 1.0,
                    "zbar_partition": 1.0,
                    "mu": 0.5,
                },
            },
        ],
    }

    def _fake_build(**kwargs):
        seen["options"] = kwargs.get("options", None)
        r = np.asarray(kwargs["r"], dtype=float)
        k = np.asarray(kwargs["k"], dtype=float)
        assert np.allclose(kwargs["n_i"], np.asarray([1.0 / 36.0, 1.0 / 18.0], dtype=float))
        vij_r = np.zeros((2, 2, r.size), dtype=float)
        vij_k = np.zeros((2, 2, k.size), dtype=float)
        n_scr_k = np.zeros((2, k.size), dtype=float)
        n_scr_k[0] = np.exp(-0.5 * k)
        n_scr_k[1] = 0.5 * np.exp(-0.5 * k)
        return SimpleNamespace(
            vij_r=vij_r,
            vij_k=vij_k,
            n_scr_k=n_scr_k,
            chi_ee_k=-1.0 / (1.0 + k * k),
            chi0_k=-0.8 / (1.0 + 0.5 * k * k),
            gee_k=0.25 * (1.0 - np.exp(-0.5 * k)),
        )

    def _fake_hnc(r, k, v_ij_r, transform, n_i, temperature_ha, **kwargs):
        seen["hnc_tail_shift"] = kwargs.get("enforce_h_tail_zero")
        n_species = int(np.asarray(v_ij_r).shape[0])
        g_r = np.ones((n_species, n_species, r.size), dtype=float)
        s_k = np.repeat(np.eye(n_species, dtype=float)[:, :, None], k.size, axis=2)
        h_r = np.zeros_like(g_r)
        c_r = np.zeros_like(g_r)
        return g_r, s_k, h_r, c_r, [1.0e-3, 1.0e-5], [{"potential_scale": 1.0, "res_final": 1.0e-5}]

    try:
        wf.build_effective_vij_from_nscr = _fake_build
        wf.hnc_solver_multicomponent_continuation = _fake_hnc
        result = wf.continue_plasma_workflow_from_electronic_result(
            wf.PlasmaWorkflowConfig(
                elements=["C", "H"],
                counts=[1.0, 2.0],
                temperature_ev=10.0,
                rho_g_cc=5.0,
                ion_temperature_ev=10.0,
            ),
            electronic_kind="mixture",
            electronic_result=electronic_result,
        )
    finally:
        wf.build_effective_vij_from_nscr = old_build
        wf.hnc_solver_multicomponent_continuation = old_hnc

    assert result["electronic"]["kind"] == "mixture"
    assert result["ion"] is not None
    assert result["ion"]["kind"] == "multicomponent"
    assert float(seen["options"].high_k_taper_start_frac) == 0.9
    assert seen["hnc_tail_shift"] is False


if __name__ == "__main__":
    test_parse_formula_composition_aggregates_repeated_symbols()
    test_parse_formula_composition_supports_single_and_multi_digit_counts()
    test_resolve_plasma_composition_supports_explicit_counts()
    test_resolve_plasma_composition_allows_single_species_without_counts()
    test_solve_plasma_workflow_dispatches_single_species_electronic()
    test_solve_plasma_workflow_dispatches_multicomponent_electronic()
    test_solve_plasma_workflow_runs_one_component_ion_structure_when_ti_is_given()
    test_solve_plasma_workflow_runs_multicomponent_ion_structure_when_ti_is_given()
    test_continue_plasma_workflow_from_electronic_result_reuses_saved_electronic_payload()
