"""Finite-temperature Thomas--Fermi AA and QOZ integration tests."""
from __future__ import annotations

import numpy as np
import pytest

import otter.electronic.thomas_fermi as tf_module
from otter.electronic.continuum.ideal import _fermi_integral_half_exact
from otter.electronic.full_external import (
    FullExternalConfig,
    solve_full_then_external,
)
from otter.electronic.potential import effective_potential_full
from otter.electronic.thomas_fermi import (
    FermiHalfEvaluator,
    ThomasFermiConfig,
    incomplete_fermi_half_lower,
    incomplete_fermi_half_upper,
    solve_thomas_fermi_full_then_external,
)
from otter.workflows import PlasmaWorkflowConfig, solve_plasma_workflow


@pytest.mark.parametrize("eta", [-15.0, -2.0, 0.0, 5.0, 20.0, 50.0])
def test_fermi_half_matches_exact_quadrature(eta: float) -> None:
    evaluator = FermiHalfEvaluator(quadrature_order=160)
    assert float(evaluator(eta)) == pytest.approx(
        _fermi_integral_half_exact(eta),
        rel=3.0e-8,
        abs=2.0e-12,
    )


def test_upper_and_lower_incomplete_fermi_integrals_close() -> None:
    evaluator = FermiHalfEvaluator(quadrature_order=160)
    eta = np.asarray([-3.0, 0.5, 8.0, 1000.0])
    split = np.asarray([0.2, 2.0, 7.0, 995.0])
    lower = incomplete_fermi_half_lower(eta, split, quadrature_order=160)
    upper = incomplete_fermi_half_upper(
        eta,
        split,
        fermi_half=evaluator,
        quadrature_order=160,
    )
    assert lower + upper == pytest.approx(
        evaluator(eta), rel=2.0e-9, abs=2.0e-10
    )


def test_tf_full_external_preserves_neutral_pseudoatom_identity() -> None:
    result = solve_thomas_fermi_full_then_external(
        ThomasFermiConfig(
            element="Al",
            temperature_ev=15.0,
            rho_g_cc=2.7,
            n_points=512,
            max_iter=200,
        )
    )
    r = np.asarray(result["r"], dtype=float)

    def charge(density: np.ndarray) -> float:
        return float(4.0 * np.pi * np.trapezoid(density * r**2, r))

    assert result["converged"] is True
    assert result["ext_status"]["converged"] is True
    assert result["xc_provenance"]["provider"] == "otter_builtin"
    assert result["xc_provenance"]["components"][0]["dois"] == [
        "10.1017/S0305004100016108"
    ]
    assert result["meta"]["xc_provenance"] == result["xc_provenance"]
    assert result["meta"]["q_full_ws"] == pytest.approx(13.0, abs=2.0e-8)
    assert charge(result["n_pa"]) == pytest.approx(13.0, abs=2.0e-5)
    assert result["n_scr"] == pytest.approx(
        result["n_pa"] - result["n_ion"],
        rel=0.0,
        abs=3.0e-8,
    )
    assert charge(result["n_scr"]) == pytest.approx(
        result["zbar_partition"], abs=2.0e-5
    )


def test_full_external_dispatches_tf_backend() -> None:
    result = solve_full_then_external(
        FullExternalConfig(
            element="Al",
            temperature_ev=15.0,
            rho_g_cc=2.7,
            electronic_model="tf",
            n_points=384,
        )
    )
    assert result["meta"]["electronic_model"] == "thomas_fermi"
    assert result["threshold_state_status"] == "not_applicable_tf"
    assert result["converged"] is True


def test_full_external_dispatches_tf_sc_controls(monkeypatch) -> None:
    """The public dispatcher must not discard the Sec. 2.4 SC inputs."""
    r = np.linspace(0.05, 4.0, 64)
    g_ii = 1.0 - np.exp(-(r / 1.2) ** 4)
    v_corr = 0.02 * np.exp(-(r / 0.8) ** 2)
    seen: list[ThomasFermiConfig] = []

    def fake_solve(cfg: ThomasFermiConfig) -> dict[str, object]:
        seen.append(cfg)
        return {"electronic_model": "tf"}

    monkeypatch.setattr(
        tf_module,
        "solve_thomas_fermi_full_then_external",
        fake_solve,
    )
    result = solve_full_then_external(
        FullExternalConfig(
            element="Al",
            temperature_ev=15.0,
            rho_g_cc=8.1,
            electronic_model="tf",
            full_fixed_mu_ha=0.42,
            g_ii_override=g_ii,
            g_ii_override_r=r,
            v_corr_full=v_corr,
            v_corr_full_r=r,
            v_corr_ext=v_corr,
            v_corr_ext_r=r,
        )
    )

    assert result == {"electronic_model": "tf"}
    assert len(seen) == 1
    cfg = seen[0]
    assert cfg.full_fixed_mu_ha == pytest.approx(0.42)
    np.testing.assert_array_equal(cfg.g_ii_override, g_ii)
    np.testing.assert_array_equal(cfg.g_ii_override_r, r)
    np.testing.assert_array_equal(cfg.v_corr_full, v_corr)
    np.testing.assert_array_equal(cfg.v_corr_ext, v_corr)


def test_tf_sc_step_keeps_is_mu_and_uses_tabulated_background(
    monkeypatch,
    tmp_path,
) -> None:
    """Exercise the TF electronic step of Starrett--Saumon Sec. 2.4.

    A fixed IS chemical potential means Eq. (3) is intentionally *not*
    re-solved.  The tabulated g_II must also disable the analytic sharp-step
    shortcut, while V_Ie^C remains a separately auditable potential term.
    """
    is_result = solve_thomas_fermi_full_then_external(
        ThomasFermiConfig(
            element="Al",
            temperature_ev=15.0,
            rho_g_cc=8.1,
            n_points=256,
            max_iter=200,
        )
    )
    r = np.asarray(is_result["r"], dtype=float)
    r_ws = float(is_result["r_ws"])
    g_ii = 1.0 - np.exp(-(r / (0.9 * r_ws)) ** 4)
    v_corr = 0.01 * np.exp(-(r / r_ws) ** 2)
    v_corr -= v_corr[-1]

    def forbidden_neutrality_solve(*args, **kwargs):
        raise AssertionError("SC-TF must retain the IS chemical potential.")

    monkeypatch.setattr(tf_module, "_neutral_mu", forbidden_neutrality_solve)
    sc_result = solve_thomas_fermi_full_then_external(
        ThomasFermiConfig(
            element="Al",
            temperature_ev=15.0,
            rho_g_cc=8.1,
            n_points=256,
            max_iter=200,
            full_fixed_mu_ha=float(is_result["mu"]),
            g_ii_override=g_ii,
            g_ii_override_r=r,
            v_corr_full=v_corr,
            v_corr_full_r=r,
            v_corr_ext=v_corr,
            v_corr_ext_r=r,
            v_full_init=np.asarray(is_result["v_full"], dtype=float),
            v_full_init_r=r,
            v_ext_init=np.asarray(is_result["v_ext"], dtype=float),
            v_ext_init_r=r,
            save_data=True,
            save_output_dir=tmp_path,
        )
    )

    assert sc_result["converged"] is True
    assert sc_result["ext_status"]["converged"] is True
    assert sc_result["mu"] == pytest.approx(is_result["mu"], abs=0.0)
    assert sc_result["meta"]["fixed_is_mu_used"] is True
    assert sc_result["meta"]["g_ii_override_used"] is True
    assert sc_result["meta"]["analytic_ion_sphere_background"] is False
    np.testing.assert_allclose(sc_result["g_ii_background"], g_ii)
    np.testing.assert_allclose(sc_result["v_corr_full"], v_corr)
    np.testing.assert_allclose(sc_result["v_corr_ext"], v_corr)
    np.testing.assert_allclose(
        sc_result["v_full"],
        sc_result["v_nuc"]
        + sc_result["v_H"]
        + sc_result["v_xc"]
        + sc_result["v_corr_full"],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        sc_result["v_ext"],
        sc_result["v_H_ext"]
        + sc_result["v_xc_ext"]
        + sc_result["v_corr_ext"],
        rtol=0.0,
        atol=1.0e-12,
    )
    # Sec. 2.4 fixes mu rather than imposing the IS sphere-neutrality Eq. (3).
    assert abs(float(sc_result["meta"]["q_full_ws"]) - 13.0) > 1.0e-3
    with np.load(sc_result["saved_paths"]["data_npz"], allow_pickle=False) as saved:
        np.testing.assert_allclose(saved["g_ii_background"], g_ii)
        np.testing.assert_allclose(saved["v_corr_full"], v_corr)
        np.testing.assert_allclose(saved["v_corr_ext"], v_corr)


def test_analytic_is_background_rejects_tabulated_gii() -> None:
    """The sharp-step shortcut must not erase an SC g_II source."""
    r = np.linspace(0.05, 6.0, 128)
    g_ii = 1.0 - np.exp(-(r / 1.5) ** 4)
    with pytest.raises(ValueError, match="literal sharp ion-sphere"):
        effective_potential_full(
            r,
            np.full_like(r, 0.1),
            0.08,
            g_ii,
            13.0,
            ion_sphere_radius=1.5,
        )


def test_tf_runs_through_chabrier_qoz_hnc() -> None:
    cfg = PlasmaWorkflowConfig(
        formula="Al",
        temperature_ev=15.0,
        rho_g_cc=2.7,
        ion_temperature_ev=15.0,
        electronic_model="tf",
        aa_overrides={"n_points": 512},
        qoz_linear_n_points=512,
        hnc_max_iter=300,
    )
    result = solve_plasma_workflow(cfg)
    electronic = result["electronic"]["result"]
    ion = result["ion"]
    assert electronic["meta"]["electronic_model"] == "thomas_fermi"
    assert cfg.qoz_response_lfc_model == "chabrier1990"
    assert ion["residual_history"][-1] <= cfg.hnc_tol
    assert np.min(ion["gii_r"]) >= 0.0
    assert np.min(ion["sii_k"]) > 0.0
