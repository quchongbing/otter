"""Fast guards for mixture AA convergence, caches, and downstream use."""

from __future__ import annotations

import numpy as np
import pytest

import otter.electronic.mixture as mixmod
from otter.electronic.mixture import MixtureConfig, solve_mixture_full
from otter.workflows import (
    PlasmaWorkflowConfig,
    _electronic_convergence_issues,
    continue_plasma_workflow_from_electronic_result,
)


def _fake_species_result(cfg_species, *, mu: float, converged: bool) -> dict:
    r = np.linspace(1.0e-3, 4.0, 32)
    return {
        "mu": float(mu),
        "r": r,
        "r_ws": float(cfg_species.r_ws_override_bohr),
        "n0": 0.1,
        "zbar": float(mixmod.element_info(cfg_species.element).z),
        "n_full": np.exp(-r),
        "n_cont": 0.5 * np.exp(-r),
        "n_ion": 0.25 * np.exp(-r),
        "v_full": -np.exp(-r),
        "v_scf": -np.exp(-r),
        "stage2_converged": bool(converged),
        "history": [{"err": 1.0e-6}],
    }


def test_mixture_defaults_to_one_parallel_worker_per_species() -> None:
    cfg = MixtureConfig(
        species=["C", "H"],
        counts=[1.0, 1.36],
        temperature_ev=10.0,
        rho_g_cc=2.94,
    )
    assert cfg.species_parallel_jobs == 2


def test_unconverged_species_result_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exact-theta revisit should retry a failed species AA solve."""
    calls = {"H": 0, "C": 0}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        return _fake_species_result(
            cfg_species,
            mu=float(1.0 / cfg_species.n_i_override_bohr3),
            converged=not (symbol == "H" and calls[symbol] == 1),
        )

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    theta = np.asarray([0.0], dtype=float)
    try:
        first = evaluator.evaluate(theta)
        second = evaluator.evaluate(theta)
    finally:
        evaluator.close()

    assert not mixmod._record_species_results_are_converged(first)
    assert mixmod._record_species_results_are_converged(second)
    assert calls == {"H": 2, "C": 1}


def test_unresolved_threshold_species_is_not_a_root_or_cache_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A box-unresolved shallow level must not manufacture a dmu root."""
    calls = {"H": 0, "C": 0}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        if symbol == "H" and calls[symbol] == 1:
            result["threshold_state_status"] = "unresolved"
        else:
            result["threshold_state_status"] = "resolved"
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
        root_threshold_refine_retry=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    theta = np.asarray([0.0], dtype=float)
    try:
        first = evaluator.evaluate(theta)
        assert not mixmod._record_species_results_are_converged(first)
        assert np.all(np.isfinite(evaluator.residual(theta)))
        second = evaluator.evaluate(theta)
    finally:
        evaluator.close()

    # ``residual`` retried the rejected point, and only the resolved revisit
    # became an exact-theta cache hit.
    assert mixmod._record_species_results_are_converged(second)
    assert calls == {"H": 2, "C": 1}


def test_unresolved_threshold_warm_start_is_retried_cold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A branch-specific shallow pole must not abort an otherwise valid root point."""
    calls: dict[str, list[bool]] = {"H": [], "C": []}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        used_warm_start = cfg_species.v_full_init is not None
        calls[symbol].append(bool(used_warm_start))
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        result = _fake_species_result(
            cfg_species,
            mu=float(np.log(volume)),
            converged=True,
        )
        result["threshold_state_status"] = (
            "unresolved" if symbol == "H" and used_warm_start else "none"
        )
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        first = evaluator.evaluate(np.asarray([0.0], dtype=float))
        second = evaluator.evaluate(np.asarray([0.1], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(first)
    assert mixmod._record_species_results_are_converged(second)
    h_result = dict(second["results"][0])
    assert h_result["threshold_state_status"] == "none"
    assert h_result["mixture_threshold_cold_retry_attempted"] is True
    assert h_result["mixture_threshold_cold_retry_selected"] is True
    assert h_result["mixture_threshold_cold_retry_initial_reasons"] == (
        "threshold_state_unresolved",
    )
    assert calls["H"] == [False, True, False]
    assert calls["C"] == [False, True]
    assert evaluator._species_threshold_cold_retries == 1


def test_threshold_failure_is_retried_with_physical_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shallow full-AA failure gets one cold, physical-domain retry."""
    calls: dict[str, list[dict[str, float | bool | int]]] = {"H": [], "C": []}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol].append({
            "refine": bool(cfg_species.bound_zero_tail_refine),
            "max_binding": float(cfg_species.bound_zero_tail_max_binding_ha),
            "scan_points": int(cfg_species.bound_zero_tail_scan_points),
            "edge_tol": float(cfg_species.bound_zero_tail_edge_rel_tol),
            "mix": float(cfg_species.scf_mix),
            "w0": float(cfg_species.scf_mixing_w0),
            "stage1_max_iter": int(cfg_species.stage1_max_iter),
            "max_iter": int(cfg_species.stage2_max_iter),
            "dn_tol": float(cfg_species.scf_dn_tol),
            "dv_tol": float(cfg_species.scf_dv_tol),
            "warm": bool(cfg_species.v_full_init is not None),
        })
        recovered = symbol != "H" or bool(cfg_species.bound_zero_tail_refine)
        result = _fake_species_result(cfg_species, mu=0.0, converged=recovered)
        result["threshold_state_status"] = "resolved" if recovered else "unresolved"
        result["shallowest_bound_energy_ha"] = -2.0e-3
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(record)
    h_result = dict(record["results"][0])
    assert h_result["mixture_threshold_refine_retry_attempted"] is True
    assert h_result["mixture_threshold_refine_retry_selected"] is True
    assert h_result["mixture_threshold_refine_retry_initial_reasons"] == (
        "stage2_unconverged",
        "threshold_state_unresolved",
    )
    assert len(calls["H"]) == 2
    retry = calls["H"][1]
    assert retry["refine"] is True
    assert retry["max_binding"] >= 1.0e-2
    assert retry["scan_points"] == 24
    assert retry["edge_tol"] == pytest.approx(0.25)
    assert retry["mix"] == pytest.approx(0.15)
    assert retry["w0"] == pytest.approx(5.0e-4)
    assert retry["stage1_max_iter"] == 0
    assert retry["max_iter"] >= 300
    assert retry["dn_tol"] <= 1.0e-6
    assert retry["dv_tol"] <= 1.0e-6
    assert retry["warm"] is False
    assert evaluator._species_threshold_refine_retries == 1


def test_nonthreshold_scf_failure_does_not_use_threshold_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guarded retry must not modify an ordinary failed AA point."""
    calls = {"H": 0, "C": 0}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        result = _fake_species_result(
            cfg_species, mu=0.0, converged=symbol != "H"
        )
        result["threshold_state_status"] = "resolved"
        result["shallowest_bound_energy_ha"] = -1.0
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert not mixmod._record_species_results_are_converged(record)
    assert calls == {"H": 1, "C": 1}
    assert evaluator._species_threshold_refine_retries == 0


def test_bound_charge_branch_flips_trigger_threshold_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final continuum frame must not hide repeated threshold crossings."""
    calls = {"H": 0, "C": 0}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        recovered = symbol != "H" or bool(cfg_species.bound_zero_tail_refine)
        result = _fake_species_result(cfg_species, mu=0.0, converged=recovered)
        result["threshold_state_status"] = "resolved"
        result["shallowest_bound_energy_ha"] = -1.0
        if symbol == "H" and not recovered:
            charges = [2.0, 2.24, 2.0, 2.24, 2.0, 2.24, 2.0, 2.24]
            result["history"] = [
                {"err": 1.0e-2, "charge_bound": value}
                for value in charges
            ]
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(record)
    assert calls == {"H": 2, "C": 1}
    h_result = dict(record["results"][0])
    assert h_result["mixture_threshold_refine_retry_selected"] is True


def test_common_mu_evaluator_uses_full_aa_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External AA is not part of the common-chemical-potential root."""
    calls: list[tuple[str, bool]] = []

    def _fake_full(cfg_species):
        calls.append((str(cfg_species.run_mode), bool(cfg_species.ext_scf_enabled)))
        return _fake_species_result(cfg_species, mu=0.0, converged=True)

    def _forbid_external(_cfg_species):
        raise AssertionError("common-mu evaluation called external AA")

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    monkeypatch.setattr(mixmod, "solve_full_then_external", _forbid_external)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(record)
    assert calls == [("full", False), ("full", False)]


def test_unresolved_auto_b3_threshold_is_retried_with_a_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolved oscillatory auto-B3 pole gets one audited simpler retry."""
    calls: dict[str, list[str]] = {"H": [], "C": []}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        model = str(cfg_species.b3_tail_model)
        calls[symbol].append(model)
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        selected = "full" if model == "auto" else model
        result["n_full_tail_meta"] = {"model_selected": selected}
        result["n_cont_tail_meta"] = {"applied": False}
        result["threshold_state_status"] = (
            "unresolved" if symbol == "H" and selected == "full" else "none"
        )
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
        aa_overrides={
            "b3_tail_model": "auto",
            "b3_tail_target": "full",
        },
        root_threshold_refine_retry=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(record)
    h_result = dict(record["results"][0])
    assert h_result["threshold_state_status"] == "none"
    assert h_result["mixture_threshold_b3_a_only_retry_attempted"] is True
    assert h_result["mixture_threshold_b3_a_only_retry_selected"] is True
    assert h_result["mixture_threshold_b3_tail_meta_source"] == "n_full_tail_meta"
    assert h_result["mixture_threshold_b3_a_only_retry_initial_reasons"] == (
        "threshold_state_unresolved",
    )
    assert calls["H"] == ["auto", "a_only"]
    assert calls["C"] == ["auto"]
    assert evaluator._species_threshold_b3_a_only_retries == 1


def test_b3_tail_meta_router_prefers_target_owner_then_compatibility_fallback() -> None:
    """Full/cont targets read their own fit metadata without breaking old files."""
    result = {
        "n_full_tail_meta": {"model_selected": "full", "target": "full"},
        "n_cont_tail_meta": {"model_selected": "a_only", "target": "cont"},
    }

    meta, source = mixmod._b3_tail_meta_for_target(
        result, b3_tail_target="full"
    )
    assert source == "n_full_tail_meta"
    assert meta["model_selected"] == "full"

    meta, source = mixmod._b3_tail_meta_for_target(
        result, b3_tail_target="both"
    )
    assert source == "n_full_tail_meta"
    assert meta["model_selected"] == "full"

    meta, source = mixmod._b3_tail_meta_for_target(
        result, b3_tail_target="cont"
    )
    assert source == "n_cont_tail_meta"
    assert meta["model_selected"] == "a_only"

    meta, source = mixmod._b3_tail_meta_for_target(
        {
            "n_full_tail_meta": {"applied": False},
            "n_cont_tail_meta": {"model_selected": "full"},
        },
        b3_tail_target="full",
    )
    assert source == "n_cont_tail_meta"
    assert meta["model_selected"] == "full"


def test_explicit_full_b3_threshold_uses_root_only_a_only_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full-B3 threshold failure may guide theta without changing the model."""
    calls: dict[str, list[str]] = {"H": [], "C": []}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        model = str(cfg_species.b3_tail_model)
        calls[symbol].append(model)
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        result["n_full_tail_meta"] = {"model_selected": model}
        result["n_cont_tail_meta"] = {"applied": False}
        result["threshold_state_status"] = (
            "unresolved" if symbol == "H" and model == "full" else "none"
        )
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
        aa_overrides={
            "b3_tail_model": "full",
            "b3_tail_target": "full",
        },
        root_threshold_refine_retry=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert mixmod._record_species_results_are_converged(record)
    assert record["root_uses_b3_surrogate"] is True
    h_result = dict(record["results"][0])
    assert h_result["mixture_threshold_b3_a_only_retry_attempted"] is True
    assert h_result["mixture_threshold_b3_a_only_retry_selected"] is True
    assert h_result["mixture_threshold_b3_a_only_retry_role"] == "root_surrogate"
    assert h_result["mixture_threshold_b3_tail_meta_source"] == "n_full_tail_meta"
    assert calls == {"H": ["full", "a_only"], "C": ["full"]}
    # The A-only surrogate must not enter either requested-model species cache.
    assert not any(key[0] == "H" for key in evaluator._species_result_cache)
    assert evaluator._species_init_cache["H"] == []


def test_explicit_full_b3_threshold_surrogate_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strict policy preserves the old all-full root eligibility guard."""
    calls = {"H": 0, "C": 0}

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        result["n_full_tail_meta"] = {"model_selected": "full"}
        result["n_cont_tail_meta"] = {"applied": False}
        result["threshold_state_status"] = (
            "unresolved" if symbol == "H" else "none"
        )
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
        aa_overrides={
            "b3_tail_model": "full",
            "b3_tail_target": "full",
        },
        root_threshold_b3_surrogate_mode="off",
        root_threshold_refine_retry=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        record = evaluator.evaluate(np.asarray([0.0], dtype=float))
    finally:
        evaluator.close()

    assert not mixmod._record_species_results_are_converged(record)
    h_result = dict(record["results"][0])
    assert h_result["mixture_threshold_b3_a_only_retry_attempted"] is False
    assert calls == {"H": 1, "C": 1}


def test_full_b3_root_surrogate_is_verified_with_requested_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxy root is returned only after an independent all-full AA solve."""
    h_full_calls = 0

    def _fake_full(cfg_species):
        nonlocal h_full_calls
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        model = str(cfg_species.b3_tail_model)
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        result["test_b3_model"] = model
        result["n_full_tail_meta"] = {"model_selected": model}
        result["n_cont_tail_meta"] = {"applied": False}
        if symbol == "H" and model == "full":
            h_full_calls += 1
            result["threshold_state_status"] = (
                "unresolved" if h_full_calls == 1 else "none"
            )
        else:
            result["threshold_state_status"] = "none"
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
        aa_overrides={
            "b3_tail_model": "full",
            "b3_tail_target": "full",
        },
        volume_weights_init=[0.5, 0.5],
        root_threshold_refine_retry=False,
    )

    result = mixmod.solve_mixture_full_only(cfg)

    assert result["meta"]["root_threshold_b3_surrogate_used"] is True
    assert result["meta"]["root_threshold_b3_full_verification_success"] is True
    assert result["meta"]["root_threshold_b3_surrogate_retries"] == 1
    assert result["meta"]["root_threshold_b3_a_only_retries"] == 1
    assert all(
        sp["result"]["test_b3_model"] == "full"
        for sp in result["species"]
    )


def test_unverified_full_b3_root_surrogate_is_never_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A converged A-only proxy cannot cross the electronic/QOZ boundary."""

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        model = str(cfg_species.b3_tail_model)
        result = _fake_species_result(cfg_species, mu=0.0, converged=True)
        result["n_full_tail_meta"] = {"model_selected": model}
        result["n_cont_tail_meta"] = {"applied": False}
        result["threshold_state_status"] = (
            "unresolved" if symbol == "H" and model == "full" else "none"
        )
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
        aa_overrides={
            "b3_tail_model": "full",
            "b3_tail_target": "full",
        },
        volume_weights_init=[0.5, 0.5],
    )

    with pytest.raises(RuntimeError, match="surrogate was not returned"):
        mixmod.solve_mixture_full_only(cfg)


def test_species_eligibility_reasons_cover_threshold_and_nonfinite_mu() -> None:
    eligible, reasons = mixmod._species_result_eligibility({
        "stage2_converged": True,
        "mu": np.nan,
        "threshold_state_status": "unresolved",
    })
    assert not eligible
    assert reasons == ("mu_nonfinite", "threshold_state_unresolved")

    assert mixmod._species_result_eligibility({
        "stage2_converged": True,
        "mu": 0.0,
        "threshold_state_status": "marginal",
    }) == (True, ())

    # Old analytic evaluators with no production-only metadata remain usable.
    assert mixmod._species_result_eligibility({"value": 1.0}) == (True, ())


def test_production_external_eligibility_rejects_post_scf_b3() -> None:
    result = {
        "stage2_converged": True,
        "mu": 0.0,
        "threshold_state_status": "none",
        "n_scr": np.ones(8),
        "ext_status": {"enabled": True, "converged": True},
        "b3_post_self_consistent": False,
        "meta": {"b3_tail_stage2_mode": "post"},
    }

    eligible, reasons = mixmod._species_result_eligibility(
        result, require_external=True
    )

    assert not eligible
    assert reasons == ("b3_post_diagnostic_only",)


def test_production_eligibility_rejects_unapplied_requested_full_b3() -> None:
    result = {
        "stage2_converged": True,
        "mu": 0.0,
        "threshold_state_status": "none",
        "n_scr": np.ones(8),
        "n_full_tail_meta": {"applied": False},
        "n_ext_tail_meta": {"applied": False},
        "ext_status": {"enabled": True, "converged": True},
        "meta": {
            "b3_tail_stage2_mode": "in_scf",
            "b3_tail_target": "full",
        },
    }

    root_eligible, root_reasons = mixmod._species_result_eligibility(result)
    qoz_eligible, qoz_reasons = mixmod._species_result_eligibility(
        result, require_external=True
    )

    assert not root_eligible
    assert root_reasons == ("b3_full_tail_unapplied",)
    assert not qoz_eligible
    assert qoz_reasons == (
        "b3_full_tail_unapplied",
        "b3_external_tail_unapplied",
    )


def test_final_species_config_preserves_selected_a_only_threshold_retry() -> None:
    """The post-root full+external solve must not recreate the rejected tail."""
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, final_run_mode="full+ext", save_data=False,
        aa_overrides={"b3_tail_model": "auto"},
    )
    full_result = {
        "r": np.asarray([0.1, 1.0, 2.0]),
        "v_full": np.asarray([-1.0, -0.1, 0.0]),
        "mu": 0.15,
        "mixture_threshold_b3_a_only_retry_selected": True,
    }

    species_cfg = mixmod._final_species_config(
        cfg,
        element_key="H",
        r_ws_bohr=2.0,
        n_i_bohr3=3.0 / (4.0 * np.pi * 2.0**3),
        extra_overrides={},
        full_result_init=full_result,
    )

    assert species_cfg.b3_tail_model == "a_only"
    assert species_cfg.v_full_init is not None


def test_final_species_config_preserves_selected_threshold_refinement() -> None:
    """The post-root full+external solve must retain the resolved shallow pole."""
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, final_run_mode="full+ext", save_data=False,
    )
    full_result = {
        "r": np.asarray([0.1, 1.0, 2.0]),
        "v_full": np.asarray([-1.0, -0.1, 0.0]),
        "mu": 0.25,
        "mixture_threshold_refine_retry_selected": True,
    }

    species_cfg = mixmod._final_species_config(
        cfg,
        element_key="C",
        r_ws_bohr=2.0,
        n_i_bohr3=3.0 / (4.0 * np.pi * 2.0**3),
        extra_overrides={},
        full_result_init=full_result,
    )

    assert species_cfg.bound_zero_tail_refine is True
    assert species_cfg.bound_zero_tail_max_binding_ha >= 1.0e-2
    assert species_cfg.stage2_max_iter >= 300
    assert species_cfg.scf_dn_tol <= 1.0e-6
    assert species_cfg.scf_dv_tol <= 1.0e-6
    assert species_cfg.stage1_max_iter == 0
    assert species_cfg.continuation_stage2_from_init is True
    assert species_cfg.continuation_mu_init == pytest.approx(0.25)
    assert species_cfg.scf_mix == pytest.approx(0.15)
    assert species_cfg.scf_mixing_w0 == pytest.approx(5.0e-4)


def test_final_species_config_keeps_screening_tail_fit_off_by_default() -> None:
    """A mixture must not silently replace one species' canonical n_scr tail."""
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=2.0,
        rho_g_cc=0.94, final_run_mode="full+ext", save_data=False,
    )

    species_cfg = mixmod._final_species_config(
        cfg,
        element_key="H",
        r_ws_bohr=2.0,
        n_i_bohr3=3.0 / (4.0 * np.pi * 2.0**3),
        extra_overrides={},
        full_result_init=None,
    )

    assert species_cfg.screening_tail_repair_mode == "off"


def test_direct_residual_hides_unresolved_threshold_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_full(cfg_species):
        result = _fake_species_result(cfg_species, mu=0.1, converged=True)
        if str(mixmod.element_info(cfg_species.element).symbol) == "H":
            result["threshold_state_status"] = "unresolved"
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, species_parallel_jobs=1, save_data=False,
    )
    evaluator = mixmod._MixtureEvaluator(cfg)
    try:
        assert np.all(np.isnan(evaluator.residual(np.asarray([0.0]))))
    finally:
        evaluator.close()


def test_unconverged_species_result_is_not_used_by_mu_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed inner AA point must not enter the tabulated mu_i(V_i) data."""
    calls = {"H": 0, "C": 0}
    recorded_points: list[dict] = []
    original_record = mixmod._record_species_samples

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        calls[symbol] += 1
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        return _fake_species_result(
            cfg_species,
            mu=float(np.log(volume) + (0.025 if symbol == "H" else 0.0)),
            converged=not (symbol == "H" and calls[symbol] == 1),
        )

    def _record_only_converged(samples, record):
        assert mixmod._record_species_results_are_converged(record)
        recorded_points.append(record)
        original_record(samples, record)

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    monkeypatch.setattr(mixmod, "_record_species_samples", _record_only_converged)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, root_maxfev=12, save_data=False,
    )
    result = solve_mixture_full(cfg)

    assert bool(result["meta"]["root_success"])
    assert recorded_points


def test_unconverged_common_mu_raises_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production mixture solves must not silently return an invalid closure."""

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        return _fake_species_result(
            cfg_species,
            mu=0.0 if symbol == "H" else 1.0,
            converged=True,
        )

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, root_maxfev=4, save_data=False,
    )
    with pytest.raises(RuntimeError, match="common-mu solve did not converge"):
        solve_mixture_full(cfg)


def test_best_effort_common_mu_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Diagnostics can still request the former best-available behavior."""

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        return _fake_species_result(
            cfg_species,
            mu=0.0 if symbol == "H" else 1.0,
            converged=True,
        )

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, root_maxfev=4, allow_unconverged_root=True,
        save_data=False,
    )
    result = solve_mixture_full(cfg)
    assert not bool(result["meta"]["root_success"])
    assert np.isclose(float(result["meta"]["mu_residual_max_ha"]), 1.0)


def test_binary_explicit_seed_expands_when_local_points_do_not_bracket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A binary warm start must not disable the remaining global root budget."""

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        # For equal fractions, theta=log(V_H/V_C).  The true root is at
        # theta=0.4, outside the explicit seed neighborhood +/-0.1 but inside
        # the bracket supplied by the broader physical seed set.
        mu = float(np.log(volume) + (0.4 if symbol == "H" else 0.0))
        return _fake_species_result(cfg_species, mu=mu, converged=True)

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        volume_weights_init=[0.5, 0.5],
        root_maxfev=12,
        save_data=False,
    )
    result = solve_mixture_full(cfg)

    assert bool(result["meta"]["root_success"])
    assert float(result["meta"]["mu_residual_max_ha"]) <= float(cfg.mu_e_tol)
    assert int(result["meta"]["root_nfev"]) > 5
    assert result["meta"]["root_method"] == "binary_observed_bracket_mu_tolerance"


def test_binary_root_probes_valid_side_of_unconverged_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed seed may guide edge probes but must never enter the bracket."""
    root_theta = 0.70
    invalid_above = 0.80
    x_c = 0.5
    x_h = 0.5
    avg_mass = 0.5 * (
        float(mixmod.element_info("C").atomic_mass)
        + float(mixmod.element_info("H").atomic_mass)
    )
    vbar = 1.0 / float(mixmod.ion_density_bohr3(1.0, avg_mass))

    def _theta_from_species_volume(symbol: str, volume: float) -> float:
        if symbol == "C":
            w_c = float(np.clip(x_c * volume / vbar, 1.0e-12, 1.0 - 1.0e-12))
            return float(np.log(w_c / (1.0 - w_c)))
        w_h = float(np.clip(x_h * volume / vbar, 1.0e-12, 1.0 - 1.0e-12))
        return float(np.log((1.0 - w_h) / w_h))

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        theta = _theta_from_species_volume(symbol, volume)
        residual = root_theta - theta
        converged = not (symbol == "C" and theta > invalid_above)
        return _fake_species_result(
            cfg_species,
            mu=float(residual if symbol == "C" else 0.0),
            converged=converged,
        )

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["C", "H"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        root_maxfev=20,
        root_brent_maxiter=16,
        save_data=False,
    )

    result = solve_mixture_full(cfg)

    assert bool(result["meta"]["root_success"])
    assert abs(float(result["theta"][0]) - root_theta) < 1.0e-3
    assert int(result["meta"]["root_n_invalid_inner"]) >= 1
    assert all(
        not (
            bool(row.get("root_eligible", False))
            and np.log(
                float(row["weight_C"]) / float(row["weight_H"])
            ) > invalid_above
        )
        for row in result["history"]
    )


def test_binary_seed_loop_stops_as_soon_as_a_bracket_is_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not spend expensive AA evaluations on unused seeds after bracketing."""

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        mu = float(np.log(volume) + (0.025 if symbol == "H" else 0.0))
        return _fake_species_result(cfg_species, mu=mu, converged=True)

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        volume_weights_init=[0.5, 0.5],
        root_maxfev=12,
        save_data=False,
    )
    result = solve_mixture_full(cfg)

    assert bool(result["meta"]["root_success"])
    assert int(result["meta"]["root_n_seed_evals"]) == 2
    assert int(result["meta"]["root_nfev"]) == 3


def test_binary_brent_has_a_separate_budget_after_last_primary_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bracket found on root_maxfev must still receive Brent iterations."""
    root_theta = -0.4

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        mu = float(np.log(volume) + (0.0 if symbol == "H" else root_theta))
        return _fake_species_result(cfg_species, mu=mu, converged=True)

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["H", "C"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        root_maxfev=4,
        root_brent_maxiter=12,
        save_data=False,
    )

    result = solve_mixture_full(cfg)

    assert bool(result["meta"]["root_success"])
    assert abs(float(result["theta"][0]) - root_theta) < 2.0e-4
    assert int(result["meta"]["root_n_seed_evals"]) == 4
    assert int(result["meta"]["root_nfev"]) > int(cfg.root_maxfev)
    assert int(result["meta"]["root_brent_maxiter"]) == 12


def test_binary_brent_recovers_valid_subbracket_beside_invalid_aa_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A narrow rejected pressure-ionization interval need not hide a valid root."""
    root_theta = 0.2605
    x_c = 1.0 / 3.0
    x_h = 2.0 / 3.0
    avg_mass = (
        float(mixmod.element_info("C").atomic_mass)
        + 2.0 * float(mixmod.element_info("H").atomic_mass)
    ) / 3.0
    vbar = 1.0 / float(mixmod.ion_density_bohr3(0.94, avg_mass))

    def _theta_from_species_volume(symbol: str, volume: float) -> float:
        if symbol == "C":
            w_c = float(np.clip(x_c * volume / vbar, 1.0e-12, 1.0 - 1.0e-12))
            return float(np.log(w_c / (1.0 - w_c)))
        w_h = float(np.clip(x_h * volume / vbar, 1.0e-12, 1.0 - 1.0e-12))
        return float(np.log((1.0 - w_h) / w_h))

    def _fake_full(cfg_species):
        symbol = str(mixmod.element_info(cfg_species.element).symbol)
        volume = 1.0 / float(cfg_species.n_i_override_bohr3)
        theta = _theta_from_species_volume(symbol, volume)
        # Positive scale with unequal endpoint values makes the first secant
        # land inside the rejected interval, while preserving one true root.
        scale = 0.363 + 0.11945 * (theta - 0.202733)
        residual = (root_theta - theta) * scale
        invalid = bool(symbol == "H" and 0.245 < theta < 0.252)
        result = _fake_species_result(
            cfg_species,
            mu=float(residual if symbol == "C" else 0.0),
            converged=not invalid,
        )
        result["threshold_state_status"] = "unresolved" if invalid else "none"
        return result

    monkeypatch.setattr(mixmod, "solve_full_only", _fake_full)
    cfg = MixtureConfig(
        species=["C", "H"],
        counts=[1.0, 2.0],
        temperature_ev=10.0,
        rho_g_cc=0.94,
        root_maxfev=8,
        root_brent_maxiter=16,
        save_data=False,
    )

    result = solve_mixture_full(cfg)

    assert bool(result["meta"]["root_success"])
    assert float(result["meta"]["mu_residual_max_ha"]) <= float(cfg.mu_e_tol)
    assert abs(float(result["theta"][0]) - root_theta) < 1.0e-3
    assert int(result["meta"]["root_n_invalid_inner"]) >= 1
    assert str(result["meta"]["root_method"]).startswith("binary_invalid_gap")


def test_final_electronic_failures_are_identified_before_qoz() -> None:
    """Both the full and external final stages are part of QOZ provenance."""
    issues = _electronic_convergence_issues([
        {
            "element": "C",
            "result": {
                "stage2_converged": False,
                "ext_status": {"converged": False},
            },
        },
        {
            "element": "H",
            "result": {
                "stage2_converged": True,
                "ext_status": {"converged": True},
            },
        },
    ])
    assert issues == [
        "C: full AA stage-2 unconverged",
        "C: external fixed-mu SCF unconverged",
    ]


def test_unresolved_threshold_state_is_identified_before_qoz() -> None:
    issues = _electronic_convergence_issues(
        [{
            "element": "H",
            "result": {
                "stage2_converged": True,
                "mu": -0.01,
                "threshold_state_status": "unresolved",
                "ext_status": {"converged": True},
            },
        }],
        require_external=True,
    )
    assert issues == ["H: unresolved threshold bound state"]


def test_final_rerun_mu_residual_is_rejected_before_qoz() -> None:
    cfg = PlasmaWorkflowConfig(
        elements=["C", "H"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        ion_temperature_ev=10.0,
    )
    electronic = {
        "meta": {
            "root_success": True,
            "mu_residual_max_ha": 1.0e-8,
            "final_mu_root_success": False,
            "final_mu_residual_max_ha": 2.0e-2,
        },
        "species": [
            {"element": "C", "result": {}},
            {"element": "H", "result": {}},
        ],
    }
    with pytest.raises(RuntimeError, match=r"final full\+external rerun lost common-mu"):
        continue_plasma_workflow_from_electronic_result(
            cfg,
            electronic_kind="mixture",
            electronic_result=electronic,
        )


def test_final_unresolved_threshold_state_is_rejected_before_qoz() -> None:
    cfg = PlasmaWorkflowConfig(
        elements=["C", "H"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        rho_g_cc=1.0,
        ion_temperature_ev=10.0,
    )
    electronic = {
        "meta": {
            "root_success": True,
            "final_mu_root_success": True,
            "final_mu_residual_max_ha": 1.0e-8,
        },
        "species": [
            {
                "element": "C",
                "result": {
                    "stage2_converged": True,
                    "mu": 0.0,
                    "threshold_state_status": "resolved",
                    "ext_status": {"converged": True},
                },
            },
            {
                "element": "H",
                "result": {
                    "stage2_converged": True,
                    "mu": 0.0,
                    "threshold_state_status": "unresolved",
                    "ext_status": {"converged": True},
                },
            },
        ],
    }
    with pytest.raises(RuntimeError, match="unresolved threshold bound state"):
        continue_plasma_workflow_from_electronic_result(
            cfg,
            electronic_kind="mixture",
            electronic_result=electronic,
        )


def test_mixture_final_rerun_records_electronic_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_result = {
        "r": np.asarray([0.1, 1.0]),
        "v_full": np.asarray([-1.0, 0.0]),
        "stage2_converged": True,
        "mu": 0.0,
        "threshold_state_status": "resolved",
    }
    mixture_full = {
        "mu_common_ha": 0.0,
        "theta": np.asarray([0.0]),
        "volume_weights": np.asarray([0.5, 0.5]),
        "history": [],
        "species": [
            {
                "element": "H", "Z": 1, "atomic_mass": 1.0,
                "count": 1.0, "x": 0.5, "volume_bohr3": 1.0,
                "r_ws_bohr": 1.0, "mu_ha": 0.0, "result": dict(base_result),
            },
            {
                "element": "C", "Z": 6, "atomic_mass": 12.0,
                "count": 1.0, "x": 0.5, "volume_bohr3": 1.0,
                "r_ws_bohr": 1.0, "mu_ha": 0.0, "result": dict(base_result),
            },
        ],
        "meta": {"root_success": True, "mu_residual_max_ha": 0.0},
    }

    monkeypatch.setattr(mixmod, "_mixture_full_only_payload", lambda cfg: mixture_full)
    monkeypatch.setattr(
        mixmod,
        "_final_species_config",
        lambda cfg, **kwargs: int(kwargs["element_key"]),
    )

    def _fake_final(z):
        return {
            "stage2_converged": True,
            "mu": 0.02 if int(z) == 1 else 0.0,
            "threshold_state_status": "unresolved" if int(z) == 1 else "resolved",
            "ext_status": {"converged": True},
        }

    monkeypatch.setattr(mixmod, "_solve_species_from_config", _fake_final)
    cfg = MixtureConfig(
        species=["H", "C"], counts=[1.0, 1.0], temperature_ev=10.0,
        rho_g_cc=1.0, final_run_mode="full+ext", species_parallel_jobs=1,
        save_data=False,
    )
    result = mixmod.solve_mixture_full_then_ext(cfg)

    assert not bool(result["meta"]["final_mu_root_success"])
    assert not bool(result["meta"]["final_electronic_eligible"])
    assert "H:threshold_state_unresolved" in result["meta"]["final_electronic_issues"]
    assert "mixture:final_mu_residual_above_tolerance" in result["meta"]["final_electronic_issues"]
