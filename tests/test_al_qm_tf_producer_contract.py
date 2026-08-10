"""Fast contract tests for the expensive Al KS-DFT/TF producer."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCER_PATH = (
    ROOT / "benchmarks" / "runners" / "regenerate_al_qm_tf.py"
)
RUNNER_PATH = ROOT / "benchmarks" / "runners" / "plot_al_qm_tf.py"


def _producer():
    spec = importlib.util.spec_from_file_location(
        "otter_al_qm_tf_producer_contract",
        PRODUCER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner():
    spec = importlib.util.spec_from_file_location(
        "otter_al_qm_tf_runner_contract",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow() -> dict:
    return {
        "electronic": {
            "result": {
                "stage2_converged": True,
                "ext_status": {"iters": 12, "err": 2.0e-6, "converged": True},
                "threshold_state_status": "resolved",
            }
        },
        "ion": {
            "hnc_converged": True,
            "hnc_output_residual": 2.0e-8,
            "closure_transform_max_abs": 3.0e-7,
        },
    }


def test_production_guard_accepts_auditable_qm_and_tf_results() -> None:
    producer = _producer()
    qm = producer._validate_production_result(
        _workflow(),
        temperature_ev=15.0,
        model="qm",
    )
    assert qm["threshold_state_status"] == "resolved"
    assert qm["hnc_output_residual"] == pytest.approx(2.0e-8)

    tf_workflow = _workflow()
    tf_workflow["electronic"]["result"][
        "threshold_state_status"
    ] = "not_applicable_tf"
    tf = producer._validate_production_result(
        tf_workflow,
        temperature_ev=15.0,
        model="tf",
    )
    assert tf["threshold_state_status"] == "not_applicable_tf"


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda result: result["electronic"]["result"].update(
                stage2_converged=False
            ),
            "stage-2",
        ),
        (
            lambda result: result["electronic"]["result"]["ext_status"].update(
                converged=False
            ),
            "external-AA",
        ),
        (
            lambda result: result["electronic"]["result"].update(
                threshold_state_status="unresolved"
            ),
            "threshold status",
        ),
        (
            lambda result: result["ion"].update(hnc_converged=False),
            "physical fixed point",
        ),
        (
            lambda result: result["ion"].update(
                hnc_output_residual=2.0e-5
            ),
            "output residual",
        ),
        (
            lambda result: result["ion"].update(
                closure_transform_max_abs=2.0e-3
            ),
            "closure mismatch",
        ),
    ),
)
def test_production_guard_rejects_each_failed_gate(
    mutation,
    match: str,
) -> None:
    producer = _producer()
    workflow = _workflow()
    mutation(workflow)
    with pytest.raises(RuntimeError, match=match):
        producer._validate_production_result(
            workflow,
            temperature_ev=15.0,
            model="qm",
        )


def _portable_convergence_payload() -> dict[str, np.ndarray]:
    return {
        "schema_version": np.asarray("otter_al_qm_tf_state_v2"),
        "model_labels": np.asarray(("qm", "tf")),
        "aa_stage2_converged": np.asarray((True, True)),
        "aa_ext_converged": np.asarray((True, True)),
        "threshold_state_status": np.asarray(
            ("resolved", "not_applicable_tf")
        ),
        "hnc_converged": np.asarray((True, True)),
        "hnc_output_residual": np.asarray((2.0e-8, 3.0e-8)),
        "closure_transform_max_abs": np.asarray((3.0e-7, 4.0e-7)),
    }


def test_offline_loader_accepts_complete_production_metadata(
    tmp_path: Path,
) -> None:
    runner = _runner()
    path = tmp_path / "state.npz"
    np.savez(path, **_portable_convergence_payload())
    loaded = runner.load_state(path)
    assert np.all(loaded["hnc_converged"])


@pytest.mark.parametrize(
    ("key", "value", "match"),
    (
        (
            "aa_ext_converged",
            np.asarray((True, False)),
            "aa_ext_converged",
        ),
        (
            "threshold_state_status",
            np.asarray(("unresolved", "not_applicable_tf")),
            "threshold",
        ),
        (
            "hnc_output_residual",
            np.asarray((2.0e-8, np.nan)),
            "HNC output residual",
        ),
        (
            "closure_transform_max_abs",
            np.asarray((3.0e-7, 2.0e-3)),
            "closure mismatch",
        ),
    ),
)
def test_offline_loader_rejects_failed_production_metadata(
    tmp_path: Path,
    key: str,
    value: np.ndarray,
    match: str,
) -> None:
    runner = _runner()
    payload = _portable_convergence_payload()
    payload[key] = value
    path = tmp_path / "state.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError, match=match):
        runner.load_state(path)
