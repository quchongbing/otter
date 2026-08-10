"""Fast offline checks for the reviewed Otter v2 numerical packages."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "al_qm_tf": {
        "directory": ROOT / "benchmarks" / "baselines" / "al_qm_tf",
        "runner": ROOT / "benchmarks" / "runners" / "plot_al_qm_tf.py",
        "producer": (
            ROOT / "benchmarks" / "runners" / "regenerate_al_qm_tf.py"
        ),
        "state_schema": "otter_al_qm_tf_state_v2",
        "temperatures_ev": (1.0, 15.0, 50.0, 100.0),
        "metric_rows": 4,
    },
    "carbon_lfc_sensitivity": {
        "directory": (
            ROOT / "benchmarks" / "baselines" / "carbon_lfc_sensitivity"
        ),
        "runner": (
            ROOT
            / "benchmarks"
            / "runners"
            / "plot_carbon_lfc_sensitivity.py"
        ),
        "producer": (
            ROOT
            / "benchmarks"
            / "runners"
            / "regenerate_carbon_lfc_sensitivity.py"
        ),
        "state_schema": "otter_carbon_lfc_sensitivity_state_v2",
        "temperatures_ev": (2.0, 100.0),
        "metric_rows": 10,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(package: dict) -> dict:
    return json.loads(
        (package["directory"] / "manifest.json").read_text(encoding="utf-8")
    )


def _load_runner(name: str, package: dict) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"otter_{name}_offline_runner", package["runner"]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {package['runner']}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_path(package: dict, state: dict) -> Path:
    relative = Path(state["data_file"])
    assert not relative.is_absolute()
    path = (package["directory"] / relative).resolve()
    assert path.parent == package["directory"].resolve()
    return path


def _assert_portable_archive(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            value = archive[key]
            assert not value.dtype.hasobject, (path, key)
            if value.dtype.kind in "fiu":
                assert np.all(np.isfinite(value)), (path, key)
            if value.dtype.kind in "SU":
                text = " ".join(str(item) for item in value.reshape(-1))
                assert "/home/" not in text
                assert "/tmp/" not in text


@pytest.mark.parametrize("name", tuple(PACKAGES))
def test_v2_manifest_provenance_hashes_and_relative_paths(name: str) -> None:
    package = PACKAGES[name]
    manifest = _manifest(package)
    assert manifest["schema_version"] == "otter_benchmark_manifest_v2"
    assert manifest["benchmark_id"] == name
    assert manifest["data_rights"] == {
        "origin_type": "project_generated_numerical_output",
        "third_party_reference_data_included": False,
        "method_citations_only": True,
    }
    assert tuple(
        float(state["temperature_ev"]) for state in manifest["states"]
    ) == package["temperatures_ev"]
    producer = manifest["producer"]
    assert producer["project"] == "Otter"
    assert len(producer["git_commit"]) == 40
    assert len(producer["script_sha256"]) == 64
    generation_hash = producer.get(
        "script_sha256_at_generation",
        producer["script_sha256"],
    )
    current_hash = producer.get(
        "script_sha256_current",
        producer["script_sha256"],
    )
    assert producer["worktree_clean_at_generation"] is True
    assert current_hash == _sha256(package["producer"])
    serialized = json.dumps(manifest)
    assert "/home/" not in serialized
    assert "/tmp/" not in serialized
    metrics_path = package["directory"] / manifest["metrics_file"]
    assert _sha256(metrics_path) == manifest["metrics_sha256"]

    for state in manifest["states"]:
        path = _state_path(package, state)
        assert _sha256(path) == state["data_sha256"]
        _assert_portable_archive(path)
        with np.load(path, allow_pickle=False) as archive:
            assert archive["schema_version"].item() == package["state_schema"]
            assert archive["benchmark_id"].item() == name
            assert archive["otter_git_commit"].item() == producer["git_commit"]
            assert (
                archive["producer_script_sha256"].item()
                == generation_hash
            )
            assert bool(archive["producer_worktree_clean"].item())
            assert np.isclose(
                float(archive["rho_g_cc"]),
                float(state["rho_g_cc"]),
            )
            assert np.isclose(
                float(archive["temperature_ev"]),
                float(state["temperature_ev"]),
            )


def test_al_qm_tf_v2_physical_and_shape_invariants() -> None:
    package = PACKAGES["al_qm_tf"]
    manifest = _manifest(package)
    assert manifest["configuration"]["rho_g_cc"] == pytest.approx(8.1)
    assert manifest["configuration"]["bound_rmax_mult"] is None
    assert manifest["configuration"]["bound_zero_tail_refine"] is False
    assert manifest["configuration"]["lfc_model"] == "chabrier1990"
    assert (
        manifest["units"]["effective_pair_potential_k"]
        == "Hartree Bohr^3"
    )
    for state in manifest["states"]:
        with np.load(_state_path(package, state), allow_pickle=False) as archive:
            assert tuple(archive["model_labels"]) == ("qm", "tf")
            assert archive["lfc_model"].item() == "chabrier1990"
            signature = json.loads(
                str(archive["producer_signature_json"].item())
            )
            assert signature["electronic_models"] == ["qm", "tf"]
            assert signature["structure_model"] == "IS"
            assert signature["bound_occ_mode"] == "fd"
            assert signature["bound_rmax_mult"] is None
            assert signature["bound_zero_tail_refine"] is False

            r = np.asarray(archive["r_ion_bohr"], dtype=float)
            k = np.asarray(archive["k_bohr_inv"], dtype=float)
            assert np.all(np.diff(r) > 0.0)
            assert np.all(np.diff(k) > 0.0)
            assert r[-1] <= 20.0
            assert k[-1] <= 20.0
            assert archive["gii_r"].shape == (2, r.size)
            assert archive["sii_k"].shape == (2, k.size)
            assert archive["vii_k_ha_bohr3"].shape == (2, k.size)
            assert np.all(archive["aa_stage2_converged"])
            assert np.all(archive["aa_ext_converged"])
            assert np.all(archive["hnc_converged"])
            assert tuple(archive["threshold_state_status"]) == (
                archive["threshold_state_status"][0],
                "not_applicable_tf",
            )
            assert archive["threshold_state_status"][0] in {
                "none",
                "resolved",
                "marginal",
            }
            assert np.max(archive["hnc_residual"]) <= 1.0e-6
            assert np.max(archive["hnc_output_residual"]) <= 1.0e-6
            assert np.max(archive["closure_transform_max_abs"]) <= 1.0e-3
            np.testing.assert_allclose(
                archive["q_scr_integral"],
                archive["zbar_partition"],
                rtol=0.0,
                atol=1.0e-2,
            )
            for model in ("qm", "tf"):
                r_e = np.asarray(archive[f"r_{model}_bohr"], dtype=float)
                assert np.all(np.diff(r_e) > 0.0)
                assert r_e[-1] <= 20.0
                for density in (
                    "n_full",
                    "n_ext",
                    "n_pa",
                    "n_bound",
                    "n_cont",
                    "n_ion",
                    "n_scr",
                ):
                    assert archive[
                        f"{density}_{model}_bohr3"
                    ].shape == r_e.shape


def test_carbon_lfc_v2_shared_input_charge_and_convergence() -> None:
    package = PACKAGES["carbon_lfc_sensitivity"]
    manifest = _manifest(package)
    configuration = manifest["configuration"]
    assert configuration["rho_g_cc"] == pytest.approx(5.0)
    assert configuration["bound_occ_mode"] == "fd"
    assert configuration["bound_rmax_mult"] is None
    assert configuration["bound_zero_tail_matching_mode"] == (
        "direct_physical_boundary"
    )
    assert configuration["b3_tail_model"] == "full"
    expected_models = (
        "none",
        "hubbard",
        "utsumiichimaru",
        "chabrier1990",
        "gregori2007",
    )
    for state in manifest["states"]:
        with np.load(_state_path(package, state), allow_pickle=False) as archive:
            assert tuple(archive["model_labels"]) == expected_models
            assert archive["reference_model"].item() == "chabrier1990"
            signature = json.loads(
                str(archive["producer_signature_json"].item())
            )
            assert signature["structure_model"] == "IS"
            assert signature["bound_occ_mode"] == "fd"
            assert signature["bound_rmax_mult"] is None
            assert signature["b3_tail_model"] == "full"
            r = np.asarray(archive["r_bohr"], dtype=float)
            k = np.asarray(archive["k_bohr_inv"], dtype=float)
            r_e = np.asarray(archive["electronic_r_bohr"], dtype=float)
            assert np.all(np.diff(r) > 0.0)
            assert np.all(np.diff(k) > 0.0)
            assert np.all(np.diff(r_e) > 0.0)
            assert r[-1] <= 20.0
            assert k[-1] <= 20.0
            assert r_e[-1] <= 20.0
            assert archive["gii_r"].shape == (5, r.size)
            assert archive["sii_k"].shape == (5, k.size)
            assert archive["vii_k_ha_bohr3"].shape == (5, k.size)
            assert archive["gee_k"].shape == (5, k.size)
            assert bool(archive["aa_stage2_converged"].item())
            assert bool(archive["aa_ext_converged"].item())
            assert archive["threshold_state_status"].item() in {
                "resolved",
                "marginal",
            }
            assert np.max(archive["hnc_output_residual"]) < 1.0e-5
            assert np.max(archive["closure_transform_max_abs"]) < 1.0e-4
            np.testing.assert_allclose(
                archive["zbar_qoz"],
                float(archive["zbar_partition"]),
                rtol=0.0,
                atol=2.0e-12,
            )


@pytest.mark.parametrize("name", tuple(PACKAGES))
def test_offline_runner_evaluates_reviewed_v2_data_from_arbitrary_cwd(
    name: str,
    monkeypatch,
    tmp_path,
) -> None:
    package = PACKAGES[name]
    monkeypatch.chdir(tmp_path)
    runner_source = package["runner"].read_text(encoding="utf-8")
    assert "solve_plasma_workflow" not in runner_source
    assert "from otter" not in runner_source
    runner = _load_runner(name, package)
    manifest = runner.load_manifest(package["directory"] / "manifest.json")
    rows, loaded = runner.evaluate_states(
        manifest,
        data_dir=package["directory"],
    )
    assert len(rows) == package["metric_rows"]
    assert len(loaded) == len(package["temperatures_ev"])
    for row in rows:
        for key, value in row.items():
            if key not in {"model", "data_file", "baseline_file"}:
                assert np.isfinite(float(value)), (name, key)


def test_v2_curated_states_capture_cold_difference_and_hot_convergence() -> None:
    al_package = PACKAGES["al_qm_tf"]
    al_runner = _load_runner("al_qm_tf_metrics", al_package)
    al_rows, _ = al_runner.evaluate_states(
        _manifest(al_package),
        data_dir=al_package["directory"],
    )
    al_by_temperature = {
        float(row["temperature_ev"]): row for row in al_rows
    }
    assert (
        float(al_by_temperature[1.0]["delta_zbar_tf_minus_ksdft"]) > 1.8
    )
    assert float(al_by_temperature[1.0]["gii_rmse_r_le_12"]) > 0.05
    assert float(al_by_temperature[100.0]["gii_rmse_r_le_12"]) < 0.003

    carbon_package = PACKAGES["carbon_lfc_sensitivity"]
    carbon_runner = _load_runner("carbon_lfc_metrics", carbon_package)
    carbon_rows, _ = carbon_runner.evaluate_states(
        _manifest(carbon_package),
        data_dir=carbon_package["directory"],
    )
    carbon_by_state = {
        (float(row["temperature_ev"]), row["model"]): row
        for row in carbon_rows
    }
    assert float(
        carbon_by_state[(2.0, "none")][
            "max_abs_dg_vs_chabrier_r_le_20"
        ]
    ) > 0.4
    assert float(
        carbon_by_state[(100.0, "none")][
            "max_abs_dg_vs_chabrier_r_le_20"
        ]
    ) < 0.005


def test_al_full_workflow_v2_augmentation_is_grid_strict(
    tmp_path: Path,
) -> None:
    path = (
        ROOT
        / "benchmarks"
        / "runners"
        / "regenerate_al_full_workflow.py"
    )
    spec = importlib.util.spec_from_file_location(
        "otter_al_full_workflow_producer_contract",
        path,
    )
    assert spec is not None and spec.loader is not None
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)

    payload = {
        "schema_version": np.asarray("old"),
        "benchmark_id": np.asarray("old"),
        "k_bohr_inv": np.asarray((0.1, 0.2, 0.3)),
    }
    portable = {
        "k_bohr_inv": np.asarray((0.1, 0.2, 0.3)),
        "n_ion_k": np.asarray(((2.8, 2.7, 2.5),)),
    }
    augmented = producer._augment_v2_payload(payload, portable)
    assert augmented["schema_version"].item() == (
        "otter_al_full_workflow_v2"
    )
    assert augmented["benchmark_id"].item() == "al_full_workflow_1ev"
    np.testing.assert_array_equal(
        augmented["n_ion_k_electrons"],
        portable["n_ion_k"][0],
    )

    bad = dict(portable)
    bad["k_bohr_inv"] = np.asarray((0.1, 0.2, 0.31))
    with pytest.raises(ValueError, match=r"f\(k\) grid"):
        producer._augment_v2_payload(payload, bad)

    candidate = tmp_path / "Al.npz"
    np.savez_compressed(candidate, **augmented)
    manifest = producer._candidate_manifest(
        candidate,
        worktree_status="",
    )
    assert manifest["status"] == "candidate_not_accepted"
    assert manifest["producer"]["worktree_clean_at_generation"] is True
    assert manifest["configuration"]["bound_rmax_mult"] is None
    assert manifest["configuration"]["bound_zero_tail_refine"] is False
    assert manifest["configuration"]["qoz_n_points_before_padding"] == 4096
    assert manifest["configuration"]["hnc_tolerance"] == 1.0e-6
    assert (
        manifest["configuration"]["hnc_transform_closure_tolerance"]
        == 1.0e-3
    )
    assert manifest["state"]["data_file"] == candidate.name
    assert manifest["state"]["data_sha256"] == _sha256(candidate)


def test_accepted_al_is_sc_example_archive_and_manifest() -> None:
    directory = (
        ROOT / "benchmarks" / "baselines" / "al_is_sc_comparison"
    )
    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "otter_example_manifest_v1"
    assert manifest["example_id"] == "al_is_sc_comparison"
    assert manifest["status"] == "accepted"
    assert manifest["producer"]["project"] == "Otter"
    assert manifest["producer"]["worktree_clean_at_generation"] is True
    assert len(manifest["producer"]["script_sha256"]) == 64
    assert manifest["data_rights"] == {
        "origin_type": "project_generated_numerical_output",
        "third_party_reference_data_included": False,
        "method_citations_only": True,
    }
    state_record = manifest["state"]
    path = (directory / state_record["data_file"]).resolve()
    assert path.parent == directory.resolve()
    assert _sha256(path) == state_record["data_sha256"]
    _assert_portable_archive(path)

    runner_path = (
        ROOT / "benchmarks" / "runners" / "plot_al_is_sc_comparison.py"
    )
    spec = importlib.util.spec_from_file_location(
        "otter_accepted_al_is_sc_runner",
        runner_path,
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    state = runner.load_state()
    assert state["schema_version"].item() == (
        "otter_al_is_sc_comparison_v1"
    )
    assert tuple(state["model_labels"]) == ("qm", "tf")
    assert tuple(state["structure_labels"]) == ("is", "sc")
    assert np.all(state["sc_converged"])
    assert np.max(state["hnc_residual"]) < 1.0e-4
    assert np.all(state["is_elapsed_s"] > 0.0)
    assert np.all(state["sc_extension_elapsed_s"] > 0.0)
    assert np.all(state["sc_total_elapsed_s"] > state["is_elapsed_s"])
    assert [row["level"] for row in runner.level_rows(state, "is")] == [
        "1s",
        "2s",
        "2p",
    ]
