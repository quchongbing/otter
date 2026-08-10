"""Fast offline checks for the Starrett et al. mixtures Figure 3 package."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = (
    ROOT
    / "benchmarks"
    / "baselines"
    / "starrett_et_al_2014_mixtures_fig3"
)
MANIFEST_PATH = BASELINE_DIR / "manifest.json"
RUNNER_PATH = (
    ROOT
    / "benchmarks"
    / "runners"
    / "plot_starrett_et_al_2014_mixtures_fig3.py"
)
PRODUCER_PATH = (
    ROOT
    / "benchmarks"
    / "runners"
    / "regenerate_starrett_et_al_2014_mixtures_fig3.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "otter_starrett_mixtures_fig3_runner", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import benchmark runner at {RUNNER_PATH}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_producer():
    spec = importlib.util.spec_from_file_location(
        "otter_starrett_mixtures_fig3_producer",
        PRODUCER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import benchmark producer at {PRODUCER_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_has_complete_unique_state_grid_and_relative_paths() -> None:
    manifest = _manifest()
    assert manifest["schema_version"] == "otter_benchmark_manifest_v1"
    assert manifest["pairs"] == ["CC", "CH", "HH"]
    assert manifest["benchmark_id"] == (
        "starrett_et_al_2014_mixtures_fig3_ch1p36"
    )
    assert manifest["publication"]["doi"] == "10.1103/PhysRevE.90.033110"
    assert manifest["publication"]["authors"] == [
        "C. E. Starrett",
        "D. Saumon",
        "J. Daligault",
        "S. Hamel",
    ]
    assert manifest["reference_data"]["redistribution_status"] == (
        "published_by_maintainer_with_attribution"
    )
    assert manifest["reference_data"]["public_release_gate"] == "resolved"
    assert manifest["reference_data"]["release_decision"][
        "decision_date"
    ] == "2026-08-10"
    assert manifest["reference_data"]["origin_type"] == (
        "independent_digitization"
    )
    assert manifest["reference_data"]["source_figure"] == "3"
    assert manifest["reference_data"]["source_curves"] == "solid IS-QM"
    assert manifest["reference_data"]["license_declared"] == "NOASSERTION"
    assert manifest["reference_data"]["license_concluded"] == "NOASSERTION"
    states = manifest["states"]
    assert len(states) == 9
    assert {
        (float(state["rho_g_cc"]), int(state["temperature_kk"]))
        for state in states
    } == {
        (rho, temperature)
        for rho in (2.94, 5.0, 15.0)
        for temperature in (20, 50, 100)
    }
    serialized = MANIFEST_PATH.read_text(encoding="utf-8")
    assert "/home/" not in serialized
    assert "/tmp/" not in serialized
    for state in states:
        assert not Path(state["reference_file"]).is_absolute()
        assert not Path(state["baseline_file"]).is_absolute()


def test_reference_baseline_and_metrics_checksums() -> None:
    manifest = _manifest()
    for state in manifest["states"]:
        reference = (BASELINE_DIR / state["reference_file"]).resolve()
        baseline = (BASELINE_DIR / state["baseline_file"]).resolve()
        assert reference.is_file()
        assert baseline.is_file()
        assert _sha256(reference) == state["reference_sha256"]
        assert _sha256(baseline) == state["baseline_sha256"]
    metrics = BASELINE_DIR / manifest["baseline"]["metrics_file"]
    assert _sha256(metrics) == manifest["baseline"]["metrics_sha256"]


def test_baselines_are_pickle_free_finite_and_converged() -> None:
    manifest = _manifest()
    for state in manifest["states"]:
        path = BASELINE_DIR / state["baseline_file"]
        with np.load(path, allow_pickle=False) as archive:
            assert all(not archive[key].dtype.hasobject for key in archive.files)
            assert archive["schema_version"].item() == (
                "otter_starrett_mixtures_fig3_baseline_v1"
            )
            assert archive["benchmark_id"].item() == (
                "starrett_et_al_2014_mixtures_fig3_ch1p36"
            )
            assert tuple(archive["species_symbols"]) == ("C", "H")
            assert tuple(archive["pair_labels"]) == ("CC", "CH", "HH")
            r = np.asarray(archive["r_bohr"], dtype=float)
            g_ab = np.asarray(archive["g_ab"], dtype=float)
            assert g_ab.shape == (3, r.size)
            assert np.all(np.isfinite(r))
            assert np.all(np.isfinite(g_ab))
            assert np.all(np.diff(r) > 0.0)
            assert r[0] >= 0.0
            assert r[-1] <= 6.0
            assert bool(archive["root_success"].item())
            assert float(archive["root_residual_ha"]) < 1.0e-4
            assert float(archive["hnc_output_residual"]) < 1.0e-5
            assert float(archive["hnc_closure_mismatch"]) < 1.0e-5
            np.testing.assert_allclose(
                archive["q_scr_used"],
                archive["zbar_qoz"],
                rtol=0.0,
                atol=2.0e-12,
            )
            for key in archive.files:
                value = archive[key]
                if value.dtype.kind in "SU":
                    text = " ".join(str(item) for item in value.reshape(-1))
                    assert "/home/" not in text
                    assert "/tmp/" not in text
            signature = json.loads(
                str(archive["producer_signature_json"].item())
            )
            assert signature["aa_overrides"]["bound_occ_mode"] == "fd"
            assert signature["qoz"]["lfc_model"] == "chabrier1990"
            assert np.isclose(
                float(signature["state"]["rho_g_cc"]),
                float(archive["rho_g_cc"]),
            )
            assert int(signature["state"]["temperature_kk"]) == int(
                archive["temperature_kk"]
            )


def test_runner_recomputes_recorded_metrics_without_a_solver(
    monkeypatch, tmp_path
) -> None:
    # Loading and evaluating the package must not depend on the caller's CWD.
    monkeypatch.chdir(tmp_path)
    runner = _load_runner()
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "solve_plasma_workflow" not in runner_source
    assert "from otter" not in runner_source
    manifest = runner.load_manifest()
    actual_rows, _ = runner.evaluate_states(manifest)
    actual = {
        (
            float(row["rho_g_cc"]),
            int(row["temperature_kk"]),
            str(row["pair"]),
        ): row
        for row in actual_rows
    }
    with (BASELINE_DIR / "metrics.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        expected_rows = list(csv.DictReader(stream))
    assert len(actual) == len(expected_rows) == 27
    numeric_fields = (
        "rmse",
        "mae",
        "max_abs",
        "bias",
        "r_peak_ref_bohr",
        "g_peak_ref",
        "r_peak_model_bohr",
        "g_peak_model",
    )
    for expected in expected_rows:
        key = (
            float(expected["rho_g_cc"]),
            int(expected["temperature_kk"]),
            expected["pair"],
        )
        row = actual[key]
        assert int(row["n_reference"]) == int(expected["n_reference"])
        for field in numeric_fields:
            assert np.isclose(
                float(row[field]),
                float(expected[field]),
                rtol=0.0,
                atol=5.0e-13,
            )
    assert max(float(row["rmse"]) for row in actual_rows) < 0.025


def test_otter_recompute_configuration_is_strict_and_candidate_only() -> None:
    producer = _load_producer()
    assert len(producer.STATES) == 9
    assert producer.OUTPUT_DIR.parts[-2:] == (
        "starrett_et_al_2014_mixtures_fig3",
        "recomputed",
    )
    assert producer.MAX_STATE_WORKERS * producer.CONTINUUM_WORKERS_PER_STATE <= 24
    state = producer.State(2.94, 20)
    cfg = producer.configuration(state)
    assert cfg.elements == ["C", "H"]
    assert cfg.counts == [1.0, 1.36]
    assert cfg.electronic_model == "qm"
    assert cfg.ion_temperature_ev == cfg.temperature_ev
    assert cfg.allow_unconverged_root is False
    assert cfg.allow_unconverged_aa is False
    assert cfg.mu_e_tol == producer.MU_E_TOL_HA
    assert cfg.hnc_require_converged is True
    assert cfg.hnc_s_projection_mode == "none"
    assert cfg.qoz_response_chi0_model == "lindhard_fd"
    assert cfg.qoz_response_lfc_model == "chabrier1990"
    assert cfg.aa_overrides["bound_occ_mode"] == "fd"
    assert cfg.aa_overrides["bound_rmax_mult"] is None
    assert cfg.aa_overrides["bound_zero_tail_refine"] is False
    assert cfg.aa_overrides["b3_tail_target"] == "full"
    assert cfg.aa_overrides["b3_tail_model"] == "full"


def test_read_only_runner_accepts_synthetic_otter_candidate(tmp_path) -> None:
    runner = _load_runner()
    r = np.linspace(0.01, 6.0, 120)
    gij = np.ones((2, 2, r.size))
    gij[0, 0] = 1.0 - np.exp(-(r / 2.0) ** 6)
    gij[0, 1] = gij[1, 0] = 1.0 - np.exp(-(r / 1.5) ** 6)
    gij[1, 1] = 1.0 - np.exp(-(r / 1.0) ** 6)
    result_path = tmp_path / "synthetic_otter.npz"
    np.savez_compressed(
        result_path,
        schema_version=np.asarray("otter_state_v1"),
        species_symbols=np.asarray(("C", "H")),
        r_bohr=r,
        gij_r=gij,
    )
    reference = (
        ROOT
        / "benchmarks"
        / "reference_data"
        / "starrett_et_al_2014_mixtures_fig3"
        / "fig3_gab_rho2p94gcc_T100kK.csv"
    )
    manifest = {
        "schema_version": "otter_starrett_fig3_recompute_manifest_v1",
        "benchmark_id": "starrett_et_al_2014_mixtures_fig3_ch1p36",
        "states": [
            {
                "rho_g_cc": 2.94,
                "temperature_kk": 100,
                "temperature_ev": 8.617333262145,
                "result_file": result_path.name,
                "result_sha256": _sha256(result_path),
                "reference_file": os.path.relpath(reference, tmp_path),
                "reference_sha256": _sha256(reference),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded_manifest = runner.load_manifest(
        manifest_path=manifest_path,
        expected_state_count=1,
    )
    metrics, loaded = runner.evaluate_states(
        loaded_manifest,
        data_dir=tmp_path,
    )
    assert len(metrics) == 3
    candidate = loaded[(2.94, 100)]["baseline"]
    assert tuple(candidate["pair_labels"]) == ("CC", "CH", "HH")
    assert candidate["g_ab"].shape == (3, r.size)
