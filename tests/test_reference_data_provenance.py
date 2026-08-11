"""Integrity, attribution, and release policy for comparison datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "benchmark_id",
    (
        "johnson_et_al_2025_two_temperature_al",
        "argha_roy_carbon_sii",
    ),
)
def test_reference_files_are_checksummed_and_release_decision_is_recorded(
    benchmark_id: str,
) -> None:
    directory = ROOT / "benchmarks" / "reference_data" / benchmark_id
    manifest = _json(directory / "manifest.json")

    assert manifest["schema_version"] == "otter_reference_manifest_v1"
    assert manifest["reference_id"] == benchmark_id
    assert manifest["redistribution_status"] == (
        "published_by_maintainer_with_attribution"
    )
    assert manifest["license_declared"] == "NOASSERTION"
    assert manifest["public_release_gate"] == "resolved"
    assert manifest["release_decision"]["decision_date"] == "2026-08-10"
    assert "does not assert" in manifest["release_decision"]["rights_note"]
    assert manifest["files"]

    for record in manifest["files"]:
        path = directory / str(record["path"])
        assert path.is_file()
        assert _sha256(path) == record["sha256"]


def test_johnson_units_and_accepted_baselines_are_explicit() -> None:
    benchmark_id = "johnson_et_al_2025_two_temperature_al"
    reference = _json(
        ROOT / "benchmarks" / "reference_data" / benchmark_id / "manifest.json"
    )
    assert reference["publication"]["doi"] == "10.1103/5c29-kdx1"
    assert reference["units"]["column_1"] == "Bohr"
    assert "r [au]" in reference["unit_audit"]["source"]
    assert len(reference["files"]) == 9

    directory = ROOT / "benchmarks" / "baselines" / benchmark_id
    manifest = _json(directory / "manifest.json")
    controller = (
        ROOT
        / "benchmarks"
        / "examples"
        / "plot_johnson_et_al_2025_two_temperature_al.py"
    )
    assert manifest["producer"]["current_controller_sha256"] == _sha256(
        controller
    )
    statuses = {record["state_id"]: record["status"] for record in manifest["states"]}
    assert statuses == {
        "al_rho2p7_te1_ti1": "accepted",
        "al_rho2p7_te10_ti1": "accepted",
        "al_rho2p7_te30_ti1": "accepted",
    }
    controller = (
        ROOT
        / str(manifest["producer"]["script_relative_path"])
    )
    assert _sha256(controller) == manifest["producer"]["current_controller_sha256"]
    _check_accepted_archives(directory, manifest)


def test_argha_attribution_uncertainty_and_current_otter_states_are_explicit() -> None:
    benchmark_id = "argha_roy_carbon_sii"
    reference = _json(
        ROOT / "benchmarks" / "reference_data" / benchmark_id / "manifest.json"
    )
    assert reference["origin_type"] == "author_provided_private_numerical_data"
    assert "Dr. Argha Roy" in reference["attribution"]
    assert "DFT-MD" in reference["attribution"]
    assert reference["method_label"] == "DFT-MD"
    assert reference["columns"][2] == "reported_uncertainty_dimensionless"
    assert len(reference["files"]) == 6

    directory = ROOT / "benchmarks" / "baselines" / benchmark_id
    manifest = _json(directory / "manifest.json")
    controller = (
        ROOT
        / "benchmarks"
        / "examples"
        / "plot_argha_roy_carbon_sii.py"
    )
    assert manifest["producer"]["current_controller_sha256"] == _sha256(
        controller
    )
    assert [record["te_ev"] for record in manifest["states"]] == [
        20.0,
        30.0,
        40.0,
        50.0,
        100.0,
    ]
    assert all(record["status"] == "accepted" for record in manifest["states"])
    assert all(
        record["threshold_state_status"] == "resolved"
        for record in manifest["states"]
    )
    assert manifest["configuration"]["bound_zero_tail_refine"] is True
    assert manifest["configuration"]["bound_rmax_mult"] is None
    controller = (
        ROOT
        / str(manifest["producer"]["script_relative_path"])
    )
    assert _sha256(controller) == manifest["producer"]["current_controller_sha256"]
    _check_accepted_archives(directory, manifest)


def _check_accepted_archives(directory: Path, manifest: dict) -> None:
    """Check only accepted files; reference-only records must not fake arrays."""
    for record in manifest["states"]:
        if record["status"] != "accepted":
            assert record.get("baseline_file") is None
            assert record.get("baseline_sha256") is None
            continue
        path = directory / str(record["baseline_file"])
        assert _sha256(path) == record["baseline_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            assert not any(
                np.asarray(archive[key]).dtype.hasobject for key in archive.files
            )
            assert str(archive["state_id"].item()) == record["state_id"]
            residual_key = (
                "hnc_output_residual"
                if "hnc_output_residual" in archive
                else "hnc_best_residual"
            )
            closure_key = (
                "closure_transform_max_abs"
                if "closure_transform_max_abs" in archive
                else "hnc_closure_mismatch"
            )
            assert float(archive[residual_key]) <= 1.0e-4
            assert float(archive[closure_key]) <= 2.5e-3
