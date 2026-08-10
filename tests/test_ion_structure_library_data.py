"""Offline integrity and physics checks for the ion-structure gallery data."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
LIBRARY_DIR = (
    ROOT / "benchmarks" / "baselines" / "ion_structure_library"
)
AL_DIR = ROOT / "benchmarks" / "baselines" / "al_full_workflow_1ev"
REFERENCE_DIR = (
    ROOT / "benchmarks" / "reference_data" / "ion_structure_library"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_library_manifest_hashes_and_portable_archives() -> None:
    manifest = _json(LIBRARY_DIR / "manifest.json")
    assert manifest["schema_version"] == "otter_benchmark_manifest_v1"
    assert manifest["benchmark_id"] == "ion_structure_library"
    assert manifest["producer"]["project"] == "Otter"
    assert len(manifest["producer"]["git_commit"]) == 40
    assert len(manifest["states"]) == 5
    assert manifest["data_rights"]["reference_redistribution_status"] == (
        "published_by_maintainer_with_attribution"
    )
    assert manifest["data_rights"]["public_release_gate"] == "resolved"
    serialized = json.dumps(manifest)
    assert "/home/" not in serialized
    assert "/tmp/" not in serialized

    for item in manifest["states"]:
        relative = Path(item["baseline_file"])
        assert not relative.is_absolute()
        path = (LIBRARY_DIR / relative).resolve()
        assert path.parent == LIBRARY_DIR.resolve()
        assert _sha256(path) == item["baseline_sha256"]
        _assert_portable_archive(path)
        with np.load(path, allow_pickle=False) as archive:
            assert archive["schema_version"].item() == (
                "otter_ion_structure_library_state_v1"
            )
            assert archive["state_id"].item() == item["state_id"]
            assert archive["otter_git_commit"].item() == (
                manifest["producer"]["git_commit"]
            )
            signature = json.loads(
                str(archive["producer_signature_json"].item())
            )
            assert signature["aa"]["bound_occ_mode"] == "fd"
            assert signature["aa"]["b3_tail_model"] == "full"
            assert signature["qoz"]["chi0_model"] == "lindhard_fd"
            assert signature["qoz"]["lfc_model"] == "chabrier1990"


def test_library_grid_charge_and_convergence_invariants() -> None:
    manifest = _json(LIBRARY_DIR / "manifest.json")
    for item in manifest["states"]:
        path = LIBRARY_DIR / item["baseline_file"]
        with np.load(path, allow_pickle=False) as archive:
            r_e = np.asarray(archive["r_e_bohr"], dtype=float)
            r = np.asarray(archive["r_bohr"], dtype=float)
            k = np.asarray(archive["k_bohr_inv"], dtype=float)
            assert np.all(np.diff(r_e) > 0.0)
            assert np.all(np.diff(r) > 0.0)
            assert np.all(np.diff(k) > 0.0)
            assert r_e[-1] <= 20.0
            assert r[-1] <= 20.0
            assert k[-1] <= 20.0
            for key in (
                "n_full_bohr3",
                "n_free_bohr3",
                "n_bound_bohr3",
                "n_ext_bohr3",
                "n_ion_bohr3",
                "n_pa_bohr3",
                "n_scr_bohr3",
                "v_full_ha",
                "v_ext_ha",
                "v_hartree_ha",
                "v_xc_ha",
            ):
                assert archive[key].shape == r_e.shape
            assert archive["gii_r"].shape == r.shape
            assert archive["vii_r_ha"].shape == r.shape
            for key in (
                "sii_k",
                "vii_k_ha_bohr3",
                "n_scr_k_electrons",
                "chi0_k_bohr3_per_ha",
                "gee_k",
            ):
                assert archive[key].shape == k.shape
            assert float(archive["hnc_best_residual"]) <= 1.0e-4
            assert float(archive["hnc_closure_mismatch"]) <= float(
                archive["hnc_closure_tolerance"]
            )
            np.testing.assert_allclose(
                float(archive["zbar_qoz"]),
                float(archive["zbar_partition"]),
                rtol=0.0,
                atol=1.0e-10,
            )
            np.testing.assert_allclose(
                float(archive["q_scr_raw"]),
                float(archive["zbar_partition"]),
                rtol=0.0,
                atol=7.0e-3,
            )


def test_reference_manifest_hashes_and_release_decision() -> None:
    manifest = _json(REFERENCE_DIR / "manifest.json")
    assert manifest["schema_version"] == "otter_reference_manifest_v1"
    assert manifest["redistribution_status"] == (
        "published_by_maintainer_with_attribution"
    )
    assert manifest["license_declared"] == "NOASSERTION"
    assert manifest["public_release_gate"] == "resolved"
    assert _sha256(REFERENCE_DIR / manifest["inventory_file"]) == (
        manifest["inventory_sha256"]
    )
    for item in manifest["files"]:
        relative = Path(item["path"])
        assert not relative.is_absolute()
        path = (REFERENCE_DIR / relative).resolve()
        assert path.is_relative_to(REFERENCE_DIR.resolve())
        assert _sha256(path) == item["sha256"]
        values = np.genfromtxt(path, delimiter=",", comments="#")
        assert values.ndim == 2
        assert values.shape[1] >= 2
        assert np.all(np.isfinite(values[:, :2]))


def test_offline_library_runner_recomputes_metrics() -> None:
    runner_path = (
        ROOT / "benchmarks" / "runners" / "plot_ion_structure_library.py"
    )
    source = runner_path.read_text(encoding="utf-8")
    assert "solve_plasma_workflow" not in source
    assert "from otter" not in source
    runner = _load_module("otter_library_offline_test", runner_path)
    states = runner.load_states(runner.load_manifest())
    rows = runner.evaluate(states)
    assert len(rows) == 15
    primary_rows = [row for row in rows if row["role"] == "primary"]
    assert len(primary_rows) == 6
    for row in rows:
        assert row["n_points"] > 5
        assert np.isfinite(row["rmse"])
        assert np.isfinite(row["mae"])
        assert np.isfinite(row["max_abs"])
    carbon = next(
        row
        for row in rows
        if row["state_id"] == "c_starrett_rho20_te50_ti50"
    )
    assert carbon["rmse"] < 0.01


def test_reference_coordinate_conversions_match_source_plot_scripts() -> None:
    runner = _load_module(
        "otter_library_units_test",
        ROOT / "benchmarks" / "runners" / "plot_ion_structure_library.py",
    )
    state = {
        "k_bohr_inv": np.asarray([1.0]),
        "sii_k": np.asarray([0.25]),
        "r_bohr": np.asarray([1.0]),
        "gii_r": np.asarray([0.75]),
    }
    k_angstrom, _ = runner._otter_curve(
        state,
        "sii",
        "angstrom^-1",
    )
    r_angstrom, _ = runner._otter_curve(state, "gii", "angstrom")
    r_bohr, _ = runner._otter_curve(state, "gii", "bohr")
    np.testing.assert_allclose(
        k_angstrom,
        [1.0 / runner.BOHR_TO_ANGSTROM],
    )
    np.testing.assert_allclose(
        r_angstrom,
        [runner.BOHR_TO_ANGSTROM],
    )
    np.testing.assert_allclose(r_bohr, [1.0])
    assert {
        series["x_unit"]
        for series in runner.REFERENCE_SERIES[
            "be_wunsch_rho5p544_te13_ti13"
        ]
    } == {"angstrom^-1", "angstrom"}
    assert {
        series["x_unit"]
        for series in runner.REFERENCE_SERIES[
            "c_starrett_rho20_te50_ti50"
        ]
    } == {"bohr"}


def test_complete_al_workflow_manifest_levels_and_pipeline() -> None:
    manifest = _json(AL_DIR / "manifest.json")
    assert manifest["schema_version"] == "otter_benchmark_manifest_v1"
    assert manifest["benchmark_id"] == "al_full_workflow_1ev"
    assert str(manifest["status"]).startswith("accepted")
    assert manifest["producer"]["project"] == "Otter"
    assert manifest["producer"]["worktree_clean_at_generation"] is False
    assert len(manifest["producer"]["script_sha256_at_generation"]) == 64
    assert manifest["producer"]["script_sha256_current"] == _sha256(
        ROOT / manifest["producer"]["script_relative_path"]
    )
    assert manifest["producer"]["script_relative_path"] == (
        "docs/examples/plot_al_full_workflow.py"
    )
    assert len(manifest["producer"]["git_status_porcelain_sha256"]) == 64
    assert manifest["configuration"]["bound_occ_mode"] == "fd"
    assert manifest["configuration"]["bound_rmax_mult"] is None
    assert manifest["configuration"]["bound_zero_tail_refine"] is True
    assert manifest["configuration"]["b3_tail_model"] == "full"
    assert manifest["configuration"]["qoz_zbar_mode"] == (
        "pseudoatom_partition"
    )
    assert manifest["configuration"]["qoz_renormalize_nscr_to_zbar"] is True
    audit = manifest["scientific_audit"]
    assert audit["q_scr_used"] == pytest.approx(audit["zbar_qoz"])
    assert audit["hnc_best_residual"] <= manifest["configuration"]["hnc_tolerance"]
    assert audit["hnc_closure_mismatch"] <= (
        manifest["configuration"]["hnc_transform_closure_tolerance"]
    )
    item = manifest["state"]
    path = AL_DIR / item["data_file"]
    assert _sha256(path) == item["data_sha256"]
    _assert_portable_archive(path)
    runner = _load_module(
        "otter_al_workflow_offline_test",
        ROOT / "benchmarks" / "runners" / "plot_al_full_workflow.py",
    )
    state = runner.load_state()
    rows = runner.bound_level_rows(state)
    assert [row["level"] for row in rows] == ["1s", "2s", "2p"]
    np.testing.assert_allclose(
        [row["occupation"] for row in rows],
        [2.0, 2.0, 6.0],
        rtol=0.0,
        atol=1.0e-8,
    )
    assert float(state["hnc_best_residual"]) <= 1.0e-4
    assert float(state["hnc_closure_mismatch"]) <= float(
        state["hnc_closure_tolerance"]
    )
    assert np.max(state["r_e_bohr"]) <= 20.0
    assert np.max(state["r_bohr"]) <= 20.0
    assert np.max(state["k_bohr_inv"]) <= 20.0
    assert state["schema_version"].item() == "otter_al_full_workflow_v2"
    assert state["n_ion_k_electrons"].shape == state["k_bohr_inv"].shape
    assert np.all(np.isfinite(state["n_ion_k_electrons"]))
    assert state["n_scr_k_raw_electrons"].shape == state["k_bohr_inv"].shape
    assert np.all(np.isfinite(state["n_scr_k_raw_electrons"]))
    assert float(state["q_scr_used"]) == pytest.approx(
        float(state["zbar_qoz"])
    )
    assert np.isclose(
        float(state["n_ion_k_electrons"][0]),
        float(np.trapezoid(
            4.0
            * np.pi
            * state["r_e_bohr"] ** 2
            * state["n_ion_bohr3"],
            state["r_e_bohr"],
        )),
        rtol=1.0e-2,
        atol=1.0e-3,
    )
