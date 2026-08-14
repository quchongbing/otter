"""Provenance, unit, and offline-gallery gates for the Starrett collection."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ID = "starrett_single_species_2013_2014"
AMU_TO_G = 1.66053906660e-24
BOHR_TO_CM = 0.529177210903e-8
PHYSICAL_DEFINITIONS = {
    "fe_10": {
        "panel_id": "fe_10",
        "element": "Fe",
        "atomic_mass": 55.845,
        "rho_g_cc": 22.5,
        "te_ev": 10.0,
        "ti_ev": 10.0,
        "reference_file": "gii_Fe_22.5gcc_10.0ev_starrett.csv",
        "reference_unit": "r_over_R_WS_dimensionless",
    },
    "h_5": {
        "panel_id": "h_5",
        "element": "H",
        "atomic_mass": 1.008,
        "rho_g_cc": 80.0,
        "te_ev": 5.0,
        "ti_ev": 5.0,
        "reference_file": "gii_H_80gcc_5.0ev_starrett.csv",
        "reference_unit": "Bohr",
    },
    "h_172": {
        "panel_id": "h_172",
        "element": "H",
        "atomic_mass": 1.008,
        "rho_g_cc": 80.0,
        "te_ev": 172.0,
        "ti_ev": 172.0,
        "reference_file": "gii_H_80gcc_172.0ev_starrett.csv",
        "reference_unit": "Bohr",
    },
    "c_64p64": {
        "panel_id": "c_64p64",
        "element": "C",
        "atomic_mass": 12.011,
        "rho_g_cc": 12.64,
        "te_ev": 64.64,
        "ti_ev": 64.64,
        "reference_file": "gii_C_12.64gcc_64.64ev_starrett.csv",
        "reference_unit": "r_over_R_WS_dimensionless",
    },
    "w_10": {
        "panel_id": "w_10",
        "element": "W",
        "atomic_mass": 183.84,
        "rho_g_cc": 40.0,
        "te_ev": 10.0,
        "ti_ev": 10.0,
        "reference_file": "gii_W_40gcc_10.0ev_starrett_TF.csv",
        "reference_unit": "Bohr",
    },
    "w_60": {
        "panel_id": "w_60",
        "element": "W",
        "atomic_mass": 183.84,
        "rho_g_cc": 40.0,
        "te_ev": 60.0,
        "ti_ev": 60.0,
        "reference_file": "gii_W_40gcc_60.0ev_starrett.csv",
        "reference_unit": "Bohr",
    },
}
STATE_DEFINITIONS = {
    f"{panel_id}_{model}": {**definition, "model": model}
    for panel_id, definition in PHYSICAL_DEFINITIONS.items()
    for model in ("qm", "tf")
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ion_sphere_radius_bohr(
    rho_g_cc: float,
    atomic_mass: float,
) -> float:
    ion_density_cm3 = rho_g_cc / (atomic_mass * AMU_TO_G)
    radius_cm = (3.0 / (4.0 * np.pi * ion_density_cm3)) ** (1.0 / 3.0)
    return float(radius_cm / BOHR_TO_CM)


def test_reference_manifest_is_complete_attributed_and_checksummed() -> None:
    directory = ROOT / "benchmarks" / "reference_data" / BENCHMARK_ID
    manifest = _json(directory / "manifest.json")
    assert manifest["schema_version"] == "otter_reference_manifest_v1"
    assert manifest["redistribution_status"] == (
        "published_by_maintainer_with_attribution"
    )
    assert manifest["license_declared"] == "NOASSERTION"
    assert manifest["public_release_gate"] == "resolved"
    assert len(manifest["files"]) == 8
    for record in manifest["files"]:
        path = directory / record["path"]
        assert path.is_file()
        assert _sha256(path) == record["sha256"]
        curve = np.asarray(np.loadtxt(path, delimiter=","), dtype=float)
        assert curve.ndim == 2
        assert curve.shape[0] >= 2
        assert curve.shape[1] == 2
        assert np.all(np.isfinite(curve))
        assert np.unique(curve[:, 0]).size == curve.shape[0]
        bad_order = np.flatnonzero(np.diff(curve[:, 0]) <= 0.0)
        if record["path"] == "gii_W_40gcc_60.0ev_starrett.csv":
            assert bad_order.tolist() == [13]
            assert "stable-sorted" in record["coordinate_order"]
        else:
            assert bad_order.size == 0


def test_reference_source_and_coordinate_units_are_explicit() -> None:
    manifest = _json(
        ROOT
        / "benchmarks"
        / "reference_data"
        / BENCHMARK_ID
        / "manifest.json"
    )
    by_name = {record["path"]: record for record in manifest["files"]}
    assert by_name["gii_C_12.64gcc_64.64ev_starrett.csv"][
        "column_1_unit"
    ] == "r_over_R_WS_dimensionless"
    assert by_name["gii_Fe_22.5gcc_10.0ev_starrett.csv"][
        "column_1_unit"
    ] == "r_over_R_WS_dimensionless"
    assert by_name["gii_H_80gcc_5.0ev_starrett.csv"][
        "column_1_unit"
    ] == "Bohr"
    assert all(
        record["citation_key"] == "StarrettSaumon2014"
        for record in manifest["files"]
    )
    assert manifest["publication"]["doi"] == (
        "10.1016/j.hedp.2013.12.001"
    )
    assert manifest["publication"]["pii"] == "S1574181813001900"


def test_baseline_coverage_and_strict_promotion_are_explicit() -> None:
    manifest = _json(
        ROOT / "benchmarks" / "baselines" / BENCHMARK_ID / "manifest.json"
    )
    assert len(manifest["states"]) == 12
    records = {record["state_id"]: record for record in manifest["states"]}
    assert {
        state_id
        for state_id, record in records.items()
        if record["status"] == "accepted"
    } == set(STATE_DEFINITIONS)
    for record in records.values():
        if record["status"] != "accepted":
            assert record["baseline_file"] is None
            assert record["baseline_sha256"] is None
            continue
        path = (
            ROOT
            / "benchmarks"
            / "baselines"
            / BENCHMARK_ID
            / record["baseline_file"]
        )
        assert _sha256(path) == record["baseline_sha256"]
        definition = STATE_DEFINITIONS[record["state_id"]]
        with np.load(path, allow_pickle=False) as archive:
            assert not any(
                np.asarray(archive[key]).dtype.hasobject
                for key in archive.files
            )
            assert archive["schema_version"].item() == (
                "otter_starrett_single_species_state_v1"
            )
            assert archive["state_id"].item() == record["state_id"]
            assert archive["panel_id"].item() == definition["panel_id"]
            assert archive["element"].item() == definition["element"]
            assert archive["electronic_model"].item() == definition["model"]
            for key in ("rho_g_cc", "te_ev", "ti_ev"):
                assert np.isclose(
                    float(archive[key]),
                    float(definition[key]),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            expected_rws = _ion_sphere_radius_bohr(
                float(definition["rho_g_cc"]),
                float(definition["atomic_mass"]),
            )
            assert np.isclose(
                float(archive["r_ws_bohr"]),
                expected_rws,
                rtol=1.0e-8,
                atol=1.0e-10,
            )
            r = np.asarray(archive["r_bohr"], dtype=float)
            g = np.asarray(archive["gii_r"], dtype=float)
            k = np.asarray(archive["k_bohr_inv"], dtype=float)
            s = np.asarray(archive["sii_k"], dtype=float)
            assert r.ndim == g.ndim == k.ndim == s.ndim == 1
            assert r.shape == g.shape
            assert k.shape == s.shape
            assert np.all(np.isfinite(r))
            assert np.all(np.isfinite(g))
            assert np.all(np.isfinite(k))
            assert np.all(np.isfinite(s))
            assert r[0] > 0.0 and np.all(np.diff(r) > 0.0)
            assert k[0] > 0.0 and np.all(np.diff(k) > 0.0)
            assert r[-1] <= 20.0
            assert k[-1] <= 20.0
            assert float(archive["hnc_output_residual"]) <= 1.0e-4
            assert float(archive["closure_transform_max_abs"]) <= 2.5e-3
            assert archive["threshold_state_status"].item() != "unresolved"
            for key in (
                "zbar_partition",
                "hnc_output_residual",
                "closure_transform_max_abs",
            ):
                assert np.isclose(
                    float(archive[key]),
                    float(record[key]),
                    rtol=1.0e-12,
                    atol=1.0e-14,
                )
    controller = (
        ROOT
        / "benchmarks"
        / "examples"
        / "plot_starrett_single_species_2013_2014.py"
    )
    assert manifest["producer"]["current_controller_sha256"] == _sha256(
        controller
    )


def test_reference_coordinate_rmse_is_independently_reproduced() -> None:
    baseline_dir = ROOT / "benchmarks" / "baselines" / BENCHMARK_ID
    reference_dir = ROOT / "benchmarks" / "reference_data" / BENCHMARK_ID
    manifest = _json(baseline_dir / "manifest.json")
    records = {record["state_id"]: record for record in manifest["states"]}

    for state_id, definition in STATE_DEFINITIONS.items():
        record = records[state_id]
        if record["status"] != "accepted":
            continue
        rws = _ion_sphere_radius_bohr(
            float(definition["rho_g_cc"]),
            float(definition["atomic_mass"]),
        )
        if "reference_files" in definition:
            reference_files = tuple(definition["reference_files"])
        else:
            reference_files = (str(definition["reference_file"]),)

        with np.load(
            baseline_dir / str(record["baseline_file"]),
            allow_pickle=False,
        ) as archive:
            otter_r_over_rws = (
                np.asarray(archive["r_bohr"], dtype=float)
                / float(archive["r_ws_bohr"])
            )
            otter_gii = np.asarray(archive["gii_r"], dtype=float)
        reproduced_rmse: dict[str, float] = {}
        for reference_file in reference_files:
            reference = np.asarray(
                np.loadtxt(reference_dir / reference_file, delimiter=","),
                dtype=float,
            )
            if definition["reference_unit"] == "Bohr":
                reference_r_over_rws = reference[:, 0] / rws
            else:
                reference_r_over_rws = reference[:, 0]
            assert reference_r_over_rws[0] >= otter_r_over_rws[0]
            assert reference_r_over_rws[-1] <= otter_r_over_rws[-1]
            predicted = np.interp(
                reference_r_over_rws,
                otter_r_over_rws,
                otter_gii,
            )
            reproduced_rmse[reference_file] = float(
                np.sqrt(np.mean((predicted - reference[:, 1]) ** 2))
            )

        if len(reference_files) == 1:
            assert np.isclose(
                reproduced_rmse[reference_files[0]],
                float(record["reference_rmse"]),
                rtol=1.0e-8,
                atol=1.0e-12,
            )
        else:
            recorded = dict(record["reference_rmse_by_curve"])
            assert set(recorded) == set(reference_files)
            for reference_file, rmse in reproduced_rmse.items():
                assert np.isclose(
                    rmse,
                    float(recorded[reference_file]),
                    rtol=1.0e-8,
                    atol=1.0e-12,
                )


def test_gallery_is_standalone_and_saves_png_and_pdf() -> None:
    path = (
        ROOT
        / "benchmarks"
        / "examples"
        / "plot_starrett_single_species_2013_2014.py"
    )
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))
    assert "USE_PRECOMPUTED_DATA = True" in source
    assert "PlasmaWorkflowConfig" in source
    assert "solve_plasma_workflow(" in source
    assert "reference_records_by_file" in source
    assert "validated_two_column_curve" in source
    assert "validate_baseline_payload" in source
    assert '"label": "Otter QM"' in source
    assert '"label": "Otter TF"' in source
    assert 'ls="none"' in source
    assert "reference-only" not in source
    assert 'formats=(\"png\", \"pdf\")' in source
    assert "benchmarks/runners" not in source
