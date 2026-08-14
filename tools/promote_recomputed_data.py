"""Promote one complete set of recomputed Otter data into the baselines.

The expensive producers write below ``benchmarks/outputs``.  This maintainer
tool validates the complete candidate set, atomically replaces only the
project-generated baseline NPZ files, refreshes their checksums and numerical
diagnostics, and removes the candidate directories.  Literature reference
data are never modified.

Run ``tools/recompute_all_data.py --fresh`` first.  Promotion is explicit::

    python tools/promote_recomputed_data.py --apply
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from otter import __version__ as otter_version
from otter.io._npz import save_npz_atomic


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "benchmarks" / "baselines"
OUTPUTS = ROOT / "benchmarks" / "outputs"


@dataclass(frozen=True)
class Package:
    name: str
    candidate_dir: Path
    manifest_name: str | None = "manifest.json"
    starrett_fig3_names: bool = False

    @property
    def baseline_dir(self) -> Path:
        return BASELINES / self.name

    @property
    def candidate_manifest(self) -> Path | None:
        if self.manifest_name is None:
            return None
        return self.candidate_dir / self.manifest_name


PACKAGES = (
    Package(
        "al_full_workflow_1ev",
        OUTPUTS / "al_full_workflow_1ev" / "recomputed",
    ),
    Package(
        "al_is_sc_comparison",
        OUTPUTS / "al_is_sc_comparison" / "recomputed",
    ),
    Package("al_qm_tf", OUTPUTS / "al_qm_tf" / "recomputed"),
    Package(
        "al_rayleigh_weight_10ev",
        OUTPUTS / "al_rayleigh_weight_10ev" / "recomputed",
        manifest_name=None,
    ),
    Package(
        "argha_roy_carbon_sii",
        OUTPUTS / "argha_roy_carbon_sii" / "gallery_recomputed",
        manifest_name=None,
    ),
    Package(
        "carbon_ionization_levels",
        OUTPUTS / "carbon_ionization_levels",
        manifest_name="C_Te100eV_density_scan.manifest.json",
    ),
    Package(
        "carbon_lfc_sensitivity",
        OUTPUTS / "carbon_lfc_sensitivity" / "recomputed",
    ),
    Package(
        "ch136_mixture_workflow_100kk",
        OUTPUTS / "ch136_mixture_workflow_100kk" / "recomputed",
        manifest_name=None,
    ),
    Package(
        "ion_structure_library",
        OUTPUTS / "ion_structure_library" / "recomputed",
    ),
    Package(
        "johnson_et_al_2025_two_temperature_al",
        OUTPUTS
        / "johnson_et_al_2025_two_temperature_al"
        / "gallery_recomputed",
        manifest_name="candidate_manifest.json",
    ),
    Package(
        "starrett_et_al_2014_mixtures_fig3",
        OUTPUTS
        / "starrett_et_al_2014_mixtures_fig3"
        / "gallery_recomputed",
        manifest_name=None,
        starrett_fig3_names=True,
    ),
    Package(
        "starrett_single_species_2013_2014",
        OUTPUTS
        / "starrett_single_species_2013_2014"
        / "gallery_recomputed",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_name(package: Package, baseline_name: str) -> str:
    if package.starrett_fig3_names:
        return baseline_name.replace("_baseline.npz", "_otter.npz")
    return baseline_name


def _npz_scalar(arrays: dict[str, np.ndarray], key: str) -> Any:
    if key not in arrays:
        return None
    value = np.asarray(arrays[key])
    if value.size != 1:
        return None
    item = value.reshape(()).item()
    if isinstance(item, np.generic):
        item = item.item()
    return item


def _load_candidate(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if not arrays or "schema_version" not in arrays:
        raise ValueError(f"{path}: missing schema_version.")
    optional_nan_diagnostics = {
        "bound_zero_tail_finite_wall_energy_ha",
        "bound_zero_tail_matched_energy_ha",
        "bound_zero_tail_exterior_probability",
    }
    for key, value in arrays.items():
        if value.dtype.hasobject:
            raise ValueError(f"{path}: object array {key!r} is not portable.")
        if value.dtype.kind in "fiu" and not np.all(np.isfinite(value)):
            if key not in optional_nan_diagnostics or np.any(np.isinf(value)):
                raise ValueError(f"{path}: non-finite numeric values in {key!r}.")
        if value.dtype.kind in "SU":
            text = " ".join(str(item) for item in value.reshape(-1))
            if "/home/" in text or "/tmp/" in text:
                raise ValueError(f"{path}: machine-local path in {key!r}.")

    for key in (
        "aa_stage2_converged",
        "aa_ext_converged",
        "stage2_converged",
        "hnc_converged",
    ):
        if key in arrays and not np.all(np.asarray(arrays[key], dtype=bool)):
            raise ValueError(f"{path}: {key} is not true for every state.")

    for key, limit in (
        ("root_residual_ha", 1.0e-4),
        ("hnc_output_residual", 1.0e-4),
        ("closure_transform_max_abs", 2.5e-3),
    ):
        if key in arrays and float(np.max(np.asarray(arrays[key], dtype=float))) > limit:
            raise ValueError(f"{path}: {key} exceeds {limit:g}.")
    return arrays


def _records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(manifest.get("states"), list):
        return [dict(item) for item in manifest["states"]]
    if isinstance(manifest.get("state"), dict):
        return [dict(manifest["state"])]
    return []


def _record_filename(record: dict[str, Any]) -> str | None:
    for key in (
        "data_file",
        "baseline_file",
        "candidate_file",
        "result_file",
    ):
        value = record.get(key)
        if value:
            return Path(str(value)).name
    return None


def _candidate_record(
    candidate_manifest: dict[str, Any] | None,
    *,
    candidate_name: str,
    baseline_record: dict[str, Any],
) -> dict[str, Any] | None:
    if candidate_manifest is None:
        return None
    records = _records(candidate_manifest)
    state_id = baseline_record.get("state_id")
    for record in records:
        if _record_filename(record) == candidate_name:
            return record
        if state_id is not None and record.get("state_id") == state_id:
            return record
    if len(records) == 1:
        return records[0]
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _manifest_record(
    manifest: dict[str, Any], baseline_name: str
) -> dict[str, Any]:
    for record in _records(manifest):
        if _record_filename(record) == baseline_name:
            return deepcopy(record)
    return {}


def _citation_keys(manifest: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for field in ("method_references", "method_publications"):
        entries = manifest.get(field, [])
        if isinstance(entries, dict):
            entries = list(entries.values())
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            key = entry.get("citation_key") or entry.get("reference_key")
            if key and str(key) not in keys:
                keys.append(str(key))
    return keys


def _method_references(manifest: dict[str, Any]) -> Any:
    for key in (
        "method_references",
        "method_publications",
        "method_publication",
        "publication",
        "literature_source",
    ):
        if key in manifest:
            return deepcopy(manifest[key])
    return []


def _compact_metadata(
    package: Package,
    *,
    manifest: dict[str, Any],
    baseline_name: str,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    scalar_diagnostics: dict[str, Any] = {}
    for key in (
        "aa_stage2_converged",
        "aa_ext_converged",
        "stage2_converged",
        "root_residual_ha",
        "root_residual_final_ha",
        "hnc_converged",
        "hnc_output_residual",
        "hnc_closure_mismatch",
        "closure_transform_max_abs",
        "threshold_state_status",
        "threshold_state_representation",
    ):
        value = _npz_scalar(arrays, key)
        if value is not None:
            scalar_diagnostics[key] = _json_safe(value)

    state = _manifest_record(manifest, baseline_name)
    for key in (
        "baseline_sha256",
        "data_sha256",
        "result_sha256",
        "candidate_sha256",
    ):
        state.pop(key, None)
    producer = deepcopy(manifest.get("producer", {}))
    if not isinstance(producer, dict):
        producer = {}
    producer.setdefault("project", "Otter")
    producer.setdefault("version", otter_version)
    return {
        "schema_version": "otter_compact_archive_metadata_v1",
        "archive_schema_version": str(_npz_scalar(arrays, "schema_version")),
        "archive_role": "project_generated_example_or_benchmark_baseline",
        "package_id": manifest.get(
            "benchmark_id", manifest.get("example_id", package.name)
        ),
        "file": baseline_name,
        "configuration": manifest.get("configuration", {}),
        "state": state,
        "producer": producer,
        "citation_keys": _citation_keys(manifest),
        "method_references": _method_references(manifest),
        "units": manifest.get("units", {}),
        "data_rights": manifest.get("data_rights", {}),
        "convergence": scalar_diagnostics,
        "fields": sorted(arrays),
    }


def _with_compact_metadata(
    package: Package,
    *,
    manifest: dict[str, Any],
    baseline_name: str,
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    payload = dict(arrays)
    metadata = _compact_metadata(
        package,
        manifest=manifest,
        baseline_name=baseline_name,
        arrays=arrays,
    )
    payload["metadata_json"] = np.asarray(
        json.dumps(_json_safe(metadata), sort_keys=True, separators=(",", ":"))
    )
    return payload


def _update_record(
    baseline_record: dict[str, Any],
    candidate_record: dict[str, Any] | None,
    arrays: dict[str, np.ndarray],
    *,
    baseline_name: str,
    digest: str,
) -> dict[str, Any]:
    record = deepcopy(baseline_record)
    if candidate_record is not None:
        for key, value in candidate_record.items():
            if key not in {
                "status",
                "data_file",
                "data_sha256",
                "baseline_file",
                "baseline_sha256",
                "candidate_file",
                "candidate_sha256",
                "result_file",
                "result_sha256",
            }:
                record[key] = deepcopy(value)

    if "data_file" in record:
        record["data_file"] = baseline_name
        record["data_sha256"] = digest
    elif "result_file" in record:
        record["result_file"] = baseline_name
        record["result_sha256"] = digest
    else:
        record["baseline_file"] = baseline_name
        record["baseline_sha256"] = digest
    if "status" in record:
        record["status"] = "accepted"

    for key in (
        "zbar_partition",
        "root_residual_ha",
        "hnc_output_residual",
        "hnc_closure_mismatch",
        "closure_transform_max_abs",
        "threshold_state_status",
        "threshold_state_representation",
    ):
        value = _npz_scalar(arrays, key)
        if value is not None:
            if isinstance(value, (np.bool_, bool)):
                value = bool(value)
            elif isinstance(value, (np.integer, int)):
                value = int(value)
            elif isinstance(value, (np.floating, float)):
                value = float(value)
            else:
                value = str(value)
            record[key] = value
    if "r_bohr" in arrays:
        record["n_model_points"] = int(np.asarray(arrays["r_bohr"]).size)
    return record


def _candidate_manifest(package: Package) -> dict[str, Any] | None:
    path = package.candidate_manifest
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_producer_metadata(manifest: dict[str, Any]) -> None:
    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        return
    relative = producer.get("script_relative_path")
    if not relative:
        return
    script = ROOT / str(relative)
    if not script.is_file():
        raise FileNotFoundError(f"Producer script is missing: {relative}")
    digest = sha256_file(script)
    producer["script_sha256_current"] = digest
    producer.setdefault(
        "script_sha256_at_generation",
        producer.get("script_sha256", digest),
    )
    if "script_sha256" in producer:
        producer["script_sha256"] = digest
    if (
        "current_controller_sha256" in producer
        or str(relative).startswith("benchmarks/examples/")
    ):
        producer["current_controller_sha256"] = digest


def _merge_manifest(
    package: Package,
    *,
    candidate_manifest: dict[str, Any] | None,
    promoted: dict[str, tuple[dict[str, np.ndarray], str]],
) -> dict[str, Any]:
    manifest_path = package.baseline_dir / "manifest.json"
    accepted = json.loads(manifest_path.read_text(encoding="utf-8"))

    if candidate_manifest is not None:
        for key in (
            "producer",
            "configuration",
            "scientific_audit",
            "shell_charge_diagnostic",
        ):
            if key in candidate_manifest:
                accepted[key] = deepcopy(candidate_manifest[key])
    if "status" in accepted:
        accepted["status"] = "accepted"
    _refresh_producer_metadata(accepted)

    if isinstance(accepted.get("states"), list):
        updated = []
        for baseline_record in accepted["states"]:
            baseline_name = _record_filename(baseline_record)
            if baseline_name is None or baseline_name not in promoted:
                raise ValueError(
                    f"{package.name}: manifest record lacks promoted data: "
                    f"{baseline_record}"
                )
            arrays, digest = promoted[baseline_name]
            candidate_name = _candidate_name(package, baseline_name)
            updated.append(
                _update_record(
                    dict(baseline_record),
                    _candidate_record(
                        candidate_manifest,
                        candidate_name=candidate_name,
                        baseline_record=dict(baseline_record),
                    ),
                    arrays,
                    baseline_name=baseline_name,
                    digest=digest,
                )
            )
        accepted["states"] = updated
    elif isinstance(accepted.get("state"), dict):
        baseline_record = dict(accepted["state"])
        baseline_name = _record_filename(baseline_record)
        if baseline_name is None or baseline_name not in promoted:
            raise ValueError(f"{package.name}: singleton state was not promoted.")
        arrays, digest = promoted[baseline_name]
        accepted["state"] = _update_record(
            baseline_record,
            _candidate_record(
                candidate_manifest,
                candidate_name=_candidate_name(package, baseline_name),
                baseline_record=baseline_record,
            ),
            arrays,
            baseline_name=baseline_name,
            digest=digest,
        )
    return accepted


def _validate_package(
    package: Package,
) -> tuple[
    dict[str, tuple[Path, dict[str, np.ndarray]]],
    dict[str, Any] | None,
]:
    baseline_names = {
        path.name for path in package.baseline_dir.glob("*.npz")
    }
    if not baseline_names:
        raise FileNotFoundError(f"{package.name}: no baseline NPZ files.")
    candidate_files: dict[str, tuple[Path, dict[str, np.ndarray]]] = {}
    for baseline_name in sorted(baseline_names):
        candidate_name = _candidate_name(package, baseline_name)
        candidate_path = package.candidate_dir / candidate_name
        if not candidate_path.is_file():
            raise FileNotFoundError(
                f"{package.name}: missing candidate {candidate_name}."
            )
        candidate_files[baseline_name] = (
            candidate_path,
            _load_candidate(candidate_path),
        )

    extra = {
        path.name
        for path in package.candidate_dir.glob("*.npz")
        if path.name
        not in {
            _candidate_name(package, baseline_name)
            for baseline_name in baseline_names
        }
    }
    if extra:
        raise ValueError(f"{package.name}: unexpected candidate files {sorted(extra)}.")
    return candidate_files, _candidate_manifest(package)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


METRIC_RUNNERS = {
    "al_qm_tf": (
        "benchmarks/runners/plot_al_qm_tf.py",
        "benchmarks/outputs/al_qm_tf/al_qm_tf_offline_metrics.csv",
        "metrics.csv",
    ),
    "carbon_lfc_sensitivity": (
        "benchmarks/runners/plot_carbon_lfc_sensitivity.py",
        "benchmarks/outputs/carbon_lfc_sensitivity/carbon_lfc_sensitivity_metrics.csv",
        "metrics.csv",
    ),
    "starrett_et_al_2014_mixtures_fig3": (
        "benchmarks/runners/plot_starrett_et_al_2014_mixtures_fig3.py",
        "benchmarks/outputs/starrett_et_al_2014_mixtures_fig3/fig3_ch1p36_offline_metrics.csv",
        "metrics.csv",
    ),
}


def _refresh_metrics(package: Package) -> None:
    if package.name not in METRIC_RUNNERS:
        return
    runner, generated_relative, baseline_name = METRIC_RUNNERS[package.name]
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", "/tmp/otter-matplotlib")
    subprocess.run(
        [sys.executable, str(ROOT / runner)],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    generated = ROOT / generated_relative
    destination = package.baseline_dir / baseline_name
    _atomic_copy(generated, destination)
    manifest_path = package.baseline_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if package.name == "starrett_et_al_2014_mixtures_fig3":
        manifest["baseline"]["metrics_file"] = baseline_name
        manifest["baseline"]["metrics_sha256"] = sha256_file(destination)
    else:
        manifest["metrics_file"] = baseline_name
        manifest["metrics_sha256"] = sha256_file(destination)
    _write_manifest(manifest_path, manifest)


def _refresh_existing_metadata(package: Package) -> None:
    promoted: dict[str, tuple[dict[str, np.ndarray], str]] = {}
    for path in sorted(package.baseline_dir.glob("*.npz")):
        arrays = _load_candidate(path)
        arrays.pop("metadata_json", None)
        manifest = json.loads(
            (package.baseline_dir / "manifest.json").read_text(encoding="utf-8")
        )
        payload = _with_compact_metadata(
            package,
            manifest=manifest,
            baseline_name=path.name,
            arrays=arrays,
        )
        save_npz_atomic(path, payload)
        promoted[path.name] = (payload, sha256_file(path))
    manifest = _merge_manifest(
        package,
        candidate_manifest=None,
        promoted=promoted,
    )
    _write_manifest(package.baseline_dir / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="replace baselines after all candidates pass validation",
    )
    parser.add_argument(
        "--keep-candidates",
        action="store_true",
        help="retain candidate directories after a successful promotion",
    )
    parser.add_argument(
        "--refresh-metadata-only",
        action="store_true",
        help="refresh embedded metadata and checksums in accepted baselines",
    )
    args = parser.parse_args()

    if args.refresh_metadata_only:
        for package in PACKAGES:
            _refresh_existing_metadata(package)
            print(f"[metadata] {package.name}")
        return

    validated = {
        package.name: _validate_package(package) for package in PACKAGES
    }
    total = sum(len(files) for files, _ in validated.values())
    print(f"Validated {total} candidate NPZ files in {len(PACKAGES)} packages.")
    if not args.apply:
        print("Dry run only; use --apply to replace project-generated baselines.")
        return

    for package in PACKAGES:
        candidate_files, candidate_manifest = validated[package.name]
        provisional = {
            baseline_name: (arrays, sha256_file(source))
            for baseline_name, (source, arrays) in candidate_files.items()
        }
        metadata_manifest = _merge_manifest(
            package,
            candidate_manifest=candidate_manifest,
            promoted=provisional,
        )
        promoted: dict[str, tuple[dict[str, np.ndarray], str]] = {}
        for baseline_name, (source, arrays) in candidate_files.items():
            destination = package.baseline_dir / baseline_name
            payload = _with_compact_metadata(
                package,
                manifest=metadata_manifest,
                baseline_name=baseline_name,
                arrays=arrays,
            )
            save_npz_atomic(destination, payload)
            promoted[baseline_name] = (payload, sha256_file(destination))
        manifest = _merge_manifest(
            package,
            candidate_manifest=candidate_manifest,
            promoted=promoted,
        )
        _write_manifest(package.baseline_dir / "manifest.json", manifest)
        _refresh_metrics(package)
        print(f"[promoted] {package.name}: {len(promoted)} state file(s)")

    if not args.keep_candidates:
        for package in PACKAGES:
            if package.name == "carbon_ionization_levels":
                for path in (
                    package.candidate_dir / "C_Te100eV_density_scan.npz",
                    package.candidate_dir
                    / "C_Te100eV_density_scan.manifest.json",
                    package.candidate_dir / "point_cache",
                    package.candidate_dir / "point_failures",
                ):
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.exists():
                        path.unlink()
            elif package.candidate_dir.exists():
                shutil.rmtree(package.candidate_dir)
    print("Promotion complete; no recomputed NPZ candidates remain.")


if __name__ == "__main__":
    main()
