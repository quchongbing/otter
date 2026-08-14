r"""
Single-species Starrett--Saumon ion-structure benchmarks
========================================================

This benchmark compares Otter IS-QOZ/HNC pair-distribution functions with
Starrett TF numerical extractions from
:cite:t:`StarrettSaumon2014` (doi:
`10.1016/j.hedp.2013.12.001
<https://doi.org/10.1016/j.hedp.2013.12.001>`__).  Published data are shown
only as open circles.  Otter KS-DFT (QM) calculations are requested for every
thermodynamic state; the tungsten panels additionally request Thomas--Fermi
(TF), matching the source comparison.  Curves are shown only when they pass
the electronic, HNC, and transform-closure gates.

Edit the input block below.  ``USE_PRECOMPUTED_DATA = True`` verifies and
loads only strictly accepted Otter archives.  With ``False``, this same script
calls the public Otter workflow for every state/model, saves accepted candidate
archives and explicit rejection records under ``benchmarks/outputs``, and
plots the newly calculated results.  It never promotes unconverged AA or HNC
best-effort output.

Every figure is saved as both a documentation PNG and a vector PDF suitable
for slides.  The publication extractions are attributed reference data with
license status ``NOASSERTION``; see :doc:`the provenance and reuse notice
</benchmarks/starrett_single_species_2013_2014>`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from otter import PlasmaWorkflowConfig, solve_plasma_workflow
from otter.plotting import grid_figsize, save_figure, set_style


# =============================================================================
# User input
# =============================================================================
USE_PRECOMPUTED_DATA = True

# Three state processes, each with six continuum workers.  Use one state
# worker on a memory-constrained host.
MAX_STATE_WORKERS = 3
CONTINUUM_WORKERS_PER_STATE = 6
QOZ_N_POINTS = int(
    os.environ.get("OTTER_STARRETT_SINGLE_QOZ_POINTS", "4096")
)

LFC_MODEL = "chabrier1990"
HNC_TOL = 1.0e-4
HNC_CLOSURE_TOL = 2.5e-3
R_RETAIN_MAX_BOHR = 20.0
K_RETAIN_MAX_BOHR_INV = 20.0

# An empty tuple means all state/model combinations.  Users may list stable
# IDs from CALCULATIONS below for a shorter live calculation.
STATE_IDS_TO_COMPUTE: tuple[str, ...] = ()

# Maintenance override; normal users only need to edit USE_PRECOMPUTED_DATA.
if os.environ.get("OTTER_RECOMPUTE_STARRETT_SINGLE", "0") == "1":
    USE_PRECOMPUTED_DATA = False
if selected_ids := os.environ.get("OTTER_STARRETT_SINGLE_STATE_IDS"):
    STATE_IDS_TO_COMPUTE = tuple(
        item.strip() for item in selected_ids.split(",") if item.strip()
    )
# =============================================================================


BENCHMARK_ID = "starrett_single_species_2013_2014"
SCHEMA = "otter_starrett_single_species_state_v1"
BOHR_TO_ANGSTROM = 0.529177210903
AMU_TO_G = 1.66053906660e-24
BOHR_TO_CM = BOHR_TO_ANGSTROM * 1.0e-8

PHYSICAL_STATES: tuple[dict[str, Any], ...] = (
    {
        "panel_id": "fe_10",
        "element": "Fe",
        "atomic_mass": 55.845,
        "rho_g_cc": 22.5,
        "te_ev": 10.0,
        "ti_ev": 10.0,
        "title": r"Fe, 22.5 g cm$^{-3}$, 10 eV",
        "x_max": 5.0,
    },
    {
        "panel_id": "h_5",
        "element": "H",
        "atomic_mass": 1.008,
        "rho_g_cc": 80.0,
        "te_ev": 5.0,
        "ti_ev": 5.0,
        "title": r"H, 80 g cm$^{-3}$, 5 eV",
        "x_max": 6.2,
    },
    {
        "panel_id": "h_172",
        "element": "H",
        "atomic_mass": 1.008,
        "rho_g_cc": 80.0,
        "te_ev": 172.0,
        "ti_ev": 172.0,
        "title": r"H, 80 g cm$^{-3}$, 172 eV",
        "x_max": 6.2,
    },
    {
        "panel_id": "c_64p64",
        "element": "C",
        "atomic_mass": 12.011,
        "rho_g_cc": 12.64,
        "te_ev": 64.64,
        "ti_ev": 64.64,
        "title": r"C, 12.64 g cm$^{-3}$, 64.64 eV",
        "x_max": 4.0,
    },
    {
        "panel_id": "w_10",
        "element": "W",
        "atomic_mass": 183.84,
        "rho_g_cc": 40.0,
        "te_ev": 10.0,
        "ti_ev": 10.0,
        "title": r"W, 40 g cm$^{-3}$, 10 eV",
        "x_max": 6.2,
    },
    {
        "panel_id": "w_60",
        "element": "W",
        "atomic_mass": 183.84,
        "rho_g_cc": 40.0,
        "te_ev": 60.0,
        "ti_ev": 60.0,
        "title": r"W, 40 g cm$^{-3}$, 60 eV",
        "x_max": 6.2,
    },
)

CALCULATIONS: tuple[dict[str, Any], ...] = tuple(
    {
        **state,
        "model": model,
        "state_id": f"{state['panel_id']}_{model}",
    }
    for state in PHYSICAL_STATES
    for model in ("qm", "tf")
)

REFERENCE_SERIES: dict[str, tuple[dict[str, str], ...]] = {
    "fe_10": (
        {
            "file": "gii_Fe_22.5gcc_10.0ev_starrett.csv",
            "label": "Starrett TF",
            "coordinate": "r_over_rws",
        },
    ),
    "h_5": (
        {
            "file": "gii_H_80gcc_5.0ev_starrett.csv",
            "label": "Starrett TF",
            "coordinate": "bohr",
        },
    ),
    "h_172": (
        {
            "file": "gii_H_80gcc_172.0ev_starrett.csv",
            "label": "Starrett TF",
            "coordinate": "bohr",
        },
    ),
    "c_64p64": (
        {
            "file": "gii_C_12.64gcc_64.64ev_starrett.csv",
            "label": "Starrett TF",
            "coordinate": "r_over_rws",
        },
    ),
    "w_10": (
        {
            "file": "gii_W_40gcc_10.0ev_starrett_TF.csv",
            "label": "Starrett TF",
            "coordinate": "bohr",
        },
    ),
    "w_60": (
        {
            "file": "gii_W_40gcc_60.0ev_starrett.csv",
            "label": "Starrett TF",
            "coordinate": "bohr",
        },
    ),
}


def repository_root() -> Path:
    """Locate the checkout when run directly or through Sphinx-Gallery."""
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    source_file = globals().get("__file__")
    if source_file is not None:
        source = Path(str(source_file)).resolve()
        candidates.extend([source.parent, *source.parents])
    for candidate in candidates:
        manifest = (
            candidate
            / "benchmarks"
            / "reference_data"
            / BENCHMARK_ID
            / "manifest.json"
        )
        if manifest.is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the Otter checkout.")


ROOT = repository_root()
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / BENCHMARK_ID
REFERENCE_DIR = ROOT / "benchmarks" / "reference_data" / BENCHMARK_ID
OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / BENCHMARK_ID
CANDIDATE_DIR = OUTPUT_DIR / "gallery_recomputed"
FIGURE_DIR = OUTPUT_DIR / "figures"
CONTROLLER_PATH = (
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_starrett_single_species_2013_2014.py"
)
_REFERENCE_RECORDS_CACHE: dict[str, dict[str, Any]] | None = None


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load only portable, pickle-free arrays."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    if any(value.dtype.hasobject for value in result.values()):
        raise TypeError(f"Object arrays are forbidden in {path}.")
    return result


def reference_records_by_file() -> dict[str, dict[str, Any]]:
    """Verify the attributed reference manifest and return file records."""
    global _REFERENCE_RECORDS_CACHE
    if _REFERENCE_RECORDS_CACHE is not None:
        return _REFERENCE_RECORDS_CACHE

    manifest_path = REFERENCE_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "otter_reference_manifest_v1":
        raise ValueError(f"Unexpected reference schema in {manifest_path}.")
    if manifest.get("reference_id") != BENCHMARK_ID:
        raise ValueError(f"Unexpected reference identifier in {manifest_path}.")
    if manifest.get("redistribution_status") != (
        "published_by_maintainer_with_attribution"
    ):
        raise ValueError("Reference-data release status was not reviewed.")

    records = {
        str(record["path"]): dict(record) for record in manifest["files"]
    }
    expected = {
        str(record["file"])
        for panel_records in REFERENCE_SERIES.values()
        for record in panel_records
    }
    if not expected.issubset(records):
        raise RuntimeError(
            "Reference manifest lacks a curve listed in REFERENCE_SERIES."
        )
    for filename, record in records.items():
        path = REFERENCE_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(f"Reference checksum mismatch for {path}.")

    _REFERENCE_RECORDS_CACHE = records
    return records


def validated_two_column_curve(path: Path) -> np.ndarray:
    """Load one finite two-column curve and return it in radius order.

    The inherited W 60 eV ``60.0ev`` extraction contains one adjacent
    coordinate inversion.  The source CSV remains byte-for-byte unchanged;
    plotting applies a stable coordinate sort after rejecting duplicate
    radii.
    """
    raw = np.asarray(np.loadtxt(path, delimiter=","), dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] < 2:
        raise ValueError(f"{path} must contain an N x 2 curve with N >= 2.")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{path} contains non-finite values.")
    if np.unique(raw[:, 0]).size != raw.shape[0]:
        raise ValueError(f"{path} contains duplicate radius coordinates.")
    order = np.argsort(raw[:, 0], kind="stable")
    ordered = raw[order]
    if not np.all(np.diff(ordered[:, 0]) > 0.0):
        raise ValueError(f"{path} radius coordinates cannot be ordered.")
    return ordered


def ion_sphere_radius_bohr(state: dict[str, Any]) -> float:
    """Return R_WS from mass density and the standard atomic mass."""
    ion_density_cm3 = float(state["rho_g_cc"]) / (
        float(state["atomic_mass"]) * AMU_TO_G
    )
    radius_cm = (3.0 / (4.0 * np.pi * ion_density_cm3)) ** (1.0 / 3.0)
    return float(radius_cm / BOHR_TO_CM)


def load_reference(panel_id: str) -> list[dict[str, Any]]:
    """Load raw publication coordinates and map only at plot time."""
    manifest_records = reference_records_by_file()
    state = next(
        item for item in PHYSICAL_STATES if item["panel_id"] == panel_id
    )
    rws = ion_sphere_radius_bohr(state)
    loaded: list[dict[str, Any]] = []
    for record in REFERENCE_SERIES[panel_id]:
        filename = str(record["file"])
        manifest_record = manifest_records[filename]
        raw = validated_two_column_curve(REFERENCE_DIR / filename)
        coordinate = str(record["coordinate"])
        if coordinate == "bohr":
            if manifest_record.get("column_1_unit") != "Bohr":
                raise ValueError(f"Unit mismatch for {filename}.")
            x = raw[:, 0] / rws
        elif coordinate == "r_over_rws":
            if (
                manifest_record.get("column_1_unit")
                != "r_over_R_WS_dimensionless"
            ):
                raise ValueError(f"Unit mismatch for {filename}.")
            x = raw[:, 0]
        else:
            raise ValueError(f"Unknown coordinate convention: {coordinate}")
        loaded.append(
            {
                **record,
                "r_over_rws": np.asarray(x, dtype=float),
                "gii": np.asarray(raw[:, 1], dtype=float),
            }
        )
    return loaded


def validate_baseline_payload(
    payload: dict[str, np.ndarray],
    state: dict[str, Any],
    record: dict[str, Any],
    path: Path,
) -> None:
    """Validate scientific metadata and arrays in one compact baseline."""
    required = {
        "schema_version",
        "state_id",
        "panel_id",
        "element",
        "electronic_model",
        "rho_g_cc",
        "te_ev",
        "ti_ev",
        "r_ws_bohr",
        "r_bohr",
        "gii_r",
        "k_bohr_inv",
        "sii_k",
        "zbar_partition",
        "hnc_output_residual",
        "closure_transform_max_abs",
        "threshold_state_status",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")

    expected_strings = {
        "schema_version": SCHEMA,
        "state_id": str(state["state_id"]),
        "panel_id": str(state["panel_id"]),
        "element": str(state["element"]),
        "electronic_model": str(state["model"]),
    }
    for key, expected in expected_strings.items():
        value = np.asarray(payload[key])
        if value.shape != () or str(value.item()) != expected:
            raise ValueError(f"{path}: invalid {key}.")

    expected_scalars = {
        "rho_g_cc": float(state["rho_g_cc"]),
        "te_ev": float(state["te_ev"]),
        "ti_ev": float(state["ti_ev"]),
    }
    for key, expected in expected_scalars.items():
        value = np.asarray(payload[key])
        if value.shape != () or not np.isfinite(float(value)):
            raise ValueError(f"{path}: invalid {key}.")
        if not np.isclose(float(value), expected, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{path}: metadata mismatch for {key}.")

    r_ws = float(np.asarray(payload["r_ws_bohr"]))
    expected_r_ws = ion_sphere_radius_bohr(state)
    if not np.isfinite(r_ws) or not np.isclose(
        r_ws, expected_r_ws, rtol=1.0e-8, atol=1.0e-10
    ):
        raise ValueError(f"{path}: inconsistent ion-sphere radius.")

    r = np.asarray(payload["r_bohr"], dtype=float)
    g = np.asarray(payload["gii_r"], dtype=float)
    k = np.asarray(payload["k_bohr_inv"], dtype=float)
    s = np.asarray(payload["sii_k"], dtype=float)
    for name, grid, values in (
        ("real-space", r, g),
        ("reciprocal-space", k, s),
    ):
        if grid.ndim != 1 or values.ndim != 1 or grid.size < 2:
            raise ValueError(f"{path}: invalid {name} array dimensions.")
        if grid.shape != values.shape:
            raise ValueError(f"{path}: mismatched {name} arrays.")
        if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: non-finite {name} data.")
        if not np.all(np.diff(grid) > 0.0) or grid[0] <= 0.0:
            raise ValueError(f"{path}: {name} grid is not positive/ordered.")
    if r[-1] > R_RETAIN_MAX_BOHR + 1.0e-10:
        raise ValueError(f"{path}: real-space data exceed the retention limit.")
    if k[-1] > K_RETAIN_MAX_BOHR_INV + 1.0e-10:
        raise ValueError(
            f"{path}: reciprocal-space data exceed the retention limit."
        )

    residual = float(np.asarray(payload["hnc_output_residual"]))
    closure = float(np.asarray(payload["closure_transform_max_abs"]))
    zbar = float(np.asarray(payload["zbar_partition"]))
    threshold = str(np.asarray(payload["threshold_state_status"]).item()).lower()
    if not np.isfinite(zbar) or zbar <= 0.0:
        raise ValueError(f"{path}: invalid pseudoatom-partition Zbar.")
    if not np.isfinite(residual) or residual > HNC_TOL:
        raise ValueError(f"{path}: HNC residual does not pass the strict gate.")
    if not np.isfinite(closure) or closure > HNC_CLOSURE_TOL:
        raise ValueError(f"{path}: transform closure does not pass the gate.")
    if threshold == "unresolved":
        raise ValueError(f"{path}: unresolved threshold representation.")

    manifest_scalars = {
        "zbar_partition": zbar,
        "hnc_output_residual": residual,
        "closure_transform_max_abs": closure,
    }
    for key, payload_value in manifest_scalars.items():
        if not np.isclose(
            float(record[key]), payload_value, rtol=1.0e-12, atol=1.0e-14
        ):
            raise ValueError(f"{path}: manifest mismatch for {key}.")


def load_precomputed_states() -> dict[str, dict[str, np.ndarray]]:
    """Verify the manifest and load all strictly accepted states."""
    manifest = json.loads(
        (BASELINE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Unexpected baseline manifest.")
    controller_hash = manifest.get("producer", {}).get(
        "current_controller_sha256"
    )
    if controller_hash != sha256_file(CONTROLLER_PATH):
        raise RuntimeError(
            "Baseline controller hash does not match this gallery script."
        )
    expected = {str(item["state_id"]) for item in CALCULATIONS}
    definitions = {str(item["state_id"]): item for item in CALCULATIONS}
    records = {
        str(record["state_id"]): record for record in manifest["states"]
    }
    if set(records) != expected:
        raise RuntimeError("Baseline manifest state coverage is incomplete.")
    terminal = {
        "accepted",
        "strict_aa_rejected",
        "strict_hnc_rejected",
        "not_calculated",
    }
    bad = {
        key: record.get("status")
        for key, record in records.items()
        if record.get("status") not in terminal
    }
    if bad:
        raise RuntimeError(f"Unreviewed baseline records: {bad}")

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for state_id, record in records.items():
        if record["status"] != "accepted":
            if record.get("baseline_file") is not None:
                raise RuntimeError(
                    f"Rejected state {state_id} names a baseline file."
                )
            continue
        path = BASELINE_DIR / str(record["baseline_file"])
        if sha256_file(path) != str(record["baseline_sha256"]):
            raise RuntimeError(f"Checksum mismatch for {path}.")
        payload = load_npz(path)
        validate_baseline_payload(
            payload, definitions[state_id], record, path
        )
        loaded[state_id] = payload
    return loaded


def workflow_config(state: dict[str, Any]) -> PlasmaWorkflowConfig:
    """Build one complete public Otter IS-QOZ/HNC calculation."""
    return PlasmaWorkflowConfig(
        elements=[str(state["element"])],
        temperature_ev=float(state["te_ev"]),
        ion_temperature_ev=float(state["ti_ev"]),
        rho_g_cc=float(state["rho_g_cc"]),
        electronic_model=str(state["model"]),
        aa_overrides={
            "cont_n_jobs": int(CONTINUUM_WORKERS_PER_STATE),
            "cont_shards": int(2 * CONTINUUM_WORKERS_PER_STATE),
        },
        qoz_linear_n_points=int(QOZ_N_POINTS),
        hnc_tol=float(HNC_TOL),
        hnc_closure_transform_tol=float(HNC_CLOSURE_TOL),
        hnc_max_iter=1000,
    )


def pack_result(
    workflow: dict[str, Any],
    state: dict[str, Any],
    elapsed_s: float,
) -> dict[str, np.ndarray]:
    """Apply strict gates and retain compact numerical output."""
    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"])
    if electronic.get("stage2_converged") is not True:
        raise RuntimeError("full average-atom stage 2 did not converge")
    if dict(electronic.get("ext_status", {})).get("converged") is not True:
        raise RuntimeError("external fixed-mu average atom did not converge")
    threshold = str(
        electronic.get("threshold_state_status", "")
    ).lower()
    if threshold == "unresolved":
        raise RuntimeError("threshold-state representation is unresolved")
    if ion.get("hnc_converged") is not True:
        raise RuntimeError("HNC did not reach a physical fixed point")
    residual = float(ion["hnc_output_residual"])
    closure = float(ion["closure_transform_max_abs"])
    if residual > HNC_TOL:
        raise RuntimeError(
            f"HNC residual {residual:.6e} exceeds {HNC_TOL:.6e}"
        )
    if closure > HNC_CLOSURE_TOL:
        raise RuntimeError(
            f"transform closure {closure:.6e} exceeds "
            f"{HNC_CLOSURE_TOL:.6e}"
        )

    r = np.asarray(ion["r"], dtype=float)
    k = np.asarray(ion["k"], dtype=float)
    r_mask = r <= R_RETAIN_MAX_BOHR
    k_mask = k <= K_RETAIN_MAX_BOHR_INV
    rws = float(electronic["r_ws"])
    return {
        "schema_version": np.asarray(SCHEMA),
        "state_id": np.asarray(str(state["state_id"])),
        "panel_id": np.asarray(str(state["panel_id"])),
        "element": np.asarray(str(state["element"])),
        "electronic_model": np.asarray(str(state["model"])),
        "rho_g_cc": np.asarray(float(state["rho_g_cc"])),
        "te_ev": np.asarray(float(state["te_ev"])),
        "ti_ev": np.asarray(float(state["ti_ev"])),
        "producer_elapsed_s": np.asarray(float(elapsed_s)),
        "r_ws_bohr": np.asarray(rws),
        "r_bohr": r[r_mask],
        "gii_r": np.asarray(ion["gii_r"], dtype=float)[r_mask],
        "k_bohr_inv": k[k_mask],
        "sii_k": np.asarray(ion["sii_k"], dtype=float)[k_mask],
        "zbar_partition": np.asarray(float(ion["zbar_partition"])),
        "hnc_output_residual": np.asarray(residual),
        "closure_transform_max_abs": np.asarray(closure),
        "threshold_state_status": np.asarray(threshold),
    }


def solve_state(state: dict[str, Any]) -> dict[str, np.ndarray]:
    """Calculate one state/model in a worker process."""
    started = time.perf_counter()
    workflow = solve_plasma_workflow(workflow_config(state))
    return pack_result(workflow, state, time.perf_counter() - started)


def git_revision() -> dict[str, Any]:
    """Record the local revision without claiming dirty work is committed."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "worktree_dirty": None}
    return {"git_commit": head, "worktree_dirty": dirty}


def save_candidates(
    solved: dict[str, dict[str, np.ndarray]],
    failures: dict[str, str],
    attempted: set[str],
) -> None:
    """Save accepted candidates and explicit non-promoted failures."""
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for state in CALCULATIONS:
        state_id = str(state["state_id"])
        if state_id not in attempted:
            records.append(
                {
                    "state_id": state_id,
                    "status": "not_attempted_by_state_filter",
                    "candidate_file": None,
                    "candidate_sha256": None,
                }
            )
        elif state_id in failures:
            records.append(
                {
                    "state_id": state_id,
                    "status": "strict_calculation_rejected",
                    "candidate_file": None,
                    "candidate_sha256": None,
                    "reason": failures[state_id],
                }
            )
        else:
            filename = f"{state_id}.npz"
            path = CANDIDATE_DIR / filename
            np.savez_compressed(path, **solved[state_id])
            records.append(
                {
                    "state_id": state_id,
                    "status": "candidate_passed_strict_gates",
                    "candidate_file": filename,
                    "candidate_sha256": sha256_file(path),
                }
            )
    controller = CONTROLLER_PATH
    manifest = {
        "schema_version": "otter_candidate_manifest_v1",
        "benchmark_id": BENCHMARK_ID,
        "producer": {
            "script_relative_path": str(controller.relative_to(ROOT)),
            "script_sha256": sha256_file(controller),
            **git_revision(),
        },
        "states": records,
    }
    (CANDIDATE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def solve_selected_states() -> dict[str, dict[str, np.ndarray]]:
    """Run selected state/models with bounded process-level parallelism."""
    selected = set(STATE_IDS_TO_COMPUTE)
    jobs = [
        state
        for state in CALCULATIONS
        if not selected or str(state["state_id"]) in selected
    ]
    unknown = selected - {str(item["state_id"]) for item in CALCULATIONS}
    if unknown:
        raise ValueError(f"Unknown STATE_IDS_TO_COMPUTE: {sorted(unknown)}")
    solved: dict[str, dict[str, np.ndarray]] = {}
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=MAX_STATE_WORKERS) as pool:
        future_map = {pool.submit(solve_state, state): state for state in jobs}
        for future in as_completed(future_map):
            state = future_map[future]
            state_id = str(state["state_id"])
            try:
                solved[state_id] = future.result()
                print(f"[accepted] {state_id}")
            except Exception as exc:  # scientific rejection is recorded
                failures[state_id] = f"{type(exc).__name__}: {exc}"
                print(f"[rejected] {state_id}: {failures[state_id]}")
    save_candidates(
        solved,
        failures,
        {str(state["state_id"]) for state in jobs},
    )
    return solved


states = (
    load_precomputed_states()
    if USE_PRECOMPUTED_DATA
    else solve_selected_states()
)

print(
    "Using "
    + (
        "checksummed, precomputed Otter states."
        if USE_PRECOMPUTED_DATA
        else "new states calculated directly by this gallery script."
    )
)
for definition in CALCULATIONS:
    state_id = str(definition["state_id"])
    if state_id not in states:
        continue
    payload = states[state_id]
    print(
        f"{state_id:18s}  "
        f"Zbar={float(payload['zbar_partition']):.8f}  "
        f"HNC={float(payload['hnc_output_residual']):.3e}  "
        f"closure={float(payload['closure_transform_max_abs']):.3e}"
    )


# %%
# All panels use :math:`r/R_{\rm WS}`.  Explicitly Bohr-valued reference
# coordinates are converted at plot time without modifying the stored CSV
# files.

set_style("thesis", palette="bing")
fig, axes = plt.subplots(
    2,
    3,
    figsize=grid_figsize(2, 3),
    squeeze=False,
)
axes_flat = axes.ravel()
reference_colors = ("#333333", "#666666")
model_styles = {
    "qm": {
        "label": "Otter QM",
        "color": "#0072B2",
        "ls": "-",
    },
    "tf": {"label": "Otter TF", "color": "#D55E00", "ls": "--"},
}

for axis, physical in zip(axes_flat, PHYSICAL_STATES, strict=True):
    panel_id = str(physical["panel_id"])
    for index, reference in enumerate(load_reference(panel_id)):
        axis.plot(
            reference["r_over_rws"],
            reference["gii"],
            ls="none",
            marker="o",
            ms=5.4,
            mfc="none",
            mew=1.15,
            color=reference_colors[index % len(reference_colors)],
            label=str(reference["label"]),
        )

    for model in ("qm", "tf"):
        state_id = f"{panel_id}_{model}"
        if state_id not in states:
            continue
        payload = states[state_id]
        style = model_styles[model]
        axis.plot(
            np.asarray(payload["r_bohr"], dtype=float)
            / float(payload["r_ws_bohr"]),
            np.asarray(payload["gii_r"], dtype=float),
            color=str(style["color"]),
            ls=str(style["ls"]),
            lw=2.0,
            label=str(style["label"]),
        )

    missing_models = [
        model
        for model in ("qm", "tf")
        if f"{panel_id}_{model}" not in states
    ]
    if missing_models:
        axis.text(
            0.03,
            0.05,
            "No accepted Otter "
            + "/".join(model.upper() for model in missing_models)
            + " result",
            transform=axis.transAxes,
            fontsize="small",
            color="0.35",
        )

    axis.axhline(1.0, color="#777777", ls=":", lw=0.8)
    axis.set_xlim(-0.5, float(physical["x_max"]))
    axis.set_ylim(-0.04, 1.86)
    axis.set_title(str(physical["title"]))
    axis.set_xlabel(r"$r/R_{\rm WS}$")
    axis.set_ylabel(r"$g_{ii}(r)$")
    axis.legend(fontsize="small", loc="best")

fig.suptitle(
    "Single-species ion structure: Otter and Starrett–Saumon references",
    y=0.985,
)
fig.text(
    0.5,
    0.006,
    "Reference data: Starrett and Saumon (2014), "
    "doi:10.1016/j.hedp.2013.12.001.",
    ha="center",
    va="bottom",
    fontsize=7.5,
)
fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.965), pad=0.55)
saved = save_figure(
    fig,
    FIGURE_DIR / "starrett_single_species_2013_2014",
    formats=("png", "pdf"),
)
print(
    "[saved] "
    + ", ".join(str(path.relative_to(ROOT)) for path in saved.values())
)
