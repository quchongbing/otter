r"""
Two-temperature aluminium: Johnson et al. (2025)
=================================================

This directly executable benchmark compares Otter IS-QOZ/HNC
:math:`g_{ii}(r)` with the three local curve families associated with
Fig. 2(a), 2(c), and 2(d) of :cite:t:`JohnsonEtAl2025`.  The states are
aluminium at :math:`\rho=2.7` g cm\ :sup:`-3`,
:math:`T_i=1` eV, and :math:`T_e=1,10,30` eV.  The paper labels the
reference methods as 2TTCP HNC+bridge, DFT-MD, and YOCP HNC+bridge.

The reference abscissa is :math:`r` in atomic units (Bohr), exactly as shown
on the published Fig. 2 axis.  The inherited ``plot_zak.py`` labelled the
same values as ångström; that was a plotting-label bug, not a conversion to
repeat here.

Edit only the input block below.  ``USE_PRECOMPUTED_DATA = True`` verifies
the dedicated accepted-baseline manifest and every NPZ checksum.  With
``False``, this same file calls the public Otter workflow, writes candidate
results under ``benchmarks/outputs``, and plots those freshly calculated
results.  It does not invoke a hidden runner.

The publication points have unresolved extraction provenance and are
published with article/panel attribution and license status ``NOASSERTION``.
See :doc:`the provenance and reuse notice
</benchmarks/johnson_et_al_2025_two_temperature_al>`.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from otter import PlasmaWorkflowConfig, solve_plasma_workflow
from otter.plotting import save_figure, set_style


# =============================================================================
# User input
# =============================================================================
USE_PRECOMPUTED_DATA = True

# Three independent states x six continuum workers uses at most about
# 18 worker processes on a 24-core workstation.
MAX_STATE_WORKERS = 3
CONTINUUM_WORKERS_PER_STATE = 6
AA_N_POINTS = 1024
QOZ_N_POINTS = 4096

LFC_MODEL = "chabrier1990"
HNC_TOL = 1.0e-4
HNC_CLOSURE_TOL = 2.5e-3
R_RETAIN_MAX_BOHR = 20.0
K_RETAIN_MAX_BOHR_INV = 20.0
# =============================================================================


BENCHMARK_ID = "johnson_et_al_2025_two_temperature_al"
STATES: tuple[dict[str, Any], ...] = (
    {
        "state_id": "al_rho2p7_te1_ti1",
        "te_ev": 1.0,
        "panel": "Fig. 2(a)",
    },
    {
        "state_id": "al_rho2p7_te10_ti1",
        "te_ev": 10.0,
        "panel": "Fig. 2(c)",
    },
    {
        "state_id": "al_rho2p7_te30_ti1",
        "te_ev": 30.0,
        "panel": "Fig. 2(d)",
    },
)
REFERENCE_METHODS: tuple[tuple[str, str, str], ...] = (
    ("AA", "2TTCP HNC+bridge", "o"),
    ("DFTMD", "DFT-MD", "x"),
    ("YOCP", "YOCP HNC+bridge", "s"),
)


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
            / "baselines"
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


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load only portable numeric/string arrays."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    if any(value.dtype.hasobject for value in result.values()):
        raise TypeError(f"Object arrays are forbidden in {path}.")
    return result


def load_precomputed_states() -> dict[str, dict[str, np.ndarray]]:
    """Load all and only accepted, checksummed dedicated baselines."""
    manifest = json.loads(
        (BASELINE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Unexpected baseline manifest.")
    expected_ids = {str(state["state_id"]) for state in STATES}
    records = {
        str(record["state_id"]): record for record in manifest["states"]
    }
    if set(records) != expected_ids:
        raise RuntimeError("Baseline manifest state coverage is incomplete.")
    terminal_statuses = {"accepted", "reference_only_strict_hnc_rejected"}
    unresolved = {
        state_id: str(record.get("status"))
        for state_id, record in records.items()
        if record.get("status") not in terminal_statuses
    }
    if unresolved:
        detail = ", ".join(
            f"{state_id}={status}"
            for state_id, status in sorted(unresolved.items())
        )
        raise RuntimeError(
            "The dedicated benchmark manifest still contains unreviewed "
            f"states ({detail}). Set USE_PRECOMPUTED_DATA = False to "
            "calculate candidates directly; review them before promotion."
        )

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for state_id, record in records.items():
        if record["status"] == "reference_only_strict_hnc_rejected":
            if record.get("baseline_file") is not None:
                raise RuntimeError(
                    f"Rejected state {state_id} must not name a baseline."
                )
            continue
        filename = record.get("baseline_file")
        expected_hash = record.get("baseline_sha256")
        if not filename or not expected_hash:
            raise RuntimeError(f"Accepted record {state_id} is incomplete.")
        path = BASELINE_DIR / str(filename)
        if sha256_file(path) != str(expected_hash):
            raise RuntimeError(f"Checksum mismatch for {path}.")
        payload = load_npz(path)
        if str(payload["state_id"].item()) != state_id:
            raise ValueError(f"State identifier mismatch in {path}.")
        loaded[state_id] = payload
    return loaded


def workflow_config(state: dict[str, Any]) -> PlasmaWorkflowConfig:
    """Build one complete public Otter IS-QOZ/HNC calculation."""
    return PlasmaWorkflowConfig(
        elements=["Al"],
        temperature_ev=float(state["te_ev"]),
        ion_temperature_ev=1.0,
        rho_g_cc=2.7,
        electronic_model="qm",
        aa_overrides={
            "n_points": int(AA_N_POINTS),
            "cont_n_jobs": int(CONTINUUM_WORKERS_PER_STATE),
            "cont_shards": int(2 * CONTINUUM_WORKERS_PER_STATE),
            "bound_occ_mode": "fd",
            "bound_rmax_mult": None,
            "bound_zero_tail_refine": False,
            "b3_tail_model": "full",
        },
        qoz_linear_n_points=int(QOZ_N_POINTS),
        qoz_pad_factor=2.0,
        qoz_zbar_mode="pseudoatom_partition",
        qoz_renormalize_nscr_to_zbar=True,
        qoz_response_chi0_model="lindhard_fd",
        qoz_response_lfc_model=str(LFC_MODEL),
        hnc_tol=float(HNC_TOL),
        hnc_closure_transform_tol=float(HNC_CLOSURE_TOL),
        hnc_max_iter=500,
        hnc_require_converged=True,
        show_progress=False,
    )


def pack_result(
    workflow: dict[str, Any],
    state: dict[str, Any],
    elapsed_s: float,
) -> dict[str, np.ndarray]:
    """Validate convergence and retain the portable benchmark arrays."""
    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"])
    if electronic.get("stage2_converged") is not True:
        raise RuntimeError("Full average-atom stage 2 did not converge.")
    if dict(electronic.get("ext_status", {})).get("converged") is not True:
        raise RuntimeError("External fixed-mu average atom did not converge.")
    if str(electronic.get("threshold_state_status", "")).lower() == "unresolved":
        raise RuntimeError("The threshold-state representation is unresolved.")
    if ion.get("hnc_converged") is not True:
        raise RuntimeError("HNC did not reach a physical fixed point.")
    if float(ion["hnc_output_residual"]) > HNC_TOL:
        raise RuntimeError("HNC residual exceeds the configured tolerance.")
    if float(ion["closure_transform_max_abs"]) > HNC_CLOSURE_TOL:
        raise RuntimeError("The g/S transform-closure audit failed.")

    r = np.asarray(ion["r"], dtype=float)
    k = np.asarray(ion["k"], dtype=float)
    r_mask = r <= R_RETAIN_MAX_BOHR
    k_mask = k <= K_RETAIN_MAX_BOHR_INV
    return {
        "schema_version": np.asarray("otter_johnson_2025_al_v1"),
        "state_id": np.asarray(str(state["state_id"])),
        "rho_g_cc": np.asarray(2.7),
        "te_ev": np.asarray(float(state["te_ev"])),
        "ti_ev": np.asarray(1.0),
        "paper_panel": np.asarray(str(state["panel"])),
        "producer_elapsed_s": np.asarray(float(elapsed_s)),
        "r_bohr": r[r_mask],
        "gii_r": np.asarray(ion["gii_r"], dtype=float)[r_mask],
        "k_bohr_inv": k[k_mask],
        "sii_k": np.asarray(ion["sii_k"], dtype=float)[k_mask],
        "zbar_partition": np.asarray(float(ion["zbar_partition"])),
        "hnc_solver_path": np.asarray(str(ion["hnc_solver_path"])),
        "hnc_fallback_used": np.asarray(bool(ion["hnc_fallback_used"])),
        "hnc_primary_best_residual": np.asarray(
            float(ion["hnc_primary_best_residual"])
        ),
        "hnc_best_residual": np.asarray(float(ion["hnc_output_residual"])),
        "hnc_s_min": np.asarray(float(ion["hnc_s_min"])),
        "hnc_closure_mismatch": np.asarray(
            float(ion["closure_transform_max_abs"])
        ),
    }


def solve_state(state: dict[str, Any]) -> dict[str, np.ndarray]:
    """Calculate one state in a worker process."""
    started = time.perf_counter()
    workflow = solve_plasma_workflow(workflow_config(state))
    return pack_result(workflow, state, time.perf_counter() - started)


def git_revision() -> dict[str, Any]:
    """Record the checkout revision without pretending dirty output is committed."""
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
    states: dict[str, dict[str, np.ndarray]],
    failures: dict[str, str],
) -> None:
    """Write candidates and a manifest with real file hashes."""
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for definition in STATES:
        state_id = str(definition["state_id"])
        if state_id in failures:
            records.append(
                {
                    "state_id": state_id,
                    "status": "strict_calculation_rejected",
                    "baseline_file": None,
                    "baseline_sha256": None,
                    "reason": failures[state_id],
                }
            )
            continue
        filename = f"{state_id}.npz"
        path = CANDIDATE_DIR / filename
        np.savez_compressed(path, **states[state_id])
        records.append(
            {
                "state_id": state_id,
                "status": "candidate_unreviewed",
                "baseline_file": filename,
                "baseline_sha256": sha256_file(path),
            }
        )
        print(f"[candidate] {path}")

    candidate_manifest = {
        "schema_version": "otter_candidate_manifest_v1",
        "benchmark_id": BENCHMARK_ID,
        "producer": {
            "script_relative_path": (
                "benchmarks/examples/"
                "plot_johnson_et_al_2025_two_temperature_al.py"
            ),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            **git_revision(),
        },
        "configuration": {
            "aa_n_points": AA_N_POINTS,
            "qoz_n_points_before_padding": QOZ_N_POINTS,
            "continuum_workers_per_state": CONTINUUM_WORKERS_PER_STATE,
            "lfc_model": LFC_MODEL,
            "hnc_tolerance": HNC_TOL,
            "hnc_transform_closure_tolerance": HNC_CLOSURE_TOL,
            "hnc_primary_solver": "anderson",
            "hnc_fallback_solver": "newton_krylov",
            "hnc_potential_scales": [
                float(value)
                for value in workflow_config(STATES[0]).hnc_potential_scales
            ],
            "hnc_adaptive_continuation": bool(
                workflow_config(STATES[0]).hnc_adaptive_continuation
            ),
        },
        "states": records,
    }
    (CANDIDATE_DIR / "candidate_manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def solve_all_states() -> dict[str, dict[str, np.ndarray]]:
    """Calculate the three independent states with bounded parallelism."""
    solved: dict[str, dict[str, np.ndarray]] = {}
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=MAX_STATE_WORKERS) as pool:
        futures = {pool.submit(solve_state, state): state for state in STATES}
        for future in as_completed(futures):
            state = futures[future]
            state_id = str(state["state_id"])
            try:
                solved[state_id] = future.result()
            except Exception as exc:
                failures[state_id] = f"{type(exc).__name__}: {exc}"
                print(f"[strictly rejected] {state_id}: {exc}")
            else:
                print(f"[computed] {state_id}")
    save_candidates(solved, failures)
    if not solved:
        raise RuntimeError("Every Otter state failed its strict checks.")
    return solved


def reference_filename(te_ev: float, token: str) -> str:
    return (
        f"Zak_2025_Al_rho2.7_Te{te_ev:.1f}_Ti1.0_{token}.csv"
    )


def load_references() -> dict[tuple[float, str], tuple[np.ndarray, np.ndarray]]:
    """Verify the attributed reference manifest and load unchanged Bohr values."""
    manifest = json.loads(
        (REFERENCE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("reference_id") != BENCHMARK_ID:
        raise ValueError("Unexpected reference-data manifest.")
    if manifest.get("units", {}).get("column_1") != "Bohr":
        raise ValueError("Johnson Fig. 2 reference radius must be Bohr.")
    by_path = {str(item["path"]): item for item in manifest["files"]}
    references: dict[tuple[float, str], tuple[np.ndarray, np.ndarray]] = {}
    for state in STATES:
        te_ev = float(state["te_ev"])
        for token, _label, _marker in REFERENCE_METHODS:
            filename = reference_filename(te_ev, token)
            record = by_path.get(filename)
            if record is None:
                raise FileNotFoundError(f"Missing manifest record {filename}.")
            path = REFERENCE_DIR / filename
            if sha256_file(path) != str(record["sha256"]):
                raise RuntimeError(f"Checksum mismatch for {path}.")
            values = np.asarray(np.genfromtxt(path, delimiter=","), dtype=float)
            if values.ndim != 2 or values.shape[1] < 2:
                raise ValueError(f"Expected two columns in {path}.")
            mask = np.isfinite(values[:, 0]) & np.isfinite(values[:, 1])
            references[(te_ev, token)] = (
                values[mask, 0],
                values[mask, 1],
            )
    return references


states = (
    load_precomputed_states()
    if USE_PRECOMPUTED_DATA
    else solve_all_states()
)
references = load_references()
print(
    "Using "
    + (
        "reviewed, checksummed Otter baselines."
        if USE_PRECOMPUTED_DATA
        else "new candidates calculated directly by this gallery script."
    )
)


def print_metrics() -> None:
    """Report interpolation errors without treating unlike methods as exact."""
    print(
        f"{'state':24s} {'reference':20s} "
        f"{'RMSE':>10s} {'MAE':>10s} {'max':>10s}"
    )
    for definition in STATES:
        state_id = str(definition["state_id"])
        if state_id not in states:
            print(f"{state_id:24s} {'Otter strict HNC rejected':20s}")
            continue
        r_otter = np.asarray(states[state_id]["r_bohr"], dtype=float)
        g_otter = np.asarray(states[state_id]["gii_r"], dtype=float)
        te_ev = float(definition["te_ev"])
        for token, label, _marker in REFERENCE_METHODS:
            r_ref, g_ref = references[(te_ev, token)]
            mask = (r_ref >= r_otter[0]) & (r_ref <= r_otter[-1])
            delta = np.interp(r_ref[mask], r_otter, g_otter) - g_ref[mask]
            print(
                f"{state_id:24s} {label:20s} "
                f"{np.sqrt(np.mean(delta**2)):10.4e} "
                f"{np.mean(np.abs(delta)):10.4e} "
                f"{np.max(np.abs(delta)):10.4e}"
            )


print_metrics()


# %%
# Pair-distribution comparison
# ----------------------------
#
# Otter is a pseudoatom IS-QOZ/HNC calculation and is not relabelled as the
# paper's 2TTCP+bridge method.  Agreement or disagreement is therefore a
# scientific comparison, not an implementation identity test.

set_style("thesis", palette="bing")
fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for panel_index, (axis, definition) in enumerate(zip(axes, STATES)):
    state_id = str(definition["state_id"])
    te_ev = float(definition["te_ev"])
    if state_id in states:
        payload = states[state_id]
        axis.plot(
            np.asarray(payload["r_bohr"]),
            np.asarray(payload["gii_r"]),
            color="black",
            lw=2.3,
            label="Otter IS-QOZ/HNC",
            zorder=2,
        )
    else:
        axis.text(
            0.97,
            0.05,
            "Otter curve withheld:\nstrict HNC rejected",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="0.25",
            fontsize=9,
        )
    for method_index, (token, label, marker) in enumerate(REFERENCE_METHODS):
        r_ref, g_ref = references[(te_ev, token)]
        options: dict[str, Any] = {
            "s": 40,
            "marker": marker,
            "linewidths": 1.3,
            "label": label,
            "zorder": 3,
        }
        color = colors[method_index % len(colors)]
        if marker == "x":
            options["color"] = color
        else:
            options["facecolors"] = "none"
            options["edgecolors"] = color
        axis.scatter(r_ref, g_ref, **options)

    axis.set_title(
        rf"{definition['panel']}: $T_e={te_ev:g}$ eV, $T_i=1$ eV"
    )
    axis.set_xlabel(r"$r$ [Bohr]")
    axis.set_xlim(-0.5, 8.2)
    axis.set_ylim(-0.05, 2.30)
    axis.axhline(1.0, color="0.55", lw=0.8, ls=":")
    if panel_index == 0:
        axis.set_ylabel(r"$g_{ii}(r)$")
        axis.legend(fontsize=9, loc="best")

fig.suptitle(
    r"Al, $\rho=2.7$ g cm$^{-3}$: "
    "Otter versus Johnson et al. (2025)",
    y=0.985,
)
fig.text(
    0.5,
    0.006,
    "Reference data: Johnson et al. (2025), "
    "doi:10.1103/5c29-kdx1.",
    ha="center",
    va="bottom",
    fontsize=7.5,
)
fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.95), pad=0.45)
saved_paths = save_figure(
    fig,
    FIGURE_DIR / "johnson_et_al_2025_two_temperature_al_gii",
)
print(
    "[figure] "
    + ", ".join(
        f"{kind}={path.relative_to(ROOT)}"
        for kind, path in saved_paths.items()
    )
)
if "agg" not in plt.get_backend().lower():
    plt.show()
