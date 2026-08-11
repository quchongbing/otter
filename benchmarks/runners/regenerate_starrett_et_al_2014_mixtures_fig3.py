"""Recompute the nine-state Starrett et al. CH1.36 Figure 3 benchmark.

This is the expensive Otter producer.  It is deliberately separate from the
read-only plotting runner and writes candidates only below
``benchmarks/outputs/starrett_et_al_2014_mixtures_fig3/recomputed``.  It never
modifies the accepted precursor arrays.

Physics provenance
------------------
The thermodynamic states and pair-distribution comparison are Figure 3 of
C. E. Starrett *et al.*, Phys. Rev. E 90, 033110 (2014),
https://doi.org/10.1103/PhysRevE.90.033110.  The orbital average-atom,
pseudoatom/QOZ construction, IS approximation, and Appendix-B density tail
follow C. E. Starrett and D. Saumon, High Energy Density Physics 10, 35--42
(2014), https://doi.org/10.1016/j.hedp.2013.12.001.  The finite-temperature
jellium local-field correction is G. Chabrier, J. Phys. France 51,
1607--1632 (1990), https://doi.org/10.1051/jphys:0199000510150160700.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

# Each average-atom continuum calculation already uses worker processes.
# Prevent BLAS/OpenMP from multiplying that explicit parallelism.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otter import PlasmaWorkflowConfig, solve_plasma_workflow  # noqa: E402
from otter.io.state import (  # noqa: E402
    StateExportOptions,
    build_state_arrays,
)


# User-editable calculation controls.  Three independent thermodynamic states
# times six continuum workers use at most about 18 explicit worker processes
# on a 24-core workstation.  Species are evaluated sequentially within each
# common-mu root, avoiding a third nested process/thread layer.
DENSITIES_G_CC = (2.94, 5.0, 15.0)
TEMPERATURES_KK = (20, 50, 100)
MAX_STATE_WORKERS = 3
CONTINUUM_WORKERS_PER_STATE = 6
SPECIES_PARALLEL_JOBS = 1

AA_N_POINTS = 4096
QOZ_N_POINTS = 4096
MU_E_TOL_HA = 1.0e-4
ROOT_TOL = 1.0e-4
ROOT_MAXFEV = 32
ROOT_BRENT_MAXITER = 24
HNC_TOL = 1.0e-5
HNC_CLOSURE_TRANSFORM_TOL = 1.0e-4
HNC_MAX_ITER = 1000
R_RETAIN_MAX_BOHR = 20.0
K_RETAIN_MAX_BOHR_INV = 20.0

COUNTS = (1.0, 1.36)
EV_PER_K = 8.617333262145e-5
SCHEMA = "otter_starrett_mixtures_fig3_candidate_v1"
BENCHMARK_ID = "starrett_et_al_2014_mixtures_fig3_ch1p36"
OUTPUT_DIR = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "starrett_et_al_2014_mixtures_fig3"
    / "recomputed"
)
REFERENCE_DIR = (
    ROOT
    / "benchmarks"
    / "reference_data"
    / "starrett_et_al_2014_mixtures_fig3"
)


@dataclass(frozen=True, order=True)
class State:
    """One thermodynamic state from Starrett et al. (2014), Figure 3."""

    rho_g_cc: float
    temperature_kk: int

    @property
    def temperature_ev(self) -> float:
        return 1000.0 * float(self.temperature_kk) * EV_PER_K

    @property
    def token(self) -> str:
        rho = f"{self.rho_g_cc:.2f}".replace(".", "p")
        return f"CH1p36_rho{rho}gcc_T{self.temperature_kk}kK_otter"


STATES = tuple(
    State(float(rho), int(temperature))
    for rho in DENSITIES_G_CC
    for temperature in TEMPERATURES_KK
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _producer_metadata() -> dict[str, Any]:
    """Record the exact source state without claiming dirty HEAD is enough."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        status = "git metadata unavailable\n"
        commit = "unknown"
    script = Path(__file__).resolve()
    return {
        "project": "Otter",
        "git_commit": commit,
        "script_relative_path": str(script.relative_to(ROOT)),
        "script_sha256": _sha256(script),
        "worktree_clean_at_generation": not bool(status),
        "git_status_porcelain_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
        "candidate_only": True,
    }


def aa_overrides() -> dict[str, Any]:
    """Return the documented IS-QM Appendix-B electronic controls."""
    return {
        "n_points": AA_N_POINTS,
        "cont_n_jobs": CONTINUUM_WORKERS_PER_STATE,
        "cont_shards": 2 * CONTINUUM_WORKERS_PER_STATE,
        "bound_occ_mode": "fd",
        # The ordinary AA radial domain is the production bound-state domain;
        # the experimental far-away bound-only box is intentionally disabled.
        "bound_rmax_mult": None,
        "bound_zero_tail_refine": False,
        "b3_tail_stage1_mode": "in_scf",
        "b3_tail_stage2_mode": "in_scf",
        "ext_b3_tail_mode": "in_scf",
        "b3_tail_target": "full",
        "b3_tail_model": "full",
        "b3_tail_fit_window_mode": "local",
        "b3_tail_local_fit_width_mult": 0.064,
        "b3_r_cut_mult": 3.0,
        "b3_r_fit_max_mult": 4.0,
        "b3_source_charge_constraint": False,
        "full_b3_use_source_closure": False,
        "ext_b3_use_source_closure": False,
        "ph_kappa": 0.0,
        "ph_kappa_iters": 0,
    }


def configuration(state: State) -> PlasmaWorkflowConfig:
    """Build the strict public Otter workflow for one Figure 3 state."""
    return PlasmaWorkflowConfig(
        elements=["C", "H"],
        counts=list(COUNTS),
        temperature_ev=state.temperature_ev,
        ion_temperature_ev=state.temperature_ev,
        rho_g_cc=state.rho_g_cc,
        electronic_model="qm",
        run_mode="full+ext",
        aa_overrides=aa_overrides(),
        mu_e_tol=MU_E_TOL_HA,
        root_tol=ROOT_TOL,
        root_maxfev=ROOT_MAXFEV,
        root_brent_maxiter=ROOT_BRENT_MAXITER,
        root_threshold_b3_surrogate_mode="a_only_when_full_unresolved",
        allow_unconverged_root=False,
        allow_unconverged_aa=False,
        species_parallel_jobs=SPECIES_PARALLEL_JOBS,
        species_parallel_backend="thread",
        qoz_linear_n_points=QOZ_N_POINTS,
        qoz_pad_factor=2.0,
        qoz_zbar_mode="pseudoatom_partition",
        qoz_renormalize_nscr_to_zbar=True,
        qoz_response_chi0_model="lindhard_fd",
        qoz_response_lfc_model="chabrier1990",
        qoz_high_k_taper_start_frac=0.9,
        hnc_tol=HNC_TOL,
        hnc_closure_transform_tol=HNC_CLOSURE_TRANSFORM_TOL,
        hnc_max_iter=HNC_MAX_ITER,
        hnc_require_converged=True,
        hnc_enforce_nodal_tail_zero=False,
        hnc_s_projection_mode="none",
        show_progress=False,
        show_mu_progress=False,
        verbose=False,
        save_data=False,
    )


def signature(state: State) -> dict[str, Any]:
    """Return every result-affecting benchmark choice in JSON-safe form."""
    return {
        "schema": 1,
        "state": {
            "rho_g_cc": state.rho_g_cc,
            "temperature_kk": state.temperature_kk,
            "temperature_ev": state.temperature_ev,
        },
        "composition": {"elements": ["C", "H"], "counts": list(COUNTS)},
        "structure_model": "IS",
        "electronic_model": "qm",
        "aa_overrides": aa_overrides(),
        "root": {
            "mu_e_tol_ha": MU_E_TOL_HA,
            "root_tol": ROOT_TOL,
            "root_maxfev": ROOT_MAXFEV,
            "root_brent_maxiter": ROOT_BRENT_MAXITER,
            "allow_unconverged_root": False,
            "allow_unconverged_aa": False,
        },
        "qoz": {
            "n_points": QOZ_N_POINTS,
            "pad_factor": 2.0,
            "zbar_mode": "pseudoatom_partition",
            "renormalize_nscr": True,
            "chi0_model": "lindhard_fd",
            "lfc_model": "chabrier1990",
            "high_k_taper_start_frac": 0.9,
        },
        "hnc": {
            "tol": HNC_TOL,
            "closure_transform_tol": HNC_CLOSURE_TRANSFORM_TOL,
            "max_iter": HNC_MAX_ITER,
            "require_converged": True,
            "s_projection_mode": "none",
        },
    }


def _strict_diagnostics(
    workflow: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Reject any common-mu, AA, external-AA, or HNC best-effort result."""
    if str(workflow["electronic"]["kind"]) != "mixture":
        raise RuntimeError("Figure 3 producer requires a C-H mixture payload.")
    electronic = dict(workflow["electronic"]["result"])
    meta = dict(electronic.get("meta", {}))
    if not bool(meta.get("root_success", False)):
        raise RuntimeError("Common-mu root did not converge.")
    if float(meta.get("mu_residual_max_ha", np.inf)) > MU_E_TOL_HA:
        raise RuntimeError("Common-mu residual exceeds the configured tolerance.")
    if not bool(meta.get("final_mu_root_success", False)):
        raise RuntimeError("Final full+external rerun lost common-mu closure.")
    if float(meta.get("final_mu_residual_max_ha", np.inf)) > MU_E_TOL_HA:
        raise RuntimeError("Final common-mu residual exceeds tolerance.")

    species = [dict(entry) for entry in electronic["species"]]
    if [str(entry["element"]) for entry in species] != ["C", "H"]:
        raise RuntimeError("Unexpected species order in mixture result.")
    for entry in species:
        result = dict(entry["result"])
        symbol = str(entry["element"])
        if not bool(result.get("stage2_converged", False)):
            raise RuntimeError(f"{symbol} full AA stage 2 did not converge.")
        if str(result.get("threshold_state_status", "")).lower() == "unresolved":
            raise RuntimeError(f"{symbol} has an unresolved threshold state.")
        ext_status = dict(result.get("ext_status", {}))
        if not bool(ext_status.get("converged", False)):
            raise RuntimeError(f"{symbol} external fixed-mu AA did not converge.")
        full_tail = dict(result.get("n_full_tail_meta", {}))
        ext_tail = dict(result.get("n_ext_tail_meta", {}))
        if not bool(full_tail.get("applied", False)):
            raise RuntimeError(f"{symbol} full-density B3 tail was not applied.")
        if not bool(ext_tail.get("applied", False)):
            raise RuntimeError(f"{symbol} external-density B3 tail was not applied.")

    ion = dict(workflow["ion"])
    if not bool(ion.get("hnc_converged", False)):
        raise RuntimeError("Multicomponent HNC did not reach its full-scale root.")
    if float(ion.get("hnc_output_residual", np.inf)) > HNC_TOL:
        raise RuntimeError("HNC residual exceeds the configured tolerance.")
    if (
        float(ion.get("closure_transform_max_abs", np.inf))
        > HNC_CLOSURE_TRANSFORM_TOL
    ):
        raise RuntimeError("The finite-grid g/S closure mismatch is too large.")
    if float(ion.get("hnc_s_min", -np.inf)) <= 0.0:
        raise RuntimeError("The converged S matrix is not positive definite.")
    return electronic, ion, species


def pack_workflow(
    workflow: dict[str, Any],
    state: State,
    *,
    elapsed_s: float,
    producer: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Convert one strict workflow to a portable, pickle-free candidate."""
    electronic, ion, species = _strict_diagnostics(workflow)
    export = build_state_arrays(
        workflow,
        options=StateExportOptions(
            # build_state_arrays uses an exclusive cutoff.  Advancing by one
            # representable float implements the documented <= 20 window.
            r_max_bohr=np.nextafter(R_RETAIN_MAX_BOHR, np.inf),
            k_max_bohr_inv=np.nextafter(K_RETAIN_MAX_BOHR_INV, np.inf),
            require_converged_hnc=True,
        ),
    )
    species_results = [dict(entry["result"]) for entry in species]
    meta = dict(electronic["meta"])
    charge = dict(ion["charge_fix"])
    payload = {
        **export,
        "benchmark_schema_version": np.asarray(SCHEMA),
        "benchmark_id": np.asarray(BENCHMARK_ID),
        "pair_labels": np.asarray(("CC", "CH", "HH")),
        "rho_g_cc": np.asarray(state.rho_g_cc),
        "temperature_kk": np.asarray(state.temperature_kk),
        "temperature_ev": np.asarray(state.temperature_ev),
        "producer_elapsed_s": np.asarray(float(elapsed_s)),
        "otter_git_commit": np.asarray(str(producer["git_commit"])),
        "producer_script_sha256": np.asarray(
            str(producer["script_sha256"])
        ),
        "producer_worktree_clean": np.asarray(
            bool(producer["worktree_clean_at_generation"])
        ),
        "producer_git_status_sha256": np.asarray(
            str(producer["git_status_porcelain_sha256"])
        ),
        "producer_signature_json": np.asarray(
            json.dumps(signature(state), sort_keys=True, separators=(",", ":"))
        ),
        "r_ws_bohr": np.asarray(
            [float(entry["r_ws_bohr"]) for entry in species]
        ),
        "mu_ha": np.asarray(
            [float(result["mu"]) for result in species_results]
        ),
        "n0_bohr3": np.asarray(
            [float(result["n0"]) for result in species_results]
        ),
        "zbar_aa_ws": np.asarray(
            [float(result["zbar"]) for result in species_results]
        ),
        "zbar_partition": np.asarray(ion["zbar_partition"], dtype=float),
        "zbar_qoz": np.asarray(ion["zbar_qoz"], dtype=float),
        "q_scr_raw": np.asarray(charge["q_scr_raw"], dtype=float),
        "q_scr_used": np.asarray(charge["q_scr_used"], dtype=float),
        "q_scr_scale": np.asarray(charge["scale_factor"], dtype=float),
        "root_success": np.asarray(True),
        "root_residual_ha": np.asarray(
            float(meta["mu_residual_max_ha"])
        ),
        "final_root_residual_ha": np.asarray(
            float(meta["final_mu_residual_max_ha"])
        ),
        "root_nfev": np.asarray(int(meta.get("root_nfev", 0))),
        "root_method": np.asarray(str(meta.get("root_method", "unknown"))),
        "hnc_output_residual": np.asarray(
            float(ion["hnc_output_residual"])
        ),
        "hnc_closure_mismatch": np.asarray(
            float(ion["closure_transform_max_abs"])
        ),
        "hnc_s_min": np.asarray(float(ion["hnc_s_min"])),
    }
    for key, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Object arrays are forbidden: {key}.")
        if array.dtype.kind in "fiu" and not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite candidate data: {key}.")
    return payload


def _solve_one(
    state: State,
    producer: dict[str, Any],
) -> tuple[State, dict[str, np.ndarray]]:
    started = time.perf_counter()
    workflow = solve_plasma_workflow(configuration(state))
    payload = pack_workflow(
        workflow,
        state,
        elapsed_s=time.perf_counter() - started,
        producer=producer,
    )
    return state, payload


def _reference_path(state: State) -> Path:
    rho = f"{state.rho_g_cc:.2f}".replace(".", "p")
    return REFERENCE_DIR / (
        f"fig3_gab_rho{rho}gcc_T{state.temperature_kk}kK.csv"
    )


def _write_manifest(
    output_dir: Path,
    *,
    producer: dict[str, Any],
    records: list[dict[str, Any]],
) -> Path:
    manifest = {
        "schema_version": "otter_starrett_fig3_recompute_manifest_v1",
        "benchmark_id": BENCHMARK_ID,
        "status": "candidate_not_accepted",
        "title": "Otter CH1.36 recomputation for Starrett et al. (2014), Figure 3",
        "producer": producer,
        "configuration": signature(STATES[0])
        | {"state": "see individual state records"},
        "publication": {
            "authors": [
                "C. E. Starrett",
                "D. Saumon",
                "J. Daligault",
                "S. Hamel",
            ],
            "journal": "Physical Review E",
            "volume": "90",
            "pages": "033110",
            "year": 2014,
            "figure": "3",
            "doi": "10.1103/PhysRevE.90.033110",
        },
        "retained_window": {
            "r_max_bohr_inclusive": R_RETAIN_MAX_BOHR,
            "k_max_bohr_inv_inclusive": K_RETAIN_MAX_BOHR_INV,
        },
        "method_references": [
            {
                "role": "IS_average_atom_pseudoatom_QOZ_HNC_and_B3_tail",
                "doi": "10.1016/j.hedp.2013.12.001",
            },
            {
                "role": "finite_temperature_jellium_LFC",
                "doi": "10.1051/jphys:0199000510150160700",
            },
        ],
        "states": records,
    }
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path


def _plot_candidate(manifest_path: Path) -> None:
    """Use the read-only benchmark renderer for the newly staged candidate."""
    runner_path = (
        ROOT
        / "benchmarks"
        / "runners"
        / "plot_starrett_et_al_2014_mixtures_fig3.py"
    )
    spec = importlib.util.spec_from_file_location(
        "otter_starrett_fig3_read_only_runner",
        runner_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import read-only runner: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    manifest = module.load_manifest(manifest_path=manifest_path)
    metrics, loaded = module.evaluate_states(
        manifest,
        data_dir=manifest_path.parent,
    )
    module.plot_states(
        loaded,
        figure_path=manifest_path.parent / "fig3_ch1p36_otter_overlay.png",
        model_label="Otter recomputation",
    )
    module.write_metrics(
        metrics,
        metrics_path=manifest_path.parent / "metrics.csv",
    )


def regenerate(
    *,
    output_dir: Path = OUTPUT_DIR,
    states: tuple[State, ...] = STATES,
    max_state_workers: int = MAX_STATE_WORKERS,
    make_plot: bool = True,
) -> Path:
    """Compute candidates, write their manifest, and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    producer = _producer_metadata()
    workers = max(1, min(int(max_state_workers), len(states)))
    solved: dict[State, dict[str, np.ndarray]] = {}
    if workers == 1:
        for state in states:
            solved_state, payload = _solve_one(state, producer)
            solved[solved_state] = payload
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_solve_one, state, producer): state
                for state in states
            }
            for future in as_completed(futures):
                solved_state, payload = future.result()
                solved[solved_state] = payload
                print(f"[computed] {solved_state.token}", flush=True)

    records: list[dict[str, Any]] = []
    for state in sorted(states):
        payload = solved[state]
        path = output_dir / f"{state.token}.npz"
        np.savez_compressed(path, **payload)
        reference = _reference_path(state)
        if not reference.is_file():
            raise FileNotFoundError(reference)
        records.append(
            {
                "rho_g_cc": state.rho_g_cc,
                "temperature_kk": state.temperature_kk,
                "temperature_ev": state.temperature_ev,
                "result_file": path.name,
                "result_sha256": _sha256(path),
                "reference_file": os.path.relpath(reference, output_dir),
                "reference_sha256": _sha256(reference),
                "root_residual_ha": float(payload["root_residual_ha"]),
                "final_root_residual_ha": float(
                    payload["final_root_residual_ha"]
                ),
                "hnc_output_residual": float(
                    payload["hnc_output_residual"]
                ),
                "hnc_closure_mismatch": float(
                    payload["hnc_closure_mismatch"]
                ),
            }
        )
        print(f"[saved] {path.relative_to(ROOT)}", flush=True)

    manifest_path = _write_manifest(
        output_dir,
        producer=producer,
        records=records,
    )
    if make_plot:
        _plot_candidate(manifest_path)
    print(f"[saved] {manifest_path.relative_to(ROOT)}", flush=True)
    return manifest_path


def main() -> None:
    print(
        "Otter Figure 3 recomputation: "
        f"{MAX_STATE_WORKERS} state workers x "
        f"{CONTINUUM_WORKERS_PER_STATE} continuum workers; "
        "accepted data will not be modified."
    )
    regenerate()


if __name__ == "__main__":
    # Required by multiprocessing on spawn-based platforms.
    main()
