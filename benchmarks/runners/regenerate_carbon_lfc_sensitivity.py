"""Regenerate the Otter carbon LFC-sensitivity comparison at 5 g/cc.

For each temperature the expensive full/external KS average atom is solved
once.  The converged electronic payload is then reused for all five local
field corrections, so differences in :math:`V_{ii}`, :math:`g_{ii}`, and
:math:`S_{ii}` isolate only the downstream jellium-response model.

Candidate files are written below
``benchmarks/outputs/carbon_lfc_sensitivity/recomputed``.  The accepted
checksummed data package is never overwritten by this program.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from otter import (  # noqa: E402
    PlasmaWorkflowConfig,
    continue_plasma_workflow_from_electronic_result,
    solve_plasma_workflow,
)


RHO_G_CC = 5.0
TEMPERATURES_EV = (2.0, 100.0)
LFC_MODELS = (
    "none",
    "hubbard",
    "utsumiichimaru",
    "chabrier1990",
    "gregori2007",
)
REFERENCE_LFC = "chabrier1990"
MAX_STATE_WORKERS = 2
CONTINUUM_WORKERS_PER_STATE = 6
AA_N_POINTS = 4096
QOZ_N_POINTS = 8192
HNC_TOL = 1.0e-5
HNC_CLOSURE_TRANSFORM_TOL = 1.0e-4
R_RETAIN_MAX_BOHR = 20.0
K_RETAIN_MAX_BOHR_INV = 20.0
OUTPUT_DIR = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "carbon_lfc_sensitivity"
    / "recomputed"
)
SCHEMA = "otter_carbon_lfc_sensitivity_state_v2"


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _git_status_porcelain() -> str:
    """Return the exact dirty-tree description used by the producer."""
    return subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _producer_metadata() -> dict[str, Any]:
    """Record enough provenance to avoid treating dirty HEAD as reproducible."""
    status = _git_status_porcelain()
    script = Path(__file__).resolve()
    script_sha256 = _sha256(script)
    return {
        "project": "Otter",
        "git_commit": _git_commit(),
        "script_relative_path": str(script.relative_to(ROOT)),
        # ``script_sha256`` is retained for compatibility with the v2 state
        # archives.  The explicit names make later documentation-only edits
        # auditable without rewriting accepted numerical data.
        "script_sha256": script_sha256,
        "script_sha256_at_generation": script_sha256,
        "script_sha256_current": script_sha256,
        "worktree_clean_at_generation": not bool(status),
        "git_status_porcelain_sha256": _text_sha256(status),
        "note": (
            "git_commit identifies HEAD only. When "
            "worktree_clean_at_generation is false, script_sha256 and the "
            "status hash disclose that uncommitted source participated."
        ),
    }


def _configuration(
    temperature_ev: float,
    *,
    ion_temperature_ev: float | None,
    lfc_model: str,
) -> PlasmaWorkflowConfig:
    return PlasmaWorkflowConfig(
        elements=["C"],
        temperature_ev=float(temperature_ev),
        ion_temperature_ev=(
            None if ion_temperature_ev is None else float(ion_temperature_ev)
        ),
        rho_g_cc=float(RHO_G_CC),
        electronic_model="qm",
        aa_overrides={
            "n_points": int(AA_N_POINTS),
            "cont_n_jobs": int(CONTINUUM_WORKERS_PER_STATE),
            "cont_shards": int(2 * CONTINUUM_WORKERS_PER_STATE),
            "bound_occ_mode": "fd",
            "b3_tail_model": "full",
            "bound_zero_tail_refine": True,
            "bound_zero_tail_max_binding_ha": 1.0e-2,
            "bound_zero_tail_scan_points": 48,
            "bound_zero_tail_edge_rel_tol": 0.1,
        },
        qoz_linear_n_points=int(QOZ_N_POINTS),
        qoz_pad_factor=2.0,
        qoz_zbar_mode="pseudoatom_partition",
        qoz_renormalize_nscr_to_zbar=True,
        qoz_response_chi0_model="lindhard_fd",
        qoz_response_lfc_model=str(lfc_model),
        hnc_tol=HNC_TOL,
        hnc_closure_transform_tol=HNC_CLOSURE_TRANSFORM_TOL,
        hnc_max_iter=1000,
        hnc_require_converged=True,
        show_progress=False,
    )


def _trim(coordinate: np.ndarray, values: np.ndarray, maximum: float) -> tuple[np.ndarray, np.ndarray]:
    coordinate = np.asarray(coordinate, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = coordinate <= float(maximum)
    return coordinate[mask], values[..., mask]


def _solve_state(temperature_ev: float) -> dict[str, Any]:
    """Solve one electronic state once and all LFC paths from that payload."""
    producer = _producer_metadata()
    electronic_started = time.perf_counter()
    electronic_workflow = solve_plasma_workflow(
        _configuration(
            temperature_ev,
            ion_temperature_ev=None,
            lfc_model=REFERENCE_LFC,
        )
    )
    electronic_elapsed_s = time.perf_counter() - electronic_started
    electronic_kind = str(electronic_workflow["electronic"]["kind"])
    electronic = dict(electronic_workflow["electronic"]["result"])
    ext_status = dict(electronic.get("ext_status", {}))
    threshold_status = str(
        electronic.get("threshold_state_status", "none")
    ).strip().lower()
    if not bool(
        electronic.get(
            "stage2_converged",
            electronic.get("converged", False),
        )
    ):
        raise RuntimeError(
            "Strict carbon producer requires a converged AA stage 2."
        )
    if not bool(ext_status.get("converged", False)):
        raise RuntimeError(
            "Strict carbon producer requires a converged external AA stage."
        )
    if threshold_status == "unresolved":
        raise RuntimeError(
            "Strict carbon producer refuses an unresolved threshold-state "
            "representation."
        )

    ion_results: list[dict[str, Any]] = []
    ion_elapsed_s: list[float] = []
    for model in LFC_MODELS:
        started = time.perf_counter()
        workflow = continue_plasma_workflow_from_electronic_result(
            _configuration(
                temperature_ev,
                ion_temperature_ev=temperature_ev,
                lfc_model=model,
            ),
            electronic_kind=electronic_kind,
            electronic_result=electronic,
        )
        ion_elapsed_s.append(time.perf_counter() - started)
        ion = dict(workflow["ion"])
        if float(ion["hnc_output_residual"]) > HNC_TOL:
            raise RuntimeError(
                f"{model} HNC residual exceeds the configured tolerance."
            )
        if (
            float(ion["closure_transform_max_abs"])
            > HNC_CLOSURE_TRANSFORM_TOL
        ):
            raise RuntimeError(
                f"{model} closure-transform mismatch exceeds tolerance."
            )
        ion_results.append(ion)

    r_e, n_full = _trim(
        electronic["r"],
        electronic["n_full"],
        R_RETAIN_MAX_BOHR,
    )
    electronic_arrays: dict[str, np.ndarray] = {
        "electronic_r_bohr": r_e,
        "n_full_bohr3": n_full,
    }
    for source, target in (
        ("n_bound", "n_bound_bohr3"),
        ("n_cont", "n_cont_bohr3"),
        ("n_ext", "n_ext_bohr3"),
        ("n_ion", "n_ion_bohr3"),
        ("n_scr", "n_scr_bohr3"),
        ("v_full", "v_full_ha"),
        ("v_xc", "v_xc_ha"),
    ):
        _, values = _trim(
            electronic["r"],
            electronic[source],
            R_RETAIN_MAX_BOHR,
        )
        electronic_arrays[target] = values

    reference_ion = ion_results[0]
    zero_tail_meta = dict(electronic.get("zero_tail_bound_meta", {}))
    zero_tail_states = list(zero_tail_meta.get("states", ()))
    exterior_matching_mode = (
        str(zero_tail_states[0].get("matching_mode", "none"))
        if zero_tail_states
        else str(zero_tail_meta.get("matching_mode", "none"))
    )
    r, _ = _trim(reference_ion["r"], reference_ion["gii_r"], R_RETAIN_MAX_BOHR)
    k, _ = _trim(reference_ion["k"], reference_ion["sii_k"], K_RETAIN_MAX_BOHR_INV)

    def on_r(ion: dict[str, Any], key: str) -> np.ndarray:
        source_r = np.asarray(ion["r"], dtype=float)
        source = np.asarray(ion[key], dtype=float)
        return np.interp(r, source_r, source)

    def on_k(ion: dict[str, Any], key: str) -> np.ndarray:
        source_k = np.asarray(ion["k"], dtype=float)
        source = np.asarray(ion[key], dtype=float)
        return np.interp(k, source_k, source)

    signature = {
        "state": {
            "element": "C",
            "rho_g_cc": RHO_G_CC,
            "temperature_ev": float(temperature_ev),
            "ion_temperature_ev": float(temperature_ev),
        },
        "structure_model": "IS",
        "lfc_models": list(LFC_MODELS),
        "reference_lfc": REFERENCE_LFC,
        "aa_n_points": AA_N_POINTS,
        "continuum_workers_per_state": CONTINUUM_WORKERS_PER_STATE,
        "bound_occ_mode": "fd",
        "bound_rmax_mult": None,
        "bound_zero_tail_refine": True,
        "bound_zero_tail_max_binding_ha": 1.0e-2,
        "bound_zero_tail_scan_points": 48,
        "bound_zero_tail_edge_rel_tol": 0.1,
        "b3_tail_model": "full",
        "qoz_n_points_before_padding": QOZ_N_POINTS,
        "chi0_model": "lindhard_fd",
        "hnc_tolerance": HNC_TOL,
        "hnc_transform_closure_tolerance": HNC_CLOSURE_TRANSFORM_TOL,
        "hnc_max_iterations": 1000,
        "threshold_reliability_edge": (
            "same bound/continuum energy_cut used by n_bound"
        ),
    }
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA),
        "benchmark_id": np.asarray("carbon_lfc_sensitivity"),
        "element_symbol": np.asarray("C"),
        "rho_g_cc": np.asarray(RHO_G_CC),
        "temperature_ev": np.asarray(float(temperature_ev)),
        "model_labels": np.asarray(LFC_MODELS),
        "reference_model": np.asarray(REFERENCE_LFC),
        "otter_git_commit": np.asarray(producer["git_commit"]),
        "producer_script_sha256": np.asarray(producer["script_sha256"]),
        "producer_worktree_clean": np.asarray(
            producer["worktree_clean_at_generation"]
        ),
        "producer_git_status_sha256": np.asarray(
            producer["git_status_porcelain_sha256"]
        ),
        "producer_signature_json": np.asarray(
            json.dumps(signature, sort_keys=True, separators=(",", ":"))
        ),
        "electronic_elapsed_s": np.asarray(float(electronic_elapsed_s)),
        "ion_elapsed_s": np.asarray(ion_elapsed_s),
        "r_bohr": r,
        "k_bohr_inv": k,
        "gii_r": np.asarray([on_r(ion, "gii_r") for ion in ion_results]),
        "sii_k": np.asarray([on_k(ion, "sii_k") for ion in ion_results]),
        "vii_r_ha": np.asarray([on_r(ion, "vii_r") for ion in ion_results]),
        "vii_k_ha_bohr3": np.asarray(
            [on_k(ion, "vii_k") for ion in ion_results]
        ),
        "chi_ee_k": np.asarray(
            [on_k(ion, "chi_ee_k") for ion in ion_results]
        ),
        "gee_k": np.asarray([on_k(ion, "gee_k") for ion in ion_results]),
        "n_scr_r_bohr3": on_r(reference_ion, "n_scr_r"),
        "n_scr_k": on_k(reference_ion, "n_scr_k"),
        "chi0_k": on_k(reference_ion, "chi0_k"),
        "zbar_qoz": np.asarray(
            [float(ion["zbar_qoz"]) for ion in ion_results]
        ),
        "n_i_bohr3": np.asarray(
            [float(ion["n_i"]) for ion in ion_results]
        ),
        "hnc_output_residual": np.asarray(
            [float(ion["hnc_output_residual"]) for ion in ion_results]
        ),
        "closure_transform_max_abs": np.asarray(
            [float(ion["closure_transform_max_abs"]) for ion in ion_results]
        ),
        "aa_stage2_converged": np.asarray(
            bool(
                electronic.get(
                    "stage2_converged",
                    electronic.get("converged", False),
                )
            )
        ),
        "aa_stage2_iters": np.asarray(
            int(
                electronic.get(
                    "stage2_iters",
                    len(electronic.get("history", [])),
                )
            )
        ),
        "aa_stage2_final_error": np.asarray(
            float(electronic["history"][-1]["err"])
        ),
        "aa_ext_converged": np.asarray(
            bool(ext_status.get("converged", False))
        ),
        "aa_ext_iters": np.asarray(int(ext_status.get("iters", 0))),
        "aa_ext_final_error": np.asarray(float(ext_status["err"])),
        "threshold_state_status": np.asarray(
            str(electronic.get("threshold_state_status", "none"))
        ),
        "threshold_state_representation": np.asarray(
            str(electronic.get("threshold_state_representation", "none"))
        ),
        "threshold_state_localization": np.asarray(
            str(electronic.get("threshold_state_localization", "none"))
        ),
        "bound_exterior_matching_mode": np.asarray(exterior_matching_mode),
        "bound_exterior_matching_applied": np.asarray(
            bool(zero_tail_meta.get("applied", False))
        ),
        "threshold_energy_cut_ha": np.asarray(
            float(electronic["meta"]["bound_energy_cut_ha"])
        ),
        "shallowest_bound_energy_ha": np.asarray(
            float(electronic["shallowest_bound_energy_ha"])
        ),
        "mu_ha": np.asarray(float(electronic["mu"])),
        "n0_bohr3": np.asarray(float(electronic["n0"])),
        "zbar_aa_ws": np.asarray(float(electronic["zbar"])),
        "zbar_partition": np.asarray(
            float(reference_ion["zbar_partition"])
        ),
        "r_ws_bohr": np.asarray(float(electronic["r_ws"])),
        "q_scr_raw_integral": np.asarray(
            float(
                4.0
                * np.pi
                * np.trapezoid(
                    np.asarray(electronic["n_scr"], dtype=float)
                    * np.asarray(electronic["r"], dtype=float) ** 2,
                    np.asarray(electronic["r"], dtype=float),
                )
            )
        ),
        **electronic_arrays,
    }
    for key, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Object arrays are forbidden: {key}.")
        if array.dtype.kind in "fiu" and not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite data in {key}.")
    return payload


def _filename(temperature_ev: float) -> str:
    token = f"{float(temperature_ev):g}".replace(".", "p")
    return f"C_rho5gcc_T{token}eV_lfc.npz"


def regenerate(*, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Compute both states and write candidate data plus a checksum manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[float, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=MAX_STATE_WORKERS) as pool:
        futures = {
            pool.submit(_solve_state, temperature_ev): temperature_ev
            for temperature_ev in TEMPERATURES_EV
        }
        for future in as_completed(futures):
            temperature_ev = futures[future]
            payloads[temperature_ev] = future.result()
            print(
                f"[done] C rho={RHO_G_CC:g} g/cc, T={temperature_ev:g} eV",
                flush=True,
            )

    paths: list[Path] = []
    state_records: list[dict[str, Any]] = []
    for temperature_ev in TEMPERATURES_EV:
        path = output_dir / _filename(temperature_ev)
        np.savez_compressed(path, **payloads[temperature_ev])
        paths.append(path)
        state_records.append(
            {
                "rho_g_cc": RHO_G_CC,
                "temperature_ev": float(temperature_ev),
                "data_file": path.name,
                "data_sha256": _sha256(path),
            }
        )
        print(f"[saved] {path.relative_to(ROOT)}", flush=True)

    manifest = {
        "schema_version": "otter_benchmark_manifest_v2",
        "benchmark_id": "carbon_lfc_sensitivity",
        "status": "candidate_not_accepted",
        "title": "Carbon finite-temperature LFC sensitivity at 5 g cm^-3",
        "producer": _producer_metadata(),
        "data_rights": {
            "origin_type": "project_generated_numerical_output",
            "third_party_reference_data_included": False,
            "method_citations_only": True,
        },
        "method_references": [
            {
                "citation_key": "StarrettSaumon2014",
                "doi": "10.1016/j.hedp.2013.12.001",
                "scope": "pseudoatom screening and effective ion interaction",
            },
            {
                "citation_key": "Hubbard1958",
                "doi": "10.1098/rspa.1958.0003",
                "scope": "Hubbard static local-field correction",
            },
            {
                "citation_key": "UtsumiIchimaru1982",
                "doi": "10.1103/PhysRevA.26.603",
                "scope": "Utsumi--Ichimaru static local-field correction",
            },
            {
                "citation_key": "Chabrier1990",
                "doi": "10.1051/jphys:0199000510150160700",
                "scope": "finite-temperature jellium local-field correction",
            },
            {
                "citation_key": "GeldartVosko1966",
                "doi": "10.1139/p66-174",
                "scope": "low-temperature branch of the Gregori interpolation",
            },
            {
                "citation_key": "GregoriEtAl2007",
                "doi": "10.1016/j.hedp.2007.02.006",
                "scope": "finite-temperature local-field interpolation",
            },
            {
                "citation_key": "LutgertEtAl2026",
                "doi": "10.1016/j.cpc.2026.110173",
                "scope": (
                    "software provenance for the attributed "
                    "Gregori/Geldart--Vosko implementation adapted from "
                    "JaXRTS"
                ),
            },
        ],
        "units": {
            "radius": "Bohr",
            "wavenumber": "Bohr^-1",
            "electron_density": "Bohr^-3",
            "electron_response": "Bohr^-3 Hartree^-1",
            "screening_cloud_k": "electron number",
            "effective_pair_potential_r": "Hartree",
            "effective_pair_potential_k": "Hartree Bohr^3",
            "gii": "dimensionless",
            "sii": "dimensionless",
        },
        "configuration": {
            "rho_g_cc": RHO_G_CC,
            "temperatures_ev": list(TEMPERATURES_EV),
            "models": list(LFC_MODELS),
            "reference_model": REFERENCE_LFC,
            "structure_model": "IS",
            "aa_n_points": AA_N_POINTS,
            "bound_occ_mode": "fd",
            "bound_rmax_mult": None,
            "bound_zero_tail_refine": True,
            "bound_zero_tail_matching_mode": "direct_physical_boundary",
            "bound_zero_tail_max_binding_ha": 1.0e-2,
            "bound_zero_tail_scan_points": 48,
            "bound_zero_tail_edge_rel_tol": 0.1,
            "b3_tail_model": "full",
            "qoz_n_points_before_padding": QOZ_N_POINTS,
            "chi0_model": "lindhard_fd",
            "hnc_tolerance": HNC_TOL,
            "hnc_transform_closure_tolerance": (
                HNC_CLOSURE_TRANSFORM_TOL
            ),
            "strict_electronic_convergence_required": True,
            "allow_unconverged_aa": False,
            "resolution_rationale": (
                "4096 radial points follow Starrett--Saumon (2014), Appendix "
                "B, after a 1024/2048/4096 threshold audit at 100 eV. A "
                "near-threshold state is checked by matching to a "
                "zero-potential exterior at the common outer SCF boundary; "
                "this numerical refinement is motivated by Starrett et al. "
                "(2019), but is not asserted to reproduce their ion-sphere "
                "boundary implementation. No separate bound-only radial "
                "extension is used."
            ),
        },
        "states": state_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"[saved] {manifest_path.relative_to(ROOT)}", flush=True)
    return paths


if __name__ == "__main__":
    regenerate()
