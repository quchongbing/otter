"""Regenerate the Otter aluminium KS-DFT/Thomas--Fermi comparison.

The scientific model follows Starrett and Saumon, HEDP 10, 35--42 (2014):
the orbital Kohn--Sham and finite-temperature Thomas--Fermi electronic
backends feed the same pseudoatom/QOZ/HNC construction.  Chabrier's
finite-temperature jellium LFC is used in both paths.

The four thermodynamic states are independent and may be computed in
parallel.  Accepted reference data are never overwritten by this script;
candidate archives and their manifest are written below
``benchmarks/outputs/al_qm_tf/recomputed``.
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

from otter import PlasmaWorkflowConfig, solve_plasma_workflow  # noqa: E402


# User-editable calculation controls.  Four simultaneous states with four
# continuum workers each use at most 16 of a 24-core workstation.
TEMPERATURES_EV = (1.0, 15.0, 50.0, 100.0)
RHO_G_CC = 8.1
MODELS = ("qm", "tf")
MAX_STATE_WORKERS = 4
CONTINUUM_WORKERS_PER_STATE = 4
AA_N_POINTS = 1024
QOZ_N_POINTS = 4096
HNC_TOL = 1.0e-6
HNC_CLOSURE_TOL = 1.0e-3
R_RETAIN_MAX_BOHR = 20.0
K_RETAIN_MAX_BOHR_INV = 20.0
OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / "al_qm_tf" / "recomputed"
SCHEMA = "otter_al_qm_tf_state_v2"


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _git_status_porcelain() -> str:
    """Return the source-tree status used for honest provenance metadata."""
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
    """Describe the exact producer state without claiming a clean checkout."""
    status = _git_status_porcelain()
    return {
        "project": "Otter",
        "git_commit": _git_commit(),
        "script_relative_path": str(
            Path(__file__).resolve().relative_to(ROOT)
        ),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "worktree_clean_at_generation": not bool(status),
        "git_status_porcelain_sha256": _text_sha256(status),
        "note": (
            "The commit identifies HEAD only. If "
            "worktree_clean_at_generation is false, the script checksum "
            "and status hash document that uncommitted source was involved; "
            "the commit alone is not a reproducible producer snapshot."
        ),
    }


def _configuration(temperature_ev: float, model: str) -> PlasmaWorkflowConfig:
    """Return one fully specified, reproducible Otter workflow."""
    return PlasmaWorkflowConfig(
        elements=["Al"],
        temperature_ev=float(temperature_ev),
        ion_temperature_ev=float(temperature_ev),
        rho_g_cc=float(RHO_G_CC),
        electronic_model=str(model),
        aa_overrides={
            "n_points": int(AA_N_POINTS),
            "cont_n_jobs": int(CONTINUUM_WORKERS_PER_STATE),
            "cont_shards": int(2 * CONTINUUM_WORKERS_PER_STATE),
            "bound_occ_mode": "fd",
            # Starrett--Saumon (2014), Appendix B, uses one common AA
            # domain.  Do not introduce a separate bound-only zero box.
            "bound_rmax_mult": None,
            "bound_zero_tail_refine": False,
            "b3_tail_model": "full",
        },
        qoz_linear_n_points=int(QOZ_N_POINTS),
        qoz_pad_factor=2.0,
        qoz_zbar_mode="pseudoatom_partition",
        qoz_renormalize_nscr_to_zbar=True,
        qoz_response_chi0_model="lindhard_fd",
        qoz_response_lfc_model="chabrier1990",
        hnc_tol=HNC_TOL,
        hnc_closure_transform_tol=HNC_CLOSURE_TOL,
        hnc_max_iter=1000,
        hnc_require_converged=True,
        show_progress=False,
    )


def _trim(
    coordinate: np.ndarray,
    values: np.ndarray,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate = np.asarray(coordinate, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = coordinate <= float(maximum)
    return coordinate[mask], values[..., mask]


def _validate_production_result(
    workflow: dict[str, Any],
    *,
    temperature_ev: float,
    model: str,
) -> dict[str, Any]:
    """Recheck every production convergence gate before archiving a state.

    ``solve_plasma_workflow`` already enforces these conditions at the
    electronic-to-QOZ boundary and in the HNC solver.  Rechecking them here is
    intentional: the generated NPZ then carries an auditable record of the
    exact AA, external-AA, pressure-ionization-threshold, and HNC conditions
    that were accepted.
    """
    label = f"Al {float(temperature_ev):g} eV {str(model)}"
    electronic_container = workflow.get("electronic")
    if not isinstance(electronic_container, dict):
        raise RuntimeError(f"{label}: missing electronic result container.")
    electronic = electronic_container.get("result")
    if not isinstance(electronic, dict):
        raise RuntimeError(f"{label}: missing electronic result.")
    ion = workflow.get("ion")
    if not isinstance(ion, dict):
        raise RuntimeError(f"{label}: missing QOZ/HNC result.")

    failures: list[str] = []
    if electronic.get("stage2_converged") is not True:
        failures.append("AA stage-2 did not converge")

    ext_status = electronic.get("ext_status")
    if not isinstance(ext_status, dict):
        failures.append("external-AA status is missing")
        ext_converged = False
    else:
        # The full+external API records convergence, iterations, and the final
        # residual; it does not expose a separate ``enabled`` flag.  Presence
        # of this status mapping is itself evidence that the external solve
        # was run.  Do not invent a metadata requirement that the solver does
        # not provide.
        ext_converged = bool(ext_status.get("converged") is True)
        if not ext_converged:
            failures.append("external-AA solve was unconverged")

    threshold_status = str(
        electronic.get("threshold_state_status", "")
    ).strip().lower()
    allowed_threshold_statuses = (
        {"not_applicable_tf"}
        if str(model).strip().lower() == "tf"
        else {"none", "resolved", "marginal"}
    )
    if threshold_status not in allowed_threshold_statuses:
        failures.append(
            "pressure-ionization threshold status is "
            f"{threshold_status or 'missing'}"
        )

    hnc_converged = ion.get("hnc_converged") is True
    if not hnc_converged:
        failures.append("OZ/HNC did not report a physical fixed point")
    try:
        hnc_output_residual = float(ion["hnc_output_residual"])
    except (KeyError, TypeError, ValueError):
        hnc_output_residual = np.inf
    if (
        not np.isfinite(hnc_output_residual)
        or hnc_output_residual > HNC_TOL
    ):
        failures.append(
            "HNC output residual "
            f"{hnc_output_residual:.3e} exceeds {HNC_TOL:.3e}"
        )
    try:
        closure_mismatch = float(ion["closure_transform_max_abs"])
    except (KeyError, TypeError, ValueError):
        closure_mismatch = np.inf
    if (
        not np.isfinite(closure_mismatch)
        or closure_mismatch > HNC_CLOSURE_TOL
    ):
        failures.append(
            "real/reciprocal-space closure mismatch "
            f"{closure_mismatch:.3e} exceeds {HNC_CLOSURE_TOL:.3e}"
        )

    if failures:
        raise RuntimeError(f"{label}: " + "; ".join(failures) + ".")
    return {
        "aa_stage2_converged": True,
        "aa_ext_converged": bool(ext_converged),
        "threshold_state_status": threshold_status,
        "hnc_converged": bool(hnc_converged),
        "hnc_output_residual": float(hnc_output_residual),
        "closure_transform_max_abs": float(closure_mismatch),
    }


def _solve_one(temperature_ev: float, model: str) -> dict[str, Any]:
    started = time.perf_counter()
    workflow = solve_plasma_workflow(_configuration(temperature_ev, model))
    elapsed_s = time.perf_counter() - started
    convergence = _validate_production_result(
        workflow,
        temperature_ev=temperature_ev,
        model=model,
    )
    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"])
    r_e, n_full = _trim(
        electronic["r"],
        electronic["n_full"],
        R_RETAIN_MAX_BOHR,
    )
    _, n_ext = _trim(electronic["r"], electronic["n_ext"], R_RETAIN_MAX_BOHR)
    _, n_pa = _trim(electronic["r"], electronic["n_pa"], R_RETAIN_MAX_BOHR)
    _, n_bound = _trim(
        electronic["r"],
        electronic["n_bound"],
        R_RETAIN_MAX_BOHR,
    )
    _, n_cont = _trim(electronic["r"], electronic["n_cont"], R_RETAIN_MAX_BOHR)
    _, n_ion = _trim(electronic["r"], electronic["n_ion"], R_RETAIN_MAX_BOHR)
    _, n_scr = _trim(electronic["r"], electronic["n_scr"], R_RETAIN_MAX_BOHR)
    _, v_full = _trim(electronic["r"], electronic["v_full"], R_RETAIN_MAX_BOHR)
    r_i, gii = _trim(ion["r"], ion["gii_r"], R_RETAIN_MAX_BOHR)
    k, sii = _trim(ion["k"], ion["sii_k"], K_RETAIN_MAX_BOHR_INV)
    _, vii_k = _trim(ion["k"], ion["vii_k"], K_RETAIN_MAX_BOHR_INV)
    return {
        "model": str(model),
        "temperature_ev": float(temperature_ev),
        "elapsed_s": float(elapsed_s),
        "r_e": r_e,
        "n_full": n_full,
        "n_ext": n_ext,
        "n_pa": n_pa,
        "n_bound": n_bound,
        "n_cont": n_cont,
        "n_ion": n_ion,
        "n_scr": n_scr,
        "v_full": v_full,
        "r_i": r_i,
        "k": k,
        "gii": gii,
        "sii": sii,
        "vii_k": vii_k,
        "n0": float(electronic["n0"]),
        "zbar_aa": float(electronic["zbar"]),
        "zbar_partition": float(ion["zbar_partition"]),
        "mu": float(electronic["mu"]),
        "r_ws": float(electronic["r_ws"]),
        "q_scr": float(
            4.0
            * np.pi
            * np.trapezoid(
                np.asarray(electronic["n_scr"], dtype=float)
                * np.asarray(electronic["r"], dtype=float) ** 2,
                np.asarray(electronic["r"], dtype=float),
            )
        ),
        **convergence,
    }


def _interpolate(
    reference: np.ndarray,
    source: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(
        np.asarray(reference, dtype=float),
        np.asarray(source, dtype=float),
        np.asarray(values, dtype=float),
    )


def _combined_payload(
    temperature_ev: float,
    by_model: dict[str, dict[str, Any]],
    producer: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Assemble one pickle-free two-model state on common ionic grids."""
    reference = by_model["qm"]
    r_ion = np.asarray(reference["r_i"], dtype=float)
    k = np.asarray(reference["k"], dtype=float)
    gii = []
    sii = []
    vii_k = []
    for model in MODELS:
        state = by_model[model]
        gii.append(_interpolate(r_ion, state["r_i"], state["gii"]))
        sii.append(_interpolate(k, state["k"], state["sii"]))
        vii_k.append(_interpolate(k, state["k"], state["vii_k"]))

    signature = {
        "state": {
            "element": "Al",
            "rho_g_cc": RHO_G_CC,
            "temperature_ev": float(temperature_ev),
            "ion_temperature_ev": float(temperature_ev),
        },
        "electronic_models": list(MODELS),
        "structure_model": "IS",
        "aa_n_points": AA_N_POINTS,
        "continuum_workers_per_state": CONTINUUM_WORKERS_PER_STATE,
        "bound_occ_mode": "fd",
        "bound_rmax_mult": None,
        "bound_zero_tail_refine": False,
        "b3_tail_model": "full",
        "qoz_n_points_before_padding": QOZ_N_POINTS,
        "lfc_model": "chabrier1990",
        "hnc_tolerance": HNC_TOL,
        "hnc_transform_closure_tolerance": HNC_CLOSURE_TOL,
        "hnc_max_iterations": 1000,
    }
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA),
        "benchmark_id": np.asarray("al_qm_tf"),
        "element_symbol": np.asarray("Al"),
        "rho_g_cc": np.asarray(RHO_G_CC),
        "temperature_ev": np.asarray(float(temperature_ev)),
        "model_labels": np.asarray(MODELS),
        "lfc_model": np.asarray("chabrier1990"),
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
        "r_ion_bohr": r_ion,
        "k_bohr_inv": k,
        "gii_r": np.asarray(gii),
        "sii_k": np.asarray(sii),
        "vii_k_ha_bohr3": np.asarray(vii_k),
        "zbar_aa_ws": np.asarray([by_model[m]["zbar_aa"] for m in MODELS]),
        "zbar_partition": np.asarray(
            [by_model[m]["zbar_partition"] for m in MODELS]
        ),
        "mu_ha": np.asarray([by_model[m]["mu"] for m in MODELS]),
        "r_ws_bohr": np.asarray([by_model[m]["r_ws"] for m in MODELS]),
        # ``hnc_residual`` remains as a v2 compatibility alias, but it is
        # explicitly the residual of the archived output rather than merely
        # the best iterate seen at some earlier iteration.
        "hnc_residual": np.asarray(
            [by_model[m]["hnc_output_residual"] for m in MODELS]
        ),
        "hnc_output_residual": np.asarray(
            [by_model[m]["hnc_output_residual"] for m in MODELS]
        ),
        "closure_transform_max_abs": np.asarray(
            [by_model[m]["closure_transform_max_abs"] for m in MODELS]
        ),
        "aa_stage2_converged": np.asarray(
            [by_model[m]["aa_stage2_converged"] for m in MODELS]
        ),
        "aa_ext_converged": np.asarray(
            [by_model[m]["aa_ext_converged"] for m in MODELS]
        ),
        "threshold_state_status": np.asarray(
            [by_model[m]["threshold_state_status"] for m in MODELS]
        ),
        "hnc_converged": np.asarray(
            [by_model[m]["hnc_converged"] for m in MODELS]
        ),
        "producer_elapsed_s": np.asarray(
            [by_model[m]["elapsed_s"] for m in MODELS]
        ),
        "n0_bohr3": np.asarray([by_model[m]["n0"] for m in MODELS]),
        "q_scr_integral": np.asarray(
            [by_model[m]["q_scr"] for m in MODELS]
        ),
    }
    for model in MODELS:
        state = by_model[model]
        payload[f"r_{model}_bohr"] = np.asarray(state["r_e"], dtype=float)
        for key in (
            "n_full",
            "n_ext",
            "n_pa",
            "n_bound",
            "n_cont",
            "n_ion",
            "n_scr",
        ):
            payload[f"{key}_{model}_bohr3"] = np.asarray(
                state[key],
                dtype=float,
            )
        payload[f"v_full_{model}_ha"] = np.asarray(
            state["v_full"],
            dtype=float,
        )
    for key, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Object arrays are forbidden: {key}.")
        if array.dtype.kind in "fiu" and not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite data in {key}.")
    return payload


def _filename(temperature_ev: float) -> str:
    token = f"{float(temperature_ev):g}".replace(".", "p")
    return f"Al_rho8p1gcc_T{token}eV_qm_tf.npz"


def regenerate(*, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Compute all states and write candidate data plus a checksum manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    producer = _producer_metadata()
    solved: dict[tuple[float, str], dict[str, Any]] = {}
    tasks = [
        (temperature_ev, model)
        for temperature_ev in TEMPERATURES_EV
        for model in MODELS
    ]
    with ProcessPoolExecutor(max_workers=MAX_STATE_WORKERS) as pool:
        futures = {
            pool.submit(_solve_one, temperature_ev, model): (
                temperature_ev,
                model,
            )
            for temperature_ev, model in tasks
        }
        for future in as_completed(futures):
            temperature_ev, model = futures[future]
            solved[(temperature_ev, model)] = future.result()
            result = solved[(temperature_ev, model)]
            print(
                f"[done] Al {temperature_ev:g} eV {model}: "
                f"{result['elapsed_s']:.2f} s, "
                f"Zbar={result['zbar_partition']:.6f}",
                flush=True,
            )

    paths: list[Path] = []
    state_records: list[dict[str, Any]] = []
    for temperature_ev in TEMPERATURES_EV:
        payload = _combined_payload(
            temperature_ev,
            {model: solved[(temperature_ev, model)] for model in MODELS},
            producer,
        )
        path = output_dir / _filename(temperature_ev)
        np.savez_compressed(path, **payload)
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
        "benchmark_id": "al_qm_tf",
        "status": "candidate_not_accepted",
        "title": (
            "Aluminium KS-DFT versus finite-temperature Thomas--Fermi "
            "at 8.1 g cm^-3"
        ),
        "producer": producer,
        "configuration": {
            "rho_g_cc": RHO_G_CC,
            "temperatures_ev": list(TEMPERATURES_EV),
            "models": list(MODELS),
            "structure_model": "IS",
            "aa_n_points": AA_N_POINTS,
            "bound_occ_mode": "fd",
            "bound_rmax_mult": None,
            "bound_zero_tail_refine": False,
            "b3_tail_model": "full",
            "qoz_n_points_before_padding": QOZ_N_POINTS,
            "lfc_model": "chabrier1990",
            "hnc_tolerance": HNC_TOL,
            "hnc_transform_closure_tolerance": HNC_CLOSURE_TOL,
            "hnc_max_iterations": 1000,
        },
        "method_references": [
            {
                "role": "average_atom_pseudoatom_qoz_hnc_and_qm_tf_models",
                "citation": (
                    "C. E. Starrett and D. Saumon, High Energy Density "
                    "Physics 10, 35-42 (2014)"
                ),
                "doi": "10.1016/j.hedp.2013.12.001",
            },
            {
                "role": "finite_temperature_jellium_lfc",
                "citation": "G. Chabrier, Journal de Physique 51, 1607-1632 (1990)",
                "doi": "10.1051/jphys:0199000510150160700",
            },
        ],
        "units": {
            "radius": "Bohr",
            "wavenumber": "Bohr^-1",
            "electronic_density": "Bohr^-3",
            "effective_pair_potential_r": "Hartree",
            "effective_pair_potential_k": "Hartree Bohr^3",
            "temperature": "eV",
            "mass_density": "g cm^-3",
            "gii_and_sii": "dimensionless",
        },
        "data_rights": {
            "origin_type": "project_generated_numerical_output",
            "third_party_reference_data_included": False,
            "method_citations_only": True,
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
