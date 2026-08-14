"""Regenerate the experimental Al IS/SC example result.

This producer compares the ion-sphere (IS) and self-consistent (SC)
average-atom/QOZ constructions for aluminium at 8.1 g/cc and
``Te = Ti = 15 eV``.  Both the orbital Kohn--Sham (``qm``) and
finite-temperature Thomas--Fermi (``tf``) electronic backends are evaluated.

The SC feedback follows C. E. Starrett and D. Saumon, *High Energy Density
Physics* **10**, 35--42 (2014), Sec. 2.4, Eqs. (19)--(20),
doi:10.1016/j.hedp.2013.12.001.  It remains an experimental Otter feature.

Accepted gallery data are never overwritten.  This script writes a candidate
archive and checksum manifest under
``benchmarks/outputs/al_is_sc_comparison/recomputed``.
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
from otter.experimental import (  # noqa: E402
    SCFeedbackConfig,
    solve_sc_feedback_workflow,
)


# User-editable numerical controls.  The two electronic models are independent
# and can use two processes; the KS continuum integration uses eight workers.
ELEMENT = "Al"
RHO_G_CC = 8.1
TE_EV = 15.0
TI_EV = 15.0
MODELS = ("qm", "tf")
MODEL_DISPLAY_LABELS = ("KS-DFT", "Thomas--Fermi")
STRUCTURES = ("is", "sc")
MAX_MODEL_WORKERS = 2
CONTINUUM_WORKERS = 8
HNC_TOL = 1.0e-4
HNC_CLOSURE_TOL = 2.5e-3
R_RETAIN_MAX_BOHR = 20.0
K_RETAIN_MAX_BOHR_INV = 20.0

# The 0.5 outer mixing converges this state more quickly than the conservative
# library default while retaining both explicit Starrett--Saumon convergence
# tests.  No unconverged result is exported.
SC_CONTROLS = SCFeedbackConfig(
    max_outer=16,
    g_tol=5.0e-4,
    v_corr_tol=5.0e-4,
    v_corr_mix=0.5,
    require_converged=True,
)

OUTPUT_DIR = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "al_is_sc_comparison"
    / "recomputed"
)
OUTPUT_PATH = OUTPUT_DIR / "Al_rho8p1gcc_Te15eV_Ti15eV_qm_tf_is_sc.npz"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
SCHEMA = "otter_al_is_sc_comparison_v1"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_status_porcelain() -> str:
    """Return the complete tracked/untracked producer status."""
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "git metadata unavailable\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def configuration(model: str) -> PlasmaWorkflowConfig:
    """Return the documented IS workflow used as the SC starting point."""
    model_key = str(model).strip().lower()
    if model_key not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}.")
    return PlasmaWorkflowConfig(
        elements=[ELEMENT],
        temperature_ev=TE_EV,
        ion_temperature_ev=TI_EV,
        rho_g_cc=RHO_G_CC,
        electronic_model=model_key,
        aa_overrides={
            "cont_n_jobs": CONTINUUM_WORKERS,
            "cont_shards": 2 * CONTINUUM_WORKERS,
        },
        hnc_closure_transform_tol=HNC_CLOSURE_TOL,
        hnc_max_iter=500,
    )


def _finite_bound_levels(electronic: dict[str, Any]) -> dict[str, np.ndarray]:
    """Flatten finite negative-energy KS levels for a portable archive."""
    energies = np.asarray(
        electronic.get("bound_energy_ha", np.empty((0, 0))),
        dtype=float,
    )
    l_values = np.asarray(
        electronic.get("bound_l_list", np.arange(energies.shape[0])),
        dtype=int,
    )
    fd = np.asarray(
        electronic.get("bound_fd", np.zeros_like(energies)),
        dtype=float,
    )
    occupations = np.asarray(
        electronic.get("bound_occ_deg_fd", np.zeros_like(energies)),
        dtype=float,
    )
    records: list[tuple[int, int, float, float, float]] = []
    for li in range(energies.shape[0]):
        for ni in range(energies.shape[1]):
            energy = float(energies[li, ni])
            if np.isfinite(energy) and energy < 0.0:
                records.append(
                    (
                        int(l_values[li]),
                        int(ni + 1),
                        energy,
                        float(fd[li, ni]),
                        float(occupations[li, ni]),
                    )
                )
    if not records:
        return {
            "bound_l": np.empty(0, dtype=int),
            "bound_n_index": np.empty(0, dtype=int),
            "bound_energy_ha": np.empty(0),
            "bound_fd": np.empty(0),
            "bound_occ_deg_fd": np.empty(0),
        }
    values = np.asarray(records, dtype=float)
    return {
        "bound_l": values[:, 0].astype(int),
        "bound_n_index": values[:, 1].astype(int),
        "bound_energy_ha": values[:, 2],
        "bound_fd": values[:, 3],
        "bound_occ_deg_fd": values[:, 4],
    }


def _extract_path(workflow: dict[str, Any]) -> dict[str, Any]:
    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"])
    r = np.asarray(ion["r"], dtype=float)
    k = np.asarray(ion["k"], dtype=float)
    r_mask = r <= R_RETAIN_MAX_BOHR
    k_mask = k <= K_RETAIN_MAX_BOHR_INV
    return {
        "r": r[r_mask],
        "k": k[k_mask],
        "gii": np.asarray(ion["gii_r"], dtype=float)[r_mask],
        "sii": np.asarray(ion["sii_k"], dtype=float)[k_mask],
        "mu": float(electronic["mu"]),
        "zbar": float(ion["zbar_partition"]),
        "hnc_residual": float(ion["hnc_output_residual"]),
        "levels": _finite_bound_levels(electronic),
    }


def _solve_model(model: str) -> dict[str, Any]:
    cfg = configuration(model)
    started = time.perf_counter()
    is_workflow = solve_plasma_workflow(cfg)
    is_elapsed_s = time.perf_counter() - started

    started = time.perf_counter()
    sc_workflow = solve_sc_feedback_workflow(
        cfg,
        is_workflow,
        feedback_cfg=SC_CONTROLS,
    )
    sc_extension_elapsed_s = time.perf_counter() - started
    feedback = dict(sc_workflow["sc_feedback"])
    history = list(feedback["history"])
    return {
        "model": str(model),
        "is": _extract_path(is_workflow),
        "sc": _extract_path(sc_workflow),
        "is_elapsed_s": float(is_elapsed_s),
        "sc_extension_elapsed_s": float(sc_extension_elapsed_s),
        "sc_total_elapsed_s": float(is_elapsed_s + sc_extension_elapsed_s),
        "sc_converged": bool(feedback["converged"]),
        "sc_iterations": int(feedback["iterations"]),
        "fixed_is_mu_ha": float(feedback["fixed_is_mu_ha"]),
        "history_iteration": np.asarray(
            [int(item["iteration"]) for item in history],
            dtype=int,
        ),
        "history_max_g_change": np.asarray(
            [float(item["max_g_change"]) for item in history],
            dtype=float,
        ),
        "history_max_v_corr_change_ha": np.asarray(
            [float(item["max_v_corr_change_ha"]) for item in history],
            dtype=float,
        ),
    }


def _interp(x: np.ndarray, source_x: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.interp(
        np.asarray(x, dtype=float),
        np.asarray(source_x, dtype=float),
        np.asarray(values, dtype=float),
    )


def _payload(by_model: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    reference = by_model["qm"]["is"]
    r = np.asarray(reference["r"], dtype=float)
    k = np.asarray(reference["k"], dtype=float)
    gii = np.empty((len(MODELS), len(STRUCTURES), r.size), dtype=float)
    sii = np.empty((len(MODELS), len(STRUCTURES), k.size), dtype=float)
    mu = np.empty((len(MODELS), len(STRUCTURES)), dtype=float)
    zbar = np.empty_like(mu)
    hnc_residual = np.empty_like(mu)
    for model_index, model in enumerate(MODELS):
        for structure_index, structure in enumerate(STRUCTURES):
            state = by_model[model][structure]
            gii[model_index, structure_index] = _interp(
                r, state["r"], state["gii"]
            )
            sii[model_index, structure_index] = _interp(
                k, state["k"], state["sii"]
            )
            mu[model_index, structure_index] = float(state["mu"])
            zbar[model_index, structure_index] = float(state["zbar"])
            hnc_residual[model_index, structure_index] = float(
                state["hnc_residual"]
            )

    signature = {
        "state": {
            "element": ELEMENT,
            "rho_g_cc": RHO_G_CC,
            "te_ev": TE_EV,
            "ti_ev": TI_EV,
        },
        "electronic_models": list(MODELS),
        "structure_models": ["IS", "SC (experimental)"],
        "aa": {
            "n_points": 4096,
            "bound_occ_mode": "fd",
            "bound_rmax_mult": None,
            "bound_zero_tail_refine": False,
            "b3_tail_model": "full",
            "continuum_workers": CONTINUUM_WORKERS,
        },
        "qoz": {
            "n_points": 4096,
            "zbar_mode": "pseudoatom_partition",
            "renormalize_nscr_to_zbar": True,
            "chi0_model": "lindhard_fd",
            "lfc_model": "chabrier1990",
        },
        "sc_feedback": {
            "reference": "StarrettSaumon2014_Sec2.4_Eqs19_20",
            "max_outer": SC_CONTROLS.max_outer,
            "g_tol": SC_CONTROLS.g_tol,
            "v_corr_tol_ha": SC_CONTROLS.v_corr_tol,
            "v_corr_mix": SC_CONTROLS.v_corr_mix,
        },
    }
    producer_status = _git_status_porcelain()
    producer_script = Path(__file__).resolve()
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA),
        "example_id": np.asarray("al_is_sc_comparison"),
        "element_symbol": np.asarray(ELEMENT),
        "rho_g_cc": np.asarray(RHO_G_CC),
        "te_ev": np.asarray(TE_EV),
        "ti_ev": np.asarray(TI_EV),
        "model_labels": np.asarray(MODELS),
        "model_display_labels": np.asarray(MODEL_DISPLAY_LABELS),
        "structure_labels": np.asarray(STRUCTURES),
        "sc_status": np.asarray("experimental"),
        "sc_reference": np.asarray(
            "Starrett and Saumon 2014 Sec. 2.4 Eqs. (19)-(20)"
        ),
        "otter_git_commit": np.asarray(_git_commit()),
        "producer_script_sha256": np.asarray(_sha256(producer_script)),
        "producer_worktree_clean": np.asarray(not bool(producer_status)),
        "producer_git_status_porcelain_sha256": np.asarray(
            _text_sha256(producer_status)
        ),
        "producer_signature_json": np.asarray(
            json.dumps(signature, sort_keys=True, separators=(",", ":"))
        ),
        "r_bohr": r,
        "k_bohr_inv": k,
        "gii_r": gii,
        "sii_k": sii,
        "mu_ha": mu,
        "zbar_partition": zbar,
        "hnc_residual": hnc_residual,
        "is_elapsed_s": np.asarray(
            [by_model[model]["is_elapsed_s"] for model in MODELS]
        ),
        "sc_extension_elapsed_s": np.asarray(
            [by_model[model]["sc_extension_elapsed_s"] for model in MODELS]
        ),
        "sc_total_elapsed_s": np.asarray(
            [by_model[model]["sc_total_elapsed_s"] for model in MODELS]
        ),
        "sc_converged": np.asarray(
            [by_model[model]["sc_converged"] for model in MODELS],
            dtype=bool,
        ),
        "sc_iterations": np.asarray(
            [by_model[model]["sc_iterations"] for model in MODELS],
            dtype=int,
        ),
        "fixed_is_mu_ha": np.asarray(
            [by_model[model]["fixed_is_mu_ha"] for model in MODELS]
        ),
        "tf_discrete_ks_levels_defined": np.asarray(False),
    }
    for model in MODELS:
        for structure in STRUCTURES:
            levels = by_model[model][structure]["levels"]
            for name, values in levels.items():
                payload[f"{model}_{structure}_{name}"] = np.asarray(values)
        payload[f"{model}_sc_history_iteration"] = np.asarray(
            by_model[model]["history_iteration"]
        )
        payload[f"{model}_sc_history_max_g_change"] = np.asarray(
            by_model[model]["history_max_g_change"]
        )
        payload[f"{model}_sc_history_max_v_corr_change_ha"] = np.asarray(
            by_model[model]["history_max_v_corr_change_ha"]
        )

    for key, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Object arrays are forbidden: {key}.")
        if array.dtype.kind in "fiu" and not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite data in {key}.")
    return payload


def regenerate(
    *,
    output_path: Path = OUTPUT_PATH,
    max_model_workers: int = MAX_MODEL_WORKERS,
) -> Path:
    """Compute both models and stage one reviewable candidate archive."""
    workers = max(1, min(int(max_model_workers), len(MODELS)))
    solved: dict[str, dict[str, Any]] = {}
    if workers == 1:
        for model in MODELS:
            solved[model] = _solve_model(model)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_solve_model, model): model for model in MODELS}
            for future in as_completed(futures):
                model = futures[future]
                solved[model] = future.result()
                print(
                    f"[done] {MODEL_DISPLAY_LABELS[MODELS.index(model)]}: "
                    f"IS={solved[model]['is_elapsed_s']:.2f} s, "
                    "SC extension="
                    f"{solved[model]['sc_extension_elapsed_s']:.2f} s",
                    flush=True,
                )

    payload = _payload(solved)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    manifest = {
        "schema_version": "otter_example_manifest_v1",
        "example_id": "al_is_sc_comparison",
        "title": (
            "Aluminium IS versus experimental SC feedback with KS-DFT "
            "and Thomas--Fermi electrons"
        ),
        "status": "candidate_not_accepted",
        "method_publication": {
            "authors": ["C. E. Starrett", "D. Saumon"],
            "title": (
                "A simple method for determining the ionic structure "
                "of warm dense matter"
            ),
            "journal": "High Energy Density Physics",
            "volume": "10",
            "pages": "35-42",
            "year": 2014,
            "doi": "10.1016/j.hedp.2013.12.001",
            "relevant_section": "2.4",
            "relevant_equations": "19-20",
        },
        "data_rights": {
            "origin_type": "project_generated_numerical_output",
            "third_party_reference_data_included": False,
            "method_citations_only": True,
        },
        "units": {
            "radius": "Bohr",
            "wavenumber": "Bohr^-1",
            "energy": "Hartree",
            "temperature": "eV",
            "mass_density": "g cm^-3",
            "gii_and_sii": "dimensionless",
            "time": "second",
        },
        "producer": {
            "project": "Otter",
            "git_commit": str(payload["otter_git_commit"].item()),
            "script_sha256": str(
                payload["producer_script_sha256"].item()
            ),
            "worktree_clean_at_generation": bool(
                payload["producer_worktree_clean"]
            ),
            "git_status_porcelain_sha256": str(
                payload["producer_git_status_porcelain_sha256"].item()
            ),
            "script_relative_path": str(
                Path(__file__).resolve().relative_to(ROOT)
            ),
        },
        "state": {
            "element": ELEMENT,
            "rho_g_cc": RHO_G_CC,
            "te_ev": TE_EV,
            "ti_ev": TI_EV,
            "data_file": output_path.name,
            "data_sha256": _sha256(output_path),
        },
    }
    manifest_path = output_path.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[saved candidate] {output_path.relative_to(ROOT)}", flush=True)
    print(
        "[note] Accepted gallery data were not modified; review before promotion.",
        flush=True,
    )
    return output_path


if __name__ == "__main__":
    regenerate()
