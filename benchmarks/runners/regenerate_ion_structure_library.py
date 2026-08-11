"""Regenerate the Otter side of the ion-structure library benchmark.

This program is intentionally separate from the offline documentation runner.
It performs expensive average-atom and QOZ/HNC calculations and writes only
to ``benchmarks/outputs/ion_structure_library/recomputed``.  It never
overwrites an accepted reference result.

The calculation is parallelized at two levels.  Independent thermodynamic
state groups run concurrently, while continuum energies within each
average-atom calculation use a smaller worker pool.  Keep the product of
``MAX_STATE_WORKERS`` and ``CONTINUUM_WORKERS_PER_STATE`` below the number of
physical CPU cores to avoid nested oversubscription.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
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


# User-editable execution controls.  Three state groups times six continuum
# workers uses at most 18 of a 24-core workstation.  The remaining cores keep
# the desktop responsive and cover BLAS/runtime overhead.
MAX_STATE_WORKERS = 3
CONTINUUM_WORKERS_PER_STATE = 6
AA_N_POINTS = 1024
QOZ_N_POINTS = 4096
R_RETAIN_MAX_BOHR = 20.0
K_RETAIN_MAX_BOHR_INV = 20.0

OUTPUT_DIR = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "ion_structure_library"
    / "recomputed"
)
SCHEMA = "otter_ion_structure_library_state_v1"


STATE_GROUPS: dict[str, tuple[dict[str, Any], ...]] = {
    "al_gill": (
        {
            "state_id": "al_gill_rho2p7_te5_ti5",
            "element": "Al",
            "rho_g_cc": 2.7,
            "te_ev": 5.0,
            "ti_ev": 5.0,
        },
    ),
    "al_clerouin": (
        {
            "state_id": "al_clerouin_rho8p1_te10_ti10",
            "element": "Al",
            "rho_g_cc": 8.1,
            "te_ev": 10.0,
            "ti_ev": 10.0,
        },
        {
            "state_id": "al_clerouin_rho8p1_te10_ti2",
            "element": "Al",
            "rho_g_cc": 8.1,
            "te_ev": 10.0,
            "ti_ev": 2.0,
        },
    ),
    "be_wunsch": (
        {
            "state_id": "be_wunsch_rho5p544_te13_ti13",
            "element": "Be",
            "rho_g_cc": 5.544,
            "te_ev": 13.0,
            "ti_ev": 13.0,
        },
    ),
    "c_starrett_hot": (
        {
            "state_id": "c_starrett_rho20_te50_ti50",
            "element": "C",
            "rho_g_cc": 20.0,
            "te_ev": 50.0,
            "ti_ev": 50.0,
        },
    ),
    "c_starrett_cold": (
        {
            "state_id": "c_starrett_rho3p7_te8p62_ti8p62",
            "element": "C",
            "rho_g_cc": 3.7,
            "te_ev": 8.62,
            "ti_ev": 8.62,
        },
    ),
}


def _git_commit() -> str:
    """Return the source revision without storing an absolute checkout path."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _configuration(state: dict[str, Any]) -> PlasmaWorkflowConfig:
    """Build the documented production configuration for one state."""
    is_cold_carbon = str(state["state_id"]) == "c_starrett_rho3p7_te8p62_ti8p62"
    state_n_points = 4096 if is_cold_carbon else int(AA_N_POINTS)
    return PlasmaWorkflowConfig(
        elements=[str(state["element"])],
        temperature_ev=float(state["te_ev"]),
        ion_temperature_ev=float(state["ti_ev"]),
        rho_g_cc=float(state["rho_g_cc"]),
        aa_overrides={
            "n_points": state_n_points,
            "cont_n_jobs": int(CONTINUUM_WORKERS_PER_STATE),
            "cont_shards": int(2 * CONTINUUM_WORKERS_PER_STATE),
            "bound_occ_mode": "fd",
            # Production uses the common AA radial domain documented by
            # Starrett--Saumon (2014), Appendix B.  The exterior threshold
            # match below acts on that same physical box; it does not create
            # a separate extended bound-only domain.
            "bound_rmax_mult": None,
            "bound_zero_tail_refine": True,
            "bound_zero_tail_max_binding_ha": 1.0e-2,
            "bound_zero_tail_scan_points": 64,
            "bound_zero_tail_edge_rel_tol": 0.1,
            "b3_tail_model": "full",
        },
        qoz_linear_n_points=int(QOZ_N_POINTS),
        qoz_pad_factor=2.0,
        qoz_zbar_mode="pseudoatom_partition",
        qoz_renormalize_nscr_to_zbar=True,
        qoz_response_chi0_model="lindhard_fd",
        qoz_response_lfc_model="chabrier1990",
        hnc_tol=1.0e-4,
        # Keep the nonlinear root strict while independently allowing the
        # measured ~2e-3 finite-DST g<->S mismatch of cold, strongly coupled
        # Al.  This does not relax positivity or fixed-point checks.
        hnc_closure_transform_tol=2.5e-3,
        hnc_max_iter=1000,
        hnc_require_converged=True,
        show_progress=False,
    )


def _finite_bound_levels(electronic: dict[str, Any]) -> dict[str, np.ndarray]:
    """Flatten only finite, negative-energy bound levels for portable output."""
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
    occ = np.asarray(
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
                        float(occ[li, ni]),
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


def _profile(
    electronic: dict[str, Any],
    name: str,
    mask: np.ndarray,
    *,
    fallback: str | None = None,
) -> np.ndarray:
    key = name if name in electronic else fallback
    if key is None or key not in electronic:
        return np.full(int(np.count_nonzero(mask)), np.nan)
    return np.asarray(electronic[key], dtype=float)[mask]


def _pack_result(
    result: dict[str, Any],
    state: dict[str, Any],
    *,
    elapsed_s: float,
) -> dict[str, np.ndarray]:
    """Convert a workflow payload to a compact pickle-free state archive."""
    electronic = dict(result["electronic"]["result"])
    ion = dict(result["ion"])
    r_e = np.asarray(electronic["r"], dtype=float)
    r_i = np.asarray(ion["r"], dtype=float)
    k = np.asarray(ion["k"], dtype=float)
    electron_mask = r_e <= R_RETAIN_MAX_BOHR
    ion_mask = r_i <= R_RETAIN_MAX_BOHR
    k_mask = k <= K_RETAIN_MAX_BOHR_INV

    state_n_points = (
        4096
        if str(state["state_id"]) == "c_starrett_rho3p7_te8p62_ti8p62"
        else int(AA_N_POINTS)
    )
    signature = {
        "state": state,
        "structure_model": "IS",
        "electronic_model": "qm",
        "aa": {
            "n_points": state_n_points,
            "bound_occ_mode": "fd",
            "bound_rmax_mult": None,
            "bound_zero_tail_refine": True,
            "bound_zero_tail_max_binding_ha": 1.0e-2,
            "bound_zero_tail_scan_points": 64,
            "bound_zero_tail_edge_rel_tol": 0.1,
            "b3_tail_model": "full",
            "continuum_workers": CONTINUUM_WORKERS_PER_STATE,
        },
        "qoz": {
            "n_points": QOZ_N_POINTS,
            "zbar_mode": "pseudoatom_partition",
            "renormalize_nscr_to_zbar": True,
            "chi0_model": "lindhard_fd",
            "lfc_model": "chabrier1990",
        },
        "hnc": {
            "tol": 1.0e-4,
            "closure_transform_tol": 2.5e-3,
            "max_iter": 1000,
            "require_converged": True,
        },
    }
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA),
        "benchmark_id": np.asarray("ion_structure_library"),
        "state_id": np.asarray(str(state["state_id"])),
        "element": np.asarray(str(state["element"])),
        "rho_g_cc": np.asarray(float(state["rho_g_cc"])),
        "te_ev": np.asarray(float(state["te_ev"])),
        "ti_ev": np.asarray(float(state["ti_ev"])),
        "otter_git_commit": np.asarray(_git_commit()),
        "producer_signature_json": np.asarray(
            json.dumps(signature, sort_keys=True, separators=(",", ":"))
        ),
        "producer_elapsed_s": np.asarray(float(elapsed_s)),
        "r_e_bohr": r_e[electron_mask],
        "n_full_bohr3": _profile(electronic, "n_full", electron_mask),
        # ``n_free`` is an internal A3-domain diagnostic and can be undefined
        # outside R_DFT,max.  The converged all-space continuum/free density is
        # ``n_cont`` after the documented B3 continuation.
        "n_free_bohr3": _profile(electronic, "n_cont", electron_mask),
        "n_cont_bohr3": _profile(electronic, "n_cont", electron_mask),
        "n_bound_bohr3": _profile(electronic, "n_bound", electron_mask),
        "n_ext_bohr3": _profile(electronic, "n_ext", electron_mask),
        "n_ion_bohr3": _profile(electronic, "n_ion", electron_mask),
        "n_pa_bohr3": _profile(electronic, "n_pa", electron_mask),
        "n_scr_bohr3": _profile(electronic, "n_scr", electron_mask),
        "v_full_ha": _profile(
            electronic,
            "v_full",
            electron_mask,
            fallback="v_scf",
        ),
        "v_ext_ha": _profile(electronic, "v_ext", electron_mask),
        "v_hartree_ha": _profile(electronic, "v_H", electron_mask),
        "v_xc_ha": _profile(electronic, "v_xc", electron_mask),
        "n0_bohr3": np.asarray(float(electronic["n0"])),
        "r_ws_bohr": np.asarray(float(electronic["r_ws"])),
        "mu_ha": np.asarray(float(electronic["mu"])),
        "zbar_aa": np.asarray(float(electronic["zbar"])),
        "zbar_partition": np.asarray(float(ion["zbar_partition"])),
        "zbar_qoz": np.asarray(float(ion["zbar_qoz"])),
        "threshold_state_status": np.asarray(
            str(electronic.get("threshold_state_status", "none"))
        ),
        "threshold_state_representation": np.asarray(
            str(electronic.get("threshold_state_representation", "none"))
        ),
        "q_scr_raw": np.asarray(
            float(ion["zbar_screening_integral_raw"])
        ),
        "r_bohr": r_i[ion_mask],
        "k_bohr_inv": k[k_mask],
        "gii_r": np.asarray(ion["gii_r"], dtype=float)[ion_mask],
        "sii_k": np.asarray(ion["sii_k"], dtype=float)[k_mask],
        "vii_r_ha": np.asarray(ion["vii_r"], dtype=float)[ion_mask],
        "vii_k_ha_bohr3": np.asarray(ion["vii_k"], dtype=float)[k_mask],
        "n_scr_k_electrons": np.asarray(
            ion["n_scr_k"],
            dtype=float,
        )[k_mask],
        "chi0_k_bohr3_per_ha": np.asarray(
            ion["chi0_k"],
            dtype=float,
        )[k_mask],
        "gee_k": np.asarray(ion["gee_k"], dtype=float)[k_mask],
        "hnc_best_residual": np.asarray(float(ion["hnc_best_residual"])),
        "hnc_closure_mismatch": np.asarray(
            float(ion["closure_transform_max_abs"])
        ),
        "hnc_closure_tolerance": np.asarray(
            float(ion["closure_transform_tol"])
        ),
        "hnc_iters": np.asarray(int(ion["hnc_iters"])),
    }
    payload.update(_finite_bound_levels(electronic))
    for key, value in payload.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Object dtype is forbidden for {key!r}.")
        if array.dtype.kind in "fiu" and not np.all(np.isfinite(array)):
            # Missing optional potential components are not acceptable in a
            # canonical output; fail loudly instead of publishing a partial
            # gallery state.
            raise ValueError(f"Non-finite values in {key!r}.")
    return payload


def _solve_group(
    group_name: str,
    states: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Solve one electronic state and all requested ion temperatures."""
    first = states[0]
    start = time.perf_counter()
    first_result = solve_plasma_workflow(_configuration(first))
    packed = {
        str(first["state_id"]): _pack_result(
            first_result,
            first,
            elapsed_s=time.perf_counter() - start,
        )
    }
    if len(states) == 1:
        return packed

    electronic_kind = str(first_result["electronic"]["kind"])
    electronic_result = dict(first_result["electronic"]["result"])
    for state in states[1:]:
        ion_start = time.perf_counter()
        result = continue_plasma_workflow_from_electronic_result(
            _configuration(state),
            electronic_kind=electronic_kind,
            electronic_result=electronic_result,
        )
        packed[str(state["state_id"])] = _pack_result(
            result,
            state,
            elapsed_s=time.perf_counter() - ion_start,
        )
    return packed


def regenerate(
    *,
    output_dir: Path = OUTPUT_DIR,
    max_state_workers: int = MAX_STATE_WORKERS,
) -> list[Path]:
    """Run all groups and return the written portable NPZ paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    failures: list[tuple[str, str]] = []
    workers = max(1, min(int(max_state_workers), len(STATE_GROUPS)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_solve_group, name, states): name
            for name, states in STATE_GROUPS.items()
        }
        for future in as_completed(futures):
            group = futures[future]
            try:
                results = future.result()
            except Exception as exc:
                failures.append((group, f"{type(exc).__name__}: {exc}"))
                print(f"[failed] group={group}: {type(exc).__name__}: {exc}")
                continue
            for state_id, payload in results.items():
                path = output_dir / f"{state_id}.npz"
                np.savez_compressed(path, **payload)
                written.append(path)
                print(
                    f"[saved] {state_id}: {path.relative_to(ROOT)} "
                    f"({float(payload['producer_elapsed_s']):.1f} s)"
                )
            print(f"[complete] group={group}")
    if failures:
        details = "; ".join(f"{group}: {reason}" for group, reason in failures)
        raise RuntimeError(
            "One or more core benchmark groups failed after successful "
            f"states were preserved: {details}"
        )
    return sorted(written)


def main() -> None:
    print(
        "Otter recomputation: "
        f"{MAX_STATE_WORKERS} state workers x "
        f"{CONTINUUM_WORKERS_PER_STATE} continuum workers"
    )
    paths = regenerate()
    print(f"Wrote {len(paths)} staged states under {OUTPUT_DIR.relative_to(ROOT)}")
    print("Accepted reference results were not modified.")


if __name__ == "__main__":
    # Guard required by multiprocessing on spawn-based platforms.
    main()
