"""Validate the complete Al 8.1 g/cc, 1 eV gallery reference result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / "al_full_workflow_1ev"
MANIFEST_PATH = BASELINE_DIR / "manifest.json"
SCHEMA = "otter_al_full_workflow_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "otter_benchmark_manifest_v1":
        raise ValueError("Unsupported benchmark manifest schema.")
    if manifest.get("benchmark_id") != "al_full_workflow_1ev":
        raise ValueError("Unexpected benchmark identifier.")
    if not str(manifest.get("status", "accepted")).startswith("accepted"):
        raise ValueError("The installed Al workflow result is not accepted.")
    return manifest


def load_state(
    *,
    path: Path | None = None,
    verify_checksum: bool = True,
) -> dict[str, np.ndarray]:
    manifest = load_manifest()
    state_record = manifest["state"]
    filename = state_record.get(
        "data_file",
        state_record.get("baseline_file"),
    )
    if not isinstance(filename, str):
        raise ValueError("The Al workflow manifest has no data filename.")
    baseline_path = (
        BASELINE_DIR / filename
        if path is None
        else Path(path)
    )
    if verify_checksum and path is None:
        expected_sha256 = state_record.get(
            "data_sha256",
            state_record.get("baseline_sha256"),
        )
        if not isinstance(expected_sha256, str):
            raise ValueError("The Al workflow manifest has no data checksum.")
        if sha256_file(baseline_path) != expected_sha256:
            raise RuntimeError(
                "Al full-workflow reference-result checksum mismatch."
            )
    with np.load(baseline_path, allow_pickle=False) as archive:
        state = {key: np.asarray(archive[key]) for key in archive.files}
    if state["schema_version"].item() != SCHEMA:
        raise ValueError("Unsupported Al full-workflow state schema.")
    for key, value in state.items():
        if value.dtype.hasobject:
            raise TypeError(f"Object dtype is forbidden for {key!r}.")
        if value.dtype.kind in "fiu" and not np.all(np.isfinite(value)):
            raise ValueError(f"Non-finite values in {key!r}.")
    if not np.isclose(float(state["rho_g_cc"]), 8.1):
        raise ValueError("Unexpected Al mass density.")
    if not np.isclose(float(state["te_ev"]), 1.0):
        raise ValueError("Unexpected electron temperature.")
    if not np.isclose(float(state["ti_ev"]), 1.0):
        raise ValueError("Unexpected ion temperature.")
    if float(state["hnc_best_residual"]) > 1.0e-4:
        raise ValueError("HNC residual exceeds the recorded production limit.")
    if float(state["hnc_closure_mismatch"]) > 2.5e-3:
        raise ValueError("Finite-DST g/S closure mismatch exceeds its limit.")
    if state["n_ion_k_electrons"].shape != state["k_bohr_inv"].shape:
        raise ValueError("f(k)=n_ion(k) does not share the QOZ reciprocal grid.")
    return state


def bound_level_rows(state: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    symbols = ("s", "p", "d", "f", "g", "h")
    rows: list[dict[str, Any]] = []
    for l_value, n_index, energy, fd, occupation in zip(
        state["bound_l"],
        state["bound_n_index"],
        state["bound_energy_ha"],
        state["bound_fd"],
        state["bound_occ_deg_fd"],
        strict=True,
    ):
        l_int = int(l_value)
        principal_n = int(n_index) + l_int
        label = (
            f"{principal_n}{symbols[l_int]}"
            if l_int < len(symbols)
            else f"n={principal_n}, l={l_int}"
        )
        rows.append(
            {
                "level": label,
                "energy_ha": float(energy),
                "energy_ev": 27.211386245988 * float(energy),
                "fd": float(fd),
                "occupation": float(occupation),
            }
        )
    return rows


def print_summary(state: dict[str, np.ndarray]) -> None:
    print(
        "Al rho=8.1 g/cc, Te=Ti=1 eV: "
        f"mu={float(state['mu_ha']):.8f} Ha, "
        f"Zbar(partition)={float(state['zbar_partition']):.8f}"
    )
    print(
        f"HNC residual={float(state['hnc_best_residual']):.3e}, "
        f"g/S closure={float(state['hnc_closure_mismatch']):.3e}"
    )
    print(
        f"{'level':>7s} {'E [Ha]':>13s} {'E [eV]':>13s} "
        f"{'FD':>10s} {'occupation':>12s}"
    )
    for row in bound_level_rows(state):
        print(
            f"{row['level']:>7s} {row['energy_ha']:13.6f} "
            f"{row['energy_ev']:13.6f} {row['fd']:10.6f} "
            f"{row['occupation']:12.6f}"
        )


if __name__ == "__main__":
    print_summary(load_state())
