"""Load and validate the precomputed Al IS/SC gallery result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / "al_is_sc_comparison"
MANIFEST_PATH = BASELINE_DIR / "manifest.json"
CANDIDATE_PATH = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "al_is_sc_comparison"
    / "recomputed"
    / "Al_rho8p1gcc_Te15eV_Ti15eV_qm_tf_is_sc.npz"
)
SCHEMA = "otter_al_is_sc_comparison_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "otter_example_manifest_v1":
        raise ValueError("Unsupported example manifest schema.")
    if manifest.get("example_id") != "al_is_sc_comparison":
        raise ValueError("Unexpected example identifier.")
    if manifest.get("status") != "accepted":
        raise ValueError("The installed IS/SC example result is not accepted.")
    return manifest


def validate_state(state: dict[str, np.ndarray]) -> None:
    """Check schema, dimensions, convergence, and fixed-IS-mu invariants."""
    if state["schema_version"].item() != SCHEMA:
        raise ValueError("Unsupported Al IS/SC state schema.")
    if tuple(state["model_labels"].tolist()) != ("qm", "tf"):
        raise ValueError("Expected the QM and TF model ordering.")
    if tuple(state["structure_labels"].tolist()) != ("is", "sc"):
        raise ValueError("Expected the IS and SC structure ordering.")
    if state["sc_status"].item() != "experimental":
        raise ValueError("SC must remain explicitly labelled experimental.")
    if not np.isclose(float(state["rho_g_cc"]), 8.1):
        raise ValueError("Unexpected aluminium mass density.")
    if not np.isclose(float(state["te_ev"]), 15.0):
        raise ValueError("Unexpected electron temperature.")
    if not np.isclose(float(state["ti_ev"]), 15.0):
        raise ValueError("Unexpected ion temperature.")

    r = np.asarray(state["r_bohr"], dtype=float)
    k = np.asarray(state["k_bohr_inv"], dtype=float)
    if r.ndim != 1 or k.ndim != 1:
        raise ValueError("The retained r and k grids must be one-dimensional.")
    if not np.all(np.diff(r) > 0.0) or not np.all(np.diff(k) > 0.0):
        raise ValueError("The retained r and k grids must be strictly increasing.")
    if r[-1] > 20.0 or k[-1] > 20.0:
        raise ValueError("The archive exceeds its documented r/k retention limits.")
    if state["gii_r"].shape != (2, 2, r.size):
        raise ValueError("Unexpected gii_r dimensions.")
    if state["sii_k"].shape != (2, 2, k.size):
        raise ValueError("Unexpected sii_k dimensions.")
    if not np.all(np.asarray(state["sc_converged"], dtype=bool)):
        raise ValueError("An experimental SC calculation is unconverged.")
    if np.any(np.asarray(state["hnc_residual"], dtype=float) > 1.0e-4):
        raise ValueError("An HNC residual exceeds the production limit.")

    fixed_mu = np.asarray(state["fixed_is_mu_ha"], dtype=float)
    mu = np.asarray(state["mu_ha"], dtype=float)
    np.testing.assert_allclose(mu[:, 0], fixed_mu, rtol=0.0, atol=2.0e-10)
    np.testing.assert_allclose(mu[:, 1], fixed_mu, rtol=0.0, atol=2.0e-10)
    if bool(state["tf_discrete_ks_levels_defined"]):
        raise ValueError("Thomas--Fermi must not advertise discrete KS levels.")
    for structure in ("is", "sc"):
        if np.asarray(state[f"tf_{structure}_bound_energy_ha"]).size != 0:
            raise ValueError("Thomas--Fermi has no discrete KS-level table.")


def load_state(
    *,
    path: Path | None = None,
    verify_checksum: bool = True,
) -> dict[str, np.ndarray]:
    """Load the accepted archive, or an explicitly supplied candidate."""
    if path is None:
        manifest = load_manifest()
        archive_path = BASELINE_DIR / manifest["state"]["data_file"]
        if verify_checksum:
            expected = manifest["state"]["data_sha256"]
            if sha256_file(archive_path) != expected:
                raise RuntimeError("Al IS/SC example-result checksum mismatch.")
    else:
        archive_path = Path(path)
    with np.load(archive_path, allow_pickle=False) as archive:
        state = {key: np.asarray(archive[key]) for key in archive.files}
    for key, value in state.items():
        if value.dtype.hasobject:
            raise TypeError(f"Object dtype is forbidden for {key!r}.")
        if value.dtype.kind in "fiu" and not np.all(np.isfinite(value)):
            raise ValueError(f"Non-finite values in {key!r}.")
    validate_state(state)
    return state


def level_rows(
    state: dict[str, np.ndarray],
    structure: str,
) -> list[dict[str, float | str]]:
    """Return a human-readable QM KS-level table for IS or SC."""
    structure_key = str(structure).strip().lower()
    if structure_key not in ("is", "sc"):
        raise ValueError("structure must be 'is' or 'sc'.")
    prefix = f"qm_{structure_key}_"
    symbols = ("s", "p", "d", "f", "g", "h")
    rows: list[dict[str, float | str]] = []
    for l_value, n_index, energy, fd, occupation in zip(
        state[prefix + "bound_l"],
        state[prefix + "bound_n_index"],
        state[prefix + "bound_energy_ha"],
        state[prefix + "bound_fd"],
        state[prefix + "bound_occ_deg_fd"],
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
    print("Al, rho=8.1 g/cc, Te=Ti=15 eV")
    for index, label in enumerate(state["model_display_labels"].tolist()):
        print(
            f"{label}: IS={float(state['is_elapsed_s'][index]):.2f} s, "
            "SC extension="
            f"{float(state['sc_extension_elapsed_s'][index]):.2f} s, "
            f"SC total={float(state['sc_total_elapsed_s'][index]):.2f} s, "
            f"SC iterations={int(state['sc_iterations'][index])}"
        )
    print("SC status: experimental; Starrett--Saumon (2014), Sec. 2.4.")


if __name__ == "__main__":
    print_summary(load_state())
