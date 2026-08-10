"""Validate, optionally regenerate, and plot the Al KS-DFT/TF comparison.

The default path is offline and reads checksummed Otter results.  Set
``USE_PRECOMPUTED_DATA = False`` below to run the explicit Otter producer;
new candidate files are written below ``benchmarks/outputs`` and never
replace reviewed data automatically.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import otter.plotting as otter_plotting


# User-editable execution mode.  Keeping this ``True`` makes the runner fast
# and deterministic.  Set it to ``False`` only when an expensive, opt-in
# Otter recomputation is intended.
USE_PRECOMPUTED_DATA = True

# Maintainers may point this at an ignored candidate directory while reviewing
# it.  End users should normally leave it as ``None`` so the reviewed package
# under ``benchmarks/baselines`` is used.
PRECOMPUTED_DATA_DIR: Path | None = None

SHOW_FIGURE = False
FIGURE_DPI = 200
R_DENSITY_PLOT_MAX_BOHR = 8.0
R_SCREENING_PLOT_MAX_BOHR = 12.0
R_ION_PLOT_MAX_BOHR = 12.0
K_PLOT_MAX_BOHR_INV = 6.0

BENCHMARKS_DIR = Path(__file__).resolve().parents[1]
PRECOMPUTED_DIR = BENCHMARKS_DIR / "baselines" / "al_qm_tf"
OUTPUT_DIR = BENCHMARKS_DIR / "outputs" / "al_qm_tf"
RECOMPUTED_DIR = OUTPUT_DIR / "recomputed"
MANIFEST_PATH = PRECOMPUTED_DIR / "manifest.json"
FIGURE_PATH = OUTPUT_DIR / "al_qm_tf_offline.png"
METRICS_PATH = OUTPUT_DIR / "al_qm_tf_offline_metrics.csv"
MODEL_ORDER = ("qm", "tf")
EXPECTED_RHO_G_CC = 8.1
EXPECTED_TEMPERATURES_EV = (1.0, 15.0, 50.0, 100.0)
MAX_HNC_OUTPUT_RESIDUAL = 1.0e-6
MAX_CLOSURE_TRANSFORM_MISMATCH = 1.0e-3
METRIC_FIELDS = (
    "rho_g_cc",
    "temperature_ev",
    "zbar_ksdft",
    "zbar_tf",
    "delta_zbar_tf_minus_ksdft",
    "q_scr_integral_ksdft",
    "q_scr_integral_tf",
    "full_shell_rmse_bohr_inv",
    "ion_shell_rmse_bohr_inv",
    "screening_shell_rmse_bohr_inv",
    "gii_rmse_r_le_12",
    "sii_rmse_k_le_6",
    "source_elapsed_ksdft_s",
    "source_elapsed_tf_s",
    "data_file",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(
    path: Path = MANIFEST_PATH,
    *,
    allow_legacy_v1: bool = True,
) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = manifest.get("schema_version")
    if schema == "otter_benchmark_manifest_v1" and allow_legacy_v1:
        if manifest.get("benchmark_id") != "al_qm_tf":
            raise ValueError("Unexpected aluminium benchmark identifier.")
        return manifest
    if schema != "otter_benchmark_manifest_v2":
        raise ValueError(
            "The Al KS-DFT/TF v2 runner requires "
            "'otter_benchmark_manifest_v2'. The reviewed data package in "
            "this checkout may still be the legacy v1 package; either "
            "select a reviewed v2 directory or set "
            "USE_PRECOMPUTED_DATA = False to generate a candidate."
        )
    if manifest.get("benchmark_id") != "al_qm_tf":
        raise ValueError("Unexpected aluminium benchmark identifier.")
    states = manifest.get("states", [])
    observed_temperatures = tuple(
        float(state["temperature_ev"]) for state in states
    )
    if observed_temperatures != EXPECTED_TEMPERATURES_EV:
        raise ValueError(
            "Expected Al temperatures "
            f"{EXPECTED_TEMPERATURES_EV}, got {observed_temperatures}."
        )
    if any(
        not np.isclose(float(state["rho_g_cc"]), EXPECTED_RHO_G_CC)
        for state in states
    ):
        raise ValueError("Every Al v2 state must use rho=8.1 g cm^-3.")
    return manifest


def resolve_data(
    state: dict[str, Any],
    data_dir: Path = PRECOMPUTED_DIR,
) -> Path:
    filename = state.get("data_file", state.get("baseline_file"))
    if not isinstance(filename, str):
        raise ValueError("State record has no portable data filename.")
    path = (Path(data_dir) / filename).resolve()
    if path.parent != Path(data_dir).resolve():
        raise ValueError("Data path escapes its package directory.")
    return path


def load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    object_keys = [key for key, value in result.items() if value.dtype.hasobject]
    if object_keys:
        raise TypeError(f"Object arrays are forbidden in {path}: {object_keys}")
    schema = str(result["schema_version"].item())
    if schema not in {
        "otter_al_qm_tf_state_v2",
        "otter_al_qm_tf_baseline_v1",
    }:
        raise ValueError(f"Unsupported state schema in {path}.")
    if tuple(str(item) for item in result["model_labels"]) != MODEL_ORDER:
        raise ValueError(f"Unexpected model order in {path}.")
    if schema == "otter_al_qm_tf_state_v2":
        required = {
            "aa_stage2_converged",
            "aa_ext_converged",
            "threshold_state_status",
            "hnc_converged",
            "hnc_output_residual",
            "closure_transform_max_abs",
        }
        missing = sorted(required.difference(result))
        if missing:
            raise ValueError(
                f"Missing production-convergence metadata in {path}: "
                + ", ".join(missing)
            )
        for key in (
            "aa_stage2_converged",
            "aa_ext_converged",
            "hnc_converged",
        ):
            values = np.asarray(result[key])
            if (
                values.shape != (2,)
                or values.dtype.kind != "b"
                or not np.all(values)
            ):
                raise ValueError(f"Rejected {key} status in {path}.")
        threshold_array = np.asarray(result["threshold_state_status"])
        if threshold_array.shape != (2,):
            raise ValueError(
                f"Rejected threshold-state metadata shape in {path}."
            )
        threshold_statuses = tuple(
            str(value).strip().lower()
            for value in threshold_array
        )
        if threshold_statuses[0] not in {"none", "resolved", "marginal"}:
            raise ValueError(
                f"Unresolved KS-DFT threshold state in {path}: "
                f"{threshold_statuses[0]!r}."
            )
        if threshold_statuses[1] != "not_applicable_tf":
            raise ValueError(
                f"Unexpected TF threshold-state status in {path}: "
                f"{threshold_statuses[1]!r}."
            )
        residuals = np.asarray(result["hnc_output_residual"], dtype=float)
        closures = np.asarray(
            result["closure_transform_max_abs"],
            dtype=float,
        )
        if (
            residuals.shape != (2,)
            or not np.all(np.isfinite(residuals))
            or np.any(residuals > MAX_HNC_OUTPUT_RESIDUAL)
        ):
            raise ValueError(f"Rejected HNC output residual in {path}.")
        if (
            closures.shape != (2,)
            or not np.all(np.isfinite(closures))
            or np.any(closures > MAX_CLOSURE_TRANSFORM_MISMATCH)
        ):
            raise ValueError(
                f"Rejected real/reciprocal closure mismatch in {path}."
            )
    return result


def _rmse(delta: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(delta, dtype=float) ** 2)))


def _shell_rmse(data: dict[str, np.ndarray], key: str) -> float:
    r_qm = np.asarray(data["r_qm_bohr"], dtype=float)
    r_tf = np.asarray(data["r_tf_bohr"], dtype=float)
    mask = r_qm <= min(
        R_SCREENING_PLOT_MAX_BOHR,
        float(r_qm[-1]),
        float(r_tf[-1]),
    )
    r = r_qm[mask]
    qm = np.asarray(data[f"{key}_qm_bohr3"], dtype=float)[mask]
    tf = np.interp(
        r,
        r_tf,
        np.asarray(data[f"{key}_tf_bohr3"], dtype=float),
    )
    return _rmse(4.0 * np.pi * r**2 * (tf - qm))


def state_metrics(data: dict[str, np.ndarray], filename: str) -> dict[str, Any]:
    r = np.asarray(data["r_ion_bohr"], dtype=float)
    k = np.asarray(data["k_bohr_inv"], dtype=float)
    gii = np.asarray(data["gii_r"], dtype=float)
    sii = np.asarray(data["sii_k"], dtype=float)
    r_mask = r <= R_ION_PLOT_MAX_BOHR
    k_mask = k <= K_PLOT_MAX_BOHR_INV
    zbar = np.asarray(data["zbar_partition"], dtype=float)
    q_scr = np.asarray(data["q_scr_integral"], dtype=float)
    elapsed = np.asarray(data["producer_elapsed_s"], dtype=float)
    schema = str(data["schema_version"].item())
    result: dict[str, Any] = {
        "rho_g_cc": float(data["rho_g_cc"]),
        "temperature_ev": float(data["temperature_ev"]),
        "zbar_tf": float(zbar[1]),
        "q_scr_integral_tf": float(q_scr[1]),
        "full_shell_rmse_bohr_inv": _shell_rmse(data, "n_full"),
        "ion_shell_rmse_bohr_inv": _shell_rmse(data, "n_ion"),
        "screening_shell_rmse_bohr_inv": _shell_rmse(data, "n_scr"),
        "gii_rmse_r_le_12": _rmse((gii[1] - gii[0])[..., r_mask]),
        "sii_rmse_k_le_6": _rmse((sii[1] - sii[0])[..., k_mask]),
        "source_elapsed_tf_s": float(elapsed[1]),
    }
    if schema == "otter_al_qm_tf_baseline_v1":
        # Preserve the v1 key order so the legacy checksum package remains
        # independently verifiable while v2 is under review.
        result = {
            "rho_g_cc": result["rho_g_cc"],
            "temperature_ev": result["temperature_ev"],
            "zbar_qm": float(zbar[0]),
            "zbar_tf": result["zbar_tf"],
            "delta_zbar_tf_minus_qm": float(zbar[1] - zbar[0]),
            "q_scr_integral_qm": float(q_scr[0]),
            "q_scr_integral_tf": result["q_scr_integral_tf"],
            "full_shell_rmse_bohr_inv": result[
                "full_shell_rmse_bohr_inv"
            ],
            "ion_shell_rmse_bohr_inv": result["ion_shell_rmse_bohr_inv"],
            "screening_shell_rmse_bohr_inv": result[
                "screening_shell_rmse_bohr_inv"
            ],
            "gii_rmse_r_le_12": result["gii_rmse_r_le_12"],
            "sii_rmse_k_le_6": result["sii_rmse_k_le_6"],
            "source_elapsed_qm_s": float(elapsed[0]),
            "source_elapsed_tf_s": result["source_elapsed_tf_s"],
            "baseline_file": filename,
        }
    else:
        result.update(
            {
                "zbar_ksdft": float(zbar[0]),
                "delta_zbar_tf_minus_ksdft": float(zbar[1] - zbar[0]),
                "q_scr_integral_ksdft": float(q_scr[0]),
                "source_elapsed_ksdft_s": float(elapsed[0]),
                "data_file": filename,
            }
        )
    return result


def evaluate_states(
    manifest: dict[str, Any],
    *,
    data_dir: Path = PRECOMPUTED_DIR,
    verify_checksums: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    metrics: list[dict[str, Any]] = []
    loaded: list[dict[str, np.ndarray]] = []
    for state in manifest["states"]:
        path = resolve_data(state, data_dir)
        actual_hash = sha256_file(path)
        expected_hash = state.get(
            "data_sha256",
            state.get("baseline_sha256"),
        )
        if not isinstance(expected_hash, str):
            raise ValueError(f"Missing SHA-256 provenance for {path}.")
        if verify_checksums and actual_hash != expected_hash:
            raise RuntimeError(f"Checksum mismatch for {path}.")
        data = load_state(path)
        if not np.isclose(float(data["rho_g_cc"]), state["rho_g_cc"]):
            raise ValueError(f"Density metadata mismatch in {path}.")
        if not np.isclose(
            float(data["temperature_ev"]), state["temperature_ev"]
        ):
            raise ValueError(f"Temperature metadata mismatch in {path}.")
        loaded.append(data)
        metrics.append(state_metrics(data, path.name))
    return metrics, loaded


def _load_module(name: str, path: Path) -> ModuleType:
    """Load an opt-in producer without importing Otter on the offline path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_data_source(
    *,
    use_precomputed_data: bool = USE_PRECOMPUTED_DATA,
    precomputed_data_dir: Path | None = PRECOMPUTED_DATA_DIR,
) -> tuple[dict[str, Any], Path]:
    """Return a v2 manifest and data directory for the requested mode.

    Recalculation is deliberately imported only in the ``False`` branch, so
    importing this validator and using reviewed data does not import Otter or
    invoke a scientific solver.
    """
    if use_precomputed_data:
        data_dir = (
            PRECOMPUTED_DIR
            if precomputed_data_dir is None
            else Path(precomputed_data_dir).expanduser().resolve()
        )
    else:
        producer = _load_module(
            "otter_al_qm_tf_regenerator",
            BENCHMARKS_DIR / "runners" / "regenerate_al_qm_tf.py",
        )
        producer.regenerate()
        data_dir = Path(producer.OUTPUT_DIR).resolve()
    return (
        load_manifest(
            data_dir / "manifest.json",
            allow_legacy_v1=False,
        ),
        data_dir,
    )


def write_metrics(metrics: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=METRIC_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(metrics)


def plot_states(loaded: list[dict[str, np.ndarray]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / ".cache"))
    import matplotlib

    if not SHOW_FIGURE:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    otter_plotting.set_style("docs", palette="deep_science")
    colors = {"qm": "tab:blue", "tf": "tab:orange"}
    styles = {"qm": "-", "tf": "--"}
    labels = {"qm": "KS-DFT", "tf": "Thomas–Fermi"}
    fig, axes = plt.subplots(
        len(loaded), 4, figsize=(15.5, 3.5 * len(loaded)), squeeze=False
    )
    for row, data in enumerate(loaded):
        for model_index, model in enumerate(MODEL_ORDER):
            r_e = np.asarray(data[f"r_{model}_bohr"], dtype=float)
            weight = 4.0 * np.pi * r_e**2
            n0 = float(np.asarray(data["n0_bohr3"])[model_index])
            axes[row, 0].plot(
                r_e,
                weight
                * (
                    np.asarray(data[f"n_full_{model}_bohr3"], dtype=float)
                    - n0
                ),
                styles[model],
                color=colors[model],
                label=f"{labels[model]} full-$n_0$",
            )
            axes[row, 1].plot(
                r_e,
                weight
                * np.asarray(data[f"n_scr_{model}_bohr3"], dtype=float),
                styles[model],
                color=colors[model],
                label=labels[model],
            )
            r_ion = np.asarray(data["r_ion_bohr"], dtype=float)
            k = np.asarray(data["k_bohr_inv"], dtype=float)
            axes[row, 2].plot(
                r_ion,
                np.asarray(data["gii_r"])[model_index],
                styles[model],
                color=colors[model],
                label=labels[model],
            )
            axes[row, 3].plot(
                k,
                np.asarray(data["sii_k"])[model_index],
                styles[model],
                color=colors[model],
                label=labels[model],
            )
        state_label = (
            f"{float(data['rho_g_cc']):g} g cm$^{{-3}}$, "
            f"{float(data['temperature_ev']):g} eV"
        )
        axes[row, 0].set_ylabel(
            state_label + "\n" + r"$4\pi r^2 n(r)$ [Bohr$^{-1}$]"
        )
        axes[row, 0].set_xlim(-1.0, R_DENSITY_PLOT_MAX_BOHR)
        axes[row, 1].set_xlim(-1.0, R_SCREENING_PLOT_MAX_BOHR)
        axes[row, 2].set_xlim(0.0, R_ION_PLOT_MAX_BOHR)
        axes[row, 3].set_xlim(0.0, K_PLOT_MAX_BOHR_INV)
        axes[row, 3].axhline(1.0, color="0.45", lw=0.8, ls=":")
        for axis in axes[row]:
            axis.grid(alpha=0.22)
    titles = (
        r"$n_{\rm full}-n_0$",
        "screening density",
        r"$g_{ii}(r)$",
        r"$S_{ii}(k)$",
    )
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title)
        axis.legend(fontsize=8)
    for axis in axes[-1, :3]:
        axis.set_xlabel(r"$r$ [Bohr]")
    axes[-1, 3].set_xlabel(r"$k$ [Bohr$^{-1}$]")
    axes[0, 1].set_ylabel(r"$4\pi r^2 n_{\rm scr}$ [Bohr$^{-1}$]")
    axes[0, 2].set_ylabel(r"$g_{ii}$")
    axes[0, 3].set_ylabel(r"$S_{ii}$")
    fig.suptitle(
        "Otter: Al KS-DFT versus Thomas–Fermi at 8.1 g cm$^{-3}$"
    )
    fig.tight_layout()
    otter_plotting.save_figure(fig, FIGURE_PATH, dpi=FIGURE_DPI)
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def print_metrics(metrics: list[dict[str, Any]]) -> None:
    print(
        f"{'rho':>6s} {'T[eV]':>7s} {'dZ(TF-KS)':>11s} "
        f"{'RMSE(g)':>11s} {'RMSE(S)':>11s}"
    )
    for row in metrics:
        delta_zbar = row.get(
            "delta_zbar_tf_minus_ksdft",
            row.get("delta_zbar_tf_minus_qm"),
        )
        print(
            f"{float(row['rho_g_cc']):6.2f} "
            f"{float(row['temperature_ev']):7.1f} "
            f"{float(delta_zbar):11.5f} "
            f"{float(row['gii_rmse_r_le_12']):11.5f} "
            f"{float(row['sii_rmse_k_le_6']):11.5f}"
        )


def main() -> None:
    manifest, data_dir = select_data_source()
    metrics, loaded = evaluate_states(manifest, data_dir=data_dir)
    write_metrics(metrics)
    plot_states(loaded)
    print_metrics(metrics)
    source_kind = "precomputed Otter results" if USE_PRECOMPUTED_DATA else (
        "newly recomputed Otter candidate results"
    )
    print(f"using {source_kind}: {data_dir}")
    print(f"saved figure: {FIGURE_PATH}")
    print(f"saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
