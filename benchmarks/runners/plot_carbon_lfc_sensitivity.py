"""Validate, optionally regenerate, and plot carbon LFC sensitivity.

The default path reads the reviewed, checksummed Otter v2 results at
``rho=5 g cm^-3``.  Set ``USE_PRECOMPUTED_DATA = False`` below to run the
strict producer.  Recalculation writes only below ``benchmarks/outputs`` and
never promotes a candidate into the reviewed data package.
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


# User-editable execution mode.  ``True`` is fast and deterministic;
# ``False`` recomputes the two electronic states and all five QOZ/HNC paths.
USE_PRECOMPUTED_DATA = True

SHOW_FIGURE = False
FIGURE_DPI = 200
R_PLOT_MAX_BOHR = 12.0
K_PLOT_MAX_BOHR_INV = 6.0
R_METRIC_MAX_BOHR = 20.0

BENCHMARKS_DIR = Path(__file__).resolve().parents[1]
REVIEWED_DIR = BENCHMARKS_DIR / "baselines" / "carbon_lfc_sensitivity"
BASELINE_DIR = REVIEWED_DIR
OUTPUT_DIR = BENCHMARKS_DIR / "outputs" / "carbon_lfc_sensitivity"
RECOMPUTED_DIR = OUTPUT_DIR / "recomputed"
MANIFEST_PATH = REVIEWED_DIR / "manifest.json"

# Maintainers may point this at an ignored candidate directory during review.
# End users should leave it as ``None`` so the accepted package is used.
PRECOMPUTED_DATA_DIR: Path | None = None

FIGURE_PATH = OUTPUT_DIR / "carbon_lfc_sensitivity.png"
METRICS_PATH = OUTPUT_DIR / "carbon_lfc_sensitivity_metrics.csv"
MODEL_ORDER = (
    "none",
    "hubbard",
    "utsumiichimaru",
    "chabrier1990",
    "gregori2007",
)
MODEL_LABELS = {
    "none": "RPA",
    "hubbard": "Hubbard",
    "utsumiichimaru": "Utsumi–Ichimaru",
    "chabrier1990": "Chabrier 1990",
    "gregori2007": "Gregori 2007",
}
EXPECTED_RHO_G_CC = 5.0
EXPECTED_TEMPERATURES_EV = (2.0, 100.0)
STATE_SCHEMA = "otter_carbon_lfc_sensitivity_state_v2"
MANIFEST_SCHEMA = "otter_benchmark_manifest_v2"
LEGACY_STATE_SCHEMA = "otter_carbon_lfc_sensitivity_baseline_v1"
LEGACY_MANIFEST_SCHEMA = "otter_benchmark_manifest_v1"
METRIC_FIELDS = (
    "rho_g_cc",
    "temperature_ev",
    "model",
    "rms_dg_vs_chabrier_r_le_20",
    "max_abs_dg_vs_chabrier_r_le_20",
    "rms_ds_vs_chabrier_k_le_6",
    "max_abs_ds_vs_chabrier_k_le_6",
    "rms_dv_vs_chabrier_ha_bohr3_k_le_6",
    "max_abs_dv_vs_chabrier_ha_bohr3_k_le_6",
    "g_peak",
    "r_peak_bohr",
    "s_kmin",
    "v_kmin_ha_bohr3",
    "gee_max",
    "hnc_output_residual",
    "closure_transform_max_abs",
    "data_file",
)
LEGACY_METRIC_FIELDS = (*METRIC_FIELDS[:-1], "baseline_file")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load a v2 manifest, or the retained v1 package for compatibility."""
    manifest_path = MANIFEST_PATH if path is None else Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema_version")
    if schema == LEGACY_MANIFEST_SCHEMA:
        if manifest.get("benchmark_id") != "carbon_lfc_sensitivity":
            raise ValueError("Unexpected carbon LFC benchmark identifier.")
        if tuple(manifest.get("models", ())) != MODEL_ORDER:
            raise ValueError("Unexpected legacy manifest LFC model order.")
        return manifest
    if schema != MANIFEST_SCHEMA:
        raise ValueError(
            "The rho=5 carbon gallery requires the v2 manifest. The legacy "
            "rho=3.7 package is intentionally not substituted."
        )
    if manifest.get("benchmark_id") != "carbon_lfc_sensitivity":
        raise ValueError("Unexpected carbon LFC benchmark identifier.")
    configuration = dict(manifest.get("configuration", {}))
    if tuple(configuration.get("models", ())) != MODEL_ORDER:
        raise ValueError("Unexpected manifest LFC model order.")
    if not np.isclose(
        float(configuration.get("rho_g_cc", np.nan)),
        EXPECTED_RHO_G_CC,
    ):
        raise ValueError("Every v2 carbon state must use rho=5 g cm^-3.")
    states = list(manifest.get("states", ()))
    temperatures = tuple(float(state["temperature_ev"]) for state in states)
    if temperatures != EXPECTED_TEMPERATURES_EV:
        raise ValueError(
            f"Expected temperatures {EXPECTED_TEMPERATURES_EV}, got "
            f"{temperatures}."
        )
    return manifest


def resolve_data(state: dict[str, Any], data_dir: Path) -> Path:
    """Resolve a manifest-relative data path without allowing traversal."""
    directory = Path(data_dir).resolve()
    path = (directory / str(state["data_file"])).resolve()
    if path.parent != directory:
        raise ValueError("Data path escapes its package directory.")
    return path


def load_state(path: Path) -> dict[str, np.ndarray]:
    """Load one pickle-free v2 state and enforce structural invariants."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    object_keys = [key for key, value in result.items() if value.dtype.hasobject]
    if object_keys:
        raise TypeError(f"Object arrays are forbidden in {path}: {object_keys}")
    if result["schema_version"].item() != STATE_SCHEMA:
        raise ValueError(f"Unsupported carbon state schema in {path}.")
    if tuple(str(item) for item in result["model_labels"]) != MODEL_ORDER:
        raise ValueError(f"Unexpected model order in {path}.")
    for name, value in result.items():
        if value.dtype.kind in "fiu" and not np.all(np.isfinite(value)):
            raise ValueError(f"Non-finite numeric array {name!r} in {path}.")
    if not bool(result["aa_stage2_converged"].item()):
        raise ValueError(f"Unconverged AA stage 2 in {path}.")
    if not bool(result["aa_ext_converged"].item()):
        raise ValueError(f"Unconverged external AA stage in {path}.")
    if str(result["threshold_state_status"].item()).lower() == "unresolved":
        raise ValueError(f"Unresolved threshold-state representation in {path}.")
    r = np.asarray(result["r_bohr"], dtype=float)
    k = np.asarray(result["k_bohr_inv"], dtype=float)
    r_e = np.asarray(result["electronic_r_bohr"], dtype=float)
    if (
        r.ndim != 1
        or k.ndim != 1
        or r_e.ndim != 1
        or np.any(np.diff(r) <= 0.0)
        or np.any(np.diff(k) <= 0.0)
        or np.any(np.diff(r_e) <= 0.0)
    ):
        raise ValueError(f"Coordinates must be increasing 1D arrays in {path}.")
    if r[-1] > 20.0 or k[-1] > 20.0 or r_e[-1] > 20.0:
        raise ValueError(f"Retained carbon data exceed the v2 20-Bohr limit in {path}.")
    expected_r_shape = (len(MODEL_ORDER), r.size)
    expected_k_shape = (len(MODEL_ORDER), k.size)
    for key in ("gii_r", "vii_r_ha"):
        if result[key].shape != expected_r_shape:
            raise ValueError(f"Unexpected {key} shape in {path}.")
    for key in ("sii_k", "vii_k_ha_bohr3", "chi_ee_k", "gee_k"):
        if result[key].shape != expected_k_shape:
            raise ValueError(f"Unexpected {key} shape in {path}.")
    if np.max(result["hnc_output_residual"]) > 1.0e-5:
        raise ValueError(f"HNC output residual exceeds the v2 tolerance in {path}.")
    if np.max(result["closure_transform_max_abs"]) > 1.0e-4:
        raise ValueError(f"Closure-transform mismatch exceeds tolerance in {path}.")
    np.testing.assert_allclose(
        result["zbar_qoz"],
        float(result["zbar_partition"]),
        rtol=0.0,
        atol=2.0e-12,
    )
    return result


def load_baseline(path: Path) -> dict[str, np.ndarray]:
    """Load the retained v1 package used by historical regression tests."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    if result["schema_version"].item() != LEGACY_STATE_SCHEMA:
        raise ValueError(f"Unsupported legacy carbon state schema in {path}.")
    if tuple(str(item) for item in result["model_labels"]) != MODEL_ORDER:
        raise ValueError(f"Unexpected legacy model order in {path}.")
    return result


def _rmse(delta: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(delta, dtype=float) ** 2)))


def state_metrics(
    data: dict[str, np.ndarray],
    filename: str,
    *,
    file_field: str = "data_file",
) -> list[dict[str, Any]]:
    """Return one metric row per LFC model."""
    r = np.asarray(data["r_bohr"], dtype=float)
    k = np.asarray(data["k_bohr_inv"], dtype=float)
    gii = np.asarray(data["gii_r"], dtype=float)
    sii = np.asarray(data["sii_k"], dtype=float)
    vii = np.asarray(data["vii_k_ha_bohr3"], dtype=float)
    gee = np.asarray(data["gee_k"], dtype=float)
    reference_index = MODEL_ORDER.index("chabrier1990")
    r_mask = r <= R_METRIC_MAX_BOHR
    k_mask = k <= K_PLOT_MAX_BOHR_INV
    peak_candidates = np.flatnonzero((r >= 0.2) & (r <= R_PLOT_MAX_BOHR))
    rows: list[dict[str, Any]] = []
    for model_index, model in enumerate(MODEL_ORDER):
        dg = gii[model_index] - gii[reference_index]
        ds = sii[model_index] - sii[reference_index]
        dv = vii[model_index] - vii[reference_index]
        peak_index = peak_candidates[
            np.argmax(gii[model_index, peak_candidates])
        ]
        rows.append(
            {
                "rho_g_cc": float(data["rho_g_cc"]),
                "temperature_ev": float(data["temperature_ev"]),
                "model": model,
                "rms_dg_vs_chabrier_r_le_20": _rmse(dg[r_mask]),
                "max_abs_dg_vs_chabrier_r_le_20": float(
                    np.max(np.abs(dg[r_mask]))
                ),
                "rms_ds_vs_chabrier_k_le_6": _rmse(ds[k_mask]),
                "max_abs_ds_vs_chabrier_k_le_6": float(
                    np.max(np.abs(ds[k_mask]))
                ),
                "rms_dv_vs_chabrier_ha_bohr3_k_le_6": _rmse(dv[k_mask]),
                "max_abs_dv_vs_chabrier_ha_bohr3_k_le_6": float(
                    np.max(np.abs(dv[k_mask]))
                ),
                "g_peak": float(gii[model_index, peak_index]),
                "r_peak_bohr": float(r[peak_index]),
                "s_kmin": float(sii[model_index, 0]),
                "v_kmin_ha_bohr3": float(vii[model_index, 0]),
                "gee_max": float(np.max(gee[model_index])),
                "hnc_output_residual": float(
                    data["hnc_output_residual"][model_index]
                ),
                "closure_transform_max_abs": float(
                    data["closure_transform_max_abs"][model_index]
                ),
                file_field: filename,
            }
        )
    return rows


def evaluate_states(
    manifest: dict[str, Any],
    *,
    data_dir: Path | None = None,
    verify_checksums: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    """Verify, load, and evaluate every state in *manifest*."""
    metrics: list[dict[str, Any]] = []
    loaded: list[dict[str, np.ndarray]] = []
    legacy = manifest.get("schema_version") == LEGACY_MANIFEST_SCHEMA
    directory = (
        REVIEWED_DIR
        if data_dir is None
        else Path(data_dir).expanduser().resolve()
    )
    for state in manifest["states"]:
        file_key = "baseline_file" if legacy else "data_file"
        hash_key = "baseline_sha256" if legacy else "data_sha256"
        path = resolve_data({**state, "data_file": state[file_key]}, directory)
        if verify_checksums and sha256_file(path) != state[hash_key]:
            raise RuntimeError(f"Checksum mismatch for {path}.")
        data = load_baseline(path) if legacy else load_state(path)
        if not np.isclose(float(data["rho_g_cc"]), state["rho_g_cc"]):
            raise ValueError(f"Density metadata mismatch in {path}.")
        if not np.isclose(
            float(data["temperature_ev"]),
            state["temperature_ev"],
        ):
            raise ValueError(f"Temperature metadata mismatch in {path}.")
        loaded.append(data)
        metrics.extend(
            state_metrics(
                data,
                path.name,
                file_field=("baseline_file" if legacy else "data_file"),
            )
        )
    return metrics, loaded


def _load_module(name: str, path: Path) -> ModuleType:
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
    """Select checksummed data or perform an explicit strict recomputation."""
    if use_precomputed_data:
        data_dir = (
            REVIEWED_DIR
            if precomputed_data_dir is None
            else Path(precomputed_data_dir).expanduser().resolve()
        )
        manifest_path = data_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No v2 carbon data at {manifest_path}. Run with "
                "USE_PRECOMPUTED_DATA = False to generate a candidate."
            )
    else:
        producer = _load_module(
            "otter_carbon_lfc_regenerator",
            BENCHMARKS_DIR
            / "runners"
            / "regenerate_carbon_lfc_sensitivity.py",
        )
        producer.regenerate()
        data_dir = Path(producer.OUTPUT_DIR).resolve()
        manifest_path = data_dir / "manifest.json"
    return load_manifest(manifest_path), data_dir


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
    """Write the compact offline comparison figure."""
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / ".cache"))
    import matplotlib

    if not SHOW_FIGURE:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    otter_plotting.set_style("docs", palette="deep_science")
    fig, axes = plt.subplots(
        len(loaded),
        5,
        figsize=(18.0, 3.6 * len(loaded)),
        squeeze=False,
    )
    for row, data in enumerate(loaded):
        r_e = np.asarray(data["electronic_r_bohr"], dtype=float)
        weight = 4.0 * np.pi * r_e**2
        electronic_mask = r_e <= R_PLOT_MAX_BOHR
        for values, label in (
            (data["n_bound_bohr3"], r"$n_{\rm bound}$"),
            (data["n_ion_bohr3"], r"$n_{\rm ion}$"),
            (data["n_scr_bohr3"], r"$n_{\rm scr}$"),
        ):
            axes[row, 0].plot(
                r_e[electronic_mask],
                (weight * np.asarray(values, dtype=float))[electronic_mask],
                lw=1.35,
                label=label,
            )
        r = np.asarray(data["r_bohr"], dtype=float)
        k = np.asarray(data["k_bohr_inv"], dtype=float)
        for model_index, model in enumerate(MODEL_ORDER):
            label = MODEL_LABELS[model]
            axes[row, 1].plot(k, data["gee_k"][model_index], lw=1.35, label=label)
            axes[row, 2].plot(
                k,
                data["vii_k_ha_bohr3"][model_index],
                lw=1.35,
                label=label,
            )
            axes[row, 3].plot(r, data["gii_r"][model_index], lw=1.35, label=label)
            axes[row, 4].plot(k, data["sii_k"][model_index], lw=1.35, label=label)
        state_label = (
            f"{float(data['rho_g_cc']):g} g cm$^{{-3}}$, "
            f"{float(data['temperature_ev']):g} eV"
        )
        axes[row, 0].set_ylabel(
            state_label + "\n" + r"$4\pi r^2 n(r)$ [Bohr$^{-1}$]"
        )
        for column, maximum in enumerate(
            (
                R_PLOT_MAX_BOHR,
                K_PLOT_MAX_BOHR_INV,
                K_PLOT_MAX_BOHR_INV,
                R_PLOT_MAX_BOHR,
                K_PLOT_MAX_BOHR_INV,
            )
        ):
            axes[row, column].set_xlim(0.0, maximum)
            axes[row, column].grid(alpha=0.22)
        axes[row, 4].axhline(1.0, color="0.45", lw=0.8, ls=":")
    for axis, title in zip(
        axes[0],
        (
            "shared electronic densities",
            r"$G_{ee}(k)$",
            r"$V_{ii}(k)$",
            r"$g_{ii}(r)$",
            r"$S_{ii}(k)$",
        ),
        strict=True,
    ):
        axis.set_title(title)
    axes[0, 0].legend(fontsize=8)
    axes[0, 4].legend(fontsize=7.5)
    for axis, label in zip(
        axes[-1],
        (
            r"$r$ [Bohr]",
            r"$k$ [Bohr$^{-1}$]",
            r"$k$ [Bohr$^{-1}$]",
            r"$r$ [Bohr]",
            r"$k$ [Bohr$^{-1}$]",
        ),
        strict=True,
    ):
        axis.set_xlabel(label)
    fig.suptitle(
        r"Otter carbon at 5 g cm$^{-3}$: finite-temperature LFC sensitivity"
    )
    fig.tight_layout()
    otter_plotting.save_figure(fig, FIGURE_PATH, dpi=FIGURE_DPI)
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def print_metrics(metrics: list[dict[str, Any]]) -> None:
    print(
        f"{'T[eV]':>7s} {'model':>18s} {'max|dg|':>11s} "
        f"{'max|dS|':>11s} {'max|dV|[Ha Bohr^3]':>20s}"
    )
    for row in metrics:
        print(
            f"{float(row['temperature_ev']):7.1f} "
            f"{str(row['model']):>18s} "
            f"{float(row['max_abs_dg_vs_chabrier_r_le_20']):11.5f} "
            f"{float(row['max_abs_ds_vs_chabrier_k_le_6']):11.5f} "
            f"{float(row['max_abs_dv_vs_chabrier_ha_bohr3_k_le_6']):20.5f}"
        )


def main() -> None:
    manifest, data_dir = select_data_source()
    metrics, loaded = evaluate_states(manifest, data_dir=data_dir)
    write_metrics(metrics)
    plot_states(loaded)
    print_metrics(metrics)
    print(f"saved figure: {FIGURE_PATH}")
    print(f"saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
