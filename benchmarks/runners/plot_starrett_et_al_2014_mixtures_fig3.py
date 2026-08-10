"""Redraw a CH1.36 Starrett et al. mixtures Figure 3 benchmark dataset.

By default this runner reads the immutable accepted precursor package.  The
``data_dir``/``manifest_path`` arguments also let it render a candidate
produced by ``regenerate_starrett_et_al_2014_mixtures_fig3.py``.  It never
invokes the average-atom or QOZ/HNC solvers itself, so it remains suitable for
documentation generation and quick data validation.

Reference
---------
C. E. Starrett, D. Saumon, J. Daligault, and S. Hamel, Physical Review E
90, 033110 (2014), Figure 3. DOI: 10.1103/PhysRevE.90.033110.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import otter.plotting as otter_plotting


# User-facing plotting controls live here; no terminal arguments are needed.
SHOW_FIGURE = False
VERIFY_SHA256 = True
FIGURE_DPI = 220
R_COMPARE_MAX_BOHR = 6.0

BENCHMARKS_DIR = Path(__file__).resolve().parents[1]
BASELINE_DIR = (
    BENCHMARKS_DIR / "baselines" / "starrett_et_al_2014_mixtures_fig3"
)
OUTPUT_DIR = (
    BENCHMARKS_DIR / "outputs" / "starrett_et_al_2014_mixtures_fig3"
)
MANIFEST_PATH = BASELINE_DIR / "manifest.json"
FIGURE_PATH = OUTPUT_DIR / "fig3_ch1p36_offline_overlay.png"
METRICS_PATH = OUTPUT_DIR / "fig3_ch1p36_offline_metrics.csv"

PAIR_ORDER = ("CC", "CH", "HH")
PAIR_COLS = {"CC": (0, 1), "CH": (4, 5), "HH": (2, 3)}
PAIR_COLORS = {"CC": "black", "CH": "red", "HH": "limegreen"}
METRIC_FIELDS = (
    "rho_g_cc",
    "temperature_kk",
    "temperature_ev",
    "pair",
    "n_reference",
    "rmse",
    "mae",
    "max_abs",
    "bias",
    "r_peak_ref_bohr",
    "g_peak_ref",
    "r_peak_model_bohr",
    "g_peak_model",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(
    *,
    manifest_path: Path = MANIFEST_PATH,
    expected_state_count: int = 9,
) -> dict[str, Any]:
    """Load and minimally validate an accepted or candidate manifest."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in {
        "otter_benchmark_manifest_v1",
        "otter_starrett_fig3_recompute_manifest_v1",
    }:
        raise ValueError("Unsupported Starrett Figure 3 manifest schema.")
    if manifest.get("benchmark_id") != (
        "starrett_et_al_2014_mixtures_fig3_ch1p36"
    ):
        raise ValueError("Manifest belongs to a different benchmark.")
    states = manifest.get("states", [])
    if len(states) != int(expected_state_count):
        raise ValueError(
            f"Expected {int(expected_state_count)} thermodynamic states, "
            f"found {len(states)}."
        )
    return manifest


def resolve_state_paths(
    state: dict[str, Any],
    *,
    data_dir: Path = BASELINE_DIR,
) -> tuple[Path, Path]:
    """Resolve one state's reference and numerical-result paths."""
    root = Path(data_dir)
    reference = (root / str(state["reference_file"])).resolve()
    result_key = "result_file" if "result_file" in state else "baseline_file"
    result = (root / str(state[result_key])).resolve()
    return reference, result


def verify_state_files(
    state: dict[str, Any],
    *,
    data_dir: Path = BASELINE_DIR,
) -> None:
    """Verify the reference and numerical-result checksums for one state."""
    reference, result = resolve_state_paths(state, data_dir=data_dir)
    result_checksum_key = (
        "result_sha256" if "result_sha256" in state else "baseline_sha256"
    )
    expected = (
        (reference, str(state["reference_sha256"])),
        (result, str(state[result_checksum_key])),
    )
    for path, checksum in expected:
        actual = sha256_file(path)
        if actual != checksum:
            raise RuntimeError(
                f"Checksum mismatch for {path}: expected {checksum}, got {actual}"
            )


def load_reference(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read the two-header, six-column digitized CSV representation."""
    data = np.genfromtxt(path, delimiter=",", skip_header=2)
    data = np.atleast_2d(np.asarray(data, dtype=float))
    if data.shape[1] != 6:
        raise ValueError(f"Expected six numeric columns in {path}, got {data.shape}.")

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pair in PAIR_ORDER:
        r_col, g_col = PAIR_COLS[pair]
        r = data[:, r_col]
        g = data[:, g_col]
        mask = np.isfinite(r) & np.isfinite(g) & (r >= 0.0)
        r = r[mask]
        g = g[mask]
        order = np.argsort(r)
        r = r[order]
        g = g[order]
        unique_r = np.unique(r)
        curves[pair] = (
            unique_r,
            np.asarray([np.mean(g[r == value]) for value in unique_r]),
        )
    return curves


def load_result(path: Path) -> dict[str, np.ndarray]:
    """Load and normalize one accepted precursor or Otter candidate."""
    with np.load(path, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    object_keys = [key for key, value in result.items() if value.dtype.hasobject]
    if object_keys:
        raise TypeError(f"Object arrays are forbidden in {path}: {object_keys}")
    schema = str(result["schema_version"].item())
    if schema == "otter_starrett_mixtures_fig3_baseline_v1":
        labels = tuple(str(value) for value in result["pair_labels"])
        if labels != PAIR_ORDER:
            raise ValueError(f"Unexpected pair order in {path}: {labels}")
        return result
    if schema == "otter_state_v1":
        symbols = tuple(str(value) for value in result["species_symbols"])
        if symbols != ("C", "H"):
            raise ValueError(f"Unexpected species order in {path}: {symbols}")
        gij = np.asarray(result["gij_r"], dtype=float)
        if gij.ndim != 3 or gij.shape[:2] != (2, 2):
            raise ValueError(f"Unexpected gij_r shape in {path}: {gij.shape}")
        return {
            **result,
            "r_bohr": np.asarray(result["r_bohr"], dtype=float),
            "g_ab": np.asarray((gij[0, 0], gij[0, 1], gij[1, 1])),
            "pair_labels": np.asarray(PAIR_ORDER),
        }
    raise ValueError(f"Unsupported numerical-result schema in {path}: {schema}")


def load_baseline(path: Path) -> dict[str, np.ndarray]:
    """Backward-compatible alias for :func:`load_result`."""
    return load_result(path)


def curve_metrics(
    r_model: np.ndarray,
    g_model: np.ndarray,
    r_reference: np.ndarray,
    g_reference: np.ndarray,
) -> dict[str, float | int]:
    """Return interpolation-based errors and first-peak diagnostics."""
    mask = (
        np.isfinite(r_reference)
        & np.isfinite(g_reference)
        & (r_reference >= max(0.0, float(r_model[0])))
        & (
            r_reference
            <= min(R_COMPARE_MAX_BOHR, float(r_model[-1]))
        )
    )
    r_ref = r_reference[mask]
    g_ref = g_reference[mask]
    if r_ref.size < 3:
        raise ValueError("Too few overlapping digitized reference points.")
    predicted = np.interp(r_ref, r_model, g_model)
    delta = predicted - g_ref

    ref_peak_indices = np.flatnonzero(
        (r_ref >= 0.2) & (r_ref <= R_COMPARE_MAX_BOHR)
    )
    model_peak_indices = np.flatnonzero(
        (r_model >= 0.2) & (r_model <= R_COMPARE_MAX_BOHR)
    )
    i_ref = ref_peak_indices[np.argmax(g_ref[ref_peak_indices])]
    i_model = model_peak_indices[np.argmax(g_model[model_peak_indices])]
    return {
        "n_reference": int(r_ref.size),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(np.abs(delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "bias": float(np.mean(delta)),
        "r_peak_ref_bohr": float(r_ref[i_ref]),
        "g_peak_ref": float(g_ref[i_ref]),
        "r_peak_model_bohr": float(r_model[i_model]),
        "g_peak_model": float(g_model[i_model]),
    }


def evaluate_states(
    manifest: dict[str, Any],
    *,
    data_dir: Path = BASELINE_DIR,
) -> tuple[list[dict[str, Any]], dict[tuple[float, int], dict[str, Any]]]:
    """Load all nine states and recompute their reference metrics."""
    metrics: list[dict[str, Any]] = []
    loaded: dict[tuple[float, int], dict[str, Any]] = {}
    for state in manifest["states"]:
        if VERIFY_SHA256:
            verify_state_files(state, data_dir=data_dir)
        reference_path, baseline_path = resolve_state_paths(
            state,
            data_dir=data_dir,
        )
        reference = load_reference(reference_path)
        baseline = load_result(baseline_path)
        rho = float(state["rho_g_cc"])
        temperature_kk = int(state["temperature_kk"])
        r_model = np.asarray(baseline["r_bohr"], dtype=float)
        g_ab = np.asarray(baseline["g_ab"], dtype=float)
        loaded[(rho, temperature_kk)] = {
            "reference": reference,
            "baseline": baseline,
        }
        for pair_index, pair in enumerate(PAIR_ORDER):
            r_ref, g_ref = reference[pair]
            metrics.append(
                {
                    "rho_g_cc": rho,
                    "temperature_kk": temperature_kk,
                    "temperature_ev": float(state["temperature_ev"]),
                    "pair": pair,
                    **curve_metrics(
                        r_model, g_ab[pair_index], r_ref, g_ref
                    ),
                }
            )
    return metrics, loaded


def plot_states(
    loaded: dict[tuple[float, int], dict[str, Any]],
    *,
    figure_path: Path = FIGURE_PATH,
    model_label: str = "Otter",
) -> None:
    """Create the 3x3 Figure 3-style overlay from already loaded data."""
    # Keep Matplotlib and font caches inside the already ignored output tree.
    # Import lazily so data-only tests do not create plotting artifacts at all.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib")
    )
    os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / ".cache"))
    import matplotlib

    if not SHOW_FIGURE:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    otter_plotting.set_style("docs", palette="deep_science")
    densities = (2.94, 5.0, 15.0)
    temperatures = (20, 50, 100)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(14.4, 10.95),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, rho in enumerate(densities):
        for column, temperature_kk in enumerate(temperatures):
            ax = axes[row, column]
            payload = loaded[(rho, temperature_kk)]
            reference = payload["reference"]
            baseline = payload["baseline"]
            r_model = np.asarray(baseline["r_bohr"], dtype=float)
            g_ab = np.asarray(baseline["g_ab"], dtype=float)
            for pair_index, pair in enumerate(PAIR_ORDER):
                r_ref, g_ref = reference[pair]
                color = PAIR_COLORS[pair]
                ax.plot(
                    r_ref,
                    g_ref,
                    color=color,
                    marker="o",
                    markersize=6.0,
                    markerfacecolor="none",
                    markeredgewidth=1.1,
                    linestyle="none",
                    alpha=0.8,
                )
                ax.plot(r_model, g_ab[pair_index], color=color, lw=1.8)
            ax.set_xlim(0.0, R_COMPARE_MAX_BOHR)
            ax.set_ylim(-0.05, 2.0)
            ax.grid(alpha=0.22)
            if row == 0:
                ax.set_title(f"{temperature_kk} kK")
            if column == 0:
                ax.set_ylabel(
                    rf"$\rho={rho:g}$ g cm$^{{-3}}$"
                    + "\n"
                    + r"$g_{ab}(r)$"
                )
            if row == 2:
                ax.set_xlabel(r"$r$ [$a_{\rm B}$]")

    pair_handles = [
        Line2D([], [], color=PAIR_COLORS[pair], lw=2.2, label=pair)
        for pair in PAIR_ORDER
    ]
    source_handles = [
        Line2D(
            [],
            [],
            color="0.3",
            marker="o",
            markersize=7.5,
            markeredgewidth=1.2,
            markerfacecolor="none",
            linestyle="none",
            label="Starrett Fig. 3 digitized",
        ),
        Line2D(
            [],
            [],
            color="0.3",
            lw=2.0,
            label=str(model_label),
        ),
    ]
    fig.legend(
        handles=pair_handles + source_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.952),
    )
    fig.suptitle(
        r"CH$_{1.36}$: offline Starrett et al. (2014) mixtures benchmark",
        y=0.992,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.065,
        top=0.885,
        wspace=0.055,
        hspace=0.09,
    )
    otter_plotting.save_figure(fig, figure_path, dpi=FIGURE_DPI)
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


def write_metrics(
    metrics: list[dict[str, Any]],
    *,
    metrics_path: Path = METRICS_PATH,
) -> None:
    """Write independently recomputed metrics beside the generated figure."""
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)


def print_metrics(metrics: list[dict[str, Any]]) -> None:
    """Print a compact benchmark table."""
    print(
        f"{'rho':>6s} {'T[kK]':>6s} {'pair':>4s} "
        f"{'RMSE':>10s} {'MAE':>10s} {'max|d|':>10s}"
    )
    for row in metrics:
        print(
            f"{float(row['rho_g_cc']):6.2f} "
            f"{int(row['temperature_kk']):6d} "
            f"{str(row['pair']):>4s} "
            f"{float(row['rmse']):10.5f} "
            f"{float(row['mae']):10.5f} "
            f"{float(row['max_abs']):10.5f}"
        )


def main(
    *,
    data_dir: Path = BASELINE_DIR,
    manifest_path: Path | None = None,
    figure_path: Path = FIGURE_PATH,
    metrics_path: Path = METRICS_PATH,
) -> None:
    """Validate data, redraw the comparison, and report fresh metrics."""
    data_dir = Path(data_dir)
    selected_manifest = (
        data_dir / "manifest.json"
        if manifest_path is None
        else Path(manifest_path)
    )
    manifest = load_manifest(manifest_path=selected_manifest)
    metrics, loaded = evaluate_states(manifest, data_dir=data_dir)
    model_label = (
        "Otter (recomputed)"
        if manifest.get("schema_version")
        == "otter_starrett_fig3_recompute_manifest_v1"
        else "Otter"
    )
    plot_states(
        loaded,
        figure_path=figure_path,
        model_label=model_label,
    )
    write_metrics(metrics, metrics_path=metrics_path)
    print_metrics(metrics)
    print(f"saved figure: {figure_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
