"""Validate and plot the curated ion-structure literature library.

This runner is offline with respect to the scientific solvers: it uses only
Otter's shared plotting style, verifies portable result checksums, converts
reference units explicitly, and recomputes pointwise interpolation metrics.
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


SHOW_FIGURES = False
ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / "ion_structure_library"
REFERENCE_DIR = (
    ROOT / "benchmarks" / "reference_data" / "ion_structure_library"
)
OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / "ion_structure_library"
MANIFEST_PATH = BASELINE_DIR / "manifest.json"
METRICS_PATH = OUTPUT_DIR / "metrics.csv"
FIGURE_SII_PATH = OUTPUT_DIR / "ion_structure_library_sii.png"
FIGURE_GII_PATH = OUTPUT_DIR / "ion_structure_library_gii.png"
BOHR_TO_ANGSTROM = 0.529177210903
SCHEMA = "otter_ion_structure_library_state_v1"


REFERENCE_SERIES: dict[str, tuple[dict[str, str], ...]] = {
    "al_gill_rho2p7_te5_ti5": (
        {
            "observable": "sii",
            "label": "Gill KS-PAMD",
            "file": "gill_et_al_2015/Sii_Al_T5ev_rho2.7_KS-PAMD_Gill.csv",
            "x_unit": "angstrom^-1",
            "role": "published_model",
        },
        {
            "observable": "sii",
            "label": "Gill TF-PAMD",
            "file": "gill_et_al_2015/Sii_Al_T5ev_rho2.7_TF-PAMD_Gill.csv",
            "x_unit": "angstrom^-1",
            "role": "published_model",
        },
        {
            "observable": "sii",
            "label": "Gill TF-DFT-MD",
            "file": "gill_et_al_2015/Sii_Al_T5ev_rho2.7_QMD_Gill.csv",
            "x_unit": "angstrom^-1",
            "role": "primary",
        },
    ),
    "al_clerouin_rho8p1_te10_ti10": (
        {
            "observable": "sii",
            "label": "Clérouin OFMD",
            "file": (
                "clerouin_et_al_2015/"
                "Jean_2015_Al_rho8.1_Te10.0_Ti10.csv"
            ),
            "x_unit": "angstrom^-1",
            "role": "primary",
        },
    ),
    "al_clerouin_rho8p1_te10_ti2": (
        {
            "observable": "sii",
            "label": "Clérouin OFMD",
            "file": (
                "clerouin_et_al_2015/"
                "Jean_2015_Al_rho8.1_Te10.0_Ti2.0_OFMD.csv"
            ),
            "x_unit": "angstrom^-1",
            "role": "primary",
        },
        {
            "observable": "sii",
            "label": "HNC-Y-SRR",
            "file": (
                "clerouin_et_al_2015/"
                "Jean_2015_Al_rho8.1_Te10.0_Ti2.0_SRR.csv"
            ),
            "x_unit": "angstrom^-1",
            "role": "published_model",
        },
    ),
    "be_wunsch_rho5p544_te13_ti13": (
        {
            "observable": "sii",
            "label": "Wünsch DFT-MD",
            "file": (
                "wunsch_et_al_2009/"
                "Sii_DFTMD_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom^-1",
            "role": "primary",
        },
        {
            "observable": "sii",
            "label": "HNC-Y-SRR",
            "file": (
                "wunsch_et_al_2009/"
                "Sii_HNC-YSRR_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom^-1",
            "role": "published_model",
        },
        {
            "observable": "sii",
            "label": "HNC-KK",
            "file": (
                "wunsch_et_al_2009/"
                "Sii_HNCKK_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom^-1",
            "role": "published_model",
        },
        {
            "observable": "sii",
            "label": "HNC-Y",
            "file": (
                "wunsch_et_al_2009/"
                "Sii_HNCY_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom^-1",
            "role": "published_model",
        },
        {
            "observable": "gii",
            "label": "Wünsch DFT-MD",
            "file": (
                "wunsch_et_al_2009/"
                "gii_DFTMD_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom",
            "role": "primary",
        },
        {
            "observable": "gii",
            "label": "HNC-Y-SRR",
            "file": (
                "wunsch_et_al_2009/"
                "gii_HNC-YSRR_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom",
            "role": "published_model",
        },
        {
            "observable": "gii",
            "label": "HNC-KK",
            "file": (
                "wunsch_et_al_2009/"
                "gii_HNC-KK_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom",
            "role": "published_model",
        },
        {
            "observable": "gii",
            "label": "HNC-Y",
            "file": (
                "wunsch_et_al_2009/"
                "gii_HNC-Y_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom",
            "role": "published_model",
        },
    ),
    "c_starrett_rho20_te50_ti50": (
        {
            "observable": "gii",
            "label": "Starrett PA-HNC",
            "file": (
                "starrett_saumon_2013/"
                "gii_C_20gcc_50.0ev_starrett.csv"
            ),
            "x_unit": "bohr",
            "role": "primary",
        },
    ),
}


STATE_TITLES = {
    "al_gill_rho2p7_te5_ti5": (
        r"Al: $\rho=2.7$ g cm$^{-3}$, $T_e=T_i=5$ eV"
    ),
    "al_clerouin_rho8p1_te10_ti10": (
        r"Al: $\rho=8.1$ g cm$^{-3}$, $T_e=T_i=10$ eV"
    ),
    "al_clerouin_rho8p1_te10_ti2": (
        r"Al: $\rho=8.1$ g cm$^{-3}$, $T_e=10$, $T_i=2$ eV"
    ),
    "be_wunsch_rho5p544_te13_ti13": (
        r"Be: $\rho=5.544$ g cm$^{-3}$, $T_e=T_i=13$ eV"
    ),
    "c_starrett_rho20_te50_ti50": (
        r"C: $\rho=20$ g cm$^{-3}$, $T_e=T_i=50$ eV"
    ),
}

OTTER_SERIES = {
    state_id: ((state_id, "Otter KS", "-"),)
    for state_id in REFERENCE_SERIES
}
OTTER_SERIES.update(
    {
        "al_clerouin_rho8p1_te10_ti10": (
            ("al_clerouin_rho8p1_te10_ti10", "Otter KS", "-"),
            ("al_clerouin_rho8p1_te10_ti10_tf", "Otter TF", "--"),
        ),
        "al_clerouin_rho8p1_te10_ti2": (
            ("al_clerouin_rho8p1_te10_ti2", "Otter KS", "-"),
            ("al_clerouin_rho8p1_te10_ti2_tf", "Otter TF", "--"),
        ),
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "otter_benchmark_manifest_v1":
        raise ValueError("Unsupported benchmark manifest schema.")
    if manifest.get("benchmark_id") != "ion_structure_library":
        raise ValueError("Unexpected benchmark identifier.")
    expected = {
        result_id
        for branches in OTTER_SERIES.values()
        for result_id, _, _ in branches
    }
    if {state["state_id"] for state in manifest["states"]} != expected:
        raise ValueError("Manifest and reference state maps differ.")
    return manifest


def load_baseline(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    if data["schema_version"].item() != SCHEMA:
        raise ValueError(f"Unsupported baseline schema in {path}.")
    for key, value in data.items():
        if value.dtype.hasobject:
            raise TypeError(f"Object dtype is forbidden: {path}:{key}")
        if value.dtype.kind in "fiu" and not np.all(np.isfinite(value)):
            raise ValueError(f"Non-finite numeric data: {path}:{key}")
    return data


def load_states(
    manifest: dict[str, Any],
    *,
    baseline_dir: Path = BASELINE_DIR,
    verify_checksums: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for state in manifest["states"]:
        path = (baseline_dir / state["baseline_file"]).resolve()
        if path.parent != baseline_dir.resolve():
            raise ValueError("Baseline path escapes its package directory.")
        if verify_checksums and baseline_dir.resolve() == BASELINE_DIR.resolve():
            if sha256_file(path) != state["baseline_sha256"]:
                raise RuntimeError(f"Checksum mismatch for {path}.")
        data = load_baseline(path)
        state_id = str(data["state_id"].item())
        if state_id != state["state_id"]:
            raise ValueError(f"State ID mismatch in {path}.")
        loaded[state_id] = data
    return loaded


def load_reference(series: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    path = (REFERENCE_DIR / series["file"]).resolve()
    if not path.is_relative_to(REFERENCE_DIR.resolve()):
        raise ValueError("Reference path escapes its package directory.")
    values = np.genfromtxt(path, delimiter=",", comments="#")
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"Expected two reference columns in {path}.")
    if not np.all(np.isfinite(values[:, :2])):
        raise ValueError(f"Non-finite reference values in {path}.")
    return values[:, 0], values[:, 1]


def _otter_curve(
    state: dict[str, np.ndarray],
    observable: str,
    x_unit: str,
) -> tuple[np.ndarray, np.ndarray]:
    if observable == "sii":
        x = np.asarray(state["k_bohr_inv"], dtype=float)
        y = np.asarray(state["sii_k"], dtype=float)
        if x_unit == "angstrom^-1":
            x = x / BOHR_TO_ANGSTROM
        elif x_unit != "bohr^-1":
            raise ValueError(f"Unsupported reciprocal unit {x_unit!r}.")
        return x, y
    if observable == "gii":
        x = np.asarray(state["r_bohr"], dtype=float)
        y = np.asarray(state["gii_r"], dtype=float)
        if x_unit == "angstrom":
            x = x * BOHR_TO_ANGSTROM
        elif x_unit != "bohr":
            raise ValueError(f"Unsupported radius unit {x_unit!r}.")
        return x, y
    raise ValueError(f"Unknown observable {observable!r}.")


def evaluate(
    states: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state_id, series_list in REFERENCE_SERIES.items():
        for result_id, model_label, _ in OTTER_SERIES[state_id]:
            state = states[result_id]
            for series in series_list:
                x_ref, y_ref = load_reference(series)
                x_otter, y_otter = _otter_curve(
                    state,
                    series["observable"],
                    series["x_unit"],
                )
                mask = (x_ref >= x_otter[0]) & (x_ref <= x_otter[-1])
                if not np.any(mask):
                    raise ValueError(
                        f"No overlap for {result_id}: {series['label']}"
                    )
                delta = (
                    np.interp(x_ref[mask], x_otter, y_otter) - y_ref[mask]
                )
                rows.append(
                    {
                        "state_id": state_id,
                        "otter_model": model_label,
                        "observable": series["observable"],
                        "reference": series["label"],
                        "role": series["role"],
                        "n_points": int(np.count_nonzero(mask)),
                        "rmse": float(np.sqrt(np.mean(delta**2))),
                        "mae": float(np.mean(np.abs(delta))),
                        "max_abs": float(np.max(np.abs(delta))),
                        "zbar_partition": float(state["zbar_partition"]),
                        "hnc_residual": float(state["hnc_best_residual"]),
                        "closure_mismatch": float(
                            state["hnc_closure_mismatch"]
                        ),
                    }
                )
    return rows


def write_metrics(rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    with METRICS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_metrics(rows: list[dict[str, Any]]) -> None:
    print(
        f"{'state':42s} {'obs':>3s} {'reference':19s} "
        f"{'RMSE':>10s} {'MAE':>10s} {'max':>10s}"
    )
    for row in rows:
        print(
            f"{row['state_id']:42s} {row['observable']:>3s} "
            f"{row['reference'][:19]:19s} "
            f"{row['rmse']:10.4e} {row['mae']:10.4e} "
            f"{row['max_abs']:10.4e}"
        )


def _plot_observable(
    states: dict[str, dict[str, np.ndarray]],
    *,
    observable: str,
    state_ids: tuple[str, ...],
    output_path: Path,
) -> Any:
    import matplotlib.pyplot as plt

    ncols = 2
    nrows = (len(state_ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=otter_plotting.grid_figsize(nrows, 2),
        squeeze=False,
    )
    marker_cycle = ("o", "s", "^", "x")
    reference_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for panel, state_id in enumerate(state_ids):
        ax = axes.ravel()[panel]
        series_for_observable = [
            item
            for item in REFERENCE_SERIES[state_id]
            if item["observable"] == observable
        ]
        display_unit = str(series_for_observable[0]["x_unit"])
        for model_index, (result_id, label, line_style) in enumerate(
            OTTER_SERIES[state_id]
        ):
            x_otter, y_otter = _otter_curve(
                states[result_id],
                observable,
                display_unit,
            )
            ax.plot(
                x_otter,
                y_otter,
                color="black" if model_index == 0 else "#D55E00",
                ls=line_style,
                lw=2.1,
                label=label,
            )
        reference_x: list[np.ndarray] = []
        for index, series in enumerate(series_for_observable):
            x_ref, y_ref = load_reference(series)
            reference_x.append(x_ref)
            marker = marker_cycle[index % len(marker_cycle)]
            if marker == "x":
                ax.scatter(
                    x_ref,
                    y_ref,
                    s=25,
                    marker=marker,
                    color=reference_colors[index % len(reference_colors)],
                    linewidths=1.2,
                    label=series["label"],
                    zorder=3,
                )
            else:
                ax.scatter(
                    x_ref,
                    y_ref,
                    s=25,
                    marker=marker,
                    facecolors="none",
                    edgecolors=reference_colors[
                        index % len(reference_colors)
                    ],
                    linewidths=1.2,
                    label=series["label"],
                    zorder=3,
                )
        ax.set_title(STATE_TITLES[state_id], fontsize=10)
        ax.set_ylabel(r"$S_{ii}(k)$" if observable == "sii" else r"$g_{ii}(r)$")
        if display_unit == "angstrom^-1":
            ax.set_xlabel(r"$k$ [$\mathrm{\AA}^{-1}$]")
        elif display_unit == "angstrom":
            ax.set_xlabel(r"$r$ [$\mathrm{\AA}$]")
        else:
            ax.set_xlabel(r"$r$ [Bohr]")
        # The stored Otter arrays retain the requested 20 Bohr / 20 Bohr^-1
        # domain, which is useful for downstream work but much wider than
        # most digitized publication panels.  Frame each visual comparison on
        # the actual reference domain so the agreement is legible.
        x_reference_all = np.concatenate(reference_x)
        x_span = float(np.max(x_reference_all) - np.min(x_reference_all))
        x_margin = max(0.03 * x_span, 1.0e-6)
        ax.set_xlim(
            max(0.0, float(np.min(x_reference_all)) - x_margin),
            float(np.max(x_reference_all)) + x_margin,
        )
        if observable == "sii":
            ax.axhline(1.0, color="0.55", lw=0.8, ls=":")
        else:
            ax.axhline(1.0, color="0.55", lw=0.8, ls=":")
        ax.grid(alpha=0.2)
        ax.legend(fontsize="small")
    for panel in range(len(state_ids), axes.size):
        axes.ravel()[panel].set_visible(False)
    fig.suptitle(
        "Otter QOZ/HNC versus curated literature curves",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    otter_plotting.save_figure(fig, output_path, dpi=200)
    return fig


def plot_all(
    states: dict[str, dict[str, np.ndarray]],
) -> tuple[Any, Any]:
    sii_figure = _plot_observable(
        states,
        observable="sii",
        state_ids=(
            "al_gill_rho2p7_te5_ti5",
            "al_clerouin_rho8p1_te10_ti10",
            "al_clerouin_rho8p1_te10_ti2",
            "be_wunsch_rho5p544_te13_ti13",
        ),
        output_path=FIGURE_SII_PATH,
    )
    gii_figure = _plot_observable(
        states,
        observable="gii",
        state_ids=(
            "be_wunsch_rho5p544_te13_ti13",
            "c_starrett_rho20_te50_ti50",
        ),
        output_path=FIGURE_GII_PATH,
    )
    return sii_figure, gii_figure


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(OUTPUT_DIR / ".cache"))
    if not SHOW_FIGURES:
        import matplotlib

        matplotlib.use("Agg")
    otter_plotting.set_style("docs", palette="deep_science")
    manifest = load_manifest()
    states = load_states(manifest)
    rows = evaluate(states)
    write_metrics(rows)
    plot_all(states)
    print_metrics(rows)
    print(f"saved {FIGURE_SII_PATH.relative_to(ROOT)}")
    print(f"saved {FIGURE_GII_PATH.relative_to(ROOT)}")

    if SHOW_FIGURES:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
