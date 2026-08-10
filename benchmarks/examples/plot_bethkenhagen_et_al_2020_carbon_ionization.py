r"""
Bethkenhagen et al. carbon-ionization context
==============================================

This definition-aware benchmark places Otter's carbon average-atom
ionization diagnostics beside the model-dependent
:math:`Z^{\rm free}` curves in Figure 3(a) of
:cite:t:`BethkenhagenEtAl2020` at :math:`T_e=100` eV.  The published
coordinates are user-supplied digitizations of the open-access figure and
are verified against the checksums in their reference-data manifest.

The quantities are not interchangeable.  Otter reports
:math:`\bar Z=Z-Q_{\rm ion}(R_{\rm WS})` and
:math:`Z^*=n_e^0/n_i`, whereas the paper's curves use several
model-dependent definitions.  In particular, its DFT-MD values use the
conduction-band conductivity and the Thomas--Reiche--Kuhn sum rule.
Consequently this page compares trends and ionization conventions; it does
not report a pointwise error metric.

``USE_PRECOMPUTED_DATA = True`` verifies and loads the accepted 62-state,
pressure-ionization-refined Otter scan.  To calculate all states directly
with this file, set
``USE_PRECOMPUTED_DATA = False`` and ``RECOMPUTE_WITH_OTTER = True`` in the
input block.  The recomputation path constructs
:class:`otter.electronic.FullExternalConfig` and calls
:func:`otter.electronic.solve_full_only` here; it does not import another
example or producer.  Fresh results are staged under
``benchmarks/outputs/bethkenhagen_et_al_2020_carbon_ionization`` and never
replace the accepted baseline automatically.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

# Each state also uses continuum worker processes.  Keep BLAS libraries from
# silently multiplying the explicitly requested process count.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

from otter.electronic import FullExternalConfig, solve_full_only
from otter.plotting import PALETTES, save_figure, style_context


# =============================================================================
# User input
# =============================================================================
USE_PRECOMPUTED_DATA = True
RECOMPUTE_WITH_OTTER = False

TEMPERATURE_EV = 100.0
DENSITIES_G_CC = np.asarray(
    (
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        1.00,
        1.05,
        1.10,
        1.15,
        1.20,
        1.25,
        1.30,
        1.40,
        1.50,
        1.60,
        1.70,
        1.80,
        1.90,
        2.00,
        2.10,
        2.20,
        2.30,
        2.35,
        2.40,
        2.50,
        2.55,
        2.60,
        2.70,
        2.80,
        2.90,
        3.00,
        3.20,
        3.50,
        3.80,
        4.00,
        4.10,
        4.20,
        4.30,
        4.40,
        4.50,
        4.60,
        4.70,
        4.80,
        4.90,
        5.00,
        5.20,
        5.40,
        5.50,
        5.60,
        5.80,
        6.00,
        8.00,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        18.0,
        20.0,
        22.0,
        25.0,
        30.0,
        34.0,
        40.0,
        50.0,
        75.0,
        100.0,
        110.0,
        150.0,
        200.0,
        220.0,
        250.0,
        300.0,
        340.0,
        400.0,
        450.0,
    ),
    dtype=float,
)

# Two outer state workers x four continuum workers uses at most eight
# explicit worker processes.
MAX_STATE_WORKERS = 2
CONTINUUM_WORKERS_PER_STATE = 4
AA_N_POINTS = 2**12
# =============================================================================


BASELINE_SCHEMA = "otter_carbon_ionization_levels_v3"
RECOMPUTED_SCHEMA = "otter_bethkenhagen_carbon_ionization_v1"


def repository_root() -> Path:
    """Locate the Otter checkout when run directly or by Sphinx-Gallery."""
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    source_file = globals().get("__file__")
    if source_file is not None:
        source = Path(str(source_file)).resolve()
        candidates.extend([source.parent, *source.parents])
    for candidate in candidates:
        if (
            candidate
            / "benchmarks"
            / "reference_data"
            / "bethkenhagen_et_al_2020_carbon_ionization"
            / "manifest.json"
        ).is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the Otter checkout.")


ROOT = repository_root()
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / "carbon_ionization_levels"
BASELINE_PATH = BASELINE_DIR / "C_Te100eV_density_scan.npz"
BASELINE_MANIFEST = BASELINE_DIR / "manifest.json"
REFERENCE_DIR = (
    ROOT
    / "benchmarks"
    / "reference_data"
    / "bethkenhagen_et_al_2020_carbon_ionization"
)
REFERENCE_MANIFEST = REFERENCE_DIR / "manifest.json"
OUTPUT_DIR = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "bethkenhagen_et_al_2020_carbon_ionization"
)
RECOMPUTED_PATH = OUTPUT_DIR / "C_Te100eV_density_scan.npz"
FIGURE_DIR = OUTPUT_DIR / "figures"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_accepted_otter_scan() -> dict[str, np.ndarray]:
    """Verify and load the reviewed 62-state Otter result."""
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "otter_example_manifest_v1"
        or manifest.get("example_id") != "carbon_ionization_levels"
        or not str(manifest.get("status", "")).startswith("accepted")
    ):
        raise ValueError("The carbon-ionization baseline is not accepted.")
    record = dict(manifest.get("state", {}))
    if record.get("data_file") != BASELINE_PATH.name:
        raise ValueError("The carbon-ionization manifest names the wrong file.")
    if sha256_file(BASELINE_PATH) != str(record.get("data_sha256")):
        raise RuntimeError("Carbon-ionization baseline checksum mismatch.")

    with np.load(BASELINE_PATH, allow_pickle=False) as archive:
        state = {key: np.asarray(archive[key]) for key in archive.files}
    if str(state["schema_version"].item()) != BASELINE_SCHEMA:
        raise ValueError("Unsupported carbon-ionization baseline schema.")
    rho = np.asarray(state["rho_g_cc"], dtype=float)
    if rho.shape != DENSITIES_G_CC.shape or not np.allclose(
        rho,
        DENSITIES_G_CC,
        rtol=0.0,
        atol=1.0e-12,
    ):
        # The accepted archive is authoritative for the offline gallery path.
        # During an incremental scan it can legitimately lag the producer's
        # requested grid until the new candidate is reviewed and promoted.
        print(
            "Using the accepted archive density grid "
            f"({rho.size} states); the producer requests "
            f"{DENSITIES_G_CC.size} states."
        )
    if not np.all(np.asarray(state["stage2_converged"], dtype=bool)):
        raise RuntimeError("The accepted scan contains an unconverged AA state.")
    return state


def load_published_curves() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load the checksummed digitization of the published Figure 3(a)."""
    manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "otter_reference_manifest_v1"
        or manifest.get("dataset_id")
        != "bethkenhagen_et_al_2020_carbon_ionization"
        or str(manifest["source"]["doi"])
        != "10.1103/PhysRevResearch.2.023260"
    ):
        raise ValueError("Unexpected Bethkenhagen et al. reference manifest.")

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for record in manifest["files"]:
        path = REFERENCE_DIR / str(record["file"])
        if sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(f"Reference checksum mismatch: {path.name}.")
        values = np.loadtxt(path, delimiter=",", comments="#")
        if (
            values.ndim != 2
            or values.shape[1] != 2
            or np.any(~np.isfinite(values))
            or np.any(values[:, 0] <= 0.0)
            or np.any(np.diff(values[:, 0]) <= 0.0)
        ):
            raise ValueError(f"Malformed reference curve: {path.name}.")
        curves[str(record["label"])] = (values[:, 0], values[:, 1])
    return curves


def aa_config(rho_g_cc: float) -> FullExternalConfig:
    """Construct one carbon full-AA state without an enlarged bound box."""
    return FullExternalConfig(
        element="C",
        temperature_ev=float(TEMPERATURE_EV),
        rho_g_cc=float(rho_g_cc),
        run_mode="full",
        n_points=int(AA_N_POINTS),
        cont_n_jobs=int(CONTINUUM_WORKERS_PER_STATE),
        cont_shards=int(2 * CONTINUUM_WORKERS_PER_STATE),
        show_scf_progress=False,
        save_data=False,
        bound_occ_mode="fd",
        bound_rmax_mult=None,
        bound_energy_cut_mode="v_frac",
        bound_energy_cut=0.70,
        bound_zero_tail_refine=True,
        bound_zero_tail_max_binding_ha=1.0e-2,
        bound_zero_tail_scan_points=64,
        bound_zero_tail_l_max=1,
        bound_zero_tail_edge_rel_tol=0.1,
        b3_tail_model="full",
    )


def solve_state(rho_g_cc: float) -> dict[str, float]:
    """Solve and reduce one independent Otter average-atom state."""
    started = time.perf_counter()
    result = solve_full_only(aa_config(float(rho_g_cc)))
    elapsed_s = time.perf_counter() - started
    if not bool(result.get("stage2_converged", False)):
        raise RuntimeError(
            f"C rho={rho_g_cc:g} g/cc did not reach stage-2 convergence."
        )
    history = list(result.get("history", ()))
    error = float(history[-1].get("err", np.nan)) if history else np.nan
    meta = dict(result["meta"])
    n_i = float(meta["n_i_bohr3"])
    n0 = float(meta["n0_final_bohr3"])
    row = {
        "rho_g_cc": float(rho_g_cc),
        "zbar": float(result["zbar"]),
        "zstar": n0 / n_i,
        "mu_ha": float(result["mu"]),
        "stage2_error": error,
        "elapsed_s": float(elapsed_s),
    }
    if not np.all(np.isfinite(tuple(row.values()))):
        raise RuntimeError(f"C rho={rho_g_cc:g} g/cc produced non-finite data.")
    return row


def recompute_otter_scan() -> dict[str, np.ndarray]:
    """Calculate all states directly and save a non-accepted candidate."""
    rows: list[dict[str, float]] = []
    workers = min(max(int(MAX_STATE_WORKERS), 1), DENSITIES_G_CC.size)
    if workers == 1:
        for rho in DENSITIES_G_CC:
            rows.append(solve_state(float(rho)))
            print(f"[computed] C, rho={float(rho):g} g/cc")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(solve_state, float(rho)): float(rho)
                for rho in DENSITIES_G_CC
            }
            for future in as_completed(futures):
                rho = futures[future]
                rows.append(future.result())
                print(f"[computed] C, rho={rho:g} g/cc")
    rows.sort(key=lambda row: row["rho_g_cc"])

    state = {
        "schema_version": np.asarray(RECOMPUTED_SCHEMA),
        "element_symbol": np.asarray("C"),
        "temperature_ev": np.asarray(TEMPERATURE_EV),
        "rho_g_cc": np.asarray([row["rho_g_cc"] for row in rows]),
        "zbar": np.asarray([row["zbar"] for row in rows]),
        "zstar": np.asarray([row["zstar"] for row in rows]),
        "mu_ha": np.asarray([row["mu_ha"] for row in rows]),
        "stage2_converged": np.ones(len(rows), dtype=bool),
        "stage2_error": np.asarray([row["stage2_error"] for row in rows]),
        "elapsed_s": np.asarray([row["elapsed_s"] for row in rows]),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = RECOMPUTED_PATH.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **state)
    temporary.replace(RECOMPUTED_PATH)
    candidate_manifest = {
        "schema_version": "otter_benchmark_candidate_manifest_v1",
        "benchmark_id": "bethkenhagen_et_al_2020_carbon_ionization",
        "status": "candidate_not_accepted",
        "state": {
            "data_file": RECOMPUTED_PATH.name,
            "data_sha256": sha256_file(RECOMPUTED_PATH),
            "states": len(rows),
            "all_stage2_converged": True,
            "max_stage2_error": float(np.max(state["stage2_error"])),
        },
        "configuration": {
            "element": "C",
            "temperature_ev": TEMPERATURE_EV,
            "densities_g_cc": DENSITIES_G_CC.tolist(),
            "aa_n_points": AA_N_POINTS,
            "continuum_workers_per_state": CONTINUUM_WORKERS_PER_STATE,
            "bound_occ_mode": "fd",
            "bound_rmax_mult": None,
            "bound_energy_cut_mode": "v_frac",
            "bound_energy_cut_value": 0.70,
            "bound_zero_tail_refine": True,
            "b3_tail_model": "full",
        },
    }
    RECOMPUTED_PATH.with_suffix(".manifest.json").write_text(
        json.dumps(candidate_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return state


def plot_comparison(
    otter_state: dict[str, np.ndarray],
    published: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Plot a definition-aware literature comparison without false scoring."""
    rho = np.asarray(otter_state["rho_g_cc"], dtype=float)
    zbar = np.asarray(otter_state["zbar"], dtype=float)
    zstar = np.asarray(otter_state["zstar"], dtype=float)
    colors = PALETTES["bing"]
    published_styles = {
        "DFT-MD": (colors[2], ":", "o"),
        "Purgatorio": (colors[3], "--", None),
        "OPAL": (colors[4], "--", None),
        "ATOMIC": (colors[5], ":", None),
        "BU-EK": (colors[6], ":", None),
        "BU-SP": (colors[7], "-.", None),
        "BU-SP + Pauli blocking": ("0.48", "--", None),
    }

    with style_context("thesis", palette="bing", figsize="docs_wide"):
        fig, ax = plt.subplots()
        for label, (rho_ref, z_ref) in published.items():
            color, line_style, marker = published_styles[label]
            ax.plot(
                rho_ref,
                z_ref,
                color=color,
                ls=line_style,
                lw=1.75,
                marker=marker,
                ms=5.0 if marker else None,
                markerfacecolor="white" if marker else None,
                markeredgewidth=1.1 if marker else None,
                alpha=0.75,
                label=rf"{label} $Z^{{\rm free}}$",
                zorder=2,
            )
        ax.plot(
            rho,
            zbar,
            color="#131313",
            lw=2.3,
            ls="-",
            marker="o",
            ms=4.0,
            alpha=0.45,
            label=r"Otter $\bar Z=Z-Q_{\rm ion}(R_{\rm WS})$",
            zorder=5,
        )
        ax.plot(
            rho,
            zstar,
            color=colors[1],
            lw=2.3,
            ls="-",
            marker="o",
            ms=4.0,
            alpha=0.45,
            label=r"Otter $Z^*=n_e^0/n_i$",
            zorder=4,
        )
        ax.set(
            xscale="log",
            xlim=(0.09, 900.0),
            ylim=(3.55, 6.15),
            xlabel=r"$\rho$ [g cm$^{-3}$]",
            ylabel="carbon ionization",
            title=r"Carbon at $T_e=100$ eV",
        )
        ax.legend(ncol=2, fontsize=8.7, loc="best")
        fig.tight_layout()
        paths = save_figure(
            fig,
            FIGURE_DIR / "bethkenhagen_2020_carbon_ionization",
        )
    print(f"saved PNG: {paths['png'].relative_to(ROOT)}")
    print(f"saved PDF: {paths['pdf'].relative_to(ROOT)}")
    if "agg" not in plt.get_backend().lower():
        plt.show()


def main() -> None:
    """Load or calculate Otter data, then make the literature comparison."""
    if bool(USE_PRECOMPUTED_DATA) == bool(RECOMPUTE_WITH_OTTER):
        raise ValueError(
            "Select exactly one path: set one of USE_PRECOMPUTED_DATA and "
            "RECOMPUTE_WITH_OTTER to True."
        )
    if USE_PRECOMPUTED_DATA:
        state = load_accepted_otter_scan()
        print(
            "Using checksummed Otter data: "
            f"{BASELINE_PATH.relative_to(ROOT)}"
        )
    else:
        state = recompute_otter_scan()
        print(
            "Using freshly recomputed Otter data: "
            f"{RECOMPUTED_PATH.relative_to(ROOT)}"
        )
    published = load_published_curves()
    print(
        f"Loaded {len(published)} digitized curves from "
        "Bethkenhagen et al. (2020), Fig. 3(a)."
    )
    print(
        "No RMSE or pointwise residual is reported because the published "
        "Zfree curves and the two Otter diagnostics use different partitions."
    )
    plot_comparison(state, published)


if __name__ == "__main__":
    main()
