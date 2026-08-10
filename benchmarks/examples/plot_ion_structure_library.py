"""
Ion-structure literature library
================================

This complete, directly executable Otter benchmark compares QOZ/HNC results
with curated literature curves for aluminium, beryllium, and carbon.  Edit
only the input block below.  ``USE_PRECOMPUTED_DATA = True`` verifies and
loads the reviewed Otter NPZ files.  With ``False``, this same file directly
constructs :class:`otter.PlasmaWorkflowConfig`, evaluates every average atom
and ion structure, saves new NPZ files below
``benchmarks/outputs/ion_structure_library/gallery_recomputed``, and plots
those new results.  It does not import another benchmark runner or producer.

In the four-panel :math:`S_{ii}(k)` figure, panel 1 uses Gill *et al.*,
Fig. 3 :cite:p:`GillEtAl2015`; panels 2 and 3 use Clérouin *et al.*, Fig. 1
:cite:p:`ClerouinEtAl2015`; and panel 4 uses Wünsch *et al.*, Fig. 2
:cite:p:`WunschEtAl2009`.  The separate real-space Wünsch comparison uses
Fig. 1(c).  A carbon extraction is attributed by the local library to
:cite:t:`StarrettSaumon2013`; its exact panel has not been independently
verified, so no figure number is claimed.
Reference coordinates remain separate data files because they are digitized
publication data; all unit conversions are explicit below.

The average-atom/pseudoatom construction follows
:cite:t:`StarrettSaumon2013,StarrettSaumon2014`, and the default
finite-temperature jellium LFC follows :cite:t:`Chabrier1990`.  See
:doc:`the provenance, coordinate units, exclusions, and data-rights audit
</benchmarks/ion_structure_library>` for the scientific interpretation.
Both benchmark figures are exported as matching PNG and vector PDF files
under ``benchmarks/outputs/ion_structure_library/figures``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from otter import (
    PlasmaWorkflowConfig,
    continue_plasma_workflow_from_electronic_result,
    solve_plasma_workflow,
)
from otter.plotting import (
    MODEL_STYLES,
    grid_figsize,
    save_figure,
    style_context,
)


# =============================================================================
# User input
# =============================================================================
USE_PRECOMPUTED_DATA = True

# Three independent state groups x six continuum workers uses at most about
# 18 worker processes on a 24-core workstation.  The two Al 8.1-g/cc states
# share one electronic calculation because only the ion temperature differs.
MAX_STATE_WORKERS = 3
CONTINUUM_WORKERS_PER_STATE = 6
AA_N_POINTS = 1024
QOZ_N_POINTS = 4096

LFC_MODEL = "chabrier1990"
HNC_TOL = 1.0e-4
HNC_CLOSURE_TOL = 2.5e-3
R_RETAIN_MAX_BOHR = 20.0
K_RETAIN_MAX_BOHR_INV = 20.0
# =============================================================================


BOHR_TO_ANGSTROM = 0.529177210903

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
}

REFERENCE_SERIES: dict[str, tuple[dict[str, str], ...]] = {
    "al_gill_rho2p7_te5_ti5": (
        {
            "observable": "sii",
            "label": "Gill KS-PAMD",
            "file": "gill_et_al_2015/Sii_Al_T5ev_rho2.7_KS-PAMD_Gill.csv",
            "x_unit": "angstrom^-1",
        },
        {
            "observable": "sii",
            "label": "Gill TF-PAMD",
            "file": "gill_et_al_2015/Sii_Al_T5ev_rho2.7_TF-PAMD_Gill.csv",
            "x_unit": "angstrom^-1",
        },
        {
            "observable": "sii",
            "label": "Gill TF-DFT-MD",
            "file": "gill_et_al_2015/Sii_Al_T5ev_rho2.7_QMD_Gill.csv",
            "x_unit": "angstrom^-1",
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
        },
        {
            "observable": "sii",
            "label": "HNC-Y-SRR",
            "file": (
                "clerouin_et_al_2015/"
                "Jean_2015_Al_rho8.1_Te10.0_Ti2.0_SRR.csv"
            ),
            "x_unit": "angstrom^-1",
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
        },
        {
            "observable": "sii",
            "label": "HNC-Y-SRR",
            "file": (
                "wunsch_et_al_2009/"
                "Sii_HNC-YSRR_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom^-1",
        },
        {
            "observable": "sii",
            "label": "HNC-KK",
            "file": (
                "wunsch_et_al_2009/"
                "Sii_HNCKK_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom^-1",
        },
        {
            "observable": "sii",
            "label": "HNC-Y",
            "file": (
                "wunsch_et_al_2009/"
                "Sii_HNCY_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom^-1",
        },
        {
            "observable": "gii",
            "label": "Wünsch DFT-MD",
            "file": (
                "wunsch_et_al_2009/"
                "gii_DFTMD_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom",
        },
        {
            "observable": "gii",
            "label": "HNC-Y-SRR",
            "file": (
                "wunsch_et_al_2009/"
                "gii_HNC-YSRR_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom",
        },
        {
            "observable": "gii",
            "label": "HNC-KK",
            "file": (
                "wunsch_et_al_2009/"
                "gii_HNC-KK_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom",
        },
        {
            "observable": "gii",
            "label": "HNC-Y",
            "file": (
                "wunsch_et_al_2009/"
                "gii_HNC-Y_Be_3rho0_T13_Z2_wunsch2009.csv"
            ),
            "x_unit": "angstrom",
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
            / "baselines"
            / "ion_structure_library"
            / "manifest.json"
        ).is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the Otter checkout.")


ROOT = repository_root()
PRECOMPUTED_DIR = (
    ROOT / "benchmarks" / "baselines" / "ion_structure_library"
)
REFERENCE_DIR = (
    ROOT / "benchmarks" / "reference_data" / "ion_structure_library"
)
OUTPUT_DIR = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "ion_structure_library"
    / "gallery_recomputed"
)
FIGURE_DIR = (
    ROOT / "benchmarks" / "outputs" / "ion_structure_library" / "figures"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        state = {key: np.asarray(archive[key]) for key in archive.files}
    if any(value.dtype.hasobject for value in state.values()):
        raise TypeError(f"Object arrays are forbidden in {path}.")
    return state


def load_precomputed_states() -> dict[str, dict[str, np.ndarray]]:
    """Verify every accepted result checksum before plotting."""
    manifest = json.loads(
        (PRECOMPUTED_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("benchmark_id") != "ion_structure_library":
        raise ValueError("Unexpected ion-structure manifest.")
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for record in manifest["states"]:
        path = PRECOMPUTED_DIR / str(record["baseline_file"])
        if sha256_file(path) != str(record["baseline_sha256"]):
            raise RuntimeError(f"Checksum mismatch for {path}.")
        state = load_npz(path)
        state_id = str(state["state_id"].item())
        if state_id != str(record["state_id"]):
            raise ValueError(f"State identifier mismatch in {path}.")
        loaded[state_id] = state
    return loaded


def workflow_config(
    state: dict[str, Any],
    *,
    ion_temperature_ev: float | None,
) -> PlasmaWorkflowConfig:
    """Build the complete public Otter workflow for one thermodynamic state."""
    return PlasmaWorkflowConfig(
        elements=[str(state["element"])],
        temperature_ev=float(state["te_ev"]),
        ion_temperature_ev=(
            None if ion_temperature_ev is None else float(ion_temperature_ev)
        ),
        rho_g_cc=float(state["rho_g_cc"]),
        electronic_model="qm",
        aa_overrides={
            "n_points": int(AA_N_POINTS),
            "cont_n_jobs": int(CONTINUUM_WORKERS_PER_STATE),
            "cont_shards": int(2 * CONTINUUM_WORKERS_PER_STATE),
            "bound_occ_mode": "fd",
            "bound_rmax_mult": None,
            "bound_zero_tail_refine": False,
            "b3_tail_model": "full",
        },
        qoz_linear_n_points=int(QOZ_N_POINTS),
        qoz_pad_factor=2.0,
        qoz_zbar_mode="pseudoatom_partition",
        qoz_renormalize_nscr_to_zbar=True,
        qoz_response_chi0_model="lindhard_fd",
        qoz_response_lfc_model=str(LFC_MODEL),
        hnc_tol=float(HNC_TOL),
        hnc_closure_transform_tol=float(HNC_CLOSURE_TOL),
        hnc_max_iter=500,
        hnc_require_converged=True,
        show_progress=False,
    )


def strict_check(
    workflow: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject unconverged electronic or HNC best-effort output."""
    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"])
    if electronic.get("stage2_converged") is not True:
        raise RuntimeError("Full average-atom stage 2 did not converge.")
    if dict(electronic.get("ext_status", {})).get("converged") is not True:
        raise RuntimeError("External fixed-mu average atom did not converge.")
    if str(electronic.get("threshold_state_status", "")).lower() == "unresolved":
        raise RuntimeError("The threshold-state representation is unresolved.")
    if ion.get("hnc_converged") is not True:
        raise RuntimeError("HNC did not reach a physical fixed point.")
    if float(ion["hnc_output_residual"]) > HNC_TOL:
        raise RuntimeError("HNC residual exceeds the configured tolerance.")
    if float(ion["closure_transform_max_abs"]) > HNC_CLOSURE_TOL:
        raise RuntimeError("The g/S transform-closure audit failed.")
    return electronic, ion


def pack_result(
    workflow: dict[str, Any],
    state: dict[str, Any],
    *,
    elapsed_s: float,
) -> dict[str, np.ndarray]:
    """Keep the portable arrays needed by this benchmark and downstream use."""
    electronic, ion = strict_check(workflow)
    r_e = np.asarray(electronic["r"], dtype=float)
    r = np.asarray(ion["r"], dtype=float)
    k = np.asarray(ion["k"], dtype=float)
    e_mask = r_e <= R_RETAIN_MAX_BOHR
    r_mask = r <= R_RETAIN_MAX_BOHR
    k_mask = k <= K_RETAIN_MAX_BOHR_INV
    return {
        "schema_version": np.asarray("otter_gallery_ion_library_v1"),
        "state_id": np.asarray(str(state["state_id"])),
        "element": np.asarray(str(state["element"])),
        "rho_g_cc": np.asarray(float(state["rho_g_cc"])),
        "te_ev": np.asarray(float(state["te_ev"])),
        "ti_ev": np.asarray(float(state["ti_ev"])),
        "producer_elapsed_s": np.asarray(float(elapsed_s)),
        "r_e_bohr": r_e[e_mask],
        "n_full_bohr3": np.asarray(electronic["n_full"])[e_mask],
        "n_scr_bohr3": np.asarray(electronic["n_scr"])[e_mask],
        "r_bohr": r[r_mask],
        "gii_r": np.asarray(ion["gii_r"])[r_mask],
        "k_bohr_inv": k[k_mask],
        "sii_k": np.asarray(ion["sii_k"])[k_mask],
        "vii_k_ha_bohr3": np.asarray(ion["vii_k"])[k_mask],
        "n_scr_k_electrons": np.asarray(ion["n_scr_k"])[k_mask],
        "zbar_partition": np.asarray(float(ion["zbar_partition"])),
        "hnc_best_residual": np.asarray(float(ion["hnc_output_residual"])),
        "hnc_closure_mismatch": np.asarray(
            float(ion["closure_transform_max_abs"])
        ),
    }


def solve_group(
    group: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, np.ndarray]]:
    """Solve one electronic state and reuse it when only Ti changes."""
    first = group[0]
    started = time.perf_counter()
    electronic_only = solve_plasma_workflow(
        workflow_config(first, ion_temperature_ev=None)
    )
    electronic_elapsed = time.perf_counter() - started
    electronic_kind = str(electronic_only["electronic"]["kind"])
    electronic = dict(electronic_only["electronic"]["result"])

    solved: dict[str, dict[str, np.ndarray]] = {}
    for state in group:
        ion_started = time.perf_counter()
        workflow = continue_plasma_workflow_from_electronic_result(
            workflow_config(
                state,
                ion_temperature_ev=float(state["ti_ev"]),
            ),
            electronic_kind=electronic_kind,
            electronic_result=electronic,
        )
        elapsed = electronic_elapsed + time.perf_counter() - ion_started
        payload = pack_result(workflow, state, elapsed_s=elapsed)
        solved[str(state["state_id"])] = payload
    return solved


def solve_all_states() -> dict[str, dict[str, np.ndarray]]:
    """Calculate all state groups with a bounded process pool."""
    groups = tuple(STATE_GROUPS.values())
    loaded: dict[str, dict[str, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=MAX_STATE_WORKERS) as pool:
        futures = {pool.submit(solve_group, group): group for group in groups}
        for future in as_completed(futures):
            group_result = future.result()
            loaded.update(group_result)
            print("[computed] " + ", ".join(sorted(group_result)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for state_id, state in loaded.items():
        path = OUTPUT_DIR / f"{state_id}.npz"
        np.savez_compressed(path, **state)
        print(f"[saved] {path}")
    return loaded


def load_reference(series: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Load one two-column digitization without changing its coordinate unit."""
    path = (REFERENCE_DIR / series["file"]).resolve()
    if not path.is_relative_to(REFERENCE_DIR.resolve()):
        raise ValueError("Reference path escapes its package directory.")
    values = np.asarray(
        np.genfromtxt(path, delimiter=",", comments="#"),
        dtype=float,
    )
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"Expected two reference columns in {path}.")
    mask = np.isfinite(values[:, 0]) & np.isfinite(values[:, 1])
    return values[mask, 0], values[mask, 1]


def otter_curve(
    state: dict[str, np.ndarray],
    observable: str,
    x_unit: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert only the Otter coordinate to the publication's stated unit."""
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


def print_metrics(states: dict[str, dict[str, np.ndarray]]) -> None:
    print(
        f"{'state':42s} {'obs':>3s} {'reference':19s} "
        f"{'RMSE':>10s} {'MAE':>10s} {'max':>10s}"
    )
    for state_id, series_list in REFERENCE_SERIES.items():
        for series in series_list:
            x_ref, y_ref = load_reference(series)
            x_otter, y_otter = otter_curve(
                states[state_id],
                str(series["observable"]),
                str(series["x_unit"]),
            )
            mask = (x_ref >= x_otter[0]) & (x_ref <= x_otter[-1])
            delta = np.interp(x_ref[mask], x_otter, y_otter) - y_ref[mask]
            print(
                f"{state_id:42s} {series['observable']:>3s} "
                f"{series['label'][:19]:19s} "
                f"{np.sqrt(np.mean(delta**2)):10.4e} "
                f"{np.mean(np.abs(delta)):10.4e} "
                f"{np.max(np.abs(delta)):10.4e}"
            )


states = (
    load_precomputed_states()
    if USE_PRECOMPUTED_DATA
    else solve_all_states()
)
print(
    "Using "
    + (
        "checksummed, precomputed Otter results."
        if USE_PRECOMPUTED_DATA
        else "new results calculated directly by this gallery script."
    )
)
print_metrics(states)


def plot_observable(
    *,
    observable: str,
    state_ids: tuple[str, ...],
) -> plt.Figure:
    """Draw one self-contained comparison figure."""
    ncols = 2
    nrows = (len(state_ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=grid_figsize(
            nrows,
            ncols,
            cell_width=5.1,
            cell_height=3.8,
        ),
        squeeze=False,
    )
    marker_cycle = ("o", "s", "^", "x")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for panel, state_id in enumerate(state_ids):
        axis = axes.ravel()[panel]
        series_list = [
            item
            for item in REFERENCE_SERIES[state_id]
            if item["observable"] == observable
        ]
        display_unit = str(series_list[0]["x_unit"])
        x_otter, y_otter = otter_curve(
            states[state_id],
            observable,
            display_unit,
        )
        axis.plot(
            x_otter,
            y_otter,
            label="Otter",
            **dict(MODEL_STYLES["otter"]),
        )
        reference_x: list[np.ndarray] = []
        for index, series in enumerate(series_list):
            x_ref, y_ref = load_reference(series)
            reference_x.append(x_ref)
            marker = marker_cycle[index % len(marker_cycle)]
            scatter_options: dict[str, Any] = {
                "s": 25,
                "marker": marker,
                "linewidths": 1.2,
                "label": series["label"],
                "zorder": 3,
            }
            color = colors[index % len(colors)]
            if marker == "x":
                scatter_options["color"] = color
            else:
                scatter_options["facecolors"] = "none"
                scatter_options["edgecolors"] = color
            axis.scatter(x_ref, y_ref, **scatter_options)
        axis.set_title(STATE_TITLES[state_id], fontsize=10)
        axis.set_ylabel(
            r"$S_{ii}(k)$" if observable == "sii" else r"$g_{ii}(r)$"
        )
        if display_unit == "angstrom^-1":
            axis.set_xlabel(r"$k$ [$\mathrm{\AA}^{-1}$]")
        elif display_unit == "angstrom":
            axis.set_xlabel(r"$r$ [$\mathrm{\AA}$]")
        else:
            axis.set_xlabel(r"$r$ [Bohr]")
        all_reference_x = np.concatenate(reference_x)
        span = float(np.ptp(all_reference_x))
        margin = max(0.03 * span, 1.0e-6)
        left = (
            -0.5
            if observable == "gii"
            else max(0.0, float(np.min(all_reference_x)) - margin)
        )
        axis.set_xlim(left, float(np.max(all_reference_x)) + margin)
        axis.axhline(1.0, color="0.55", lw=0.8, ls=":")
        axis.legend(fontsize=8)
    for panel in range(len(state_ids), axes.size):
        axes.ravel()[panel].set_visible(False)
    fig.suptitle("Otter QOZ/HNC versus curated literature curves", y=0.985)
    source_line = (
        "Reference data: Gill et al. (2020); Clérouin et al. (2015); "
        "Wunsch et al. (2009)."
        if observable == "sii"
        else "Reference data: Wunsch et al. (2009); "
        "Starrett and Saumon (2013)."
    )
    fig.text(
        0.5,
        0.006,
        source_line,
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.965), pad=0.55)
    return fig


# %%
# Static ion structure factors
# ----------------------------
#
# Literature wave numbers are stored in inverse ångström.  Otter's native
# inverse-Bohr grid is converted explicitly inside ``otter_curve``.

plot_style = ExitStack()
plot_style.enter_context(style_context("thesis", palette="bing"))
fig_sii = plot_observable(
    observable="sii",
    state_ids=(
        "al_gill_rho2p7_te5_ti5",
        "al_clerouin_rho8p1_te10_ti10",
        "al_clerouin_rho8p1_te10_ti2",
        "be_wunsch_rho5p544_te13_ti13",
    ),
)


# %%
# Pair distribution functions
# ---------------------------
#
# The Be coordinates are ångström, while the carbon digitization uses Bohr.

fig_gii = plot_observable(
    observable="gii",
    state_ids=(
        "be_wunsch_rho5p544_te13_ti13",
        "c_starrett_rho20_te50_ti50",
    ),
)

save_figure(
    fig_sii,
    FIGURE_DIR / "ion_structure_library_sii",
    close=False,
)
save_figure(
    fig_gii,
    FIGURE_DIR / "ion_structure_library_gii",
    close=False,
)
plot_style.close()

if "agg" not in plt.get_backend().lower():
    plt.show()
