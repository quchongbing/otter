r"""
CH1.36: multicomponent electronic-to-ionic workflow
===================================================

This example calculates one genuine two-species plasma with the public Otter
workflow:

.. math::

   \mathrm{mixture\ AA}
   \rightarrow \{q_{\rm C}(k),q_{\rm H}(k)\}
   \rightarrow V_{ab}(k)
   \rightarrow \mathrm{mixture\ OZ/HNC}
   \rightarrow \{g_{ab}(r),S_{ab}(k)\}.

The state is ``CH1.36``, ``rho=5 g/cc`` and
``Te=Ti=100 kK``.  Set ``RECOMPUTE_WITH_OTTER=True`` below to run this
calculation in the present file.  The default verifies and loads a reviewed
result produced by the same current-Otter calculation so documentation builds
remain quick.  No digitized or third-party numerical curve is used here.

The full/external pseudoatom and QOZ construction follow
:cite:t:`StarrettSaumon2014`; the multicomponent equations follow
:cite:t:`StarrettEtAl2014`; and the finite-temperature jellium local-field
correction follows :cite:t:`Chabrier1990`.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from otter import PlasmaWorkflowConfig, solve_plasma_workflow
from otter.plotting import PAIR_COLORS, grid_figsize, save_figure, style_context


# =============================================================================
# User input
# =============================================================================
RECOMPUTE_WITH_OTTER = False
if os.environ.get("OTTER_RECOMPUTE_CH136_EXAMPLE", "0") == "1":
    RECOMPUTE_WITH_OTTER = True

ELEMENTS = ("C", "H")
COUNTS = (1.0, 1.36)
RHO_G_CC = 5.0
TEMPERATURE_K = 100_000.0
EV_PER_K = 8.617333262145e-5
TE_EV = TEMPERATURE_K * EV_PER_K
TI_EV = TE_EV

CONTINUUM_WORKERS = int(os.environ.get("OTTER_CONTINUUM_WORKERS", "6"))
CONTINUUM_SHARDS = 32
COMMON_MU_TOL_HA = 1.0e-4
HNC_TOL = 1.0e-5
HNC_CLOSURE_TOL = 1.0e-4
# =============================================================================


SCHEMA = "otter_ch136_mixture_workflow_v1"
PAIR_ORDER = (("CC", 0, 0), ("CH", 0, 1), ("HH", 1, 1))


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
            / "ch136_mixture_workflow_100kk"
            / "manifest.json"
        ).is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the Otter repository root.")


ROOT = repository_root()
BASELINE_DIR = (
    ROOT / "benchmarks" / "baselines" / "ch136_mixture_workflow_100kk"
)
OUTPUT_DIR = (
    ROOT / "benchmarks" / "outputs" / "ch136_mixture_workflow_100kk"
)
FIGURE_DIR = OUTPUT_DIR / "figures"
RECOMPUTED_PATH = OUTPUT_DIR / "recomputed" / "CH1p36_5gcc_100kK.npz"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workflow_config() -> PlasmaWorkflowConfig:
    """Return the complete public configuration used on this page."""
    return PlasmaWorkflowConfig(
        elements=list(ELEMENTS),
        counts=list(COUNTS),
        temperature_ev=TE_EV,
        ion_temperature_ev=TI_EV,
        rho_g_cc=RHO_G_CC,
        aa_overrides={
            "cont_n_jobs": CONTINUUM_WORKERS,
            "cont_shards": CONTINUUM_SHARDS,
            "b3_tail_target": "full",
            "b3_r_cut_mult": 3.0,
            "b3_r_fit_max_mult": 4.0,
            "full_b3_use_source_closure": False,
            "ext_b3_use_source_closure": False,
        },
        root_maxfev=32,
        root_brent_maxiter=24,
        hnc_tol=HNC_TOL,
        hnc_closure_transform_tol=HNC_CLOSURE_TOL,
        hnc_max_iter=1000,
        show_mu_progress=True,
    )


def strict_audit(
    workflow: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Reject any unconverged common-mu, AA, external-AA, or HNC result."""
    electronic_block = dict(workflow["electronic"])
    if str(electronic_block["kind"]) != "mixture":
        raise RuntimeError("A two-species mixture result is required.")
    electronic = dict(electronic_block["result"])
    meta = dict(electronic.get("meta", {}))
    if not bool(meta.get("root_success", False)):
        raise RuntimeError("The common-mu root did not converge.")
    if float(meta.get("mu_residual_max_ha", np.inf)) > COMMON_MU_TOL_HA:
        raise RuntimeError("The initial common-mu residual exceeds tolerance.")
    if not bool(meta.get("final_mu_root_success", False)):
        raise RuntimeError("The full+external rerun lost common-mu closure.")
    if (
        float(meta.get("final_mu_residual_max_ha", np.inf))
        > COMMON_MU_TOL_HA
    ):
        raise RuntimeError("The final common-mu residual exceeds tolerance.")

    species = [dict(entry) for entry in electronic["species"]]
    if tuple(str(entry["element"]) for entry in species) != ELEMENTS:
        raise RuntimeError("Unexpected species order.")
    for entry in species:
        symbol = str(entry["element"])
        result = dict(entry["result"])
        if result.get("stage2_converged") is not True:
            raise RuntimeError(f"{symbol}: full AA stage 2 did not converge.")
        if str(result.get("threshold_state_status", "")).lower() == "unresolved":
            raise RuntimeError(f"{symbol}: unresolved threshold state.")
        if dict(result.get("ext_status", {})).get("converged") is not True:
            raise RuntimeError(f"{symbol}: external AA did not converge.")

    ion = dict(workflow["ion"])
    if ion.get("hnc_converged") is not True:
        raise RuntimeError("The mixture HNC did not converge.")
    if float(ion["hnc_output_residual"]) > HNC_TOL:
        raise RuntimeError("The HNC fixed-point residual exceeds tolerance.")
    if float(ion["closure_transform_max_abs"]) > HNC_CLOSURE_TOL:
        raise RuntimeError("The finite-lattice g/S closure audit failed.")
    return electronic, ion, species


def result_arrays(workflow: dict[str, Any], elapsed_s: float) -> dict[str, np.ndarray]:
    """Convert one strictly accepted workflow into a pickle-free archive."""
    electronic, ion, species = strict_audit(workflow)
    meta = dict(electronic["meta"])
    charge = dict(ion["charge_fix"])
    mu_species = np.asarray(
        [float(dict(entry["result"])["mu"]) for entry in species], dtype=float
    )
    arrays = {
        "schema_version": np.asarray(SCHEMA),
        "species_symbols": np.asarray(ELEMENTS),
        "species_counts": np.asarray(COUNTS, dtype=float),
        "rho_g_cc": np.asarray(RHO_G_CC),
        "temperature_k": np.asarray(TEMPERATURE_K),
        "temperature_ev": np.asarray(TE_EV),
        "r_bohr": np.asarray(ion["r"], dtype=float),
        "k_bohr_inv": np.asarray(ion["k"], dtype=float),
        "gij_r": np.asarray(ion["gij_r"], dtype=float),
        "sij_k": np.asarray(ion["sij_k"], dtype=float),
        "q_k": np.asarray(ion["n_scr_k"], dtype=float),
        "vij_k": np.asarray(ion["vij_k"], dtype=float),
        "mu_species_ha": mu_species,
        "mu_common_ha": np.asarray(float(np.mean(mu_species))),
        "root_residual_initial_ha": np.asarray(
            float(meta["mu_residual_max_ha"])
        ),
        "root_residual_final_ha": np.asarray(
            float(meta["final_mu_residual_max_ha"])
        ),
        "zbar_partition": np.asarray(ion["zbar_partition"], dtype=float),
        "zbar_qoz": np.asarray(ion["zbar_qoz"], dtype=float),
        "zbar_aa_ws": np.asarray(ion["zbar_aa_ws"], dtype=float),
        "q_scr_native_raw": np.asarray(
            charge["q_scr_native_raw"], dtype=float
        ),
        "q_scr_dst_raw": np.asarray(charge["q_scr_dst_raw"], dtype=float),
        "q_scr_dst_used": np.asarray(charge["q_scr_dst_used"], dtype=float),
        "q_scr_scale_factor": np.asarray(
            charge["scale_factor"], dtype=float
        ),
        "hnc_output_residual": np.asarray(
            float(ion["hnc_output_residual"])
        ),
        "hnc_closure_mismatch": np.asarray(
            float(ion["closure_transform_max_abs"])
        ),
        "hnc_s_min": np.asarray(float(ion["hnc_s_min"])),
        "hnc_s_max": np.asarray(float(ion["hnc_s_max"])),
        "producer_elapsed_s": np.asarray(float(elapsed_s)),
    }
    if any(value.dtype.hasobject for value in arrays.values()):
        raise TypeError("Gallery archives cannot contain object arrays.")
    return arrays


def calculate_with_otter() -> dict[str, np.ndarray]:
    """Run and save the complete calculation defined above."""
    started = time.perf_counter()
    workflow = solve_plasma_workflow(workflow_config())
    elapsed_s = time.perf_counter() - started
    arrays = result_arrays(workflow, elapsed_s)
    RECOMPUTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RECOMPUTED_PATH, **arrays)
    print(f"[saved] {RECOMPUTED_PATH}")
    return arrays


def load_reviewed_result() -> dict[str, np.ndarray]:
    """Verify and load the reviewed current-Otter result."""
    manifest = json.loads(
        (BASELINE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != "otter_example_manifest_v1":
        raise ValueError("Unsupported mixture-example manifest.")
    record = dict(manifest["state"])
    path = BASELINE_DIR / str(record["data_file"])
    if sha256_file(path) != str(record["data_sha256"]):
        raise RuntimeError(f"Checksum mismatch for {path}.")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if str(arrays["schema_version"].item()) != SCHEMA:
        raise ValueError("Unsupported mixture-example archive.")
    if any(value.dtype.hasobject for value in arrays.values()):
        raise TypeError("Object arrays are forbidden in gallery archives.")
    return arrays


result = (
    calculate_with_otter() if RECOMPUTE_WITH_OTTER else load_reviewed_result()
)

symbols = tuple(str(value) for value in result["species_symbols"])
mu = np.asarray(result["mu_species_ha"], dtype=float)
zbar_partition = np.asarray(result["zbar_partition"], dtype=float)
zbar_qoz = np.asarray(result["zbar_qoz"], dtype=float)
zbar_aa = np.asarray(result["zbar_aa_ws"], dtype=float)
q_native = np.asarray(result["q_scr_native_raw"], dtype=float)
q_dst_raw = np.asarray(result["q_scr_dst_raw"], dtype=float)
q_dst_used = np.asarray(result["q_scr_dst_used"], dtype=float)
q_scale = np.asarray(result["q_scr_scale_factor"], dtype=float)

print(
    "CH1.36  rho=5 g/cc  Te=Ti=100 kK "
    f"({float(result['temperature_ev']):.8f} eV)"
)
print(
    f"common mu = {float(result['mu_common_ha']):.8f} Ha; "
    f"max|Delta mu| = {float(result['root_residual_final_ha']):.3e} Ha"
)
print(
    f"HNC residual = {float(result['hnc_output_residual']):.3e}; "
    f"g/S closure = {float(result['hnc_closure_mismatch']):.3e}; "
    f"min eig(S) = {float(result['hnc_s_min']):.6f}"
)
print(
    f"{'sp':>3s} {'mu[Ha]':>12s} {'Zbar(AA)':>11s} "
    f"{'Zbar(part)':>12s} {'Zbar(QOZ)':>11s} "
    f"{'Qnative':>11s} {'Qdst raw':>11s} {'Qdst used':>11s} {'scale':>10s}"
)
for index, symbol in enumerate(symbols):
    print(
        f"{symbol:>3s} {mu[index]:12.8f} {zbar_aa[index]:11.7f} "
        f"{zbar_partition[index]:12.7f} {zbar_qoz[index]:11.7f} "
        f"{q_native[index]:11.7f} {q_dst_raw[index]:11.7f} "
        f"{q_dst_used[index]:11.7f} {q_scale[index]:10.7f}"
    )
print(f"reviewed producer wall time = {float(result['producer_elapsed_s']):.2f} s")


# %%
# Pair structure and pseudoatom inputs
# ------------------------------------
#
# The three pair channels use one stable colour assignment throughout.  The
# Ashcroft--Langreth convention used for ``S_ab`` is recorded by the public
# workflow.  The charge table above distinguishes the native pseudoatom
# integral from the DST-lattice charge actually used by QOZ.

r = np.asarray(result["r_bohr"], dtype=float)
k = np.asarray(result["k_bohr_inv"], dtype=float)
gij = np.asarray(result["gij_r"], dtype=float)
sij = np.asarray(result["sij_k"], dtype=float)
q_k = np.asarray(result["q_k"], dtype=float)
vij_k = np.asarray(result["vij_k"], dtype=float)

with style_context("thesis", palette="bing"):
    fig, axes = plt.subplots(2, 2, figsize=grid_figsize(2, 2))
    for label, i, j in PAIR_ORDER:
        axes[0, 0].plot(
            r, gij[i, j], color=PAIR_COLORS[label], label=label
        )
        axes[0, 1].plot(
            k, sij[i, j], color=PAIR_COLORS[label], label=label
        )
        axes[1, 1].plot(
            k, vij_k[i, j], color=PAIR_COLORS[label], label=label
        )

    axes[1, 0].plot(
        k, q_k[0], color=PAIR_COLORS["CC"], label=r"$q_{\rm C}(k)$"
    )
    axes[1, 0].plot(
        k, q_k[1], color=PAIR_COLORS["HH"], label=r"$q_{\rm H}(k)$"
    )

    axes[0, 0].set(
        title=r"$g_{ab}(r)$",
        xlabel=r"$r$ [Bohr]",
        ylabel=r"$g_{ab}(r)$",
        xlim=(-0.5, 8.0),
        ylim=(-0.05, None),
    )
    axes[0, 1].set(
        title=r"$S_{ab}(k)$",
        xlabel=r"$k$ [Bohr$^{-1}$]",
        ylabel=r"$S_{ab}(k)$",
        xlim=(0.0, 8.0),
    )
    axes[1, 0].set(
        title="Screening clouds",
        xlabel=r"$k$ [Bohr$^{-1}$]",
        ylabel=r"$q_a(k)$",
        xlim=(0.0, 8.0),
    )
    axes[1, 1].set(
        title=r"$V_{ab}(k)$",
        xlabel=r"$k$ [Bohr$^{-1}$]",
        ylabel=r"$V_{ab}(k)$ [Ha]",
        xlim=(0.0, 8.0),
    )
    for axis in axes.flat:
        axis.legend(frameon=False)

    fig.suptitle(
        r"CH$_{1.36}$: $\rho=5$ g cm$^{-3}$, "
        r"$T_e=T_i=100$ kK",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    save_figure(
        fig,
        FIGURE_DIR / "ch136_mixture_full_workflow",
        close=False,
    )

if "agg" not in plt.get_backend().lower():
    plt.show()
