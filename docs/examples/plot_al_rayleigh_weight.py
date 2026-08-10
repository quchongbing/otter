r"""
Al: elastic Rayleigh weight versus density
==========================================

This self-contained example evaluates the static ionic Rayleigh weight

.. math::

   W(k)=\left|q(k)+f(k)\right|^2 S_{ii}(k),

for aluminium at :math:`T_e=T_i=10\,\mathrm{eV}` and three mass densities.
Here ``q(k)`` is Otter's charge-closed screening-cloud transform and
``f(k)`` is the ionic form factor (the transform of the bound-electron
density).  Both are exported by :func:`otter.io.state.build_state_arrays`,
so the expression uses exactly the quantities supplied to the QOZ/HNC
workflow.  The convention is the static form-factor factor used in the
ion-structure construction of Starrett and Saumon (2014),
:cite:t:`StarrettSaumon2014`.

The default loads a checksummed archive produced by this same script.  Set
``RECOMPUTE_WITH_OTTER=True`` below, or run with
``OTTER_RECOMPUTE_AL_RAYLEIGH=1``, to perform all three Otter calculations
again.  The script writes both PNG and vector PDF figures.
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
from otter.io.state import StateExportOptions, build_state_arrays
from otter.plotting import grid_figsize, save_figure, style_context


# =============================================================================
# User input
# =============================================================================
USE_PRECOMPUTED_DATA = True
RECOMPUTE_WITH_OTTER = False
if os.environ.get("OTTER_RECOMPUTE_AL_RAYLEIGH", "0") == "1":
    USE_PRECOMPUTED_DATA = False
    RECOMPUTE_WITH_OTTER = True
if USE_PRECOMPUTED_DATA == RECOMPUTE_WITH_OTTER:
    raise ValueError("Select exactly one data mode: precomputed or recompute.")

ELEMENT = "Al"
RHO_VALUES_G_CC = (2.7, 8.1, 15.0)
TE_EV = 10.0
TI_EV = 10.0
AA_N_POINTS = 1024
CONTINUUM_WORKERS = 8
QOZ_N_POINTS = 4096
R_MAX_BOHR = 20.0
K_MAX_BOHR_INV = 20.0
HNC_TOL = 1.0e-6
HNC_CLOSURE_TOL = 1.0e-3
# =============================================================================


SCHEMA = "otter_al_rayleigh_weight_10ev_v1"


def repository_root() -> Path:
    """Find this checkout when called directly or by Sphinx-Gallery."""
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    source_file = globals().get("__file__")
    if source_file is not None:
        source = Path(str(source_file)).resolve()
        candidates.extend([source.parent, *source.parents])
    for candidate in candidates:
        is_checkout = (candidate / "src" / "otter" / "__init__.py").is_file()
        has_archive = (
            candidate / "benchmarks" / "baselines" / "al_rayleigh_weight_10ev"
        ).is_dir()
        if is_checkout and (has_archive or candidate.name == "otter"):
            return candidate
    raise FileNotFoundError("Cannot locate the Otter repository root.")


ROOT = repository_root()
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / "al_rayleigh_weight_10ev"
OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / "al_rayleigh_weight_10ev"
FIGURE_DIR = OUTPUT_DIR / "figures"
RECOMPUTED_PATH = OUTPUT_DIR / "recomputed" / "Al_Te10eV_rayleigh.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workflow_config(rho_g_cc: float) -> PlasmaWorkflowConfig:
    """Build the public Otter QM -> QOZ/HNC configuration for one density."""
    return PlasmaWorkflowConfig(
        elements=[ELEMENT],
        temperature_ev=TE_EV,
        ion_temperature_ev=TI_EV,
        rho_g_cc=float(rho_g_cc),
        electronic_model="qm",
        aa_overrides={
            "n_points": AA_N_POINTS,
            "cont_n_jobs": CONTINUUM_WORKERS,
            "cont_shards": 2 * CONTINUUM_WORKERS,
            "bound_occ_mode": "fd",
            "bound_rmax_mult": None,
            "bound_zero_tail_refine": False,
            "b3_tail_model": "full",
        },
        qoz_linear_n_points=QOZ_N_POINTS,
        qoz_pad_factor=2.0,
        qoz_zbar_mode="pseudoatom_partition",
        qoz_renormalize_nscr_to_zbar=True,
        qoz_response_chi0_model="lindhard_fd",
        qoz_response_lfc_model="chabrier1990",
        hnc_tol=HNC_TOL,
        hnc_closure_transform_tol=HNC_CLOSURE_TOL,
        hnc_max_iter=1000,
        hnc_require_converged=True,
        show_progress=False,
    )


def pack_workflow(workflow: dict[str, Any], rho_g_cc: float, elapsed_s: float) -> dict[str, np.ndarray]:
    """Extract q, f, S and W from one completed Otter workflow."""
    exported = build_state_arrays(
        workflow,
        options=StateExportOptions(
            r_max_bohr=R_MAX_BOHR,
            k_max_bohr_inv=K_MAX_BOHR_INV,
        ),
    )
    k = np.asarray(exported["k_bohr_inv"], dtype=float)
    q = np.asarray(exported["q_k"], dtype=float)[0]
    f = np.asarray(exported["f_k"], dtype=float)[0]
    sii = np.asarray(exported["sij_k"], dtype=float)[0, 0]
    if not (k.shape == q.shape == f.shape == sii.shape):
        raise ValueError("q, f, Sii and k do not share a reciprocal grid.")
    weight = np.abs(q + f) ** 2 * sii
    if not np.all(np.isfinite(weight)) or np.any(weight < -1.0e-10):
        raise RuntimeError("Rayleigh weight is not finite and non-negative.")
    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"])
    return {
        "schema_version": np.asarray(SCHEMA),
        "element": np.asarray(ELEMENT),
        "rho_g_cc": np.asarray(float(rho_g_cc)),
        "te_ev": np.asarray(TE_EV),
        "ti_ev": np.asarray(TI_EV),
        "k_bohr_inv": k,
        "q_k": q,
        "f_k": f,
        "sii_k": sii,
        "rayleigh_weight": weight,
        "mu_ha": np.asarray(float(electronic["mu"])),
        "zbar": np.asarray(float(ion["zbar_qoz"])),
        "elapsed_s": np.asarray(float(elapsed_s)),
    }


def calculate_states() -> list[dict[str, np.ndarray]]:
    """Run the three independent Otter calculations and write a portable NPZ."""
    states: list[dict[str, np.ndarray]] = []
    started = time.perf_counter()
    for rho in RHO_VALUES_G_CC:
        print(f"[compute] Al, rho={rho:g} g/cc, Te=Ti={TE_EV:g} eV")
        state_started = time.perf_counter()
        workflow = solve_plasma_workflow(workflow_config(rho))
        state = pack_workflow(workflow, rho, time.perf_counter() - state_started)
        states.append(state)
        print(
            f"  mu={float(state['mu_ha']):.8f} Ha, "
            f"Zbar={float(state['zbar']):.6f}, "
            f"HNC={float(workflow['ion']['hnc_best_residual']):.3e}"
        )
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA),
        "state_count": np.asarray(len(states)),
        "rho_values_g_cc": np.asarray(RHO_VALUES_G_CC),
        "producer_elapsed_s": np.asarray(time.perf_counter() - started),
    }
    for index, state in enumerate(states):
        for key, value in state.items():
            if key == "schema_version":
                continue
            payload[f"state_{index}_{key}"] = np.asarray(value)
    RECOMPUTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RECOMPUTED_PATH, **payload)
    print(f"Saved newly calculated states: {RECOMPUTED_PATH}")
    return states


def load_states() -> list[dict[str, np.ndarray]]:
    """Verify the checksummed archive and unpack its three states."""
    manifest = json.loads(
        (BASELINE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    record = manifest["state"]
    archive_path = BASELINE_DIR / str(record["data_file"])
    if sha256_file(archive_path) != str(record["data_sha256"]):
        raise RuntimeError(f"Checksum mismatch for {archive_path}.")
    with np.load(archive_path, allow_pickle=False) as archive:
        if str(archive["schema_version"].item()) != SCHEMA:
            raise ValueError("Unsupported Al Rayleigh-weight archive schema.")
        count = int(archive["state_count"])
        states = []
        for index in range(count):
            prefix = f"state_{index}_"
            state = {
                key[len(prefix):]: np.asarray(archive[key])
                for key in archive.files
                if key.startswith(prefix)
            }
            states.append(state)
    return states


states = calculate_states() if RECOMPUTE_WITH_OTTER else load_states()
print(
    "Using "
    + (
        "newly calculated Otter states."
        if RECOMPUTE_WITH_OTTER
        else "the checksummed current-Otter states."
    )
)
for state in states:
    print(
        f"rho={float(state['rho_g_cc']):g} g/cc: "
        f"mu={float(state['mu_ha']):.8f} Ha, "
        f"max W={float(np.max(state['rayleigh_weight'])):.6e}"
    )


with style_context("thesis", palette="bing"):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=grid_figsize(2, 2, cell_width=4.6, cell_height=3.0),
        squeeze=False,
    )
    colours = ("#2E5EAA", "#E76F51", "#23BB62")
    for state, colour in zip(states, colours, strict=True):
        rho = float(state["rho_g_cc"])
        label = rf"$\rho={rho:g}\ \mathrm{{g\,cm^{{-3}}}}$"
        k = np.asarray(state["k_bohr_inv"], dtype=float)
        axes[0, 0].plot(k, state["q_k"], color=colour, label=label + r", $q$")
        axes[0, 0].plot(k, state["f_k"], color=colour, ls="--", label=label + r", $f$")
        axes[0, 1].plot(k, state["sii_k"], color=colour, label=label)
        axes[1, 0].plot(k, state["q_k"] + state["f_k"], color=colour, label=label)
        axes[1, 1].plot(k, state["rayleigh_weight"], color=colour, label=label)

    axes[0, 0].set(title=r"Form-factor components", ylabel=r"$q(k),\ f(k)$")
    axes[0, 1].set(title=r"Ionic structure", ylabel=r"$S_{ii}(k)$")
    axes[1, 0].set(title=r"Total form factor", ylabel=r"$q(k)+f(k)$")
    axes[1, 1].set(title=r"Rayleigh weight", ylabel=r"$W(k)$")
    for axis in axes.flat:
        axis.set_xlabel(r"$k$ [Bohr$^{-1}$]")
        axis.set_xlim(-0.05, 12.0)
        axis.legend(frameon=False)
    fig.suptitle(
        r"Al: static Rayleigh weight, $T_e=T_i=10$ eV",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    # Keep the figure open so Sphinx-Gallery embeds the same multi-density
    # result on the ``Al: elastic Rayleigh weight versus density`` HTML page.
    # ``save_figure`` still writes the slide-ready PNG/PDF copies; the open
    # figure is only additionally used by the documentation renderer.
    save_figure(fig, FIGURE_DIR / "al_rayleigh_weight_10ev", close=False)
