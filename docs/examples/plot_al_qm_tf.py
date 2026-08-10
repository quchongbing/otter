"""
Aluminium KS-DFT and Thomas--Fermi comparison
==============================================

This is a complete, directly executable Otter example.  It compares
Kohn--Sham density-functional theory (KS-DFT, internally ``qm``) and
finite-temperature Thomas--Fermi (TF) average atoms for aluminium at
``rho=8.1 g/cc`` and ``T=1, 15, 50, 100 eV``.  Both electronic models feed
the same ion-sphere pseudoatom, QOZ/HNC, and Chabrier-1990 LFC construction.

Only the parameters in the input block below need to be edited.
``USE_PRECOMPUTED_DATA = True`` verifies and loads reviewed NPZ files.
Set ``RECOMPUTE_WITH_OTTER = True`` (and
``USE_PRECOMPUTED_DATA = False``) to make this same file construct
:class:`otter.PlasmaWorkflowConfig` objects, call
:func:`otter.solve_plasma_workflow`, save new NPZ files under
``benchmarks/outputs/al_qm_tf/gallery_recomputed``, and plots them.  It does
not import another benchmark runner or producer.

The model follows :cite:t:`StarrettSaumon2014` and the finite-temperature
jellium LFC follows :cite:t:`Chabrier1990`.  See :doc:`the model-comparison
notes </user_guide/al_qm_tf>` for convergence criteria, units, and
interpretation.
The two figures are exported as matching PNG and vector PDF files under
``benchmarks/outputs/al_qm_tf/figures`` for documentation and slides.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from otter import PlasmaWorkflowConfig, solve_plasma_workflow
from otter.plotting import grid_figsize, save_figure, style_context


# =============================================================================
# User input
# =============================================================================
USE_PRECOMPUTED_DATA = True
RECOMPUTE_WITH_OTTER = False
if os.environ.get("OTTER_RECOMPUTE_AL_QM_TF", "0") == "1":
    USE_PRECOMPUTED_DATA = False
    RECOMPUTE_WITH_OTTER = True

if USE_PRECOMPUTED_DATA == RECOMPUTE_WITH_OTTER:
    raise ValueError(
        "Select exactly one data mode: precomputed or recompute with Otter."
    )

RHO_G_CC = 8.1
TEMPERATURES_EV = (1.0, 15.0, 50.0, 100.0)
ELECTRONIC_MODELS = ("qm", "tf")

# Four state workers x four continuum workers use at most about 16 workers.
MAX_STATE_WORKERS = 4
CONTINUUM_WORKERS_PER_STATE = 4
AA_N_POINTS = 1024
QOZ_N_POINTS = 4096

HNC_TOL = 1.0e-6
HNC_CLOSURE_TOL = 1.0e-3
R_RETAIN_MAX_BOHR = 20.0
K_RETAIN_MAX_BOHR_INV = 20.0
# =============================================================================


def repository_root() -> Path:
    """Locate this source checkout when run directly or by Sphinx-Gallery."""
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
            / "al_qm_tf"
            / "manifest.json"
        ).is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the Otter checkout.")


ROOT = repository_root()
PRECOMPUTED_DIR = ROOT / "benchmarks" / "baselines" / "al_qm_tf"
OUTPUT_DIR = (
    ROOT / "benchmarks" / "outputs" / "al_qm_tf" / "gallery_recomputed"
)
FIGURE_DIR = ROOT / "benchmarks" / "outputs" / "al_qm_tf" / "figures"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load one pickle-free numerical state."""
    with np.load(path, allow_pickle=False) as archive:
        state = {key: np.asarray(archive[key]) for key in archive.files}
    object_keys = [key for key, value in state.items() if value.dtype.hasobject]
    if object_keys:
        raise TypeError(f"Object arrays are forbidden in {path}: {object_keys}")
    return state


def load_precomputed_states() -> list[dict[str, np.ndarray]]:
    """Verify the reviewed manifest and load the four accepted states."""
    manifest = json.loads(
        (PRECOMPUTED_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("benchmark_id") != "al_qm_tf":
        raise ValueError("The selected manifest is not the Al QM/TF benchmark.")
    states: list[dict[str, np.ndarray]] = []
    for record in manifest["states"]:
        path = PRECOMPUTED_DIR / str(record["data_file"])
        if sha256_file(path) != str(record["data_sha256"]):
            raise RuntimeError(f"Checksum mismatch for {path}.")
        state = load_npz(path)
        if tuple(str(value) for value in state["model_labels"]) != (
            "qm",
            "tf",
        ):
            raise ValueError(f"Unexpected electronic-model order in {path}.")
        states.append(state)
    return states


def workflow_config(temperature_ev: float, model: str) -> PlasmaWorkflowConfig:
    """Build the complete public Otter calculation for one model/state."""
    return PlasmaWorkflowConfig(
        elements=["Al"],
        temperature_ev=float(temperature_ev),
        ion_temperature_ev=float(temperature_ev),
        rho_g_cc=float(RHO_G_CC),
        electronic_model=str(model),
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
        qoz_response_lfc_model="chabrier1990",
        hnc_tol=float(HNC_TOL),
        hnc_closure_transform_tol=float(HNC_CLOSURE_TOL),
        hnc_max_iter=1000,
        hnc_require_converged=True,
        show_progress=False,
    )


def solve_model(temperature_ev: float, model: str) -> dict[str, object]:
    """Run AA -> external pseudoatom -> QOZ/HNC for one electronic model."""
    started = time.perf_counter()
    workflow = solve_plasma_workflow(workflow_config(temperature_ev, model))
    elapsed_s = time.perf_counter() - started
    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"])
    ext_status = dict(electronic.get("ext_status", {}))
    if electronic.get("stage2_converged") is not True:
        raise RuntimeError(f"{model}: full average atom did not converge.")
    if ext_status.get("converged") is not True:
        raise RuntimeError(f"{model}: external average atom did not converge.")
    if ion.get("hnc_converged") is not True:
        raise RuntimeError(f"{model}: HNC did not reach a physical root.")
    if float(ion["hnc_output_residual"]) > HNC_TOL:
        raise RuntimeError(f"{model}: HNC residual exceeds tolerance.")
    if float(ion["closure_transform_max_abs"]) > HNC_CLOSURE_TOL:
        raise RuntimeError(f"{model}: g/S transform closure exceeds tolerance.")
    return {
        "model": model,
        "elapsed_s": elapsed_s,
        "electronic": electronic,
        "ion": ion,
    }


def trim_grid(
    coordinate: np.ndarray,
    values: np.ndarray,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate = np.asarray(coordinate, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = coordinate <= maximum
    return coordinate[mask], values[..., mask]


def combine_live_models(
    temperature_ev: float,
    solved: dict[str, dict[str, object]],
) -> dict[str, np.ndarray]:
    """Convert two live workflows to the same compact arrays used by plotting."""
    qm_ion = dict(solved["qm"]["ion"])
    r_ion, _ = trim_grid(
        qm_ion["r"], qm_ion["gii_r"], R_RETAIN_MAX_BOHR
    )
    k, _ = trim_grid(
        qm_ion["k"], qm_ion["sii_k"], K_RETAIN_MAX_BOHR_INV
    )
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray("otter_gallery_al_qm_tf_v1"),
        "rho_g_cc": np.asarray(RHO_G_CC),
        "temperature_ev": np.asarray(temperature_ev),
        "model_labels": np.asarray(ELECTRONIC_MODELS),
        "r_ion_bohr": r_ion,
        "k_bohr_inv": k,
    }
    gii: list[np.ndarray] = []
    sii: list[np.ndarray] = []
    zbar: list[float] = []
    elapsed: list[float] = []
    n0: list[float] = []
    for model in ELECTRONIC_MODELS:
        entry = solved[model]
        electronic = dict(entry["electronic"])
        ion = dict(entry["ion"])
        source_r = np.asarray(ion["r"], dtype=float)
        source_k = np.asarray(ion["k"], dtype=float)
        gii.append(np.interp(r_ion, source_r, ion["gii_r"]))
        sii.append(np.interp(k, source_k, ion["sii_k"]))
        zbar.append(float(ion["zbar_partition"]))
        elapsed.append(float(entry["elapsed_s"]))
        n0.append(float(electronic["n0"]))
        r_e, n_full = trim_grid(
            electronic["r"], electronic["n_full"], R_RETAIN_MAX_BOHR
        )
        _, n_scr = trim_grid(
            electronic["r"], electronic["n_scr"], R_RETAIN_MAX_BOHR
        )
        payload[f"r_{model}_bohr"] = r_e
        payload[f"n_full_{model}_bohr3"] = n_full
        payload[f"n_scr_{model}_bohr3"] = n_scr
    payload["gii_r"] = np.asarray(gii)
    payload["sii_k"] = np.asarray(sii)
    payload["zbar_partition"] = np.asarray(zbar)
    payload["producer_elapsed_s"] = np.asarray(elapsed)
    payload["n0_bohr3"] = np.asarray(n0)
    return payload


def solve_all_states() -> list[dict[str, np.ndarray]]:
    """Run all eight independent workflows and save compact gallery results."""
    jobs = [
        (float(temperature), model)
        for temperature in TEMPERATURES_EV
        for model in ELECTRONIC_MODELS
    ]
    solved: dict[tuple[float, str], dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=MAX_STATE_WORKERS) as pool:
        future_map = {
            pool.submit(solve_model, temperature, model): (temperature, model)
            for temperature, model in jobs
        }
        for future in as_completed(future_map):
            key = future_map[future]
            solved[key] = future.result()
            print(f"[computed] Al, T={key[0]:g} eV, model={key[1]}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    states: list[dict[str, np.ndarray]] = []
    for temperature in TEMPERATURES_EV:
        state = combine_live_models(
            float(temperature),
            {
                model: solved[(float(temperature), model)]
                for model in ELECTRONIC_MODELS
            },
        )
        token = f"{float(temperature):g}".replace(".", "p")
        path = OUTPUT_DIR / f"Al_rho8p1gcc_T{token}eV_qm_tf.npz"
        np.savez_compressed(path, **state)
        print(f"[saved] {path}")
        states.append(state)
    return states


states = (
    load_precomputed_states()
    if USE_PRECOMPUTED_DATA
    else solve_all_states()
    if RECOMPUTE_WITH_OTTER
    else []
)

print(
    "Using "
    + (
        "checksummed, precomputed Otter results."
        if USE_PRECOMPUTED_DATA
        else "new results calculated directly by this gallery script."
    )
)
print(f"{'rho':>6s} {'T[eV]':>7s} {'dZ(TF-KS)':>11s} {'RMSE(g)':>11s} {'RMSE(S)':>11s}")
for state in states:
    dz = float(np.diff(np.asarray(state["zbar_partition"], dtype=float))[0])
    dg = np.diff(np.asarray(state["gii_r"], dtype=float), axis=0)[0]
    ds = np.diff(np.asarray(state["sii_k"], dtype=float), axis=0)[0]
    r_metric = np.asarray(state["r_ion_bohr"], dtype=float) <= 12.0
    k_metric = np.asarray(state["k_bohr_inv"], dtype=float) <= 6.0
    print(
        f"{float(state['rho_g_cc']):6.2f} "
        f"{float(state['temperature_ev']):7.1f} "
        f"{dz:11.5f} "
        f"{np.sqrt(np.mean(dg[r_metric]**2)):11.5f} "
        f"{np.sqrt(np.mean(ds[k_metric]**2)):11.5f}"
    )


# %%
# The density panels use radial shell densities, :math:`4\pi r^2 n(r)`, so
# their signed outer-shell structure remains visible beside the ionic
# observables.  Solid lines are KS-DFT and dashed lines are TF.

models = ELECTRONIC_MODELS
labels = {"qm": "KS-DFT", "tf": "Thomas–Fermi"}

plot_style = ExitStack()
plot_style.enter_context(style_context("thesis", palette="bing"))
colors = dict(
    zip(models, plt.rcParams["axes.prop_cycle"].by_key()["color"])
)
styles = {"qm": "-", "tf": "--"}
fig_density, density_axes = plt.subplots(
    len(states),
    2,
    figsize=grid_figsize(
        len(states),
        2,
        cell_width=5.0,
        cell_height=2.2,
    ),
    squeeze=False,
)
for row, state in enumerate(states):
    for model_index, model in enumerate(models):
        r_e = np.asarray(state[f"r_{model}_bohr"], dtype=float)
        shell_weight = 4.0 * np.pi * r_e**2
        n0 = float(np.asarray(state["n0_bohr3"])[model_index])
        density_mask = r_e <= 8.0
        screening_mask = r_e <= 12.0
        density_axes[row, 0].plot(
            r_e[density_mask],
            (
                shell_weight
                * (
                    np.asarray(
                        state[f"n_full_{model}_bohr3"],
                        dtype=float,
                    )
                    - n0
                )
            )[density_mask],
            styles[model],
            color=colors[model],
            label=f"{labels[model]}: full-$n_0$",
        )
        density_axes[row, 1].plot(
            r_e[screening_mask],
            (
                shell_weight
                * np.asarray(state[f"n_scr_{model}_bohr3"], dtype=float)
            )[screening_mask],
            styles[model],
            color=colors[model],
            label=labels[model],
        )

    rho = float(state["rho_g_cc"])
    temperature = float(state["temperature_ev"])
    density_axes[row, 0].set_ylabel(
        rf"{rho:g} g cm$^{{-3}}$, {temperature:g} eV"
        + "\n"
        + r"$4\pi r^2 n(r)$ [Bohr$^{-1}$]"
    )
    density_axes[row, 0].set_xlim(-0.5, 4.0)
    density_axes[row, 1].set_xlim(-0.5, 12.0)

density_titles = (r"$n_{\rm full}-n_0$", "screening density")
for axis, title in zip(density_axes[0], density_titles, strict=True):
    axis.set_title(title)
    axis.legend(frameon=False)
for axis in density_axes[-1]:
    axis.set_xlabel(r"$r$ [Bohr]")
density_axes[0, 1].set_ylabel(
    r"$4\pi r^2 n_{\rm scr}$ [Bohr$^{-1}$]"
)
fig_density.suptitle(
    "Aluminium electronic structure: KS-DFT versus Thomas–Fermi",
    y=0.985,
)
fig_density.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))


# %%
# The same electronic outputs are then propagated through the common
# pseudoatom/QOZ/HNC construction.

fig_ionic, ionic_axes = plt.subplots(
    len(states),
    2,
    figsize=grid_figsize(
        len(states),
        2,
        cell_width=5.0,
        cell_height=2.2,
    ),
    squeeze=False,
)
for row, state in enumerate(states):
    r_ion = np.asarray(state["r_ion_bohr"], dtype=float)
    k = np.asarray(state["k_bohr_inv"], dtype=float)
    for model_index, model in enumerate(models):
        ionic_axes[row, 0].plot(
            r_ion,
            np.asarray(state["gii_r"], dtype=float)[model_index],
            styles[model],
            color=colors[model],
            label=labels[model],
        )
        ionic_axes[row, 1].plot(
            k,
            np.asarray(state["sii_k"], dtype=float)[model_index],
            styles[model],
            color=colors[model],
            label=labels[model],
        )
    rho = float(state["rho_g_cc"])
    temperature = float(state["temperature_ev"])
    ionic_axes[row, 0].set_ylabel(
        rf"{rho:g} g cm$^{{-3}}$, {temperature:g} eV"
        + "\n"
        + r"$g_{ii}(r)$"
    )
    ionic_axes[row, 0].set_xlim(-0.5, 12.0)
    ionic_axes[row, 1].set_xlim(0.0, 6.0)
    ionic_axes[row, 1].axhline(1.0, color="0.45", lw=0.8, ls=":")
ionic_axes[0, 0].set_title(r"$g_{ii}(r)$")
ionic_axes[0, 1].set_title(r"$S_{ii}(k)$")
ionic_axes[0, 0].legend(frameon=False)
ionic_axes[0, 1].legend(frameon=False)
ionic_axes[-1, 0].set_xlabel(r"$r$ [Bohr]")
ionic_axes[-1, 1].set_xlabel(r"$k$ [Bohr$^{-1}$]")
ionic_axes[0, 1].set_ylabel(r"$S_{ii}$")
fig_ionic.suptitle(
    "Aluminium ionic structure: KS-DFT versus Thomas–Fermi",
    y=0.985,
)
fig_ionic.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

save_figure(
    fig_density,
    FIGURE_DIR / "al_qm_tf_electronic_structure",
    close=False,
)
save_figure(
    fig_ionic,
    FIGURE_DIR / "al_qm_tf_ionic_structure",
    close=False,
)
plot_style.close()

if "agg" not in plt.get_backend().lower():
    plt.show()
