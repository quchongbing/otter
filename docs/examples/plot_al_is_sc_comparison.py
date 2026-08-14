r"""
Experimental IS/SC feedback for aluminium
=========================================

This example compares the ion-sphere (IS) construction with Otter's
**experimental** self-consistent (SC) average-atom/QOZ feedback for aluminium
at :math:`\rho=8.1\ {\rm g\,cm^{-3}}` and
:math:`T_e=T_i=15\ {\rm eV}`.  Both orbital Kohn--Sham DFT (KS-DFT) and
finite-temperature Thomas--Fermi (TF) electrons are shown.

The SC loop is based on Sec. 2.4 and Eqs. (19)--(20) of
:cite:t:`StarrettSaumon2014` (doi:
`10.1016/j.hedp.2013.12.001
<https://doi.org/10.1016/j.hedp.2013.12.001>`_).  It feeds the ion structure
and ion--electron correlation potential back into the electronic calculation
while keeping the converged IS chemical potential fixed.  Otter labels this
path experimental because validation across a broader state space is still
in progress.

Set ``RECOMPUTE_WITH_OTTER = True`` below to calculate every curve directly
with the public Otter API.  The default ``False`` path loads the reviewed,
checksummed Otter result so that the gallery builds quickly.  A fresh result
is saved separately and never overwrites the reviewed baseline.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from otter import PlasmaWorkflowConfig, solve_plasma_workflow
from otter.experimental import SCFeedbackConfig, solve_sc_feedback_workflow
from otter.plotting import (
    MODEL_STYLES,
    PALETTES,
    grid_figsize,
    save_figure,
    style_context,
)


# %%
# User controls
# -------------
# ``RECOMPUTE_WITH_OTTER`` is the only switch required for a normal run.
# Two independent electronic models may be evaluated concurrently.  Reduce
# either worker count on a small machine.

RECOMPUTE_WITH_OTTER = False
if os.environ.get("OTTER_RECOMPUTE_AL_IS_SC", "0") == "1":
    RECOMPUTE_WITH_OTTER = True
RECOMPUTE_MODEL_WORKERS = 2
CONTINUUM_WORKERS = 8

ELEMENT = "Al"
RHO_G_CC = 8.1
TE_EV = 15.0
TI_EV = 15.0
MODELS = ("qm", "tf")
MODEL_DISPLAY_LABELS = ("KS-DFT", "Thomas--Fermi")
STRUCTURES = ("is", "sc")

HNC_TOL = 1.0e-4
HNC_CLOSURE_TOL = 2.5e-3
R_RETAIN_MAX_BOHR = 20.0
K_RETAIN_MAX_BOHR_INV = 20.0
SC_CONTROLS = SCFeedbackConfig(
    max_outer=16,
    g_tol=5.0e-4,
    v_corr_tol=5.0e-4,
    v_corr_mix=0.5,
    require_converged=True,
)

SCHEMA = "otter_al_is_sc_comparison_v1"
HARTREE_TO_EV = 27.211386245988


def _repository_root() -> Path:
    """Locate the repository from either the source tree or a gallery build."""
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
            / "al_is_sc_comparison"
            / "manifest.json"
        ).is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the Otter repository root.")


ROOT = _repository_root()
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / "al_is_sc_comparison"
MANIFEST_PATH = BASELINE_DIR / "manifest.json"
OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / "al_is_sc_comparison"
CANDIDATE_PATH = (
    OUTPUT_DIR
    / "recomputed"
    / "Al_rho8p1gcc_Te15eV_Ti15eV_qm_tf_is_sc.npz"
)
FIGURE_DIR = OUTPUT_DIR / "figures"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workflow_config(model: str) -> PlasmaWorkflowConfig:
    """Return the complete IS workflow used to initialise the SC loop."""
    model_key = str(model).strip().lower()
    if model_key not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}.")
    return PlasmaWorkflowConfig(
        elements=[ELEMENT],
        temperature_ev=TE_EV,
        ion_temperature_ev=TI_EV,
        rho_g_cc=RHO_G_CC,
        electronic_model=model_key,
        aa_overrides={
            "cont_n_jobs": CONTINUUM_WORKERS,
            "cont_shards": 2 * CONTINUUM_WORKERS,
        },
        hnc_closure_transform_tol=HNC_CLOSURE_TOL,
        hnc_max_iter=500,
    )


def _finite_bound_levels(electronic: dict[str, Any]) -> dict[str, np.ndarray]:
    """Flatten finite negative-energy KS levels into numeric arrays."""
    energies = np.asarray(
        electronic.get("bound_energy_ha", np.empty((0, 0))),
        dtype=float,
    )
    l_values = np.asarray(
        electronic.get("bound_l_list", np.arange(energies.shape[0])),
        dtype=int,
    )
    fd = np.asarray(
        electronic.get("bound_fd", np.zeros_like(energies)),
        dtype=float,
    )
    occupations = np.asarray(
        electronic.get("bound_occ_deg_fd", np.zeros_like(energies)),
        dtype=float,
    )
    records: list[tuple[int, int, float, float, float]] = []
    for l_index in range(energies.shape[0]):
        for n_index in range(energies.shape[1]):
            energy = float(energies[l_index, n_index])
            if np.isfinite(energy) and energy < 0.0:
                records.append(
                    (
                        int(l_values[l_index]),
                        int(n_index + 1),
                        energy,
                        float(fd[l_index, n_index]),
                        float(occupations[l_index, n_index]),
                    )
                )
    if not records:
        return {
            "bound_l": np.empty(0, dtype=int),
            "bound_n_index": np.empty(0, dtype=int),
            "bound_energy_ha": np.empty(0),
            "bound_fd": np.empty(0),
            "bound_occ_deg_fd": np.empty(0),
        }
    values = np.asarray(records, dtype=float)
    return {
        "bound_l": values[:, 0].astype(int),
        "bound_n_index": values[:, 1].astype(int),
        "bound_energy_ha": values[:, 2],
        "bound_fd": values[:, 3],
        "bound_occ_deg_fd": values[:, 4],
    }


def _extract_path(workflow: dict[str, Any]) -> dict[str, Any]:
    """Extract the quantities retained by this example."""
    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"])
    r = np.asarray(ion["r"], dtype=float)
    k = np.asarray(ion["k"], dtype=float)
    r_mask = r <= R_RETAIN_MAX_BOHR
    k_mask = k <= K_RETAIN_MAX_BOHR_INV
    return {
        "r": r[r_mask],
        "k": k[k_mask],
        "gii": np.asarray(ion["gii_r"], dtype=float)[r_mask],
        "sii": np.asarray(ion["sii_k"], dtype=float)[k_mask],
        "mu": float(electronic["mu"]),
        "zbar": float(ion["zbar_partition"]),
        "hnc_residual": float(ion["hnc_output_residual"]),
        "levels": _finite_bound_levels(electronic),
    }


def _solve_model(model: str) -> dict[str, Any]:
    """Calculate one IS result and continue it to experimental SC."""
    config = workflow_config(model)
    started = time.perf_counter()
    is_workflow = solve_plasma_workflow(config)
    is_elapsed_s = time.perf_counter() - started

    started = time.perf_counter()
    sc_workflow = solve_sc_feedback_workflow(
        config,
        is_workflow,
        feedback_cfg=SC_CONTROLS,
    )
    sc_extension_elapsed_s = time.perf_counter() - started
    feedback = dict(sc_workflow["sc_feedback"])
    history = list(feedback["history"])
    return {
        "is": _extract_path(is_workflow),
        "sc": _extract_path(sc_workflow),
        "is_elapsed_s": float(is_elapsed_s),
        "sc_extension_elapsed_s": float(sc_extension_elapsed_s),
        "sc_total_elapsed_s": float(
            is_elapsed_s + sc_extension_elapsed_s
        ),
        "sc_converged": bool(feedback["converged"]),
        "sc_iterations": int(feedback["iterations"]),
        "fixed_is_mu_ha": float(feedback["fixed_is_mu_ha"]),
        "history_iteration": np.asarray(
            [int(item["iteration"]) for item in history],
            dtype=int,
        ),
        "history_max_g_change": np.asarray(
            [float(item["max_g_change"]) for item in history],
            dtype=float,
        ),
        "history_max_v_corr_change_ha": np.asarray(
            [float(item["max_v_corr_change_ha"]) for item in history],
            dtype=float,
        ),
    }


def _interpolate(
    target: np.ndarray,
    source: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(
        np.asarray(target, dtype=float),
        np.asarray(source, dtype=float),
        np.asarray(values, dtype=float),
    )


def _pack_state(by_model: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """Pack recomputed results in the reviewed, pickle-free schema."""
    r = np.asarray(by_model["qm"]["is"]["r"], dtype=float)
    k = np.asarray(by_model["qm"]["is"]["k"], dtype=float)
    gii = np.empty((len(MODELS), len(STRUCTURES), r.size), dtype=float)
    sii = np.empty((len(MODELS), len(STRUCTURES), k.size), dtype=float)
    mu = np.empty((len(MODELS), len(STRUCTURES)), dtype=float)
    zbar = np.empty_like(mu)
    hnc_residual = np.empty_like(mu)

    for model_index, model in enumerate(MODELS):
        for structure_index, structure in enumerate(STRUCTURES):
            path = by_model[model][structure]
            gii[model_index, structure_index] = _interpolate(
                r, path["r"], path["gii"]
            )
            sii[model_index, structure_index] = _interpolate(
                k, path["k"], path["sii"]
            )
            mu[model_index, structure_index] = float(path["mu"])
            zbar[model_index, structure_index] = float(path["zbar"])
            hnc_residual[model_index, structure_index] = float(
                path["hnc_residual"]
            )

    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA),
        "example_id": np.asarray("al_is_sc_comparison"),
        "element_symbol": np.asarray(ELEMENT),
        "rho_g_cc": np.asarray(RHO_G_CC),
        "te_ev": np.asarray(TE_EV),
        "ti_ev": np.asarray(TI_EV),
        "model_labels": np.asarray(MODELS),
        "model_display_labels": np.asarray(MODEL_DISPLAY_LABELS),
        "structure_labels": np.asarray(STRUCTURES),
        "sc_status": np.asarray("experimental"),
        "sc_reference": np.asarray(
            "Starrett and Saumon 2014 Sec. 2.4 Eqs. (19)-(20)"
        ),
        "r_bohr": r,
        "k_bohr_inv": k,
        "gii_r": gii,
        "sii_k": sii,
        "mu_ha": mu,
        "zbar_partition": zbar,
        "hnc_residual": hnc_residual,
        "is_elapsed_s": np.asarray(
            [by_model[model]["is_elapsed_s"] for model in MODELS]
        ),
        "sc_extension_elapsed_s": np.asarray(
            [
                by_model[model]["sc_extension_elapsed_s"]
                for model in MODELS
            ]
        ),
        "sc_total_elapsed_s": np.asarray(
            [by_model[model]["sc_total_elapsed_s"] for model in MODELS]
        ),
        "sc_converged": np.asarray(
            [by_model[model]["sc_converged"] for model in MODELS],
            dtype=bool,
        ),
        "sc_iterations": np.asarray(
            [by_model[model]["sc_iterations"] for model in MODELS],
            dtype=int,
        ),
        "fixed_is_mu_ha": np.asarray(
            [by_model[model]["fixed_is_mu_ha"] for model in MODELS]
        ),
        "tf_discrete_ks_levels_defined": np.asarray(False),
    }
    for model in MODELS:
        for structure in STRUCTURES:
            levels = by_model[model][structure]["levels"]
            for name, values in levels.items():
                payload[f"{model}_{structure}_{name}"] = np.asarray(values)
        payload[f"{model}_sc_history_iteration"] = np.asarray(
            by_model[model]["history_iteration"]
        )
        payload[f"{model}_sc_history_max_g_change"] = np.asarray(
            by_model[model]["history_max_g_change"]
        )
        payload[f"{model}_sc_history_max_v_corr_change_ha"] = np.asarray(
            by_model[model]["history_max_v_corr_change_ha"]
        )
    return payload


def recompute_state() -> dict[str, np.ndarray]:
    """Run the public Otter API and stage a fresh local candidate."""
    workers = max(1, min(int(RECOMPUTE_MODEL_WORKERS), len(MODELS)))
    solved: dict[str, dict[str, Any]] = {}
    if workers == 1:
        for model in MODELS:
            solved[model] = _solve_model(model)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_solve_model, model): model for model in MODELS
            }
            for future in as_completed(futures):
                model = futures[future]
                solved[model] = future.result()
                print(
                    f"[computed] {MODEL_DISPLAY_LABELS[MODELS.index(model)]}: "
                    f"IS={solved[model]['is_elapsed_s']:.2f} s, "
                    "SC extension="
                    f"{solved[model]['sc_extension_elapsed_s']:.2f} s"
                )
    state = _pack_state(solved)
    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CANDIDATE_PATH, **state)
    print(f"Saved fresh candidate: {CANDIDATE_PATH}")
    return state


def validate_state(state: dict[str, np.ndarray]) -> None:
    """Reject an incompatible or unconverged stored result."""
    if str(state["schema_version"].item()) != SCHEMA:
        raise ValueError("Unsupported Al IS/SC state schema.")
    if tuple(state["model_labels"].tolist()) != MODELS:
        raise ValueError("Expected QM and TF model ordering.")
    if tuple(state["structure_labels"].tolist()) != STRUCTURES:
        raise ValueError("Expected IS and SC structure ordering.")
    if str(state["sc_status"].item()) != "experimental":
        raise ValueError("SC must remain explicitly labelled experimental.")
    if not np.all(np.asarray(state["sc_converged"], dtype=bool)):
        raise ValueError("An experimental SC calculation is unconverged.")
    if np.any(np.asarray(state["hnc_residual"], dtype=float) > HNC_TOL):
        raise ValueError("An HNC residual exceeds the documented tolerance.")
    r = np.asarray(state["r_bohr"], dtype=float)
    k = np.asarray(state["k_bohr_inv"], dtype=float)
    if (
        r.ndim != 1
        or k.ndim != 1
        or not np.all(np.diff(r) > 0.0)
        or not np.all(np.diff(k) > 0.0)
    ):
        raise ValueError("Stored r and k grids must be strictly increasing.")
    if np.asarray(state["gii_r"]).shape != (2, 2, r.size):
        raise ValueError("Unexpected gii_r dimensions.")
    if np.asarray(state["sii_k"]).shape != (2, 2, k.size):
        raise ValueError("Unexpected sii_k dimensions.")
    fixed_mu = np.asarray(state["fixed_is_mu_ha"], dtype=float)
    mu = np.asarray(state["mu_ha"], dtype=float)
    np.testing.assert_allclose(mu[:, 0], fixed_mu, atol=2.0e-10, rtol=0.0)
    np.testing.assert_allclose(mu[:, 1], fixed_mu, atol=2.0e-10, rtol=0.0)
    if bool(state["tf_discrete_ks_levels_defined"]):
        raise ValueError("Thomas--Fermi cannot define discrete KS levels.")


def load_reviewed_state() -> dict[str, np.ndarray]:
    """Load and checksum the accepted, project-generated Otter result."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "otter_example_manifest_v1"
        or manifest.get("example_id") != "al_is_sc_comparison"
        or manifest.get("status") != "accepted"
    ):
        raise ValueError("The installed Al IS/SC manifest is not accepted.")
    archive_path = BASELINE_DIR / manifest["state"]["data_file"]
    if _sha256(archive_path) != manifest["state"]["data_sha256"]:
        raise RuntimeError("Al IS/SC baseline checksum mismatch.")
    with np.load(archive_path, allow_pickle=False) as archive:
        state = {key: np.asarray(archive[key]) for key in archive.files}
    return state


def level_rows(
    state: dict[str, np.ndarray],
    structure: str,
) -> list[dict[str, float | str]]:
    """Return the finite KS levels for one structure model."""
    structure_key = str(structure).strip().lower()
    if structure_key not in STRUCTURES:
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
                "energy_ev": HARTREE_TO_EV * float(energy),
                "fd": float(fd),
                "occupation": float(occupation),
            }
        )
    return rows


def print_summary(state: dict[str, np.ndarray]) -> None:
    """Print chemical potentials, ionisation, timings, and KS levels."""
    print(
        "\nAl: rho=8.1 g/cc, Te=Ti=15 eV"
        "\nSC status: experimental; Starrett--Saumon (2014), "
        "Sec. 2.4, Eqs. (19)--(20)."
    )
    print(
        "\nmodel             path      mu [Ha]       Zbar       "
        "HNC residual   wall time [s]"
    )
    print("-" * 82)
    mu = np.asarray(state["mu_ha"], dtype=float)
    zbar = np.asarray(state["zbar_partition"], dtype=float)
    residual = np.asarray(state["hnc_residual"], dtype=float)
    for model_index, model_name in enumerate(
        state["model_display_labels"].tolist()
    ):
        values = (
            (
                "IS",
                float(state["is_elapsed_s"][model_index]),
            ),
            (
                "SC",
                float(state["sc_total_elapsed_s"][model_index]),
            ),
        )
        for structure_index, (path_name, elapsed) in enumerate(values):
            print(
                f"{model_name:<17s} {path_name:<5s} "
                f"{mu[model_index, structure_index]:12.8f} "
                f"{zbar[model_index, structure_index]:10.6f} "
                f"{residual[model_index, structure_index]:14.3e} "
                f"{elapsed:15.2f}"
            )

    is_levels = {row["level"]: row for row in level_rows(state, "is")}
    sc_levels = {row["level"]: row for row in level_rows(state, "sc")}
    labels = list(dict.fromkeys([*is_levels, *sc_levels]))
    print(
        "\nKS-DFT finite bound levels"
        "\nlevel    E_IS [Ha]    E_SC [Ha]    FD_IS      FD_SC"
    )
    print("-" * 57)
    for label in labels:
        is_row = is_levels.get(label)
        sc_row = sc_levels.get(label)
        print(
            f"{label:<6s} "
            f"{float(is_row['energy_ha']) if is_row else np.nan:11.6f} "
            f"{float(sc_row['energy_ha']) if sc_row else np.nan:11.6f} "
            f"{float(is_row['fd']) if is_row else np.nan:10.6f} "
            f"{float(sc_row['fd']) if sc_row else np.nan:10.6f}"
        )
    print(
        "Thomas--Fermi is a semiclassical density model and has no "
        "discrete KS-level table.\n"
    )


state = (
    recompute_state() if RECOMPUTE_WITH_OTTER else load_reviewed_state()
)
validate_state(state)
print(
    "Using a fresh Otter calculation."
    if RECOMPUTE_WITH_OTTER
    else "Using the reviewed, checksummed Otter calculation."
)
print_summary(state)


# %%
# Ion structure
# -------------
# Solid lines are IS; dashed lines are the experimental SC extension.  Rows
# distinguish KS-DFT from TF electrons.

r = np.asarray(state["r_bohr"], dtype=float)
k = np.asarray(state["k_bohr_inv"], dtype=float)
gii = np.asarray(state["gii_r"], dtype=float)
sii = np.asarray(state["sii_k"], dtype=float)
model_names = tuple(state["model_display_labels"].tolist())
structure_names = ("IS", "SC (experimental)")
structure_styles = (
    dict(MODEL_STYLES["is"]),
    dict(MODEL_STYLES["sc"]),
)
r_mask = r <= 10.0
k_mask = k <= 6.0

with style_context("thesis", palette="bing"):
    fig_structure, axes = plt.subplots(
        2,
        2,
        figsize=grid_figsize(2, 2),
        sharex="col",
    )
    for model_index, model_name in enumerate(model_names):
        for structure_index, structure_name in enumerate(structure_names):
            style = structure_styles[structure_index]
            axes[model_index, 0].plot(
                r[r_mask],
                gii[model_index, structure_index, r_mask],
                label=structure_name,
                **style,
            )
            axes[model_index, 1].plot(
                k[k_mask],
                sii[model_index, structure_index, k_mask],
                label=structure_name,
                **style,
            )
        axes[model_index, 0].set_ylabel(
            model_name + "\n" + r"$g_{ii}(r)$"
        )
        axes[model_index, 1].set_ylabel(r"$S_{ii}(k)$")
        axes[model_index, 0].axhline(
            1.0, color="0.55", ls=":", lw=0.9
        )
        axes[model_index, 1].axhline(
            1.0, color="0.55", ls=":", lw=0.9
        )
        axes[model_index, 0].set_xlim(-0.5, 10.0)
        axes[model_index, 1].set_xlim(0.0, 6.0)
        for axis in axes[model_index]:
            axis.legend()
    axes[0, 0].set_title("Pair distribution")
    axes[0, 1].set_title("Static structure factor")
    axes[-1, 0].set_xlabel(r"$r$ [Bohr]")
    axes[-1, 1].set_xlabel(r"$k$ [Bohr$^{-1}$]")
    fig_structure.suptitle(
        r"Al: $\rho=8.1$ g cm$^{-3}$, $T_e=T_i=15$ eV",
        y=0.985,
    )
    fig_structure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_figure(
        fig_structure,
        FIGURE_DIR / "al_is_sc_ionic_structure",
        close=False,
    )


# %%
# KS-DFT level shifts
# -------------------
# The deep 1s level and valence levels use separate linear axes so that the
# SC-induced changes remain visible.  Fermi--Dirac occupations are printed in
# the terminal table above.  TF has no discrete KS eigenvalue spectrum.

is_level_map = {row["level"]: row for row in level_rows(state, "is")}
sc_level_map = {row["level"]: row for row in level_rows(state, "sc")}
all_level_names = list(dict.fromkeys([*is_level_map, *sc_level_map]))
deep_levels = [name for name in all_level_names if name == "1s"]
valence_levels = [name for name in all_level_names if name != "1s"]


def _draw_level_paths(
    axis: Any,
    labels: list[str],
    *,
    title: str,
) -> None:
    for color, label in zip(PALETTES["bing"][1:], labels, strict=False):
        is_row = is_level_map.get(label)
        sc_row = sc_level_map.get(label)
        values = [
            np.nan if is_row is None else float(is_row["energy_ev"]),
            np.nan if sc_row is None else float(sc_row["energy_ev"]),
        ]
        axis.plot(
            (0.0, 1.0),
            values,
            color=color,
            marker="o",
            ms=6.0,
            label=label,
        )
    axis.set(
        xticks=(0.0, 1.0),
        xticklabels=("IS", "SC"),
        ylabel="KS energy [eV]",
        title=title,
    )
    axis.set_xlim(-0.25, 1.25)
    axis.legend()


with style_context("thesis", palette="bing"):
    fig_levels, level_axes = plt.subplots(
        1,
        2,
        figsize=grid_figsize(1, 2),
    )
    _draw_level_paths(
        level_axes[0],
        deep_levels,
        title="Deep level",
    )
    _draw_level_paths(
        level_axes[1],
        valence_levels,
        title="Valence levels",
    )
    fig_levels.suptitle(
        "KS-DFT bound levels under IS and experimental SC feedback",
        y=0.985,
    )
    fig_levels.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_figure(
        fig_levels,
        FIGURE_DIR / "al_is_sc_bound_levels",
        close=False,
    )


# %%
# Runtime and SC convergence
# --------------------------
# ``SC extension`` excludes the initial IS calculation.  Timings are machine
# dependent and document only these stored runs.  A valid SC result must meet
# both the :math:`g_{ii}` and correlation-potential change tolerances.

is_time = np.asarray(state["is_elapsed_s"], dtype=float)
sc_extension_time = np.asarray(state["sc_extension_elapsed_s"], dtype=float)
x = np.arange(len(model_names), dtype=float)
width = 0.34

with style_context("thesis", palette="bing"):
    fig_convergence, (ax_time, ax_history) = plt.subplots(
        1,
        2,
        figsize=grid_figsize(1, 2),
    )
    for offset, values, label, color in (
        (-0.5 * width, is_time, "IS", PALETTES["bing"][1]),
        (
            0.5 * width,
            sc_extension_time,
            "SC",
            PALETTES["bing"][2],
        ),
    ):
        bars = ax_time.bar(
            x + offset,
            values,
            width,
            label=label,
            color=color,
        )
        ax_time.bar_label(bars, fmt="%.1f s", padding=2, fontsize=9)
    ax_time.set(
        xticks=x,
        xticklabels=model_names,
        ylabel="wall time [s]",
        title="Recorded calculation time",
    )
    ax_time.set_ylim(
        0.0,
        1.18 * float(max(np.max(is_time), np.max(sc_extension_time))),
    )
    ax_time.legend()

    for model, model_name, color in zip(
        MODELS,
        model_names,
        PALETTES["bing"][1:3],
        strict=True,
    ):
        iteration = np.asarray(
            state[f"{model}_sc_history_iteration"], dtype=int
        )
        dg = np.asarray(
            state[f"{model}_sc_history_max_g_change"], dtype=float
        )
        dv = np.asarray(
            state[f"{model}_sc_history_max_v_corr_change_ha"],
            dtype=float,
        )
        ax_history.semilogy(
            iteration,
            dg,
            color=color,
            marker="o",
            ms=4.5,
            label=model_name + r" $\max|\Delta g|$",
        )
        ax_history.semilogy(
            iteration,
            dv,
            color=color,
            ls="--",
            marker="s",
            ms=4.0,
            label=model_name + r" $\max|\Delta V^C|$ [Ha]",
        )
    ax_history.axhline(
        SC_CONTROLS.g_tol,
        color="0.45",
        ls=":",
        lw=1.0,
        label="tolerance",
    )
    ax_history.set(
        xlabel="SC outer iteration",
        ylabel="maximum change",
        title="Experimental SC convergence",
    )
    ax_history.legend(fontsize=8.5)
    fig_convergence.suptitle(
        r"Al: $\rho=8.1$ g cm$^{-3}$, $T_e=T_i=15$ eV",
        y=0.985,
    )
    fig_convergence.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    save_figure(
        fig_convergence,
        FIGURE_DIR / "al_is_sc_timing_convergence",
        close=False,
    )

if "agg" not in plt.get_backend().lower():
    plt.show()
