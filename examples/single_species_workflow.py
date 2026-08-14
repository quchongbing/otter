"""Minimal one-component electronic -> ionic structure workflow."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib.pyplot as plt
import numpy as np

from otter import PlasmaWorkflowConfig, solve_plasma_workflow
from otter.plotting import grid_figsize, save_figure, set_style

# ===========================================
#            user input parameters
# -------------------------------------------
ELEMENT = "Al"
RHO_G_CC = 8.1
TE_EV = 15.0
TI_EV = 15.0
# -------------------------------------------

SAVE_NPZ = True
OUTPUT_PATH = ROOT / "outputs" / "al_rho8p1_T15_state.npz"
SAVE_FIGURES = True
FIGURE_STEM = ROOT / "outputs" / "single_species_aa_qoz_hnc"
# ===========================================


DENSITY_LABELS = {
    "n_full": r"$n_{\mathrm{full}}$",
    "n_bound": r"$n_{\mathrm{bound}}$",
    "n_cont": r"$n_{\mathrm{cont}}$",
    "n_ion": r"$n_{\mathrm{ion}}$",
    "n_ext": r"$n_{\mathrm{ext}}$",
    "n_pa": r"$n_{\mathrm{pa}}$",
    "n_scr": r"$n_{\mathrm{scr}}$",
}


def _weighted_density(r: np.ndarray, values: np.ndarray) -> np.ndarray:
    r_arr = np.asarray(r, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    return 4.0 * np.pi * r_arr**2 * values_arr


def _r_ws_from_result(electronic: dict) -> float | None:
    if "r_ws" in electronic:
        return float(electronic["r_ws"])
    meta = electronic.get("meta", {})
    if isinstance(meta, dict) and "r_ws_bohr" in meta:
        return float(meta["r_ws_bohr"])
    return None


def _plot_profiles(
    ax,
    r: np.ndarray,
    payload: dict,
    names: tuple[str, ...],
) -> None:
    for name in names:
        if name not in payload:
            continue
        values = np.asarray(payload[name], dtype=float)
        if values.shape != np.asarray(r).shape:
            continue
        ax.plot(r, _weighted_density(r, values), label=DENSITY_LABELS.get(name, name))


def main() -> None:
    cfg = PlasmaWorkflowConfig(
        elements=[ELEMENT],
        temperature_ev=TE_EV,
        rho_g_cc=RHO_G_CC,
        ion_temperature_ev=TI_EV,
        save_state_npz=SAVE_NPZ,
        save_state_path=OUTPUT_PATH,
        #electronic_model='tf', # run semiclassical Thomas-Fermi model for the electronic structure; default is ks-dft (QM)
    )
    result = solve_plasma_workflow(cfg)
    ion = result["ion"]
    electronic = result["electronic"]["result"]

    set_style("docs", palette="deep_science")
    fig, axes = plt.subplots(
        2,
        2,
        figsize=grid_figsize(2, 2),
        constrained_layout=True,
    )
    ax_density, ax_potential, ax_gii, ax_sii = axes.ravel()

    r_e = np.asarray(electronic["r"], dtype=float)
    density_names = ("n_full", "n_bound", "n_cont", "n_ion", "n_ext", "n_pa", "n_scr")
    _plot_profiles(ax_density, r_e, electronic, density_names)
    if "n0" in electronic:
        n0_profile = np.full_like(r_e, float(electronic["n0"]))
        ax_density.plot(
            r_e,
            _weighted_density(r_e, n0_profile),
            color="black",
            linestyle="--",
            linewidth=1.3,
            label=r"$n_0$",
        )
    r_ws = _r_ws_from_result(electronic)
    if r_ws is not None:
        ax_density.axvline(
            r_ws,
            color="black",
            linestyle=":",
            linewidth=1.1,
            label=r"$R_{\mathrm{ws}}$",
        )
    ax_density.set_xlabel(r"$r\,[a_0]$")
    ax_density.set_ylabel(r"$4\pi r^2 n(r)\,[a_0^{-1}]$")
    ax_density.legend()
    ax_density.set_ylim(-0.5, 15)
    ax_density.set_xlim(-0.5, 8)

    if "v_full" in electronic:
        ax_potential.plot(
            r_e,
            np.asarray(electronic["v_full"], dtype=float),
            label=r"$V_{\mathrm{eff}}^{\mathrm{full}}$",
        )
    if "v_ext" in electronic:
        ax_potential.plot(
            r_e,
            np.asarray(electronic["v_ext"], dtype=float),
            label=r"$V_{\mathrm{eff}}^{\mathrm{ext}}$",
        )
    ax_potential.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    if r_ws is not None:
        ax_potential.axvline(
            r_ws,
            color="black",
            linestyle=":",
            linewidth=1.1,
            label=r"$R_{\mathrm{ws}}$",
        )
    ax_potential.set_xlabel(r"$r\,[a_0]$")
    ax_potential.set_ylabel(r"$V_{\mathrm{eff}}(r)\,[\mathrm{Ha}]$")
    ax_potential.set_xlim(-0.5, 8.0)
    ax_potential.set_ylim(-1.0, 1.0)
    ax_potential.legend()

    r_ion = np.asarray(ion["r"], dtype=float)
    ax_gii.plot(r_ion, ion["gii_r"], label=r"$g_{ii}(r)$")
    ax_gii.set_xlabel(r"$r\,[a_0]$")
    ax_gii.set_ylabel(r"$g_{ii}(r)$")
    ax_gii.set_xlim(-0.5, 20.0)
    ax_gii.legend()

    k_ion = np.asarray(ion["k"], dtype=float)
    ax_sii.plot(k_ion, ion["sii_k"], label=r"$S_{ii}(k)$")
    ax_sii.set_xlabel(r"$k\,[a_0^{-1}]$")
    ax_sii.set_ylabel(r"$S_{ii}(k)$")
    ax_sii.set_xlim(0.0, 20.0)
    ax_sii.legend()
    if SAVE_FIGURES:
        paths = save_figure(fig, FIGURE_STEM)
        print(
            "saved figures "
            + ", ".join(str(path) for path in paths.values())
        )
    plt.show()
    if SAVE_NPZ:
        print(f"saved {result['saved_paths']['state_npz']}")


if __name__ == "__main__":
    main()
