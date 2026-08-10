"""Minimal multicomponent workflow for g_ij(r) and S_ij(k)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
from otter import PlasmaWorkflowConfig, solve_plasma_workflow
from otter.plotting import save_figure, set_style

# ===========================================
#            user input parameters
# -------------------------------------------
ELEMENTS = ["C", "H"]
COUNTS = [1.0, 1.36]
RHO_G_CC = 5.0
TE_EV = 10.0
TI_EV = TE_EV

# ===========================================
SHOW_SCF_PROGRESS = False
SHOW_MIXTURE_ROOT_PROGRESS = True
SAVE_NPZ = True
OUTPUT_PATH = ROOT / "outputs" / "ch136_state.npz"
SAVE_FIGURES = True
FIGURE_STEM = ROOT / "outputs" / "ch136_gij_sij"


def _pair_label(species: list[str], i: int, j: int) -> str:
    return f"{species[i]}-{species[j]}"


def _plot_pair_matrices(ion: dict) -> None:
    species = [str(item) for item in ion["species"]]
    r = np.asarray(ion["r"], dtype=float)
    k = np.asarray(ion["k"], dtype=float)
    gij = np.asarray(ion["gij_r"], dtype=float)
    sij = np.asarray(ion["sij_k"], dtype=float)
    n_species = len(species)

    set_style("docs", palette="deep_science")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax_g, ax_s = axes
    for i in range(n_species):
        for j in range(i, n_species):
            label = _pair_label(species, i, j)
            ax_g.plot(r, gij[i, j], label=label)
            ax_s.plot(k, sij[i, j], label=label)

    ax_g.set_xlabel("r [Bohr]")
    ax_g.set_ylabel("g_ij(r)")
    ax_g.set_xlim(0.0, 20.0)
    ax_g.legend(fontsize=8)

    ax_s.set_xlabel("k [1/Bohr]")
    ax_s.set_ylabel("S_ij(k)")
    ax_s.set_xlim(0.0, 20.0)
    ax_s.legend(fontsize=8)
    if SAVE_FIGURES:
        paths = save_figure(fig, FIGURE_STEM)
        print(
            "saved figures "
            + ", ".join(str(path) for path in paths.values())
        )
    plt.show()


def main() -> None:
    cfg = PlasmaWorkflowConfig(
        elements=ELEMENTS,
        counts=COUNTS,
        temperature_ev=TE_EV,
        rho_g_cc=RHO_G_CC,
        ion_temperature_ev=TI_EV,
        show_progress=SHOW_SCF_PROGRESS,
        show_mu_progress=SHOW_MIXTURE_ROOT_PROGRESS,
        # Production defaults use FD bound occupations, the full B3/Friedel
        # tail, Chabrier (1990) LFC, charge-closed QOZ, and a physical
        # multicomponent HNC root.
        save_state_npz=SAVE_NPZ,
        save_state_path=OUTPUT_PATH,
    )
    result = solve_plasma_workflow(cfg)
    ion = result["ion"]

    print(f"species={ion['species']}")
    print(f"zbar={np.asarray(ion['zbar'])}")
    print(
        f"HNC iterations={ion['hnc_iters']}  qoz={ion['qoz_build_s']:.3f}s hnc={ion['hnc_solve_s']:.3f}s"
    )
    mix_meta = result["electronic"]["result"].get("meta", {})
    print(
        "mixture root "
        f"success={mix_meta.get('root_success')} "
        f"nfev={mix_meta.get('root_nfev')} "
        f"max|dmu|={float(mix_meta.get('mu_residual_max_ha', np.nan)):.3e} Ha "
        f"method={mix_meta.get('root_method')}"
    )

    _plot_pair_matrices(ion)

    if SAVE_NPZ:
        print(f"saved {result['saved_paths']['state_npz']}")


if __name__ == "__main__":
    main()
