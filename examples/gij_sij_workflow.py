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
OUTPUT_PATH = ROOT / "outputs" / "gij_sij_demo.npz"


def _pair_label(species: list[str], i: int, j: int) -> str:
    return f"{species[i]}-{species[j]}"


def _plot_pair_matrices(ion: dict) -> None:
    species = [str(item) for item in ion["species"]]
    r = np.asarray(ion["r"], dtype=float)
    k = np.asarray(ion["k"], dtype=float)
    gij = np.asarray(ion["gij_r"], dtype=float)
    sij = np.asarray(ion["sij_k"], dtype=float)
    n_species = len(species)

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
        qoz_linear_n_points=2**12,
        qoz_pad_factor=2.0,
        qoz_renormalize_nscr_to_zbar=True,
        qoz_high_k_taper_start_frac=0.9,
        hnc_mixing_scheme="anderson",
        aa_overrides={
            "cont_n_jobs": 1,
            "cont_rmax_mult": 8.0,
        },
    )
    result = solve_plasma_workflow(cfg)
    ion = result["ion"]

    print(f"species={ion['species']}")
    print(f"zbar={np.asarray(ion['zbar'])}")
    print(f"HNC iterations={ion['hnc_iters']}  qoz={ion['qoz_build_s']:.3f}s hnc={ion['hnc_solve_s']:.3f}s")
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
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUTPUT_PATH,
            r=ion["r"],
            k=ion["k"],
            species=np.asarray(ion["species"], dtype="<U8"),
            zbar=np.asarray(ion["zbar"], dtype=float),
            n_i=np.asarray(ion["n_i"], dtype=float),
            gij_r=np.asarray(ion["gij_r"], dtype=float),
            sij_k=np.asarray(ion["sij_k"], dtype=float),
            vij_r=np.asarray(ion["vij_r"], dtype=float),
            vij_k=np.asarray(ion["vij_k"], dtype=float),
        )
        print(f"saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
