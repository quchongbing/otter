"""Minimal multicomponent workflow for g_ij(r) and S_ij(k)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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

SAVE_NPZ = True
OUTPUT_PATH = ROOT / "outputs" / "gij_sij_demo.npz"


def main() -> None:
    cfg = PlasmaWorkflowConfig(
        elements=ELEMENTS,
        counts=COUNTS,
        temperature_ev=TE_EV,
        rho_g_cc=RHO_G_CC,
        ion_temperature_ev=TI_EV,
        qoz_linear_n_points=2**12,
        qoz_pad_factor=2.0,
        qoz_renormalize_nscr_to_zbar=True,
        qoz_high_k_taper_start_frac=0.9,
        hnc_mixing_scheme="anderson",
        aa_overrides={
            "cont_n_jobs": 1,
            "cont_rmax_mult": 8.0,
            "b3_r_fit_max_mult": 5.0,
            "b3_r_cut_mult": 4.0,
            "geometry_r_ws_cap_bohr": 3.0,
        },
    )
    result = solve_plasma_workflow(cfg)
    ion = result["ion"]

    print(f"species={ion['species']}")
    print(f"zbar={np.asarray(ion['zbar'])}")
    print(f"HNC iterations={ion['hnc_iters']}  qoz={ion['qoz_build_s']:.3f}s hnc={ion['hnc_solve_s']:.3f}s")

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
