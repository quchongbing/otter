"""Minimal one-component electronic -> ionic structure workflow."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt

from otter import PlasmaWorkflowConfig, solve_plasma_workflow

# ===========================================
#            user input parameters
# -------------------------------------------
ELEMENT = "C"
RHO_G_CC = 20
TE_EV = 50.0
TI_EV = 50.0
# ===========================================

def main() -> None:
    cfg = PlasmaWorkflowConfig(
        elements=[ELEMENT],
        temperature_ev=TE_EV,
        rho_g_cc=RHO_G_CC,
        ion_temperature_ev=TI_EV,
        qoz_linear_n_points=2**12,
        qoz_pad_factor=2.0,
        qoz_renormalize_nscr_to_zbar=True,
        qoz_high_k_taper_start_frac=0.9,
        aa_overrides={
            "cont_n_jobs": 1,
            "cont_rmax_mult": 15.0,
        },
    )
    result = solve_plasma_workflow(cfg)
    ion = result["ion"]
    electronic = result["electronic"]["result"]

    print(f"element={ELEMENT} Te={TE_EV:g} eV Ti={TI_EV:g} eV rho={RHO_G_CC:g} g/cc")
    print(f"mu={electronic['mu']:.8f} Ha  zbar={electronic['zbar']:.8f}")
    print(f"HNC iterations={ion['hnc_iters']}  qoz={ion['qoz_build_s']:.3f}s hnc={ion['hnc_solve_s']:.3f}s")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(ion["r"], ion["vii_r"])
    axes[0].set_xlabel("r [Bohr]")
    axes[0].set_ylabel("V_ii(r) [Ha]")

    axes[1].plot(ion["r"], ion["gii_r"])
    axes[1].set_xlabel("r [Bohr]")
    axes[1].set_ylabel("g_ii(r)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
