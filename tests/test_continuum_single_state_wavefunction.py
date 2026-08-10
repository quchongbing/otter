"""
tests/test_continuum_single_state_wavefunction.py

Purpose
-------
Exercise the continuum Numerov solver at a single (energy, l) and visualize
the normalized wavefunction and its density contribution.

Methods
-------
- Build a sqrt grid and a screened Coulomb V_eff(r).
- Propagate the regular Numerov solution for a chosen (e, l).
- Normalize u(r) by matching to asymptotic free/Coulomb basis.
- Plot u(r), R(r)=u/r, and the single-channel density contribution.

Equations
---------
Single-channel density (energy-normalized):
  n_{e,l}(r) = f_FD(e; mu, T) * 2(2l+1)/(4pi) * |u(r)/r|^2

References
----------
- C. E. Starrett & D. Saumon (2014), Eq. (A3).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from otter.numerics.grids import create_sqrt_grid
from otter.numerics.constants import EV_TO_HA
from otter.electronic.continuum.scattering import (
    _numerov_propagate_sqrt,
    _match_scattering_u,
    fermi_dirac,
)
from otter.plotting import save_figure, set_style


def test_continuum_single_state_wavefunction(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_style("docs", palette="deep_science")
    print("\n=== Continuum Single-State Wavefunction ===")

    grid = create_sqrt_grid(rmax=12.0, N=800)
    r = grid.r

    z_eff = 3.0
    r_screen = 2.5
    v_eff = -z_eff * np.exp(-r / r_screen) / r

    energy = 0.5
    l_val = 1
    mu = 0.2
    temperature = 5.0 * EV_TO_HA

    u_raw = _numerov_propagate_sqrt(r, v_eff, energy, l_val, grid.dxi)
    u_norm, delta, amp_target, meta = _match_scattering_u(
        u_raw,
        r,
        v_eff,
        energy,
        l_val,
        match_fraction=0.25,
        match_r_cut=0.7 * r[-1],
        match_kr_min=None,
        match_v_tol=None,
        match_min_points=16,
        match_asymptotic="auto",
        match_coulomb_tol=0.2,
        match_allow_shift=True,
        match_fallback="free",
    )

    R = u_norm / r
    occ = fermi_dirac(np.array([energy]), mu, temperature)[0]
    factor = (2.0 * (2 * l_val + 1)) / (4.0 * np.pi)
    n_l = occ * factor * (np.abs(R) ** 2)
    n_l_radial = 4.0 * np.pi * r ** 2 * n_l

    print(f"Grid: N={grid.N}, rmax={grid.rmax:.2f}")
    print(f"Energy={energy:.3f} Ha, l={l_val}, mu={mu:.3f}, T={temperature:.3f}")
    print(f"Delta_l={delta:.6f}, amp_target={amp_target:.6e}")
    print(f"Match meta: {meta}")

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True)

    axes[0].plot(r, u_norm, label="u(r) normalized", lw=2.0)
    axes[0].set_ylabel("u(r)")
    axes[0].set_title("Single-state scattering solution")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(r, R, label="R(r)=u/r", lw=2.0)
    axes[1].set_ylabel("R(r)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(r, n_l_radial, label="4pi r^2 n_l(r)", lw=2.0)
    axes[2].set_xlabel("r (Bohr)")
    axes[2].set_ylabel("4pi r^2 n_l")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    out_path = Path(__file__).with_suffix("").name + ".png"
    fig.tight_layout()
    paths = save_figure(fig, out_path, close=True)
    print(f"Saved {paths['png']} and {paths['pdf']}")


if __name__ == "__main__":
    test_continuum_single_state_wavefunction()
