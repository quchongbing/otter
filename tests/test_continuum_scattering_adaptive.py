"""
tests/test_continuum_scattering_adaptive.py

Purpose
-------
Validate adaptive energy integration for scattering continuum and check
whether resonant structure is visible in energy-resolved diagnostics.

Methods
-------
- Use adaptive energy refinement for a screened Coulomb (Yukawa) potential.
- Report energy-evaluation counts and accuracy vs ideal n0.
- Visualize n_c(r) vs r, DOS-like curve vs E, dE vs E, and phase-shift trends.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from otter.numerics.grids import create_sqrt_grid
from otter.electronic.continuum import scattering as qmod
from otter.electronic.continuum.scattering import (
    continuum_density_scattering_adaptive,
    fermi_dirac,
)
from otter.electronic.continuum.ideal import ideal_unbound_density
from otter.plotting import save_figure, set_style


# --- User-configurable settings ---
# Grid and potential (sqrt grid for continuum).
GRID_RMAX = 30.0        # Outer radius Rmax (Bohr). A3/B3 match uses this grid.
GRID_N = 2**10          # Number of radial points (sqrt grid).

# Thermodynamic state (Ha).
MU = 0.5                # Chemical potential (Ha).
TEMPERATURE = 0.5       # Temperature (Ha).

# Energy integration bounds (Ha).
E_MIN = 1e-3            # Lower bound of A3 integral (Ha).
E_MAX = 15           # Upper bound of A3 integral (Ha).

# Yukawa effective potential parameters (deeper/less screened to expose resonances).
YUKAWA_Z = 6.0          # Effective charge (dimensionless).
YUKAWA_KAPPA = 0.1      # Screening strength (Bohr^-1).

# Tail-matching / l-cut strategy.
R_CUT = 20.0            # Radius where A3 is trusted (Bohr); B3 should take over beyond.
MATCH_WIDTH_FRAC = 0.15 # Fraction of Rmax added beyond r_cut for the match window.
L_CAP_STRATEGY = "match"  # "match" uses r_m_end-Δr; "rmax" uses k*Rmax.

ADAPTIVE_PARAMS = {
    "match_fraction": 0.3,            # Fractional tail window (used if match_r_cut is None).
    "match_fraction_mode": "r",       # Interpret match_fraction in physical r (not index).
    "match_width": None,              # Will be set to MATCH_WIDTH_FRAC * Rmax at runtime.
    "match_kr_min": 3.0,              # Require k*r >= max(match_kr_min, l+1) in match window.
    "match_v_tol": None,              # Require |V_eff| <= match_v_tol in match window.
    "match_min_points": 16,           # Minimum number of points in match window.
    "e_tol": 1e-3,                    # Adaptive Simpson relative error target.
    "e_max_depth": 12,               # Max adaptive depth.
    "e_min_width": 5e-5,             # Minimum energy interval width.
    "n_e_base": 30,                  # Base energy nodes for adaptive refinement.
    "resonance_tol": 1e-3,           # Enable curvature-based resonance forcing.
    "resonance_r_fractions": (0.25, 0.5, 0.75),
    "resonance_floor": 1e-8,
    "adaptive_mode": "bisection",   # Use phase-shift windows + local Simpson refinement.
    "delta_mode": "max",            # Phase-shift jump metric for bisection.
    "delta_tol": np.pi / 2.0,        # Phase-shift jump threshold (radians).
    "bisection_max_depth": 12,       # Max bisection depth for resonance windows.
    "resonance_window_factor": 6.0,  # Window width scaling for phase-shift peaks.
}

BACKEND_CASES = [
    {"name": "numba_single", "n_jobs": 1},
]

def _yukawa(r: np.ndarray, z: float, kappa: float, r_soft: float) -> np.ndarray:
    return -z * np.exp(-kappa * r) / np.sqrt(r * r + r_soft * r_soft)


def test_continuum_scattering_adaptive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_style("docs", palette="deep_science")
    print("\n=== Continuum Scattering Adaptive Test (Yukawa) ===")

    grid = create_sqrt_grid(rmax=GRID_RMAX, N=GRID_N)
    v_eff = _yukawa(grid.r, z=YUKAWA_Z, kappa=YUKAWA_KAPPA, r_soft=grid.r[0])

    mu = float(MU)
    temperature = float(TEMPERATURE)

    e_min = float(E_MIN)
    e_max = float(E_MAX)
    k_max = np.sqrt(2.0 * e_max)
    l_max = int(np.ceil(k_max * grid.rmax)) + 2

    # Build a finite match window [r_cut, r_cut + match_width].
    match_width = MATCH_WIDTH_FRAC * grid.rmax
    match_r_cut = float(R_CUT)
    ADAPTIVE_PARAMS["match_width"] = match_width

    print(f"Grid: N={grid.N}, rmax={grid.rmax:.1f}")
    print(f"Using l_max={l_max}, e_min={e_min:.1e}, e_max={e_max:.1f}")
    print(f"Match window: r_m_end = {match_r_cut + match_width:.2f} (r_cut={match_r_cut:.2f}, width={match_width:.2f})")

    for case in BACKEND_CASES:
        label = case["name"]
        n_jobs = int(case["n_jobs"])
        print(f"\n[case] {label}: numerov_mode=numba, n_jobs={n_jobs}")

        t0 = time.perf_counter()
        energy_cache = {}
        n_adapt, meta = continuum_density_scattering_adaptive(
            v_eff,
            grid.r,
            mu,
            temperature,
            e_min,
            e_max,
            l_max,
            "sqrt",
            grid.dxi,
            l_pad=2,
            match_r_cut=match_r_cut,
            n_jobs=n_jobs,
            l_cap_strategy=L_CAP_STRATEGY,
            match_fraction=ADAPTIVE_PARAMS["match_fraction"],
            match_fraction_mode=ADAPTIVE_PARAMS["match_fraction_mode"],
            match_width=ADAPTIVE_PARAMS["match_width"],
            match_kr_min=ADAPTIVE_PARAMS["match_kr_min"],
            match_v_tol=ADAPTIVE_PARAMS["match_v_tol"],
            match_min_points=ADAPTIVE_PARAMS["match_min_points"],
            e_tol=ADAPTIVE_PARAMS["e_tol"],
            e_max_depth=ADAPTIVE_PARAMS["e_max_depth"],
            e_min_width=ADAPTIVE_PARAMS["e_min_width"],
            n_e_base=ADAPTIVE_PARAMS["n_e_base"],
            resonance_tol=ADAPTIVE_PARAMS["resonance_tol"],
            resonance_r_fractions=ADAPTIVE_PARAMS["resonance_r_fractions"],
            resonance_floor=ADAPTIVE_PARAMS["resonance_floor"],
            adaptive_mode=ADAPTIVE_PARAMS["adaptive_mode"],
            delta_mode=ADAPTIVE_PARAMS["delta_mode"],
            delta_tol=ADAPTIVE_PARAMS["delta_tol"],
            bisection_max_depth=ADAPTIVE_PARAMS["bisection_max_depth"],
            resonance_window_factor=ADAPTIVE_PARAMS["resonance_window_factor"],
            energy_cache=energy_cache,
        )
        t1 = time.perf_counter()

        n_ideal = ideal_unbound_density(mu, temperature)

        print(f"Adaptive time: {t1 - t0:.3f}s")
        print(
            f"Adaptive evals: {meta['n_eval']} (max_depth={meta['max_depth']}, "
            f"resonance_hits={meta['resonance_hits']}, delta_hits={meta['delta_hits']})"
        )
        print(f"Ideal n0 (full) = {n_ideal:.6e}")
        # Diagnostics for the effective l cutoff at the test energy.
        k_probe = np.sqrt(2.0 * max(e_min, 1e-12))
        i0 = int(np.searchsorted(grid.r, match_r_cut))
        i1 = int(np.searchsorted(grid.r, match_r_cut + match_width))
        i0 = max(1, min(i0, grid.r.size - 2))
        i1 = max(i0 + 1, min(i1, grid.r.size))
        r_m_end = float(grid.r[i1 - 1])
        dr_local = float(np.median(np.diff(grid.r[i0:i1]))) if (i1 - i0) >= 2 else float(np.median(np.diff(grid.r)))
        delta_r = max(int(ADAPTIVE_PARAMS["match_min_points"]), 2) * dr_local
        l_cap_match = int(np.floor(k_probe * (r_m_end - delta_r) - 1.0))
        print(f"r_m_end = {r_m_end:.4f}, Δr ≈ {delta_r:.4f}, l_cap_match(E_min) = {l_cap_match}")

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(grid.r, n_adapt, label="n_cont (adaptive)")
        ax.axhline(n_ideal, color="k", linestyle=":", label="ideal n0")
        ax.set_xlabel("r (Bohr)")
        ax.set_ylabel("n_c(r)")
        ax.set_title("Scattering continuum (Yukawa): n_cont vs r")
        ax.grid(True, alpha=0.3)
        ax.legend()

        out_path = f"quantum_continuum_scattering_adaptive_{label}.png"
        fig.tight_layout()
        paths = save_figure(fig, out_path, close=True)
        print(f"Saved {paths['png']} and {paths['pdf']}")

    # Energy-resolved DOS-like quantity and adaptive dE spacing.
    if energy_cache:
        e_samples = np.array(sorted(energy_cache.keys()), dtype=float)
        x_e = np.zeros_like(e_samples)
        for i, e in enumerate(e_samples):
            n_e, _ = energy_cache[e]
            integrand = 4.0 * np.pi * (grid.r ** 2) * n_e
            x_e[i] = float(np.trapezoid(integrand, grid.r))


        de = np.diff(e_samples)
        e_mid = 0.5 * (e_samples[:-1] + e_samples[1:])
        f_occ = fermi_dirac(e_samples, mu, temperature)
        dos_like = np.full_like(x_e, np.nan)
        occ_mask = f_occ > 1e-4
        dos_like[occ_mask] = x_e[occ_mask] / f_occ[occ_mask]

        fig2, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)
        dos_mask = occ_mask & np.isfinite(dos_like) & (dos_like > 0.0)
        x_mask = np.isfinite(x_e) & (x_e > 0.0)
        axes[0].plot(e_samples[dos_mask], dos_like[dos_mask], lw=2.0, label="DOS-like (X/f)")
        axes[0].plot(e_samples[x_mask], x_e[x_mask], lw=1.2, alpha=0.6, label="FD-weighted X(E)")
        axes[0].set_xlabel("Energy (Ha)")
        axes[0].set_ylabel("X(E)")
        axes[0].set_title("Adaptive continuum DOS-like curve")
        #axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(e_mid, de, lw=1.5)
        axes[1].set_xlabel("Energy (Ha)")
        axes[1].set_ylabel("dE")
        axes[1].set_title("Adaptive energy spacing (dE vs E)")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale("log")
        out_path2 = f"quantum_continuum_scattering_adaptive_energy_{label}.png"
        fig2.tight_layout()
        paths2 = save_figure(fig2, out_path2, close=True)
        print(f"Saved {paths2['png']} and {paths2['pdf']}")

        # Phase-shift diagnostics (resonance indicators).
        delta_all = np.vstack([energy_cache[e][1] for e in e_samples])
        delta_all_unwrap = np.unwrap(delta_all, axis=0, discont=np.pi)
        weights = 2.0 * np.arange(delta_all.shape[1]) + 1.0
        delta_sum = np.sum(delta_all_unwrap, axis=1)
        delta_weighted = np.sum(delta_all_unwrap * weights[None, :], axis=1)
        delta_max = np.max(np.abs(delta_all_unwrap), axis=1)
        d_delta_sum = np.gradient(delta_sum, e_samples)
        d_delta_weighted = np.gradient(delta_weighted, e_samples)
        d_delta_max = np.gradient(delta_max, e_samples)

        fig3, axes3 = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axes3[0].plot(e_samples, delta_sum, lw=2.0, label="sum δ_l (unwrapped)")
        axes3[0].plot(e_samples, delta_weighted, lw=1.7, alpha=0.9, label="sum (2l+1)δ_l")
        axes3[0].plot(e_samples, delta_max, lw=1.2, alpha=0.7, label="max |δ_l|")
        axes3[0].set_ylabel("Phase shift (rad)")
        axes3[0].set_title("Phase-shift trends vs energy")
        axes3[0].grid(True, alpha=0.3)
        axes3[0].legend()

        axes3[1].plot(e_samples, np.abs(d_delta_sum), lw=2.0, label="|d/dE sum δ|")
        axes3[1].plot(e_samples, np.abs(d_delta_weighted), lw=1.7, alpha=0.9, label="|d/dE sum (2l+1)δ|")
        axes3[1].plot(e_samples, np.abs(d_delta_max), lw=1.2, alpha=0.7, label="|d/dE max δ|")
        axes3[1].set_xlabel("Energy (Ha)")
        axes3[1].set_ylabel("|dδ/dE|")
        axes3[1].set_yscale("log")
        axes3[1].grid(True, alpha=0.3)
        axes3[1].legend()

        out_path3 = f"quantum_continuum_scattering_adaptive_phase_{label}.png"
        fig3.tight_layout()
        paths3 = save_figure(fig3, out_path3, close=True)
        print(f"Saved {paths3['png']} and {paths3['pdf']}")

        # Selected delta_l(E) channels to identify resonance structure directly.
        sample_delta = energy_cache[e_samples[0]][1]
        n_l = int(sample_delta.size)
        l_samples = list(range(min(6, n_l)))
        if l_samples:
            delta_l = np.zeros((e_samples.size, len(l_samples)))
            for i, e in enumerate(e_samples):
                _, dvec = energy_cache[e]
                for j, l_val in enumerate(l_samples):
                    delta_l[i, j] = dvec[l_val]

            # Unwrap phase shifts to remove ±π branch cuts.
            delta_l_unwrap = np.unwrap(delta_l, axis=0, discont=np.pi)

            fig4, ax4 = plt.subplots(1, 1, figsize=(8, 4.5))
            for j, l_val in enumerate(l_samples):
                ax4.plot(e_samples, delta_l_unwrap[:, j], lw=1.6, label=f"l={l_val}")
            ax4.set_xlabel("Energy (Ha)")
            ax4.set_ylabel("δ_l (rad, unwrapped)")
            ax4.set_title("Selected phase-shift channels δ_l(E) (unwrapped)")
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            out_path4 = f"quantum_continuum_scattering_adaptive_delta_l_{label}.png"
            fig4.tight_layout()
            paths4 = save_figure(fig4, out_path4, close=True)
            print(f"Saved {paths4['png']} and {paths4['pdf']}")

            # Optional: highlight rapid phase change (resonance indicator).
            fig5, ax5 = plt.subplots(1, 1, figsize=(8, 4.5))
            for j, l_val in enumerate(l_samples):
                d_delta = np.gradient(delta_l_unwrap[:, j], e_samples)
                ax5.plot(e_samples, np.abs(d_delta), lw=1.4, label=f"l={l_val}")
            ax5.set_xlabel("Energy (Ha)")
            ax5.set_ylabel("|dδ_l/dE|")
            ax5.set_yscale("log")
            ax5.set_title("Phase-shift slopes (resonance indicator)")
            ax5.grid(True, alpha=0.3)
            ax5.legend()
            out_path5 = f"quantum_continuum_scattering_adaptive_delta_slope_{label}.png"
            fig5.tight_layout()
            paths5 = save_figure(fig5, out_path5, close=True)
            print(f"Saved {paths5['png']} and {paths5['pdf']}")

            # Uniform-energy derivative to reduce artifacts from non-uniform energy grids.
            e_uniform = np.linspace(float(e_samples.min()), float(e_samples.max()), 2000)
            fig6, ax6 = plt.subplots(1, 1, figsize=(8, 4.5))
            for j, l_val in enumerate(l_samples):
                delta_interp = np.interp(e_uniform, e_samples, delta_l_unwrap[:, j])
                d_delta_uniform = np.gradient(delta_interp, e_uniform)
                ax6.plot(e_uniform, np.abs(d_delta_uniform), lw=1.4, label=f"l={l_val}")
            ax6.set_xlabel("Energy (Ha)")
            ax6.set_ylabel("|dδ_l/dE| (uniform grid)")
            ax6.set_yscale("log")
            ax6.set_title("Phase-shift slopes on uniform energy grid")
            ax6.grid(True, alpha=0.3)
            ax6.legend()
            out_path6 = f"quantum_continuum_scattering_adaptive_delta_slope_uniform_{label}.png"
            fig6.tight_layout()
            paths6 = save_figure(fig6, out_path6, close=True)
            print(f"Saved {paths6['png']} and {paths6['pdf']}")


if __name__ == "__main__":
    test_continuum_scattering_adaptive()
