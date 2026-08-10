"""
tests/test_continuum_tail_b3_recovery.py

Purpose
-------
Validate B3 tail matching directly on an A3 continuum profile produced from a
Yukawa effective potential.

What this test checks
---------------------
1) Build `n_cont(r)` from A3 using `QuantumContinuumScattering`.
2) Apply B3 at `r_cut` with `apply_tail_match` (manual splice).
3) Apply B3 again through `QuantumContinuumScattering` (`tail_match=True`).
4) Compare A3 vs B3 in the tail and verify the two B3 implementations agree.

Notes
-----
- This test is for diagnostics/validation, not strict CI-level regression.
- Keep `tail_r_cut < match_r_cut` so B3 replacement starts inside the region
  where A3 is still used for scattering normalization.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from otter.numerics.grids import create_sqrt_grid
from otter.electronic.continuum import scattering as qmod
from otter.electronic.continuum.scattering import QuantumContinuumScattering
from otter.electronic.continuum.ideal import ideal_unbound_density
from otter.electronic.continuum.hybrid import ideal_density_range
from otter.electronic.continuum.tail import apply_tail_match
from otter.plotting import save_figure, set_style


# -------- User-configurable diagnostics --------
VERBOSE = 1
PLOT = 1

GRID_RMAX = 35.0
GRID_N = 2**10

MU = 0.1
TEMPERATURE = 0.5
E_MIN = 1e-4
E_MAX = 10.0

# Use the same A3 settings as test_continuum_scattering_diagnostics
# so the A3 baseline is directly comparable.
ENERGY_MODE = "adaptive"          # "adaptive" or "linear"
N_E_LINEAR = 1000                 # used only when ENERGY_MODE="linear"
ADAPTIVE_MODE = "simpson"         # "simpson" or "bisection"
N_E_BASE = 20                     # used only when ENERGY_MODE="adaptive"
E_TOL = 0.1                       # used only when ENERGY_MODE="adaptive"

YUKAWA_Z = 1.0
YUKAWA_KAPPA = 0.5

# Asymptotic matching window (A3 normalization support window).
MATCH_R_CUT = 15.0
MATCH_WIDTH_FRAC = 0.15
# B3 replacement starts before the A3 matching window.
TAIL_R_CUT = 12.0

L_CAP_STRATEGY = "match"
FIT_POINTS = 40
BLEND_POINTS = 30


def _yukawa(r: np.ndarray, z: float, kappa: float, r_soft: float) -> np.ndarray:
    return -z * np.exp(-kappa * r) / np.sqrt(r * r + r_soft * r_soft)


def _build_a3_params(r: np.ndarray, v_eff: np.ndarray, l_max: int, grid_dxi: float) -> dict:
    match_r_cut = float(MATCH_R_CUT)
    match_width = float(MATCH_WIDTH_FRAC * r[-1])
    tail_r_cut = float(TAIL_R_CUT)
    if tail_r_cut >= match_r_cut:
        raise ValueError(
            f"Require tail_r_cut < match_r_cut, got {tail_r_cut:.4f} >= {match_r_cut:.4f}."
        )
    if ENERGY_MODE not in ("linear", "adaptive"):
        raise ValueError("ENERGY_MODE must be 'linear' or 'adaptive'.")
    params = {
        "v_eff": v_eff,
        "grid_kind": "sqrt",
        "grid_dxi": float(grid_dxi),
        "l_max": int(l_max),
        "l_pad": 2,
        "e_min": float(E_MIN),
        "e_max": float(E_MAX),
        "n_e": int(N_E_LINEAR),
        "n_jobs": 1,
        "energy_mode": str(ENERGY_MODE),
        "l_cap_strategy": L_CAP_STRATEGY,
        "match_fraction": 0.3,
        "match_fraction_mode": "r",
        "match_r_cut": match_r_cut,
        "match_width": match_width,
        "match_kr_min": 3.0,
        "match_v_tol": 1e-4,
        "match_min_points": 16,
        "match_asymptotic": "auto",
        "tail_match": False,
        "tail_match_target": "cont",
        "tail_r_cut": tail_r_cut,
        "tail_fit_points": FIT_POINTS,
        "tail_blend_points": BLEND_POINTS,
    }
    if ENERGY_MODE == "adaptive":
        params["adaptive_mode"] = ADAPTIVE_MODE
        params["n_e_base"] = int(N_E_BASE)
        params["e_tol"] = float(E_TOL)
    return params


def test_continuum_tail_b3_recovery(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_style("docs", palette="deep_science")
    print("\n=== Continuum Tail B3 Recovery (Yukawa A3 -> B3) ===")

    # (1) Build A3 continuum profile on a Yukawa potential.
    grid = create_sqrt_grid(rmax=GRID_RMAX, N=GRID_N)
    r = grid.r
    v_eff = _yukawa(r, z=YUKAWA_Z, kappa=YUKAWA_KAPPA, r_soft=r[0])
    k_max = np.sqrt(2.0 * E_MAX)
    l_max = int(np.ceil(k_max * r[-1])) + 2
    model = QuantumContinuumScattering()

    params_a3 = _build_a3_params(r, v_eff, l_max=l_max, grid_dxi=grid.dxi)
    n_a3 = model.density(r, mu=MU, temperature=TEMPERATURE, params=params_a3)

    # B3 reference state.
    n0 = float(ideal_unbound_density(MU, TEMPERATURE))
    n0_range = float(ideal_density_range(MU, TEMPERATURE, E_MIN, E_MAX, n_samples=2000))
    tail_r_cut = float(params_a3["tail_r_cut"])
    match_r_cut = float(params_a3["match_r_cut"])
    idx_cut = int(np.searchsorted(r, tail_r_cut))

    # (2) Manual B3 splice on top of A3.
    n_b3_manual, meta_manual = apply_tail_match(
        r,
        n_a3,
        n0,
        MU,
        TEMPERATURE,
        idx_cut,
        fit_points=FIT_POINTS,
        blend_points=BLEND_POINTS,
    )

    # (3) Built-in B3 splice through the continuum model.
    params_b3 = dict(params_a3)
    params_b3["tail_match"] = True
    params_b3["tail_n0"] = n0
    params_b3["tail_mu_id"] = MU
    n_b3_model = model.density(r, mu=MU, temperature=TEMPERATURE, params=params_b3)

    # (4) Compare A3 vs B3 and model-vs-manual B3 consistency.
    near_mask = (r >= tail_r_cut) & (r <= min(match_r_cut, tail_r_cut + 0.2 * r[-1]))
    tail_mask = (r >= 0.80 * r[-1]) & (r <= 0.95 * r[-1])

    near_rel = float(np.mean(np.abs(n_b3_manual[near_mask] - n_a3[near_mask])) / max(n0, 1e-12))
    tail_rel = float(np.mean(np.abs(n_b3_manual[tail_mask] - n_a3[tail_mask])) / max(n0, 1e-12))
    model_manual_max = float(np.max(np.abs(n_b3_model - n_b3_manual)))
    continuity_rel = float(abs(n_b3_manual[idx_cut] - n_a3[idx_cut]) / max(n0, 1e-12))
    mid_mask = (r >= 0.30 * r[-1]) & (r <= 0.40 * r[-1])
    mid_mean = float(np.mean(n_a3[mid_mask]))

    if VERBOSE:
        print(f"Grid: N={grid.N}, rmax={grid.rmax:.2f}")
        print(f"Potential: Yukawa(z={YUKAWA_Z}, kappa={YUKAWA_KAPPA})")
        print(
            f"l_max={l_max}, e in [{E_MIN:.1e}, {E_MAX:.1f}], "
            f"energy_mode={ENERGY_MODE}"
        )
        if ENERGY_MODE == "adaptive":
            print(f"adaptive: mode={ADAPTIVE_MODE}, n_e_base={N_E_BASE}, e_tol={E_TOL}")
        else:
            print(f"linear: n_e={N_E_LINEAR}")
        print(
            f"match window: [{match_r_cut:.3f}, {min(r[-1], match_r_cut + params_a3['match_width']):.3f}], "
            f"tail_r_cut={tail_r_cut:.3f}"
        )
        print(f"Ideal n0 (full) = {n0:.6e}")
        print(f"Ideal n0 (range) = {n0_range:.6e}")
        print(f"A3 mid mean n(r in [0.30,0.40]Rmax) = {mid_mean:.6e}")
        print(
            f"tail endpoint: A3={n_a3[-1]:.6e}, B3(manual)={n_b3_manual[-1]:.6e}, "
            f"B3(model)={n_b3_model[-1]:.6e}"
        )
        print(
            f"B3 fit params: A={meta_manual['A']:.3e}, "
            f"B={meta_manual['B']:.3e}, delta={meta_manual['delta']:.3e}"
        )
        print(f"A3 vs B3 near-cut mean rel = {near_rel:.3e}")
        print(f"A3 vs B3 far-tail mean rel = {tail_rel:.3e}")
        print(f"B3(model) vs B3(manual) max abs = {model_manual_max:.3e}")
        print(f"splice continuity at r_cut rel = {continuity_rel:.3e}")

    if PLOT:
        fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True)

        ax0 = axes[0]
        ax0.plot(r, n_a3, lw=1.8, label="A3 n_cont")
        ax0.plot(r, n_b3_manual, lw=1.8, ls="--", label="A3 + B3 (manual)")
        ax0.plot(r, n_b3_model, lw=1.2, ls=":", label="A3 + B3 (model)")
        ax0.axhline(n0, color="k", lw=1.0, ls=":", label="ideal n0")
        ax0.axvline(tail_r_cut, color="tab:red", ls="--", lw=1.2, label="tail r_cut")
        ax0.axvline(match_r_cut, color="tab:purple", ls="--", lw=1.2, label="match r_cut")
        ax0.set_ylabel("n_cont(r)")
        ax0.set_title("Yukawa continuum: A3 vs B3 tail replacement")
        ax0.grid(True, alpha=0.3)
        ax0.legend()

        ax1 = axes[1]
        ax1.plot(r, n_b3_manual - n_a3, lw=1.8, label="manual B3 - A3")
        ax1.plot(r, n_b3_model - n_b3_manual, lw=1.2, label="model B3 - manual B3")
        ax1.axhline(0.0, color="k", lw=1.0, ls=":")
        ax1.axvline(tail_r_cut, color="tab:red", ls="--", lw=1.2)
        ax1.axvline(match_r_cut, color="tab:purple", ls="--", lw=1.2)
        ax1.set_xlabel("r (Bohr)")
        ax1.set_ylabel("Delta n(r)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        out_path = "quantum_continuum_tail_b3_recovery_yukawa.png"
        fig.tight_layout()
        paths = save_figure(fig, out_path, close=True)
        print(f"Saved {paths['png']} and {paths['pdf']}")

    # Continuity at r_cut should be good because of splice/blend.
    assert np.isfinite(near_rel)
    assert continuity_rel < 5e-2
    # Model path and direct apply_tail_match path should agree closely.
    assert model_manual_max < 1e-5

if __name__ == "__main__":
    test_continuum_tail_b3_recovery()
