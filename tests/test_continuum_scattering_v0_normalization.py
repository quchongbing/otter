"""
tests/test_continuum_scattering_v0_normalization.py

Purpose
-------
Regression for scattering continuum normalization in the V_eff=0 limit.

Methods
-------
- Use a sqrt grid and V_eff = 0.
- Compute n_c(r) from Numerov scattering.
- Compare tail mean density to ideal n0.
- Visualize n_c(r).

Equations
---------
Continuum density (QM form):
  n_c(r) = integral dE g_k sum_l [2(2l+1)/(4*pi)] |y_{k,l}(r)/r|^2

References
----------
- C. E. Starrett & D. Saumon (2014), Eq. (A3).
"""
import numpy as np
import matplotlib.pyplot as plt

from otter.numerics.grids import create_sqrt_grid
from otter.electronic.continuum.scattering import QuantumContinuumScattering
from otter.electronic.continuum.ideal import ideal_unbound_density
from otter.plotting import save_figure, set_style


def test_continuum_scattering_v0_normalization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_style("docs", palette="deep_science")
    print("\n=== Continuum Scattering V=0 Normalization ===")

    grid = create_sqrt_grid(rmax=10.0, N=2**10)
    v_eff = np.zeros_like(grid.r)

    mu = 0.2
    temperature = 1

    e_min = 1e-4
    e_max = 10.0
    k_max = np.sqrt(2.0 * e_max)
    l_max = int(np.ceil(k_max * grid.rmax)) + 2

    params = {
        "v_eff": v_eff,
        "grid_kind": "sqrt",
        "grid_dxi": grid.dxi,
        "l_max": l_max,
        "l_pad": 2,
        "e_min": e_min,
        "e_max": e_max,
        "n_e": 160,
        "match_fraction": 0.2,
        "match_kr_min": 4.0,
        "match_v_tol": 1e-6,
        "match_min_points": 16,
        "match_asymptotic": "auto",
    }

    model = QuantumContinuumScattering()
    n_r = model.density(grid.r, mu, temperature, params=params)

    n0 = ideal_unbound_density(mu, temperature)
    tail_mask = grid.r >= 2.0
    n_tail = float(np.mean(n_r[tail_mask]))
    rel_err = abs(n_tail - n0) / max(n0, 1e-12)

    print(f"Grid: N={grid.N}, rmax={grid.rmax:.1f}")
    print(f"Using l_max={l_max}, e_max={e_max:.2f}")
    print(f"Ideal n0 = {n0:.6e}")
    print(f"Tail mean n = {n_tail:.6e}, rel_err={rel_err:.3e}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(grid.r, n_r, label="scattering n_c(r)")
    ax.axhline(n0, color="k", linestyle="--", label="ideal n0")
    ax.set_xlabel("r (Bohr)")
    ax.set_ylabel("n_c(r)")
    ax.set_title("Scattering continuum normalization (V=0)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = "quantum_continuum_scattering_v0_normalization.png"
    fig.tight_layout()
    paths = save_figure(fig, out_path, close=True)
    print(f"Saved {paths['png']} and {paths['pdf']}")

    assert rel_err < 0.2


if __name__ == "__main__":
    test_continuum_scattering_v0_normalization()
