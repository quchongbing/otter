"""
tests/test_continuum_quantum_free.py

Purpose
-------
Test the quantum continuum free-electron prototype and compare to ideal density.

Methods
-------
- Compute n_c(r) from energy integral + partial waves.
- Compare mean density to ideal-unbound approximation.
- Visualize n_c(r) vs r.
"""
import numpy as np
import matplotlib.pyplot as plt

from otter.electronic.continuum.scattering import QuantumContinuumFree
from otter.electronic.continuum.ideal import ideal_unbound_density
from otter.plotting import save_figure, set_style


def test_quantum_continuum_free(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    set_style("docs", palette="deep_science")
    print("\n=== Quantum Continuum (Free) Test ===")

    r = np.linspace(0.2, 10.0, 200)
    mu = 0.2
    temperature = 0.1

    model = QuantumContinuumFree()
    params = {"l_max": 6, "e_max": 3.0, "n_e": 200}
    n_r = model.density(r, mu, temperature, params=params)

    n_ideal = ideal_unbound_density(mu, temperature)
    n_mean = float(np.mean(n_r))
    n_std = float(np.std(n_r))
    rel_err = abs(n_mean - n_ideal) / max(n_ideal, 1e-12)

    print(f"Ideal n0 = {n_ideal:.6e}")
    print(f"Quantum mean n = {n_mean:.6e}, std = {n_std:.3e}")
    print(f"Relative mean error = {rel_err:.3e}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(r, n_r, label="quantum continuum (free)")
    ax.axhline(n_ideal, color="k", linestyle="--", label="ideal n0")
    ax.set_xlabel("r (Bohr)")
    ax.set_ylabel("n_c(r)")
    ax.set_title("Quantum continuum (free-electron) vs ideal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = "quantum_continuum_free.png"
    fig.tight_layout()
    paths = save_figure(fig, out_path, close=True)
    print(f"Saved {paths['png']} and {paths['pdf']}")


if __name__ == "__main__":
    test_quantum_continuum_free()
