"""
tests/test_continuum_scattering_cache_reuse.py

Purpose
-------
Demonstrate reuse of adaptive energy evaluations via an external cache.

Methods
-------
- Run adaptive continuum twice with the same energy_cache.
- Report cache hits and evaluation counts and compare results.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from otter.numerics.grids import create_sqrt_grid
from otter.electronic.continuum.scattering import continuum_density_scattering_adaptive
from otter.plotting import save_figure, set_style


def test_continuum_scattering_cache_reuse(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_style("docs", palette="deep_science")
    print("\n=== Continuum Scattering Cache Reuse (V=0) ===")

    grid = create_sqrt_grid(rmax=10.0, N=450)
    r = grid.r
    v_eff = np.zeros_like(r)

    mu = 0.2
    temperature = 0.1

    e_min = 1e-4
    e_max = 2.0
    k_max = np.sqrt(2.0 * e_max)
    l_max = int(np.ceil(k_max * grid.rmax))

    energy_cache = {}

    t0 = time.perf_counter()
    n_first, meta_first = continuum_density_scattering_adaptive(
        v_eff,
        r,
        mu,
        temperature,
        e_min,
        e_max,
        l_max,
        "sqrt",
        grid.dxi,
        l_pad=2,
        match_fraction=0.2,
        e_tol=1e-3,
        e_max_depth=8,
        e_min_width=1e-4,
        n_e_base=8,
        energy_cache=energy_cache,
    )
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    n_second, meta_second = continuum_density_scattering_adaptive(
        v_eff,
        r,
        mu,
        temperature,
        e_min,
        e_max,
        l_max,
        "sqrt",
        grid.dxi,
        l_pad=2,
        match_fraction=0.2,
        e_tol=1e-3,
        e_max_depth=8,
        e_min_width=1e-4,
        n_e_base=8,
        energy_cache=energy_cache,
    )
    t3 = time.perf_counter()

    max_diff = float(np.max(np.abs(n_first - n_second)))

    print(f"Grid: N={grid.N}, rmax={grid.rmax:.1f}")
    print(f"First run:  n_eval={meta_first['n_eval']} cache_hits={meta_first['n_cache_hits']} time={t1 - t0:.3f}s")
    print(f"Second run: n_eval={meta_second['n_eval']} cache_hits={meta_second['n_cache_hits']} time={t3 - t2:.3f}s")
    print(f"Cache size: {meta_second['n_cache_total']}")
    print(f"Max |n1 - n2| = {max_diff:.3e}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(r, n_first, label="first run")
    ax.plot(r, n_second, label="second run", linestyle="--")
    ax.set_xlabel("r (Bohr)")
    ax.set_ylabel("n_c(r)")
    ax.set_title("Adaptive cache reuse (V=0)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = "quantum_continuum_scattering_cache_reuse.png"
    fig.tight_layout()
    paths = save_figure(fig, out_path, close=True)
    print(f"Saved {paths['png']} and {paths['pdf']}")


if __name__ == "__main__":
    test_continuum_scattering_cache_reuse()
