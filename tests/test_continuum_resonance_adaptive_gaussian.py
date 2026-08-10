"""
tests/test_continuum_resonance_adaptive_gaussian.py

Purpose
-------
Compare physical energy sampling strategies around a narrow resonance:
- Bisection on phase-shift changes (delta_l).
- Global adaptive Simpson on a physical proxy (Q_l(E)).

Methods
-------
- Build a Gaussian well + barrier V_eff(r) that produces a shape resonance.
- Use delta_l(E) to identify the resonance and to drive bisection sampling.
- Use adaptive Simpson on Q_l(E) to generate a second sampling pattern.
- Visualize Q_l(E), delta_l(E), and energy samples for both methods.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from otter.numerics.constants import EV_TO_HA
from otter.numerics.grids import create_sqrt_grid
from otter.electronic.continuum.scattering import (
    _numerov_propagate_sqrt,
    _match_scattering_u,
    fermi_dirac,
)
from otter.plotting import save_figure, set_style


def gaussian_well_barrier(r: np.ndarray,
                          depth: float,
                          width: float,
                          barrier: float,
                          barrier_r: float,
                          barrier_w: float) -> np.ndarray:
    """
    Build a short-range Gaussian well plus a Gaussian barrier.
    """
    well = -depth * np.exp(-(r / width) ** 2)
    bump = barrier * np.exp(-((r - barrier_r) / barrier_w) ** 2)
    return well + bump


def _scan_delta_l(r, v_eff, energy_grid, l_val, grid_step) -> np.ndarray:
    """
    Compute delta_l(E) for a fixed l over an energy grid.
    """
    delta_vals = np.zeros_like(energy_grid)
    for i, e in enumerate(energy_grid):
        u_raw = _numerov_propagate_sqrt(r, v_eff, float(e), l_val, grid_step)
        _, delta, _, _ = _match_scattering_u(
            u_raw,
            r,
            v_eff,
            float(e),
            l_val,
            match_fraction=0.2,
            match_slice=None,
            match_r_cut=0.6 * r[-1],
            match_kr_min=2.0,
            match_v_tol=1e-6,
            match_min_points=16,
            match_asymptotic="auto",
            match_coulomb_tol=0.1,
            match_allow_shift=True,
            match_fallback="free",
        )
        delta_vals[i] = delta
    return delta_vals


def _channel_charge_at_energy(r,
                              v_eff,
                              energy,
                              l_val,
                              grid_step,
                              mu,
                              temperature,
                              r_max=None) -> float:
    """
    Compute Q_l(E): the resonant-channel charge inside r_max.
    """
    if r_max is None:
        r_mask = slice(None)
    else:
        r_mask = r <= r_max
    u_raw = _numerov_propagate_sqrt(r, v_eff, float(energy), l_val, grid_step)
    u_norm, _, _, _ = _match_scattering_u(
        u_raw,
        r,
        v_eff,
        float(energy),
        l_val,
        match_fraction=0.2,
        match_slice=None,
        match_r_cut=0.6 * r[-1],
        match_kr_min=2.0,
        match_v_tol=1e-6,
        match_min_points=16,
        match_asymptotic="auto",
        match_coulomb_tol=0.1,
        match_allow_shift=True,
        match_fallback="free",
    )
    R = u_norm / r
    factor = (2.0 * (2 * l_val + 1)) / (4.0 * np.pi)
    f_occ = float(fermi_dirac(np.array([energy]), mu, temperature)[0])
    n_l = f_occ * factor * (np.abs(R) ** 2)
    integrand = 4.0 * np.pi * (r ** 2) * n_l
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(integrand[r_mask], r[r_mask]))
    return float(np.trapz(integrand[r_mask], r[r_mask]))


def _channel_charge_grid(r, v_eff, energy_grid, l_val, grid_step, mu, temperature, r_max=None) -> np.ndarray:
    """
    Evaluate Q_l(E) on a grid of energies.
    """
    return np.array(
        [
            _channel_charge_at_energy(r, v_eff, e, l_val, grid_step, mu, temperature, r_max=r_max)
            for e in energy_grid
        ],
        dtype=float,
    )


def _delta_at_energy(r, v_eff, energy, l_val, grid_step, match_params, cache) -> float:
    """
    Compute delta_l(E) with caching.
    """
    key = float(energy)
    if key in cache:
        return cache[key]
    u_raw = _numerov_propagate_sqrt(r, v_eff, key, l_val, grid_step)
    _, delta, _, _ = _match_scattering_u(
        u_raw,
        r,
        v_eff,
        key,
        l_val,
        match_fraction=match_params["match_fraction"],
        match_slice=None,
        match_r_cut=match_params["match_r_cut"],
        match_kr_min=match_params["match_kr_min"],
        match_v_tol=match_params["match_v_tol"],
        match_min_points=match_params["match_min_points"],
        match_asymptotic=match_params["match_asymptotic"],
        match_coulomb_tol=match_params["match_coulomb_tol"],
        match_allow_shift=match_params["match_allow_shift"],
        match_fallback=match_params["match_fallback"],
    )
    cache[key] = float(delta)
    return float(delta)


def _bisect_delta_interval(a,
                           b,
                           max_depth,
                           min_width,
                           r,
                           v_eff,
                           l_val,
                           grid_step,
                           match_params,
                           cache):
    """
    Strict halving bisection driven by delta_l changes.
    """
    da = _delta_at_energy(r, v_eff, a, l_val, grid_step, match_params, cache)
    db = _delta_at_energy(r, v_eff, b, l_val, grid_step, match_params, cache)
    samples = [float(a), float(b)]
    for _ in range(int(max_depth)):
        if (b - a) <= min_width:
            break
        m = 0.5 * (a + b)
        dm = _delta_at_energy(r, v_eff, m, l_val, grid_step, match_params, cache)
        samples.append(float(m))
        if abs(dm - da) >= abs(db - dm):
            b, db = m, dm
        else:
            a, da = m, dm
    return float(a), float(b), samples


def _build_bisection_samples(e_min,
                             e_max,
                             n_base,
                             delta_tol,
                             max_depth,
                             min_width,
                             r,
                             v_eff,
                             l_val,
                             grid_step,
                             match_params):
    """
    Build energy samples via delta-driven bisection and symmetric mirroring.
    """
    base = np.linspace(float(e_min), float(e_max), int(n_base) + 1)
    cache = {}
    for e in base:
        _delta_at_energy(r, v_eff, e, l_val, grid_step, match_params, cache)

    energies = np.array(sorted(cache.keys()))
    deltas = np.array([cache[e] for e in energies])
    deltas_unwrap = np.unwrap(deltas)
    delta_diff = np.abs(np.diff(deltas_unwrap))
    idx_peak = int(np.argmax(delta_diff))
    a = float(energies[idx_peak])
    b = float(energies[idx_peak + 1])

    if delta_diff[idx_peak] > delta_tol:
        a, b, samples = _bisect_delta_interval(
            a,
            b,
            max_depth,
            min_width,
            r,
            v_eff,
            l_val,
            grid_step,
            match_params,
            cache,
        )
    else:
        samples = [a, b]

    center = 0.5 * (a + b)
    offsets = sorted({abs(float(s) - center) for s in samples})
    sym_set = set()
    for off in offsets:
        for sign in (-1.0, 1.0):
            val = center + sign * off
            if e_min <= val <= e_max:
                sym_set.add(float(val))
    for val in sym_set:
        _delta_at_energy(r, v_eff, val, l_val, grid_step, match_params, cache)

    e_samples = np.array(sorted(cache.keys()))
    meta = {
        "bracket": (float(a), float(b)),
        "center": float(center),
        "n_eval": len(cache),
        "ddelta_peak": float(delta_diff[idx_peak]),
    }
    return e_samples, meta


def _eval_cached(func, x, cache, samples):
    """
    Evaluate func(x) with caching and sample tracking.
    """
    key = float(x)
    if key not in cache:
        cache[key] = float(func(key))
        samples.append(key)
    return cache[key]


def _adaptive_simpson(func,
                      a,
                      b,
                      fa,
                      fb,
                      fm,
                      tol,
                      depth,
                      max_depth,
                      min_width,
                      cache,
                      samples):
    """
    Recursive adaptive Simpson refinement on a scalar function.
    """
    if (b - a) <= min_width or depth >= max_depth:
        return (b - a) * (fa + 4.0 * fm + fb) / 6.0

    m = 0.5 * (a + b)
    lm = 0.5 * (a + m)
    rm = 0.5 * (m + b)
    flm = _eval_cached(func, lm, cache, samples)
    frm = _eval_cached(func, rm, cache, samples)

    simp = (b - a) * (fa + 4.0 * fm + fb) / 6.0
    left = (m - a) * (fa + 4.0 * flm + fm) / 6.0
    right = (b - m) * (fm + 4.0 * frm + fb) / 6.0
    err = abs((left + right) - simp) / (abs(simp) + 1e-12)

    if err > tol:
        left_val = _adaptive_simpson(func, a, m, fa, fm, flm, tol, depth + 1,
                                     max_depth, min_width, cache, samples)
        right_val = _adaptive_simpson(func, m, b, fm, fb, frm, tol, depth + 1,
                                      max_depth, min_width, cache, samples)
        return left_val + right_val
    return left + right


def _integrate_simpson_adaptive(func, e_min, e_max, tol, max_depth, min_width):
    """
    Global adaptive Simpson integration with sample tracking.
    """
    cache = {}
    samples = []
    fa = _eval_cached(func, e_min, cache, samples)
    fb = _eval_cached(func, e_max, cache, samples)
    mid = 0.5 * (e_min + e_max)
    fm = _eval_cached(func, mid, cache, samples)
    integral = _adaptive_simpson(func, e_min, e_max, fa, fb, fm, tol, 0,
                                 max_depth, min_width, cache, samples)
    return integral, cache, samples


def _select_resonance_potential(r, grid_step, e_scan):
    """
    Select a potential with the strongest phase-shift jump on e_scan.
    """
    candidates = [
        {"depth": 30.0, "width": 0.6, "barrier": 18.0, "barrier_r": 2.0, "barrier_w": 0.35, "l_res": 2},
        {"depth": 36.0, "width": 0.55, "barrier": 24.0, "barrier_r": 2.0, "barrier_w": 0.3, "l_res": 3},
        {"depth": 42.0, "width": 0.5, "barrier": 30.0, "barrier_r": 1.9, "barrier_w": 0.25, "l_res": 3},
    ]
    best = None
    for cand in candidates:
        v_eff = gaussian_well_barrier(
            r,
            depth=cand["depth"],
            width=cand["width"],
            barrier=cand["barrier"],
            barrier_r=cand["barrier_r"],
            barrier_w=cand["barrier_w"],
        )
        delta_l = _scan_delta_l(r, v_eff, e_scan, cand["l_res"], grid_step)
        delta_unwrap = np.unwrap(delta_l)
        delta_diff = np.abs(np.diff(delta_unwrap))
        idx_peak = int(np.argmax(delta_diff))
        ddelta_peak = float(delta_diff[idx_peak])
        e_res = 0.5 * (e_scan[idx_peak] + e_scan[idx_peak + 1])
        if best is None or ddelta_peak > best["ddelta_peak"]:
            best = {
                "params": cand,
                "v_eff": v_eff,
                "ddelta_peak": ddelta_peak,
                "e_res": e_res,
            }
    if best is None:
        raise RuntimeError("No resonance candidate found.")
    return best


def test_continuum_resonance_adaptive_gaussian(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_style("docs", palette="deep_science")
    """
    Compare bisection-vs-Simpson sampling around a physical resonance.
    """
    print("\n=== Continuum Resonance Sampling (Gaussian Well) ===")

    grid = create_sqrt_grid(rmax=15.0, N=700)
    r = grid.r

    mu = 0.2
    temperature = 5.0 * EV_TO_HA

    e_min = 0.1
    e_max = 4.0

    print(f"Grid: N={grid.N}, rmax={grid.rmax:.1f}")
    print(f"Energy range: [{e_min:.2f}, {e_max:.2f}] Ha, T=5 eV")

    e_scan = np.linspace(e_min, e_max, 80)
    best = _select_resonance_potential(r, grid.dxi, e_scan)
    params = best["params"]
    v_eff = best["v_eff"]
    l_res = params["l_res"]

    print(
        "Selected V_eff: depth={:.2f} Ha, width={:.2f} Bohr, barrier={:.2f} "
        "at r={:.2f} (w={:.2f}), l_res={} (ddelta={:.3f})".format(
            params["depth"],
            params["width"],
            params["barrier"],
            params["barrier_r"],
            params["barrier_w"],
            l_res,
            best["ddelta_peak"],
        )
    )

    e_fine = np.linspace(max(e_min, best["e_res"] - 0.25), min(e_max, best["e_res"] + 0.25), 220)
    e_dense = np.linspace(max(e_min, best["e_res"] - 0.06), min(e_max, best["e_res"] + 0.06), 1000)
    e_plot = np.unique(np.concatenate([e_scan, e_fine, e_dense]))

    delta_plot = _scan_delta_l(r, v_eff, e_plot, l_res, grid.dxi)
    delta_unwrap = np.unwrap(delta_plot)
    ddelta = np.abs(np.diff(delta_unwrap))
    idx_ddelta = int(np.argmax(ddelta))
    e_peak_ddelta = 0.5 * (e_plot[idx_ddelta] + e_plot[idx_ddelta + 1])

    r_capture = params["barrier_r"] + 0.5 * params["barrier_w"]
    q_plot = _channel_charge_grid(r, v_eff, e_plot, l_res, grid.dxi, mu, temperature, r_max=r_capture)
    idx_q = int(np.argmax(q_plot))
    e_peak_q = float(e_plot[idx_q])

    match_params = {
        "match_fraction": 0.2,
        "match_r_cut": 0.6 * r[-1],
        "match_kr_min": 6.0,
        "match_v_tol": 1e-6,
        "match_min_points": 16,
        "match_asymptotic": "auto",
        "match_coulomb_tol": 0.1,
        "match_allow_shift": True,
        "match_fallback": "free",
    }

    delta_tol = max(0.2 * best["ddelta_peak"], 0.4)
    e_bisect, bis_meta = _build_bisection_samples(
        e_min,
        e_max,
        n_base=20,
        delta_tol=delta_tol,
        max_depth=10,
        min_width=1e-4,
        r=r,
        v_eff=v_eff,
        l_val=l_res,
        grid_step=grid.dxi,
        match_params=match_params,
    )

    def q_func(x):
        return _channel_charge_at_energy(
            r,
            v_eff,
            x,
            l_res,
            grid.dxi,
            mu,
            temperature,
            r_max=r_capture,
        )

    _, sim_cache, sim_samples = _integrate_simpson_adaptive(
        q_func,
        e_min,
        e_max,
        tol=1e-3,
        max_depth=10,
        min_width=1e-4,
    )
    e_simpson = np.array(sorted(sim_cache.keys()), dtype=float)

    print(f"Bisection samples: {len(e_bisect)} evals (bracket={bis_meta['bracket']})")
    print(f"Simpson samples:   {len(e_simpson)} evals")

    fig, axes = plt.subplots(4, 1, figsize=(8.0, 12.5), sharex=False)
    axes[0].plot(e_plot, q_plot, lw=2.0)
    axes[0].axvline(e_peak_q, color="k", ls="--", lw=1.0, label="peak Q_l")
    axes[0].axvline(e_peak_ddelta, color="tab:orange", ls="--", lw=1.0, label="peak ddelta")
    axes[0].set_xlabel("Energy (Ha)")
    axes[0].set_ylabel("Q_l(E)")
    axes[0].set_title("Energy-resolved channel charge (resonant l; DOS-like proxy)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(e_plot, delta_unwrap, lw=2.0)
    axes[1].axvline(e_peak_ddelta, color="k", ls="--", lw=1.0, label="peak ddelta")
    axes[1].axvline(e_peak_q, color="tab:orange", ls="--", lw=1.0, label="peak Q_l")
    axes[1].set_xlabel("Energy (Ha)")
    axes[1].set_ylabel("delta_l (rad)")
    axes[1].set_title("Phase shift scan (Gaussian well + barrier)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].scatter(e_simpson, np.zeros_like(e_simpson), s=14, marker="|", label="simpson")
    axes[2].scatter(e_bisect, np.ones_like(e_bisect), s=14, marker="|", label="bisection")
    axes[2].axvline(e_peak_q, color="k", ls="--", lw=1.0, label="peak Q_l")
    axes[2].axvline(e_peak_ddelta, color="tab:orange", ls="--", lw=1.0, label="peak ddelta")
    axes[2].axvline(bis_meta["bracket"][0], color="tab:orange", ls=":", lw=1.0, label="bisection bracket")
    axes[2].axvline(bis_meta["bracket"][1], color="tab:orange", ls=":", lw=1.0)
    axes[2].set_xlabel("Energy (Ha)")
    axes[2].set_yticks([0, 1], labels=["simpson", "bisection"])
    axes[2].set_title("Energy samples")
    axes[2].grid(True, axis="x", alpha=0.3)
    axes[2].legend()

    e_bisect_sorted = np.array(sorted(e_bisect), dtype=float)
    de_bisect = np.diff(e_bisect_sorted)
    e_bisect_mid = 0.5 * (e_bisect_sorted[:-1] + e_bisect_sorted[1:])
    de_simpson = np.diff(e_simpson)
    e_simpson_mid = 0.5 * (e_simpson[:-1] + e_simpson[1:])

    axes[3].plot(e_simpson_mid, de_simpson, lw=1.5, label="simpson dE")
    axes[3].plot(e_bisect_mid, de_bisect, lw=1.5, label="bisection dE")
    axes[3].axvline(e_peak_q, color="k", ls="--", lw=1.0, label="peak Q_l")
    axes[3].axvline(e_peak_ddelta, color="tab:orange", ls="--", lw=1.0, label="peak ddelta")
    axes[3].set_xlabel("Energy (Ha)")
    axes[3].set_ylabel("dE")
    axes[3].set_title("Adaptive energy spacing (dE vs E)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    out_path = Path(__file__).with_suffix("").name + ".png"
    fig.tight_layout()
    paths = save_figure(fig, out_path, close=True)
    print(f"Saved {paths['png']} and {paths['pdf']}")


if __name__ == "__main__":
    test_continuum_resonance_adaptive_gaussian()
