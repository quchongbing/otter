"""
tests/test_continuum_scattering_coulomb_phase_shift.py

Purpose
-------
Validate Coulomb matching for a single (energy, l) scattering state and
confirm that phase shifts are stable across a small energy increment.

Methods
-------
- Build a sqrt grid and a softened Coulomb potential.
- Propagate the regular Numerov solution for two nearby energies.
- Normalize using Coulomb asymptotic matching and extract delta_l.
- Compare the normalized tail to the fitted Coulomb combination.

Equations
---------
u(r) ~ A [F_l(eta, kr) cos(delta) - G_l(eta, kr) sin(delta)]
u_norm = (sqrt(2/pi)/sqrt(k_use)) * u / A

References
----------
- C. E. Starrett & D. Saumon (2014), Eq. (A3).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from otter.numerics.grids import create_sqrt_grid
from otter.electronic.continuum import scattering as qmod
from otter.plotting import save_figure, set_style


def _soft_coulomb(r: np.ndarray, z: float, r_soft: float) -> np.ndarray:
    return -z / np.sqrt(r * r + r_soft * r_soft)


def _match_and_fit(r, v_eff, energy, l_val, grid_step, match_params):
    u_raw = qmod._numerov_propagate_sqrt(r, v_eff, energy, l_val, grid_step)
    u_norm, delta, amp_target, meta = qmod._match_scattering_u(
        u_raw,
        r,
        v_eff,
        energy,
        l_val,
        match_fraction=match_params["match_fraction"],
        match_slice=None,
        match_r_cut=match_params["match_r_cut"],
        match_fraction_mode="index",
        match_width=None,
        match_kr_min=match_params["match_kr_min"],
        match_v_tol=match_params["match_v_tol"],
        match_min_points=match_params["match_min_points"],
        match_asymptotic=match_params["match_asymptotic"],
        match_coulomb_tol=match_params["match_coulomb_tol"],
        match_allow_shift=match_params["match_allow_shift"],
        match_fallback=match_params["match_fallback"],
    )

    match_slice, _ = qmod._select_match_window(
        r,
        v_eff,
        energy,
        l_val,
        None,
        match_params["match_fraction"],
        match_params["match_r_cut"],
        None,
        "index",
        match_params["match_kr_min"],
        match_params["match_v_tol"],
        match_params["match_min_points"],
    )
    i0, i1 = match_slice
    rm = r[i0:i1]
    v_tail = v_eff[i0:i1]
    u_f, u_g, meta_basis = qmod._build_asymptotic_basis(
        rm,
        v_tail,
        energy,
        l_val,
        match_params["match_asymptotic"],
        0.0 if match_params["match_v_tol"] is None else float(match_params["match_v_tol"]),
        match_params["match_coulomb_tol"],
        match_params["match_allow_shift"],
    )
    u_fit = amp_target * (np.cos(delta) * u_f - np.sin(delta) * u_g)
    rel_err = np.linalg.norm(u_norm[i0:i1] - u_fit) / max(np.linalg.norm(u_fit), 1e-12)
    return u_norm, delta, rel_err, meta, meta_basis, match_slice


def test_continuum_scattering_coulomb_phase_shift(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    set_style("docs", palette="deep_science")
    print("\n=== Continuum Coulomb Phase-Shift Test ===")
    print(f"HAVE_COULOMB={qmod._HAVE_COULOMB}")
    print(f"COULOMB_BACKEND={getattr(qmod, '_COULOMB_BACKEND', 'unknown')}")
    print(f"HAVE_COULOMB_ASYM={getattr(qmod, '_HAVE_COULOMB_ASYM', False)}")

    grid = create_sqrt_grid(rmax=30.0, N=900)
    r = grid.r
    z_eff = 1.0
    r_soft = r[0]
    v_eff = _soft_coulomb(r, z_eff, r_soft)

    l_val = 0
    energy = 0.5
    de = 0.02

    match_params = {
        "match_fraction": 0.25,
        "match_r_cut": 0.6 * r[-1],
        "match_kr_min": 6.0,
        "match_v_tol": None,
        "match_min_points": 24,
        "match_asymptotic": "coulomb",
        "match_coulomb_tol": 0.1,
        "match_allow_shift": False,
        "match_fallback": "free",
    }

    u0, delta0, rel0, meta0, meta_basis0, slice0 = _match_and_fit(
        r, v_eff, energy, l_val, grid.dxi, match_params
    )
    u1, delta1, rel1, meta1, meta_basis1, slice1 = _match_and_fit(
        r, v_eff, energy + de, l_val, grid.dxi, match_params
    )
    delta_diff = delta1 - delta0

    print(f"Grid: N={grid.N}, rmax={grid.rmax:.1f}, r0={r[0]:.3e}")
    print(f"Energy={energy:.3f} Ha, dE={de:.3f} Ha, l={l_val}")
    print(f"Delta(E)={delta0:.6f}, Delta(E+dE)={delta1:.6f}, dDelta={delta_diff:.6f}")
    print(f"Fit rel_err: E={rel0:.3e}, E+dE={rel1:.3e}")
    print(f"Match meta: {meta0}")
    print(f"Basis meta: {meta_basis0}")

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.5), sharex=True)
    i0, i1 = slice0
    axes[0].plot(r[i0:i1], u0[i0:i1], label="u_norm (E)", lw=2.0)
    axes[0].plot(r[i0:i1], u1[i0:i1], label="u_norm (E+dE)", lw=1.5, alpha=0.8)
    axes[0].set_ylabel("u(r)")
    axes[0].set_title("Coulomb-matched normalized u(r) in tail window")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot([energy, energy + de], [delta0, delta1], marker="o", lw=2.0)
    axes[1].set_xlabel("Energy (Ha)")
    axes[1].set_ylabel("delta_l (rad)")
    axes[1].set_title("Phase shift vs energy (Coulomb reference)")
    axes[1].grid(True, alpha=0.3)

    out_path = Path(__file__).with_suffix("").name + ".png"
    fig.tight_layout()
    paths = save_figure(fig, out_path, close=True)
    print(f"Saved {paths['png']} and {paths['pdf']}")

    if meta_basis0.get("kind") != "coulomb":
        raise AssertionError(f"Expected Coulomb basis, got {meta_basis0.get('kind')}")
    backend = meta_basis0.get("coulomb_backend")
    if backend is None:
        raise AssertionError("Coulomb backend missing; basis not constructed.")
    assert backend == "asymptotic"
    assert abs(delta_diff) < 0.3
    rel_tol = 0.35 if backend == "asymptotic" else 0.2
    assert rel0 < rel_tol


if __name__ == "__main__":
    test_continuum_scattering_coulomb_phase_shift()
