"""Lock the diagnostic (non-self-consistent) semantics of post-SCF B3."""

from __future__ import annotations

import numpy as np

import otter.electronic.full_external as full_external


def test_post_b3_keeps_fixed_point_potential_separate(monkeypatch) -> None:
    r = np.linspace(1.0e-4, 15.0, 128)
    n0 = 0.02
    n_cont = n0 + 1.0e-3 * np.exp(-0.3 * r)
    n_bound = 2.0e-3 * np.exp(-2.0 * r)
    v_full = -np.exp(-r)
    rebuilt = 0.5 * np.exp(-0.5 * r)

    def fake_tail_match(*args, **kwargs):
        density = np.asarray(args[1], dtype=float).copy()
        density[r >= 5.0] = n0
        return density, {"model_selected": "a_only"}

    def fake_effective_potential(*args, **kwargs):
        return rebuilt.copy()

    monkeypatch.setattr(full_external, "apply_tail_match", fake_tail_match)
    monkeypatch.setattr(
        full_external,
        "effective_potential_full",
        fake_effective_potential,
    )

    cfg = full_external.FullExternalConfig(
        element="H",
        temperature_ev=10.0,
        rho_g_cc=1.0,
        b3_tail_stage2_mode="post",
        b3_r_cut_mult=5.0,
        b3_r_fit_max_mult=7.0,
        save_data=False,
    )
    result = {
        "r": r,
        "n0": n0,
        "mu": 0.1,
        "n_cont": n_cont,
        "n_bound": n_bound,
        "n_full": n_cont + n_bound,
        "v_full": v_full,
        "g_ii": np.where(r < 1.0, 0.0, 1.0),
        "Z": 1.0,
    }

    out = full_external._apply_b3_post_to_full_result(
        result,
        cfg,
        r_ws=1.0,
        rmax=15.0,
        stage_mode="post",
    )

    assert not np.array_equal(out["n_cont"], n_cont)
    assert np.array_equal(out["v_full"], v_full)
    assert np.array_equal(out["v_full_fixed_point"], v_full)
    assert np.array_equal(out["v_post_from_density"], rebuilt)
    assert "v_scf" not in out
    assert out["b3_post_self_consistent"] is False
    assert out["b3_tail_meta"]["diagnostic_only"] is True
    assert out["b3_tail_meta"]["self_consistent"] is False


def test_external_b3_rebuilds_continuum_for_full_target(monkeypatch) -> None:
    """The no-nucleus external system has no separate bound contribution."""
    r = np.linspace(1.0e-3, 2.0, 32)
    seen_targets: list[str] = []

    class DummyContinuum:
        @staticmethod
        def density(r_eval, mu, temperature, params):
            return np.full_like(r_eval, 0.02)

    def fake_rebuild(r_full, n_eval, *, params, n0, **kwargs):
        seen_targets.append(str(params["tail_match_target"]))
        density = np.full_like(r_full, float(n0))
        return density, density.copy(), {"applied": True}

    monkeypatch.setattr(
        "otter.electronic.ks_dft._select_continuum_model",
        lambda model: DummyContinuum(),
    )
    monkeypatch.setattr(full_external, "_rebuild_continuum_on_full_grid", fake_rebuild)
    monkeypatch.setattr(
        full_external,
        "effective_potential_external",
        lambda r, *args, **kwargs: np.zeros_like(r),
    )

    full_external._external_fixed_mu_scf(
        r=r,
        mu=0.1,
        temperature_ha=0.2,
        n0=0.02,
        ext_params_base={
            "tail_match": True,
            "tail_match_target": "full",
            "solve_rmax": 1.0,
            "tail_fallback_on_error": False,
            "source_closure": False,
        },
        mix=0.25,
        dn_tol=1.0e-4,
        dv_tol=1.0e-4,
        max_iter=1,
        adaptive_mix=False,
        mixing_scheme="linear",
        mixing_m=2,
        mixing_w0=5.0e-4,
        ext_b3_tail_mode="in_scf",
        verbose=False,
        print_every=1,
    )

    assert seen_targets == ["cont"]
