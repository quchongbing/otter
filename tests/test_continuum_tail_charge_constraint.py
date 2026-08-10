"""Unit tests for the opt-in charge-constrained B3 tail fit."""

from __future__ import annotations

import numpy as np
import pytest

import otter.electronic.ks_dft as ks_dft
import otter.electronic.full_external as full_external
from otter.electronic.ks_dft import (
    KSDTFConfig,
    _apply_charge_constrained_b3_tail,
    _charge_constraint_failure_meta,
    _source_background_charge,
    _source_electron_charge_target,
    solve_ks_dft_is,
)
from otter.electronic.continuum.tail import (
    _solve_tail_fit_system,
    apply_tail_match,
    linear_response_tail,
)
from otter.data.helpers import trapz_integral
from otter.electronic.full_external import (
    FullExternalConfig,
    _build_continuum_params,
    solve_full_then_external,
)


def _charge(r: np.ndarray, density: np.ndarray) -> float:
    return float(4.0 * np.pi * trapz_integral((r**2) * density, r))


def test_b3_charge_constraint_includes_hermite_bridge() -> None:
    r = np.linspace(0.08, 18.0, 1400)
    n0 = 0.035
    mu = 0.42
    temperature = 0.24
    idx_cut = 520
    n_raw = linear_response_tail(
        r,
        n0,
        mu,
        temperature,
        A=0.18,
        B=0.07,
        delta=0.35,
    )
    # Add an inner-only perturbation to verify that the retained prefix enters
    # the finite-box charge equation but is not modified by the tail fit.
    n_raw[:idx_cut] += 0.004 * np.exp(-0.6 * r[:idx_cut])
    target = _charge(r, n_raw) + 0.025

    constrained, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        mu,
        temperature,
        idx_cut,
        fit_points=36,
        r_fit_max=float(r[idx_cut + 180]),
        fit_window_mode="physical",
        blend_points=12,
        model="full",
        charge_target=target,
    )

    assert np.array_equal(constrained[:idx_cut], n_raw[:idx_cut])
    assert abs(_charge(r, constrained) - target) < 2.0e-9
    assert bool(meta["charge_constraint_applied"])
    assert abs(float(meta["charge_constraint_residual"])) < 2.0e-9
    assert np.isfinite(float(meta["charge_constraint_unconstrained_fit_rms"]))
    assert np.isfinite(float(meta["charge_constraint_fit_rms_ratio"]))
    assert np.isfinite(float(meta["charge_constraint_coeff_delta_rel"]))
    assert np.isfinite(float(meta["charge_constraint_profile_delta_rel"]))
    assert bool(meta["charge_constraint_accepted"])
    assert float(meta["charge_constraint_tail_min"]) >= 0.0


def test_b3_charge_row_survives_tiny_yukawa_basis_without_cancellation() -> None:
    """A tiny A column must not vanish when the response is added to n0."""
    r = np.linspace(0.08, 30.0, 3000)
    n0 = 0.035
    mu = 10.0
    temperature = 1.0e-3
    idx_cut = int(np.searchsorted(r, 20.0))
    n_raw = linear_response_tail(
        r,
        n0,
        mu,
        temperature,
        A=0.18,
        B=0.07,
        delta=0.35,
    )
    target = _charge(r, n_raw) + 1.0e-3

    out, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        mu,
        temperature,
        idx_cut,
        fit_points=48,
        r_fit_max=24.0,
        fit_window_mode="physical",
        blend_points=12,
        model="full",
        charge_target=target,
        charge_constraint_fit_rms_ratio_max=None,
        charge_constraint_profile_delta_rel_max=None,
    )

    # The former splice(unit)-splice(zero) construction rounded this response
    # to exactly zero because the local basis is about 1e-22 while n0 is 0.035.
    assert 0.0 < abs(float(meta["charge_constraint_row_A"])) < 1.0e-15
    assert abs(_charge(r, out) - target) < 2.0e-9
    assert bool(meta["charge_constraint_accepted"])


def test_b3_charge_constraint_rejects_degraded_fit_and_can_disable_quality_guard() -> None:
    r = np.linspace(0.08, 18.0, 1400)
    n0 = 0.035
    mu = 0.42
    temperature = 0.24
    idx_cut = 520
    n_raw = linear_response_tail(
        r, n0, mu, temperature, A=0.18, B=0.07, delta=0.35
    )
    n_raw[:idx_cut] += 0.004 * np.exp(-0.6 * r[:idx_cut])
    target = _charge(r, n_raw) - 1.0
    kwargs = dict(
        fit_points=36,
        r_fit_max=float(r[idx_cut + 180]),
        fit_window_mode="physical",
        blend_points=12,
        model="full",
        charge_target=target,
    )

    with pytest.raises(ValueError, match="fit RMS ratio"):
        apply_tail_match(
            r,
            n_raw,
            n0,
            mu,
            temperature,
            idx_cut,
            **kwargs,
        )

    out, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        mu,
        temperature,
        idx_cut,
        **kwargs,
        charge_constraint_fit_rms_ratio_max=None,
        charge_constraint_profile_delta_rel_max=None,
    )
    assert bool(meta["charge_constraint_accepted"])
    assert np.min(out[idx_cut:]) >= 0.0
    assert abs(_charge(r, out) - target) < 2.0e-9


def test_b3_charge_constraint_always_rejects_negative_tail() -> None:
    r = np.linspace(0.08, 18.0, 1400)
    n0 = 0.035
    mu = 0.42
    temperature = 0.24
    idx_cut = 520
    n_raw = linear_response_tail(
        r, n0, mu, temperature, A=0.18, B=0.07, delta=0.35
    )
    target = _charge(r, n_raw) - 100.0

    with pytest.raises(ValueError, match="minimum density"):
        apply_tail_match(
            r,
            n_raw,
            n0,
            mu,
            temperature,
            idx_cut,
            fit_points=36,
            r_fit_max=float(r[idx_cut + 180]),
            fit_window_mode="physical",
            blend_points=12,
            model="full",
            charge_target=target,
            charge_constraint_fit_rms_ratio_max=None,
            charge_constraint_profile_delta_rel_max=None,
        )


def test_b3_charge_constraint_is_opt_in() -> None:
    r = np.linspace(0.1, 12.0, 800)
    n0 = 0.02
    n_raw = linear_response_tail(r, n0, 0.25, 0.3, 0.1, 0.03, -0.2)
    out, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        0.25,
        0.3,
        300,
        fit_points=24,
        blend_points=6,
        model="full",
    )
    assert np.all(np.isfinite(out))
    assert not bool(meta["charge_constraint_applied"])
    assert np.isnan(float(meta["charge_constraint_residual"]))


def test_apply_tail_match_preserves_unconstrained_auto_selection() -> None:
    """The orchestration layer must report the model resolved by the local fit."""
    r = np.linspace(1.0, 8.0, 500)
    n0 = 0.05
    n_raw = linear_response_tail(
        r,
        n0,
        0.8,
        0.08,
        A=-0.03,
        B=0.8,
        delta=0.4,
    )
    _, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        0.8,
        0.08,
        80,
        fit_points=40,
        model="auto",
    )
    assert meta["model_requested"] == "auto"
    assert meta["model_selected"] == "full"


def test_auto_model_is_selected_before_charge_constraint() -> None:
    """A global charge equality must not erase a locally resolved Friedel term."""
    r = np.linspace(0.08, 18.0, 1400)
    n0 = 0.035
    mu = 0.42
    temperature = 0.03
    idx_cut = int(np.searchsorted(r, 6.0))
    n_raw = linear_response_tail(
        r,
        n0,
        mu,
        temperature,
        A=0.18,
        B=1.0,
        delta=0.35,
    )
    n_raw[:idx_cut] += 0.004 * np.exp(-0.6 * r[:idx_cut])

    out, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        mu,
        temperature,
        idx_cut,
        fit_points=60,
        r_fit_max=10.0,
        fit_window_mode="physical",
        blend_points=12,
        model="auto",
        charge_target=_charge(r, n_raw) - 1.0,
        charge_constraint_fit_rms_ratio_max=None,
        charge_constraint_profile_delta_rel_max=None,
    )

    assert np.all(np.isfinite(out))
    assert meta["model_requested"] == "auto"
    assert meta["model_selection_basis"] == "unconstrained_local_fit"
    assert meta["model_selection_unconstrained_model"] == "full"
    assert meta["model_selected"] == "full"
    # This is the regression condition: before the fix, the constrained
    # residual comparison was below the 15% auto threshold and selected
    # ``a_only`` even though the unconstrained data clearly required B3.
    assert float(meta["model_selection_unconstrained_fit_rel_improve_full"]) > 0.15
    assert float(meta["charge_constrained_fit_rel_improve_full"]) < 0.15
    assert abs(_charge(r, out) - (_charge(r, n_raw) - 1.0)) < 2.0e-9


def test_zero_integral_row_with_zero_target_is_redundant() -> None:
    r = np.linspace(0.1, 1.0, 6)
    coeffs, _, _, diag = _solve_tail_fit_system(
        np.zeros((4, 1)),
        np.zeros(4),
        r=r,
        n_r=np.zeros_like(r),
        idx_cut=1,
        r_fit=r[1:5],
        deriv_row=np.zeros(1),
        match_value_weight=0.0,
        match_slope_weight=0.0,
        integral_row=np.zeros(1),
        integral_target=0.0,
    )
    np.testing.assert_array_equal(coeffs, np.zeros(1))
    assert bool(diag["integral_constraint_applied"])
    assert bool(diag["integral_constraint_redundant"])
    assert float(diag["integral_constraint_residual"]) == 0.0


def test_extreme_singular_value_ratio_reports_infinity_without_warning() -> None:
    from otter.electronic.continuum.tail import (
        _finite_singular_value_condition_number,
    )

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        condition = _finite_singular_value_condition_number(
            np.asarray((1.0, np.nextafter(0.0, 1.0)))
        )

    assert condition == float("inf")


def test_default_tail_match_works_without_numpy_trapezoid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-out/default path remains compatible with NumPy 1.x."""
    r = np.linspace(0.1, 12.0, 800)
    n0 = 0.02
    n_raw = linear_response_tail(r, n0, 0.25, 0.3, 0.1, 0.03, -0.2)
    monkeypatch.delattr(np, "trapezoid", raising=False)

    out, meta = apply_tail_match(
        r,
        n_raw,
        n0,
        0.25,
        0.3,
        300,
        fit_points=24,
        blend_points=6,
        model="full",
    )
    assert np.all(np.isfinite(out))
    assert not bool(meta["charge_constraint_applied"])


def test_source_charge_targets_use_full_Z_and_external_zero() -> None:
    r = np.linspace(0.05, 10.0, 900)
    n0 = 0.04
    g_ii = np.ones_like(r)
    g_ii[r < 1.7] = 0.0
    q_background = _source_background_charge(r, n0, g_ii)
    electron_full = _source_electron_charge_target(r, n0, g_ii, 6.0)
    electron_ext = _source_electron_charge_target(r, n0, g_ii, 0.0)

    assert abs((electron_full + q_background) - 6.0) < 1.0e-11
    assert abs(electron_ext + q_background) < 1.0e-11


def test_full_external_config_keeps_constraint_disabled_by_default() -> None:
    cfg = FullExternalConfig(element="C", temperature_ev=10.0, rho_g_cc=3.0)
    params_default = _build_continuum_params(
        cfg,
        l_max=4,
        r_ws=2.0,
        rmax=20.0,
        adaptive_mode="shared",
        b3_stage_mode="in_scf",
        e_max_mode="fixed",
    )
    assert params_default["b3_source_charge_constraint"] is False

    cfg.b3_source_charge_constraint = True
    params_enabled = _build_continuum_params(
        cfg,
        l_max=4,
        r_ws=2.0,
        rmax=20.0,
        adaptive_mode="shared",
        b3_stage_mode="in_scf",
        e_max_mode="fixed",
    )
    assert params_enabled["b3_source_charge_constraint"] is True
    assert params_enabled["b3_charge_constraint_fit_rms_ratio_max"] == 10.0
    assert params_enabled["b3_charge_constraint_profile_delta_rel_max"] == 10.0

    cfg.b3_charge_constraint_fit_rms_ratio_max = None
    cfg.b3_charge_constraint_profile_delta_rel_max = None
    params_disabled_guards = _build_continuum_params(
        cfg,
        l_max=4,
        r_ws=2.0,
        rmax=20.0,
        adaptive_mode="shared",
        b3_stage_mode="in_scf",
        e_max_mode="fixed",
    )
    assert params_disabled_guards["b3_charge_constraint_fit_rms_ratio_max"] is None
    assert params_disabled_guards["b3_charge_constraint_profile_delta_rel_max"] is None


def test_charge_constrained_helper_respects_fixed_tail_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_apply_tail_match(
        r: np.ndarray,
        density: np.ndarray,
        n0: float,
        mu_id: float,
        temperature: float,
        idx_cut: int,
        **kwargs: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        captured.update(
            n0=n0,
            mu_id=mu_id,
            fit_max=kwargs["charge_constraint_fit_rms_ratio_max"],
            profile_max=kwargs["charge_constraint_profile_delta_rel_max"],
        )
        return np.asarray(density, dtype=float), {
            "charge_constraint_applied": True,
            "charge_constraint_accepted": True,
        }

    monkeypatch.setattr(
        "otter.electronic.continuum.tail.apply_tail_match", fake_apply_tail_match
    )
    r = np.linspace(0.1, 10.0, 100)
    density = np.full_like(r, 0.03)
    _, meta = _apply_charge_constrained_b3_tail(
        r,
        density,
        n0=0.03,
        mu_id=0.2,
        temperature=0.1,
        params={
            "tail_r_cut": 6.0,
            "tail_n0_fixed": 0.031,
            "tail_mu_id_fixed": 0.24,
            "b3_charge_constraint_fit_rms_ratio_max": 7.0,
            "b3_charge_constraint_profile_delta_rel_max": None,
        },
        electron_charge_target=15.0,
    )

    assert captured == {
        "n0": 0.031,
        "mu_id": 0.24,
        "fit_max": 7.0,
        "profile_max": None,
    }
    assert meta["tail_n0_used"] == 0.031
    assert meta["tail_mu_id_used"] == 0.24
    assert bool(meta["charge_constraint_requested"])


def test_charge_constraint_failure_meta_is_explicit() -> None:
    meta = _charge_constraint_failure_meta(
        {"applied": True, "model_selected": "full"},
        ValueError("quality guard"),
    )
    assert bool(meta["charge_constraint_requested"])
    assert not bool(meta["charge_constraint_applied"])
    assert not bool(meta["charge_constraint_accepted"])
    assert meta["charge_constraint_failure_reason"] == "quality guard"


def _empty_bound_solution(
    potential: np.ndarray,
    r: np.ndarray,
    step: float,
    l_list: np.ndarray,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a shape-correct no-bound-state spectrum for lightweight SCF tests."""
    del potential, step, kwargs
    n_l = int(np.asarray(l_list).size)
    return (
        np.empty((n_l, 0), dtype=float),
        np.empty((n_l, 0, np.asarray(r).size), dtype=float),
    )


def _patch_lightweight_bound_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the sparse eigen solve without changing continuum/B3 control flow."""
    monkeypatch.setattr(
        ks_dft, "solve_bound_states_sparse_numerov", _empty_bound_solution
    )
    monkeypatch.setattr(
        ks_dft,
        "_refine_shallow_bound_states_zero_tail",
        lambda *args, **kwargs: (
            args[2],
            args[3],
            {"applied": False, "states": []},
        ),
    )
    monkeypatch.setattr(
        ks_dft,
        "bound_state_reliability_diagnostics",
        lambda *args, **kwargs: {
            "states": [],
            "shallowest": None,
            "shallowest_status": "none",
        },
    )


def test_combined_full_external_exact_b3_failure_is_not_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful full constraint must not hide a failed external constraint."""
    _patch_lightweight_bound_channel(monkeypatch)
    exact_calls = 0

    def fake_exact_tail(
        r: np.ndarray,
        density: np.ndarray,
        *,
        n0: float,
        mu_id: float,
        temperature: float,
        params: dict[str, object],
        electron_charge_target: float,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del n0, mu_id, temperature, params, electron_charge_target
        nonlocal exact_calls
        exact_calls += 1
        # With target="cont", each SCF iteration constrains the full
        # continuum first and the external continuum second.
        if exact_calls % 2 == 0:
            raise ValueError("synthetic external quality rejection")
        return np.asarray(density, dtype=float).copy(), {
            "charge_constraint_requested": True,
            "charge_constraint_applied": True,
            "charge_constraint_accepted": True,
            "charge_constraint_residual": 0.0,
        }

    monkeypatch.setattr(
        ks_dft, "_apply_charge_constrained_b3_tail", fake_exact_tail
    )
    cfg = KSDTFConfig(
        Z=1,
        temperature=0.5,
        mu=0.2,
        r_ws=1.0,
        rmax=4.0,
        rmax_mult=None,
        n_points=48,
        l_list=np.array([0]),
        n_states=1,
        continuum_model="ideal",
        compute_external=True,
        mu_mode="fixed",
        n0_mode="ideal",
        mixing_scheme="linear",
        mix=1.0,
        max_iter=3,
        tol=1.0e6,
        dn_tol=1.0e6,
        dv_tol=1.0e6,
        bound_zero_tail_refine=False,
        continuum_params={
            "tail_match": True,
            "tail_mode": "in_scf",
            "tail_match_target": "cont",
            "tail_r_cut": 2.5,
            "tail_fit_points": 8,
            "tail_blend_points": 0,
            "tail_model": "a_only",
            "tail_fallback_on_error": True,
            "b3_source_charge_constraint": True,
        },
    )

    with pytest.warns(
        RuntimeWarning, match="Charge-constrained external B3 fit failed"
    ):
        result = solve_ks_dft_is(cfg)

    assert exact_calls == 2 * cfg.max_iter
    assert not bool(result["converged"])
    assert all(
        bool(item["b3_charge_constraint_full_applied"])
        and not bool(item["b3_charge_constraint_ext_applied"])
        for item in result["history"]
    )
    assert bool(result["b3_charge_constraint_full_applied"])
    assert not bool(result["b3_charge_constraint_ext_applied"])
    assert (
        result["n_ext_tail_meta"]["charge_constraint_failure_reason"]
        == "synthetic external quality rejection"
    )


def test_final_refresh_constraint_failure_revokes_prior_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refreshed return state, not the preceding iterate, owns convergence."""
    _patch_lightweight_bound_channel(monkeypatch)
    potential_updates = 0

    def fake_effective_potential(
        r: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        del args, kwargs
        nonlocal potential_updates
        potential_updates += 1
        # Iteration 1 hands iteration 2 a recognizable nonzero map. Iteration
        # 2 then hands the final-refresh evaluation the zero map.
        value = -0.5 if potential_updates == 1 else 0.0
        return np.full_like(np.asarray(r, dtype=float), value)

    def fake_exact_tail(
        r: np.ndarray,
        density: np.ndarray,
        *,
        params: dict[str, object],
        **kwargs: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del r, kwargs
        v_eff = np.asarray(params["v_eff"], dtype=float)
        if np.all(v_eff == 0.0):
            raise ValueError("synthetic final-refresh rejection")
        return np.asarray(density, dtype=float).copy(), {
            "charge_constraint_requested": True,
            "charge_constraint_applied": True,
            "charge_constraint_accepted": True,
            "charge_constraint_residual": 0.0,
        }

    monkeypatch.setattr(
        ks_dft, "effective_potential_full", fake_effective_potential
    )
    monkeypatch.setattr(
        ks_dft, "_apply_charge_constrained_b3_tail", fake_exact_tail
    )
    cfg = KSDTFConfig(
        Z=1,
        temperature=0.5,
        mu=0.0,
        mu_mode="neutral",
        mu_strategy="inner",
        mu_bounds=(-2.0, 2.0),
        mu_tol=1.0e-8,
        mu_max_iter=30,
        r_ws=1.0,
        rmax=4.0,
        rmax_mult=None,
        n_points=48,
        l_list=np.array([0]),
        n_states=1,
        continuum_model="ideal",
        compute_external=False,
        n0_mode="ideal",
        mixing_scheme="linear",
        mix=1.0,
        max_iter=2,
        tol=1.0e6,
        dn_tol=1.0e6,
        dv_tol=1.0e6,
        bound_zero_tail_refine=False,
        continuum_params={
            "tail_match": True,
            "tail_mode": "in_scf",
            "tail_match_target": "cont",
            "tail_r_cut": 2.5,
            "tail_fit_points": 8,
            "tail_blend_points": 0,
            "tail_model": "a_only",
            "tail_fallback_on_error": True,
            "b3_source_charge_constraint": True,
        },
    )

    with pytest.warns(
        RuntimeWarning, match="synthetic final-refresh rejection"
    ):
        result = solve_ks_dft_is(cfg)

    assert potential_updates == cfg.max_iter
    assert len(result["history"]) == cfg.max_iter
    assert bool(result["history"][-1]["b3_charge_constraint_full_applied"])
    assert not bool(result["converged"])
    assert not bool(result["b3_charge_constraint_full_applied"])
    assert (
        result["n_cont_tail_meta"]["charge_constraint_failure_reason"]
        == "synthetic final-refresh rejection"
    )


def _synthetic_full_result(cfg: KSDTFConfig) -> dict[str, object]:
    """Minimal, internally consistent high-level payload for metadata tests."""
    r = np.linspace(0.05, float(cfg.rmax), int(cfg.n_points))
    n0 = 0.025
    n_bound = np.zeros_like(r)
    n_cont = np.full_like(r, n0)
    n_full = n_bound + n_cont
    n_ext = np.full_like(r, n0)
    n_ion = np.zeros_like(r)
    n_pa = n_full - n_ext
    return {
        "Z": float(cfg.Z),
        "r": r,
        "r_bound": r.copy(),
        "g_ii": (r >= float(cfg.r_ws)).astype(float),
        "n0": float(n0),
        "n_bound": n_bound,
        "n_ion": n_ion,
        "n_cont": n_cont,
        "n_free": n_cont.copy(),
        "n_cont_dft_raw": n_cont.copy(),
        "n_cont_pre_tail": n_cont.copy(),
        "n_cont_tail_meta": {
            "charge_constraint_requested": False,
            "charge_constraint_applied": False,
        },
        "n_full": n_full,
        "n_full_pre_tail": n_full.copy(),
        "n_full_source": n_full.copy(),
        "n_full_source_provenance": "synthetic_fixed_point_candidate",
        "n_full_tail_meta": {
            "charge_constraint_requested": True,
            "charge_constraint_applied": True,
            "charge_constraint_accepted": True,
            "charge_constraint_residual": 0.0,
        },
        "n_ext": n_ext,
        "n_pa": n_pa,
        "n_scr": n_pa - n_ion,
        "v_full": -np.exp(-r),
        "v_ext": np.zeros_like(r),
        "history": [],
        "converged": True,
        "r_ws": float(cfg.r_ws),
        "mu": float(cfg.mu),
        "zbar": 1.0,
        "zero_tail_bound_meta": {"applied": False, "states": []},
        "bound_state_diagnostics": {
            "states": [],
            "shallowest": None,
            "shallowest_status": "none",
        },
    }


def test_legacy_b3_aliases_are_resolved_in_high_level_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metadata must describe the controls actually passed to the KS solver."""
    monkeypatch.setattr(
        full_external, "solve_ks_dft_is", lambda cfg: _synthetic_full_result(cfg)
    )
    monkeypatch.setattr(
        full_external, "_build_bound_tables_and_dos", lambda **kwargs: {}
    )
    monkeypatch.setattr(
        full_external, "_build_scattering_continuum_dos", lambda **kwargs: {}
    )

    cfg = FullExternalConfig(
        element="H",
        temperature_ev=10.0,
        rho_g_cc=1.0,
        n_points=48,
        rmax_mult=4.0,
        run_mode="full",
        ext_scf_enabled=False,
        save_data=False,
        show_scf_progress=False,
        verbose=False,
        # Exercise both historical aliases: the empty explicit controls
        # resolve from cont_tail_match and cont_tail_match_target.
        b3_tail_stage1_mode="",
        b3_tail_stage2_mode="",
        b3_tail_target="",
        cont_tail_match=True,
        cont_tail_match_target="full",
        b3_source_charge_constraint=True,
        ext_b3_tail_mode="off",
    )
    result = solve_full_then_external(cfg)
    meta = result["meta"]

    assert meta["b3_tail_stage2_mode_raw"] == ""
    assert meta["b3_tail_stage2_mode"] == "in_scf"
    assert meta["b3_tail_target_raw"] == ""
    assert meta["b3_tail_target"] == "full"
    assert bool(meta["b3_charge_constraint_requested"])
    assert bool(meta["b3_charge_constraint_applied"])
    assert float(meta["b3_charge_constraint_residual"]) == 0.0


def test_high_level_external_constraint_state_replaces_full_only_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public result-level external metadata must describe the ext SCF."""
    monkeypatch.setattr(
        full_external, "solve_ks_dft_is", lambda cfg: _synthetic_full_result(cfg)
    )
    monkeypatch.setattr(
        full_external, "_build_bound_tables_and_dos", lambda **kwargs: {}
    )
    monkeypatch.setattr(
        full_external, "_build_scattering_continuum_dos", lambda **kwargs: {}
    )

    ext_tail_meta = {
        "charge_constraint_requested": True,
        "charge_constraint_applied": True,
        "charge_constraint_accepted": True,
        "charge_constraint_residual": 0.0,
    }

    def fake_external_scf(**kwargs):
        r = np.asarray(kwargs["r"], dtype=float)
        n0 = float(kwargs["n0"])
        return (
            np.full_like(r, n0),
            np.zeros_like(r),
            {
                "iters": 1,
                "err": 0.0,
                "converged": True,
                "history": [],
                "final_ph_kappa": 0.0,
                "tail_meta": dict(ext_tail_meta),
            },
        )

    monkeypatch.setattr(
        full_external, "_external_fixed_mu_scf", fake_external_scf
    )
    cfg = FullExternalConfig(
        element="H",
        temperature_ev=10.0,
        rho_g_cc=1.0,
        n_points=48,
        rmax_mult=4.0,
        run_mode="full+ext",
        ext_scf_enabled=True,
        save_data=False,
        show_scf_progress=False,
        verbose=False,
        b3_tail_stage1_mode="in_scf",
        b3_tail_stage2_mode="in_scf",
        ext_b3_tail_mode="in_scf",
        b3_tail_target="full",
        b3_source_charge_constraint=True,
    )

    result = solve_full_then_external(cfg)

    assert result["n_ext_tail_meta"] == ext_tail_meta
    assert bool(result["b3_charge_constraint_ext_applied"])
    assert bool(result["meta"]["ext_b3_charge_constraint_applied"])
