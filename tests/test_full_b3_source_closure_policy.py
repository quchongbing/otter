"""Regression tests for target-aware full-branch B3 source closure."""

from __future__ import annotations

from otter.electronic.full_external import (
    FullExternalConfig,
    _resolve_b3_tail_controls,
    _resolve_full_b3_source_closure_policy,
)


def test_auto_source_closure_uses_matched_full_density_directly() -> None:
    assert not _resolve_full_b3_source_closure_policy(
        setting=None,
        tail_target="full",
    )
    assert not _resolve_full_b3_source_closure_policy(
        setting=None,
        tail_target="both",
    )


def test_auto_source_closure_preserves_legacy_continuum_target() -> None:
    assert _resolve_full_b3_source_closure_policy(
        setting=None,
        tail_target="cont",
    )


def test_explicit_source_closure_setting_overrides_auto_policy() -> None:
    assert _resolve_full_b3_source_closure_policy(
        setting=True,
        tail_target="full",
    )
    assert not _resolve_full_b3_source_closure_policy(
        setting=False,
        tail_target="cont",
    )


def test_full_external_config_accepts_auto_source_closure() -> None:
    cfg = FullExternalConfig(
        element="H",
        temperature_ev=2.0,
        rho_g_cc=1.0,
        full_b3_use_source_closure="auto",
    )

    assert cfg.full_b3_use_source_closure is None


def test_auto_fit_window_preserves_cont_target_and_spans_full_target() -> None:
    cfg_cont = FullExternalConfig(
        element="H",
        temperature_ev=2.0,
        rho_g_cc=1.0,
        b3_tail_target="cont",
        b3_tail_fit_window_mode="auto",
    )
    cfg_full = FullExternalConfig(
        element="H",
        temperature_ev=2.0,
        rho_g_cc=1.0,
        b3_tail_target="full",
        b3_tail_fit_window_mode="auto",
    )

    controls_cont = _resolve_b3_tail_controls(
        cfg_cont,
        r_ws=2.0,
        rmax=30.0,
        stage_mode="in_scf",
    )
    controls_full = _resolve_b3_tail_controls(
        cfg_full,
        r_ws=2.0,
        rmax=30.0,
        stage_mode="in_scf",
    )

    assert controls_cont["fit_window_mode"] == "local"
    assert controls_full["fit_window_mode"] == "auto"


def test_explicit_fit_window_mode_overrides_target_policy() -> None:
    cfg = FullExternalConfig(
        element="H",
        temperature_ev=2.0,
        rho_g_cc=1.0,
        b3_tail_target="cont",
        b3_tail_fit_window_mode="physical",
    )

    controls = _resolve_b3_tail_controls(
        cfg,
        r_ws=2.0,
        rmax=30.0,
        stage_mode="in_scf",
    )

    assert controls["fit_window_mode"] == "physical"
