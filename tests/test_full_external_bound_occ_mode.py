"""Regression tests for the high-level physical bound occupation mode."""
from __future__ import annotations

import pytest

from otter.electronic.full_external import FullExternalConfig, _build_ks_config


def _ks_bound_occ_mode(cfg: FullExternalConfig) -> str:
    ks_cfg = _build_ks_config(
        cfg,
        z_nuc=6,
        temperature_ha=100.0 / 27.211386245988,
        n_i=0.01,
        r_ws=2.0,
        rmax=20.0,
        mu_guess=0.1,
        mu_bounds=(-5.0, 5.0),
        max_iter=4,
        cont_params={"tail_mode": "off"},
        compute_external=True,
    )
    return str(ks_cfg.bound_occ_mode)


def test_bound_occ_mode_defaults_to_physical_fd() -> None:
    cfg = FullExternalConfig(
        element="C",
        temperature_ev=100.0,
        rho_g_cc=2.0,
    )
    assert _ks_bound_occ_mode(cfg) == "fd"


def test_bound_occ_mode_can_use_legacy_fdm_diagnostic() -> None:
    cfg = FullExternalConfig(
        element="C",
        temperature_ev=100.0,
        rho_g_cc=2.0,
        bound_occ_mode="fd_m",
    )
    assert _ks_bound_occ_mode(cfg) == "fd_m"


def test_bound_occ_mode_rejects_invalid_value() -> None:
    with pytest.raises(
        ValueError,
        match=r"bound_occ_mode must be 'fd' or 'fd_m'",
    ):
        FullExternalConfig(
            element="C",
            temperature_ev=100.0,
            rho_g_cc=2.0,
            bound_occ_mode="bad",
        )
