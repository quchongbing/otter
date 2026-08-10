"""Atomic-commit regressions for full/external Appendix-B tail fits."""

from __future__ import annotations

import numpy as np
import pytest

from otter.electronic import ks_dft as ks_dft_module
from otter.electronic.ks_dft import _apply_paired_full_external_b3_tail
from otter.electronic.continuum import tail as tail_module


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    r = np.linspace(0.05, 12.0, 256)
    n0 = 0.02
    n_full = n0 + 0.3 * np.exp(-r)
    n_ext = n0 + 0.1 * np.exp(-r)
    params: dict[str, object] = {
        "tail_r_cut": 4.0,
        "tail_fit_points": 20,
        "tail_fit_window_mode": "local",
        "tail_blend_points": 0,
        "tail_model": "full",
    }
    return r, n_full, n_ext, params


def test_paired_tail_does_not_mutate_inputs_when_external_fit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful full fit must not leak out when the external fit fails."""
    r, n_full, n_ext, params = _inputs()
    full_before = n_full.copy()
    ext_before = n_ext.copy()
    calls = 0

    def _fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        density = np.asarray(args[1], dtype=float)
        if calls == 2:
            raise ValueError("synthetic external-fit failure")
        return density + 1.0, {"model_selected": "full"}

    monkeypatch.setattr(tail_module, "apply_tail_match", _fit)

    with pytest.raises(ValueError, match="external-fit failure"):
        _apply_paired_full_external_b3_tail(
            r,
            n_full,
            n_ext,
            n0=0.02,
            mu_id=0.1,
            temperature=0.05,
            params=params,
            source_electron_target_full=6.0,
            source_electron_target_ext=0.0,
            source_charge_target_full=6.0,
            source_charge_target_ext=0.0,
            charge_constrained=False,
        )

    assert calls == 2
    assert np.array_equal(n_full, full_before)
    assert np.array_equal(n_ext, ext_before)


def test_paired_tail_returns_both_candidates_after_both_fits_succeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The paired helper exposes candidates only after both fits succeed."""
    r, n_full, n_ext, params = _inputs()
    calls = 0

    def _fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        density = np.asarray(args[1], dtype=float)
        return density + float(calls), {"model_selected": "full"}

    monkeypatch.setattr(tail_module, "apply_tail_match", _fit)

    full_out, full_meta, ext_out, ext_meta = (
        _apply_paired_full_external_b3_tail(
            r,
            n_full,
            n_ext,
            n0=0.02,
            mu_id=0.1,
            temperature=0.05,
            params=params,
            source_electron_target_full=6.0,
            source_electron_target_ext=0.0,
            source_charge_target_full=6.0,
            source_charge_target_ext=0.0,
            charge_constrained=False,
        )
    )

    assert calls == 2
    assert np.allclose(full_out, n_full + 1.0)
    assert ext_out is not None
    assert np.allclose(ext_out, n_ext + 2.0)
    assert full_meta["paired_full_external_commit"] is True
    assert ext_meta is not None
    assert ext_meta["paired_full_external_commit"] is True


def test_charge_constrained_pair_does_not_expose_full_candidate_on_external_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact-charge variant has the same all-or-nothing semantics."""
    r, n_full, n_ext, params = _inputs()
    full_before = n_full.copy()
    ext_before = n_ext.copy()
    calls = 0

    def _fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        density = np.asarray(args[1], dtype=float)
        if calls == 2:
            raise ValueError("synthetic constrained external-fit failure")
        return density + 1.0, {"model_selected": "full"}

    monkeypatch.setattr(
        ks_dft_module,
        "_apply_charge_constrained_b3_tail",
        _fit,
    )

    with pytest.raises(ValueError, match="constrained external-fit failure"):
        _apply_paired_full_external_b3_tail(
            r,
            n_full,
            n_ext,
            n0=0.02,
            mu_id=0.1,
            temperature=0.05,
            params=params,
            source_electron_target_full=6.0,
            source_electron_target_ext=0.0,
            source_charge_target_full=6.0,
            source_charge_target_ext=0.0,
            charge_constrained=True,
        )

    assert calls == 2
    assert np.array_equal(n_full, full_before)
    assert np.array_equal(n_ext, ext_before)
