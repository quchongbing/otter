"""
tests/test_qoz_screening_charge_consistency.py

Purpose
-------
Unit tests for the reduced-QOZ screening-charge normalization helper.

The QOZ mapping is sensitive to the low-k content of `n_scr(k)`. These tests
keep the correction logic lightweight and verify that enforcing
`4*pi*int r^2 n_scr dr = Zbar` behaves as expected.
"""
from __future__ import annotations

import numpy as np
import pytest

from otter.ionic import (
    QOZPotentialOptions,
    QOZResponseOptions,
    build_effective_vii_from_nscr,
    dst_lattice_zero_moment,
    enforce_screening_charge_consistency,
    precompute_dst_lattice_transform_like,
    qoz_zbar_from_nscr,
    radial_charge_trapezoid,
)


def _gaussian_nscr(r: np.ndarray, *, zbar: float, width: float) -> np.ndarray:
    """Return a smooth positive screening cloud normalized to `zbar`."""
    n_scr = np.exp(-(r / float(width)) ** 2)
    q_scr = 4.0 * np.pi * np.trapezoid((r**2) * n_scr, r)
    return n_scr * (float(zbar) / max(float(q_scr), 1.0e-14))


def test_screening_charge_consistency_enforces_zbar() -> None:
    """The helper should rescale the integrated screening charge to `Zbar`."""
    r = np.linspace(1.0e-4, 12.0, 2400)
    zbar = 4.0
    n_scr_ref = _gaussian_nscr(r, zbar=zbar, width=2.0)
    n_scr_bad = 0.85 * n_scr_ref

    fix = enforce_screening_charge_consistency(r, n_scr_bad, zbar=zbar, renormalize=True)
    q_scr_used = 4.0 * np.pi * np.trapezoid((r**2) * np.asarray(fix.n_scr_r, dtype=float), r)

    assert abs(float(fix.q_scr_raw) - 0.85 * zbar) < 5.0e-4
    assert abs(float(fix.q_scr_used) - zbar) < 5.0e-10
    assert abs(float(q_scr_used) - zbar) < 5.0e-10
    assert float(fix.q_scr_rel) > 0.1


def test_dst_charge_consistency_uses_the_transform_zero_moment() -> None:
    """Production normalization must close the zero mode of the actual DST."""
    # A deliberately coarse, compact cloud makes the endpoint weighting
    # difference large enough that this test would fail clearly if the legacy
    # trapezoidal normalization were accidentally restored.
    transform = precompute_dst_lattice_transform_like(
        np.linspace(1.0e-4, 12.0, 256)
    )
    r = transform.r
    zbar = 4.0
    n_scr_ref = _gaussian_nscr(r, zbar=zbar, width=0.7)
    n_scr_bad = 0.85 * n_scr_ref

    legacy = enforce_screening_charge_consistency(
        r,
        n_scr_bad,
        zbar=zbar,
        renormalize=True,
    )
    legacy_dst_charge = float(
        dst_lattice_zero_moment(legacy.n_scr_r, transform)
    )
    assert abs(legacy_dst_charge - zbar) > 1.0e-4

    fixed = enforce_screening_charge_consistency(
        r,
        n_scr_bad,
        zbar=zbar,
        renormalize=True,
        transform=transform,
    )
    fixed_dst_charge = float(
        dst_lattice_zero_moment(fixed.n_scr_r, transform)
    )
    fixed_trap_charge = float(radial_charge_trapezoid(r, fixed.n_scr_r))

    assert fixed.normalization_measure == "dst_zero_moment"
    assert abs(fixed_dst_charge - zbar) < 1.0e-12
    assert abs(float(fixed.q_scr_used) - zbar) < 1.0e-12
    assert abs(float(fixed.q_scr_dst_used) - zbar) < 1.0e-12
    assert float(fixed.q_scr_raw) == float(fixed.q_scr_dst_raw)
    assert np.isclose(
        float(fixed.scale_factor),
        zbar / float(fixed.q_scr_dst_raw),
        atol=1.0e-15,
        rtol=1.0e-15,
    )
    assert np.isclose(
        float(fixed.q_scr_trapz_used),
        fixed_trap_charge,
        atol=1.0e-13,
        rtol=1.0e-13,
    )


def test_charge_closure_rejects_an_unnormalizable_cloud() -> None:
    r = np.linspace(1.0e-4, 8.0, 128)
    with pytest.raises(ValueError, match="Cannot normalize n_scr"):
        enforce_screening_charge_consistency(
            r,
            np.zeros_like(r),
            zbar=1.0,
            renormalize=True,
        )


def test_qoz_zbar_distinguishes_partition_raw_integral_and_legacy_aa() -> None:
    """Production PA partition, raw closure audit, and legacy WS are distinct."""
    r = np.linspace(1.0e-4, 18.0, 3000)
    screening_charge = np.asarray([3.25, 0.87], dtype=float)
    n_scr = np.asarray(
        [
            _gaussian_nscr(r, zbar=screening_charge[0], width=2.0),
            _gaussian_nscr(r, zbar=screening_charge[1], width=1.1),
        ],
        dtype=float,
    )
    electronic_zbar = np.asarray([4.0, 1.0], dtype=float)
    partition_zbar = np.asarray([3.4, 1.0], dtype=float)

    production = np.asarray(
        qoz_zbar_from_nscr(
            r,
            n_scr,
            partition_zbar=partition_zbar,
            electronic_zbar=electronic_zbar,
        ),
        dtype=float,
    )
    raw_integral = np.asarray(
        qoz_zbar_from_nscr(
            r,
            n_scr,
            electronic_zbar=electronic_zbar,
            mode="screening_integral",
        ),
        dtype=float,
    )
    legacy = np.asarray(
        qoz_zbar_from_nscr(
            r,
            n_scr,
            electronic_zbar=electronic_zbar,
            mode="electronic",
        ),
        dtype=float,
    )

    assert np.array_equal(production, partition_zbar)
    assert np.allclose(raw_integral, screening_charge, atol=5.0e-12, rtol=0.0)
    assert np.array_equal(legacy, electronic_zbar)
    assert not np.allclose(production, raw_integral, atol=1.0e-3, rtol=1.0e-3)
    assert not np.allclose(raw_integral, legacy, atol=1.0e-3, rtol=1.0e-3)


def test_screening_charge_consistency_recovers_low_k_vii() -> None:
    """
    Charge renormalization should recover the reference low-k potential.

    A pure amplitude mismatch in `n_scr(r)` should mainly pollute the low-k
    part of `V_ii(k)`. After enforcing `Q_scr = Zbar`, the corrected profile
    should return to the reference result.
    """
    r_like = np.linspace(1.0e-4, 16.0, 3200)
    zbar = 3.0
    n_i = 3.0 / (4.0 * np.pi * 2.5**3)
    t_ha = 10.0 / 27.211386245988
    transform = precompute_dst_lattice_transform_like(r_like)
    r = transform.r
    k = transform.k
    n_scr_ref = enforce_screening_charge_consistency(
        r,
        _gaussian_nscr(r, zbar=zbar, width=2.6),
        zbar=zbar,
        renormalize=True,
        transform=transform,
    ).n_scr_r
    n_scr_bad = 0.9 * n_scr_ref
    opts = QOZPotentialOptions(
        response=QOZResponseOptions(
            chi0_model="lindhard_fd",
            lfc_model="gregori2007",
            electron_temperature_ha=t_ha,
            lindhard_p_points=2**10,
            lindhard_p_max_mult=8.0,
            lindhard_p_max_extra=20.0,
        )
    )

    ref = build_effective_vii_from_nscr(
        r=r,
        n_scr=n_scr_ref,
        zbar=zbar,
        n_i=n_i,
        ion_temperature_ha=t_ha,
        k=k,
        transform=transform,
        options=opts,
    )
    raw = build_effective_vii_from_nscr(
        r=r,
        n_scr=n_scr_bad,
        zbar=zbar,
        n_i=n_i,
        ion_temperature_ha=t_ha,
        k=k,
        transform=transform,
        options=opts,
    )
    fixed_profile = enforce_screening_charge_consistency(
        r,
        n_scr_bad,
        zbar=zbar,
        renormalize=True,
        transform=transform,
    )
    fixed = build_effective_vii_from_nscr(
        r=r,
        n_scr=np.asarray(fixed_profile.n_scr_r, dtype=float),
        zbar=zbar,
        n_i=n_i,
        ion_temperature_ha=t_ha,
        k=k,
        transform=transform,
        options=opts,
    )

    raw_low_k_err = abs(float(raw.vii_k[0]) - float(ref.vii_k[0]))
    fixed_low_k_err = abs(float(fixed.vii_k[0]) - float(ref.vii_k[0]))

    assert fixed_low_k_err < 1.0e-10
    assert raw_low_k_err > fixed_low_k_err * 1.0e3


if __name__ == "__main__":
    test_screening_charge_consistency_enforces_zbar()
    test_dst_charge_consistency_uses_the_transform_zero_moment()
    test_qoz_zbar_distinguishes_partition_raw_integral_and_legacy_aa()
    test_screening_charge_consistency_recovers_low_k_vii()
    print("test_qoz_screening_charge_consistency: ok")
