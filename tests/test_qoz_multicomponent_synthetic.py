"""
tests/test_qoz_multicomponent_synthetic.py

Purpose
-------
Exercise the new multicomponent QOZ/HNC path on smooth analytic screening
clouds. The goal is to keep the regression fast while locking down the matrix
assembly and symmetry conventions.
"""
from __future__ import annotations

import numpy as np
import pytest

from otter.numerics.grids import create_linear_grid
from otter.ionic import (
    QOZPotentialOptions,
    QOZResponseOptions,
    build_effective_vij_from_nscr,
    build_effective_vii_from_nscr,
    dst_lattice_zero_moment,
    enforce_screening_charge_consistency_many,
    hnc_solver_multicomponent,
    hnc_solver_multicomponent_continuation,
    precompute_dst_lattice_transform_like,
)
from otter.ionic.qoz import (
    _stable_effective_pair_potential_k,
    decompose_effective_pair_potential_k,
)
from otter.numerics.transforms import radial_forward


def test_qoz_multicomponent_synthetic_pipeline() -> None:
    """Synthetic multicomponent screening clouds should yield finite symmetric outputs."""
    grid = create_linear_grid(rmax=32.0, N=256)
    r = np.asarray(grid.r, dtype=float)
    transform = precompute_dst_lattice_transform_like(r)
    k = np.asarray(transform.k, dtype=float)

    zbar = np.asarray([0.8, 1.2, 1.6], dtype=float)
    n_i = np.asarray([8.0e-4, 5.0e-4, 3.0e-4], dtype=float)
    decay = np.asarray([0.9, 0.6, 0.45], dtype=float)
    n_scr = np.zeros((3, r.size), dtype=float)
    for idx in range(3):
        n_scr[idx] = np.exp(-decay[idx] * r)

    charge_fix = enforce_screening_charge_consistency_many(
        r,
        n_scr,
        zbar=zbar,
        renormalize=True,
        transform=transform,
    )
    assert np.all(np.isfinite(charge_fix.n_scr_r))
    assert charge_fix.normalization_measure == "dst_zero_moment"
    assert np.allclose(charge_fix.q_scr_used, zbar, atol=1e-12, rtol=0.0)
    assert np.allclose(charge_fix.q_scr_dst_used, zbar, atol=1e-12, rtol=0.0)
    assert np.allclose(
        dst_lattice_zero_moment(charge_fix.n_scr_r, transform),
        zbar,
        atol=1e-12,
        rtol=0.0,
    )
    assert np.array_equal(charge_fix.q_scr_raw, charge_fix.q_scr_dst_raw)

    te_ha = 6.0
    qoz = build_effective_vij_from_nscr(
        r=transform.r,
        n_scr=charge_fix.n_scr_r,
        zbar=zbar,
        n_i=n_i,
        ion_temperature_ha=te_ha,
        k=k,
        transform=transform,
        options=QOZPotentialOptions(
            response=QOZResponseOptions(
                chi0_model="lindhard_fd",
                lfc_model="gregori2007",
                electron_temperature_ha=te_ha,
                lindhard_p_points=192,
            )
        ),
    )

    assert qoz.vij_r.shape == (3, 3, r.size)
    assert qoz.vij_k.shape == (3, 3, k.size)
    assert np.allclose(qoz.vij_r, np.swapaxes(qoz.vij_r, 0, 1), atol=1e-10, rtol=1e-10)
    assert np.allclose(qoz.vij_k, np.swapaxes(qoz.vij_k, 0, 1), atol=1e-10, rtol=1e-10)
    assert np.all(np.isfinite(qoz.vij_r))
    assert np.all(np.isfinite(qoz.vij_k))

    multi_single = build_effective_vij_from_nscr(
        r=transform.r,
        n_scr=charge_fix.n_scr_r[:1],
        zbar=zbar[:1],
        n_i=n_i[:1],
        ion_temperature_ha=te_ha,
        k=k,
        transform=transform,
        options=QOZPotentialOptions(
            response=QOZResponseOptions(
                chi0_model="lindhard_fd",
                lfc_model="gregori2007",
                electron_temperature_ha=te_ha,
                lindhard_p_points=192,
            )
        ),
    )
    single = build_effective_vii_from_nscr(
        r=transform.r,
        n_scr=charge_fix.n_scr_r[0],
        zbar=float(zbar[0]),
        n_i=float(n_i[0]),
        ion_temperature_ha=te_ha,
        k=k,
        transform=transform,
        options=QOZPotentialOptions(
            response=QOZResponseOptions(
                chi0_model="lindhard_fd",
                lfc_model="gregori2007",
                electron_temperature_ha=te_ha,
                lindhard_p_points=192,
            )
        ),
    )
    assert np.allclose(multi_single.vij_k[0, 0], single.vii_k, atol=1e-10, rtol=1e-10)
    assert np.allclose(multi_single.vij_r[0, 0], single.vii_r, atol=1e-10, rtol=1e-10)

    # Use a deliberately weak pair potential for the HNC smoke test. The
    # multicomponent HNC solver is exercised independently of the stronger
    # low-k behavior of the synthetic QOZ screening clouds above.
    v_hnc = np.zeros_like(qoz.vij_r)
    for i in range(3):
        for j in range(3):
            amp = 1.5e-3 if i == j else 1.0e-3
            v_hnc[i, j] = amp * np.exp(-0.45 * transform.r)

    g_r, s_k, h_r, c_r, res_hist = hnc_solver_multicomponent(
        transform.r,
        k,
        v_hnc,
        transform,
        n_i,
        te_ha,
        mix=0.12,
        tol=5.0e-5,
        max_iter=60,
        mixing_scheme="picard",
        tail_points=24,
    )

    assert g_r.shape == (3, 3, r.size)
    assert s_k.shape == (3, 3, k.size)
    assert h_r.shape == (3, 3, r.size)
    assert c_r.shape == (3, 3, r.size)
    assert len(res_hist) > 0
    assert np.all(np.isfinite(g_r))
    assert np.all(np.isfinite(s_k))
    assert np.allclose(g_r, np.swapaxes(g_r, 0, 1), atol=1e-10, rtol=1e-10)
    assert np.allclose(s_k, np.swapaxes(s_k, 0, 1), atol=1e-10, rtol=1e-10)
    tail_mean = np.mean(g_r[:, :, -24:], axis=2)
    assert np.max(np.abs(tail_mean - 1.0)) < 0.25
    for ik in range(k.size):
        eig = np.linalg.eigvalsh(s_k[:, :, ik])
        assert float(np.min(eig)) > 0.0

    g_c, s_c, h_c, c_c, res_c, stage_meta = hnc_solver_multicomponent_continuation(
        transform.r,
        k,
        v_hnc,
        transform,
        n_i,
        te_ha,
        potential_scales=(0.4, 0.7, 1.0),
        mix=0.12,
        tol=5.0e-5,
        max_iter=40,
        mixing_scheme="picard",
        tail_points=24,
    )

    assert g_c.shape == g_r.shape
    assert s_c.shape == s_k.shape
    assert h_c.shape == h_r.shape
    assert c_c.shape == c_r.shape
    assert len(res_c) > 0
    assert len(stage_meta) == 3
    for stage in stage_meta:
        assert float(stage["potential_scale"]) > 0.0
        assert float(stage["s0_min_eig"]) > 0.0

    # The production strong-coupling backend solves the unprojected equations.
    # Lock down both the raw positive-definite S matrix and the independent
    # closure identity, rather than accepting a small fixed-point residual alone.
    g_n, s_n, _h_n, _c_n, res_n = hnc_solver_multicomponent(
        transform.r,
        k,
        v_hnc,
        transform,
        n_i,
        te_ha,
        tol=5.0e-5,
        max_iter=20,
        mixing_scheme="newton_krylov",
        s_projection_mode="none",
        c_map_clip=0.0,
        enforce_h_tail_zero=False,
        tail_points=24,
    )
    assert float(res_n[-1]) < 5.0e-5
    raw_eig = np.linalg.eigvalsh(np.moveaxis(s_n, -1, 0))
    assert float(np.min(raw_eig)) > 0.0
    sqrt_n = np.sqrt(np.outer(n_i, n_i))
    s_from_g = (
        np.eye(n_i.size)[:, :, np.newaxis]
        + sqrt_n[:, :, np.newaxis]
        * radial_forward(np.asarray(g_n) - 1.0, transform)
    )
    assert float(np.max(np.abs(s_n - s_from_g))) < 1.0e-8


def test_strict_continuation_rejects_a_projected_false_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny reported residual cannot hide a broken OZ/closure identity."""
    import otter.ionic.qoz as qoz_module

    transform = precompute_dst_lattice_transform_like(
        create_linear_grid(rmax=12.0, N=64).r
    )
    n_i = np.asarray([1.0e-3, 2.0e-3], dtype=float)
    v_r = np.zeros((2, 2, transform.r.size), dtype=float)

    def _fake_projected_solver(*args, **kwargs):
        g_r = np.ones_like(v_r)
        h_r = np.zeros_like(v_r)
        c_r = np.zeros_like(v_r)
        s_k = 2.0 * np.eye(2)[:, :, np.newaxis] * np.ones(transform.k.size)
        return g_r, s_k, h_r, c_r, [1.0e-12]

    monkeypatch.setattr(qoz_module, "hnc_solver_multicomponent", _fake_projected_solver)
    with pytest.raises(RuntimeError, match="PSD projection is not a physical substitute"):
        qoz_module.hnc_solver_multicomponent_continuation(
            transform.r,
            transform.k,
            v_r,
            transform,
            n_i,
            1.0,
            potential_scales=(1.0,),
            adaptive=False,
            require_converged=True,
            s_projection_mode="clip",
        )


def test_stable_pair_potential_matches_legacy_algebra_on_regular_modes() -> None:
    """The stable rearrangement must remain algebraically identical away from k=0."""
    transform = precompute_dst_lattice_transform_like(
        create_linear_grid(rmax=24.0, N=192).r
    )
    r = transform.r
    k = transform.k
    zbar = np.asarray([0.9, 1.4], dtype=float)
    n_i = np.asarray([7.0e-4, 4.0e-4], dtype=float)
    n_scr = np.asarray(
        [np.exp(-0.8 * r), np.exp(-0.5 * r)],
        dtype=float,
    )
    charge_fix = enforce_screening_charge_consistency_many(
        r,
        n_scr,
        zbar=zbar,
        renormalize=True,
        transform=transform,
    )
    options = QOZPotentialOptions(
        response=QOZResponseOptions(
            chi0_model="lindhard_fd",
            lfc_model="gregori2007",
            electron_temperature_ha=5.0,
            lindhard_p_points=192,
        ),
        high_k_taper_start_frac=None,
    )
    qoz = build_effective_vij_from_nscr(
        r=r,
        n_scr=charge_fix.n_scr_r,
        zbar=zbar,
        n_i=n_i,
        ion_temperature_ha=5.0,
        k=k,
        transform=transform,
        options=options,
    )

    coulomb_k = 4.0 * np.pi / k**2
    legacy = (
        zbar[:, np.newaxis, np.newaxis]
        * zbar[np.newaxis, :, np.newaxis]
        * coulomb_k
        + qoz.n_scr_k[:, np.newaxis, :]
        * qoz.n_scr_k[np.newaxis, :, :]
        / qoz.chi_ee_k
    )
    regular = (k >= 0.5) & (k <= 5.0)
    assert np.any(regular)
    assert np.allclose(
        qoz.vij_k[:, :, regular],
        legacy[:, :, regular],
        atol=2.0e-11,
        rtol=2.0e-12,
    )


def test_stable_pair_potential_avoids_ultralow_k_cancellation() -> None:
    """The rearranged formula should retain terms lost by direct 1/k^2 subtraction."""
    k = np.asarray([3.0e-8], dtype=float)
    zbar = np.asarray([3.0, 1.0], dtype=float)
    # One representable ULP below each charge creates a very small but exactly
    # known charge residual.  At this k the direct Coulomb terms are O(1e16),
    # while their physically relevant remainder is O(1).
    n_scr_k = np.nextafter(zbar, 0.0)[:, np.newaxis]
    chi0_k = np.asarray([-0.25], dtype=float)
    gee_k = 0.05 * k**2

    stable = _stable_effective_pair_potential_k(
        n_scr_k=n_scr_k,
        zbar=zbar,
        k=k,
        chi0_k=chi0_k,
        gee_k=gee_k,
    )[:, :, 0]
    components = decompose_effective_pair_potential_k(
        n_scr_k=n_scr_k,
        zbar=zbar,
        k=k,
        chi0_k=chi0_k,
        gee_k=gee_k,
    )
    assert all(component.shape == (2, 2, 1) for component in components)
    assert all(np.all(np.isfinite(component)) for component in components)
    assert np.array_equal(sum(components)[:, :, 0], stable)

    coulomb = 4.0 * np.pi / float(k[0] ** 2)
    inverse_chi = 1.0 / float(chi0_k[0]) - coulomb * (1.0 - float(gee_k[0]))
    legacy = (
        coulomb * np.outer(zbar, zbar)
        + np.outer(n_scr_k[:, 0], n_scr_k[:, 0]) * inverse_chi
    )

    # Use extended precision only to form an independent reference for this
    # cancellation test; the production routine itself remains float64.
    ld = np.longdouble
    z_ld = zbar.astype(ld)
    q_ld = n_scr_k[:, 0].astype(ld)
    k_ld = ld(k[0])
    d_ld = q_ld - z_ld
    gee_ld = ld(0.05) * k_ld**2
    reference = (
        ld(4.0)
        * ld(np.pi)
        / k_ld**2
        * (
            -z_ld[:, None] * d_ld[None, :]
            - z_ld[None, :] * d_ld[:, None]
            - d_ld[:, None] * d_ld[None, :]
            + gee_ld * q_ld[:, None] * q_ld[None, :]
        )
        + q_ld[:, None] * q_ld[None, :] / ld(chi0_k[0])
    ).astype(float)

    stable_error = float(np.max(np.abs(stable - reference)))
    legacy_error = float(np.max(np.abs(legacy - reference)))
    assert stable_error < 1.0e-10
    assert legacy_error > 1.0
    assert stable_error < legacy_error * 1.0e-8

if __name__ == "__main__":
    test_qoz_multicomponent_synthetic_pipeline()
    test_stable_pair_potential_matches_legacy_algebra_on_regular_modes()
    test_stable_pair_potential_avoids_ultralow_k_cancellation()
    print("test_qoz_multicomponent_synthetic: ok")
