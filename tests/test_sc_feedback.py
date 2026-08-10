from __future__ import annotations

import numpy as np
import pytest

from otter.numerics.constants import EV_TO_HA
from otter.workflows import PlasmaWorkflowConfig, solve_plasma_workflow
import otter.experimental.sc_feedback as sc_feedback_module
from otter.experimental.sc_feedback import (
    SCFeedbackConfig,
    estimate_mixture_correlation_potentials,
    mixture_ionic_background_profiles,
    solve_sc_feedback_workflow,
)
from otter.numerics.transforms import (
    precompute_dst_lattice_transform_like,
    radial_forward,
    radial_inverse,
)


def test_sc_feedback_fails_closed_by_default() -> None:
    assert SCFeedbackConfig().require_converged is True


def test_mixture_ionic_background_uses_every_pair_channel() -> None:
    gij = np.asarray(
        [
            [[0.2, 0.8, 1.0], [0.4, 0.9, 1.0]],
            [[0.4, 0.9, 1.0], [0.6, 0.95, 1.0]],
        ],
        dtype=float,
    )
    weights = np.asarray([0.25, 0.75], dtype=float)

    actual = mixture_ionic_background_profiles(gij, weights)

    expected = np.asarray(
        [
            0.25 * gij[0, 0] + 0.75 * gij[0, 1],
            0.25 * gij[1, 0] + 0.75 * gij[1, 1],
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(actual[:, -1], 1.0, rtol=0.0, atol=1.0e-14)


def test_one_component_correlation_potential_matches_starrett_eq19_limit() -> None:
    r = 0.08 * np.arange(1, 129, dtype=float)
    transform = precompute_dst_lattice_transform_like(r, n_grid=r.size + 1)
    k = np.asarray(transform.k, dtype=float)
    g = 1.0 - 0.85 * np.exp(-((r / 1.3) ** 2))
    gij = g[np.newaxis, np.newaxis, :]
    n_scr_r = 0.12 * np.exp(-((r / 0.9) ** 2))
    n_scr_k = radial_forward(n_scr_r, transform)
    chi_ee_k = -0.4 / (1.0 + 0.15 * k**2)
    zbar = 2.8
    n_i = 0.025
    n0 = 0.061
    te_ev = 12.0

    actual = estimate_mixture_correlation_potentials(
        r=r,
        k=k,
        gij_r=gij,
        n_scr_k=n_scr_k[np.newaxis, :],
        chi_ee_k=chi_ee_k,
        zbar=np.asarray([zbar]),
        partial_ion_density=np.asarray([n_i]),
        field_free_electron_density=n0,
        electron_temperature_ev=te_ev,
    )[0]

    beta = 1.0 / (te_ev * EV_TO_HA)
    c_ie_k = -beta * n_scr_k / chi_ee_k
    c_tilde_bar_k = c_ie_k - (4.0 * np.pi * beta / np.maximum(k**2, 1.0e-24)) * zbar
    zstar = n0 / n_i
    c_tilde_k = (zstar / zbar) * c_tilde_bar_k
    h_k = radial_forward(g - 1.0, transform)
    expected = radial_inverse(-(n_i / beta) * c_tilde_k * h_k, transform)
    expected = expected - expected[-1]

    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-13)


@pytest.mark.parametrize("electronic_model", ["qm", "tf"])
def test_mixture_sc_feedback_keeps_is_mu_and_uses_full_gij_background(
    monkeypatch,
    electronic_model: str,
) -> None:
    r = 0.12 * np.arange(1, 33, dtype=float)
    transform = precompute_dst_lattice_transform_like(r, n_grid=r.size + 1)
    k = np.asarray(transform.k, dtype=float)
    g00 = 1.0 - 0.7 * np.exp(-((r / 0.8) ** 2))
    g01 = 1.0 - 0.4 * np.exp(-((r / 1.0) ** 2))
    g11 = 1.0 - 0.2 * np.exp(-((r / 1.2) ** 2))
    gij = np.asarray([[g00, g01], [g01, g11]], dtype=float)
    weights = np.asarray([0.3, 0.7], dtype=float)

    species = []
    for symbol, volume in (("C", 12.0), ("H", 28.0)):
        species.append(
            {
                "element": symbol,
                "Z": 6 if symbol == "C" else 1,
                "count": 1.0,
                "x": 0.5,
                "volume_bohr3": volume,
                "r_ws_bohr": float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)),
                "mu_ha": 0.4,
                "result": {
                    "mu": 0.4,
                    "n0": 0.05,
                    "r": r,
                    "v_full": np.zeros_like(r),
                    "v_ext": np.zeros_like(r),
                    "zbar": 3.0 if symbol == "C" else 0.8,
                    "meta": {},
                },
            }
        )
    ion = {
        "kind": "multicomponent",
        "species": ["C", "H"],
        "r": r,
        "k": k,
        "gij_r": gij,
        "sij_k": np.zeros_like(gij),
        "n_scr_k": np.asarray(
            [
                radial_forward(0.08 * np.exp(-((r / 0.8) ** 2)), transform),
                radial_forward(0.03 * np.exp(-((r / 1.0) ** 2)), transform),
            ]
        ),
        "chi_ee_k": -0.3 / (1.0 + k**2),
        "zbar": np.asarray([3.0, 0.8]),
        "n_i": np.asarray([0.015, 0.035]),
    }
    is_workflow = {
        "species_symbols": ["C", "H"],
        "species_counts": [1.0, 1.0],
        "electronic": {
            "kind": "mixture",
            "result": {
                "mu_common_ha": 0.4,
                "volume_weights": weights,
                "species": species,
                "meta": {"root_success": True},
            },
        },
        "ion": ion,
    }
    workflow_cfg = PlasmaWorkflowConfig(
        elements=["C", "H"],
        counts=[1.0, 1.0],
        temperature_ev=10.0,
        ion_temperature_ev=10.0,
        rho_g_cc=3.0,
        electronic_model=electronic_model,
        aa_overrides={"bound_occ_mode": "fd"},
    )

    seen_configs = []

    def fake_solve(cfg):
        seen_configs.append(cfg)
        return {
            "mu": float(cfg.full_fixed_mu_ha),
            "n0": 0.05,
            "r": r,
            "v_full": np.zeros_like(r),
            "v_ext": np.zeros_like(r),
            "zbar": 3.0 if str(cfg.element) == "C" else 0.8,
            "meta": {},
        }

    def fake_continue(cfg, *, electronic_kind, electronic_result):
        return {
            **is_workflow,
            "electronic": {"kind": electronic_kind, "result": electronic_result},
            "ion": dict(ion),
        }

    monkeypatch.setattr(sc_feedback_module, "solve_full_then_external", fake_solve)
    monkeypatch.setattr(
        sc_feedback_module,
        "continue_plasma_workflow_from_electronic_result",
        fake_continue,
    )

    result = solve_sc_feedback_workflow(
        workflow_cfg,
        is_workflow,
        feedback_cfg=SCFeedbackConfig(
            max_outer=1,
            g_tol=1.0e-12,
            v_corr_tol=1.0e12,
            v_corr_mix=0.35,
            use_continuation=False,
        ),
    )

    expected_background = mixture_ionic_background_profiles(gij, weights)
    assert result["structure_model"] == "SC"
    assert result["sc_feedback"]["converged"] is True
    assert len(seen_configs) == 2
    for idx, cfg in enumerate(seen_configs):
        assert cfg.bound_occ_mode == "fd"
        assert cfg.full_fixed_mu_ha == 0.4
        assert cfg.n0_mode_override == "ideal"
        assert cfg.electronic_model == electronic_model
        np.testing.assert_allclose(cfg.g_ii_override, expected_background[idx])
        assert cfg.v_corr_full is not None
        assert cfg.v_corr_ext is not None
    assert result["sc_feedback"]["electronic_model"] == electronic_model
    assert "doi:10.1016/j.hedp.2013.12.001" in result["sc_feedback"]["reference"]
    assert result["sc_feedback"]["v_corr_r_bohr"] is not None
    assert result["sc_feedback"]["v_corr_species_ha"] is not None
    for idx, entry in enumerate(result["electronic"]["result"]["species"]):
        np.testing.assert_allclose(
            entry["result"]["g_ii_background"],
            expected_background[idx],
        )


def test_single_component_tf_sc_outer_iteration_uses_tf_backend() -> None:
    """Integration guard against silently falling back from TF-SC to QM."""
    cfg = PlasmaWorkflowConfig(
        formula="Al",
        temperature_ev=15.0,
        ion_temperature_ev=15.0,
        rho_g_cc=8.1,
        electronic_model="tf",
        aa_overrides={"n_points": 128},
        qoz_linear_n_points=128,
        hnc_max_iter=300,
    )
    is_result = solve_plasma_workflow(cfg)
    sc_result = solve_sc_feedback_workflow(
        cfg,
        is_result,
        feedback_cfg=SCFeedbackConfig(
            max_outer=1,
            require_converged=False,
        ),
    )

    electronic = sc_result["electronic"]["result"]
    assert electronic["meta"]["electronic_model"] == "thomas_fermi"
    assert electronic["meta"]["fixed_is_mu_used"] is True
    assert electronic["meta"]["analytic_ion_sphere_background"] is False
    assert electronic["mu"] == is_result["electronic"]["result"]["mu"]
    assert np.max(np.abs(electronic["v_corr_full"])) > 0.0
    assert sc_result["sc_feedback"]["electronic_model"] == "tf"
