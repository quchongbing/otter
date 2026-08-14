"""Fast contract tests for portable q/f/g/S state files."""

from __future__ import annotations

import json

import numpy as np
import pytest

from otter.io.state import (
    STATE_SCHEMA_VERSION,
    StateExportOptions,
    build_state_arrays,
    load_plasma_state,
    save_plasma_state,
    validate_state_arrays,
)
from otter.numerics.transforms import (
    precompute_dst_lattice_transform_like,
    radial_forward,
)
from otter.workflows import PlasmaWorkflowConfig


def _synthetic_workflow(n_species: int = 1) -> dict:
    transform = precompute_dst_lattice_transform_like(np.linspace(1.0e-4, 32.0, 257))
    r = np.asarray(transform.r)
    k = np.asarray(transform.k)
    symbols = ["C"] if n_species == 1 else ["C", "H"]
    counts = [1.0] if n_species == 1 else [1.0, 2.0]

    entries = []
    n_ion = np.empty((n_species, r.size))
    n_scr = np.empty((n_species, r.size))
    for idx, symbol in enumerate(symbols):
        n_ion[idx] = (0.3 + 0.1 * idx) * np.exp(-(0.8 + idx) * r)
        n_scr[idx] = (0.2 + 0.05 * idx) * np.exp(-(0.5 + idx) * r)
        result = {
            "element": symbol,
            "Z": 6 if symbol == "C" else 1,
            "r": r,
            "r_ws": 2.0 - 0.2 * idx,
            "mu": 0.2,
            "n_i": 0.05 + 0.01 * idx,
            "n0": 0.1 + 0.02 * idx,
            "n_full": 0.1 + n_ion[idx] + n_scr[idx],
            "n_bound": n_ion[idx],
            "n_cont": 0.1 + n_scr[idx],
            "n_free": 0.1 + n_scr[idx],
            "n_ext": np.full_like(r, 0.1),
            "n_pa": n_ion[idx] + n_scr[idx],
            "n_ion": n_ion[idx],
            "n_scr": n_scr[idx],
            "v_full": -np.exp(-r) / r,
            "v_scf": -np.exp(-r) / r,
            "v_ext": 0.05 * np.exp(-r),
            "v_nuc": -(6 if symbol == "C" else 1) / r,
            "v_H": np.exp(-r) / r,
            "v_xc": -0.01 * np.exp(-r),
            "zbar": 2.0 if symbol == "C" else 1.0,
            "zbar_partition": 1.9 if symbol == "C" else 0.9,
            "zbar_ws": 2.0 if symbol == "C" else 1.0,
            "bound_energy_cut_ha": 0.0,
            "bound_l_list": np.asarray([0, 1]),
            "bound_energy_ha": np.asarray([[-1.0, np.inf], [-0.2, np.inf]]),
            "bound_fd": np.asarray([[0.9, 0.0], [0.6, 0.0]]),
            "bound_m": np.asarray([[1.0, 0.0], [0.8, 0.0]]),
            "bound_fdm": np.asarray([[0.9, 0.0], [0.48, 0.0]]),
            "bound_occ_deg_fd": np.asarray([[1.8, 0.0], [3.6, 0.0]]),
            "bound_occ_deg_fdm": np.asarray([[1.8, 0.0], [2.88, 0.0]]),
            "bound_q_ion_ws": np.asarray([[1.7, np.nan], [2.5, np.nan]]),
            "bound_orbital_density_r": np.asarray(
                [
                    [0.6 * n_ion[idx], np.zeros_like(r)],
                    [0.4 * n_ion[idx], np.zeros_like(r)],
                ]
            ),
            "ion_orbital_density_r": np.asarray(
                [
                    [0.7 * n_ion[idx], np.zeros_like(r)],
                    [0.3 * n_ion[idx], np.zeros_like(r)],
                ]
            ),
        }
        entries.append(
            {
                "element": symbol,
                "Z": result["Z"],
                "count": counts[idx],
                "x": counts[idx] / sum(counts),
                "volume_bohr3": 20.0 - idx,
                "r_ws_bohr": result["r_ws"],
                "mu_ha": result["mu"],
                "result": result,
            }
        )

    q_k = radial_forward(n_scr, transform)
    temperature_ha = 10.0 / 27.211386245988
    chi_ee_k = -np.ones_like(k)
    v_ie_k = q_k / chi_ee_k
    c_ie_k = -v_ie_k / temperature_ha
    v_ee_k = 4.0 * np.pi / k**2
    c_ee_k = -v_ee_k / temperature_ha
    gij = np.ones((n_species, n_species, r.size))
    hij = np.zeros_like(gij)
    cij = np.zeros_like(gij)
    vij_r = np.ones_like(gij)
    vij_k = np.ones((n_species, n_species, k.size))
    sij = np.zeros((n_species, n_species, k.size))
    for i in range(n_species):
        sij[i, i] = 1.0

    if n_species == 1:
        electronic = {
            "kind": "single_species",
            "result": entries[0]["result"],
        }
        n_scr_r_out = n_scr[0]
        n_scr_k_out = q_k[0]
        gij_out = gij[0, 0]
        sij_out = sij[0, 0]
        zbar_out = 2.0
        n_i_out = 0.05
        v_ie_k_out = v_ie_k[0]
        c_ie_k_out = c_ie_k[0]
    else:
        electronic = {
            "kind": "mixture",
            "result": {"species": entries},
        }
        n_scr_r_out = n_scr
        n_scr_k_out = q_k
        gij_out = gij
        sij_out = sij
        zbar_out = np.asarray([2.0, 1.0])
        n_i_out = np.asarray([0.02, 0.04])
        v_ie_k_out = v_ie_k
        c_ie_k_out = c_ie_k

    return {
        "configuration": {
            "temperature_ev": 10.0,
            "rho_g_cc": 1.0,
            "qoz_response_lfc_model": "chabrier1990",
        },
        "citation_keys": ["StarrettSaumon2014", "Chabrier1990"],
        "formula": None,
        "species_symbols": symbols,
        "species_counts": counts,
        "temperature_ev": 10.0,
        "ion_temperature_ev": 10.0,
        "rho_g_cc": 1.0,
        "electronic": electronic,
        "ion": {
            "kind": "one_component" if n_species == 1 else "multicomponent",
            "sij_convention": "ashcroft_langreth",
            "r": r,
            "k": k,
            "n_scr_r": n_scr_r_out,
            "n_scr_k": n_scr_k_out,
            "v_ie_k": v_ie_k_out,
            "v_ei_k": v_ie_k_out,
            "v_ee_k": v_ee_k,
            "c_ie_k": c_ie_k_out,
            "c_ee_k": c_ee_k,
            "chi0_k": -2.0 * np.ones_like(k),
            "chi_ee_k": chi_ee_k,
            "gee_k": 0.2 * np.ones_like(k),
            "gij_r": gij_out,
            "sij_k": sij_out,
            "hij_r": hij[0, 0] if n_species == 1 else hij,
            "cij_r": cij[0, 0] if n_species == 1 else cij,
            "vij_r": vij_r[0, 0] if n_species == 1 else vij_r,
            "vij_k": vij_k[0, 0] if n_species == 1 else vij_k,
            "zbar": zbar_out,
            "n_i": n_i_out,
            "hnc_converged": True,
            "hnc_best_residual": 1.0e-7,
            "closure_transform_max_abs": 2.0e-8,
            "qoz_zbar_mode": "pseudoatom_partition",
            "qoz_response_chi0_model": "lindhard_fd",
            "qoz_response_lfc_model": "chabrier1990",
            "charge_fix": {"q_scr_rel": [1.0e-8] * n_species},
            "residual_history": [1.0e-3, 1.0e-7],
        },
    }


@pytest.mark.parametrize("n_species", [1, 2])
def test_state_arrays_preserve_q_f_g_s_contract(n_species: int) -> None:
    arrays = build_state_arrays(_synthetic_workflow(n_species))
    assert arrays["schema_version"].item() == STATE_SCHEMA_VERSION
    assert arrays["q_k"].shape[0] == n_species
    assert arrays["f_k"].shape[0] == n_species
    assert arrays["gij_r"].shape[:2] == (n_species, n_species)
    assert arrays["sij_k"].shape[:2] == (n_species, n_species)
    assert arrays["v_ie_k"].shape[0] == n_species
    np.testing.assert_array_equal(arrays["v_ie_k"], arrays["v_ei_k"])
    assert arrays["c_ie_k"].shape[0] == n_species
    assert arrays["v_ee_k"].ndim == 1
    assert arrays["c_ee_k"].ndim == 1
    assert arrays["gee_k"].shape == arrays["k_bohr_inv"].shape
    np.testing.assert_array_equal(arrays["gee_k"], arrays["g_ee_k"])
    assert arrays["chi0_k"].shape == arrays["k_bohr_inv"].shape
    assert arrays["chi_ee_k"].shape == arrays["k_bohr_inv"].shape
    assert arrays["vij_r"].shape[:2] == (n_species, n_species)
    assert arrays["vij_k"].shape[:2] == (n_species, n_species)
    assert arrays["species_0_bound_energy_ha"].shape == (2,)
    np.testing.assert_array_equal(
        arrays["species_0_bound_principal_n"], [1, 2]
    )
    assert arrays["species_0_bound_orbital_density_r"].shape[0] == 2
    assert arrays["species_0_ion_orbital_density_r"].shape[0] == 2
    assert arrays["mu_ha"].shape == (n_species,)
    assert arrays["zstar"].shape == (n_species,)
    np.testing.assert_array_equal(arrays["q_k"], arrays["n_scr_k"])
    np.testing.assert_array_equal(arrays["f_k"], arrays["n_ion_k"])
    assert float(np.max(arrays["r_bohr"])) < 20.0
    assert float(np.max(arrays["k_bohr_inv"])) < 20.0
    metadata = json.loads(str(arrays["metadata_json"].item()))
    assert metadata["model"]["structure_model"] == "IS"
    assert metadata["model"]["lfc_model"] == "chabrier1990"
    assert metadata["units"]["q_k"] == "electron number"
    assert metadata["units"]["v_ie_k"] == "Hartree Bohr^3"
    assert metadata["units"]["c_ee_k"] == "Bohr^3"
    assert metadata["configuration"]["rho_g_cc"] == 1.0
    assert metadata["citation_keys"] == [
        "StarrettSaumon2014",
        "Chabrier1990",
    ]
    assert metadata["convergence"]["electronic"][0]["species"] == "C"


def test_state_file_round_trips_without_pickle(tmp_path) -> None:
    path = save_plasma_state(
        tmp_path / "state.npz",
        _synthetic_workflow(2),
    )
    loaded = load_plasma_state(path)
    with np.load(path, allow_pickle=False) as raw:
        assert all(raw[key].dtype.kind != "O" for key in raw.files)
    np.testing.assert_allclose(loaded["q_k"], loaded["n_scr_k"])
    np.testing.assert_allclose(loaded["f_k"], loaded["n_ion_k"])


def test_single_species_state_recovers_ion_density_from_rws() -> None:
    workflow = _synthetic_workflow(1)
    result = workflow["electronic"]["result"]
    result.pop("n_i")
    arrays = build_state_arrays(workflow)
    expected = 3.0 / (4.0 * np.pi * float(result["r_ws"]) ** 3)
    np.testing.assert_allclose(arrays["n_i_bohr3"], [expected])


def test_state_validator_accepts_legacy_v1_without_interaction_channels() -> None:
    arrays = build_state_arrays(_synthetic_workflow(1))
    legacy = {
        key: value
        for key, value in arrays.items()
        if key not in {"v_ie_k", "v_ei_k", "v_ee_k", "c_ie_k", "c_ee_k"}
    }
    legacy["schema_version"] = np.asarray("otter_state_v1")
    metadata = json.loads(str(legacy["metadata_json"].item()))
    metadata["schema_version"] = "otter_state_v1"
    legacy["metadata_json"] = np.asarray(json.dumps(metadata))
    validate_state_arrays(legacy)


def test_state_save_adds_npz_suffix_and_leaves_no_temporary_file(tmp_path) -> None:
    path = save_plasma_state(tmp_path / "state", _synthetic_workflow(1))
    assert path == tmp_path / "state.npz"
    assert path.is_file()
    assert not list(tmp_path.glob(".state.npz.*.tmp"))


def test_state_export_respects_custom_exclusive_window() -> None:
    arrays = build_state_arrays(
        _synthetic_workflow(1),
        options=StateExportOptions(
            r_max_bohr=8.0,
            k_max_bohr_inv=5.0,
        ),
    )
    assert float(np.max(arrays["r_bohr"])) < 8.0
    assert float(np.max(arrays["k_bohr_inv"])) < 5.0


def test_state_validation_uses_metadata_window_and_rejects_nonfinite_data() -> None:
    arrays = build_state_arrays(
        _synthetic_workflow(1),
        options=StateExportOptions(r_max_bohr=8.0, k_max_bohr_inv=5.0),
    )
    bad_window = dict(arrays)
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["window"]["r_max_bohr_exclusive"] = float(arrays["r_bohr"][-1])
    bad_window["metadata_json"] = np.asarray(json.dumps(metadata))
    with pytest.raises(ValueError, match=r"contains r >="):
        validate_state_arrays(bad_window)

    bad_q = dict(arrays)
    bad_q["q_k"] = np.array(arrays["q_k"], copy=True)
    bad_q["n_scr_k"] = np.array(arrays["n_scr_k"], copy=True)
    bad_q["q_k"][0, 0] = np.nan
    bad_q["n_scr_k"][0, 0] = np.nan
    with pytest.raises(ValueError, match="q_k contains non-finite"):
        validate_state_arrays(bad_q)

    missing_configuration = dict(arrays)
    metadata = json.loads(str(arrays["metadata_json"].item()))
    del metadata["configuration"]
    missing_configuration["metadata_json"] = np.asarray(json.dumps(metadata))
    with pytest.raises(ValueError, match="workflow configuration"):
        validate_state_arrays(missing_configuration)


def test_state_export_rejects_missing_or_unconverged_ion_stage() -> None:
    missing = _synthetic_workflow(1)
    missing["ion"] = None
    with pytest.raises(ValueError, match="ion-structure stage"):
        build_state_arrays(missing)

    unconverged = _synthetic_workflow(1)
    unconverged["ion"]["hnc_converged"] = False
    with pytest.raises(ValueError, match="missing or unconverged HNC"):
        build_state_arrays(unconverged)

    unknown = _synthetic_workflow(1)
    del unknown["ion"]["hnc_converged"]
    with pytest.raises(ValueError, match="missing or unconverged HNC"):
        build_state_arrays(unknown)

    sc_unconverged = _synthetic_workflow(1)
    sc_unconverged["structure_model"] = "SC"
    sc_unconverged["sc_feedback"] = {"converged": False}
    with pytest.raises(ValueError, match="unconverged SC-feedback"):
        build_state_arrays(sc_unconverged)


def test_workflow_state_save_requires_ion_temperature() -> None:
    with pytest.raises(ValueError, match="save_state_npz requires"):
        PlasmaWorkflowConfig(
            elements=["C"],
            temperature_ev=10.0,
            rho_g_cc=1.0,
            save_state_npz=True,
        )
