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
            "n_ion": n_ion[idx],
            "n_scr": n_scr[idx],
            "zbar": 2.0 if symbol == "C" else 1.0,
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
    gij = np.ones((n_species, n_species, r.size))
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

    return {
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
            "gij_r": gij_out,
            "sij_k": sij_out,
            "zbar": zbar_out,
            "n_i": n_i_out,
            "hnc_converged": True,
            "hnc_best_residual": 1.0e-7,
            "closure_transform_max_abs": 2.0e-8,
            "qoz_zbar_mode": "pseudoatom_partition",
            "qoz_response_chi0_model": "lindhard_fd",
            "qoz_response_lfc_model": "chabrier1990",
            "charge_fix": {"q_scr_rel": [1.0e-8] * n_species},
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
    np.testing.assert_array_equal(arrays["q_k"], arrays["n_scr_k"])
    np.testing.assert_array_equal(arrays["f_k"], arrays["n_ion_k"])
    assert float(np.max(arrays["r_bohr"])) < 20.0
    assert float(np.max(arrays["k_bohr_inv"])) < 20.0
    metadata = json.loads(str(arrays["metadata_json"].item()))
    assert metadata["model"]["lfc_model"] == "chabrier1990"
    assert metadata["units"]["q_k"] == "electron number"


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
