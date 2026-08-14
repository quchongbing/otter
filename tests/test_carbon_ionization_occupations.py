from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs" / "examples" / "plot_carbon_ionization_levels.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("carbon_ionization_levels", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the carbon ionization example module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_carbon_producer_requires_4096_radial_points() -> None:
    module = _load_example_module()
    assert module.AA_N_POINTS == 2**12
    assert module._configuration(2.0).n_points == 2**12
    assert module.SCHEMA == "otter_carbon_ionization_levels_v3"
    seeds = module._accepted_seed_rows(module.DENSITIES_G_CC)
    assert np.all(np.diff(module.DENSITIES_G_CC) > 0.0)
    assert np.all(module.DENSITIES_G_CC > 0.0)
    with np.load(module.BASELINE_PATH, allow_pickle=False) as archive:
        accepted_rho = np.asarray(archive["rho_g_cc"], dtype=float)
    expected_seed_count = np.count_nonzero(
        np.isin(accepted_rho, module.DENSITIES_G_CC)
    )
    assert len(seeds) == expected_seed_count
    assert {row["point_source"] for row in seeds} == {
        "accepted_baseline_seed"
    }


def test_main_extends_a_changed_density_grid_incrementally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module()
    staged_state = {"state": np.asarray("candidate")}
    observed: list[dict[str, np.ndarray]] = []

    def mismatch():
        raise module.DensityGridMismatchError("changed grid")

    monkeypatch.setattr(module, "RECOMPUTE_WITH_OTTER", False)
    monkeypatch.setattr(module, "_load_precomputed", mismatch)
    monkeypatch.setattr(module, "_compute_and_stage", lambda: staged_state)
    monkeypatch.setattr(module, "_print_state_table", observed.append)
    monkeypatch.setattr(module, "_plot", observed.append)

    module.main()

    assert observed == [staged_state, staged_state]


def test_failed_point_cache_keeps_nonconvergence_out_of_aa_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_example_module()
    monkeypatch.setattr(module, "POINT_CACHE_DIR", tmp_path)
    record = {
        "record_type": "stage2_nonconvergence",
        "rho_g_cc": 500.0,
        "elapsed_s": 1.0,
        "stage": "full_aa_stage2",
        "stage2_converged": False,
        "stage2_error": np.nan,
        "stage2_iters": 180,
        "bound_charge_min_e": 0.0,
        "bound_charge_max_e": 0.9,
        "message": "did not converge",
        "point_source": "fresh_otter_solve",
    }

    module._save_point_failure(record)
    loaded = module._load_point_failure(500.0)

    assert loaded is not None
    assert loaded["stage2_converged"] is False
    assert "zbar" not in loaded


def test_level_ion_charge_reduction_preserves_shell_labels_and_closure() -> None:
    module = _load_example_module()
    result = {
        "bound_l_list": np.asarray((0, 1)),
        "bound_q_ion_ws": np.asarray(
            (
                (1.0, np.nan),
                (0.25, 0.10),
            )
        ),
        "q_ion_ws": 1.35,
    }

    charges = module._finite_level_ion_charges(result)

    assert charges == pytest.approx({"1s": 1.0, "2p": 0.25, "3p": 0.10})

    bad = dict(result)
    bad["q_ion_ws"] = 1.5
    with pytest.raises(RuntimeError, match="does not close"):
        module._finite_level_ion_charges(bad)
