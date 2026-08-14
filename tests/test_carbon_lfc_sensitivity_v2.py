"""Fast configuration checks for the candidate carbon LFC v2 protocol."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str, module_name: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_carbon_lfc_v2_producer_uses_reviewed_strict_resolution() -> None:
    producer = _load(
        "benchmarks/runners/regenerate_carbon_lfc_sensitivity.py",
        "otter_carbon_lfc_v2_producer",
    )
    config = producer._configuration(
        100.0,
        ion_temperature_ev=100.0,
        lfc_model="chabrier1990",
    )

    assert producer.RHO_G_CC == 5.0
    assert producer.QOZ_N_POINTS == 8192
    assert producer.REFERENCE_LFC == "chabrier1990"
    assert config.aa_overrides.get("n_points", 4096) == 4096
    assert config.aa_overrides.get("bound_occ_mode", "fd") == "fd"
    assert config.aa_overrides.get("b3_tail_model", "full") == "full"
    assert config.aa_overrides["bound_zero_tail_refine"] is True
    assert "bound_rmax_mult" not in config.aa_overrides
    assert config.hnc_require_converged is True
    assert config.hnc_tol == producer.HNC_TOL
    assert (
        config.hnc_closure_transform_tol
        == producer.HNC_CLOSURE_TRANSFORM_TOL
    )
    assert getattr(config, "allow_unconverged_aa", False) is False


def test_carbon_lfc_v2_runner_defaults_to_reviewed_package() -> None:
    runner = _load(
        "benchmarks/runners/plot_carbon_lfc_sensitivity.py",
        "otter_carbon_lfc_v2_runner",
    )

    assert runner.USE_PRECOMPUTED_DATA is True
    assert runner.EXPECTED_RHO_G_CC == 5.0
    assert runner.EXPECTED_TEMPERATURES_EV == (2.0, 100.0)
    assert runner.PRECOMPUTED_DATA_DIR is None
    assert runner.RECOMPUTED_DIR != runner.REVIEWED_DIR


def test_carbon_lfc_v2_candidate_manifest_is_explicit_in_source() -> None:
    source = (
        ROOT
        / "benchmarks"
        / "runners"
        / "regenerate_carbon_lfc_sensitivity.py"
    ).read_text(encoding="utf-8")
    assert '"status": "candidate_not_accepted"' in source
