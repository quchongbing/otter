"""Fast, offline contracts for Otter's capability-oriented example gallery."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "docs" / "examples"

GALLERY_SCRIPTS = {
    "al_full": {
        "path": EXAMPLE_DIR / "plot_al_full_workflow.py",
        "solve_tokens": ("solve_plasma_workflow(",),
        "has_radial_axis": True,
    },
    "al_is_sc": {
        "path": EXAMPLE_DIR / "plot_al_is_sc_comparison.py",
        "solve_tokens": (
            "solve_plasma_workflow(",
            "solve_sc_feedback_workflow(",
        ),
        "has_radial_axis": True,
    },
    "carbon_lfc": {
        "path": EXAMPLE_DIR / "plot_carbon_lfc_sensitivity.py",
        "solve_tokens": (
            "solve_plasma_workflow(",
            "continue_plasma_workflow_from_electronic_result(",
        ),
        "has_radial_axis": True,
    },
    "al_qm_tf": {
        "path": EXAMPLE_DIR / "plot_al_qm_tf.py",
        "solve_tokens": ("solve_plasma_workflow(",),
        "has_radial_axis": True,
    },
    "al_rayleigh_weight": {
        "path": EXAMPLE_DIR / "plot_al_rayleigh_weight.py",
        "solve_tokens": ("solve_plasma_workflow(",),
        "has_radial_axis": False,
    },
    "carbon_ionization": {
        "path": EXAMPLE_DIR / "plot_carbon_ionization_levels.py",
        "solve_tokens": ("solve_full_only(",),
        "has_radial_axis": False,
    },
    "ch136_mixture": {
        "path": EXAMPLE_DIR / "plot_ch136_mixture_workflow.py",
        "solve_tokens": ("solve_plasma_workflow(",),
        "has_radial_axis": True,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.parametrize(
    "example", GALLERY_SCRIPTS.values(), ids=GALLERY_SCRIPTS
)
def test_gallery_script_is_standalone_reproducible_and_styled(
    example: dict[str, object],
) -> None:
    """Each page contains its calculation, reviewed-data path, and plotting."""
    path = example["path"]
    assert isinstance(path, Path)
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(path))

    # The gallery default is an offline, checksummed reviewed result; changing
    # one in-file switch executes the scientific calculation in this script.
    assert "RECOMPUTE_WITH_OTTER = False" in source
    assert "if RECOMPUTE_WITH_OTTER" in source
    assert "np.load(" in source
    assert "allow_pickle=False" in source
    assert "sha256" in source.lower()
    assert all(token in source for token in example["solve_tokens"])

    # A capability example must call Otter directly, not delegate its
    # calculation to a benchmark runner or another dynamically loaded script.
    assert "from otter" in source or "import otter" in source
    for forbidden in (
        "benchmarks.runners",
        "benchmarks/runners",
        "importlib",
        "runpy",
        "subprocess",
    ):
        assert forbidden not in source

    # The shared helper writes PNG and PDF by default and applies tight
    # bounding-box output; all gallery figures use the thesis/bing style.
    assert 'style_context("thesis", palette="bing")' in source
    assert "save_figure(" in source
    assert ".savefig(" not in source
    assert ".grid(" not in source
    if bool(example["has_radial_axis"]):
        assert "-0.5" in source


def test_gallery_detail_pages_enable_results_before_source() -> None:
    """Sphinx-Gallery keeps full source/downloads but previews figures first."""
    conf = (ROOT / "docs" / "source" / "conf.py").read_text(encoding="utf-8")
    javascript = (
        ROOT / "docs" / "source" / "_static" / "gallery_results_first.js"
    ).read_text(encoding="utf-8")
    css = (
        ROOT / "docs" / "source" / "_static" / "custom.css"
    ).read_text(encoding="utf-8")

    assert 'html_js_files = ["gallery_results_first.js"]' in conf
    assert "img.sphx-glr-single-img" in javascript
    assert "img.sphx-glr-multi-img" in javascript
    assert ".sphx-glr-script-out" in javascript
    assert "otter-results-first" in javascript
    assert "insertBefore(preview, firstCode)" in javascript
    assert ".otter-results-first" in css
    assert ".otter-result-grid" in css


def test_carbon_ionization_reviewed_archive_is_complete_and_checksummed() -> None:
    directory = ROOT / "benchmarks" / "baselines" / "carbon_ionization_levels"
    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    record = manifest["state"]
    archive_path = directory / record["data_file"]

    assert manifest["schema_version"] == "otter_example_manifest_v1"
    assert manifest["example_id"] == "carbon_ionization_levels"
    assert str(manifest["status"]).startswith("accepted")
    assert manifest["producer"]["script_sha256_current"] == _sha256(
        ROOT / manifest["producer"]["script_relative_path"]
    )
    assert _sha256(archive_path) == record["data_sha256"]
    assert any(
        reference["doi"] == "10.1016/j.hedp.2013.12.001"
        for reference in manifest["method_references"]
    )

    with np.load(archive_path, allow_pickle=False) as archive:
        assert archive["schema_version"].item() == (
            "otter_carbon_ionization_levels_v3"
        )
        assert archive["element_symbol"].item() == "C"
        assert float(archive["temperature_ev"]) == pytest.approx(100.0)
        rho = np.asarray(archive["rho_g_cc"], dtype=float)
        zbar = np.asarray(archive["zbar"], dtype=float)
        zstar = np.asarray(archive["zstar"], dtype=float)
        mu = np.asarray(archive["mu_ha"], dtype=float)
        status = np.asarray(archive["threshold_status"]).astype(str)
        edge_mode = str(archive["bound_energy_cut_mode"].item())
        edge_value = float(archive["bound_energy_cut_value"])
        labels = np.asarray(archive["level_labels"]).astype(str)
        energies = np.asarray(
            archive["level_energy_relative_edge_ev"], dtype=float
        )
        is_bound = np.asarray(archive["level_is_bound"], dtype=bool)
        stage2_converged = np.asarray(
            archive["stage2_converged"], dtype=bool
        )
        stage2_error = np.asarray(archive["stage2_error"], dtype=float)
        point_source = np.asarray(archive["point_source"]).astype(str)
        level_q_ion = np.asarray(archive["level_q_ion_ws"], dtype=float)
        level_q_ion_mask = np.asarray(
            archive["level_q_ion_ws_is_available"], dtype=bool
        )
        failed_rho = np.asarray(archive["failed_rho_g_cc"], dtype=float)
        aa_n_points = int(archive["aa_n_points"])

    assert rho.shape == (72,)
    assert np.all(np.diff(rho) > 0.0)
    assert rho[[0, -1]] == pytest.approx((0.1, 450.0))
    assert np.all(np.isfinite(zbar))
    assert np.all(np.isfinite(zstar))
    assert np.all(np.isfinite(mu))
    assert np.all((0.0 <= zbar) & (zbar <= 6.0))
    assert np.all((0.0 <= zstar) & (zstar <= 6.0))
    assert edge_mode == "v_frac"
    assert edge_value == pytest.approx(0.70)
    assert labels.tolist() == ["1s", "2s", "2p", "3s", "3p", "3d"]
    assert energies.shape == is_bound.shape == (rho.size, labels.size)
    assert np.all(energies[is_bound] < 0.0)
    assert np.all(energies[~is_bound] == 0.0)
    assert stage2_converged.shape == rho.shape
    assert stage2_error.shape == rho.shape
    assert point_source.shape == rho.shape
    assert np.all(stage2_converged)
    fresh = point_source == "fresh_otter_solve"
    accepted = point_source == "accepted_baseline_seed"
    assert np.count_nonzero(accepted) == 62
    assert np.count_nonzero(fresh) == 10
    assert rho[fresh] == pytest.approx(
        (1.05, 1.15, 1.30, 1.60, 1.80, 1.90, 2.10, 2.30, 2.35, 2.55)
    )
    assert np.all(np.isfinite(stage2_error))
    assert aa_n_points == 2**12
    assert failed_rho.size == 0
    assert level_q_ion.shape == level_q_ion_mask.shape == energies.shape
    displayed_q_ion = np.sum(
        np.where(level_q_ion_mask, level_q_ion, 0.0), axis=1
    )
    assert displayed_q_ion == pytest.approx(6.0 - zbar, abs=4.0e-6)
    assert np.count_nonzero(status == "resolved") == 66
    assert np.count_nonzero(status == "marginal") == 2
    assert np.count_nonzero(status == "unresolved") == 4
    assert rho[status == "marginal"] == pytest.approx((4.5, 4.6))
    assert rho[status == "unresolved"] == pytest.approx(
        (0.15, 0.3, 0.4, 0.45)
    )
    three_s = labels.tolist().index("3s")
    assert rho[is_bound[:, three_s]] == pytest.approx(
        (0.1, 0.2, 0.25, 0.35, 0.5, 0.55, 0.6)
    )
    two_s = labels.tolist().index("2s")
    assert rho[is_bound[:, two_s]] == pytest.approx(
        (
            0.1, 0.2, 0.25, 0.35, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
            1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6,
            1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.35, 2.4, 2.5,
            2.55, 2.6, 2.7, 2.8, 2.9, 3.0, 3.2, 3.5, 3.8, 4.0,
            4.1, 4.2, 4.3, 4.4, 4.7, 4.8, 4.9, 5.0,
        )
    )
    two_p = labels.tolist().index("2p")
    assert rho[is_bound[:, two_p]] == pytest.approx(
        (
            0.1, 0.2, 0.25, 0.35, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
            1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6,
            1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.35, 2.4, 2.5,
            2.55, 2.6,
        )
    )
    three_p = labels.tolist().index("3p")
    assert rho[is_bound[:, three_p]] == pytest.approx((0.1, 0.2, 0.25, 0.35))
    three_d = labels.tolist().index("3d")
    assert rho[is_bound[:, three_d]] == pytest.approx((0.1, 0.2, 0.25))


def test_ch136_mixture_archive_is_current_otter_and_strictly_accepted() -> None:
    directory = (
        ROOT / "benchmarks" / "baselines" / "ch136_mixture_workflow_100kk"
    )
    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    record = manifest["state"]
    archive_path = directory / record["data_file"]

    assert manifest["schema_version"] == "otter_example_manifest_v1"
    assert manifest["example_id"] == "ch136_mixture_workflow_100kk"
    assert str(manifest["status"]).startswith("accepted")
    assert manifest["data_rights"]["third_party_reference_data_included"] is False
    assert {
        reference["doi"] for reference in manifest["method_references"]
    } == {
        "10.1016/j.hedp.2013.12.001",
        "10.1103/PhysRevE.90.033110",
        "10.1051/jphys:0199000510150160700",
    }
    assert record["producer_project"] == "Otter"
    assert record["worktree_clean_at_generation"] is False
    assert record["producer_sha256_current"] == _sha256(
        ROOT / record["producer"]
    )
    assert _sha256(archive_path) == record["data_sha256"]
    acceptance = record["acceptance"]
    assert acceptance["common_mu_converged"] is True
    assert acceptance["all_species_aa_converged"] is True
    assert acceptance["all_species_external_aa_converged"] is True
    assert acceptance["hnc_converged"] is True
    assert acceptance["common_mu_residual_ha"] <= 1.0e-4
    assert acceptance["hnc_output_residual"] <= 1.0e-5
    assert acceptance["hnc_closure_mismatch"] <= 1.0e-4
    assert acceptance["hnc_min_eigenvalue"] > 0.0

    with np.load(archive_path, allow_pickle=False) as archive:
        assert archive["schema_version"].item() == (
            "otter_ch136_mixture_workflow_v1"
        )
        assert archive["species_symbols"].astype(str).tolist() == ["C", "H"]
        assert archive["species_counts"] == pytest.approx([1.0, 1.36])
        assert float(archive["rho_g_cc"]) == pytest.approx(5.0)
        assert float(archive["temperature_k"]) == pytest.approx(100_000.0)
        r = np.asarray(archive["r_bohr"], dtype=float)
        k = np.asarray(archive["k_bohr_inv"], dtype=float)
        gij = np.asarray(archive["gij_r"], dtype=float)
        sij = np.asarray(archive["sij_k"], dtype=float)
        q_k = np.asarray(archive["q_k"], dtype=float)
        vij_k = np.asarray(archive["vij_k"], dtype=float)
        zbar = np.asarray(archive["zbar_qoz"], dtype=float)
        q_used = np.asarray(archive["q_scr_dst_used"], dtype=float)
        mu = np.asarray(archive["mu_species_ha"], dtype=float)

    assert gij.shape == sij.shape == vij_k.shape == (2, 2, r.size)
    assert q_k.shape == (2, k.size)
    assert r.shape == k.shape
    assert np.all(np.diff(r) > 0.0)
    assert np.all(np.diff(k) > 0.0)
    assert np.all(np.isfinite(gij))
    assert np.all(np.isfinite(sij))
    assert np.all(np.isfinite(q_k))
    assert np.all(np.isfinite(vij_k))
    assert np.max(np.abs(gij - np.swapaxes(gij, 0, 1))) < 1.0e-12
    assert np.max(np.abs(sij - np.swapaxes(sij, 0, 1))) < 1.0e-12
    assert np.max(np.abs(vij_k - np.swapaxes(vij_k, 0, 1))) < 1.0e-12
    assert q_used == pytest.approx(zbar, rel=1.0e-12, abs=1.0e-12)
    assert np.max(mu) - np.min(mu) <= 1.0e-4


def test_al_rayleigh_weight_archive_is_checksummed_and_nonnegative() -> None:
    """The three-density Rayleigh example stores only finite Otter outputs."""
    directory = ROOT / "benchmarks" / "baselines" / "al_rayleigh_weight_10ev"
    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    archive_path = directory / str(manifest["state"]["data_file"])
    assert manifest["benchmark_id"] == "al_rayleigh_weight_10ev"
    assert _sha256(archive_path) == manifest["state"]["data_sha256"]
    assert manifest["producer"]["script_sha256_current"] == _sha256(
        ROOT / manifest["producer"]["script_relative_path"]
    )
    with np.load(archive_path, allow_pickle=False) as archive:
        assert archive["schema_version"].item() == (
            "otter_al_rayleigh_weight_10ev_v1"
        )
        assert int(archive["state_count"]) == 3
        rho = np.asarray(archive["rho_values_g_cc"], dtype=float)
        np.testing.assert_allclose(rho, (2.7, 8.1, 15.0))
        for index in range(3):
            prefix = f"state_{index}_"
            q = np.asarray(archive[prefix + "q_k"], dtype=float)
            f = np.asarray(archive[prefix + "f_k"], dtype=float)
            sii = np.asarray(archive[prefix + "sii_k"], dtype=float)
            weight = np.asarray(
                archive[prefix + "rayleigh_weight"], dtype=float
            )
            np.testing.assert_allclose(weight, np.abs(q + f) ** 2 * sii)
            assert np.all(np.isfinite(weight))
            assert np.min(weight) >= -1.0e-10
