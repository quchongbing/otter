"""Repository-wide policy for maintained scientific figures."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOT_SOURCE_DIRECTORIES = (
    ROOT / "benchmarks" / "examples",
    ROOT / "benchmarks" / "runners",
    ROOT / "docs" / "examples",
    ROOT / "examples",
    ROOT / "tools" / "diagnostics",
    ROOT / "tools" / "studies",
)
DIAGNOSTIC_TESTS = (
    "test_continuum_quantum_free.py",
    "test_continuum_resonance_adaptive_gaussian.py",
    "test_continuum_scattering_adaptive.py",
    "test_continuum_scattering_cache_reuse.py",
    "test_continuum_scattering_coulomb_phase_shift.py",
    "test_continuum_scattering_v0_normalization.py",
    "test_continuum_single_state_wavefunction.py",
    "test_continuum_tail_b3_recovery.py",
)


def _matplotlib_figure_sources() -> list[Path]:
    paths: list[Path] = []
    for directory in PLOT_SOURCE_DIRECTORIES:
        for path in directory.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if (
                "matplotlib.pyplot" in source
                and ("plt.subplots" in source or "plt.figure" in source)
            ):
                paths.append(path)
    paths.extend(ROOT / "tests" / name for name in DIAGNOSTIC_TESTS)
    return sorted(set(paths))


def test_every_repository_figure_uses_shared_style_and_dual_export() -> None:
    """Every maintained plot must use Otter style and PNG/PDF export."""
    paths = _matplotlib_figure_sources()
    assert paths
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert (
            "from otter.plotting import" in source
            or "import otter.plotting" in source
        ), path
        assert "set_style" in source or "style_context" in source, path
        assert "save_figure" in source, path
        assert ".savefig(" not in source, path


def test_public_galleries_use_thesis_bing_without_grids() -> None:
    """Capability and benchmark galleries share one publication style."""
    gallery_paths = sorted(
        (ROOT / "docs" / "examples").glob("plot_*.py")
    ) + sorted((ROOT / "benchmarks" / "examples").glob("plot_*.py"))
    assert gallery_paths
    for path in gallery_paths:
        source = path.read_text(encoding="utf-8")
        assert '"thesis", palette="bing"' in source, path
        assert ".grid(" not in source, path
