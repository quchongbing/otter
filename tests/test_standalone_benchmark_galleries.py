"""Contracts for the downloadable, independently executable benchmarks."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GALLERIES = (
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_bethkenhagen_et_al_2020_carbon_ionization.py",
    ROOT / "benchmarks" / "examples" / "plot_ion_structure_library.py",
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_johnson_et_al_2025_two_temperature_al.py",
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_argha_roy_carbon_sii.py",
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_starrett_et_al_2014_mixtures_fig3.py",
    ROOT
    / "benchmarks"
    / "examples"
    / "plot_starrett_single_species_2013_2014.py",
)


@pytest.mark.parametrize("path", GALLERIES, ids=lambda path: path.stem)
def test_benchmark_gallery_is_one_complete_otter_script(path: Path) -> None:
    """A live gallery must not delegate its scientific work to another file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    assert "USE_PRECOMPUTED_DATA = True" in source
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == "otter" or node.module.startswith("otter."))
        for node in ast.walk(tree)
    )
    plasma_workflow = (
        "PlasmaWorkflowConfig" in source
        and "solve_plasma_workflow(" in source
    )
    full_average_atom = (
        "FullExternalConfig" in source and "solve_full_only(" in source
    )
    assert plasma_workflow or full_average_atom
    assert "from otter.plotting import" in source
    assert "save_figure(" in source

    assert "importlib" not in source
    assert "benchmarks/runners" not in source
    assert "benchmarks\" / \"runners" not in source
    assert "regenerate_" not in source
