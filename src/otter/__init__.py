"""Public API for otter."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from otter.workflows import (
    PlasmaWorkflowConfig,
    continue_plasma_workflow_from_electronic_result,
    parse_formula_composition,
    resolve_plasma_composition,
    run_formula_workflow,
    solve_plasma_workflow,
)
from otter.io.state import (
    StateExportOptions,
    load_plasma_state,
    save_plasma_state,
)
from otter.literature import (
    CitationMixin,
    bibliography_entries,
    get_bibtex_ref_string,
    get_cite_ref_string,
    get_formatted_ref_string,
    write_citations_markdown,
)

try:
    __version__ = version("otter")
except PackageNotFoundError:
    __version__ = "0+source"

__all__ = [
    "__version__",
    "CitationMixin",
    "PlasmaWorkflowConfig",
    "StateExportOptions",
    "continue_plasma_workflow_from_electronic_result",
    "bibliography_entries",
    "get_bibtex_ref_string",
    "get_cite_ref_string",
    "get_formatted_ref_string",
    "parse_formula_composition",
    "load_plasma_state",
    "resolve_plasma_composition",
    "run_formula_workflow",
    "save_plasma_state",
    "solve_plasma_workflow",
    "write_citations_markdown",
]
