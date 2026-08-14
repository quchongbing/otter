"""Sphinx configuration for the Otter documentation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

from sphinx_gallery.sorting import FileNameSortKey

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "otter"

# Always document this checkout, even when another editable Otter install is
# active in the Sphinx environment.
sys.path.insert(0, str(SOURCE_ROOT))

project = "Otter"
author = "Chongbing Qu and Dominik Kraus"
copyright = "2026, Chongbing Qu"

try:
    release = version("otter")
except PackageNotFoundError:
    # This fallback keeps source-tree documentation introspection useful before
    # the first editable install. CI and release builds install the package.
    release = "0+unknown"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
]

source_suffix = {".rst": "restructuredtext"}
root_doc = "index"
language = "en"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "sg_execution_times.rst",
    "**/sg_execution_times.rst",
]

autosummary_generate = True
autoclass_content = "both"
autodoc_default_options = {
    "show-inheritance": True,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False

# Sphinx-Gallery's documented sorting key is a callable class and is therefore
# intentionally not serialised into Sphinx's environment cache.
suppress_warnings = ["config.cache"]


sphinx_gallery_conf = {
    "examples_dirs": ["../examples", "../../benchmarks/examples"],
    "gallery_dirs": ["gen_examples", "benchmarks/gen_benchmarks"],
    "filename_pattern": r"plot_",
    "reference_url": {"otter": None},
    "within_subsection_order": FileNameSortKey,
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("otter",),
    "remove_config_comments": True,
    # Examples load checksummed reference results during documentation builds.
    # Scientific calculation times, where relevant, are reported explicitly
    # by the example rather than as Sphinx-Gallery file-execution overhead.
    "min_reported_time": float("inf"),
}

bibtex_bibfiles = [str(PACKAGE_ROOT / "literature.bib")]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"

html_theme = "sphinx_rtd_theme"
html_title = f"Otter {release} documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["gallery_results_first.js"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
}
