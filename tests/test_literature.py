from __future__ import annotations

from pathlib import Path
import re

from otter.electronic.full_external import FullExternalConfig
from otter.electronic.mixture import MixtureConfig
from otter.ionic import LFC_MODEL_CITATION_KEYS
from otter.ionic.qoz import QOZResponseOptions
from otter.literature import (
    bibliography_entries,
    citation_keys_for_chi0_model,
    citation_keys_for_lfc_model,
    citation_keys_for_xc_model,
    get_bibtex_ref_string,
    get_cite_ref_string,
    get_formatted_ref_string,
    write_citations_markdown,
)
from otter.workflows import PlasmaWorkflowConfig


def test_bibliography_and_formatters_cover_every_key() -> None:
    entries = bibliography_entries()
    assert len(entries) >= 30
    for key in entries:
        plain = get_formatted_ref_string(key)
        bibtex = get_bibtex_ref_string(key)
        cite = get_cite_ref_string(key)
        assert plain
        assert bibtex.startswith("@")
        assert cite == rf"\cite{{{key}}}"


def test_model_dispatchers_return_registered_canonical_keys() -> None:
    entries = bibliography_entries()
    assert citation_keys_for_chi0_model("lindhard_fd") == ("Mermin1970",)
    assert set(citation_keys_for_chi0_model("lindhard_fd")) <= set(entries)
    for model in LFC_MODEL_CITATION_KEYS:
        assert set(citation_keys_for_lfc_model(model)) <= set(entries)
    assert citation_keys_for_xc_model("pbe") == (
        "LehtolaEtAl2018",
        "PerdewBurkeErnzerhof1996",
        "PerdewBurkeErnzerhof1997",
    )
    assert "LehtolaEtAl2018" in citation_keys_for_xc_model("libxc:gga_x_pbe")


def test_configuration_citation_api_is_auditable() -> None:
    configs = [
        FullExternalConfig("C", 2.0, 1.0, xc_model="lda_pw"),
        PlasmaWorkflowConfig(
            formula="C",
            temperature_ev=2.0,
            rho_g_cc=1.0,
            electronic_model="tf",
            qoz_response_lfc_model="hubbard",
        ),
        QOZResponseOptions(lfc_model="gregori2007"),
        MixtureConfig(
            species=("C", "H"),
            counts=(1.0, 1.0),
            temperature_ev=2.0,
            rho_g_cc=1.0,
            aa_overrides={"xc_model": "pbe"},
        ),
    ]
    for config in configs:
        assert config.citation_keys
        assert "\\cite{" in config.citation(style="cite")
        assert "@" in config.citation(style="bibtex")
        assert config.citation(style="plain")


def test_citation_registry_writer(tmp_path: Path) -> None:
    output = write_citations_markdown(tmp_path / "CITATIONS.md")
    text = output.read_text(encoding="utf-8")
    assert "# Otter citation policy and registry" in text
    assert "`StarrettSaumon2014`" in text
    assert "`LehtolaEtAl2018`" in text


def test_repository_citation_keys_are_registered() -> None:
    """Catch stale inline citations and benchmark-manifest keys early."""
    root = Path(__file__).resolve().parents[1]
    patterns = (
        # Citation roles in prose contain only comma-separated BibTeX keys;
        # the contributor guide's literal ``:cite:p:`` spelling is excluded.
        re.compile(
            r":cite:[pt]:`([A-Za-z][A-Za-z0-9]*(?:,[A-Za-z][A-Za-z0-9]*)*)`"
        ),
        re.compile(r'"citation_key"\s*:\s*"([^"]+)"'),
    )
    used: set[str] = set()
    for folder in ("src", "docs", "examples", "benchmarks"):
        for path in (root / folder).rglob("*"):
            if path.suffix not in {".py", ".rst", ".md", ".json"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in patterns:
                for match in pattern.findall(text):
                    used.update(part.strip() for part in match.split(","))
    assert used <= set(bibliography_entries())
