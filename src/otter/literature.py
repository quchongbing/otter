"""Runtime literature registry and citation formatting helpers.

The canonical bibliography is :mod:`otter`'s packaged ``literature.bib``.
This module deliberately has no dependency on ``pybtex`` so installed Otter
workflows can expose citations without installing the documentation extras.

The public :class:`CitationMixin` follows the small, explicit API used by
JaXRTS model objects: ``plain`` returns readable references, ``bibtex``
returns copyable entries, and ``cite`` returns unevaluated LaTeX citation
keys.  Configuration classes use the same interface to make the references
for a complete calculation auditable before it is run.
"""

from __future__ import annotations

from importlib.resources import files
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from otter._version import __version__


CitationEntry = str | tuple[str, str] | tuple[Sequence[str], str]


def _bibliography_text() -> str:
    return files("otter").joinpath("literature.bib").read_text(encoding="utf-8")


def _balanced_end(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        char = text[index]
        # BibTeX accent commands such as ``{\\"o}`` contain a quote, but
        # that quote does not delimit the outer entry. Brace balancing alone
        # is therefore more reliable here than treating every quote as a
        # string delimiter.
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("Unbalanced braces in literature.bib.")


def _entries_from_bibtex(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    pattern = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)
    for match in pattern.finditer(text):
        end = _balanced_end(text, text.find("{", match.start()))
        entries[match.group(1)] = text[match.start() : end + 1].strip()
    return entries


def _field_value(entry: str, field: str) -> str:
    match = re.search(
        rf"\b{re.escape(field)}\s*=\s*", entry, flags=re.IGNORECASE
    )
    if match is None:
        return ""
    index = match.end()
    while index < len(entry) and entry[index].isspace():
        index += 1
    if index >= len(entry):
        return ""
    if entry[index] == "{":
        end = _balanced_end(entry, index)
        return entry[index + 1 : end]
    if entry[index] == '"':
        end = index + 1
        escaped = False
        while end < len(entry):
            if escaped:
                escaped = False
            elif entry[end] == "\\":
                escaped = True
            elif entry[end] == '"':
                break
            end += 1
        return entry[index + 1 : end]
    end = entry.find(",", index)
    return entry[index:] if end < 0 else entry[index:end]


def _plain_tex(value: str) -> str:
    value = value.replace("--", "–").replace("~", " ")
    accent_map = {
        '"': {"a": "ä", "e": "ë", "i": "ï", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"},
        "'": {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "c": "ć", "A": "Á", "E": "É"},
        "`": {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù"},
        "^": {"a": "â", "e": "ê", "i": "î", "o": "ô", "u": "û"},
    }

    def replace_accent(match: re.Match[str]) -> str:
        mark, letter = match.group(1), match.group(2)
        return accent_map.get(mark, {}).get(letter, letter)

    value = re.sub(r'\\(["\'`^])\s*\{?([A-Za-z])\}?', replace_accent, value)
    value = re.sub(r"\\[{}]", "", value)
    value = value.replace("{", "").replace("}", "")
    return " ".join(value.split())


def _format_authors(value: str) -> str:
    authors: list[str] = []
    for author in re.split(r"\s+and\s+", value):
        author = _plain_tex(author)
        if "," in author:
            family, given = (part.strip() for part in author.split(",", 1))
            author = f"{given} {family}".strip()
        authors.append(author)
    return ", ".join(authors)


def _reference_string(key: str, comment: str | None = None) -> str:
    raw = bibliography_entries()[key]
    authors = _format_authors(_field_value(raw, "author"))
    title = _plain_tex(_field_value(raw, "title"))
    journal = _plain_tex(_field_value(raw, "journal"))
    volume = _plain_tex(_field_value(raw, "volume"))
    number = _plain_tex(_field_value(raw, "number"))
    pages = _plain_tex(_field_value(raw, "pages"))
    year = _plain_tex(_field_value(raw, "year"))
    doi = _plain_tex(_field_value(raw, "doi"))
    if not authors and not title:
        result = raw
    else:
        result = f"{authors}. {title}."
        if journal:
            result += f" {journal}"
        if volume:
            result += f" {volume}"
        if number:
            result += f"({number})"
        if pages:
            result += f", {pages}"
        if year:
            result += f" ({year})"
        result += "."
        if doi:
            result += f" DOI: https://doi.org/{doi}."
    if comment:
        result += f" {comment.rstrip('.')}."
    return result


_BIBLIOGRAPHY_CACHE: dict[str, str] | None = None


def bibliography_entries() -> dict[str, str]:
    """Return the packaged raw BibTeX entries keyed by citation key."""
    global _BIBLIOGRAPHY_CACHE
    if _BIBLIOGRAPHY_CACHE is None:
        _BIBLIOGRAPHY_CACHE = _entries_from_bibtex(_bibliography_text())
    return dict(_BIBLIOGRAPHY_CACHE)


def get_formatted_ref_string(key: str, comment: str | None = None) -> str:
    """Return one human-readable bibliography entry."""
    if key not in bibliography_entries():
        raise KeyError(f"Unknown Otter citation key: {key!r}")
    return _reference_string(key, comment)


def get_bibtex_ref_string(key: str, comment: str | None = None) -> str:
    """Return one raw BibTeX entry, optionally preceded by a comment."""
    if key not in bibliography_entries():
        raise KeyError(f"Unknown Otter citation key: {key!r}")
    prefix = f"% {comment}\n" if comment else ""
    return prefix + bibliography_entries()[key]


def get_cite_ref_string(key: str, comment: str | None = None) -> str:
    """Return one unevaluated LaTeX citation command."""
    if key not in bibliography_entries():
        raise KeyError(f"Unknown Otter citation key: {key!r}")
    prefix = f"% {comment}\n" if comment else ""
    return prefix + rf"\cite{{{key}}}"


def _expand_entries(entries: Iterable[CitationEntry]) -> list[tuple[str, str | None]]:
    expanded: list[tuple[str, str | None]] = []
    for entry in entries:
        if isinstance(entry, str):
            expanded.append((entry, None))
            continue
        keys, comment = entry
        if isinstance(keys, str):
            expanded.append((keys, comment))
        else:
            expanded.extend((str(key), comment) for key in keys)
    return expanded


class CitationMixin:
    """Mixin adding the plain/BibTeX/LaTeX citation API to model configs."""

    @property
    def citation_keys(self) -> Sequence[CitationEntry]:
        return getattr(self, "cite_keys", ())

    def citation(
        self,
        style: str = "plain",
        comment: str | None = None,
    ) -> str:
        """Return references for this configured model.

        Parameters
        ----------
        style
            ``"plain"``, ``"bibtex"``, or ``"cite"``.
        comment
            Optional provenance note appended to plain output or emitted as a
            BibTeX/LaTeX comment.
        """
        if style not in {"plain", "bibtex", "cite"}:
            raise ValueError("style must be 'plain', 'bibtex', or 'cite'.")
        expanded = _expand_entries(self.citation_keys)
        # A complete workflow often reaches the same paper through several
        # layers (for example the pseudoatom and QOZ options). Keep output
        # deterministic and readable while preserving the first comment.
        unique: list[tuple[str, str | None]] = []
        seen: set[str] = set()
        for key, entry_comment in expanded:
            if key in seen:
                continue
            seen.add(key)
            unique.append((key, entry_comment))
        expanded = unique
        if style == "cite":
            keys = ",".join(key for key, _ in expanded)
            prefix = f"% {comment}\n" if comment else ""
            return prefix + rf"\cite{{{keys}}}"
        formatter = (
            get_formatted_ref_string if style == "plain" else get_bibtex_ref_string
        )
        out: list[str] = []
        for key, entry_comment in expanded:
            note = entry_comment or comment
            if entry_comment and comment:
                note = f"{comment.rstrip('.')}. {entry_comment}"
            out.append(formatter(key, note))
        return "\n".join(out)


_CHI0_CITATION_KEYS = {
    # Otter evaluates the collisionless static finite-temperature Lindhard
    # response. Mermin (1970) records how this response enters a conserving
    # relaxation-time dielectric construction; ``lindhard_fd`` itself does
    # not add a finite collision frequency.
    "lindhard_fd": ("Mermin1970",),
}


def citation_keys_for_chi0_model(model: str) -> tuple[str, ...]:
    """Return canonical bibliography keys for a chi0 dispatcher choice."""
    key = str(model).strip().lower().replace("-", "_")
    aliases = {
        "lindhard": "lindhard_fd",
        "finite_temperature_lindhard": "lindhard_fd",
        "finite_t_lindhard": "lindhard_fd",
    }
    return _CHI0_CITATION_KEYS.get(aliases.get(key, key), ())


_LFC_CITATION_KEYS = {
    "none": ("PinesBohm1952",),
    "hubbard": ("Hubbard1958",),
    "utsumiichimaru": ("UtsumiIchimaru1982",),
    "chabrier1990": (
        "UtsumiIchimaru1982",
        "IchimaruEtAl1987",
        "Chabrier1990",
    ),
    "vashistasingwi": ("VashishtaSingwi1972", "Chabrier1990"),
    "chabrier_hubbard": ("Hubbard1958", "Chabrier1990"),
    "geldartvosko": ("GeldartVosko1966", "GregoriEtAl2007"),
    "gregori2007": (
        "UtsumiIchimaru1982",
        "GeldartVosko1966",
        "GregoriEtAl2007",
    ),
}


def citation_keys_for_lfc_model(model: str) -> tuple[str, ...]:
    """Return canonical bibliography keys for an LFC dispatcher choice."""
    key = str(model).strip().lower().replace("-", "_")
    aliases = {
        "ui": "utsumiichimaru",
        "utsumi_ichimaru": "utsumiichimaru",
        "chabrier_1990": "chabrier1990",
        "chabrier_ui": "chabrier1990",
        "chabrier_hubbard": "chabrier_hubbard",
        "ch": "chabrier_hubbard",
        "gregori": "gregori2007",
    }
    return _LFC_CITATION_KEYS.get(aliases.get(key, key), ())


_XC_CITATION_KEYS = {
    "dirac": ("Dirac1930",),
    "none": (),
    "pbe": (
        "LehtolaEtAl2018",
        "PerdewBurkeErnzerhof1996",
        "PerdewBurkeErnzerhof1997",
    ),
    "lda_pw": (
        "LehtolaEtAl2018",
        "Dirac1930",
        "Bloch1929",
        "PerdewWang1992",
    ),
    "lda_pz": (
        "LehtolaEtAl2018",
        "Dirac1930",
        "Bloch1929",
        "PerdewZunger1981",
    ),
    "lda_vwn": (
        "LehtolaEtAl2018",
        "Dirac1930",
        "Bloch1929",
        "VoskoWilkNusair1980",
    ),
}


def citation_keys_for_xc_model(model: str) -> tuple[str, ...]:
    """Return canonical bibliography keys for an XC model alias."""
    key = str(model).strip().lower()
    if key in _XC_CITATION_KEYS:
        return _XC_CITATION_KEYS[key]
    if key.startswith("libxc:"):
        names = [part.strip() for part in key[6:].split("+") if part.strip()]
        by_id = {
            "lda_x": ("Dirac1930", "Bloch1929"),
            "lda_c_pw": ("PerdewWang1992",),
            "lda_c_pz": ("PerdewZunger1981",),
            "lda_c_vwn": ("VoskoWilkNusair1980",),
            "gga_x_pbe": ("PerdewBurkeErnzerhof1996", "PerdewBurkeErnzerhof1997"),
            "gga_c_pbe": ("PerdewBurkeErnzerhof1996", "PerdewBurkeErnzerhof1997"),
        }
        keys = ["LehtolaEtAl2018"]
        for name in names:
            keys.extend(by_id.get(name, ()))
        return tuple(dict.fromkeys(keys))
    return ("LehtolaEtAl2018",) if key else ()


def write_citations_markdown(path: str | Path = "CITATIONS.md") -> Path:
    """Write the repository-wide citation policy and complete key registry."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Otter citation policy and registry",
        "",
        "This file is generated from `src/otter/literature.bib`. It is the",
        "developer-facing citation contract for source code, documentation,",
        "examples, benchmark manifests, and generated HTML.",
        "",
        "## For users",
        "",
        "If you use Otter in a scientific publication, please cite Chongbing",
        f"Qu, *Otter*, version {__version__}, computer software (2026),",
        "https://github.com/otter-hed/otter.",
        "",
        "Model references record scientific provenance; they are not additional",
        "software-citation requirements.",
        "Configuration objects expose:",
        "",
        "```python",
        "config.citation(style=\"plain\")   # readable references",
        "config.citation(style=\"bibtex\")  # copyable BibTeX",
        "config.citation(style=\"cite\")    # \\cite{...}",
        "```",
        "",
        "The canonical HTML bibliography is generated by Sphinx from the same",
        "BibTeX file: `docs/source/bibliography.rst`.",
        "",
        "## For contributors",
        "",
        "1. Add or update the primary paper in `src/otter/literature.bib`.",
        "2. Use a stable citation key and DOI; do not invent uncited shorthand.",
        "3. Add a `:cite:p:`/`:cite:t:` reference beside equations or model",
        "   descriptions in RST, and a `References` section in Python docstrings.",
        "4. For selectable models, expose the selected keys through a",
        "   `citation_keys` property and test all three citation styles.",
        "5. Mark Otter-specific numerical choices separately from the cited",
        "   physical model; do not attribute implementation details to papers.",
        "6. Run `python tools/update_citations.py`, the citation tests, and the",
        "   documentation build before submitting a model or benchmark.",
        "",
        "## Selectable-model mappings",
        "",
        "The runtime keeps these mappings in `citation_keys_for_chi0_model`,",
        "`citation_keys_for_lfc_model`, and `citation_keys_for_xc_model`; use",
        "those helpers instead of duplicating",
        "paper lists in examples or output manifests.",
        "",
        "### Non-interacting electron response",
        "",
    ]
    for model in sorted(_CHI0_CITATION_KEYS):
        keys = ", ".join(f"`{key}`" for key in _CHI0_CITATION_KEYS[model])
        lines.append(f"- `{model}`: {keys}")
    lines.extend(
        [
            "",
            "### Ionic local-field corrections",
            "",
        ]
    )
    for model in sorted(_LFC_CITATION_KEYS):
        keys = ", ".join(f"`{key}`" for key in _LFC_CITATION_KEYS[model])
        lines.append(f"- `{model}`: {keys}")
    lines.extend(
        [
            "",
            "### Exchange-correlation aliases",
            "",
        ]
    )
    for model in sorted(_XC_CITATION_KEYS):
        keys = ", ".join(f"`{key}`" for key in _XC_CITATION_KEYS[model]) or "(none)"
        lines.append(f"- `{model}`: {keys}")
    lines.extend(
        [
            "- `libxc:<functional>[+<functional>...]`: `LehtolaEtAl2018` plus",
            "  the primary references returned by the installed Libxc runtime.",
            "  The exact version, IDs, references, and DOIs are retained in",
            "  `xc_provenance`; unknown functionals must not be hand-catalogued.",
            "",
        "## Complete bibliography",
        "",
        ]
    )
    for key in sorted(bibliography_entries()):
        lines.extend([f"### `{key}`", "", get_formatted_ref_string(key), ""])
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target


__all__ = [
    "CitationEntry",
    "CitationMixin",
    "bibliography_entries",
    "citation_keys_for_chi0_model",
    "citation_keys_for_lfc_model",
    "citation_keys_for_xc_model",
    "get_bibtex_ref_string",
    "get_cite_ref_string",
    "get_formatted_ref_string",
    "write_citations_markdown",
]
