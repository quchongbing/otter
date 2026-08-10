"""Regenerate the repository-wide citation registry from package BibTeX."""

from __future__ import annotations

from otter.literature import write_citations_markdown


if __name__ == "__main__":
    print(write_citations_markdown())
