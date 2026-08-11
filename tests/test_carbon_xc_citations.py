"""Citation-provenance contract for the carbon XC comparison."""
from __future__ import annotations

import json
from pathlib import Path
import runpy


EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "studies"
    / "carbon_xc_comparison.py"
)


def test_comparison_writes_machine_and_human_readable_xc_citations(
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path(str(EXAMPLE))
    records = namespace["_write_xc_provenance"](tmp_path, ["dirac"])

    assert records["dirac"]["provider"] == "otter_builtin"
    payload = json.loads((tmp_path / "xc_provenance.json").read_text())
    assert payload["schema"] == "otter-xc-provenance-v1"
    assert payload["models"]["dirac"]["components"][0]["dois"] == [
        "10.1017/S0305004100016108"
    ]

    citations = (tmp_path / "CITATIONS.md").read_text()
    assert "Otter numerical regularization" in citations
    assert "https://doi.org/10.1017/S0305004100016108" in citations
