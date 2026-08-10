"""Benchmark manifests must not contain unresolved public-release actions."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_reference_data_release_decisions_clear_public_release_gate(
    tmp_path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_public_release.py"),
            "--root",
            str(ROOT),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Public-release benchmark rights gates are clear." in completed.stdout
