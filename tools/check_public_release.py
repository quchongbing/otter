#!/usr/bin/env python3
"""Fail when a benchmark manifest still blocks public redistribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BLOCKING_VALUES = {
    "permission_not_established",
    "private_validation_only",
    "review_or_remove_before_public_release",
}

CLEARED_GATE_VALUES = {
    "cleared",
    "not_applicable",
    "resolved",
}


def _walk(value: Any, *, path: str = "") -> list[tuple[str, str]]:
    blockers: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if (
                key == "public_release_gate"
                and isinstance(child, str)
                and child.strip()
                and child.strip().lower() not in CLEARED_GATE_VALUES
            ):
                # A manifest author should not need to add each newly worded
                # release action to BLOCKING_VALUES.  The field name itself is
                # a contract: any non-cleared action blocks publication.
                blockers.append((child_path, child))
            else:
                blockers.extend(_walk(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            blockers.extend(_walk(child, path=f"{path}[{index}]"))
    elif isinstance(value, str) and value in BLOCKING_VALUES:
        blockers.append((path, value))
    return blockers


def public_release_blockers(root: Path) -> list[dict[str, str]]:
    """Return unresolved redistribution gates under ``root/benchmarks``."""
    records: list[dict[str, str]] = []
    for manifest in sorted((root / "benchmarks").rglob("manifest.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for key_path, value in _walk(payload):
            records.append(
                {
                    "manifest": manifest.relative_to(root).as_posix(),
                    "field": key_path,
                    "value": value,
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of this script).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the blocker list as JSON.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    blockers = public_release_blockers(root)
    if args.json:
        print(json.dumps(blockers, indent=2, sort_keys=True))
    elif blockers:
        print("PUBLIC RELEASE BLOCKED:")
        for item in blockers:
            print(f"- {item['manifest']}: {item['field']}={item['value']}")
    else:
        print("Public-release benchmark rights gates are clear.")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
