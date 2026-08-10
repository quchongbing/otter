"""Private helpers for safe NumPy archive writes."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


def save_npz_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    compressed: bool = True,
) -> Path:
    """Write an NPZ beside ``path`` and atomically replace the destination."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if bool(compressed) else np.savez
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            saver(temporary, **payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output


__all__ = ["save_npz_atomic"]
