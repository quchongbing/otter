r"""Portable, machine-readable plasma-state exports.

The public state format stores the quantities most often exchanged between
average-atom, QOZ/HNC, and XRTS workflows:

``q(k)``
    The charge-closed pseudoatom screening cloud
    :math:`n_\mathrm{scr}(k)`.
``f(k)``
    The ion-associated electron density :math:`n_\mathrm{ion}(k)`.
``g_ij(r)``, ``S_ij(k)``
    Ashcroft--Langreth partial pair distributions and static structure
    factors.

Only numeric arrays and fixed-width Unicode strings are written, so files can
always be loaded with ``allow_pickle=False``.  The default public window is
``r < 20 Bohr`` and ``k < 20 Bohr**-1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from otter.io._npz import save_npz_atomic
from otter.numerics.transforms import (
    precompute_dst_lattice_transform_like,
    radial_forward,
)


STATE_SCHEMA_VERSION = "otter_state_v1"


@dataclass(frozen=True)
class StateExportOptions:
    """Controls for one portable state export."""

    r_max_bohr: float = 20.0
    k_max_bohr_inv: float = 20.0
    require_converged_hnc: bool = True
    compressed: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.r_max_bohr)) or float(self.r_max_bohr) <= 0.0:
            raise ValueError("r_max_bohr must be finite and positive.")
        if (
            not np.isfinite(float(self.k_max_bohr_inv))
            or float(self.k_max_bohr_inv) <= 0.0
        ):
            raise ValueError("k_max_bohr_inv must be finite and positive.")


def _json_safe(value: Any) -> Any:
    """Convert nested NumPy-rich diagnostics to ordinary JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _package_version() -> str:
    try:
        return str(version("otter"))
    except PackageNotFoundError:
        return "0+source"


def _as_species_axis(values: Any, *, n_species: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if n_species == 1 and arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2 or arr.shape[0] != n_species:
        raise ValueError(
            f"{name} must have shape (n_species, n_grid); got {arr.shape}."
        )
    return arr


def _as_pair_axes(values: Any, *, n_species: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if n_species == 1 and arr.ndim == 1:
        arr = arr[np.newaxis, np.newaxis, :]
    if arr.ndim != 3 or arr.shape[:2] != (n_species, n_species):
        raise ValueError(
            f"{name} must have shape (n_species, n_species, n_grid); "
            f"got {arr.shape}."
        )
    return arr


def _species_entries(workflow: Mapping[str, Any]) -> list[dict[str, Any]]:
    electronic = dict(workflow["electronic"])
    electronic_result = dict(electronic["result"])
    if str(electronic["kind"]) == "single_species":
        symbol = str(
            electronic_result.get(
                "element",
                list(workflow.get("species_symbols", ["?"]))[0],
            )
        )
        return [{"element": symbol, "result": electronic_result}]
    return [
        {
            **dict(entry),
            "result": dict(entry["result"]),
        }
        for entry in electronic_result["species"]
    ]


def _interpolate_ion_density(
    *,
    entries: list[dict[str, Any]],
    r_target: np.ndarray,
) -> np.ndarray:
    out = np.zeros((len(entries), r_target.size), dtype=float)
    for idx, entry in enumerate(entries):
        result = dict(entry["result"])
        if "r" not in result or "n_ion" not in result:
            raise ValueError(
                f"Species {entry.get('element', idx)!r} lacks r/n_ion data "
                "required to construct f(k)."
            )
        r_src = np.asarray(result["r"], dtype=float)
        n_ion_src = np.asarray(result["n_ion"], dtype=float)
        if (
            r_src.ndim != 1
            or n_ion_src.shape != r_src.shape
            or r_src.size < 2
            or np.any(np.diff(r_src) <= 0.0)
        ):
            raise ValueError("Each native n_ion profile must use a monotone 1D grid.")
        out[idx] = np.interp(
            r_target,
            r_src,
            n_ion_src,
            left=float(n_ion_src[0]),
            right=0.0,
        )
    return out


def _metadata(
    workflow: Mapping[str, Any],
    ion: Mapping[str, Any],
    options: StateExportOptions,
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "producer": "otter",
        "producer_version": _package_version(),
        "units": {
            "r_bohr": "Bohr",
            "k_bohr_inv": "Bohr^-1",
            "density_r": "Bohr^-3",
            "q_k": "electron number",
            "f_k": "electron number",
            "gij_r": "dimensionless",
            "sij_k": "dimensionless",
            "vij_k": "Hartree Bohr^3",
        },
        "state": {
            "formula": workflow.get("formula"),
            "species_symbols": workflow.get("species_symbols"),
            "species_counts": workflow.get("species_counts"),
            "temperature_ev": workflow.get("temperature_ev"),
            "ion_temperature_ev": workflow.get("ion_temperature_ev"),
            "rho_g_cc": workflow.get("rho_g_cc"),
        },
        "model": {
            "electronic_kind": dict(workflow.get("electronic", {})).get("kind"),
            "structure_model": workflow.get("structure_model", "IS"),
            "sij_convention": ion.get(
                "sij_convention",
                "ashcroft_langreth",
            ),
            "qoz_zbar_mode": ion.get("qoz_zbar_mode"),
            "chi0_model": ion.get("qoz_response_chi0_model"),
            "lfc_model": ion.get("qoz_response_lfc_model"),
        },
        "window": {
            "r_max_bohr_exclusive": float(options.r_max_bohr),
            "k_max_bohr_inv_exclusive": float(options.k_max_bohr_inv),
        },
        "convergence": {
            "hnc_converged": ion.get("hnc_converged"),
            "hnc_best_residual": ion.get("hnc_best_residual"),
            "hnc_output_residual": ion.get("hnc_output_residual"),
            "closure_transform_max_abs": ion.get("closure_transform_max_abs"),
            "charge_fix": ion.get("charge_fix"),
        },
    }


def build_state_arrays(
    workflow: Mapping[str, Any],
    *,
    options: StateExportOptions | None = None,
) -> dict[str, np.ndarray]:
    """Build a portable state payload from a completed plasma workflow.

    The workflow must include the ionic stage.  This guarantees that all
    reciprocal-space quantities share the exact QOZ/DST lattice and that
    ``q(k)`` is the same charge-closed screening cloud used to construct the
    effective pair potential.
    """
    opts = options or StateExportOptions()
    ion_raw = workflow.get("ion")
    if not isinstance(ion_raw, Mapping) or not ion_raw:
        raise ValueError(
            "A completed ion-structure stage is required to export q/f/g/S."
        )
    ion = dict(ion_raw)
    if bool(opts.require_converged_hnc) and ion.get("hnc_converged") is not True:
        raise ValueError(
            "Refusing to export a missing or unconverged HNC status. Set "
            "require_converged_hnc=False only for an explicit diagnostic file."
        )
    structure_model = str(
        workflow.get("structure_model", ion.get("structure_model", "IS"))
    ).upper()
    if structure_model == "SC" and bool(opts.require_converged_hnc):
        feedback = workflow.get("sc_feedback", ion.get("sc_feedback"))
        if not isinstance(feedback, Mapping) or feedback.get("converged") is not True:
            raise ValueError(
                "Refusing to export a missing or unconverged SC-feedback "
                "status. Disable require_converged_hnc only for an explicit "
                "diagnostic file."
            )

    symbols = [str(value) for value in workflow["species_symbols"]]
    counts = np.asarray(workflow["species_counts"], dtype=float)
    if len(symbols) == 0 or counts.shape != (len(symbols),):
        raise ValueError("Workflow species symbols/counts are inconsistent.")
    fractions = counts / float(np.sum(counts))
    n_species = len(symbols)

    r_full = np.asarray(ion["r"], dtype=float)
    k_full = np.asarray(ion["k"], dtype=float)
    if (
        r_full.ndim != 1
        or k_full.ndim != 1
        or r_full.size < 2
        or r_full.size != k_full.size
    ):
        raise ValueError("Ion r/k grids must be equal-length one-dimensional arrays.")

    transform = precompute_dst_lattice_transform_like(r_full)
    if not np.allclose(transform.r, r_full, rtol=1.0e-12, atol=1.0e-13):
        raise ValueError("Ion r grid is not the strict DST lattice expected by QOZ.")
    if not np.allclose(transform.k, k_full, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("Ion k grid is inconsistent with its real-space DST lattice.")

    entries = _species_entries(workflow)
    if len(entries) != n_species:
        raise ValueError("Electronic and ionic species counts differ.")
    n_ion_r_full = _interpolate_ion_density(entries=entries, r_target=r_full)
    f_k_full = np.asarray(radial_forward(n_ion_r_full, transform), dtype=float)

    q_k_full = _as_species_axis(
        ion["n_scr_k"],
        n_species=n_species,
        name="n_scr_k",
    )
    n_scr_r_full = _as_species_axis(
        ion["n_scr_r"],
        n_species=n_species,
        name="n_scr_r",
    )
    gij_full = _as_pair_axes(
        ion.get("gij_r", ion.get("gii_r")),
        n_species=n_species,
        name="gij_r",
    )
    sij_full = _as_pair_axes(
        ion.get("sij_k", ion.get("sii_k")),
        n_species=n_species,
        name="sij_k",
    )

    r_mask = r_full < float(opts.r_max_bohr)
    k_mask = k_full < float(opts.k_max_bohr_inv)
    if not np.any(r_mask) or not np.any(k_mask):
        raise ValueError("Requested export window contains no grid points.")

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(STATE_SCHEMA_VERSION),
        "species_symbols": np.asarray(symbols, dtype="<U8"),
        "species_counts": counts,
        "species_number_fraction": fractions,
        "r_bohr": r_full[r_mask],
        "k_bohr_inv": k_full[k_mask],
        "n_ion_r": n_ion_r_full[:, r_mask],
        "n_scr_r": n_scr_r_full[:, r_mask],
        "f_k": f_k_full[:, k_mask],
        "q_k": q_k_full[:, k_mask],
        # Explicit aliases make the physics names self-documenting while
        # preserving the q/f notation commonly used by XRTS workflows.
        "n_ion_k": f_k_full[:, k_mask],
        "n_scr_k": q_k_full[:, k_mask],
        "gij_r": gij_full[..., r_mask],
        "sij_k": sij_full[..., k_mask],
        "zbar": np.atleast_1d(np.asarray(ion.get("zbar", np.nan), dtype=float)),
        "n_i_bohr3": np.atleast_1d(np.asarray(ion.get("n_i", np.nan), dtype=float)),
    }
    if "vij_k" in ion:
        arrays["vij_k"] = _as_pair_axes(
            ion["vij_k"],
            n_species=n_species,
            name="vij_k",
        )[..., k_mask]

    metadata_text = json.dumps(
        _json_safe(_metadata(workflow, ion, opts)),
        sort_keys=True,
        separators=(",", ":"),
    )
    arrays["metadata_json"] = np.asarray(metadata_text)
    validate_state_arrays(arrays)
    return arrays


def validate_state_arrays(arrays: Mapping[str, Any]) -> None:
    """Validate the public state schema without loading pickled objects."""
    required = {
        "schema_version",
        "species_symbols",
        "species_counts",
        "species_number_fraction",
        "r_bohr",
        "k_bohr_inv",
        "n_ion_r",
        "n_scr_r",
        "f_k",
        "q_k",
        "n_ion_k",
        "n_scr_k",
        "gij_r",
        "sij_k",
        "metadata_json",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"State payload is missing fields: {', '.join(missing)}")

    converted = {key: np.asarray(value) for key, value in arrays.items()}
    object_keys = [key for key, value in converted.items() if value.dtype.kind == "O"]
    if object_keys:
        raise ValueError(
            "Portable state payloads cannot contain object arrays: "
            + ", ".join(sorted(object_keys))
        )
    if str(converted["schema_version"].item()) != STATE_SCHEMA_VERSION:
        raise ValueError("Unsupported state schema version.")

    try:
        metadata = json.loads(str(converted["metadata_json"].item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("metadata_json is not valid scalar JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to an object.")
    if metadata.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("metadata_json has an unsupported schema version.")

    n_species = int(converted["species_symbols"].size)
    nr = int(converted["r_bohr"].size)
    nk = int(converted["k_bohr_inv"].size)
    if n_species < 1 or nr < 1 or nk < 1:
        raise ValueError("State species and r/k grids must be non-empty.")
    counts = converted["species_counts"]
    fractions = converted["species_number_fraction"]
    if counts.shape != (n_species,) or fractions.shape != (n_species,):
        raise ValueError("Species counts/fractions have inconsistent shapes.")
    if (
        np.any(~np.isfinite(counts))
        or np.any(counts <= 0.0)
        or np.any(~np.isfinite(fractions))
        or np.any(fractions <= 0.0)
        or not np.isclose(float(np.sum(fractions)), 1.0, atol=2.0e-14)
    ):
        raise ValueError("Species counts/fractions must be finite and positive.")

    r_grid = converted["r_bohr"]
    k_grid = converted["k_bohr_inv"]
    if (
        r_grid.ndim != 1
        or k_grid.ndim != 1
        or np.any(~np.isfinite(r_grid))
        or np.any(~np.isfinite(k_grid))
        or np.any(r_grid <= 0.0)
        or np.any(k_grid <= 0.0)
        or np.any(np.diff(r_grid) <= 0.0)
        or np.any(np.diff(k_grid) <= 0.0)
    ):
        raise ValueError("State r/k grids must be finite, positive, and increasing.")

    if converted["n_ion_r"].shape != (n_species, nr):
        raise ValueError("n_ion_r shape is inconsistent with species/r grids.")
    if converted["n_scr_r"].shape != (n_species, nr):
        raise ValueError("n_scr_r shape is inconsistent with species/r grids.")
    for key in ("f_k", "q_k", "n_ion_k", "n_scr_k"):
        if converted[key].shape != (n_species, nk):
            raise ValueError(f"{key} shape is inconsistent with species/k grids.")
    if converted["gij_r"].shape != (n_species, n_species, nr):
        raise ValueError("gij_r shape is inconsistent with species/r grids.")
    if converted["sij_k"].shape != (n_species, n_species, nk):
        raise ValueError("sij_k shape is inconsistent with species/k grids.")
    if "vij_k" in converted and converted["vij_k"].shape != (
        n_species,
        n_species,
        nk,
    ):
        raise ValueError("vij_k shape is inconsistent with species/k grids.")
    for key in (
        "species_counts",
        "species_number_fraction",
        "n_ion_r",
        "n_scr_r",
        "f_k",
        "q_k",
        "n_ion_k",
        "n_scr_k",
        "gij_r",
        "sij_k",
        "zbar",
        "n_i_bohr3",
        "vij_k",
    ):
        if key in converted and np.any(~np.isfinite(converted[key])):
            raise ValueError(f"{key} contains non-finite values.")
    if not np.array_equal(converted["f_k"], converted["n_ion_k"]):
        raise ValueError("f_k must be the explicit alias of n_ion_k.")
    if not np.array_equal(converted["q_k"], converted["n_scr_k"]):
        raise ValueError("q_k must be the explicit alias of n_scr_k.")

    try:
        r_limit = float(metadata["window"]["r_max_bohr_exclusive"])
        k_limit = float(metadata["window"]["k_max_bohr_inv_exclusive"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("metadata_json lacks valid exclusive r/k windows.") from exc
    if (
        not np.isfinite(r_limit)
        or r_limit <= 0.0
        or not np.isfinite(k_limit)
        or k_limit <= 0.0
    ):
        raise ValueError("metadata_json contains invalid exclusive r/k windows.")
    if np.any(r_grid >= r_limit):
        raise ValueError(f"State export contains r >= {r_limit:g} Bohr.")
    if np.any(k_grid >= k_limit):
        raise ValueError(f"State export contains k >= {k_limit:g} Bohr^-1.")


def save_plasma_state(
    path: str | Path,
    workflow: Mapping[str, Any],
    *,
    options: StateExportOptions | None = None,
) -> Path:
    """Save one completed workflow as a portable compressed NPZ state.

    The file is written beside the destination and atomically replaced only
    after NumPy has completed the archive.  A failed or interrupted export
    therefore cannot expose a partially written production state.
    """
    opts = options or StateExportOptions()
    arrays = build_state_arrays(workflow, options=opts)
    output = Path(path)
    if output.suffix.lower() != ".npz":
        output = output.with_suffix(output.suffix + ".npz")
    return save_npz_atomic(output, arrays, compressed=bool(opts.compressed))


def load_plasma_state(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate one portable state file with pickle disabled."""
    with np.load(Path(path), allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    validate_state_arrays(arrays)
    return arrays


__all__ = [
    "STATE_SCHEMA_VERSION",
    "StateExportOptions",
    "build_state_arrays",
    "load_plasma_state",
    "save_plasma_state",
    "validate_state_arrays",
]
