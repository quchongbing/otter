"""
otter/io/results.py

I/O helpers for saving full/external electronic-structure runs.

This module writes one self-contained NPZ file that includes:
- radial profiles (n_ion, n_cont, n_full, ...),
- serialized metadata (`meta_json`),
- selected scalar metadata entries as `meta_<key>` fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import re

import numpy as np


def _to_jsonable(value: Any) -> Any:
    """
    Convert nested objects to JSON-serializable values.

    Notes
    -----
    - numpy scalars/arrays are converted to Python scalars/lists.
    - tuples are converted to lists.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


def _safe_token(text: str) -> str:
    """Return a filename-safe token."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    token = token.strip("._")
    return token or "run"


def save_full_external_data(
    *,
    output_dir: str | Path,
    element_symbol: str,
    z: int,
    temperature_ev: float,
    rho_g_cc: float,
    suffix: str = "",
    result: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, str]:
    """
    Save one full/external run to disk.

    Parameters
    ----------
    output_dir
        Target directory for output files.
    element_symbol
        Chemical symbol (e.g. "Al").
    z
        Atomic number.
    temperature_ev
        Electron temperature in eV.
    rho_g_cc
        Mass density in g/cc.
    suffix
        Optional user suffix appended to filename.
    result
        Solver result dictionary returned by the high-level API.
    metadata
        Metadata dictionary (input settings + derived state scalars).

    Returns
    -------
    dict
        Paths of written files.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    symbol_tok = _safe_token(str(element_symbol))
    te_tok = _safe_token(f"{float(temperature_ev):.6g}")
    rho_tok = _safe_token(f"{float(rho_g_cc):.6g}")
    suffix_tok = _safe_token(str(suffix)) if str(suffix).strip() else ""
    stem = f"{symbol_tok}Z{int(z)}_Te{te_tok}_rho{rho_tok}"
    if suffix_tok:
        stem = f"{stem}_{suffix_tok}"
    data_path = out_dir / f"{stem}.npz"

    # Core radial arrays requested by the workflow. Missing keys are skipped
    # so the same function works for full-only or full+external runs.
    key_order = (
        "r",
        "n_ion",
        "n_cont",
        "n_full",
        "v_scf",
        "v_xc",
        "v_H",
        "n_ext",
        "n_scr",
        "v_ext",
        # Bound-state tables.
        "bound_l_list",
        "bound_n_index",
        "bound_energy_ha",
        "bound_fd",
        "bound_m",
        "bound_fdm",
        "bound_occ_deg_fd",
        "bound_occ_deg_fdm",
        # DOS arrays.
        "dos_energy_ha",
        "dos_bound",
        "dos_bound_fd",
        "dos_cont_ideal",
        "dos_cont_ideal_fd",
        "dos_cont_energy_ha",
        "dos_cont_scattering",
        "dos_cont_scattering_fd",
        "dos_cont_fd",
    )
    arrays: dict[str, np.ndarray] = {}
    for key in key_order:
        if key not in result:
            continue
        arr = np.asarray(result[key], dtype=float)
        if arr.ndim > 2:
            continue
        arrays[key] = arr

    meta_jsonable = _to_jsonable(dict(metadata))
    payload: dict[str, Any] = dict(arrays)
    payload["meta_json"] = np.array(
        json.dumps(meta_jsonable, ensure_ascii=False, sort_keys=True, indent=2)
    )

    # Also store scalar metadata entries as direct npz keys for quick access.
    if isinstance(meta_jsonable, dict):
        for k, v in meta_jsonable.items():
            if isinstance(v, (int, float, bool, str)) or v is None:
                payload[f"meta_{k}"] = np.array(v)

    np.savez_compressed(data_path, **payload)
    return {"data_npz": str(data_path)}


def _interp_profile_linear(
    *,
    r_src: np.ndarray,
    y_src: np.ndarray,
    r_dst: np.ndarray,
    right_value: float | None = None,
) -> np.ndarray:
    """Interpolate one radial profile to a common linear grid."""
    r_src = np.asarray(r_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    r_dst = np.asarray(r_dst, dtype=float)
    if r_src.ndim != 1 or y_src.ndim != 1 or r_src.size != y_src.size:
        raise ValueError("Expected one-dimensional radial profile arrays.")
    if r_src.size == 0:
        return np.zeros_like(r_dst)
    left = float(y_src[0])
    right = float(y_src[-1]) if right_value is None else float(right_value)
    return np.interp(r_dst, r_src, y_src, left=left, right=right)



def save_mixture_data(
    *,
    output_dir: str | Path,
    mixture_label: str,
    temperature_ev: float,
    rho_g_cc: float,
    result: Mapping[str, Any],
    suffix: str = "",
    save_common_linear_grid: bool = True,
    linear_n_points: int = 4096,
) -> dict[str, str]:
    """
    Save one multicomponent mixture AA run to disk.

    Parameters
    ----------
    output_dir
        Target directory for output files.
    mixture_label
        Compact composition label such as `C1H2O1`.
    temperature_ev
        Mixture temperature in eV.
    rho_g_cc
        Total mass density in g/cc.
    result
        Mixture solver result returned by `solve_mixture_full_then_ext`.
    suffix
        Optional filename suffix.
    save_common_linear_grid
        If True, interpolate all species profiles to one shared linear grid.
    linear_n_points
        Number of points on the optional shared linear grid.

    Returns
    -------
    dict
        Paths of written files.

    Notes
    -----
    The payload iterates over an arbitrary number of species stored in
    `result["species"]`.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mix_tok = _safe_token(str(mixture_label))
    te_tok = _safe_token(f"{float(temperature_ev):.6g}")
    rho_tok = _safe_token(f"{float(rho_g_cc):.6g}")
    suffix_tok = _safe_token(str(suffix)) if str(suffix).strip() else ""
    stem = f"mixture_{mix_tok}_Te{te_tok}_rho{rho_tok}"
    if suffix_tok:
        stem = f"{stem}_{suffix_tok}"
    data_path = out_dir / f"{stem}.npz"

    payload: dict[str, Any] = {}
    meta = dict(result.get("meta", {}))

    hist = list(result.get("history", []))
    if hist:
        keys = sorted({str(k) for row in hist for k in row.keys()})
        for key in keys:
            vals = [row.get(key, np.nan) for row in hist]
            try:
                payload[f"history_{key}"] = np.asarray(vals, dtype=float)
            except Exception:
                payload[f"history_{key}"] = np.asarray(vals, dtype=object)

    species_entries = list(result.get("species", []))
    if len(species_entries) < 2:
        raise ValueError("save_mixture_data expects at least two species entries.")

    native_profiles: dict[str, dict[str, Any]] = {}
    common_rmax = 0.0
    common_rmin = np.inf
    for sp in species_entries:
        symbol = _safe_token(str(sp["element"]))
        final = dict(sp["result"])
        native_profiles[symbol] = final
        r = np.asarray(final["r"], dtype=float)
        payload[f"{symbol}_r_native"] = r
        common_rmax = max(common_rmax, float(r[-1]))
        common_rmin = min(common_rmin, float(r[0]))
        for key in (
            "n_ion",
            "n_cont",
            "n_full",
            "n_ext",
            "n_scr",
            "v_scf",
            "v_H",
            "v_xc",
            "v_ext",
        ):
            if key in final:
                payload[f"{symbol}_{key}_native"] = np.asarray(final[key], dtype=float)
        for scalar_key in (
            "mu",
            "n0",
            "zbar",
            "q_full_ws",
            "q_cont_ws",
            "q_ext_ws",
            "r_ws",
        ):
            if scalar_key in final:
                payload[f"{symbol}_{scalar_key}"] = np.array(final[scalar_key])
        payload[f"{symbol}_count"] = np.array(sp["count"])
        payload[f"{symbol}_x"] = np.array(sp["x"])
        payload[f"{symbol}_volume_bohr3"] = np.array(sp["volume_bohr3"])
        payload[f"{symbol}_r_ws_bohr"] = np.array(sp["r_ws_bohr"])

    if save_common_linear_grid:
        r_linear = np.linspace(float(common_rmin), float(common_rmax), int(linear_n_points))
        payload["r_linear_common"] = r_linear
        for symbol, final in native_profiles.items():
            n0_val = float(final.get("n0", 0.0))
            interp_rules = {
                "n_ion": 0.0,
                "n_scr": 0.0,
                "n_cont": n0_val,
                "n_full": n0_val,
                "n_ext": n0_val,
                "v_scf": 0.0,
                "v_H": 0.0,
                "v_xc": 0.0,
                "v_ext": 0.0,
            }
            r_src = np.asarray(final["r"], dtype=float)
            for key, right_value in interp_rules.items():
                if key not in final:
                    continue
                payload[f"{symbol}_{key}_linear"] = _interp_profile_linear(
                    r_src=r_src,
                    y_src=np.asarray(final[key], dtype=float),
                    r_dst=r_linear,
                    right_value=right_value,
                )

    meta_jsonable = _to_jsonable(meta)
    payload["meta_json"] = np.array(
        json.dumps(meta_jsonable, ensure_ascii=False, sort_keys=True, indent=2)
    )
    if isinstance(meta_jsonable, dict):
        for k, v in meta_jsonable.items():
            if isinstance(v, (int, float, bool, str)) or v is None:
                payload[f"meta_{k}"] = np.array(v)

    np.savez_compressed(data_path, **payload)
    return {"data_npz": str(data_path)}
