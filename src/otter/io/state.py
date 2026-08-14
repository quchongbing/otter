r"""Portable, machine-readable plasma-state exports.

The public state format stores the converged average-atom, pseudoatom, and
QOZ/HNC quantities needed for analysis or downstream XRTS calculations:

``q(k)``
    The charge-closed pseudoatom screening cloud
    :math:`n_\mathrm{scr}(k)`.
``f(k)``
    The ion-associated electron density :math:`n_\mathrm{ion}(k)`.
``g_ij(r)``, ``S_ij(k)``
    Ashcroft--Langreth partial pair distributions and static structure
    factors.
``V_Ie(k)``, ``V_ee(k)``, ``C_Ie(k)``, ``C_ee(k)``
    Electron--ion and electron--electron potentials and direct-correlation
    channels used by the QOZ construction.
``chi0_k``, ``chi_ee_k``, ``gee_k``
    Ideal and interacting electron responses and the selected local-field
    correction :math:`G_{ee}(k)`.
``species_<i>_*``
    Native-grid electronic densities, potential components, mean-ionization
    definitions, bound levels, and per-level density contributions.

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
    radial_inverse,
)


STATE_SCHEMA_VERSION = "otter_state_v3"
_SUPPORTED_STATE_SCHEMA_VERSIONS = {
    "otter_state_v1",
    "otter_state_v2",
    STATE_SCHEMA_VERSION,
}


@dataclass(frozen=True)
class StateExportOptions:
    """Controls for one portable state export."""

    r_max_bohr: float = 20.0
    k_max_bohr_inv: float = 20.0
    require_converged_hnc: bool = True
    compressed: bool = True
    include_electronic_profiles: bool = True
    include_orbital_densities: bool = True

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


_ELECTRON_DENSITY_FIELDS = {
    "n_e": "n_e_r",
    "n_e_base": "n_e_base_r",
    "n_full": "n_full_r",
    "n_bound": "n_bound_r",
    "n_cont": "n_cont_r",
    "n_ext": "n_ext_r",
    "n_ext_pre_tail": "n_ext_pre_tail_r",
    "n_pa": "n_pa_r",
    "n_pa_repaired": "n_pa_repaired_r",
    "n_scr": "n_scr_r_native",
    "n_scr_repaired": "n_scr_repaired_r",
    "n_ion": "n_ion_r_native",
    "n_full_source": "n_full_source_r",
    "n_cont_dft_raw": "n_cont_dft_raw_r",
    "n_positive_energy_tf": "n_positive_energy_tf_r",
    "n_negative_tf": "n_negative_tf_r",
}

_ELECTRON_POTENTIAL_FIELDS = {
    "v_full": "v_full_r_ha",
    "v_scf": "v_scf_r_ha",
    "v_ext": "v_ext_r_ha",
    "v_nuc": "v_nuc_r_ha",
    "v_H": "v_hartree_r_ha",
    "v_xc": "v_xc_r_ha",
    "v_H_ext": "v_hartree_ext_r_ha",
    "v_xc_ext": "v_xc_ext_r_ha",
    "v_corr_full": "v_corr_full_r_ha",
    "v_corr_ext": "v_corr_ext_r_ha",
}

_ELECTRON_SPECTRAL_FIELDS = (
    "dos_energy_ha",
    "dos_bound",
    "dos_bound_fd",
    "dos_cont_ideal",
    "dos_cont_ideal_fd",
    "dos_cont_energy_ha",
    "dos_cont_scattering",
    "dos_cont_scattering_fd",
    "dos_cont_fd",
    "cont_phase_energy_ha",
    "cont_phase_shift_rad",
)


def _finite_numeric(value: Any) -> np.ndarray | None:
    """Return one finite numeric array, or ``None`` when unavailable."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind not in "biufc" or np.any(~np.isfinite(array)):
        return None
    return array


def _entry_n_i(entry: Mapping[str, Any]) -> float:
    """Return one species ion density without evaluating an absent fallback."""
    result = dict(entry["result"])
    if "n_i" in result:
        return float(result["n_i"])
    if "n_i_bohr3" in result:
        return float(result["n_i_bohr3"])
    if "volume_bohr3" in entry:
        return 1.0 / float(entry["volume_bohr3"])
    r_ws = result.get("r_ws", entry.get("r_ws_bohr"))
    if r_ws is not None:
        r_ws_value = float(r_ws)
        if np.isfinite(r_ws_value) and r_ws_value > 0.0:
            return float(3.0 / (4.0 * np.pi * r_ws_value**3))
    raise ValueError(
        f"Species {entry.get('element', '?')!r} lacks n_i, volume, and Rws data."
    )


def _species_vector(
    ion: Mapping[str, Any],
    key: str,
    *,
    entries: list[dict[str, Any]],
    fallback_keys: tuple[str, ...],
) -> np.ndarray:
    """Return one finite scalar per species from the ion or AA result."""
    n_species = len(entries)
    if key in ion:
        value = np.atleast_1d(np.asarray(ion[key], dtype=float))
        if value.shape == (n_species,) and np.all(np.isfinite(value)):
            return value
    values = []
    for entry in entries:
        result = dict(entry["result"])
        value = np.nan
        for fallback_key in fallback_keys:
            if fallback_key in result:
                value = float(result[fallback_key])
                break
        values.append(value)
    return np.asarray(values, dtype=float)


def _add_species_electronic_arrays(
    arrays: dict[str, np.ndarray],
    *,
    entries: list[dict[str, Any]],
    r_max_bohr: float,
    include_profiles: bool,
    include_orbital_densities: bool,
) -> None:
    """Add native-grid AA fields using stable species-index prefixes."""
    for species_index, entry in enumerate(entries):
        result = dict(entry["result"])
        prefix = f"species_{species_index}_"
        r_native = np.asarray(result.get("r", ()), dtype=float)
        if (
            r_native.ndim != 1
            or r_native.size < 1
            or np.any(~np.isfinite(r_native))
            or np.any(np.diff(r_native) <= 0.0)
        ):
            continue
        r_mask = r_native < float(r_max_bohr)
        if not np.any(r_mask):
            continue
        arrays[prefix + "r_bohr"] = r_native[r_mask]

        n_i = _entry_n_i(entry)
        n0 = float(result.get("n0", np.nan))
        scalar_values = {
            "nuclear_charge": result.get("Z", entry.get("Z", np.nan)),
            "r_ws_bohr": result.get("r_ws", entry.get("r_ws_bohr", np.nan)),
            "mu_ha": result.get("mu", entry.get("mu_ha", np.nan)),
            "n_i_bohr3": n_i,
            "n0_bohr3": n0,
            "zbar_aa": result.get("zbar", np.nan),
            "zbar_partition": result.get("zbar_partition", np.nan),
            "zbar_ws": result.get("zbar_ws", result.get("zbar", np.nan)),
            "zstar": n0 / n_i if np.isfinite(n0) and n_i > 0.0 else np.nan,
            "bound_energy_cut_ha": result.get("bound_energy_cut_ha", np.nan),
            "shallowest_bound_energy_ha": result.get(
                "shallowest_bound_energy_ha", np.nan
            ),
            "runtime_s": result.get("runtime_s", np.nan),
        }
        for name, value in scalar_values.items():
            value_array = _finite_numeric(value)
            if value_array is not None and value_array.size == 1:
                arrays[prefix + name] = np.asarray(float(value_array.item()))

        if include_profiles:
            for source, target in {
                **_ELECTRON_DENSITY_FIELDS,
                **_ELECTRON_POTENTIAL_FIELDS,
            }.items():
                value = _finite_numeric(result.get(source))
                if value is not None and value.shape == r_native.shape:
                    arrays[prefix + target] = np.asarray(value[r_mask], dtype=float)
            if "n_free" in result:
                n_free = np.asarray(result["n_free"], dtype=float)
                if n_free.shape == r_native.shape:
                    free_mask = r_mask & np.isfinite(n_free)
                    if np.any(free_mask):
                        arrays[prefix + "n_free_r_bohr"] = r_native[free_mask]
                        arrays[prefix + "n_free_r"] = n_free[free_mask]
            g_background = _finite_numeric(result.get("g_ii_background"))
            if g_background is not None and g_background.shape == r_native.shape:
                arrays[prefix + "g_ii_background_r"] = np.asarray(
                    g_background[r_mask], dtype=float
                )

        try:
            energies = np.asarray(result.get("bound_energy_ha"), dtype=float)
        except (TypeError, ValueError):
            energies = np.empty((0, 0), dtype=float)
        angular = _finite_numeric(result.get("bound_l_list"))
        if energies.ndim == 2 and energies.size and angular is not None:
            l_values = np.asarray(angular, dtype=int).reshape(-1)
            if l_values.shape == (energies.shape[0],):
                energy_cut = float(result.get("bound_energy_cut_ha", 0.0))
                selected = np.isfinite(energies) & (energies < energy_cut)
                l_grid = np.broadcast_to(l_values[:, None], energies.shape)
                n_grid = np.broadcast_to(
                    np.arange(1, energies.shape[1] + 1, dtype=int)[None, :],
                    energies.shape,
                )
                arrays[prefix + "bound_l"] = l_grid[selected]
                arrays[prefix + "bound_n_index"] = n_grid[selected]
                arrays[prefix + "bound_principal_n"] = (
                    n_grid[selected] + l_grid[selected]
                )
                arrays[prefix + "bound_energy_ha"] = energies[selected]
                for source in (
                    "bound_fd",
                    "bound_m",
                    "bound_fdm",
                    "bound_occ_deg_fd",
                    "bound_occ_deg_fdm",
                    "bound_q_ion_ws",
                ):
                    try:
                        value = np.asarray(result.get(source), dtype=float)
                    except (TypeError, ValueError):
                        continue
                    if value.shape == energies.shape and np.all(
                        np.isfinite(value[selected])
                    ):
                        arrays[prefix + source] = np.asarray(
                            value[selected], dtype=float
                        )
                if include_orbital_densities:
                    for source in (
                        "bound_orbital_density_r",
                        "ion_orbital_density_r",
                    ):
                        value = _finite_numeric(result.get(source))
                        if value is not None and value.shape == (*energies.shape, r_native.size):
                            arrays[prefix + source] = np.asarray(
                                value[selected][:, r_mask], dtype=float
                            )

        for source in _ELECTRON_SPECTRAL_FIELDS:
            value = _finite_numeric(result.get(source))
            if value is not None:
                arrays[prefix + source] = np.asarray(value)

        for source in (
            "bound_occ_mode",
            "threshold_state_status",
            "threshold_state_representation",
            "threshold_spectral_representation_status",
        ):
            value = result.get(source, dict(result.get("meta", {})).get(source))
            if value is not None:
                arrays[prefix + source] = np.asarray(str(value), dtype="<U96")


def _metadata(
    workflow: Mapping[str, Any],
    ion: Mapping[str, Any],
    options: StateExportOptions,
) -> dict[str, Any]:
    electronic = dict(workflow.get("electronic", {}))
    electronic_result = dict(electronic.get("result", {}))
    entries = _species_entries(workflow)
    electronic_convergence = []
    for entry in entries:
        result = dict(entry["result"])
        meta = dict(result.get("meta", {}))
        ext_status = dict(result.get("ext_status", {}))
        electronic_convergence.append(
            {
                "species": entry.get("element"),
                "stage1_converged": result.get(
                    "stage1_converged", meta.get("stage1_converged")
                ),
                "stage2_converged": result.get(
                    "stage2_converged", meta.get("stage2_converged")
                ),
                "external_converged": ext_status.get(
                    "converged", meta.get("ext_converged")
                ),
                "threshold_state_status": result.get(
                    "threshold_state_status", meta.get("threshold_state_status")
                ),
                "threshold_state_representation": result.get(
                    "threshold_state_representation",
                    meta.get("threshold_state_representation"),
                ),
                "runtime_s": result.get("runtime_s", meta.get("runtime_s")),
            }
        )
    mixture_meta = dict(electronic_result.get("meta", {}))
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "producer": "otter",
        "producer_version": _package_version(),
        "configuration": workflow.get("configuration", {}),
        "citation_keys": workflow.get("citation_keys", ()),
        "units": {
            "r_bohr": "Bohr",
            "k_bohr_inv": "Bohr^-1",
            "density_r": "Bohr^-3",
            "q_k": "electron number",
            "f_k": "electron number",
            "gij_r": "dimensionless",
            "sij_k": "dimensionless",
            "hij_r": "dimensionless",
            "cij_r": "dimensionless",
            "vij_r": "Hartree",
            "vij_k": "Hartree Bohr^3",
            "v_ie_k": "Hartree Bohr^3",
            "v_ei_k": "Hartree Bohr^3",
            "v_ee_k": "Hartree Bohr^3",
            "c_ie_k": "Bohr^3",
            "c_ee_k": "Bohr^3",
            "v_ie_r": "Hartree",
            "v_ee_r": "Hartree",
            "c_ie_r": "dimensionless",
            "c_ee_r": "dimensionless",
            "chi0_k": "Bohr^-3 Hartree^-1",
            "chi_ee_k": "Bohr^-3 Hartree^-1",
            "gee_k": "dimensionless",
            "g_ee_k": "dimensionless",
            "electronic_density_r": "Bohr^-3",
            "electronic_potential_r": "Hartree",
            "bound_energy_ha": "Hartree",
            "mu_ha": "Hartree",
        },
        "array_layout": {
            "common_species": "(species, grid)",
            "pairs": "(species_i, species_j, grid)",
            "native_electronic": "species_<index>_<field>",
            "orbital_density": "(bound_level, native_r)",
            "species_order": workflow.get("species_symbols"),
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
            "electronic": electronic_convergence,
            "common_mu_residual_max_ha": mixture_meta.get(
                "final_mu_residual_max_ha"
            ),
            "common_mu_success": mixture_meta.get("final_mu_root_success"),
            "hnc_converged": ion.get("hnc_converged"),
            "hnc_best_residual": ion.get("hnc_best_residual"),
            "hnc_output_residual": ion.get("hnc_output_residual"),
            "closure_transform_max_abs": ion.get("closure_transform_max_abs"),
            "charge_fix": ion.get("charge_fix"),
        },
        "definitions": {
            "zbar": "QOZ charge selected by qoz_zbar_mode",
            "zstar": "n0 / n_i",
            "bound_orbital_density_r": (
                "per-level contribution to n_bound using bound_occ_mode"
            ),
            "ion_orbital_density_r": (
                "per-level contribution to n_ion including M(E) and f_cut(r)"
            ),
            "real_space_electron_channels": (
                "inverse transform on the finite QOZ DST lattice"
            ),
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
    v_ie_k_full = _as_species_axis(
        ion["v_ie_k"],
        n_species=n_species,
        name="v_ie_k",
    )
    c_ie_k_full = _as_species_axis(
        ion["c_ie_k"],
        n_species=n_species,
        name="c_ie_k",
    )
    v_ee_k_full = np.asarray(ion["v_ee_k"], dtype=float)
    c_ee_k_full = np.asarray(ion["c_ee_k"], dtype=float)
    chi0_k_full = np.asarray(ion["chi0_k"], dtype=float)
    chi_ee_k_full = np.asarray(ion["chi_ee_k"], dtype=float)
    gee_k_full = np.asarray(ion["gee_k"], dtype=float)
    if any(
        value.shape != k_full.shape
        for value in (
            v_ee_k_full,
            c_ee_k_full,
            chi0_k_full,
            chi_ee_k_full,
            gee_k_full,
        )
    ):
        raise ValueError(
            "Electron response and common interaction channels must share "
            "the ion reciprocal grid."
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
    hij_full = _as_pair_axes(
        ion.get("hij_r", ion.get("hii_r")),
        n_species=n_species,
        name="hij_r",
    )
    cij_full = _as_pair_axes(
        ion.get("cij_r", ion.get("cii_r")),
        n_species=n_species,
        name="cij_r",
    )
    vij_r_full = _as_pair_axes(
        ion.get("vij_r", ion.get("vii_r")),
        n_species=n_species,
        name="vij_r",
    )
    vij_k_full = _as_pair_axes(
        ion.get("vij_k", ion.get("vii_k")),
        n_species=n_species,
        name="vij_k",
    )

    v_ie_r_full = np.asarray(radial_inverse(v_ie_k_full, transform), dtype=float)
    c_ie_r_full = np.asarray(radial_inverse(c_ie_k_full, transform), dtype=float)
    v_ee_r_full = np.asarray(radial_inverse(v_ee_k_full, transform), dtype=float)
    c_ee_r_full = np.asarray(radial_inverse(c_ee_k_full, transform), dtype=float)

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
        "v_ie_k": v_ie_k_full[:, k_mask],
        "v_ei_k": v_ie_k_full[:, k_mask],
        "v_ee_k": v_ee_k_full[k_mask],
        "c_ie_k": c_ie_k_full[:, k_mask],
        "c_ee_k": c_ee_k_full[k_mask],
        "chi0_k": chi0_k_full[k_mask],
        "chi_ee_k": chi_ee_k_full[k_mask],
        "gee_k": gee_k_full[k_mask],
        "g_ee_k": gee_k_full[k_mask],
        "gij_r": gij_full[..., r_mask],
        "sij_k": sij_full[..., k_mask],
        "hij_r": hij_full[..., r_mask],
        "cij_r": cij_full[..., r_mask],
        "vij_r": vij_r_full[..., r_mask],
        "vij_k": vij_k_full[..., k_mask],
        "zbar": _species_vector(
            ion, "zbar", entries=entries, fallback_keys=("zbar_partition", "zbar")
        ),
        "zbar_qoz": _species_vector(
            ion,
            "zbar_qoz",
            entries=entries,
            fallback_keys=("zbar_partition", "zbar"),
        ),
        "zbar_partition": _species_vector(
            ion,
            "zbar_partition",
            entries=entries,
            fallback_keys=("zbar_partition", "zbar"),
        ),
        "zbar_aa_ws": _species_vector(
            ion,
            "zbar_aa_ws",
            entries=entries,
            fallback_keys=("zbar_ws", "zbar"),
        ),
        "n_i_bohr3": np.asarray([_entry_n_i(entry) for entry in entries]),
    }
    arrays.update(
        {
            "v_ie_r": v_ie_r_full[:, r_mask],
            "v_ei_r": v_ie_r_full[:, r_mask],
            "v_ee_r": v_ee_r_full[r_mask],
            "c_ie_r": c_ie_r_full[:, r_mask],
            "c_ee_r": c_ee_r_full[r_mask],
        }
    )

    mu_values = []
    r_ws_values = []
    n0_values = []
    zstar_values = []
    for entry in entries:
        electronic_result = dict(entry["result"])
        n_i_value = _entry_n_i(entry)
        n0_value = float(electronic_result.get("n0", np.nan))
        mu_values.append(float(electronic_result.get("mu", entry.get("mu_ha", np.nan))))
        r_ws_values.append(
            float(electronic_result.get("r_ws", entry.get("r_ws_bohr", np.nan)))
        )
        n0_values.append(n0_value)
        zstar_values.append(n0_value / n_i_value)
    arrays["mu_ha"] = np.asarray(mu_values, dtype=float)
    arrays["r_ws_bohr"] = np.asarray(r_ws_values, dtype=float)
    arrays["n0_bohr3"] = np.asarray(n0_values, dtype=float)
    arrays["zstar"] = np.asarray(zstar_values, dtype=float)

    residual_history = _finite_numeric(ion.get("residual_history"))
    if residual_history is not None:
        arrays["hnc_residual_history"] = np.asarray(
            residual_history, dtype=float
        ).reshape(-1)

    _add_species_electronic_arrays(
        arrays,
        entries=entries,
        r_max_bohr=float(opts.r_max_bohr),
        include_profiles=bool(opts.include_electronic_profiles),
        include_orbital_densities=bool(opts.include_orbital_densities),
    )

    metadata = _metadata(workflow, ion, opts)
    metadata["fields"] = sorted(arrays)
    metadata_text = json.dumps(
        _json_safe(metadata),
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
    schema_version = str(converted["schema_version"].item())
    if schema_version not in _SUPPORTED_STATE_SCHEMA_VERSIONS:
        raise ValueError("Unsupported state schema version.")

    try:
        metadata = json.loads(str(converted["metadata_json"].item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("metadata_json is not valid scalar JSON.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to an object.")
    if metadata.get("schema_version") != schema_version:
        raise ValueError("metadata_json has an unsupported schema version.")
    if schema_version == STATE_SCHEMA_VERSION:
        if not isinstance(metadata.get("configuration"), dict):
            raise ValueError("metadata_json lacks the workflow configuration.")
        if not isinstance(metadata.get("citation_keys"), list):
            raise ValueError("metadata_json lacks the citation-key list.")
        recorded_fields = metadata.get("fields")
        if not isinstance(recorded_fields, list) or set(recorded_fields) != (
            set(converted) - {"metadata_json"}
        ):
            raise ValueError("metadata_json field inventory is inconsistent.")
        interaction_fields = {
            "v_ie_k",
            "v_ei_k",
            "v_ee_k",
            "c_ie_k",
            "c_ee_k",
            "v_ie_r",
            "v_ei_r",
            "v_ee_r",
            "c_ie_r",
            "c_ee_r",
            "chi0_k",
            "chi_ee_k",
            "gee_k",
            "g_ee_k",
            "hij_r",
            "cij_r",
            "vij_r",
            "vij_k",
            "zbar_qoz",
            "zbar_partition",
            "zbar_aa_ws",
            "zstar",
            "mu_ha",
            "r_ws_bohr",
            "n0_bohr3",
        }
        missing_interactions = sorted(interaction_fields.difference(converted))
        if missing_interactions:
            raise ValueError(
                "State payload is missing electron-interaction fields: "
                + ", ".join(missing_interactions)
            )

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
    for key in ("v_ie_k", "v_ei_k", "c_ie_k"):
        if key in converted and converted[key].shape != (n_species, nk):
            raise ValueError(f"{key} shape is inconsistent with species/k grids.")
    for key in (
        "v_ee_k",
        "c_ee_k",
        "chi0_k",
        "chi_ee_k",
        "gee_k",
        "g_ee_k",
    ):
        if key in converted and converted[key].shape != (nk,):
            raise ValueError(f"{key} shape is inconsistent with the k grid.")
    if converted["gij_r"].shape != (n_species, n_species, nr):
        raise ValueError("gij_r shape is inconsistent with species/r grids.")
    if converted["sij_k"].shape != (n_species, n_species, nk):
        raise ValueError("sij_k shape is inconsistent with species/k grids.")
    for key in ("hij_r", "cij_r", "vij_r"):
        if key in converted and converted[key].shape != (
            n_species,
            n_species,
            nr,
        ):
            raise ValueError(f"{key} shape is inconsistent with species/r grids.")
    if "vij_k" in converted and converted["vij_k"].shape != (
        n_species,
        n_species,
        nk,
    ):
        raise ValueError("vij_k shape is inconsistent with species/k grids.")
    for key in ("v_ie_r", "v_ei_r", "c_ie_r"):
        if key in converted and converted[key].shape != (n_species, nr):
            raise ValueError(f"{key} shape is inconsistent with species/r grids.")
    for key in ("v_ee_r", "c_ee_r"):
        if key in converted and converted[key].shape != (nr,):
            raise ValueError(f"{key} shape is inconsistent with the r grid.")
    for key in (
        "zbar",
        "zbar_qoz",
        "zbar_partition",
        "zbar_aa_ws",
        "zstar",
        "mu_ha",
        "r_ws_bohr",
        "n0_bohr3",
        "n_i_bohr3",
    ):
        if key in converted and converted[key].shape != (n_species,):
            raise ValueError(f"{key} shape is inconsistent with the species axis.")
    for key in (
        "species_counts",
        "species_number_fraction",
        "n_ion_r",
        "n_scr_r",
        "f_k",
        "q_k",
        "n_ion_k",
        "n_scr_k",
        "v_ie_k",
        "v_ei_k",
        "v_ee_k",
        "c_ie_k",
        "c_ee_k",
        "chi0_k",
        "chi_ee_k",
        "gee_k",
        "g_ee_k",
        "v_ie_r",
        "v_ei_r",
        "v_ee_r",
        "c_ie_r",
        "c_ee_r",
        "gij_r",
        "sij_k",
        "hij_r",
        "cij_r",
        "vij_r",
        "zbar",
        "zbar_qoz",
        "zbar_partition",
        "zbar_aa_ws",
        "zstar",
        "mu_ha",
        "r_ws_bohr",
        "n0_bohr3",
        "n_i_bohr3",
        "vij_k",
    ):
        if key in converted and np.any(~np.isfinite(converted[key])):
            raise ValueError(f"{key} contains non-finite values.")
    if not np.array_equal(converted["f_k"], converted["n_ion_k"]):
        raise ValueError("f_k must be the explicit alias of n_ion_k.")
    if not np.array_equal(converted["q_k"], converted["n_scr_k"]):
        raise ValueError("q_k must be the explicit alias of n_scr_k.")
    if "v_ei_k" in converted and not np.array_equal(
        converted["v_ie_k"], converted["v_ei_k"]
    ):
        raise ValueError("v_ei_k must be the explicit alias of v_ie_k.")
    if "v_ei_r" in converted and not np.array_equal(
        converted["v_ie_r"], converted["v_ei_r"]
    ):
        raise ValueError("v_ei_r must be the explicit alias of v_ie_r.")
    if "g_ee_k" in converted and not np.array_equal(
        converted["gee_k"], converted["g_ee_k"]
    ):
        raise ValueError("g_ee_k must be the explicit alias of gee_k.")

    for key, value in converted.items():
        if value.dtype.kind in "biufc" and np.any(~np.isfinite(value)):
            raise ValueError(f"{key} contains non-finite values.")

    if schema_version == STATE_SCHEMA_VERSION:
        for species_index in range(n_species):
            prefix = f"species_{species_index}_"
            r_key = prefix + "r_bohr"
            if r_key not in converted:
                raise ValueError(f"State payload is missing {r_key}.")
            r_native = converted[r_key]
            if (
                r_native.ndim != 1
                or r_native.size < 1
                or np.any(r_native <= 0.0)
                or np.any(np.diff(r_native) <= 0.0)
            ):
                raise ValueError(f"{r_key} must be positive and increasing.")
            n_level = int(converted.get(prefix + "bound_energy_ha", np.empty(0)).size)
            for name in ("bound_l", "bound_n_index", "bound_principal_n"):
                key = prefix + name
                if key in converted and converted[key].shape != (n_level,):
                    raise ValueError(f"{key} is not aligned with bound levels.")
            for name in ("bound_orbital_density_r", "ion_orbital_density_r"):
                key = prefix + name
                if key in converted and converted[key].shape != (
                    n_level,
                    r_native.size,
                ):
                    raise ValueError(f"{key} has an inconsistent orbital/r shape.")

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
