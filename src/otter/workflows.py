"""
otter/workflows.py

Unified composition-driven plasma workflow.

This module provides one high-level entry point that accepts only the minimal
plasma state:

  composition + T_e + rho [+ optional T_i]

and dispatches to the existing validated electronic-structure solvers:

  - single species: `FullExternalConfig` / `solve_full_then_external`
  - multicomponent mixture: `MixtureConfig` / `solve_mixture_full_then_ext`

When `T_i` is provided, the workflow continues to the current QOZ/HNC ion
structure stage using the screening densities returned by the electronic solve.
Composition can be specified either explicitly with `elements + counts` or via
one compact chemical `formula` convenience string.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from otter.numerics.constants import EV_TO_HA
from otter.numerics.grids import create_linear_grid
from otter.data.elements import element as element_info
from otter.ionic import (
    QOZPotentialOptions,
    QOZResponseOptions,
    build_effective_vii_from_nscr,
    build_effective_vij_from_nscr,
    enforce_screening_charge_consistency,
    enforce_screening_charge_consistency_many,
    hnc_solver,
    hnc_solver_multicomponent_continuation,
    precompute_dst_lattice_transform_like,
)

from otter.electronic.full_external import FullExternalConfig, solve_full_then_external
from otter.electronic.mixture import MixtureConfig, solve_mixture_full_then_ext


_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)([0-9]*)")


def parse_formula_composition(formula: str) -> tuple[list[str], list[float]]:
    """
    Parse one simple chemical formula into distinct species and counts.

    Parameters
    ----------
    formula
        Chemical formula such as `Al`, `CH2`, or `C10H8O4`.

    Returns
    -------
    list[str], list[float]
        Distinct species symbols and their stoichiometric counts.

    Notes
    -----
    This parser intentionally targets the compact formulas used by the current
    AA/mixture workflows. Parentheses and charge annotations are rejected.
    Repeated symbols are accumulated while preserving first appearance order,
    so structural formulas such as `CH3COOH` become `C2H4O2`.
    """
    text = str(formula).strip()
    if not text:
        raise ValueError("formula must be a non-empty chemical formula string.")
    if "(" in text or ")" in text:
        raise ValueError("formula parser does not support parentheses yet.")

    symbols: list[str] = []
    counts: list[float] = []
    counts_by_symbol: dict[str, float] = {}
    pos = 0
    while pos < len(text):
        match = _FORMULA_TOKEN_RE.match(text, pos)
        if match is None:
            raise ValueError(f"Could not parse formula near position {pos}: {text!r}")
        symbol = str(element_info(match.group(1)).symbol)
        count_txt = str(match.group(2))
        count_val = 1.0 if count_txt == "" else float(int(count_txt))
        if count_val <= 0.0:
            raise ValueError("Formula stoichiometric counts must be positive.")
        if symbol not in counts_by_symbol:
            symbols.append(symbol)
            counts.append(0.0)
            counts_by_symbol[symbol] = 0.0
        counts_by_symbol[symbol] += count_val
        counts[symbols.index(symbol)] = counts_by_symbol[symbol]
        pos = match.end()

    if pos != len(text):
        raise ValueError(f"Failed to consume the full chemical formula: {text!r}")
    return symbols, [float(val) for val in counts]


def _resolve_explicit_composition(
    *,
    elements: list[int | str] | tuple[int | str, ...] | None,
    counts: list[float] | tuple[float, ...] | None,
    number_fraction: list[float] | tuple[float, ...] | None,
) -> tuple[list[str], list[float]]:
    """
    Resolve one explicit composition into canonical symbols and positive counts.

    Notes
    -----
    The outer AA/mixture solvers only care about relative stoichiometric
    weights, so `number_fraction` can be passed through unchanged after
    normalization. `counts` remains the preferred explicit user input because
    it mirrors the chemical formula directly.
    """
    if elements is None:
        raise ValueError("Explicit composition requires elements=[...].")
    symbols = [str(element_info(spec).symbol) for spec in list(elements)]
    if len(symbols) == 0:
        raise ValueError("elements must contain at least one species.")

    if counts is not None and number_fraction is not None:
        raise ValueError("Specify only one of counts or number_fraction.")

    if counts is None and number_fraction is None:
        if len(symbols) == 1:
            return symbols, [1.0]
        raise ValueError(
            "For multicomponent states, provide either counts=[...] or number_fraction=[...]."
        )

    values = counts if counts is not None else number_fraction
    values_arr = np.asarray(values, dtype=float)
    if values_arr.shape != (len(symbols),):
        raise ValueError("Composition weights must have one entry per species.")
    if np.any(values_arr <= 0.0):
        raise ValueError("Composition weights must be strictly positive.")
    if number_fraction is not None:
        values_arr = values_arr / float(np.sum(values_arr))
    return symbols, [float(val) for val in values_arr]


def resolve_plasma_composition(
    *,
    formula: str | None,
    elements: list[int | str] | tuple[int | str, ...] | None,
    counts: list[float] | tuple[float, ...] | None,
    number_fraction: list[float] | tuple[float, ...] | None,
) -> tuple[list[str], list[float]]:
    """
    Resolve one composition specification to canonical symbols and counts.

    Exactly one composition style must be used:

    - `formula="C10H8O4"`
    - `elements=["C", "H", "O"], counts=[10, 8, 4]`
    - `elements=["C", "H", "O"], number_fraction=[10/22, 8/22, 4/22]`
    """
    has_formula = formula is not None and str(formula).strip() != ""
    has_explicit = elements is not None
    if has_formula and has_explicit:
        raise ValueError("Specify either formula or elements-based composition, not both.")
    if has_formula:
        return parse_formula_composition(str(formula))
    return _resolve_explicit_composition(
        elements=elements,
        counts=counts,
        number_fraction=number_fraction,
    )


def _interp_profile_linear(
    *,
    r_src: np.ndarray,
    y_src: np.ndarray,
    r_dst: np.ndarray,
    right_value: float,
) -> np.ndarray:
    """Interpolate one radial profile onto one shared linear grid."""
    r_src_arr = np.asarray(r_src, dtype=float)
    y_src_arr = np.asarray(y_src, dtype=float)
    r_dst_arr = np.asarray(r_dst, dtype=float)
    if r_src_arr.ndim != 1 or y_src_arr.ndim != 1 or r_src_arr.size != y_src_arr.size:
        raise ValueError("Expected one-dimensional radial profiles.")
    if r_src_arr.size == 0:
        return np.zeros_like(r_dst_arr)
    return np.interp(
        r_dst_arr,
        r_src_arr,
        y_src_arr,
        left=float(y_src_arr[0]),
        right=float(right_value),
    )


def _build_qoz_linear_grid(
    *,
    r_max: float,
    n_linear: int,
    pad_factor: float,
) -> np.ndarray:
    """Build one strict linear grid, optionally padded to reduce `k_min`."""
    lin = create_linear_grid(float(r_max), int(n_linear))
    r_lin = np.asarray(lin.r, dtype=float)
    pad = max(float(pad_factor), 1.0)
    if pad <= 1.0 + 1.0e-12:
        return r_lin
    dr = float(lin.dr)
    n_pad = int(np.round((pad * float(r_lin[-1])) / dr))
    n_pad = max(n_pad, r_lin.size)
    return dr * np.arange(1, n_pad + 1, dtype=float)


def _species_parallel_jobs_default(cfg: "PlasmaWorkflowConfig", n_species: int) -> int:
    """Resolve the species-level parallel worker count."""
    if cfg.species_parallel_jobs is not None:
        return int(cfg.species_parallel_jobs)
    return max(int(n_species), 1) if int(n_species) > 1 else 1


def _single_species_entry(
    *,
    symbol: str,
    count: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Wrap one single-species full+ext result in the mixture-like payload shape."""
    elem = element_info(symbol)
    r_ws_bohr = float(result.get("r_ws", result.get("meta", {}).get("r_ws", np.nan)))
    if not np.isfinite(r_ws_bohr) or r_ws_bohr <= 0.0:
        raise ValueError("Single-species workflow requires a finite positive r_ws.")
    volume_bohr3 = float(4.0 * np.pi * r_ws_bohr**3 / 3.0)
    return {
        "element": str(elem.symbol),
        "Z": int(elem.z),
        "atomic_mass": float(elem.atomic_mass),
        "count": float(count),
        "x": 1.0,
        "volume_bohr3": float(volume_bohr3),
        "r_ws_bohr": float(r_ws_bohr),
        "mu_ha": float(result["mu"]),
        "result": dict(result),
    }


def _species_entries_from_electronic(
    *,
    symbols: list[str],
    counts: list[float],
    electronic_kind: str,
    electronic_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalize single-species and mixture outputs to one shared species list."""
    if str(electronic_kind) == "single_species":
        return [
            _single_species_entry(
                symbol=str(symbols[0]),
                count=float(counts[0]),
                result=dict(electronic_result),
            )
        ]
    return [{**dict(sp), "result": dict(sp["result"])} for sp in electronic_result["species"]]


@dataclass
class PlasmaWorkflowConfig:
    """
    Unified high-level plasma workflow configuration.

    Parameters
    ----------
    formula
        Optional chemical formula such as `Al`, `CH2`, or `C10H8O4`.
    elements, counts
        Preferred explicit composition input. For example,
        `elements=["C", "H"]`, `counts=[1, 2]` represents `CH2`.
        When `elements` contains exactly one species, `counts` may be omitted.
    number_fraction
        Optional alternative explicit composition input for non-integer
        mixtures. Only one of `counts` or `number_fraction` may be provided.
    temperature_ev
        Electron temperature in eV.
    rho_g_cc
        Mass density in g/cc.
    ion_temperature_ev
        Optional ion temperature in eV. When provided, the workflow continues
        from AA/full+ext to the QOZ/HNC ion-structure stage.
    aa_overrides
        Optional single-AA controls. These are forwarded unchanged; the
        workflow does not modify AA defaults unless the user explicitly
        requests it here.
    species_overrides
        Optional per-species AA overrides for mixture runs.
    """

    temperature_ev: float
    rho_g_cc: float
    formula: str | None = None
    elements: list[int | str] | tuple[int | str, ...] | None = None
    counts: list[float] | tuple[float, ...] | None = None
    number_fraction: list[float] | tuple[float, ...] | None = None
    ion_temperature_ev: float | None = None

    aa_overrides: dict[str, Any] = field(default_factory=dict)
    species_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    run_mode: str = "full+ext"
    mu_e_tol: float = 1.0e-4
    root_tol: float = 1.0e-4
    root_maxfev: int = 20
    volume_weights_init: list[float] | tuple[float, ...] | None = None
    species_parallel_jobs: int | None = None
    species_parallel_backend: str = "thread"

    show_progress: bool = False
    show_mu_progress: bool = False
    verbose: bool = False

    save_data: bool = False
    save_output_dir: str | Path = "outputs"
    save_suffix: str = ""
    save_common_linear_grid: bool = True
    save_linear_n_points: int = 4096

    qoz_linear_n_points: int = 4096
    qoz_pad_factor: float = 2.0
    qoz_renormalize_nscr_to_zbar: bool = True
    qoz_response_chi0_model: str = "lindhard_fd"
    qoz_response_lfc_model: str = "gregori2007"
    qoz_high_k_taper_start_frac: float | None = 0.9

    hnc_mix: float = 0.03
    hnc_tol: float = 1.0e-4
    hnc_max_iter: int = 160
    hnc_mixing_scheme: str = "anderson"
    hnc_tail_points: int = 32
    hnc_potential_scales: tuple[float, ...] = (0.35, 0.6, 0.8, 1.0)

    def __post_init__(self) -> None:
        resolve_plasma_composition(
            formula=self.formula,
            elements=self.elements,
            counts=self.counts,
            number_fraction=self.number_fraction,
        )
        if float(self.temperature_ev) < 0.0:
            raise ValueError("temperature_ev must be non-negative.")
        if float(self.rho_g_cc) <= 0.0:
            raise ValueError("rho_g_cc must be positive.")
        if self.ion_temperature_ev is not None and float(self.ion_temperature_ev) < 0.0:
            raise ValueError("ion_temperature_ev must be non-negative when provided.")
        if int(self.qoz_linear_n_points) < 32:
            raise ValueError("qoz_linear_n_points must be at least 32.")
        if float(self.qoz_pad_factor) < 1.0:
            raise ValueError("qoz_pad_factor must be >= 1.")
        if self.qoz_high_k_taper_start_frac is not None and not (
            0.0 < float(self.qoz_high_k_taper_start_frac) < 1.0
        ):
            raise ValueError("qoz_high_k_taper_start_frac must lie in (0, 1) when provided.")
        if float(self.hnc_tol) <= 0.0:
            raise ValueError("hnc_tol must be positive.")
        if int(self.hnc_max_iter) < 4:
            raise ValueError("hnc_max_iter must be at least 4.")
        if str(self.run_mode).strip().lower() not in ("full", "full+ext", "full_ext"):
            raise ValueError("run_mode must be 'full' or 'full+ext'.")
        if self.ion_temperature_ev is not None and str(self.run_mode).strip().lower() == "full":
            raise ValueError("Ion-structure workflow requires run_mode='full+ext' to provide n_scr.")


def _solve_electronic_structure(
    cfg: PlasmaWorkflowConfig,
    *,
    symbols: list[str],
    counts: list[float],
) -> tuple[str, dict[str, Any]]:
    """Dispatch one unified formula state to single-AA or mixture AA."""
    if len(symbols) == 1:
        cfg_single = FullExternalConfig(
            element=str(symbols[0]),
            temperature_ev=float(cfg.temperature_ev),
            rho_g_cc=float(cfg.rho_g_cc),
            run_mode=str(cfg.run_mode),
            show_scf_progress=bool(cfg.show_progress),
            verbose=bool(cfg.verbose),
            save_data=bool(cfg.save_data),
            save_output_dir=cfg.save_output_dir,
            save_suffix=str(cfg.save_suffix),
            **dict(cfg.aa_overrides),
        )
        return "single_species", solve_full_then_external(cfg_single)

    cfg_mix = MixtureConfig(
        species=list(symbols),
        counts=[float(val) for val in counts],
        temperature_ev=float(cfg.temperature_ev),
        rho_g_cc=float(cfg.rho_g_cc),
        aa_overrides=dict(cfg.aa_overrides),
        species_overrides={key: dict(val) for key, val in cfg.species_overrides.items()},
        mu_e_tol=float(cfg.mu_e_tol),
        root_tol=float(cfg.root_tol),
        root_maxfev=int(cfg.root_maxfev),
        volume_weights_init=(
            None if cfg.volume_weights_init is None else [float(val) for val in cfg.volume_weights_init]
        ),
        show_progress=bool(cfg.show_progress),
        show_mu_progress=bool(cfg.show_mu_progress),
        verbose=bool(cfg.verbose),
        final_run_mode=str(cfg.run_mode),
        species_parallel_jobs=_species_parallel_jobs_default(cfg, len(symbols)),
        species_parallel_backend=str(cfg.species_parallel_backend),
        save_data=bool(cfg.save_data),
        save_output_dir=cfg.save_output_dir,
        save_suffix=str(cfg.save_suffix),
        save_common_linear_grid=bool(cfg.save_common_linear_grid),
        save_linear_n_points=int(cfg.save_linear_n_points),
    )
    return "mixture", solve_mixture_full_then_ext(cfg_mix)


def _one_component_ion_structure(
    cfg: PlasmaWorkflowConfig,
    *,
    species_entry: dict[str, Any],
) -> dict[str, Any]:
    """Run one-component QOZ/HNC from one single-species AA/full+ext result."""
    final = dict(species_entry["result"])
    r_src = np.asarray(final["r"], dtype=float)
    r_work = _build_qoz_linear_grid(
        r_max=float(r_src[-1]),
        n_linear=int(cfg.qoz_linear_n_points),
        pad_factor=float(cfg.qoz_pad_factor),
    )
    transform = precompute_dst_lattice_transform_like(r_work)
    r = np.asarray(transform.r, dtype=float)
    k = np.asarray(transform.k, dtype=float)
    n_scr = _interp_profile_linear(
        r_src=r_src,
        y_src=np.asarray(final["n_scr"], dtype=float),
        r_dst=r,
        right_value=0.0,
    )
    charge_fix = enforce_screening_charge_consistency(
        r,
        n_scr,
        zbar=float(final["zbar"]),
        renormalize=bool(cfg.qoz_renormalize_nscr_to_zbar),
    )
    ion_temperature_ha = float(cfg.ion_temperature_ev) * EV_TO_HA
    t_qoz = time.perf_counter()
    qoz = build_effective_vii_from_nscr(
        r=r,
        n_scr=charge_fix.n_scr_r,
        zbar=float(final["zbar"]),
        n_i=float(1.0 / float(species_entry["volume_bohr3"])),
        ion_temperature_ha=ion_temperature_ha,
        k=k,
        transform=transform,
        options=QOZPotentialOptions(
            response=QOZResponseOptions(
                chi0_model=str(cfg.qoz_response_chi0_model),
                lfc_model=str(cfg.qoz_response_lfc_model),
                electron_temperature_ha=float(cfg.temperature_ev) * EV_TO_HA,
            ),
            high_k_taper_start_frac=(
                None if cfg.qoz_high_k_taper_start_frac is None else float(cfg.qoz_high_k_taper_start_frac)
            ),
        ),
    )
    qoz_build_s = time.perf_counter() - t_qoz
    t_hnc = time.perf_counter()
    g_r, s_k, h_r, c_r, residual_history = hnc_solver(
        r,
        k,
        qoz.vii_r,
        transform,
        float(1.0 / float(species_entry["volume_bohr3"])),
        ion_temperature_ha,
        mix=float(cfg.hnc_mix),
        tol=float(cfg.hnc_tol),
        max_iter=int(cfg.hnc_max_iter),
        mixing_scheme=str(cfg.hnc_mixing_scheme),
        tail_points=int(cfg.hnc_tail_points),
    )
    hnc_solve_s = time.perf_counter() - t_hnc
    gij_r = np.asarray(g_r, dtype=float)[None, None, :]
    sij_k = np.asarray(s_k, dtype=float)[None, None, :]
    hij_r = np.asarray(h_r, dtype=float)[None, None, :]
    cij_r = np.asarray(c_r, dtype=float)[None, None, :]
    vij_r = np.asarray(qoz.vii_r, dtype=float)[None, None, :]
    vij_k = np.asarray(qoz.vii_k, dtype=float)[None, None, :]
    return {
        "kind": "one_component",
        "species": [str(species_entry["element"])],
        "r": r,
        "k": k,
        "n_scr_r": np.asarray(charge_fix.n_scr_r, dtype=float),
        "n_scr_k": np.asarray(qoz.n_scr_k, dtype=float),
        "zbar": float(final["zbar"]),
        "n_i": float(1.0 / float(species_entry["volume_bohr3"])),
        "gii_r": np.asarray(g_r, dtype=float),
        "sii_k": np.asarray(s_k, dtype=float),
        "gij_r": gij_r,
        "sij_k": sij_k,
        "hii_r": np.asarray(h_r, dtype=float),
        "cii_r": np.asarray(c_r, dtype=float),
        "hij_r": hij_r,
        "cij_r": cij_r,
        "vii_r": np.asarray(qoz.vii_r, dtype=float),
        "vii_k": np.asarray(qoz.vii_k, dtype=float),
        "vij_r": vij_r,
        "vij_k": vij_k,
        "chi_ee_k": np.asarray(qoz.chi_ee_k, dtype=float),
        "chi0_k": np.asarray(qoz.chi0_k, dtype=float),
        "gee_k": np.asarray(qoz.gee_k, dtype=float),
        "qoz_build_s": float(qoz_build_s),
        "hnc_solve_s": float(hnc_solve_s),
        "hnc_iters": int(len(residual_history)),
        "ion_total_s": float(qoz_build_s + hnc_solve_s),
        "residual_history": list(residual_history),
        "charge_fix": {
            "q_scr_raw": float(charge_fix.q_scr_raw),
            "q_scr_used": float(charge_fix.q_scr_used),
            "q_scr_rel": float(charge_fix.q_scr_rel),
        },
    }


def _multicomponent_ion_structure(
    cfg: PlasmaWorkflowConfig,
    *,
    species_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run multicomponent QOZ/HNC from one mixture full+ext result."""
    common_rmax = max(float(np.asarray(sp["result"]["r"], dtype=float)[-1]) for sp in species_entries)
    r_work = _build_qoz_linear_grid(
        r_max=float(common_rmax),
        n_linear=int(cfg.qoz_linear_n_points),
        pad_factor=float(cfg.qoz_pad_factor),
    )
    transform = precompute_dst_lattice_transform_like(r_work)
    r = np.asarray(transform.r, dtype=float)
    k = np.asarray(transform.k, dtype=float)

    n_species = len(species_entries)
    n_scr = np.zeros((n_species, r.size), dtype=float)
    zbar = np.zeros(n_species, dtype=float)
    n_i = np.zeros(n_species, dtype=float)
    x = np.asarray([float(sp["x"]) for sp in species_entries], dtype=float)
    vbar_bohr3 = float(
        np.sum(x * np.asarray([float(sp["volume_bohr3"]) for sp in species_entries], dtype=float))
    )
    n_mix = 1.0 / max(float(vbar_bohr3), 1.0e-300)
    symbols = [str(sp["element"]) for sp in species_entries]
    for idx, sp in enumerate(species_entries):
        final = dict(sp["result"])
        n_scr[idx] = _interp_profile_linear(
            r_src=np.asarray(final["r"], dtype=float),
            y_src=np.asarray(final["n_scr"], dtype=float),
            r_dst=r,
            right_value=0.0,
        )
        zbar[idx] = float(final["zbar"])
        n_i[idx] = float(x[idx] * n_mix)

    charge_fix = enforce_screening_charge_consistency_many(
        r,
        n_scr,
        zbar=zbar,
        renormalize=bool(cfg.qoz_renormalize_nscr_to_zbar),
    )
    ion_temperature_ha = float(cfg.ion_temperature_ev) * EV_TO_HA
    t_qoz = time.perf_counter()
    qoz = build_effective_vij_from_nscr(
        r=r,
        n_scr=charge_fix.n_scr_r,
        zbar=zbar,
        n_i=n_i,
        ion_temperature_ha=ion_temperature_ha,
        k=k,
        transform=transform,
        options=QOZPotentialOptions(
            response=QOZResponseOptions(
                chi0_model=str(cfg.qoz_response_chi0_model),
                lfc_model=str(cfg.qoz_response_lfc_model),
                electron_temperature_ha=float(cfg.temperature_ev) * EV_TO_HA,
            ),
            high_k_taper_start_frac=(
                None if cfg.qoz_high_k_taper_start_frac is None else float(cfg.qoz_high_k_taper_start_frac)
            ),
        ),
    )
    qoz_build_s = time.perf_counter() - t_qoz
    t_hnc = time.perf_counter()
    g_r, s_k, h_r, c_r, residual_history, stage_meta = hnc_solver_multicomponent_continuation(
        r,
        k,
        qoz.vij_r,
        transform,
        n_i,
        ion_temperature_ha,
        potential_scales=tuple(float(val) for val in cfg.hnc_potential_scales),
        mix=float(cfg.hnc_mix),
        tol=float(cfg.hnc_tol),
        max_iter=int(cfg.hnc_max_iter),
        mixing_scheme=str(cfg.hnc_mixing_scheme),
        tail_points=int(cfg.hnc_tail_points),
        s_min_floor=1.0e-8,
        s_max_ceil=1.0e6,
    )
    hnc_solve_s = time.perf_counter() - t_hnc
    gii = np.asarray([g_r[idx, idx] for idx in range(n_species)], dtype=float)
    sii = np.asarray([s_k[idx, idx] for idx in range(n_species)], dtype=float)
    return {
        "kind": "multicomponent",
        "species": symbols,
        "r": r,
        "k": k,
        "n_scr_r": np.asarray(charge_fix.n_scr_r, dtype=float),
        "n_scr_k": np.asarray(qoz.n_scr_k, dtype=float),
        "zbar": np.asarray(zbar, dtype=float),
        "n_i": np.asarray(n_i, dtype=float),
        "gii_r": gii,
        "sii_k": sii,
        "gij_r": np.asarray(g_r, dtype=float),
        "sij_k": np.asarray(s_k, dtype=float),
        "hij_r": np.asarray(h_r, dtype=float),
        "cij_r": np.asarray(c_r, dtype=float),
        "vij_r": np.asarray(qoz.vij_r, dtype=float),
        "vij_k": np.asarray(qoz.vij_k, dtype=float),
        "chi_ee_k": np.asarray(qoz.chi_ee_k, dtype=float),
        "chi0_k": np.asarray(qoz.chi0_k, dtype=float),
        "gee_k": np.asarray(qoz.gee_k, dtype=float),
        "qoz_build_s": float(qoz_build_s),
        "hnc_solve_s": float(hnc_solve_s),
        "hnc_iters": int(len(residual_history)),
        "ion_total_s": float(qoz_build_s + hnc_solve_s),
        "residual_history": list(residual_history),
        "stage_meta": [dict(stage) for stage in stage_meta],
        "charge_fix": {
            "q_scr_raw": np.asarray(charge_fix.q_scr_raw, dtype=float),
            "q_scr_used": np.asarray(charge_fix.q_scr_used, dtype=float),
            "q_scr_rel": np.asarray(charge_fix.q_scr_rel, dtype=float),
        },
    }


def solve_plasma_workflow(cfg: PlasmaWorkflowConfig) -> dict[str, Any]:
    """
    Solve one unified plasma workflow from one composition specification.

    Parameters
    ----------
    cfg
        Unified workflow configuration.

    Returns
    -------
    dict
        Top-level payload containing parsed composition, electronic structure,
        and optional ion-structure outputs.
    """
    symbols, counts = resolve_plasma_composition(
        formula=cfg.formula,
        elements=cfg.elements,
        counts=cfg.counts,
        number_fraction=cfg.number_fraction,
    )
    electronic_kind, electronic_result = _solve_electronic_structure(
        cfg,
        symbols=symbols,
        counts=counts,
    )
    return continue_plasma_workflow_from_electronic_result(
        cfg,
        electronic_kind=electronic_kind,
        electronic_result=electronic_result,
    )


def continue_plasma_workflow_from_electronic_result(
    cfg: PlasmaWorkflowConfig,
    *,
    electronic_kind: str,
    electronic_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Continue the unified workflow from an already available electronic result.

    Parameters
    ----------
    cfg
        Unified workflow configuration. The composition and thermodynamic state
        must match the supplied electronic payload.
    electronic_kind
        Either `"single_species"` or `"mixture"`.
    electronic_result
        Previously solved electronic payload in the same shape returned by the
        workflow electronic stage.

    Returns
    -------
    dict
        Top-level workflow payload with optional ion-structure results.

    Notes
    -----
    This helper exists so expensive AA/full+ext results can be cached and then
    reused while iterating only on the downstream QOZ/HNC controls.
    """
    symbols, counts = resolve_plasma_composition(
        formula=cfg.formula,
        elements=cfg.elements,
        counts=cfg.counts,
        number_fraction=cfg.number_fraction,
    )
    species_entries = _species_entries_from_electronic(
        symbols=symbols,
        counts=counts,
        electronic_kind=electronic_kind,
        electronic_result=electronic_result,
    )

    ion_result: dict[str, Any] | None = None
    if cfg.ion_temperature_ev is not None:
        if len(species_entries) == 1:
            ion_result = _one_component_ion_structure(
                cfg,
                species_entry=dict(species_entries[0]),
            )
        else:
            ion_result = _multicomponent_ion_structure(
                cfg,
                species_entries=[dict(sp) for sp in species_entries],
            )

    result = {
        "formula": (None if cfg.formula is None else str(cfg.formula)),
        "species_symbols": list(symbols),
        "species_counts": [float(val) for val in counts],
        "temperature_ev": float(cfg.temperature_ev),
        "rho_g_cc": float(cfg.rho_g_cc),
        "ion_temperature_ev": (
            None if cfg.ion_temperature_ev is None else float(cfg.ion_temperature_ev)
        ),
        "electronic": {
            "kind": str(electronic_kind),
            "result": electronic_result,
        },
        "ion": ion_result,
    }
    saved_paths = electronic_result.get("saved_paths", None)
    if saved_paths is not None:
        result["saved_paths"] = dict(saved_paths)
    return result


def run_formula_workflow(
    formula: str,
    temperature_ev: float,
    rho_g_cc: float,
    *,
    ion_temperature_ev: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience wrapper around `PlasmaWorkflowConfig` + `solve_plasma_workflow`."""
    cfg = PlasmaWorkflowConfig(
        formula=str(formula),
        temperature_ev=float(temperature_ev),
        rho_g_cc=float(rho_g_cc),
        ion_temperature_ev=(
            None if ion_temperature_ev is None else float(ion_temperature_ev)
        ),
        **kwargs,
    )
    return solve_plasma_workflow(cfg)


__all__ = [
    "PlasmaWorkflowConfig",
    "continue_plasma_workflow_from_electronic_result",
    "parse_formula_composition",
    "resolve_plasma_composition",
    "run_formula_workflow",
    "solve_plasma_workflow",
]
