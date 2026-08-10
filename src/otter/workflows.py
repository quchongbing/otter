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

The single-component pseudoatom/QOZ construction follows C. E. Starrett and
D. Saumon, High Energy Density Physics 10, 35--42 (2014),
DOI 10.1016/j.hedp.2013.12.001.  The shared-chemical-potential mixture and
effective-potential construction follows C. E. Starrett et al., Physical
Review E 90, 033110 (2014), DOI 10.1103/PhysRevE.90.033110.  Workflow
dispatch, validation gates, caching, and portable state export are Otter
software conventions.
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
from otter.literature import (
    CitationMixin,
    citation_keys_for_chi0_model,
    citation_keys_for_lfc_model,
    citation_keys_for_xc_model,
)
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
    qoz_zbar_from_nscr,
    radial_charge_trapezoid,
    radial_forward,
)

from otter.electronic.full_external import FullExternalConfig, solve_full_then_external
from otter.electronic.mixture import (
    MixtureConfig,
    _species_result_eligibility,
    solve_mixture_full_then_ext,
)


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


def _electronic_convergence_issues(
    species_entries: list[dict[str, Any]],
    *,
    require_external: bool = False,
) -> list[str]:
    """Describe unsafe final AA/full+external stages before ion structure."""
    issues: list[str] = []
    for entry in species_entries:
        symbol = str(entry.get("element", "?"))
        result = dict(entry.get("result", {}))
        _, reasons = _species_result_eligibility(
            result,
            require_external=bool(require_external),
        )
        reason_messages = {
            "stage2_unconverged": "full AA stage-2 unconverged",
            "mu_nonfinite": "non-finite electron chemical potential",
            "threshold_state_unresolved": "unresolved threshold bound state",
            "b3_full_tail_unapplied": (
                "requested full-density B3 tail was not applied"
            ),
            "b3_external_tail_unapplied": (
                "requested external-density B3 tail was not applied"
            ),
            "external_status_missing": "external fixed-mu SCF status missing",
            "external_disabled": "external fixed-mu SCF disabled",
            "external_unconverged": "external fixed-mu SCF unconverged",
            "b3_post_diagnostic_only": (
                "post-SCF B3 is diagnostic/non-self-consistent and is not a production QOZ input"
            ),
        }
        issues.extend(
            f"{symbol}: {reason_messages.get(reason, reason)}" for reason in reasons
        )

        # Keep reporting an explicitly active failed external stage to callers
        # that use this diagnostic without the stricter QOZ requirement.
        if not require_external:
            ext_raw = result.get("ext_status", None)
            if isinstance(ext_raw, dict) and ext_raw:
                ext_status = dict(ext_raw)
                ext_active = bool(ext_status.get("enabled", True))
                if ext_active and "converged" in ext_status and not bool(ext_status["converged"]):
                    issues.append(f"{symbol}: external fixed-mu SCF unconverged")
    return issues


@dataclass
class PlasmaWorkflowConfig(CitationMixin):
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
    allow_unconverged_root
        Permit explicit diagnostic best-effort mixture output. By default an
        unconverged common-mu state is rejected before QOZ/HNC.
    allow_unconverged_aa
        Permit explicit diagnostic ion-structure continuation from a failed
        final full or external AA stage. The production default rejects it.
    """

    temperature_ev: float
    rho_g_cc: float
    formula: str | None = None
    elements: list[int | str] | tuple[int | str, ...] | None = None
    counts: list[float] | tuple[float, ...] | None = None
    number_fraction: list[float] | tuple[float, ...] | None = None
    ion_temperature_ev: float | None = None
    electronic_model: str = "qm"
    # ``qm``: orbital KS average atom; ``tf``: finite-temperature
    # Thomas--Fermi full/external pseudoatom (Starrett--Saumon 2014).
    xc_model: str = "dirac"
    # dirac is dependency-free; pbe and libxc:<names> require pylibxc.
    gga_core_mode: str = "finite"
    gga_core_zr: float = 0.05

    aa_overrides: dict[str, Any] = field(default_factory=dict)
    species_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    run_mode: str = "full+ext"
    mu_e_tol: float = 1.0e-4
    root_tol: float = 1.0e-4
    root_maxfev: int = 20
    root_brent_maxiter: int = 16
    root_threshold_b3_surrogate_mode: str = "a_only_when_full_unresolved"
    allow_unconverged_root: bool = False
    allow_unconverged_aa: bool = False
    volume_weights_init: list[float] | tuple[float, ...] | None = None
    species_parallel_jobs: int | None = None
    species_parallel_backend: str = "thread"

    show_progress: bool = False
    show_mu_progress: bool = False
    verbose: bool = False

    save_data: bool = False
    save_output_dir: str | Path = "outputs"
    save_suffix: str = ""
    save_state_npz: bool = False
    # Save the portable q(k), f(k), gij(r), Sij(k) state contract after a
    # successful ion-structure calculation.
    save_state_path: str | Path | None = None
    state_r_max_bohr: float = 20.0
    state_k_max_bohr_inv: float = 20.0
    save_common_linear_grid: bool = True
    save_linear_n_points: int = 4096

    qoz_linear_n_points: int = 4096
    qoz_pad_factor: float = 2.0
    qoz_renormalize_nscr_to_zbar: bool = True
    # Enforce the neutral-pseudoatom identities of Starrett & Saumon before
    # closing Eq. (15) on the actual DST lattice.  "screening_integral" keeps
    # the raw finite-box integral for diagnostics; "electronic" is legacy WS.
    qoz_zbar_mode: str = "pseudoatom_partition"
    qoz_response_chi0_model: str = "lindhard_fd"
    # Chabrier's finite-temperature jellium extension is the production
    # default validated against Starrett's CH1.36 Fig. 3 benchmark. Simpler
    # Hubbard and other LFC branches remain explicit diagnostic choices.
    qoz_response_lfc_model: str = "chabrier1990"
    qoz_high_k_taper_start_frac: float | None = 0.9

    hnc_mix: float = 0.03
    hnc_tol: float = 1.0e-4
    hnc_closure_transform_tol: float | None = None
    # Independent acceptance limit for
    # |S(k)-[1+n_i FT(g(r)-1)]| on the finite DST lattice.  None preserves
    # the historical max(10*hnc_tol, 1e-6) rule.  Keeping this separate from
    # hnc_tol lets a strongly coupled finite-box calculation demand a tight
    # nonlinear fixed-point residual without stopping early merely to widen
    # the transform-consistency allowance.
    hnc_max_iter: int = 160
    # "auto" first uses the mature direct Anderson one-component solver and
    # falls back to raw potential-strength continuation/Newton only if its
    # strict physical audit fails. Mixtures use matrix-free Newton directly.
    hnc_mixing_scheme: str = "auto"
    hnc_tail_points: int = 32
    # A hard clip on N=h-c changes the closure fixed point.  Keep the low-level
    # legacy guard available, but solve the unmodified equations in production.
    hnc_nodal_clip: float = 0.0
    # Production multicomponent calculations must solve the original OZ/HNC
    # equations.  Eigenvalue clipping is retained only as an explicitly
    # selected diagnostic aid in the low-level solver; it changes the fixed
    # point and can otherwise turn a finite-k instability into a plausible-
    # looking but nonphysical S_ij(k).
    hnc_s_projection_mode: str = "none"
    hnc_fallback_mixing_scheme: str | None = "newton_krylov"
    hnc_newton_max_iter: int = 60
    hnc_adaptive_continuation: bool = True
    hnc_min_scale_step: float = 2.0e-3
    hnc_max_stage_attempts: int = 32
    hnc_require_converged: bool = True
    # The legacy solver option subtracts a constant from the nodal N=h-c
    # tail, despite its historical name `enforce_h_tail_zero`.  The finite-box
    # shift slightly breaks S_ij = delta_ij + sqrt(n_i n_j) FT[g_ij-1], so the
    # high-level production path leaves it disabled and reports that identity.
    hnc_enforce_nodal_tail_zero: bool = False
    hnc_potential_scales: tuple[float, ...] = (0.05, 0.15, 0.35, 0.6, 0.8, 1.0)

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
        model_key = str(self.electronic_model).strip().lower().replace("-", "_")
        if model_key in {"ks", "ks_aa", "quantum"}:
            model_key = "qm"
        elif model_key in {"thomas_fermi", "thomasfermi"}:
            model_key = "tf"
        if model_key not in {"qm", "tf"}:
            raise ValueError("electronic_model must be 'qm' or 'tf'.")
        self.electronic_model = model_key
        self.gga_core_mode = str(self.gga_core_mode).strip().lower()
        if self.gga_core_mode not in {"finite", "strict"}:
            raise ValueError("gga_core_mode must be 'finite' or 'strict'.")
        if (
            not np.isfinite(float(self.gga_core_zr))
            or float(self.gga_core_zr) <= 0.0
        ):
            raise ValueError("gga_core_zr must be finite and positive.")
        if float(self.mu_e_tol) <= 0.0 or float(self.root_tol) <= 0.0:
            raise ValueError("mu_e_tol and root_tol must be positive.")
        if (
            self.hnc_closure_transform_tol is not None
            and float(self.hnc_closure_transform_tol) <= 0.0
        ):
            raise ValueError("hnc_closure_transform_tol must be positive.")
        if int(self.root_maxfev) < 4:
            raise ValueError("root_maxfev must be at least 4.")
        if int(self.root_brent_maxiter) < 1:
            raise ValueError("root_brent_maxiter must be at least 1.")
        if str(self.root_threshold_b3_surrogate_mode).strip().lower() not in (
            "off",
            "a_only_when_full_unresolved",
        ):
            raise ValueError(
                "root_threshold_b3_surrogate_mode must be 'off' or "
                "'a_only_when_full_unresolved'."
            )
        if self.ion_temperature_ev is not None and float(self.ion_temperature_ev) < 0.0:
            raise ValueError("ion_temperature_ev must be non-negative when provided.")
        if bool(self.save_state_npz) and self.ion_temperature_ev is None:
            raise ValueError(
                "save_state_npz requires ion_temperature_ev so q/f/g/S share "
                "the converged QOZ/HNC lattice."
            )
        if float(self.state_r_max_bohr) <= 0.0:
            raise ValueError("state_r_max_bohr must be positive.")
        if float(self.state_k_max_bohr_inv) <= 0.0:
            raise ValueError("state_k_max_bohr_inv must be positive.")
        if int(self.qoz_linear_n_points) < 32:
            raise ValueError("qoz_linear_n_points must be at least 32.")
        if float(self.qoz_pad_factor) < 1.0:
            raise ValueError("qoz_pad_factor must be >= 1.")
        qoz_zbar_mode_key = str(self.qoz_zbar_mode).strip().lower().replace("-", "_")
        if qoz_zbar_mode_key not in {
            "pseudoatom_partition",
            "partition",
            "pa_partition",
            "z_minus_qion_all",
            "screening_integral",
            "nscr_integral",
            "eq15_raw",
            "electronic",
            "aa",
            "legacy",
        }:
            raise ValueError(
                "qoz_zbar_mode must be 'pseudoatom_partition', "
                "'screening_integral', or 'electronic'."
            )
        if qoz_zbar_mode_key in {
            "pseudoatom_partition", "partition", "pa_partition", "z_minus_qion_all"
        } and not bool(self.qoz_renormalize_nscr_to_zbar):
            raise ValueError(
                "pseudoatom_partition requires qoz_renormalize_nscr_to_zbar=True "
                "so Eq. (15) is closed on the QOZ lattice."
            )
        if self.qoz_high_k_taper_start_frac is not None and not (
            0.0 < float(self.qoz_high_k_taper_start_frac) < 1.0
        ):
            raise ValueError("qoz_high_k_taper_start_frac must lie in (0, 1) when provided.")
        if float(self.hnc_tol) <= 0.0:
            raise ValueError("hnc_tol must be positive.")
        if int(self.hnc_max_iter) < 4:
            raise ValueError("hnc_max_iter must be at least 4.")
        if str(self.hnc_mixing_scheme).strip().lower() not in {
            "auto",
            "picard",
            "anderson",
            "newton_krylov",
        }:
            raise ValueError(
                "hnc_mixing_scheme must be 'auto', 'picard', 'anderson', or "
                "'newton_krylov'."
            )
        if str(self.hnc_s_projection_mode).strip().lower().replace("-", "_") not in {
            "none",
            "raw",
            "clip",
        }:
            raise ValueError("hnc_s_projection_mode must be 'none' or 'clip'.")
        if float(self.hnc_min_scale_step) <= 0.0:
            raise ValueError("hnc_min_scale_step must be positive.")
        if float(self.hnc_nodal_clip) < 0.0:
            raise ValueError("hnc_nodal_clip must be non-negative.")
        if int(self.hnc_max_stage_attempts) < len(tuple(self.hnc_potential_scales)):
            raise ValueError(
                "hnc_max_stage_attempts must be at least len(hnc_potential_scales)."
            )
        if int(self.hnc_newton_max_iter) < 2:
            raise ValueError("hnc_newton_max_iter must be at least 2.")
        if str(self.run_mode).strip().lower() not in ("full", "full+ext", "full_ext"):
            raise ValueError("run_mode must be 'full' or 'full+ext'.")
        if self.ion_temperature_ev is not None and str(self.run_mode).strip().lower() == "full":
            raise ValueError("Ion-structure workflow requires run_mode='full+ext' to provide n_scr.")

    @property
    def citation_keys(self) -> tuple[str, ...]:
        """Primary papers for the selected electronic and QOZ models."""
        keys = [
            "StarrettSaumon2014",
            "StarrettEtAl2014",
            *citation_keys_for_xc_model(self.xc_model),
            *citation_keys_for_chi0_model(self.qoz_response_chi0_model),
            *citation_keys_for_lfc_model(self.qoz_response_lfc_model),
        ]
        return tuple(dict.fromkeys(keys))


def _solve_electronic_structure(
    cfg: PlasmaWorkflowConfig,
    *,
    symbols: list[str],
    counts: list[float],
) -> tuple[str, dict[str, Any]]:
    """Dispatch one unified formula state to single-AA or mixture AA."""
    aa_overrides = dict(cfg.aa_overrides)
    configured_model = aa_overrides.pop("electronic_model", None)
    if configured_model is not None:
        configured_key = str(configured_model).strip().lower().replace("-", "_")
        aliases = {
            "ks": "qm",
            "ks_aa": "qm",
            "quantum": "qm",
            "thomas_fermi": "tf",
            "thomasfermi": "tf",
        }
        configured_key = aliases.get(configured_key, configured_key)
        if configured_key != str(cfg.electronic_model):
            raise ValueError(
                "Conflicting electronic_model values in PlasmaWorkflowConfig "
                "and aa_overrides."
            )
    aa_overrides["electronic_model"] = str(cfg.electronic_model)
    configured_xc_model = aa_overrides.pop("xc_model", None)
    if (
        configured_xc_model is not None
        and str(configured_xc_model).strip().lower()
        != str(cfg.xc_model).strip().lower()
    ):
        raise ValueError(
            "Conflicting xc_model values in PlasmaWorkflowConfig and aa_overrides."
        )
    aa_overrides["xc_model"] = str(cfg.xc_model)
    configured_gga_core_mode = aa_overrides.pop("gga_core_mode", None)
    if (
        configured_gga_core_mode is not None
        and str(configured_gga_core_mode).strip().lower()
        != str(cfg.gga_core_mode).strip().lower()
    ):
        raise ValueError(
            "Conflicting gga_core_mode values in PlasmaWorkflowConfig and "
            "aa_overrides."
        )
    configured_gga_core_zr = aa_overrides.pop("gga_core_zr", None)
    if (
        configured_gga_core_zr is not None
        and not np.isclose(float(configured_gga_core_zr), float(cfg.gga_core_zr))
    ):
        raise ValueError(
            "Conflicting gga_core_zr values in PlasmaWorkflowConfig and "
            "aa_overrides."
        )
    aa_overrides["gga_core_mode"] = str(cfg.gga_core_mode)
    aa_overrides["gga_core_zr"] = float(cfg.gga_core_zr)
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
            **aa_overrides,
        )
        return "single_species", solve_full_then_external(cfg_single)

    cfg_mix = MixtureConfig(
        species=list(symbols),
        counts=[float(val) for val in counts],
        temperature_ev=float(cfg.temperature_ev),
        rho_g_cc=float(cfg.rho_g_cc),
        aa_overrides=aa_overrides,
        species_overrides={key: dict(val) for key, val in cfg.species_overrides.items()},
        mu_e_tol=float(cfg.mu_e_tol),
        root_tol=float(cfg.root_tol),
        root_maxfev=int(cfg.root_maxfev),
        root_brent_maxiter=int(cfg.root_brent_maxiter),
        root_threshold_b3_surrogate_mode=str(
            cfg.root_threshold_b3_surrogate_mode
        ),
        allow_unconverged_root=bool(cfg.allow_unconverged_root),
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


def _screening_density_for_qoz(
    final: dict[str, Any],
) -> tuple[np.ndarray, str]:
    """
    Return the canonical screening density used by QOZ.

    Older electronic caches may contain an applied ``constrained_b3``
    post-process that overwrote ``n_scr`` while preserving the defining
    ``n_full-n_ext-n_ion`` profile as ``n_scr_raw``.  That fitted profile was
    not obtained from the external electronic SCF and can generate a
    threshold-dependent low-k potential.  Recover the canonical profile when
    such a cache is encountered.  New results never overwrite ``n_scr`` and
    therefore take the ordinary branch below.
    """
    repair = final.get("screening_tail_repair", {})
    repair = dict(repair) if isinstance(repair, dict) else {}
    if bool(repair.get("applied", False)) and "n_scr_raw" in final:
        raw = np.asarray(final["n_scr_raw"], dtype=float)
        current = np.asarray(final["n_scr"], dtype=float)
        if raw.shape == current.shape and np.all(np.isfinite(raw)):
            return raw, "n_scr_raw_legacy_repair_bypass"
    return np.asarray(final["n_scr"], dtype=float), "n_scr"


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
    n_scr_src, screening_density_source = _screening_density_for_qoz(final)
    zbar_electronic = float(final["zbar"])
    q_scr_native_raw = float(radial_charge_trapezoid(r_src, n_scr_src))
    if "zbar_partition" in final and np.isfinite(float(final["zbar_partition"])):
        zbar_partition = float(final["zbar_partition"])
    elif "n_ion" in final and "Z" in species_entry:
        zbar_partition = float(species_entry["Z"]) - float(
            radial_charge_trapezoid(r_src, np.asarray(final["n_ion"], dtype=float))
        )
    else:
        mode_key = str(cfg.qoz_zbar_mode).strip().lower().replace("-", "_")
        if mode_key in {
            "pseudoatom_partition", "partition", "pa_partition", "z_minus_qion_all"
        }:
            raise ValueError(
                "The pseudoatom_partition QOZ convention requires zbar_partition "
                "or both Z and the native n_ion profile; this electronic cache is incompatible."
            )
        zbar_partition = np.nan
    zbar_qoz = float(
        qoz_zbar_from_nscr(
            r_src,
            n_scr_src,
            partition_zbar=zbar_partition,
            electronic_zbar=zbar_electronic,
            mode=str(cfg.qoz_zbar_mode),
        )
    )
    n_scr = _interp_profile_linear(
        r_src=r_src,
        y_src=n_scr_src,
        r_dst=r,
        right_value=0.0,
    )
    charge_fix = enforce_screening_charge_consistency(
        r,
        n_scr,
        zbar=zbar_qoz,
        renormalize=bool(cfg.qoz_renormalize_nscr_to_zbar),
        transform=transform,
    )
    ion_temperature_ha = float(cfg.ion_temperature_ev) * EV_TO_HA
    t_qoz = time.perf_counter()
    qoz = build_effective_vii_from_nscr(
        r=r,
        n_scr=charge_fix.n_scr_r,
        zbar=zbar_qoz,
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
    direct_mixing_scheme = (
        "anderson"
        if str(cfg.hnc_mixing_scheme).strip().lower() in {"auto", "newton_krylov"}
        else str(cfg.hnc_mixing_scheme)
    )
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
        mixing_scheme=direct_mixing_scheme,
        tail_points=int(cfg.hnc_tail_points),
        c_map_clip=float(cfg.hnc_nodal_clip),
        enforce_h_tail_zero=bool(cfg.hnc_enforce_nodal_tail_zero),
    )
    closure_tol = (
        max(10.0 * float(cfg.hnc_tol), 1.0e-6)
        if cfg.hnc_closure_transform_tol is None
        else float(cfg.hnc_closure_transform_tol)
    )

    def _diagnose_one_component_hnc(
        g_value: np.ndarray,
        s_value: np.ndarray,
        history_value: list[float],
    ) -> dict[str, float | bool]:
        """Audit one returned iterate against the strict production gates."""
        g_arr_local = np.asarray(g_value, dtype=float)
        s_arr_local = np.asarray(s_value, dtype=float)
        s_from_g_local = 1.0 + float(
            1.0 / float(species_entry["volume_bohr3"])
        ) * radial_forward(g_arr_local - 1.0, transform)
        closure_local = float(
            np.max(np.abs(s_arr_local - np.asarray(s_from_g_local, dtype=float)))
        )
        residual_local = np.asarray(history_value, dtype=float)
        best_residual_local = float(
            np.min(residual_local[np.isfinite(residual_local)])
            if np.any(np.isfinite(residual_local))
            else np.inf
        )
        s_min_local = (
            float(np.min(s_arr_local))
            if np.all(np.isfinite(s_arr_local))
            else -np.inf
        )
        s_max_local = (
            float(np.max(s_arr_local))
            if np.all(np.isfinite(s_arr_local))
            else np.inf
        )
        g_min_local = (
            float(np.min(g_arr_local))
            if np.all(np.isfinite(g_arr_local))
            else -np.inf
        )
        g_tail_local = float(
            np.mean(
                g_arr_local[
                    -min(max(int(cfg.hnc_tail_points), 8), g_arr_local.size) :
                ]
            )
        )
        converged_local = bool(
            residual_local.size > 0
            and best_residual_local <= float(cfg.hnc_tol)
            and np.all(np.isfinite(g_arr_local))
            and g_min_local >= 0.0
            and np.all(np.isfinite(s_arr_local))
            and s_min_local > 0.0
            and closure_local <= closure_tol
        )
        return {
            "converged": converged_local,
            "best_residual": best_residual_local,
            "s_min": s_min_local,
            "s_max": s_max_local,
            "g_min": g_min_local,
            "g_tail_mean": g_tail_local,
            "closure_transform_max_abs": closure_local,
        }

    primary_hnc = _diagnose_one_component_hnc(g_r, s_k, residual_history)
    fallback_used = False
    stage_meta: list[dict[str, float | bool | str]] = []
    hnc_solver_path = f"direct_{direct_mixing_scheme}"

    # Strongly coupled one-component states can have a physical HNC root even
    # when a cold start at the full pair potential stalls on an inadmissible
    # branch.  The mixture workflow already handles this by continuing the
    # potential strength and warm-starting each accepted stage.  Apply the
    # same raw OZ/HNC equations in a 1x1 matrix representation only after the
    # mature direct solver fails, so established direct solutions are
    # unchanged.  Potential-strength continuation is an Otter numerical
    # method; Newton iteration for strongly coupled HNC is discussed by
    # Starrett et al., Phys. Rev. E 90, 033110 (2014), Sec. III.
    if (
        not bool(primary_hnc["converged"])
        and cfg.hnc_fallback_mixing_scheme is not None
    ):
        requested_scheme = str(cfg.hnc_mixing_scheme).strip().lower()
        continuation_scheme = (
            "newton_krylov" if requested_scheme == "auto" else requested_scheme
        )
        try:
            (
                g_matrix,
                s_matrix,
                h_matrix,
                c_matrix,
                residual_history,
                stage_meta,
            ) = hnc_solver_multicomponent_continuation(
                r,
                k,
                np.asarray(qoz.vii_r, dtype=float)[None, None, :],
                transform,
                np.asarray(
                    [1.0 / float(species_entry["volume_bohr3"])],
                    dtype=float,
                ),
                ion_temperature_ha,
                potential_scales=tuple(
                    float(val) for val in cfg.hnc_potential_scales
                ),
                mix=float(cfg.hnc_mix),
                tol=float(cfg.hnc_tol),
                max_iter=int(cfg.hnc_max_iter),
                mixing_scheme=continuation_scheme,
                tail_points=int(cfg.hnc_tail_points),
                c_map_clip=float(cfg.hnc_nodal_clip),
                enforce_h_tail_zero=bool(cfg.hnc_enforce_nodal_tail_zero),
                s_min_floor=1.0e-8,
                s_max_ceil=1.0e6,
                s_projection_mode=str(cfg.hnc_s_projection_mode),
                adaptive=bool(cfg.hnc_adaptive_continuation),
                min_scale_step=float(cfg.hnc_min_scale_step),
                max_stage_attempts=int(cfg.hnc_max_stage_attempts),
                require_converged=bool(cfg.hnc_require_converged),
                fallback_mixing_scheme=cfg.hnc_fallback_mixing_scheme,
                newton_max_iter=int(cfg.hnc_newton_max_iter),
            )
        except RuntimeError as exc:
            if bool(cfg.hnc_require_converged):
                raise RuntimeError(
                    "One-component OZ/HNC direct solve failed and its "
                    f"potential-strength continuation failed: {exc}"
                ) from exc
        else:
            g_r = np.asarray(g_matrix[0, 0], dtype=float)
            s_k = np.asarray(s_matrix[0, 0], dtype=float)
            h_r = np.asarray(h_matrix[0, 0], dtype=float)
            c_r = np.asarray(c_matrix[0, 0], dtype=float)
            fallback_used = True
            hnc_solver_path = (
                f"direct_{direct_mixing_scheme}->continuation_"
                f"{continuation_scheme}"
            )

    hnc_solve_s = time.perf_counter() - t_hnc
    final_hnc = _diagnose_one_component_hnc(g_r, s_k, residual_history)
    closure_transform_max_abs = float(
        final_hnc["closure_transform_max_abs"]
    )
    hnc_best_residual = float(final_hnc["best_residual"])
    hnc_s_min = float(final_hnc["s_min"])
    hnc_s_max = float(final_hnc["s_max"])
    hnc_g_min = float(final_hnc["g_min"])
    hnc_g_tail_mean = float(final_hnc["g_tail_mean"])
    hnc_converged = bool(final_hnc["converged"])
    if bool(cfg.hnc_require_converged) and not hnc_converged:
        raise RuntimeError(
            "One-component OZ/HNC did not reach a physical fixed point: "
            f"best residual={hnc_best_residual:.3e} "
            f"(tol={float(cfg.hnc_tol):.3e}), "
            f"min S(k)={hnc_s_min:.3e}, max S(k)={hnc_s_max:.3e}, "
            f"min g(r)={hnc_g_min:.3e}, tail mean g(r)={hnc_g_tail_mean:.6f}, "
            f"closure mismatch={closure_transform_max_abs:.3e} "
            f"(limit={closure_tol:.3e}). "
            "The returned iterate would only be a diagnostic map, not a "
            "production HNC solution. Set hnc_require_converged=False only "
            "for explicit diagnostic output."
        )
    gij_r = np.asarray(g_r, dtype=float)[None, None, :]
    sij_k = np.asarray(s_k, dtype=float)[None, None, :]
    hij_r = np.asarray(h_r, dtype=float)[None, None, :]
    cij_r = np.asarray(c_r, dtype=float)[None, None, :]
    vij_r = np.asarray(qoz.vii_r, dtype=float)[None, None, :]
    vij_k = np.asarray(qoz.vii_k, dtype=float)[None, None, :]
    return {
        "kind": "one_component",
        "sij_convention": "ashcroft_langreth",
        "species": [str(species_entry["element"])],
        "r": r,
        "k": k,
        "n_scr_r": np.asarray(charge_fix.n_scr_r, dtype=float),
        "n_scr_k": np.asarray(qoz.n_scr_k, dtype=float),
        "screening_density_source": str(screening_density_source),
        "zbar": float(zbar_qoz),
        "zbar_qoz": float(zbar_qoz),
        "zbar_partition": float(zbar_partition),
        "zbar_screening_integral_raw": float(q_scr_native_raw),
        "zbar_aa_ws": float(zbar_electronic),
        "zbar_electronic": float(zbar_electronic),
        "qoz_zbar_mode": str(cfg.qoz_zbar_mode),
        "qoz_response_chi0_model": str(cfg.qoz_response_chi0_model),
        "qoz_response_lfc_model": str(cfg.qoz_response_lfc_model),
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
        "hnc_converged": bool(hnc_converged),
        "hnc_best_residual": float(hnc_best_residual),
        "hnc_output_residual": float(hnc_best_residual),
        "hnc_s_min": float(hnc_s_min),
        "hnc_s_max": float(hnc_s_max),
        "hnc_g_min": float(hnc_g_min),
        "hnc_g_tail_mean": float(hnc_g_tail_mean),
        "hnc_require_converged": bool(cfg.hnc_require_converged),
        "hnc_solver_path": str(hnc_solver_path),
        "hnc_fallback_used": bool(fallback_used),
        "hnc_primary_converged": bool(primary_hnc["converged"]),
        "hnc_primary_best_residual": float(primary_hnc["best_residual"]),
        "hnc_primary_s_min": float(primary_hnc["s_min"]),
        "hnc_primary_s_max": float(primary_hnc["s_max"]),
        "hnc_primary_closure_transform_max_abs": float(
            primary_hnc["closure_transform_max_abs"]
        ),
        "stage_meta": [dict(stage) for stage in stage_meta],
        "ion_total_s": float(qoz_build_s + hnc_solve_s),
        "residual_history": list(residual_history),
        "closure_transform_max_abs": float(closure_transform_max_abs),
        "closure_transform_tol": float(closure_tol),
        "hnc_mixing_scheme": str(hnc_solver_path),
        "hnc_nodal_clip": float(cfg.hnc_nodal_clip),
        "hnc_enforce_nodal_tail_zero": bool(cfg.hnc_enforce_nodal_tail_zero),
        "charge_fix": {
            "q_scr_raw": float(charge_fix.q_scr_raw),
            "q_scr_used": float(charge_fix.q_scr_used),
            "q_scr_rel": float(charge_fix.q_scr_rel),
            "normalization_measure": str(charge_fix.normalization_measure),
            "scale_factor": float(charge_fix.scale_factor),
            "total_scale_factor": float(charge_fix.scale_factor),
            "q_scr_trapz_raw": float(charge_fix.q_scr_trapz_raw),
            "q_scr_trapz_used": float(charge_fix.q_scr_trapz_used),
            "q_scr_dst_raw": float(charge_fix.q_scr_dst_raw),
            "q_scr_dst_used": float(charge_fix.q_scr_dst_used),
            "zbar_target": float(zbar_qoz),
            "zbar_partition": float(zbar_partition),
            "q_scr_native_raw": float(q_scr_native_raw),
            "pa_charge_residual_native": float(q_scr_native_raw - zbar_partition),
            "zbar_aa_ws": float(zbar_electronic),
            "zbar_electronic": float(zbar_electronic),
            "zbar_target_source": str(cfg.qoz_zbar_mode),
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
    zbar_partition = np.zeros(n_species, dtype=float)
    q_scr_native_raw = np.zeros(n_species, dtype=float)
    screening_density_source: list[str] = []
    zbar_electronic = np.zeros(n_species, dtype=float)
    n_i = np.zeros(n_species, dtype=float)
    x = np.asarray([float(sp["x"]) for sp in species_entries], dtype=float)
    vbar_bohr3 = float(
        np.sum(x * np.asarray([float(sp["volume_bohr3"]) for sp in species_entries], dtype=float))
    )
    n_mix = 1.0 / max(float(vbar_bohr3), 1.0e-300)
    symbols = [str(sp["element"]) for sp in species_entries]
    for idx, sp in enumerate(species_entries):
        final = dict(sp["result"])
        r_native = np.asarray(final["r"], dtype=float)
        n_scr_native, source = _screening_density_for_qoz(final)
        screening_density_source.append(str(source))
        n_scr[idx] = _interp_profile_linear(
            r_src=r_native,
            y_src=n_scr_native,
            r_dst=r,
            right_value=0.0,
        )
        zbar_electronic[idx] = float(final["zbar"])
        q_scr_native_raw[idx] = float(
            radial_charge_trapezoid(r_native, n_scr_native)
        )
        if "zbar_partition" in final and np.isfinite(float(final["zbar_partition"])):
            zbar_partition[idx] = float(final["zbar_partition"])
        elif "n_ion" in final and "Z" in sp:
            zbar_partition[idx] = float(sp["Z"]) - float(
                radial_charge_trapezoid(
                    r_native,
                    np.asarray(final["n_ion"], dtype=float),
                )
            )
        else:
            mode_key = str(cfg.qoz_zbar_mode).strip().lower().replace("-", "_")
            if mode_key in {
                "pseudoatom_partition", "partition", "pa_partition", "z_minus_qion_all"
            }:
                raise ValueError(
                    "The pseudoatom_partition QOZ convention requires zbar_partition "
                    "or both Z and the native n_ion profile; this electronic cache is incompatible."
                )
            zbar_partition[idx] = np.nan
        zbar[idx] = float(
            qoz_zbar_from_nscr(
                r_native,
                n_scr_native,
                partition_zbar=float(zbar_partition[idx]),
                electronic_zbar=float(zbar_electronic[idx]),
                mode=str(cfg.qoz_zbar_mode),
            )
        )
        n_i[idx] = float(x[idx] * n_mix)

    charge_fix = enforce_screening_charge_consistency_many(
        r,
        n_scr,
        zbar=zbar,
        renormalize=bool(cfg.qoz_renormalize_nscr_to_zbar),
        transform=transform,
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
        mixing_scheme=(
            "newton_krylov"
            if str(cfg.hnc_mixing_scheme).strip().lower() == "auto"
            else str(cfg.hnc_mixing_scheme)
        ),
        tail_points=int(cfg.hnc_tail_points),
        c_map_clip=float(cfg.hnc_nodal_clip),
        enforce_h_tail_zero=bool(cfg.hnc_enforce_nodal_tail_zero),
        s_min_floor=1.0e-8,
        s_max_ceil=1.0e6,
        s_projection_mode=str(cfg.hnc_s_projection_mode),
        adaptive=bool(cfg.hnc_adaptive_continuation),
        min_scale_step=float(cfg.hnc_min_scale_step),
        max_stage_attempts=int(cfg.hnc_max_stage_attempts),
        require_converged=bool(cfg.hnc_require_converged),
        fallback_mixing_scheme=cfg.hnc_fallback_mixing_scheme,
        newton_max_iter=int(cfg.hnc_newton_max_iter),
    )
    hnc_solve_s = time.perf_counter() - t_hnc
    sqrt_n = np.sqrt(np.outer(n_i, n_i))
    s_from_g = (
        np.eye(n_species, dtype=float)[:, :, np.newaxis]
        + sqrt_n[:, :, np.newaxis] * radial_forward(np.asarray(g_r, dtype=float) - 1.0, transform)
    )
    closure_transform_max_abs = float(
        np.max(np.abs(np.asarray(s_k, dtype=float) - np.asarray(s_from_g, dtype=float)))
    )
    residual_arr = np.asarray(residual_history, dtype=float)
    hnc_best_residual = float(
        np.min(residual_arr[np.isfinite(residual_arr)])
        if np.any(np.isfinite(residual_arr))
        else np.inf
    )
    s_arr = np.asarray(s_k, dtype=float)
    s_symmetric = 0.5 * (s_arr + np.swapaxes(s_arr, 0, 1))
    if np.all(np.isfinite(s_symmetric)):
        s_eigenvalues = np.linalg.eigvalsh(np.moveaxis(s_symmetric, 2, 0))
        hnc_s_min = float(np.min(s_eigenvalues))
        hnc_s_max = float(np.max(s_eigenvalues))
    else:
        hnc_s_min = -np.inf
        hnc_s_max = np.inf
    final_stage = dict(stage_meta[-1]) if stage_meta else {}
    requested_scale = float(max(cfg.hnc_potential_scales))
    hnc_output_residual = float(
        final_stage.get("res_final", hnc_best_residual)
    )
    hnc_converged = bool(
        final_stage.get("converged", False)
        and float(final_stage.get("potential_scale", 0.0))
        >= requested_scale - 1.0e-12
        and np.isfinite(hnc_output_residual)
        and hnc_output_residual <= float(cfg.hnc_tol)
    )
    gii = np.asarray([g_r[idx, idx] for idx in range(n_species)], dtype=float)
    sii = np.asarray([s_k[idx, idx] for idx in range(n_species)], dtype=float)
    return {
        "kind": "multicomponent",
        "sij_convention": "ashcroft_langreth",
        "species": symbols,
        "r": r,
        "k": k,
        "n_scr_r": np.asarray(charge_fix.n_scr_r, dtype=float),
        "n_scr_k": np.asarray(qoz.n_scr_k, dtype=float),
        "screening_density_source": list(screening_density_source),
        "zbar": np.asarray(zbar, dtype=float),
        "zbar_qoz": np.asarray(zbar, dtype=float),
        "zbar_partition": np.asarray(zbar_partition, dtype=float),
        "zbar_screening_integral_raw": np.asarray(q_scr_native_raw, dtype=float),
        "zbar_aa_ws": np.asarray(zbar_electronic, dtype=float),
        "zbar_electronic": np.asarray(zbar_electronic, dtype=float),
        "qoz_zbar_mode": str(cfg.qoz_zbar_mode),
        "qoz_response_chi0_model": str(cfg.qoz_response_chi0_model),
        "qoz_response_lfc_model": str(cfg.qoz_response_lfc_model),
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
        "hnc_converged": bool(hnc_converged),
        "hnc_best_residual": float(hnc_best_residual),
        "hnc_output_residual": float(hnc_output_residual),
        "hnc_s_min": float(hnc_s_min),
        "hnc_s_max": float(hnc_s_max),
        "ion_total_s": float(qoz_build_s + hnc_solve_s),
        "residual_history": list(residual_history),
        "stage_meta": [dict(stage) for stage in stage_meta],
        "closure_transform_max_abs": float(closure_transform_max_abs),
        "hnc_mixing_scheme": (
            "newton_krylov"
            if str(cfg.hnc_mixing_scheme).strip().lower() == "auto"
            else str(cfg.hnc_mixing_scheme)
        ),
        "hnc_s_projection_mode": str(cfg.hnc_s_projection_mode),
        "hnc_nodal_clip": float(cfg.hnc_nodal_clip),
        "hnc_require_converged": bool(cfg.hnc_require_converged),
        "hnc_enforce_nodal_tail_zero": bool(cfg.hnc_enforce_nodal_tail_zero),
        "charge_fix": {
            "q_scr_raw": np.asarray(charge_fix.q_scr_raw, dtype=float),
            "q_scr_used": np.asarray(charge_fix.q_scr_used, dtype=float),
            "q_scr_rel": np.asarray(charge_fix.q_scr_rel, dtype=float),
            "normalization_measure": str(charge_fix.normalization_measure),
            "scale_factor": np.asarray(charge_fix.scale_factor, dtype=float),
            "total_scale_factor": np.asarray(charge_fix.scale_factor, dtype=float),
            "q_scr_trapz_raw": np.asarray(charge_fix.q_scr_trapz_raw, dtype=float),
            "q_scr_trapz_used": np.asarray(charge_fix.q_scr_trapz_used, dtype=float),
            "q_scr_dst_raw": np.asarray(charge_fix.q_scr_dst_raw, dtype=float),
            "q_scr_dst_used": np.asarray(charge_fix.q_scr_dst_used, dtype=float),
            "zbar_target": np.asarray(zbar, dtype=float),
            "zbar_partition": np.asarray(zbar_partition, dtype=float),
            "q_scr_native_raw": np.asarray(q_scr_native_raw, dtype=float),
            "pa_charge_residual_native": np.asarray(
                q_scr_native_raw - zbar_partition, dtype=float
            ),
            "zbar_aa_ws": np.asarray(zbar_electronic, dtype=float),
            "zbar_electronic": np.asarray(zbar_electronic, dtype=float),
            "zbar_target_source": str(cfg.qoz_zbar_mode),
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

    if str(electronic_kind) == "mixture" and cfg.ion_temperature_ev is not None:
        mixture_meta = dict(electronic_result.get("meta", {}))
        if not bool(mixture_meta.get("root_success", True)) and not bool(cfg.allow_unconverged_root):
            raise RuntimeError(
                "Refusing to continue an unconverged mixture common-mu state into QOZ/HNC: "
                f"max|dmu|={float(mixture_meta.get('mu_residual_max_ha', np.nan)):.6e} Ha. "
                "Recompute the electronic state, or set allow_unconverged_root=True only "
                "for an explicit diagnostic best-effort continuation."
            )
        final_mu_success = mixture_meta.get("final_mu_root_success", None)
        final_mu_residual = mixture_meta.get("final_mu_residual_max_ha", None)
        if final_mu_success is None and final_mu_residual is not None:
            try:
                final_mu_value = float(final_mu_residual)
            except (TypeError, ValueError):
                final_mu_value = np.inf
            final_mu_success = bool(
                np.isfinite(final_mu_value)
                and final_mu_value <= float(cfg.mu_e_tol)
            )
        if final_mu_success is not None and not bool(final_mu_success) and not bool(cfg.allow_unconverged_root):
            raise RuntimeError(
                "Refusing to continue a mixture whose final full+external rerun "
                "lost common-mu closure into QOZ/HNC: "
                f"max|dmu_final|={float(mixture_meta.get('final_mu_residual_max_ha', np.nan)):.6e} Ha. "
                "Recompute the electronic state, or set allow_unconverged_root=True only "
                "for an explicit diagnostic best-effort continuation."
            )

    if cfg.ion_temperature_ev is not None:
        convergence_issues = _electronic_convergence_issues(
            species_entries,
            require_external=True,
        )
        if convergence_issues and not bool(cfg.allow_unconverged_aa):
            raise RuntimeError(
                "Refusing to continue unconverged electronic structure into QOZ/HNC: "
                + "; ".join(convergence_issues)
                + ". Set allow_unconverged_aa=True only for an explicit diagnostic continuation."
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
    if bool(cfg.save_state_npz):
        from otter.io.state import StateExportOptions, save_plasma_state

        if cfg.save_state_path is None:
            species_label = "-".join(str(symbol) for symbol in symbols)
            filename = (
                f"{species_label}_rho{float(cfg.rho_g_cc):.6g}_"
                f"Te{float(cfg.temperature_ev):.6g}_"
                f"Ti{float(cfg.ion_temperature_ev):.6g}_state.npz"
            )
            state_path = Path(cfg.save_output_dir) / filename
        else:
            state_path = Path(cfg.save_state_path)
        saved_state = save_plasma_state(
            state_path,
            result,
            options=StateExportOptions(
                r_max_bohr=float(cfg.state_r_max_bohr),
                k_max_bohr_inv=float(cfg.state_k_max_bohr_inv),
            ),
        )
        result.setdefault("saved_paths", {})
        result["saved_paths"]["state_npz"] = str(saved_state)
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
