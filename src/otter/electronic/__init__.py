"""Electronic-structure solvers and average-atom workflow drivers."""
from __future__ import annotations

from otter.electronic.full_external import FullExternalConfig, solve_full_only, solve_full_then_external
from otter.electronic.ks_dft import KSDTFConfig, solve_ks_dft_is
from otter.electronic.mixture import MixtureConfig, solve_mixture_full_then_ext, solve_mixture_full_only
from otter.electronic.thomas_fermi import (
    FermiHalfEvaluator,
    ThomasFermiConfig,
    incomplete_fermi_half_lower,
    incomplete_fermi_half_upper,
    solve_thomas_fermi_full_then_external,
)
from otter.electronic.xc import xc_provenance

__all__ = [
    "FermiHalfEvaluator",
    "FullExternalConfig",
    "KSDTFConfig",
    "MixtureConfig",
    "ThomasFermiConfig",
    "incomplete_fermi_half_lower",
    "incomplete_fermi_half_upper",
    "solve_full_only",
    "solve_full_then_external",
    "solve_ks_dft_is",
    "solve_mixture_full_only",
    "solve_mixture_full_then_ext",
    "solve_thomas_fermi_full_then_external",
    "xc_provenance",
]
