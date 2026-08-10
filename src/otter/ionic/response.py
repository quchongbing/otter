"""Backward-compatible imports for electron local-field corrections.

The implementations live in :mod:`otter.ionic.lfc`. This compatibility
module preserves the public imports used by Otter 0.1 while keeping the
electron-response models separate from the QOZ and HNC solvers.
"""
from __future__ import annotations

from otter.ionic.lfc import (
    LFC_MODEL_CITATION_KEYS,
    LFC_MODEL_REFERENCE_KEYS,
    cee_from_gee,
    cee_jellium_from_lfc,
    chabrier1990_gamma0,
    coupling_parameter_ee_au,
    fermi_energy_au,
    fermi_wavenumber_au,
    gee_jellium,
    gee_jellium_chabrier1990,
    gee_jellium_chabrier_hubbard,
    gee_jellium_geldartvosko,
    gee_jellium_gregori2007,
    gee_jellium_hubbard,
    gee_jellium_utsumiichimaru,
    gee_jellium_vashistasingwi,
    iit_exchange_correlation_free_energy_per_electron_au,
    wigner_seitz_radius_au,
)

__all__ = [
    "LFC_MODEL_CITATION_KEYS",
    "LFC_MODEL_REFERENCE_KEYS",
    "cee_from_gee",
    "cee_jellium_from_lfc",
    "chabrier1990_gamma0",
    "coupling_parameter_ee_au",
    "fermi_energy_au",
    "fermi_wavenumber_au",
    "gee_jellium",
    "gee_jellium_chabrier1990",
    "gee_jellium_chabrier_hubbard",
    "gee_jellium_geldartvosko",
    "gee_jellium_gregori2007",
    "gee_jellium_hubbard",
    "gee_jellium_utsumiichimaru",
    "gee_jellium_vashistasingwi",
    "iit_exchange_correlation_free_energy_per_electron_au",
    "wigner_seitz_radius_au",
]
