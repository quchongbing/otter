"""Continuum-electron models."""
from __future__ import annotations

from otter.electronic.continuum.hybrid import QuantumContinuumHybrid
from otter.electronic.continuum.ideal import IdealContinuum, ideal_unbound_density
from otter.electronic.continuum.scattering import (
    QuantumContinuumFree,
    QuantumContinuumScattering,
    continuum_density_free,
    continuum_density_scattering,
    continuum_density_scattering_adaptive,
    continuum_density_scattering_basis,
    fermi_dirac,
    gamma_from_phase_shift_cache,
)
from otter.electronic.continuum.tail import apply_tail_match, tail_parameters

__all__ = [
    "IdealContinuum",
    "QuantumContinuumFree",
    "QuantumContinuumHybrid",
    "QuantumContinuumScattering",
    "apply_tail_match",
    "continuum_density_free",
    "continuum_density_scattering",
    "continuum_density_scattering_adaptive",
    "continuum_density_scattering_basis",
    "fermi_dirac",
    "gamma_from_phase_shift_cache",
    "ideal_unbound_density",
    "tail_parameters",
]
