"""
otter/numerics/constants.py

Purpose
-------
Define common physical constants and unit conversions (atomic units).

Methods
-------
- Use fixed scalars for eV <-> Hartree and Bohr <-> cm.

References
----------
- CODATA 2018 values :cite:p:`TiesingaEtAl2021` (rounded for convenience);
  these constants are supplied as fixed atomic-unit conversion factors by
  Otter.
"""

EV_TO_HA = 0.03674932217565499
HA_TO_EV = 1.0 / EV_TO_HA
BOHR_TO_CM = 5.29177210903e-9
CM_TO_BOHR = 1.0 / BOHR_TO_CM
