Johnson et al. 2025 two-temperature aluminium
===============================================

This benchmark compares Otter IS-QOZ/HNC pair distributions with Figure 2 of
:cite:t:`JohnsonEtAl2025` at
:math:`\rho=2.7\ {\rm g\,cm^{-3}}`, :math:`T_i=1` eV, and
:math:`T_e=1,3,10,30` eV.  The reference package contains the paper's
2TTCP HNC+bridge, DFT-MD, and YOCP HNC+bridge curves for panels 2(a)--2(d).

Otter uses the pseudoatom construction of
:cite:t:`StarrettSaumon2013,StarrettSaumon2014`, finite-temperature Lindhard
response, the jellium LFC of :cite:t:`Chabrier1990`, and an HNC ion closure.
It does not include the bridge correction used by the 2TTCP reference, so the
comparison does not equate the two models.  Johnson *et al.* used PAW DFT-MD
with 11 explicit electrons per Al atom and 64-atom cells at 1 eV and 32-atom
cells at 3, 10, and 30 eV :cite:p:`JohnsonEtAl2025`.

Coordinate unit
---------------

The publication labels the horizontal axis as :math:`r` in atomic units, so
the archived coordinate and Otter radius are compared directly in Bohr.  A
source plotting file labelled the same values as ångström; the benchmark does
not repeat that label or rescale the data.

Reproduce the comparison
------------------------

The standalone script
``benchmarks/examples/plot_johnson_et_al_2025_two_temperature_al.py`` contains
the thermodynamic inputs, Otter workflow calls, convergence checks, checksum
validation, and plotting.  Set

.. code-block:: python

   USE_PRECOMPUTED_DATA = True

to load checksummed results, or set it to ``False`` to run all four
calculations.  New files are written under
``benchmarks/outputs/johnson_et_al_2025_two_temperature_al/gallery_recomputed``
and do not overwrite accepted results.  The 30 eV calculation uses
potential-strength continuation with the Newton--Krylov HNC backend; its
residual and transform-closure diagnostics are recorded in the manifest.

The rendered comparison is available at
:ref:`sphx_glr_benchmarks_gen_benchmarks_plot_johnson_et_al_2025_two_temperature_al.py`.

Reference-data notice
---------------------

The project maintainer digitized the reference curves from Figure 2(a)--2(d).
The files record article panels, units, method labels, and SHA-256 checksums.
The article does not attach an open-data license to these point sets, so they
are distributed with attribution and license status ``NOASSERTION`` and are
not covered by Otter's BSD software license.  See the manifest under
``benchmarks/reference_data/johnson_et_al_2025_two_temperature_al`` before
reuse.
