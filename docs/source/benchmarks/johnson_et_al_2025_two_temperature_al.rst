Johnson et al. 2025 two-temperature aluminium
===============================================

This benchmark compares Otter IS-QOZ/HNC pair distributions with the
two-temperature aluminium curves in Fig. 2 of :cite:t:`JohnsonEtAl2025`.
All states use
:math:`\rho=2.7\ {\rm g\,cm^{-3}}` and :math:`T_i=1` eV:

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - Publication panel
     - :math:`T_e`
     - Local reference curves
   * - Fig. 2(a)
     - 1 eV
     - 2TTCP HNC+bridge, DFT-MD, YOCP HNC+bridge
   * - Fig. 2(c)
     - 10 eV
     - 2TTCP HNC+bridge, DFT-MD, YOCP HNC+bridge
   * - Fig. 2(d)
     - 30 eV
     - 2TTCP HNC+bridge, DFT-MD, YOCP HNC+bridge

The 3 eV state in Fig. 2(b) is absent from the inherited local data library
and is not reconstructed or invented.

Scientific interpretation
-------------------------

The solid 2TTCP reference is the paper's quantum Ornstein--Zernike model with
a bridge correction.  The other references are DFT-MD and a Yukawa
one-component-plasma HNC calculation with a bridge correction
:cite:p:`JohnsonEtAl2025`.  Otter's line is labelled
``Otter IS-QOZ/HNC``: it follows the pseudoatom construction of
:cite:t:`StarrettSaumon2013,StarrettSaumon2014`, finite-temperature Lindhard
response, the jellium LFC of :cite:t:`Chabrier1990`, and an HNC ion closure.
It is a comparison method, not a claim to be identical to the paper's 2TTCP
or bridge implementation.

Johnson et al. used VASP PAW DFT-MD with 11 explicit electrons per Al atom.
The reported cells contain 64 atoms at 1 and 3 eV and 32 atoms at 10 and
30 eV :cite:p:`JohnsonEtAl2025`.  Those finite-cell reference details are
important when interpreting small differences from an integral-equation
curve.

Coordinate-unit correction
--------------------------

The publication labels the Fig. 2 horizontal axis :math:`r\,[{\rm au}]`;
therefore the archived first column and Otter radius are compared directly in
Bohr.  The inherited ``plot_zak.py`` labelled the unchanged values as
ångström.  Direct inspection of the cited figure establishes that as a
legacy plotting-label bug.  This benchmark does **not** rescale those
coordinates.

Reproduce or reuse
------------------

The gallery file is a complete standalone program:

``benchmarks/examples/plot_johnson_et_al_2025_two_temperature_al.py``
   Contains the thermodynamic inputs, public ``PlasmaWorkflowConfig`` call,
   convergence checks, baseline/reference checksum validation, candidate
   output writer, metrics-ready arrays, and figure construction.

At its top, select:

.. code-block:: python

   USE_PRECOMPUTED_DATA = True

``True`` accepts only a complete set of reviewed NPZ baselines whose hashes
match the dedicated manifest.  All three states passed the strict electronic,
HNC residual, raw :math:`S(k)>0`, and transform-closure checks.

The 30 eV state is also a solver-regression case.  A cold-start
full-potential Anderson iteration stalls at residual 1 and produces negative
:math:`S(k)`.  This is not evidence that the physical HNC root is absent:
using the same :math:`V_{ii}` with potential-strength continuation and the
raw matrix-free Newton--Krylov backend reaches scale one with residual
:math:`3.07\times10^{-10}`, minimum :math:`S(k)=5.99\times10^{-3}`, and
transform-closure mismatch :math:`2.84\times10^{-9}`.  Otter therefore uses
continuation only as a strict fallback after the mature direct
one-component solve fails.  Potential-strength continuation is an Otter
numerical method; the need for Newton iteration in strongly coupled HNC
calculations is discussed by :cite:t:`StarrettEtAl2014`.

Set the switch to ``False`` to run all three Otter calculations directly.
The default process layout is three state workers times six continuum
workers, or at most about 18 workers on a 24-core machine.  New results are
written under
``benchmarks/outputs/johnson_et_al_2025_two_temperature_al/gallery_recomputed``
and never overwrite reviewed baselines.

Every figure call writes both a high-resolution PNG for HTML and a vector PDF
for papers or slides through :func:`otter.plotting.save_figure`.  The files
are placed under
``benchmarks/outputs/johnson_et_al_2025_two_temperature_al/figures``.

Data provenance and reuse notice
--------------------------------

The nine CSV files are copied byte-for-byte from the user's local
``form factors/Zak`` library.  Their per-file SHA-256 values, panels, units,
and method labels are recorded in
``benchmarks/reference_data/johnson_et_al_2025_two_temperature_al/manifest.json``.
The inherited library does not record whether the points were digitized or
author-provided, so their extraction provenance remains unresolved.

The article says its data are available from the corresponding author upon
reasonable request; it does not attach an open-data license to these point
sets.  The maintainer has chosen to publish the extracted coordinates and
derived comparisons with article/panel attribution and license status
``NOASSERTION``.  They are not covered by Otter's BSD-3-Clause software
license.  This repository policy does not assert that APS or the authors
supplied an open-data license.

The publication PDF is linked by DOI and is not copied into the repository:
`10.1103/5c29-kdx1 <https://doi.org/10.1103/5c29-kdx1>`__.

After an accepted baseline is reviewed, the rendered gallery comparison is
available at
:ref:`sphx_glr_benchmarks_gen_benchmarks_plot_johnson_et_al_2025_two_temperature_al.py`.
