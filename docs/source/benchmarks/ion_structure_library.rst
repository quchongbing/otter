Ion-structure reference library
===============================

This suite compares Otter calculations with selected curves from the
local form-factor library.  The core states are traceable to:

* Al at 2.7 g cm\ :sup:`-3`, 5 eV: Fig. 3 of
  :cite:t:`GillEtAl2015`;
* Al at 8.1 g cm\ :sup:`-3`, 10 eV: Fig. 1 of
  :cite:t:`ClerouinEtAl2015`;
* Be at 5.544 g cm\ :sup:`-3`, 13 eV:
  :math:`g_{ii}` from Fig. 1(c) and :math:`S_{ii}` from Fig. 2 of
  :cite:t:`WunschEtAl2009`; and
* the selected carbon extraction attributed by the local library to
  :cite:t:`StarrettSaumon2013`.

The thermodynamic class and observable are kept explicit:

.. list-table::
   :header-rows: 1
   :widths: 18 19 17 21 25

   * - Material
     - State
     - Observable
     - Class
     - Source
   * - Al
     - 2.7 g cm\ :sup:`-3`, 5 eV
     - :math:`S_{ii}(k)`
     - equilibrium
     - :cite:t:`GillEtAl2015`
   * - Al
     - 8.1 g cm\ :sup:`-3`, :math:`T_e=T_i=10` eV
     - :math:`S_{ii}(k)`
     - equilibrium
     - :cite:t:`ClerouinEtAl2015`
   * - Al
     - 8.1 g cm\ :sup:`-3`, :math:`T_e=10`, :math:`T_i=2` eV
     - :math:`S_{ii}(k)`
     - two-temperature
     - :cite:t:`ClerouinEtAl2015`
   * - Be
     - 5.544 g cm\ :sup:`-3`, 13 eV
     - :math:`g_{ii}(r)`, :math:`S_{ii}(k)`
     - equilibrium
     - :cite:t:`WunschEtAl2009`
   * - C
     - 20 g cm\ :sup:`-3`, 50 eV
     - :math:`g_{ii}(r)`
     - equilibrium
     - :cite:t:`StarrettSaumon2013`

For that final carbon curve the exact source panel has not yet been
independently verified.  The benchmark therefore reports the paper but does
not invent a figure number.

The four panels of ``ion_structure_library_sii`` have an explicit source
map: panel 1 is Gill *et al.*, Fig. 3 :cite:p:`GillEtAl2015`; panels 2 and 3
are Clérouin *et al.*, Fig. 1 :cite:p:`ClerouinEtAl2015`; panel 4 is Wünsch
*et al.*, Fig. 2 :cite:p:`WunschEtAl2009`.  The separate real-space Wünsch
comparison uses Fig. 1(c) of the same paper.

The full data audit is recorded in
``benchmarks/reference_data/ion_structure_library/inventory.json``.  It
distinguishes validated reference families from duplicate project output,
uncited local data, ambiguous unit conversions, and invalid legacy caches.
In particular, no old pickle-bearing PAMD NPZ is accepted as an Otter
reference result.

Coordinate units were verified against the original plotting programs:
Gill, Clérouin, and Wünsch reciprocal curves use
:math:`\mathrm{\AA}^{-1}`; Wünsch real-space curves use
:math:`\mathrm{\AA}`; the selected Starrett carbon curves use
:math:`a_B`.  The plotting runner converts the Otter grid, never the
archived reference columns.  The Fe and specially scaled carbon curves remain
outside this compact core figure but are now audited in
:doc:`starrett_single_species_2013_2014`.

Other families found in the same workstation library were not silently
discarded:

* the Johnson--Shaffer--Murillo two-temperature Al data now have a dedicated
  :doc:`Johnson 2025 benchmark
  <johnson_et_al_2025_two_temperature_al>`;
* the author-provided carbon DFT-MD data now have a dedicated
  :doc:`Dr. Argha Roy comparison <argha_roy_carbon_sii>`;
* the ``max2022`` files map to Fig. 2 of :cite:t:`SchornerEtAl2022`, but the
  local extraction cannot be assigned consistently to its DFT-MD points or
  NN-MD curve, so it remains permission- and identity-gated;
* the extra Starrett H/C/Fe/W files now have a dedicated
  :doc:`Starrett--Saumon benchmark
  <starrett_single_species_2013_2014>` with corrected units, an erratum
  citation, and explicit strict-rejection states; and
* ``non_equlibrum`` contains only old project output, so it is a future
  sensitivity example rather than an independent literature benchmark.

Reproduce or reuse
------------------

The downloadable gallery script exposes one code-level choice:

.. code-block:: python

   USE_PRECOMPUTED_DATA = True

``True`` verifies and uses the versioned, precomputed Otter outputs.  With
``False`` the same downloadable file directly runs the average-atom and
QOZ/HNC calculations, saves candidate files under
``benchmarks/outputs/ion_structure_library/gallery_recomputed``, computes the
unit-aware comparison metrics, and plots the figures.  It does not import a
second Python runner or producer and never overwrites accepted results.  The
recomputation uses bounded state-level and continuum-level process pools; the
default maximum is about 18 workers.

The HTML gallery displays PNG figures and lets Sphinx-Gallery attach the exact
runnable Python source and notebook.  The same script also exports a vector
PDF for slides under
``benchmarks/outputs/ion_structure_library/figures``.

Data provenance and reuse notice
--------------------------------

Publication does not place digitized curves under Otter's software license.
The maintainer has chosen to publish the bundled values with per-file source
attribution and license status ``NOASSERTION``.  This does not assert a
publisher- or author-supplied data license.  See the reference README and
manifest before reuse.

See the rendered comparison in
:doc:`gen_benchmarks/plot_ion_structure_library`.
