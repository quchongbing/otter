Ion-structure literature library
================================

This benchmark compares Otter with published and author-provided
ion-structure curves for aluminium, beryllium, and carbon.

.. list-table::
   :header-rows: 1
   :widths: 18 28 18 36

   * - Material
     - State
     - Observable
     - Reference
   * - Al
     - 2.7 g cm\ :sup:`-3`, :math:`T_e=T_i=5` eV
     - :math:`S_{ii}(k)`
     - Fig. 3 of :cite:t:`GillEtAl2015`
   * - Al
     - 8.1 g cm\ :sup:`-3`, :math:`T_e=T_i=10` eV
     - :math:`S_{ii}(k)`
     - Fig. 1 of :cite:t:`ClerouinEtAl2015`
   * - Al
     - 8.1 g cm\ :sup:`-3`, :math:`T_e=10`, :math:`T_i=2` eV
     - :math:`S_{ii}(k)`
     - Fig. 1 of :cite:t:`ClerouinEtAl2015`
   * - Be
     - 5.544 g cm\ :sup:`-3`, :math:`T_e=T_i=13` eV
     - :math:`g_{ii}(r)`, :math:`S_{ii}(k)`
     - Figs. 1(c) and 2 of :cite:t:`WunschEtAl2009`
   * - C
     - 20 g cm\ :sup:`-3`, :math:`T_e=T_i=50` eV
     - :math:`g_{ii}(r)`
     - C. E. Starrett, private communication (unpublished)

Each panel states whether it is an equilibrium or two-temperature comparison.
The two Clérouin panels include independent Otter KS and Thomas--Fermi
average-atom calculations; both use the same QOZ/HNC settings.

Units
-----

The Gill, Clérouin, and Wünsch reciprocal-space coordinates are in
:math:`\mathrm{\AA}^{-1}`; the Wünsch real-space coordinate is in
:math:`\mathrm{\AA}`; and the Starrett carbon coordinate is in Bohr.  The
plotting script converts Otter coordinates to the reference unit without
modifying the archived reference columns.

Reproduce the comparison
------------------------

The downloadable script
:doc:`gen_benchmarks/plot_ion_structure_library` exposes one switch:

.. code-block:: python

   USE_PRECOMPUTED_DATA = True

``True`` verifies and loads checksummed Otter results.  ``False`` runs the
average-atom and QOZ/HNC calculations in the same script, writes new files
under ``benchmarks/outputs/ion_structure_library/gallery_recomputed``, and
plots those results.  Accepted files are never overwritten automatically.
Each run exports PNG and PDF figures.

Reference-data notice
---------------------

The publication-derived and author-provided coordinates are not covered by
Otter's BSD software license.  Their attributions, per-file checksums, units,
and license status
``NOASSERTION`` are recorded in
``benchmarks/reference_data/ion_structure_library``.  Consult those records
before redistributing the numerical values.
