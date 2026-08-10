Ion-structure library migration note
====================================

This note records the decisions behind the private-development
``ion_structure_library`` benchmark.  It is a maintenance record, not a
redistribution determination for the publication-derived curves.

Accepted core families
----------------------

* Gill 2020: Al, ``2.7 g/cc``, ``5 eV``, :math:`S_{ii}(k)`;
* Clérouin *et al.* 2015: Al, ``8.1 g/cc``, ``Te=10 eV`` with
  ``Ti=10`` and ``2 eV``, :math:`S_{ii}(k)`;
* Wünsch *et al.* 2009: Be, ``5.544 g/cc``, ``13 eV``,
  :math:`g_{ii}(r)` and :math:`S_{ii}(k)`;
* Starrett and Saumon 2013: C, ``20 g/cc``, ``50 eV``,
  :math:`g_{ii}(r)`.

Every paired numerical result was recomputed by Otter.  Legacy NPZ
files were not reused because several required Python pickle, disagreed with
their filenames, or contained invalid values.

Unit audit
----------

The original plotting programs are the coordinate-unit evidence:

* Clérouin and Wünsch leave the reference :math:`k` values unchanged, convert
  the project grid by dividing its inverse-Bohr values by ``0.529177...``,
  and label the result :math:`\mathrm{\AA}^{-1}`;
* Wünsch real-space data use ångström;
* the selected Starrett carbon values are left unchanged on an
  :math:`a_B` axis.

The exceptional Fe conversion and special carbon scaling were not
generalized.  A later panel-by-panel audit established that both PRE axes are
:math:`r/R_{\rm WS}`; those states now live in the dedicated
:doc:`/benchmarks/starrett_single_species_2013_2014` package.

Numerical acceptance
--------------------

The producer requires a nonlinear HNC residual below :math:`10^{-4}`.
The finite DST relation

.. math::

   S(k) = 1 + n_i\mathcal{F}[g(r)-1]

is checked independently.  Cold, strongly coupled Al has a converged
nonlinear residual near :math:`7\times10^{-5}` and a finite-box transform
mismatch near :math:`2\times10^{-3}`.  Otter therefore exposes a separate
``hnc_closure_transform_tol``; increasing ``hnc_tol`` would stop the nonlinear
iteration too early and is not an acceptable substitute.

Excluded or pending data
------------------------

* C at ``3.7 g/cc, 8.62 eV`` remains a pressure-ionization validation target,
  but Otter correctly refuses to continue its unresolved threshold state.
* The additional Starrett C/Fe/H/W files are now the dedicated
  :doc:`/benchmarks/starrett_single_species_2013_2014` benchmark.  It records
  the PRE erratum, corrected coordinates, and missing/rejected Otter states
  instead of reusing legacy PAMD archives.
* ``Zak_2025`` is now the dedicated
  :doc:`/benchmarks/johnson_et_al_2025_two_temperature_al` benchmark.  Its
  legacy ångström axis label was corrected to Bohr after checking Fig. 2 of
  :cite:t:`JohnsonEtAl2025`.
* ``argha`` is now the dedicated
  :doc:`/benchmarks/argha_roy_carbon_sii` benchmark.  The reference
  curves are labelled ``DFT-MD data provided by Dr. Argha Roy``, matching the
  inherited plotting source and the maintainer's current identification.
  The intermediate ``DMF-MD`` name is retained only as provenance.
  The files are published by maintainer decision with attribution and data
  license status ``NOASSERTION``.
* ``max2022`` is linked to :cite:t:`SchornerEtAl2022`, but the inherited
  extraction has not yet been distinguished reliably between the nearly
  overlapping DFT-MD points and NN-MD curve.  Its 5 eV file even contains
  wave numbers below the minimum reciprocal vector of the cited 125-ion
  DFT-MD cell.  It remains catalogued rather than being given an invented
  method identity.
* ``non_equlibrum`` contains project sensitivity output rather than an
  independent reference.

Reproduction
------------

The offline runner validates checksums and recreates metrics and figures:

.. code-block:: console

   python benchmarks/runners/plot_ion_structure_library.py

The expensive native calculation stages candidates without changing accepted
baselines:

.. code-block:: console

   python benchmarks/runners/regenerate_ion_structure_library.py

The downloadable gallery is independent of both maintenance programs.  Its
``USE_PRECOMPUTED_DATA`` switch selects checksum-verified loading or a direct
Otter calculation implemented entirely in that one gallery file.
