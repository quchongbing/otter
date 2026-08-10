Starrett et al. 2014 mixture Figure 3
======================================

This benchmark compares the three pair distributions of
:math:`\mathrm{CH}_{1.36}` with the solid IS-QM curves digitized from Figure 3
of :cite:t:`StarrettEtAl2014`.  It covers nine states:

* :math:`\rho = 2.94,\ 5,\ 15\ {\rm g\,cm^{-3}}`;
* :math:`T = 20,\ 50,\ 100\ {\rm kK}`;
* C--C, C--H, and H--H pair channels.

.. important::

   The solid curves are imported, validated **private precursor** results.
   They are a migration target and provenance record, not evidence that every
   state has already been recomputed by the current Otter solver.  The figure
   uses the concise legend ``Otter`` requested for the migrated project; this
   provenance note and the manifest retain the exact numerical origin.

This is the four-author *Physical Review E* mixture paper, ``90``, 033110
(2014): `DOI/publisher record
<https://doi.org/10.1103/PhysRevE.90.033110>`__,
`APS accepted manuscript
<https://link.aps.org/accepted/10.1103/PhysRevE.90.033110>`__, and
`author preprint <https://arxiv.org/abs/1408.3548>`__.  It is distinct from the
two-author methodology paper :cite:t:`StarrettSaumon2014` in *High Energy
Density Physics*, ``10``, 35--42 (2014).

Offline comparison
------------------

The benchmark gallery below reads only immutable digitized CSV files and
portable numeric NPZ results.  It verifies their SHA-256 checksums and
redraws the comparison.  Its source contains the explicit code-level switch
``USE_PRECOMPUTED_DATA = True``.  This default does not run average-atom,
pseudoatom, or HNC solvers and therefore keeps the documentation build fast
and deterministic.

:ref:`sphx_glr_benchmarks_gen_benchmarks_plot_starrett_et_al_2014_mixtures_fig3.py`

The 27 pair/state comparisons have a median RMSE of approximately
``0.00729``.  The largest RMSE is ``0.02176`` for H--H at
:math:`2.94\ {\rm g\,cm^{-3}}` and 50 kK.  These metrics are recomputed by the
standalone read-only runner, rather than copied from the figure.

Provenance and reproducibility
------------------------------

The benchmark data are split into two immutable layers, while the gallery is
the complete executable controller:

``benchmarks/reference_data/starrett_et_al_2014_mixtures_fig3``
   Digitized publication curves.  These are derived from the cited article
   and are not asserted to inherit Otter's source-code license.

``benchmarks/baselines/starrett_et_al_2014_mixtures_fig3``
   Pure-numeric, pickle-free private-precursor arrays, configuration,
   convergence diagnostics, checksums, and the source revision.

``benchmarks/examples/plot_starrett_et_al_2014_mixtures_fig3.py``
   The downloadable gallery source.  It contains both modes in one file:
   checksum-verified loading, or direct construction and solution of all nine
   Otter workflows through the public API.  It also validates convergence,
   saves the selected arrays, evaluates errors, and plots the overlay.  It
   does not dynamically import another Python program.

.. note::

   The independent digitizations are published by maintainer decision with
   source attribution and license status ``NOASSERTION``.  They are not
   covered by Otter's BSD-3-Clause software license, and the repository does
   not assert an APS- or author-supplied data license.  Consult the manifest
   before reuse.

The imported producer configuration used the IS structure model, orbital
quantum average atoms, Fermi--Dirac bound occupations, finite-temperature
Lindhard response, and the Chabrier-1990 LFC.  The manifest is authoritative
for all result-affecting options.

Otter recomputation
--------------------

To recompute the nine states instead of loading the accepted arrays, edit the
gallery source and set:

.. code-block:: python

   USE_PRECOMPUTED_DATA = False

Then execute
``benchmarks/examples/plot_starrett_et_al_2014_mixtures_fig3.py``.  The
same script constructs the nine ``PlasmaWorkflowConfig`` values, calls
``solve_plasma_workflow``, writes candidates below
``benchmarks/outputs/starrett_et_al_2014_mixtures_fig3/gallery_recomputed``,
and plots the new data.

The script evaluates three independent states concurrently and assigns six
continuum workers to each state, so its explicit concurrency is at most about
18 workers on a 24-core machine.  A full nine-state orbital mixture run is
substantially more expensive than the read-only documentation path; its wall
time depends strongly on threshold-state convergence and common-chemical-
potential root evaluations.  It is therefore never launched automatically by
Sphinx.

The live branch implements the IS average-atom/pseudoatom construction and
the Appendix-B full-density damped-Friedel tail described by
:cite:t:`StarrettSaumon2014`: orbital KS-DFT, ordinary Fermi--Dirac bound
occupations, ``full/full`` B3 target/model, strict full and external AA
convergence, and strict common-:math:`\mu` closure.  The ionic path follows
the multicomponent QOZ/HNC construction of :cite:t:`StarrettEtAl2014` with
finite-temperature Lindhard response and the finite-temperature jellium LFC
of :cite:t:`Chabrier1990`.  Candidate files retain
:math:`g_{CC}`, :math:`g_{CH}`, and :math:`g_{HH}` through
:math:`r\leq20\,a_{\rm B}`, together with the state, ionization,
common-:math:`\mu`, HNC, and timing diagnostics required to audit this
Figure-3 comparison.  The in-memory workflow contains the additional
screening, potential, and structure-factor arrays, but this focused gallery
archive does not duplicate arrays that it does not plot.

Separate programs remain under ``benchmarks/runners`` for maintainer release
audits, but they are not dependencies of the gallery script.

To execute the complete downloadable benchmark:

.. code-block:: console

   python benchmarks/examples/plot_starrett_et_al_2014_mixtures_fig3.py
