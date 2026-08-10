Carbon Otter PA-HNC and DFT-MD comparison
=========================================

This benchmark compares current Otter carbon static ion structure
factors with DFT-MD data provided by Dr. Argha Roy (private communication;
unpublished). The displayed state grid is

.. math::

   \rho = 3.51538\ {\rm g\,cm^{-3}},\qquad
   T_e=T_i=20,\ 30,\ 40,\ 50,\ 100\ {\rm eV}.

Every black curve is recalculated with the current Otter
average-atom :math:`\rightarrow` pseudoatom :math:`\rightarrow` QOZ/HNC
workflow. No historical PAMD curve is used. All five accepted archives pass
the full-AA stage-2, external-AA, threshold-state, HNC fixed-point, and
real/reciprocal transform-closure gates. The shallow states between 20 and
50 eV are resolved by analytic negative-energy exterior matching on the
physical AA box; no unconverged result and no artificial extended bound-only
box is accepted.

The supplied reciprocal-space coordinate is
:math:`\mathrm{\AA}^{-1}`. Otter calculates on an inverse-Bohr grid, and the
gallery converts only the Otter coordinate before comparison. The third
reference column is shown as *reported uncertainty*. It is not identified as
one standard deviation because that statistical interpretation has not been
confirmed.

Reproduce, inspect, and export
------------------------------

The downloadable gallery source is a complete calculation program. At its
top, choose

.. code-block:: python

   USE_PRECOMPUTED_DATA = True

to verify and load the five reviewed current-Otter archives, or change it to
``False`` to run all average-atom and QOZ/HNC calculations directly in the
same file. Live results are saved under
``benchmarks/outputs/argha_roy_carbon_sii/gallery_recomputed`` and never
overwrite accepted baselines.

Every rendered figure is written as a high-resolution PNG for the web and a
vector PDF for papers and presentation slides under
``benchmarks/outputs/argha_roy_carbon_sii/figures``.

The complete executable source, rendered figure, and downloadable notebook
are available at
:doc:`gen_benchmarks/plot_argha_roy_carbon_sii`.

Scientific provenance and data notice
-------------------------------------

The Otter side uses the average-atom and pseudoatom/QOZ construction of
:cite:t:`StarrettSaumon2013,StarrettSaumon2014` with the
:cite:t:`Chabrier1990` static jellium local-field correction.

The reference numerical files are attributed to Dr. Argha Roy (private
communication; unpublished), published by maintainer decision with license
status ``NOASSERTION``, and not covered by Otter's BSD-3-Clause software
license.  This repository policy does not assert an open-data license or
replace permission from the provider.  The decision and every reference
checksum are stored in
``benchmarks/reference_data/argha_roy_carbon_sii/manifest.json``.
