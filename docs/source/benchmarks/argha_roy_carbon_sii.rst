Carbon PA-HNC and DFT-MD comparison
===================================

This benchmark compares Otter carbon static ion structure factors with
unpublished DFT-MD data provided by Dr. Argha Roy (private communication) at

.. math::

   \rho=3.51538\ {\rm g\,cm^{-3}},\qquad
   T_e=T_i=20,30,40,50,100\ {\rm eV}.

Otter uses the average-atom and pseudoatom/QOZ method of
:cite:t:`StarrettSaumon2013,StarrettSaumon2014` with the static jellium LFC of
:cite:t:`Chabrier1990`.  Each displayed Otter result passes full/external AA,
threshold-state, HNC fixed-point, and transform-closure checks.

The reference wavenumber is in :math:`\mathrm{\AA}^{-1}`; only the Otter
coordinate is converted before comparison.  The third reference column is
shown as *reported uncertainty*.  It is not identified as one standard
deviation because that interpretation has not been confirmed.

Reproduce the comparison
------------------------

The downloadable script :doc:`gen_benchmarks/plot_argha_roy_carbon_sii`
contains the calculation and plotting code.  Set

.. code-block:: python

   USE_PRECOMPUTED_DATA = True

to verify and load the five checksummed Otter results.  Set it to ``False``
to run all five states and write new results under
``benchmarks/outputs/argha_roy_carbon_sii/gallery_recomputed``.  Existing
accepted files are not overwritten.  The figure is exported as PNG and PDF.

Reference-data notice
---------------------

The DFT-MD coordinates are attributed to Dr. Argha Roy and remain
unpublished.  No open-data license is recorded, so the repository assigns
license status ``NOASSERTION`` and does not cover these values under Otter's
BSD software license.  Per-file checksums and attribution are stored in
``benchmarks/reference_data/argha_roy_carbon_sii/manifest.json``.
