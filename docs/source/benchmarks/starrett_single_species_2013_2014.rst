Starrett--Saumon single-species ion structure
==============================================

This benchmark compares Otter with local numerical extractions from
:cite:t:`StarrettSaumon2014`, *High Energy Density Physics* **10**, 35--42
(2014), `doi:10.1016/j.hedp.2013.12.001
<https://doi.org/10.1016/j.hedp.2013.12.001>`__
(PII ``S1574181813001900``). Starrett TF values are open circles; accepted
Otter QM and TF calculations are solid blue and dashed orange lines.

The directly executable program contains the state definitions, public
:class:`~otter.PlasmaWorkflowConfig` calls, strict reliability gates,
checksum validation, and plotting:

:ref:`sphx_glr_benchmarks_gen_benchmarks_plot_starrett_single_species_2013_2014.py`

``USE_PRECOMPUTED_DATA = True`` loads checksummed current-Otter baselines.
Set it to ``False`` to calculate the selected states locally. The script saves
both PNG and PDF.

Accepted calculations
---------------------

Both QM and TF pass for H at 5 and 172 eV and C at 64.64 eV. Fe at 10 eV and
W at 10 eV have accepted TF curves. The W 10 eV TF calculation uses an
8192-point QOZ grid; all other accepted curves use 4096 points before padding.

Fe QM and W 60 eV TF remain above the independent transform-closure limit
after a 4096-to-8192-point grid check. W 10 and 60 eV QM do not reach a
physical HNC fixed point. These attempts and numerical reasons are retained
in the baseline manifest, but failed iterates are not plotted.

Method
------

The IS-QM path solves the finite-temperature KS average atom; the IS-TF path
uses the Thomas--Fermi electronic model. Both feed the same pseudoatom,
finite-temperature Lindhard response, Chabrier LFC, and QOZ/HNC workflow
described by :cite:t:`StarrettSaumon2014`. Ordinary Fermi--Dirac bound
occupations are used. A candidate must pass full and external electronic
convergence, the threshold-state check, the nonlinear HNC residual, and the
independent Fourier-transform closure check.

Reference-data notice
---------------------

The local CSV files do not retain an open-data license or a complete
digitization history.  They are published by maintainer decision with source
attribution and license status ``NOASSERTION`` and are not covered by Otter's
BSD-3-Clause source-code license.  This repository policy does not assert a
publisher- or author-supplied data license.
