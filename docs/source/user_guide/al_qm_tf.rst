Aluminium KS-DFT versus Thomas--Fermi
=====================================

This capability example compares the Kohn--Sham density-functional-theory
(KS-DFT, internally ``qm``) and finite-temperature Thomas--Fermi (TF)
electronic backends in Otter.  Both electronic models feed the same
ion-sphere pseudoatom, QOZ/HNC, and Chabrier-1990 local-field-correction
construction.  The sequence

.. math::

   \rho=8.1\ {\rm g\,cm^{-3}},\qquad
   T_e=T_i\in\{1,15,50,100\}\ {\rm eV}

therefore isolates temperature-dependent electronic-model sensitivity
without simultaneously changing the ion density.

The TF construction follows Eqs. (2)--(11) of
:cite:t:`StarrettSaumon2014`, while the finite-temperature jellium
local-field correction follows :cite:t:`Chabrier1990`.  These citations
define the methods; no numerical curve is extracted from either paper.

Example and execution modes
---------------------------

:ref:`sphx_glr_gen_examples_plot_al_qm_tf.py`

The example has one user setting:

.. code-block:: python

   RECOMPUTE_WITH_OTTER = False

The default verifies and loads four checksummed Otter state files.  Set
``RECOMPUTE_WITH_OTTER=True`` to make the same script construct
``PlasmaWorkflowConfig`` objects and call ``solve_plasma_workflow`` for all
eight calculations.  New files are written under
``benchmarks/outputs/al_qm_tf/gallery_recomputed`` and never overwrite or
replace accepted data automatically.

What is plotted
---------------

The electronic figure contains exactly two quantities:

* :math:`4\pi r^2[n_{\rm full}(r)-n_0]` for KS-DFT and TF, shown over
  :math:`-1\leq r\leq8` Bohr; and
* :math:`4\pi r^2n_{\rm scr}(r)` for both models, shown over
  :math:`-1\leq r\leq12` Bohr.

The negative lower limits are visual margins only; the radial grids begin at
positive :math:`r`.  The ionic-density partition is retained in the portable
state schema for diagnostics but is intentionally not drawn in this
comparison.  Separate panels propagate the same electronic results into
:math:`g_{ii}(r)` and :math:`S_{ii}(k)`.

Temperature sequence
--------------------

At 1 and 15 eV, the orbital shell structure produces a large difference in
the pseudoatom-partition ionization and in the screening cloud.  The
difference decreases at 50 eV.  By 100 eV, the two ionic structures are very
close for this state even though the electronic decompositions are not
identical.  The four points illustrate a trend; they do not define a
universal temperature boundary for the validity of TF theory.

The table below is recomputed from the reviewed v2 files.  ``Delta Z`` is
:math:`\bar Z_{\rm TF}-\bar Z_{\rm KS-DFT}` using the QOZ
pseudoatom-partition ionization.  RMSEs use :math:`r\leq12` Bohr and
:math:`k\leq6` Bohr\ :sup:`-1`.

.. list-table::
   :header-rows: 1
   :widths: 12 15 15 15

   * - :math:`T` [eV]
     - :math:`\Delta Z`
     - RMSE :math:`g_{ii}`
     - RMSE :math:`S_{ii}`
   * - 1
     - +1.84337
     - 0.06149
     - 0.07227
   * - 15
     - +1.81427
     - 0.04222
     - 0.03499
   * - 50
     - +0.67313
     - 0.01246
     - 0.01197
   * - 100
     - -0.18648
     - 0.00209
     - 0.00160

Numerical record
----------------

The four pickle-free NPZ files record converged full/external AA stages,
threshold-state status, HNC residuals below :math:`10^{-6}`, transform-closure
mismatches below :math:`10^{-3}`, and the model settings and checksums needed
to reproduce the figures.  Arrays are restricted to :math:`r,k<20` in the
following units:

* density: Bohr\ :sup:`-3`
* radius: Bohr
* wavenumber: Bohr\ :sup:`-1`
* chemical and real-space potential energy: Hartree
* reciprocal-space pair potential: Hartree Bohr\ :sup:`3`
* :math:`g_{ii}` and :math:`S_{ii}`: dimensionless

The API model labels are ``("qm", "tf")``; figures identify ``qm`` as
KS-DFT.
