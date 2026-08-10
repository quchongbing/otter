Self-consistent ion-structure feedback
======================================

.. warning::

   The AA :math:`\leftrightarrow` QOZ/HNC feedback implementation is
   **experimental and not a production workflow**.  The production default is
   the ion-sphere (IS) construction.

The module :mod:`otter.experimental.sc_feedback` implements an outer iteration
inspired by the self-consistent construction of
:cite:t:`StarrettSaumon2014`.  Starting from a completed IS calculation, it:

#. forms the ionic background seen by each average atom from the current
   :math:`g_{ij}(r)`;
#. estimates an ion--electron correlation potential;
#. reruns the selected full/external average atom (Kohn--Sham or
   Thomas--Fermi) at the fixed IS chemical potential and, for mixtures, fixed
   IS volume partition;
#. reconstructs the pseudoatoms and solves QOZ/HNC again;
#. repeats until both :math:`g_{ij}` and the correlation potential satisfy
   their requested tolerances.

For a mixture, the central species receives the charge-weighted sum of all
pair channels, not only its diagonal :math:`g_{ii}`.  This multicomponent
extension is especially provisional: the published mixture calculations in
:cite:t:`StarrettEtAl2014` used the IS approximation.

Implemented single-component coupling
-------------------------------------

For one component the implementation follows Sec. 2.4 of
:cite:t:`StarrettSaumon2014` directly.  The QOZ/HNC result replaces the
ion-sphere step in the source of the full and external potentials, Eqs. (4)
and (7), and the same additive correlation potential enters both maps:

.. math::

   V_{\mathrm{Ie}}^{\mathrm C}(r)
   =-\frac{n_{\mathrm I}^{0}}{\beta}
     \int \widetilde C_{\mathrm{Ie}}(
       |\mathbf r-\mathbf r'|, n_{\mathrm e}^{0})
     [g_{\mathrm{II}}(r')-1]\,d\mathbf r'.

This is Eq. (19); the density rescaling of
:math:`\widetilde C_{\mathrm{Ie}}` is Eq. (20).  The field-free density
:math:`n_{\mathrm e}^{0}` is evaluated at the chemical potential of the
preceding IS solve.  In particular, an SC electronic step does **not**
reimpose the ion-sphere neutrality condition (3).  The TF backend also
disables its analytic sharp-step Hartree shortcut when a tabulated
:math:`g_{\mathrm{II}}` is supplied, so the feedback is not silently replaced
by the IS source.

The electronic payload retains ``g_ii_background``, ``v_corr_full`` and
``v_corr_ext`` on its radial grid.  The final ``sc_feedback`` metadata also
retains the mixed correlation potential and grid used by the last outer
iteration, making Eq. (19) auditable.

Minimal diagnostic use
----------------------

.. code-block:: python

   from otter import PlasmaWorkflowConfig, solve_plasma_workflow
   from otter.experimental import (
       SCFeedbackConfig,
       solve_sc_feedback_workflow,
   )

   config = PlasmaWorkflowConfig(
       elements=["C"],
       temperature_ev=20.0,
       ion_temperature_ev=20.0,
       rho_g_cc=3.7,
       aa_overrides={"bound_occ_mode": "fd"},
   )

   is_result = solve_plasma_workflow(config)
   sc_result = solve_sc_feedback_workflow(
       config,
       is_result,
       feedback_cfg=SCFeedbackConfig(
           max_outer=10,
           g_tol=5.0e-4,
           v_corr_tol=5.0e-4,
           v_corr_mix=0.35,
           require_converged=True,
       ),
   )

   history = sc_result["sc_feedback"]["history"]

Keep ``is_result`` as the comparison reference.  The experimental API fails
closed by default when the outer iteration is unconverged.  Set
``require_converged=False`` only to inspect an explicitly labelled
best-effort diagnostic; the portable production state writer rejects such a
result.

Scope and limitations
---------------------

* Both the orbital Kohn--Sham and Thomas--Fermi electronic backends implement
  the single-component coupling above.  TF has no discrete bound-orbital
  table.
* ``bound_occ_mode="fd"`` is required by the Kohn--Sham path; it is not a TF
  parameter.
* The IS chemical potential and mixture volume partition remain fixed during
  the outer iteration.
* Convergence of the two numerical differences does not establish uniqueness
  or improved physical accuracy.
* Pressure-ionization thresholds can still make the inner AA solve
  discontinuous or expensive.
* The mixture generalization needs independent literature and simulation
  validation before production use.

The returned ``sc_feedback`` metadata records tolerances, mixing, fixed
chemical potential, volume weights, per-iteration changes, and convergence.
These fields should accompany every reported comparison.

API
---

.. currentmodule:: otter.experimental

.. autosummary::
   :toctree: ../_autosummary

   SCFeedbackConfig
   solve_sc_feedback_workflow
   mixture_ionic_background_profiles
   estimate_mixture_correlation_potentials
