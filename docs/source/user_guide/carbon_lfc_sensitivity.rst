Carbon local-field-correction sensitivity
=========================================

This model-sensitivity example isolates the static jellium electron
local-field correction (LFC) for carbon at
:math:`\rho=5\ {\rm g\,cm^{-3}}`.  At each temperature, all five paths reuse
one strict KS average-atom/external-pseudoatom result: the same
:math:`\bar Z`, :math:`n_{\rm scr}(r)`, and finite-temperature Lindhard
:math:`\chi^0_{ee}(k)`.  Only :math:`G_{ee}(k)` and the downstream
:math:`\chi_{ee}`, :math:`V_{ii}`, QOZ/HNC, :math:`g_{ii}`, and
:math:`S_{ii}` change.

The models are RPA (:math:`G_{ee}=0`), Hubbard
:cite:p:`Hubbard1958`, Utsumi--Ichimaru
:cite:p:`UtsumiIchimaru1982`, Chabrier-1990
:cite:p:`Chabrier1990`, and the Gregori finite-temperature interpolation
:cite:p:`GeldartVosko1966,GregoriEtAl2007`.  Chabrier-1990 is used as the
comparison reference; this is a sensitivity study, not a claim that the
reference is exact.  The pseudoatom/QOZ construction follows
:cite:t:`StarrettSaumon2014`.

Example and execution mode
--------------------------

:ref:`sphx_glr_gen_examples_plot_carbon_lfc_sensitivity.py`

The example begins with one user-facing switch:

.. code-block:: python

   RECOMPUTE_WITH_OTTER = False

``False`` verifies and plots the checksummed v2 result.  Changing it to
``True`` runs Otter's full
``KS AA -> external pseudoatom -> five LFC/QOZ/HNC`` calculation and writes a
new candidate below
``benchmarks/outputs/carbon_lfc_sensitivity/gallery_recomputed``.
The downloadable example file directly calls
``solve_plasma_workflow`` and
``continue_plasma_workflow_from_electronic_result``; it does not load a
separate Python producer.  Recalculation never overwrites reviewed data or
silently continues an unconverged electronic state.

The two states are :math:`T_e=T_i=2` and :math:`100` eV.  The cold state
exposes LFC sensitivity in the ionic correlations; the hot state provides a
weak-correlation comparison.  All displayed curves and printed metrics are
computed from the selected data source by the same standalone gallery file.

Why overlapping response curves can give different potentials
---------------------------------------------------------------

The response panel now places :math:`G_{ee}(k)/k^2` between
:math:`\chi_{ee}(k)` and :math:`V_{ii}(k)`, and the example includes a
dedicated four-panel low-:math:`k` audit.  This distinction is required
because the Starrett--Saumon reduction contains the inverse response
:cite:p:`StarrettSaumon2014`,

.. math::

   V_{ii}(k)=\frac{4\pi\bar Z^2}{k^2}
             +\frac{q(k)^2}{\chi_{ee}(k)},
   \qquad q(k)=n_{\rm scr}(k).

For the LFC models used here,
:math:`G_{ee}(k)=a k^2+O(k^4)`.  Their interacting responses therefore share
the leading Coulomb limit

.. math::

   \chi_{ee}(k)=-\frac{k^2}{4\pi}+O(k^4),

and look almost identical close to the origin.  The corresponding LFC
contribution to the effective potential instead approaches the finite,
model-dependent value

.. math::

   V_{\rm LFC}(k)
   =\frac{4\pi}{k^2}G_{ee}(k)q(k)^2
   \longrightarrow 4\pi a\bar Z^2.

The diagnostic evaluates the same stable three-term decomposition used by
the production QOZ path,

.. math::

   V_{ii}=V_{\rm charge}+V_{\rm LFC}+V_{\chi_0},

and verifies that it reconstructs the archived :math:`V_{ii}`.  At the first
2 eV reciprocal point, :math:`k=0.0563291\ {\rm Bohr}^{-1}`, the RPA and
Chabrier responses differ by only :math:`0.0309\%`, while their
:math:`V_{ii}` values differ by
:math:`19.56\ {\rm Ha\,Bohr^3}`.  The printed table and the low-:math:`k`
figure expose this inverse-response amplification directly.  The shared
:math:`q(k)`, :math:`\chi_0(k)`, and :math:`\bar Z` are kept fixed, so the
comparison isolates the LFC.

Strict threshold-state audit
----------------------------

At :math:`100` eV, a finite Dirichlet box returns a very diffuse 2s
eigenvalue close to the nominal zero.  It lies *above* the bound/continuum
edge used by the AA density partition,

.. math::

   E_{\rm cut}=V_{\rm eff}(0.7R_{\rm max}),

and therefore is not part of :math:`n_{\rm bound}` or :math:`n_{\rm ion}`.
An earlier diagnostic nevertheless inspected every nominally negative box
eigenvalue and incorrectly labelled this excluded level ``unresolved``.

Otter now applies the same :math:`E_{\rm cut}` in the density construction
and the reliability test.  For each included level the diagnostic uses the
gauge-invariant binding energy :math:`E_{\rm cut}-E`, and compares the outer
potential to :math:`V_{\rm eff}-E_{\rm cut}`.  A common energy offset
therefore cannot change the decision.

At the literature-resolution grid, the 2s state remains occupied but a
Dirichlet wall spans too few decay lengths.  Otter therefore imposes an
analytic decaying exterior condition at the common outer SCF boundary and
normalizes the orbital through infinity.  The match is accepted only when
the outer :math:`V_{\rm eff}` is small on the candidate binding-energy scale.
This numerical treatment is motivated by the negative-energy asymptotics in
:cite:t:`StarrettEtAl2019` and :cite:t:`WilsonEtAl2006`; it is not an
identical reproduction of either implementation.  It neither sets
``allow_unconverged_aa`` nor creates a separate enlarged bound-only box.

The producer uses 4096 radial points, consistent with the 4095-point
quadratic grid in :cite:t:`StarrettSaumon2014`, Appendix B, after a
1024/2048/4096 threshold audit.  The v2 archives record the continuum edge,
shallowest included bound energy, threshold classification, full/external
SCF convergence, HNC residual, closure-transform mismatch, raw screening
charge, and QOZ charge.  The manifest records the matching mode explicitly
as ``outer_scf_exterior_match`` and ``bound_rmax_mult`` as ``null``: this
label denotes an exterior match on the common outer SCF domain, not a
separate enlarged bound-state box.

V2 data protocol
----------------

For release engineering, the independent producer is
``benchmarks/runners/regenerate_carbon_lfc_sensitivity.py``; the independent
validator/plotter is
``benchmarks/runners/plot_carbon_lfc_sensitivity.py``.  Candidate archives use
``otter_carbon_lfc_sensitivity_state_v2`` and a manifest with
``otter_benchmark_manifest_v2``.

The manifest fixes:

* :math:`\rho=5\ {\rm g\,cm^{-3}}`, both temperatures, and the five-model
  order;
* 4096-point AA and 8192-point pre-padding QOZ grids;
* FD bound occupation, the full B3 tail, finite-temperature Lindhard
  response, and strict convergence tolerances;
* units, method citations, data rights, producer script checksum, dirty-tree
  disclosure, and each NPZ SHA-256 digest.

The NPZ files are pickle-free and contain no absolute paths.  They retain
:math:`r,k\leq20` in Bohr units.  Densities use Bohr\ :sup:`-3`,
:math:`\chi` uses Bohr\ :sup:`-3` Hartree\ :sup:`-1`,
:math:`V_{ii}(r)` uses Hartree, and :math:`V_{ii}(k)` uses Hartree
Bohr\ :sup:`3`; :math:`n_{\rm scr}(k)` is an electron number.

The reviewed v2 files live in
``benchmarks/baselines/carbon_lfc_sensitivity``.  Their manifest records a
clean Otter producer commit, the producer-script SHA-256, strict AA/external
and HNC convergence metadata, and each NPZ checksum.  A fresh run still writes
only to the ignored ``recomputed`` directory; promotion remains a deliberate
maintainer review action rather than a side effect of the generator.

Those maintenance programs are not required by the downloadable gallery
example.  The gallery is itself a complete standalone calculation and plot
script.  For the separate release-audit figure and CSV metric table:

.. code-block:: console

   python benchmarks/runners/plot_carbon_lfc_sensitivity.py
