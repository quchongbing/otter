Exchange-correlation functionals
================================

Otter keeps the spin-unpolarized Dirac exchange model as its dependency-free
default.  Optional Libxc bindings add ground-state LDA and GGA functionals to
both the orbital Kohn--Sham and Thomas--Fermi full/external solvers.

Installation
------------

Libxc is optional, so importing and running the default model does not require
it.  Install the Python bindings from conda-forge before selecting a Libxc
model:

.. code-block:: bash

   conda install -c conda-forge pylibxc

The Libxc project documents its Python interface and available functionals at
https://libxc.gitlab.io/installation/ and
https://libxc.gitlab.io/functionals/.

Citation and provenance
-----------------------

Calculations that use a Libxc-backed model must cite Libxc
:cite:p:`LehtolaEtAl2018` and the primary papers for every selected functional.
This is the `official Libxc citation policy
<https://libxc.gitlab.io/#citing-libxc>`_: report that Libxc was used, report
its version, and report the references returned by Libxc for the selected
functional IDs.  Reporting the exact implementation is also important for
reproducibility :cite:p:`LehtolaMarques2023`.

Otter follows that policy programmatically.  ``xc_provenance(model)`` queries
the installed pylibxc runtime and returns the Libxc version, exact string and
numeric functional IDs, software citation, and the references/DOIs supplied
by each functional.  High-level electronic results retain this dictionary as
``result["xc_provenance"]`` and saved metadata retains the same information.
The carbon comparison additionally writes ``xc_provenance.json`` and
``CITATIONS.md`` beside its numerical archives and prints a concise provenance
notice once at startup.

.. code-block:: python

   from otter.electronic import xc_provenance

   citation_record = xc_provenance("pbe")

Selecting a model
-----------------

Set xc_model on the top-level plasma workflow or the direct electronic
configuration:

.. code-block:: python

   from otter import PlasmaWorkflowConfig, solve_plasma_workflow

   config = PlasmaWorkflowConfig(
       elements=["C"],
       temperature_ev=100.0,
       rho_g_cc=3.7,
       xc_model="pbe",
   )
   result = solve_plasma_workflow(config)

The same option is accepted by FullExternalConfig, ThomasFermiConfig,
KSDTFConfig, and run_minimal.

The built-in aliases are:

.. list-table::
   :header-rows: 1

   * - Model
     - Components
     - Primary functional references
   * - dirac
     - Built-in Dirac exchange; no Libxc dependency
     - :cite:p:`Dirac1930`
   * - none
     - Zero exchange-correlation potential
     - Not applicable
   * - pbe
     - gga_x_pbe + gga_c_pbe
     - :cite:p:`PerdewBurkeErnzerhof1996,PerdewBurkeErnzerhof1997`
   * - lda_pw
     - lda_x + lda_c_pw
     - :cite:p:`Dirac1930,Bloch1929,PerdewWang1992`
   * - lda_pz
     - lda_x + lda_c_pz
     - :cite:p:`Dirac1930,Bloch1929,PerdewZunger1981`
   * - lda_vwn
     - lda_x + lda_c_vwn
     - :cite:p:`Dirac1930,Bloch1929,VoskoWilkNusair1980`

Other spin-unpolarized LDA and GGA combinations use a plus-separated explicit
Libxc specification, for example:

.. code-block:: python

   config = PlasmaWorkflowConfig(
       elements=["Al"],
       temperature_ev=10.0,
       rho_g_cc=2.7,
       xc_model="libxc:gga_x_pbe+gga_c_pbe",
   )

GGA potential
-------------

For a spherical density, Libxc returns partial derivatives with respect to the
density and the contracted gradient.  Otter evaluates the exact derivative of
its shell-weighted discrete XC energy.  If ``D`` is the radial derivative
matrix and ``W`` contains the spherical shell volumes, the implemented form is

.. math::

   v_{xc} = v_{\rho} + W^{-1}D^T
      \left[2 W v_{\sigma} s^2 Dn\right].

Here :math:`v_\sigma` is the partial derivative of the energy density with
respect to the squared density gradient.  On the production square-root grid,
``D`` differentiates in the uniform :math:`\xi=\sqrt r` coordinate and applies
the analytic chain rule.  The transpose form is the discrete analogue of the
spherical divergence and is variationally consistent with the reported XC
energy density.

Finite GGA core regularization
------------------------------

GGA functionals contain a second radial derivative and can amplify tiny
origin-grid errors in all-electron densities.  Otter therefore uses a finite
GGA core by default in its electronic workflows.  For nuclear charge ``Z``,
the transition radius is

.. math::

   r_c = \mathtt{gga\_core\_zr}/Z,

with ``gga_core_zr=0.05``.  Inside that radius the density gradient is
multiplied by the C2 switch

.. math::

   s(t)=10t^3-15t^4+6t^5, \qquad t=r/r_c,

and :math:`s=1` outside.  Both the energy input
:math:`\sigma=(s Dn)^2` and its potential derivative use the same switch, so
this is a defined regularized functional rather than clipping the resulting
potential.  At the nucleus it approaches the zero-gradient LDA limit; outside
the core it is exactly the selected GGA.

This switch and Otter's shell-weighted discrete adjoint are Otter numerical
methods.  They are not part of PBE :cite:p:`PerdewBurkeErnzerhof1996` and are
not supplied by Libxc :cite:p:`LehtolaEtAl2018`.  Consequently the output
records ``xc_provenance`` and the separate ``gga_core_*`` fields; citing the
PBE or Libxc papers must not be used to attribute this regularization to those
works.

The behavior is controlled at any high-level entry point:

.. code-block:: python

   config = PlasmaWorkflowConfig(
       elements=["C"],
       temperature_ev=2.0,
       rho_g_cc=1.0,
       xc_model="pbe",
       gga_core_mode="finite",  # default
       gga_core_zr=0.05,
   )

Use ``gga_core_mode="strict"`` only to inspect or reproduce the unregularized
GGA nuclear behavior.  The finite mode requires at least eight radial points
inside ``r_c`` and reports an actionable resolution error otherwise.  The
uniform background subtraction uses the same switch, so its gradient term
still vanishes exactly.  Result metadata records the mode, radius, number of
core points, density-cusp error, and nuclear-potential turning diagnostic.

Current scope
-------------

Only unpolarized LDA and GGA functionals are supported.  Meta-GGAs require
kinetic-energy-density or Laplacian inputs that the current density-only
solver interface does not provide.  Hybrid functionals additionally require
a nonlocal exact-exchange operator.  Otter rejects both families with an
explicit error instead of silently dropping their missing terms.

PBE and the listed LDA aliases are ground-state functionals.  Selecting one at
finite electron temperature does not turn it into a finite-temperature XC
free-energy functional.  The result metadata records the selected xc_model so
that this model choice remains visible.
