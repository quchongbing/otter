Quick start
===========

Unified workflow
----------------

The main entry point is :class:`otter.PlasmaWorkflowConfig`.  The following
configuration describes carbon at a mass density of
:math:`3.7\,\mathrm{g\,cm^{-3}}` and an electron temperature of 100 eV:

.. code-block:: python

   from otter import PlasmaWorkflowConfig, solve_plasma_workflow

   config = PlasmaWorkflowConfig(
       elements=["C"],
       temperature_ev=100.0,
       rho_g_cc=3.7,
   )
   result = solve_plasma_workflow(config)

With no ion temperature, the workflow stops after electronic structure.  Set
``ion_temperature_ev`` to continue through the pseudoatom QOZ/HNC calculation:

.. code-block:: python

   config = PlasmaWorkflowConfig(
       elements=["C"],
       temperature_ev=100.0,
       ion_temperature_ev=100.0,
       rho_g_cc=3.7,
   )
   result = solve_plasma_workflow(config)

   electronic = result["electronic"]
   ionic = result["ion"]

Mixtures
--------

Composition can be given either as a formula or as explicit species counts:

.. code-block:: python

   ch = PlasmaWorkflowConfig(
       elements=["C", "H"],
       counts=[1.0, 1.36],
       temperature_ev=8.617333,
       ion_temperature_ev=8.617333,
       rho_g_cc=2.94,
   )

For non-integer mixtures, explicit ``counts`` avoids ambiguity in formula
parsing.

Runtime and cached electronic structure
---------------------------------------

Quantum continuum calculations can take minutes or longer, particularly near
pressure ionization.  The unified workflow therefore separates the expensive
electronic stage from the downstream ionic stage.  The function
:func:`otter.continue_plasma_workflow_from_electronic_result` can reuse a
previously validated electronic result while changing QOZ/HNC controls.

Do not interpret solver completion alone as physical validation.  Inspect
charge closure, SCF convergence, HNC residuals, and the model's applicability;
the :doc:`benchmarks/index` pages show the expected reporting pattern.

Consistent PNG and PDF figures
------------------------------

The optional plotting helpers apply the same typography, sizes, ticks, and
color cycles throughout Otter.  ``save_figure`` writes a web PNG and a vector
PDF from the same Matplotlib figure:

.. code-block:: python

   import matplotlib.pyplot as plt
   from otter.plotting import save_figure, set_style

   set_style("docs", palette="nature")
   fig, ax = plt.subplots()
   ax.plot(ionic["k"], ionic["sii_k"])
   ax.set(xlabel=r"$k$ [Bohr$^{-1}$]", ylabel=r"$S_{ii}(k)$")

   paths = save_figure(fig, "outputs/carbon_sii")
   print(paths["png"], paths["pdf"])

No additional plotting package is introduced: these helpers use Matplotlib,
which is installed by the existing ``otter[plot]`` optional dependency.
