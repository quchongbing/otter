Quick start
===========

Single-species aluminium
------------------------

Run the :download:`single-species workflow
<../../examples/single_species_workflow.py>` from the repository root:

.. code-block:: console

   $ poetry run python examples/single_species_workflow.py

The default state is Al at :math:`8.1\,\mathrm{g\,cm^{-3}}` and
:math:`T_e=T_i=15\,\mathrm{eV}`.  The input block controls the state and output.
The script plots the electronic density, effective potential,
:math:`g_{ii}(r)`, and :math:`S_{ii}(k)`.

Mixtures
--------

Run the :download:`mixture workflow <../../examples/mixture_workflow.py>`:

.. code-block:: console

   $ poetry run python examples/mixture_workflow.py

Runtime and cached electronic structure
---------------------------------------

Quantum continuum calculations can be slow near pressure ionization.  The
function
:func:`otter.continue_plasma_workflow_from_electronic_result` can reuse a
validated electronic result while changing QOZ/HNC controls.

Do not interpret solver completion alone as physical validation.  Inspect
charge closure, SCF convergence, HNC residuals, and the model's applicability;
the :doc:`benchmarks/index` pages show the expected reporting pattern.

Consistent PNG and PDF figures
------------------------------

``save_figure`` writes PNG and PDF files from the same Matplotlib figure:

.. code-block:: python

   import matplotlib.pyplot as plt
   from otter.plotting import save_figure, set_style

   set_style("docs", palette="nature")
   fig, ax = plt.subplots()
   ax.plot(ionic["k"], ionic["sii_k"])
   ax.set(xlabel=r"$k$ [Bohr$^{-1}$]", ylabel=r"$S_{ii}(k)$")

   paths = save_figure(fig, "outputs/al_sii")
   print(paths["png"], paths["pdf"])
   plt.show()
