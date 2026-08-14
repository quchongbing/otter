API reference
=============

Stable high-level interface
---------------------------

The public interface is intentionally smaller than the internal solver
implementation.  Applications should begin with the unified workflow objects
below.

.. currentmodule:: otter

.. autosummary::
   :toctree: ../_autosummary

   PlasmaWorkflowConfig
   solve_plasma_workflow
   continue_plasma_workflow_from_electronic_result
   run_formula_workflow
   parse_formula_composition
   resolve_plasma_composition
   StateExportOptions
   save_plasma_state
   load_plasma_state

Portable state schema
---------------------

The stable state-file interface stores the native average-atom profiles and
levels together with :math:`q(k)`, :math:`f(k)`, electron response and LFC,
interaction channels, :math:`V_{ij}`, :math:`g_{ij}`, and :math:`S_{ij}`
without pickled Python objects.  See
:doc:`../user_guide/state_exports` for shapes, units, and examples.

.. currentmodule:: otter.io.state

.. autosummary::
   :toctree: ../_autosummary

   build_state_arrays
   validate_state_arrays

Scientific plotting
-------------------

Otter provides one shared Matplotlib style and a dual web/slide export path.
By default :func:`save_figure` writes both a high-resolution PNG and an
editable vector PDF.

.. currentmodule:: otter.plotting

.. autosummary::
   :toctree: ../_autosummary

   set_style
   style_context
   style_rcparams
   grid_figsize
   add_panel_label
   save_figure

Internal modules
----------------

The ``otter.electronic``, ``otter.ionic``, ``otter.numerics``, and
``otter.io`` packages contain lower-level model and diagnostic APIs.  They are
documented progressively as their contracts stabilize.  Code outside Otter
should not rely on private names or dictionary fields that are absent from the
public result schema.

Experimental APIs are documented separately under
:doc:`../experimental/index`; their presence does not imply the stability
guarantees of this page.

API stability
-------------

Public additions require:

* a NumPy-style docstring with units and array shapes;
* a literature reference for physical models;
* validation of accepted values and failure modes;
* at least one focused test;
* an example when the behavior is not evident from the signature.
