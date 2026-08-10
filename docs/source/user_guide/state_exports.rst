Portable plasma-state files
===========================

Otter can export the quantities most often exchanged between pseudoatom,
QOZ/HNC, and XRTS calculations as a portable NumPy ``.npz`` file:

.. math::

   q_i(k) = n_{\mathrm{scr},i}(k), \qquad
   f_i(k) = n_{\mathrm{ion},i}(k),

together with :math:`g_{ij}(r)` and :math:`S_{ij}(k)`.  The default file
contains only :math:`r < 20\,a_{\rm B}` and
:math:`k < 20\,a_{\rm B}^{-1}`.  Both limits are **exclusive**.

The export is deliberately restricted to numeric and fixed-width Unicode
arrays.  It can therefore be loaded with ``allow_pickle=False`` and does not
embed Python objects or workstation-specific paths.  The archive is written
beside its destination and atomically replaced only after the write completes.

Save directly from a workflow
-----------------------------

Set ``save_state_npz`` on a workflow that includes the ion-structure stage:

.. code-block:: python

   from otter import PlasmaWorkflowConfig, solve_plasma_workflow

   config = PlasmaWorkflowConfig(
       elements=["C", "H"],
       counts=[1.0, 1.36],
       temperature_ev=8.617333,
       ion_temperature_ev=8.617333,
       rho_g_cc=2.94,
       save_state_npz=True,
       save_state_path="outputs/ch1p36_state.npz",
       state_r_max_bohr=20.0,
       state_k_max_bohr_inv=20.0,
   )
   result = solve_plasma_workflow(config)
   print(result["saved_paths"]["state_npz"])

An ion temperature is required because :math:`q(k)`, :math:`f(k)`, and
:math:`S_{ij}(k)` must share the actual converged QOZ/DST reciprocal-space
lattice.  Otter rejects a missing or unconverged HNC status by default.  For
experimental SC feedback it also requires an explicitly converged outer
feedback status.

Save an existing result
-----------------------

A completed in-memory workflow can be exported separately:

.. code-block:: python

   from otter import StateExportOptions, save_plasma_state

   path = save_plasma_state(
       "outputs/state.npz",
       result,
       options=StateExportOptions(
           r_max_bohr=12.0,
           k_max_bohr_inv=15.0,
           require_converged_hnc=True,
           compressed=True,
       ),
   )

Setting ``require_converged_hnc=False`` is intended only for a clearly
labelled diagnostic snapshot.  Such a file is not a physical production
result.

Load and inspect
----------------

Use :func:`otter.load_plasma_state` to validate the schema while loading:

.. code-block:: python

   import json
   from otter import load_plasma_state

   state = load_plasma_state("outputs/state.npz")
   metadata = json.loads(str(state["metadata_json"].item()))

   i_c = list(state["species_symbols"]).index("C")
   k = state["k_bohr_inv"]
   q_c = state["q_k"][i_c]
   f_c = state["f_k"][i_c]
   s_cc = state["sij_k"][i_c, i_c]

The same file may also be inspected directly with
``numpy.load(path, allow_pickle=False)``.

Schema and array shapes
-----------------------

For :math:`N_s` species, :math:`N_r` retained radial points, and :math:`N_k`
retained reciprocal-space points, the principal arrays are:

.. list-table::
   :header-rows: 1
   :widths: 24 24 52

   * - Key
     - Shape
     - Meaning
   * - ``species_symbols``
     - ``(N_s,)``
     - Species order used by every species and pair axis.
   * - ``r_bohr``
     - ``(N_r,)``
     - Radial grid in Bohr.
   * - ``k_bohr_inv``
     - ``(N_k,)``
     - Reciprocal grid in inverse Bohr.
   * - ``n_ion_r``
     - ``(N_s, N_r)``
     - Ion-associated electron density in :math:`a_{\rm B}^{-3}`.
   * - ``n_scr_r``
     - ``(N_s, N_r)``
     - Charge-closed screening density in :math:`a_{\rm B}^{-3}`.
   * - ``f_k`` / ``n_ion_k``
     - ``(N_s, N_k)``
     - Identical aliases for :math:`n_{\rm ion}(k)`.
   * - ``q_k`` / ``n_scr_k``
     - ``(N_s, N_k)``
     - Identical aliases for the screening cloud used by QOZ.
   * - ``gij_r``
     - ``(N_s, N_s, N_r)``
     - Pair distribution functions.
   * - ``sij_k``
     - ``(N_s, N_s, N_k)``
     - Ashcroft--Langreth partial static structure factors.
   * - ``vij_k``
     - ``(N_s, N_s, N_k)``
     - Effective pair potentials, when present in the workflow.
   * - ``metadata_json``
     - scalar string
     - Units, state, models, windows, and convergence diagnostics.

The pair axes always follow ``species_symbols``; for example, element
``sij_k[i, j]`` is the :math:`i`--:math:`j` partial.  The aliases are stored
explicitly so that both pseudoatom and XRTS notation remain unambiguous.

API
---

The high-level public functions are :class:`otter.StateExportOptions`,
:func:`otter.save_plasma_state`, and :func:`otter.load_plasma_state`.
Lower-level schema construction and validation are available from
:mod:`otter.io.state`.
