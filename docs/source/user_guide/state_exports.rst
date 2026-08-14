Workflow results and portable NPZ files
=======================================

The return value of :func:`otter.solve_plasma_workflow` is the primary API.
It contains the converged electronic result and, when ``ion_temperature_ev``
is set, the QOZ/HNC result.  The same state can be written as a portable NPZ
archive.  NPZ files contain only numeric arrays and fixed-width strings and
are loaded with ``allow_pickle=False``.

This page describes the portable workflow-state schema
``otter_state_v3``.  Benchmark baselines may instead use compact,
benchmark-specific schemas because one archive can contain several model or
thermodynamic states.  Such plotting archives are validated by their own
producer/loader and are not inputs to :func:`otter.load_plasma_state`.  Every
project-generated baseline also embeds scalar ``metadata_json`` with its
configuration, state identifier, producer, references, units, convergence
diagnostics, and field inventory; the adjacent manifest records checksums and
package-level provenance.

In-memory access
----------------

For a single species:

.. code-block:: python

   from otter import PlasmaWorkflowConfig, solve_plasma_workflow

   config = PlasmaWorkflowConfig(
       elements=["Al"],
       temperature_ev=15.0,
       ion_temperature_ev=15.0,
       rho_g_cc=8.1,
   )
   result = solve_plasma_workflow(config)
   aa = result["electronic"]["result"]
   ion = result["ion"]

   r_aa = aa["r"]
   n_full = aa["n_full"]
   n_bound = aa["n_bound"]
   n_cont = aa["n_cont"]
   energies = aa["bound_energy_ha"]
   level_density = aa["bound_orbital_density_r"]
   ion_level_density = aa["ion_orbital_density_r"]
   V_eff = aa["v_full"]
   V_nuc = aa["v_nuc"]
   V_H = aa["v_H"]
   V_xc = aa["v_xc"]
   Zbar = aa["zbar_partition"]
   Zstar = aa["n0"] / ion["n_i"]
   mu = aa["mu"]

   k = ion["k"]
   q = ion["q_k"]          # n_scr(k)
   f = ion["f_k"]          # n_ion(k)
   G = ion["g_ee_k"]
   chi0 = ion["chi0_k"]
   chi_ee = ion["chi_ee_k"]
   V_ei = ion["v_ei_k"]
   C_ie = ion["c_ie_k"]
   V_ii = ion["vij_k"][0, 0]
   g_ii = ion["gij_r"][0, 0]
   S_ii = ion["sij_k"][0, 0]

For a mixture, each average-atom result is in
``result["electronic"]["result"]["species"][i]["result"]``.  The order is
``result["species_symbols"]``.  Species axes in QOZ arrays use the same order.

Save and load
-------------

Set ``save_state_npz`` on a workflow that includes the ion-structure stage:

.. code-block:: python

   from otter import PlasmaWorkflowConfig, load_plasma_state, solve_plasma_workflow

   config = PlasmaWorkflowConfig(
       elements=["C", "H"],
       counts=[1.0, 1.36],
       temperature_ev=8.617333,
       ion_temperature_ev=8.617333,
       rho_g_cc=2.94,
       save_state_npz=True,
       save_state_path="outputs/ch1p36_state.npz",
   )
   result = solve_plasma_workflow(config)
   state = load_plasma_state(result["saved_paths"]["state_npz"])

``load_plasma_state`` checks the schema, shapes, aliases, finite values, and
the recorded export windows.  Plain NumPy access is also available when
validation is not required:

.. code-block:: python

   import numpy as np

   with np.load("outputs/ch1p36_state.npz", allow_pickle=False) as archive:
       print(archive.files)
       k = archive["k_bohr_inv"]
       q = archive["q_k"]
       G = archive["g_ee_k"]
       S = archive["sij_k"]

By default the archive retains :math:`r < 20\,a_{\rm B}` and
:math:`k < 20\,a_{\rm B}^{-1}`; both limits are exclusive.  Change them with
``state_r_max_bohr`` and ``state_k_max_bohr_inv``.  A completed in-memory
workflow can also be saved explicitly:

.. code-block:: python

   from otter import StateExportOptions, save_plasma_state

   save_plasma_state(
       "outputs/state.npz",
       result,
       options=StateExportOptions(r_max_bohr=12.0, k_max_bohr_inv=15.0),
   )

Production export requires a converged HNC result.  The file is written
atomically, so an interrupted write does not replace an existing state.

Species and pair axes
---------------------

For :math:`N_s` species, :math:`N_r` common radial points, and :math:`N_k`
common reciprocal points:

.. list-table:: Common QOZ/HNC arrays
   :header-rows: 1
   :widths: 27 23 50

   * - Key
     - Shape
     - Quantity
   * - ``species_symbols``
     - ``(N_s,)``
     - Species order for every species and pair axis.
   * - ``r_bohr``, ``k_bohr_inv``
     - ``(N_r,)``, ``(N_k,)``
     - Common QOZ/DST grids.
   * - ``n_ion_r``, ``n_scr_r``
     - ``(N_s, N_r)``
     - Ion-associated and charge-closed screening densities.
   * - ``f_k`` / ``n_ion_k``
     - ``(N_s, N_k)``
     - Identical aliases for :math:`n_{\rm ion}(k)`.
   * - ``q_k`` / ``n_scr_k``
     - ``(N_s, N_k)``
     - Identical aliases for the screening clouds used by QOZ.
   * - ``chi0_k``, ``chi_ee_k``
     - ``(N_k,)``
     - :math:`\chi^0_{ee}(k)` and :math:`\chi_{ee}(k)`.
   * - ``g_ee_k`` / ``gee_k``
     - ``(N_k,)``
     - Identical aliases for the selected local-field correction
       :math:`G_{ee}(k)`.
   * - ``v_ie_k`` / ``v_ei_k``
     - ``(N_s, N_k)``
     - Electron--ion potential; the two names are explicit aliases.
   * - ``v_ee_k``
     - ``(N_k,)``
     - Electron--electron potential.
   * - ``c_ie_k``, ``c_ee_k``
     - ``(N_s, N_k)``, ``(N_k,)``
     - Electron--ion and electron--electron direct correlations.
   * - ``v_ie_r`` / ``v_ei_r``, ``v_ee_r``
     - species/common radial arrays
     - Finite-DST real-space representations of the corresponding channels.
   * - ``c_ie_r``, ``c_ee_r``
     - species/common radial arrays
     - Finite-DST real-space direct-correlation channels.
   * - ``vij_r``, ``vij_k``
     - ``(N_s, N_s, N_r/k)``
     - Effective ion--ion pair potentials.
   * - ``gij_r``, ``hij_r``, ``cij_r``
     - ``(N_s, N_s, N_r)``
     - Pair, total-correlation, and ion direct-correlation functions.
   * - ``sij_k``
     - ``(N_s, N_s, N_k)``
     - Ashcroft--Langreth partial static structure factors.
   * - ``zbar``, ``zbar_qoz``
     - ``(N_s,)``
     - Mean ionic charge used in QOZ.
   * - ``zbar_partition``, ``zbar_aa_ws``, ``zstar``
     - ``(N_s,)``
     - Pseudoatom-partition, WS, and :math:`Z^*=n_0/n_i` definitions.
   * - ``mu_ha``, ``r_ws_bohr``, ``n0_bohr3``, ``n_i_bohr3``
     - ``(N_s,)``
     - Chemical potential, WS radius, background electron density, and ion
       density.

Pair access is direct.  For example:

.. code-block:: python

   symbols = list(state["species_symbols"])
   i_c = symbols.index("C")
   i_h = symbols.index("H")
   g_ch = state["gij_r"][i_c, i_h]
   s_cc = state["sij_k"][i_c, i_c]

Native average-atom arrays
--------------------------

Each species retains its native electronic grid under a stable prefix:
``species_0_*``, ``species_1_*``, and so on.  Use
``species_symbols`` to identify the prefix.

.. code-block:: python

   i_c = list(state["species_symbols"]).index("C")
   prefix = f"species_{i_c}_"
   r = state[prefix + "r_bohr"]
   n_full = state[prefix + "n_full_r"]
   n_bound = state[prefix + "n_bound_r"]
   V_eff = state[prefix + "v_full_r_ha"]
   V_nuc = state[prefix + "v_nuc_r_ha"]
   V_H = state[prefix + "v_hartree_r_ha"]
   V_xc = state[prefix + "v_xc_r_ha"]

Available profiles are saved when the selected electronic model produces
them.  They include ``n_full_r``, ``n_bound_r``, ``n_cont_r``, ``n_ext_r``,
``n_pa_r``, native ``n_scr_r_native`` and ``n_ion_r_native``, TF positive-
and negative-energy densities, tail/source profiles, and repaired diagnostic
profiles.  Potential fields include the full and external effective
potentials and their nuclear, Hartree, exchange--correlation, and correlation
components.

The direct positive-energy A3 density can have a shorter numerical domain.
It therefore uses its own paired arrays ``species_i_n_free_r_bohr`` and
``species_i_n_free_r`` instead of inserting NaNs on the full native grid.

Bound levels and orbital densities
----------------------------------

Resolved bound levels are flattened into aligned one-dimensional arrays:

``species_i_bound_l``, ``species_i_bound_n_index``, ``species_i_bound_principal_n``
   Angular momentum, radial index, and spectroscopic principal quantum number
   :math:`n=n_{\rm radial}+l`.
``species_i_bound_energy_ha``
   Numerical level energy.  ``species_i_bound_energy_cut_ha`` records the
   continuum edge used for the bound/free classification.
``species_i_bound_fd``, ``species_i_bound_m``, ``species_i_bound_fdm``
   Fermi--Dirac factor, pressure-ionization weight, and their product.
``species_i_bound_occ_deg_fd``, ``species_i_bound_occ_deg_fdm``
   Occupations including the :math:`2(2l+1)` degeneracy.
``species_i_bound_q_ion_ws``
   Per-level contribution to ionic charge inside the WS sphere.
``species_i_bound_orbital_density_r``
   Per-level contribution to :math:`n_{\rm bound}(r)` using the workflow's
   ``bound_occ_mode``.
``species_i_ion_orbital_density_r``
   Per-level contribution to :math:`n_{\rm ion}(r)`, including :math:`M(E)`
   and the radial partition :math:`f_{\rm cut}(r)`.

The two density tables have shape ``(N_level, N_native_r)`` and use the same
level order as ``bound_l``, ``bound_n_index``, and ``bound_energy_ha``:

.. code-block:: python

   E = state[prefix + "bound_energy_ha"]
   l = state[prefix + "bound_l"]
   n = state[prefix + "bound_principal_n"]
   n_level = state[prefix + "bound_orbital_density_r"]
   n_ion_level = state[prefix + "ion_orbital_density_r"]

The scalar strings ``species_i_bound_occ_mode``,
``species_i_threshold_state_status``, and
``species_i_threshold_state_representation`` retain the occupation and
near-threshold classification used by the electronic solve.

Metadata and discovery
----------------------

``metadata_json`` records the complete :class:`otter.PlasmaWorkflowConfig`
snapshot, citation keys, units, thermodynamic state, model choices,
electronic/common-chemical-potential/HNC convergence diagnostics, export
windows, definitions, and the actual field list.  Programmatic discovery
does not require a hard-coded list:

.. code-block:: python

   import json

   metadata = json.loads(str(state["metadata_json"].item()))
   print(metadata["configuration"])
   print(metadata["citation_keys"])
   print(metadata["model"])
   print(metadata["convergence"])
   print(metadata["units"])
   print(metadata["fields"])

The schema is validated by :func:`otter.load_plasma_state`.  Direct loading
with ``numpy.load(path, allow_pickle=False)`` is also supported.

API
---

The public interface is :class:`otter.StateExportOptions`,
:func:`otter.save_plasma_state`, and :func:`otter.load_plasma_state`.
Lower-level construction and validation are in :mod:`otter.io.state`.
