Scientific benchmarks
=====================

Benchmarks are part of Otter's scientific interface.  They record the plasma
state, model choices, source reference, numerical tolerances, software
revision, and quantitative comparison metrics.

Published pages plot compact, curated arrays.  They do not rerun the
expensive quantum electronic solver during a documentation build.  Select a
card for its physical definition, literature provenance, quantitative audit,
and reproducible plotting or recomputation entry point.

All gallery figures use :mod:`otter.plotting` and are exported twice when a
script runs: a high-resolution PNG for the web and a vector PDF for papers or
slides.  The PDF paths are printed by each script and live under the
corresponding ``benchmarks/outputs/<benchmark>/figures`` directory.

Definition-aware ionization context
-----------------------------------

The carbon ionization example is the main visual entry point for this
definition-aware comparison.  It shows Otter's :math:`\bar Z` and :math:`Z^*`
alongside the model-dependent :math:`Z^{\rm free}` curves from Figure 3(a) of
:cite:t:`BethkenhagenEtAl2020`, and provides a second figure with the tracked
orbital levels.  Because these quantities use different electron partitions,
the comparison is intended to show definitions and trends, not to assign a
misleading pointwise error.  The complete, directly executable example is
shown here; the publication-data audit remains available in the dedicated
Bethkenhagen benchmark page.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Carbon ionization, pressure-ionization levels, and definition-aware comparison with Bethkenhagen et al. (2020).">

.. only:: html

   .. image:: /gen_examples/images/thumb/sphx_glr_plot_carbon_ionization_levels_thumb.png
      :alt: Otter carbon ionization and pressure-ionization levels

   :doc:`/gen_examples/plot_carbon_ionization_levels`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Carbon ionization and pressure-ionization levels</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

The literature-only comparison and its checksummed reference-data audit are
available at
:doc:`Bethkenhagen et al. (2020) carbon ionization
<gen_benchmarks/plot_bethkenhagen_et_al_2020_carbon_ionization>`.

Equilibrium pair distributions: :math:`g_{ii}(r)`
--------------------------------------------------

These pages compare radial pair distributions at
:math:`T_e=T_i`.  Published curves are markers; Otter calculations are
continuous lines.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Nine-state CH1.36 pair-distribution comparison with the digitized IS-QM curves in Starrett et al. (2014), Figure 3.">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_starrett_et_al_2014_mixtures_fig3_thumb.png
      :alt: Starrett et al. 2014 CH1.36 mixture benchmark

   :doc:`gen_benchmarks/plot_starrett_et_al_2014_mixtures_fig3`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Starrett 2014 CH1.36 mixtures</div>
    </div>

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Single-species pair-distribution comparisons with Starrett and Saumon, High Energy Density Physics 10, 35–42 (2014).">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_starrett_single_species_2013_2014_thumb.png
      :alt: Starrett and Saumon single-species ion-structure benchmark

   :doc:`gen_benchmarks/plot_starrett_single_species_2013_2014`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Starrett--Saumon single species</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Equilibrium structure factors: :math:`S_{ii}(k)`
-------------------------------------------------

The carbon page is an equilibrium :math:`S_{ii}` comparison:
:math:`T_e=T_i` is stated for every displayed curve.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Current Otter PA-HNC carbon structure factors compared with DFT-MD data provided by Dr. Argha Roy at five equilibrium temperatures.">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_argha_roy_carbon_sii_thumb.png
      :alt: Current Otter PA-HNC and DFT-MD comparison using data provided by Dr. Argha Roy

   :doc:`gen_benchmarks/plot_argha_roy_carbon_sii`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Carbon Otter PA-HNC and Dr. Argha Roy DFT-MD data</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Two-temperature / non-equilibrium ionic structure
--------------------------------------------------

Here :math:`T_e\ne T_i`; the page reports both temperatures and compares
only like-for-like two-temperature states.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two-temperature aluminium pair distributions compared with Johnson, Shaffer, and Murillo (2025), Figure 2.">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_johnson_et_al_2025_two_temperature_al_thumb.png
      :alt: Johnson et al. 2025 two-temperature aluminium benchmark

   :doc:`gen_benchmarks/plot_johnson_et_al_2025_two_temperature_al`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Johnson 2025 two-temperature aluminium</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Cross-observable literature library
-----------------------------------

This library is intentionally separate from the three homogeneous categories
above.  It contains both :math:`g_{ii}(r)` and :math:`S_{ii}(k)`, and includes
equilibrium and explicitly labelled two-temperature states.  Its page records
the observable, :math:`T_e`, :math:`T_i`, coordinate units, source figure, and
reference for every panel.

.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Audited Otter gii and Sii comparisons with Al, Be, and C literature curves. Equilibrium and two-temperature panels are labelled separately.">

.. only:: html

   .. image:: /benchmarks/gen_benchmarks/images/thumb/sphx_glr_plot_ion_structure_library_thumb.png
      :alt: Cross-observable ion-structure literature library

   :doc:`gen_benchmarks/plot_ion_structure_library`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ion-structure literature library</div>
    </div>

.. thumbnail-parent-div-close

.. raw:: html

    </div>

.. toctree::
   :hidden:
   :maxdepth: 1

   starrett_et_al_2014_mixtures_fig3
   ion_structure_library
   starrett_single_species_2013_2014
   johnson_et_al_2025_two_temperature_al
   argha_roy_carbon_sii
   gen_benchmarks/index
   validation_policy

Validation and data governance
------------------------------

Visual agreement is useful but insufficient.  Benchmark reports should include
machine-readable errors, peak positions, charge closure, SCF/HNC residuals, and
the exact physical-model configuration.  Digitized literature data must be
clearly distinguished from native author data.  See
:doc:`validation_policy` for the acceptance and public-release rules, or open
the complete :doc:`runnable scientific benchmark gallery
<gen_benchmarks/index>` to browse and download every Python script and
notebook.
