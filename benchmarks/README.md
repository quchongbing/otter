# Otter benchmarks

This directory separates immutable literature data, precomputed numerical
reference results, and lightweight plotting programs:

```text
reference_data/   Digitized or tabulated results from cited publications
baselines/        Portable, pure-numeric Otter/precursor calculation results
examples/         Sphinx-Gallery benchmark and validation pages
runners/          Offline programs that read those files and regenerate plots
```

The plotting runners do not perform average-atom or QOZ/HNC calculations.
This keeps documentation builds and ordinary continuous integration fast and
deterministic. Full recomputation belongs in a separately marked release
benchmark and must never silently overwrite an accepted reference result.

Every reference-result dataset must:

- load with `numpy.load(..., allow_pickle=False)`;
- use array names that include physical units where appropriate;
- contain no absolute workstation paths;
- provide convergence metadata and a result-affecting configuration;
- be listed in a manifest with SHA-256 checksums and publication provenance.

Literature-derived and author-provided data are not covered by Otter's
source-code license unless their manifest explicitly says otherwise.  The
bundled reference sets are published by maintainer decision with detailed
source attribution and license status `NOASSERTION`; this is not an assertion
of a publisher- or provider-supplied open-data license.  Consult the
[reference-data notice](reference_data/README.md), dataset README, and
manifest before reuse.  The release checker rejects any future manifest with
an unresolved `public_release_gate`.

Curated packages currently included are:

- `baselines/al_qm_tf`: project-generated QM/TF applicability comparison;
- `baselines/carbon_lfc_sensitivity`: project-generated finite-temperature
  LFC sensitivity comparison;
- `baselines/starrett_et_al_2014_mixtures_fig3`: project-generated CH1.36
  precursor results paired with separately gated, digitized literature curves.
- `baselines/ion_structure_library`: Otter Al/Be/C states paired with
  the provenance-audited portion of the local reference library;
- `baselines/starrett_single_species_2013_2014`: strict native Otter C and H
  results, plus explicit rejected/not-calculated Fe and W records, paired
  with a panel-by-panel audited Starrett--Saumon reference collection;
- `baselines/johnson_et_al_2025_two_temperature_al`: Otter aluminium
  calculations paired with the two-temperature Figure 2 curves of Johnson,
  Shaffer, and Murillo (2025);
- `baselines/argha_roy_carbon_sii`: accepted Otter carbon states paired with
  DFT-MD data provided by Dr. Argha Roy (private communication; unpublished);
  and
- `baselines/al_full_workflow_1ev`: the complete Otter Al
  electronic-to-QOZ/HNC gallery state.

Expensive producer programs are named ``regenerate_*.py``.  They write to
``benchmarks/outputs/**/recomputed`` and do not modify accepted reference
results.

Public gallery programs are complete, directly executable scripts.  Their
``USE_PRECOMPUTED_DATA`` switch selects checksum-verified Otter arrays or a
fresh calculation performed in that same file.  Figures use
``otter.plotting`` and write both a high-resolution PNG and a vector PDF
under ``benchmarks/outputs/<benchmark>/figures``; the PDF is intended for
papers and presentation slides.
