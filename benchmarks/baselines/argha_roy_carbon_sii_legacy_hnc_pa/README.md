# Historical HNC-PA curves for the Argha Roy carbon comparison

This package contains five compact, pickle-free extracts from the legacy PAMD
archives used by the inherited `plot_argha.py` figure:

- carbon at `rho = 3.51538 g cm^-3`;
- `Te = Ti = 20, 30, 40, 50, 100 eV`; and
- `Sii(k)` through the first point above `12 Å^-1`.

The archived curves are labelled **HNC-PA**, matching the original figure.
They are project-generated historical output, not external reference data.
They do not contain the current Otter convergence metadata, HNC residual, or
forward/inverse-transform closure audit.  Consequently they are accepted only
for reproducibly redrawing the historical comparison and must never be
described as current, strict Otter benchmark results.

The modern, convergence-checked 50 and 100 eV Otter states remain separately
under `benchmarks/baselines/argha_roy_carbon_sii`.  The author-provided DFT-MD
curves and uncertainty columns remain separately release-gated under
`benchmarks/reference_data/argha_roy_carbon_sii`.
