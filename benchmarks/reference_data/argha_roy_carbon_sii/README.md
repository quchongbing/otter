# Carbon DFT-MD structure factors provided by Dr. Argha Roy

This reference package contains carbon static ion structure factors
at

- `rho = 3.51538 g cm^-3`;
- `Te = Ti = 1, 20, 30, 40, 50, 100 eV`; and
- wave number in inverse ångström.

The three columns are the reported wave number, `Sii(k)`, and reported
uncertainty.  The uncertainty is deliberately not described as one standard
deviation because that interpretation has not been confirmed.

The numerical data were provided by Dr. Argha Roy (private communication;
unpublished) and are labelled **DFT-MD**, matching the inherited
`plot_argha.py` source and the maintainer's current identification.  An
intermediate Otter migration changed that legend to `DMF-MD`; the manifest
records this naming history so generated documentation cannot silently retain
the superseded label.

The files are published by explicit maintainer decision with attribution to
Dr. Argha Roy and license status `NOASSERTION`.  They are not covered by
Otter's BSD-3-Clause software license.  This repository decision does not
assert an open-data license or replace permission from the data provider; see
`manifest.json` and the parent reference-data notice before reuse.

The corresponding current Otter calculations are project-generated outputs
and are kept separately under `benchmarks/baselines/argha_roy_carbon_sii`.
The displayed 20, 30, 40, 50, and 100 eV states pass the strict electronic,
threshold-state, HNC fixed-point, and transform-closure gates. The 1 eV
reference remains available in this package but is not part of the
displayed five-temperature benchmark.
