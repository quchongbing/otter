# Carbon DFT-MD structure factors provided by Dr. Argha Roy

This package contains unpublished carbon static ion structure factors
provided by Dr. Argha Roy (private communication) at
`rho = 3.51538 g cm^-3` and `Te = Ti = 1, 20, 30, 40, 50, 100 eV`.

The three columns are wavenumber in inverse ångström, dimensionless
`Sii(k)`, and reported uncertainty.  The uncertainty is not identified as one
standard deviation because that interpretation has not been confirmed.

The data are labelled **DFT-MD**.  No open-data license is recorded; the files
therefore carry license status `NOASSERTION` and are not covered by Otter's
BSD-3-Clause software license.  See `manifest.json` and the parent
reference-data notice before reuse.

Current Otter calculations are stored separately under
`benchmarks/baselines/argha_roy_carbon_sii`.  The displayed 20, 30, 40, 50,
and 100 eV states pass the electronic, threshold-state, HNC fixed-point, and
transform-closure checks.  The 1 eV reference is retained but not displayed.
