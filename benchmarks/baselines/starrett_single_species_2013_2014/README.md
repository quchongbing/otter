# Starrett–Saumon single-species Otter baselines

The standalone gallery controller requests both current-Otter QM (KS-DFT)
and TF calculations for all six thermodynamic states. Eight of twelve
calculations pass the full/external electronic, threshold-state, HNC, and
independent transform-closure gates:

- Fe 10 eV: TF;
- H 5 eV: QM and TF;
- H 172 eV: QM and TF;
- C 64.64 eV: QM and TF;
- W 10 eV: TF on an 8192-point QOZ grid.

Fe QM and W 60 eV TF remain above the transform-closure threshold after
doubling the QOZ grid. W 10/60 eV QM do not reach a physical HNC fixed point.
Those attempts and their numerical diagnostics are recorded in
`manifest.json`; no failed iterate is saved as a baseline or plotted.

The reference curves are local numerical extractions from C. E. Starrett and
D. Saumon, *High Energy Density Physics* **10**, 35–42 (2014), DOI
[10.1016/j.hedp.2013.12.001](https://doi.org/10.1016/j.hedp.2013.12.001),
PII `S1574181813001900`.

The extracted reference coordinates are published by maintainer decision
with source attribution and license status `NOASSERTION`; they are not
covered by Otter's BSD-3-Clause software license.  See the adjacent reference
manifest before reuse.  The accepted Otter NPZ files in this directory are
project-generated outputs.
