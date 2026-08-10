# Johnson et al. (2025) two-temperature aluminium reference data

This reference package contains the local two-column point sets
associated with Fig. 2(a), 2(c), and 2(d) of:

> Z. A. Johnson, N. R. Shaffer, and M. S. Murillo, “Quantum
> Ornstein-Zernike theory for two-temperature two-component plasmas,”
> *Physical Review E* **112**, 025207 (2025),
> <https://doi.org/10.1103/5c29-kdx1>.

All three states are aluminium at
\(\rho=2.7\ {\rm g\,cm^{-3}}\) and \(T_i=1\ {\rm eV}\):

| Paper panel | \(T_e\) (eV) | Curves retained |
|---|---:|---|
| Fig. 2(a) | 1 | 2TTCP HNC+bridge, DFT-MD, YOCP HNC+bridge |
| Fig. 2(c) | 10 | 2TTCP HNC+bridge, DFT-MD, YOCP HNC+bridge |
| Fig. 2(d) | 30 | 2TTCP HNC+bridge, DFT-MD, YOCP HNC+bridge |

The paper's Fig. 2(b), \(T_e=3\ {\rm eV}\), is not present in the local
source library and is intentionally not represented here.  The historical
filename token `AA` denotes the paper's 2TTCP HNC+bridge curve; `YOCP`
denotes its Yukawa one-component-plasma HNC+bridge curve.

## Coordinates and integrity

The CSV files are copied byte-for-byte from the local research data library.
Column 1 is \(r\) in atomic units (Bohr), matching the \(r\,[{\rm au}]\)
axis printed in Johnson et al. Fig. 2, and column 2 is dimensionless
\(g_{ii}(r)\).  No smoothing, interpolation, clipping, or unit conversion has
been applied. `manifest.json` records a SHA-256 digest for every file.

The inherited `plot_zak.py` labelled this coordinate as ångström. Direct
inspection of the cited figure establishes that label as a legacy plotting
bug; the CSV values themselves have not been rescaled.

The local library did not record whether these points were digitized from the
published figure or supplied by an author.  Their extraction provenance is
therefore explicitly unresolved; Otter must not describe them as
author-supplied.

## Data-rights notice

The article states that data are available from the corresponding author upon
reasonable request; it does not declare these numerical point sets as an open
dataset.  An APS article citation or DOI does not by itself grant permission
to redistribute extracted numerical data under Otter's software license.

The maintainer has explicitly chosen to publish these nine extracted curves
with article, figure-panel, and method attribution and license status
`NOASSERTION`.  They are not covered by Otter's BSD-3-Clause software
license.  This repository policy does not assert that APS or the authors
supplied an open-data license; downstream users should consult the article,
`manifest.json`, and the parent reference-data notice before reuse.

The publication PDF is not vendored.
