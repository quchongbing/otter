# Ion-structure reference library

This package contains the small numerical curves selected
from the user-supplied `form factors` library after an explicit provenance and
unit audit.  Otter does not depend on that workstation directory at runtime.

The core imported families are:

- N. M. Gill, R. A. Heinonen, C. E. Starrett, and D. Saumon,
  “Ion-ion dynamic structure factor of warm dense mixtures,” *Physical
  Review E* **91**, 063109 (2015), DOI
  [`10.1103/PhysRevE.91.063109`](https://doi.org/10.1103/PhysRevE.91.063109),
  Fig. 3.  The Al state is `rho=2.7 g/cc`, `Te=Ti=5 eV`.
- J. Clérouin *et al.*, “Evidence for out-of-equilibrium states in warm dense
  matter probed by x-ray Thomson scattering,” *Physical Review E* **91**,
  011101 (2015), DOI
  [`10.1103/PhysRevE.91.011101`](https://doi.org/10.1103/PhysRevE.91.011101),
  Fig. 1.
- K. Wünsch, J. Vorberger, and D. O. Gericke, “Ion structure in warm dense
  matter: benchmarking solutions of hypernetted-chain equations by
  first-principle simulations,” *Physical Review E* **79**, 010201(R) (2009),
  DOI
  [`10.1103/PhysRevE.79.010201`](https://doi.org/10.1103/PhysRevE.79.010201).
  The imported Be real-space curves are from Fig. 1(c), and the structure
  factors are from Fig. 2.
- C. E. Starrett and D. Saumon, “Electronic and ionic structures of warm and
  hot dense matter,” *Physical Review E* **87**, 013104 (2013), DOI
  [`10.1103/PhysRevE.87.013104`](https://doi.org/10.1103/PhysRevE.87.013104).
  The local library attributes its C 20 g/cc, 50 eV curve to this work, but
  the exact source panel has not yet been independently verified.  Otter
  therefore does not invent a figure number.

`inventory.json` is the authoritative audit of every family found in the
source directory.  It also records why several tempting datasets are *not*
yet treated as public benchmarks.  In particular, the C
`rho=3.7 g/cc, T=8.62 eV` reference is retained for future pressure-ionization
work, but is not paired with an accepted Otter result while Otter's threshold-state
reliability gate rejects that electronic solution.

Two additional source families were moved into dedicated benchmark
packages rather than crowding this core figure:

- Johnson, Shaffer, and Murillo (2025), Fig. 2(a,c,d), is stored under
  `reference_data/johnson_et_al_2025_two_temperature_al`.  Direct inspection
  of the paper shows that its radius is in Bohr; the inherited `plot_zak.py`
  ångström label was a plotting bug.
- Carbon DFT-MD data provided by Dr. Argha Roy are stored under
  `reference_data/argha_roy_carbon_sii`, including all six temperatures and
  the reported uncertainty column.

The inventory also records the publication and Fig. 2 identity for
`max2022`, while retaining its *method-curve* identity as unresolved: at
least one file contains wave numbers inaccessible to the cited 125-ion
DFT-MD cell and must not be relabelled DFT-MD.  Old `non_equlibrum` arrays
are project output rather than an independent reference.  Additional
Starrett H/C/Fe/W curves have moved to the dedicated
`starrett_single_species_2013_2014` package after a panel and unit audit.

## Units

The coordinate units were checked against the original local plotting
programs, not inferred from filenames:

| family | reference coordinate | local plotting evidence |
|---|---|---|
| Gill *et al.* | `k` in inverse ångström | Gill *et al.* (2015), Fig. 3 axis |
| Clérouin | `k` in inverse ångström | `plot_jean.py` leaves reference `k` unchanged and divides the project Bohr grid by `0.529177249` |
| Wünsch | `k` in inverse ångström; `r` in ångström | `plot_wunscher.py` applies the same reciprocal conversion; real-space curves follow the publication axis |
| selected Starrett C | `r` in Bohr | `plot_starrett.py` leaves these C columns unchanged and labels the axis `a_B` |

The exceptional Fe conversion and the `1.364781` rescaling used only for the
Starrett C `12.64 g/cc, 64.64 eV` panel are not generalized in this core
package.  Their dedicated benchmark identifies both raw coordinates as
dimensionless :math:`r/R_{\rm WS}` and performs conversion only in its
plotting layer.  All dependent variables are dimensionless; original CSV
values are never rewritten.

## Rights and release policy

The curves are digitized or extracted numerical values from cited figures;
they are not source-code outputs covered by Otter's software license.  The
maintainer has explicitly chosen to publish the listed files with per-file
publication attribution and license status `NOASSERTION`.  That repository
policy is recorded in `manifest.json`; it does not assert a publisher- or
author-supplied data license.  Downstream users should consult the cited
sources and the parent reference-data notice before reuse.  Source PDFs are
not copied into the repository.
