# Ion-structure reference library

This package contains numerical curves used by the ion-structure benchmark:

- Al at `2.7 g/cc`, `Te=Ti=5 eV`: Fig. 3 of Gill *et al.*,
  [doi:10.1103/PhysRevE.91.063109](https://doi.org/10.1103/PhysRevE.91.063109).
- Al at `8.1 g/cc`, `Te=10 eV`, `Ti=10 or 2 eV`: Fig. 1 of Clérouin
  *et al.*,
  [doi:10.1103/PhysRevE.91.011101](https://doi.org/10.1103/PhysRevE.91.011101).
- Be at `5.544 g/cc`, `Te=Ti=13 eV`: Figs. 1(c) and 2 of Wünsch *et al.*,
  [doi:10.1103/PhysRevE.79.010201](https://doi.org/10.1103/PhysRevE.79.010201).
- C at `20 g/cc`, `Te=Ti=50 eV`: PA-HNC data provided by C. E. Starrett
  (private communication; unpublished).

## Units

Gill, Clérouin, and Wünsch reciprocal-space coordinates are in inverse
ångström.  Wünsch real-space coordinates are in ångström.  The selected
Starrett carbon radius is in Bohr.  Plotting code converts Otter coordinates
to these units without modifying the archived reference columns.

## Reuse

The files are publication-derived or author-provided numerical values, not
source-code outputs.
No third-party open-data license is asserted; their license status is
`NOASSERTION`.  Per-file checksums and citations are recorded in
`manifest.json`.  See the parent reference-data notice before reuse.
