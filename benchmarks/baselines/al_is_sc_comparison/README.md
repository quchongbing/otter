# Al IS/SC comparison

This directory contains the reviewed, project-generated Otter result used by
the example-gallery comparison of ion-sphere (IS) and experimental
self-consistent (SC) feedback for Al at `rho=8.1 g/cc` and
`Te=Ti=15 eV`.

The orbital KS-DFT and finite-temperature Thomas–Fermi paths use the same
pseudoatom/QOZ/HNC construction.  The SC extension follows Sec. 2.4,
Eqs. (19)–(20), of C. E. Starrett and D. Saumon, *High Energy Density
Physics* **10**, 35–42 (2014),
[doi:10.1016/j.hedp.2013.12.001](https://doi.org/10.1016/j.hedp.2013.12.001).
Otter labels this path experimental because broader state-space validation is
still required.

`manifest.json` records the exact producer commit and script checksum,
physical state, units, convergence policy, data rights, and NPZ checksum.
The archive is numeric-only and can be loaded with `allow_pickle=False`.
