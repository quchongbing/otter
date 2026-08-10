# Aluminium KS-DFT/Thomas–Fermi results

This reviewed, project-generated Otter package compares orbital Kohn–Sham
DFT (`qm`) and finite-temperature Thomas–Fermi (`tf`) average atoms for Al at
`rho=8.1 g/cc` and `T=1, 15, 50, 100 eV`. Both paths feed the same IS
pseudoatom/QOZ/HNC construction with the Chabrier-1990 finite-temperature
jellium local-field correction.

The average-atom, TF, pseudoatom, and QOZ/HNC construction follows C. E.
Starrett and D. Saumon, *High Energy Density Physics* **10**, 35–42 (2014),
[doi:10.1016/j.hedp.2013.12.001](https://doi.org/10.1016/j.hedp.2013.12.001).
The citation specifies the method; no numerical curve was extracted from that
paper.

Each pickle-free NPZ contains the two native electronic grids; full,
external, pseudoatom, bound, continuum, ionic, and screening densities;
`V_ii(k)`, `g_ii(r)`, and `S_ii(k)`; ionization; explicit AA/external,
threshold-state, HNC, and transform-closure audit fields; and producer
provenance. Arrays retain `r,k <= 20` in Bohr units.

`manifest.json` is the authoritative state/file/checksum map, and
`metrics.csv` records model differences. Recreate the plot and independently
recompute the metrics with:

```bash
python benchmarks/runners/plot_al_qm_tf.py
```

Set `USE_PRECOMPUTED_DATA = False` in that runner to stage a fresh candidate
under `benchmarks/outputs`; it never overwrites this accepted package.
