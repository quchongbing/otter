# Otter two-temperature aluminium benchmark baselines

This package contains reviewed Otter calculations for the 1, 3, 10, and
30 eV electron-temperature states in Johnson et al. (2025).  All four passed the
strict full/external average-atom, threshold-state, HNC residual, positivity,
and transform-closure gates, and their real SHA-256 values are recorded in
`manifest.json`.

The original 30 eV-electron/1 eV-ion attempt used a cold-start, full-strength
one-component Anderson iteration.  That numerical path stalled with an
order-unity residual and negative `S(k)`.  Re-solving the unchanged raw
OZ/HNC equations by potential-strength continuation and matrix-free
Newton--Krylov reaches the full physical potential with residual
`3.07e-10`, minimum `S(k)=5.99e-3`, and transform-closure mismatch
`2.84e-9`.  The failed diagnostic map was not promoted; this package stores
only the converged continuation result.

The self-contained gallery program
`benchmarks/examples/plot_johnson_et_al_2025_two_temperature_al.py` has two
explicit modes:

- `USE_PRECOMPUTED_DATA = True` loads the four accepted files from this
  package and verifies their SHA-256 checksums.
- `USE_PRECOMPUTED_DATA = False` calls the public Otter workflow directly and
  writes candidate NPZ files plus a candidate manifest under
  `benchmarks/outputs/johnson_et_al_2025_two_temperature_al/gallery_recomputed`.

Future recalculations remain candidates until their convergence diagnostics,
curves, exact producer revision, and checksums have been reviewed.  Do not
bypass this review gate merely to make the documentation build.

The publication curves live in the adjacent reference-data package.  They are
published by maintainer decision with attribution and license status
`NOASSERTION`; Otter-generated baselines do not inherit that third-party data
status.  Consult the reference manifest before reuse.
