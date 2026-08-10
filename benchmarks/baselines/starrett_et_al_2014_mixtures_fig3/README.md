# Validated CH1.36 mixtures Figure 3 precursor results

The nine compressed NPZ files contain the minimum data needed to reproduce
the Starrett et al. mixtures Figure 3 comparison without rerunning the
expensive electronic- and ionic-structure calculation:

- `r_bohr` and the three rows of `g_ab`, ordered C-C, C-H, H-H;
- thermodynamic state and species metadata;
- average-atom, pseudoatom-partition, and QOZ charges;
- common-chemical-potential and HNC convergence diagnostics;
- the complete result-affecting producer signature;
- the original cache filename and SHA-256 checksum.

Only the published comparison interval `0 <= r <= 6 Bohr` is retained. All
arrays are numeric or fixed-width Unicode and load with
`allow_pickle=False`. No absolute local paths are stored.

The results were imported from a validated private precursor calculation.
They are a migration target and provenance record, not a claim that the
current Otter implementation has already reproduced every state. Once the
migrated Otter solver passes the same physical checks, a newly generated
result may be added with a new schema/version rather than overwriting this
one.

`manifest.json` is the authoritative state/file map. `metrics.csv` records
the comparison with the digitized reference curves and shares their
maintainer-published, attributed `NOASSERTION` data status. The pure precursor
calculation NPZ files are project-generated and are tracked separately from
the third-party-derived reference data.

Regenerate the figure and independently recompute the metrics with:

```bash
python benchmarks/runners/plot_starrett_et_al_2014_mixtures_fig3.py
```

To perform a new nine-state Otter calculation, use the separate producer:

```bash
python benchmarks/runners/regenerate_starrett_et_al_2014_mixtures_fig3.py
```

It writes only to
`benchmarks/outputs/starrett_et_al_2014_mixtures_fig3/recomputed`. The
accepted precursor files in this directory are never overwritten. The
benchmark-gallery source also exposes `USE_PRECOMPUTED_DATA`; setting it to
`False` invokes this Otter producer and plots the staged candidate.
