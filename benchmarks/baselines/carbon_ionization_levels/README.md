# Carbon ionization/level gallery state

This directory contains project-generated Otter output for
`docs/examples/plot_carbon_ionization_levels.py`.  It is a capability example,
not third-party benchmark data.

The v3 archive stores one 4096-point full-AA density scan for carbon at
\(T_e=100\) eV: \(\bar Z=Z-Q_{\rm ion}(R_{\rm WS})\),
\(Z^*=n_e^0/n_i\), the chemical potential, and the 1s/2s/2p/3s/3p/3d energies relative
to the configured numerical edge
\(E_{\rm cut}=V_{\rm eff}(0.70R_{\rm max})\).  These two mean-ionization
definitions are diagnostics rather than unique observables; their distinct
pressure-ionization behaviour is discussed by Starrett *et al.* (2019),
Sec. 4.2.  All 72 AA states from 0.1 through 450 g/cc converged with the
current production `bound_occ_mode="fd"` state sum.  The archive also stores
each displayed shell's direct contribution to \(Q_{\rm ion}(R_{\rm WS})\),
including the Starrett--Saumon pressure-ionization and radial-cutoff weights.
Ten additional states at 1--3 g/cc resolve the pressure-ionization interval;
the original 62 states remain unchanged and are reused as checksummed seeds.
The 3s branch is stored through 0.60 g/cc and is absent at 0.65 g/cc; this
brackets a numerical threshold
interval rather than defining an exact pressure-ionization density.  The
3p branch is found at 0.10, 0.20, 0.25, and 0.35 g/cc; the low-density 3d
branch is displayed where present with reduced opacity.  Shallow-level values
are omitted when their numerical threshold classification is marginal or
unresolved; these classifications are recorded in the manifest, not drawn
as plot annotations.

To reproduce the state, set `RECOMPUTE_WITH_OTTER = True` in the gallery
script, or run:

```bash
OTTER_RECOMPUTE_CARBON_IONIZATION=1 \
PYTHONPATH=src python docs/examples/plot_carbon_ionization_levels.py
```

The ordinary recompute path seeds itself from this accepted 4096-point v3
baseline and calculates only requested densities that are absent there.  Set
`REUSE_ACCEPTED_POINTS_WHEN_RECOMPUTING = False` only for a deliberate
independent calculation of the full grid.  Each completed density point is
checkpointed under
`benchmarks/outputs/carbon_ionization_levels/point_cache`, so an interrupted
scan can resume without changing this accepted archive.

Method context:

- Starrett and Saumon, *High Energy Density Physics* **10**, 35–42 (2014),
  DOI [10.1016/j.hedp.2013.12.001](https://doi.org/10.1016/j.hedp.2013.12.001).
- Starrett *et al.*, *Computer Physics Communications* **235**, 50–62 (2019),
  DOI [10.1016/j.cpc.2018.10.002](https://doi.org/10.1016/j.cpc.2018.10.002).
