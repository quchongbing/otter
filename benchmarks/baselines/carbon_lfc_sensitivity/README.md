# Carbon finite-temperature LFC sensitivity

This reviewed, project-generated Otter package isolates the static jellium
electron local-field correction for carbon at `rho=5 g/cc` and
`T=2, 100 eV`. At each temperature one strict KS-DFT full/external
pseudoatom result is reused for RPA, Hubbard, Utsumi–Ichimaru,
Chabrier-1990, and Gregori-2007; only the downstream response,
`V_ii`, and QOZ/HNC solution change.

Physical references are:

- J. Hubbard, *Proc. R. Soc. A* **243**, 336–352 (1958),
  [doi:10.1098/rspa.1958.0003](https://doi.org/10.1098/rspa.1958.0003);
- K. Utsumi and S. Ichimaru, *Phys. Rev. A* **26**, 603–610 (1982),
  [doi:10.1103/PhysRevA.26.603](https://doi.org/10.1103/PhysRevA.26.603);
- G. Chabrier, *J. Phys. France* **51**, 1607–1632 (1990),
  [doi:10.1051/jphys:0199000510150160700](https://doi.org/10.1051/jphys:0199000510150160700);
- D. J. W. Geldart and S. H. Vosko, *Can. J. Phys.* **44**, 2137–2171
  (1966), [doi:10.1139/p66-174](https://doi.org/10.1139/p66-174); and
- G. Gregori et al., *High Energy Density Physics* **3**, 99–108 (2007),
  [doi:10.1016/j.hedp.2007.02.006](https://doi.org/10.1016/j.hedp.2007.02.006).

The Gregori/Geldart–Vosko software implementation is adapted from JaXRTS as
recorded in `THIRD_PARTY_NOTICES.md`; the papers above remain the physical
references.

Each pickle-free v2 NPZ stores the shared electronic structure and stacks
`G_ee(k)`, `chi_ee(k)`, `V_ii(r/k)`, `g_ii(r)`, and `S_ii(k)` in the fixed
model order. It includes AA/external, threshold-state, HNC, closure, charge,
unit, and clean-producer provenance metadata. Arrays retain `r,k <= 20`.
The standalone example also evaluates `G_ee(k)/k^2` and the stable
`V_charge + V_LFC + V_chi0` decomposition, then writes a dedicated low-k
audit figure explaining why nearly coincident `chi_ee(k)` curves can still
produce separated effective potentials.

Recreate the capability-gallery plot with the complete standalone example:

```bash
python docs/examples/plot_carbon_lfc_sensitivity.py
```

Set `RECOMPUTE_WITH_OTTER = True` in that file (or export
`OTTER_RECOMPUTE_CARBON_LFC=1`) to stage a fresh current-Otter candidate
under `benchmarks/outputs`; accepted files are never replaced automatically.
The lower-level reviewed-data maintenance runner remains
`benchmarks/runners/regenerate_carbon_lfc_sensitivity.py`.
