# CH1.36 complete mixture-workflow example

This directory contains the checksum-gated, current-Otter result displayed by
`docs/examples/plot_ch136_mixture_workflow.py`.

- composition: CH1.36
- mass density: 5 g/cc
- electron and ion temperature: 100 kK (8.617333262145 eV)
- electronic model: orbital KS average atom, full + external pseudoatom
- ion structure: multicomponent QOZ/HNC
- response: finite-temperature Lindhard with Chabrier (1990) LFC

The archive contains only Otter-computed arrays and scalar audits. It contains
no digitized or third-party numerical data. Set
`OTTER_RECOMPUTE_CH136_EXAMPLE=1` when running the gallery script to reproduce
the result in `benchmarks/outputs/ch136_mixture_workflow_100kk/recomputed`.

Method references:

- C. E. Starrett and D. Saumon, *High Energy Density Physics* **10**,
  35--42 (2014),
  <https://doi.org/10.1016/j.hedp.2013.12.001>.
- C. E. Starrett, D. Saumon, J. Daligault, and S. Hamel,
  *Phys. Rev. E* **90**, 033110 (2014),
  <https://doi.org/10.1103/PhysRevE.90.033110>.
- G. Chabrier, *J. Phys. France* **51**, 1607--1632 (1990),
  <https://doi.org/10.1051/jphys:0199000510150160700>.
