# Starrett et al. (2014) mixtures, Figure 3

These nine CSV files contain digitized solid-line IS-QM/PA
pair-distribution functions for CH1.36 from Figure 3 of:

C. E. Starrett, D. Saumon, J. Daligault, and S. Hamel, “Integral equation
model for warm and hot dense mixtures,” *Physical Review E* **90**, 033110
(2014), DOI: `10.1103/PhysRevE.90.033110`.

Primary article links:

- DOI / publisher record:
  <https://doi.org/10.1103/PhysRevE.90.033110>
- APS accepted manuscript:
  <https://link.aps.org/accepted/10.1103/PhysRevE.90.033110>
- author preprint:
  <https://arxiv.org/abs/1408.3548>

The state grid is:

- mass density: 2.94, 5, and 15 g cm^-3;
- temperature: 20, 50, and 100 kK;
- pairs: C-C, C-H, and H-H.

Each file has two header rows and three independent `(r, g_ab)` pairs. The
column order in the source files is C-C, H-H, C-H. Radius is measured in
Bohr. Small negative digitization values are retained as extracted; plotting
and metrics discard only non-finite values and negative radii.

These CSV files are project-created digitizations of the solid IS-QM curves
shown in the cited figure. They are not publisher-supplied source data and
were not supplied or endorsed by the article's authors. They were imported
from a private precursor benchmark snapshot, which did not record the
digitization tool or extraction date.

The article is accessible as an accepted manuscript through APS/CHORUS, but
no Creative Commons or data-redistribution license has been identified for
the figure or these extracted curves.

A complete citation supplies attribution and scientific provenance, but it is
not by itself a copyright or database-rights redistribution license.  The
maintainer has explicitly chosen to publish these nine independent
digitizations, their overlays, and their comparison metrics with source
attribution and license status `NOASSERTION`.  They are not covered by
Otter's BSD-3-Clause software license.  This repository policy does not
assert an APS- or author-supplied data license; downstream users should
consult the cited source, the manifest, and the parent reference-data notice
before reuse.

File checksums are recorded in
`../../baselines/starrett_et_al_2014_mixtures_fig3/manifest.json`.
