# Changelog

This project follows [Semantic Versioning](https://semver.org/). Changes that
have not yet been released are collected under “Unreleased”.

## Unreleased

## 0.2.1 - 2026-08-14

### Added

- Expanded ``otter_state_v3`` archives with electronic profiles, orbital
  densities, response functions, QOZ interaction channels, pair potentials,
  structure factors, and calculation metadata.
- Added a Colab introduction and optional Libxc installation extra.

### Changed

- Added a Poetry lock file and one cross-platform `poetry install` path with
  runtime, plotting, tests, documentation, and editable source installation.
- Added package-install smoke tests on all three operating systems and on
  supported CPython release lines.
- Replaced the duplicated quick-start snippets with the canonical
  ``examples/single_species_workflow.py`` example, which now exercises the
  production defaults directly.
- Kept the top-level examples focused on single-species and mixture workflows;
  numerical diagnostics and model studies now live under ``tools``.
- Stabilized mixture roots near pressure ionization and preserved validated
  external-density tails through the final electronic solve.
- Recomputed and promoted all accepted example and benchmark NPZ baselines
  with field inventories and provenance metadata.
- Simplified the software citation and refreshed the example and benchmark
  documentation.

## 0.2.0 - 2026-08-10

### Added

- Unified single-species and mixture AA → pseudoatom → QOZ/HNC workflow.
- Orbital Kohn–Sham and finite-temperature Thomas–Fermi electronic backends.
- Finite-temperature Chabrier (1990) jellium local-field correction.
- Portable, pickle-free `q(k)`, `f(k)`, `g_ij(r)`, and `S_ij(k)` state files.
- Cached, provenance-checked Starrett et al. mixture benchmark.
- Experimental AA ↔ QOZ/HNC self-consistent feedback API.
- Configurable exchange-correlation models, including dependency-free Dirac
  exchange and optional Libxc-backed LDA/PBE functionals with recorded
  software and functional provenance.
- Self-contained capability and scientific-benchmark galleries with
  results-first HTML pages and publication-ready PNG/PDF figures.

### Changed

- Physical Fermi–Dirac bound occupation is the production default.
- Pseudoatom charge closure is enforced on the QOZ/DST lattice.
- HNC production paths reject unconverged or projected nonphysical roots.
- Experimental SC feedback and production state export now fail closed on
  missing convergence status.
- Continuum threshold, phase-shift resonance, weak-bound-state, and B3/Friedel
  tail diagnostics were strengthened.
- Scientific reference datasets now carry explicit source, checksum, rights,
  and redistribution metadata separate from Otter's software license.

## 0.1.0

- Initial Otter project structure and documentation prototype.
