# Changelog

This project follows [Semantic Versioning](https://semver.org/). Changes that
have not yet been released are collected under “Unreleased”.

## Unreleased

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
