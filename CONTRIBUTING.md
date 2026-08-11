# Contributing to Otter

Thank you for helping improve Otter. Numerical plasma models can produce
smooth-looking but incorrect curves, so every change should be reviewable at
three levels: implementation, numerical invariants, and physical validation.

## Development setup

Use Python 3.12 or newer and Poetry 2.1 or newer. Install the locked runtime,
tests, and documentation tools:

```bash
poetry install
```

Run the fast regression suite and strict documentation build:

```bash
poetry run pytest -q
poetry run make -C docs strict
```

On Windows without `make`, use
`poetry run python -m sphinx -E -a -W --keep-going -b html docs/source docs/build/html`.

## Change requirements

- Keep model defaults conservative. Diagnostic approximations must be explicit
  options and clearly labelled in result metadata.
- Add focused tests for each numerical invariant or failure mode.
- For physics changes, add or update a benchmark with a citable source,
  immutable input data, units, provenance, and quantitative metrics.
- Do not regenerate expensive AA calculations in ordinary tests or docs.
  Commit small, pickle-free reference arrays and validate their hashes instead.
- Do not hide failed common-chemical-potential, charge-closure, or HNC solves by
  clipping or silently returning a best-effort result.
- Add the primary literature citation near the implementation and to
  `src/otter/literature.bib`.
- Keep generated Sphinx pages, caches, plots, and local environments out of
  version control.

The validation policy and architecture notes are in `docs/source/benchmarks`
and `docs/source/development`.

## Pull requests

Keep each pull request focused. In the description, state:

1. the physical or engineering problem;
2. the model equations or references affected;
3. tests and benchmark states run;
4. any changes to defaults, schemas, or numerical tolerances;
5. known validity limits.

Changes to public result schemas or model defaults require a changelog entry.
Before creating any public artifact, run
`python tools/check_public_release.py`; a nonzero result is a hard release
blocker, not an informational warning.
