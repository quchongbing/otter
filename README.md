# Otter

Otter calculates electronic and ionic structure in warm and hot dense matter.
From composition, mass density, and temperature it can solve a quantum
average-atom or finite-temperature Thomas–Fermi model, construct neutral
pseudoatoms, build effective ion–ion potentials, and solve one- or
multicomponent QOZ/HNC equations.

The implementation follows the average-atom and pseudoatom framework of
Starrett and collaborators. Model choices, charge closure, nonlinear-solver
status, and validation metadata remain visible in the result.

[Documentation](https://quchongbing.github.io/otter/) ·
[Example gallery](https://quchongbing.github.io/otter/gen_examples/) ·
[Scientific benchmarks](https://quchongbing.github.io/otter/benchmarks/gen_benchmarks/)

> **Status:** Otter is under active development. A converged numerical solve
> is not, by itself, evidence that an average-atom/HNC model is applicable to a
> new thermodynamic regime. Use convergence diagnostics and benchmarks.

## Capabilities

- orbital Kohn–Sham full/external average atoms;
- finite-temperature Thomas–Fermi full/external average atoms;
- pressure-ionization, weak-bound-state, continuum phase-shift, and B3/Friedel
  tail diagnostics;
- single-species and general-mixture common-chemical-potential construction;
- finite-temperature Lindhard response and several local-field corrections,
  with Chabrier (1990) as the validated production default;
- charge-closed one- and multicomponent QOZ effective potentials;
- HNC solvers that reject unconverged or projected nonphysical roots;
- portable, pickle-free `q(k)`, `f(k)`, `g_ij(r)`, and `S_ij(k)` state files;
- cached, provenance-checked literature and model-sensitivity benchmarks.

The AA ↔ QOZ/HNC self-consistent feedback loop is deliberately isolated under
`otter.experimental`; the ion-sphere construction is the production workflow.

## Install

Otter requires Python 3.12 or newer.

```bash
git clone https://github.com/quchongbing/otter.git
cd otter
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Plotting examples require the optional plotting extra:

```bash
python -m pip install -e ".[plot]"
```

All maintained Otter plots use the shared `otter.plotting` style and export
both a 300 dpi PNG for screens and a vector PDF for papers or slides.

For tests or documentation, use `.[dev]` or `.[docs]`.

## Quick start

The same high-level interface handles one component or a mixture. With
`ion_temperature_ev` set, the workflow continues through QOZ/HNC:

```python
from otter import PlasmaWorkflowConfig, solve_plasma_workflow

config = PlasmaWorkflowConfig(
    elements=["C", "H"],
    counts=[1.0, 1.36],
    temperature_ev=8.617333,      # 100 kK
    ion_temperature_ev=8.617333,
    rho_g_cc=2.94,
)
result = solve_plasma_workflow(config)

electronic = result["electronic"]["result"]
ionic = result["ion"]
g_cc = ionic["gij_r"][0, 0]
s_cc = ionic["sij_k"][0, 0]
```

Select the Thomas–Fermi backend with `electronic_model="tf"`. The default
`"qm"` backend retains orbital shell structure.

### Save `q/f/g/S`

```python
config = PlasmaWorkflowConfig(
    elements=["C"],
    temperature_ev=100.0,
    ion_temperature_ev=100.0,
    rho_g_cc=3.7,
    save_state_npz=True,
    save_state_path="outputs/carbon_state.npz",
)
result = solve_plasma_workflow(config)
```

The versioned NPZ schema stores:

- `q_k == n_scr_k` and `f_k == n_ion_k`;
- `gij_r`, `sij_k`, and (when available) `vij_k`;
- species ordering, units, model settings, and convergence metadata.

Its default exclusive windows are `r < 20 Bohr` and
`k < 20 Bohr^-1`; it loads with `allow_pickle=False` and is atomically written
only after a successful archive is complete.

Quantum continuum calculations can take minutes or longer near pressure
ionization. `continue_plasma_workflow_from_electronic_result` reuses an
already validated electronic result while iterating on downstream QOZ/HNC
controls.

## Validation and documentation

The documentation connects equations to implementation, records validity
limits, and redraws compact cached benchmarks without rerunning expensive
average-atom calculations:

**Online documentation:** https://quchongbing.github.io/otter/

```bash
python -m pip install -e ".[docs]"
make -C docs strict
```

Open `docs/build/html/index.html` after the build. Start with:

- [documentation source](docs/source/index.rst);
- [capability example gallery](docs/examples/README.rst);
- [scientific benchmark gallery](docs/source/benchmarks/index.rst);
- [validation policy](docs/source/benchmarks/validation_policy.rst);
- [portable state schema](docs/source/user_guide/state_exports.rst);
- [migration ledger](docs/source/development/migration.rst);
- [development roadmap](docs/source/development/roadmap.rst).

Digitized publication curves and author-provided numerical data have separate
provenance and rights manifests; they are not covered by Otter's BSD software
license unless a dataset explicitly says otherwise.  The current bundled
reference sets are published by maintainer decision with source attribution
and license status `NOASSERTION`.  Read the
[reference-data notice](benchmarks/reference_data/README.md) before reuse.
The executable gate `python tools/check_public_release.py` rejects any future
manifest that reintroduces an unresolved public-release action.

## Development

```bash
python -m pip install -e ".[dev,docs]"
pytest -q
make -C docs strict
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for numerical and benchmark review
requirements and [CHANGELOG.md](CHANGELOG.md) for user-visible changes.

## Citation

Use [CITATION.cff](CITATION.cff) to cite the software, and cite the primary
model papers listed in the documentation for the features used. The central
method reference is:

C. E. Starrett and D. Saumon, “A simple method for determining the ionic
structure of warm dense matter,” *High Energy Density Physics* **10**, 35–42
(2014), [doi:10.1016/j.hedp.2013.12.001](https://doi.org/10.1016/j.hedp.2013.12.001).

Runs using optional Libxc functionals must additionally cite Libxc and every
selected functional. Otter records the installed Libxc version, exact
functional IDs, and Libxc-provided references in `xc_provenance`; see the
[XC citation guide](docs/source/user_guide/xc_functionals.rst).

The repository-wide citation contract is documented in
[`CITATIONS.md`](CITATIONS.md). Runtime configuration objects provide
`config.citation(style="plain"|"bibtex"|"cite")` and expose their canonical
`citation_keys`, so reports can record exactly which physical models were
selected.

Otter is distributed under the [BSD 3-Clause License](LICENSE).
