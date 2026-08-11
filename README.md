# Otter

Otter calculates electronic and ionic structure in warm and hot dense matter.
From composition, mass density, and temperature it can solve a quantum
average-atom or Thomas–Fermi model, construct neutral
pseudoatoms, build effective ion–ion potentials, and solve one- or
multicomponent QOZ/HNC equations.

Otter is based primarily on the pseudoatom model of
[Starrett and Saumon (2014)](https://doi.org/10.1016/j.hedp.2013.12.001).

## Capabilities

- finite-temperature quantum and Thomas–Fermi electronic structure; the
  quantum model provides orbital levels, occupations, and density components;
- pseudoatom densities `n_ion(r)` and `n_scr(r)`, with form factors
  `f(k)=n_ion(k)` and `q(k)=n_scr(k)`;
- effective ion–ion potentials `V_ij(r)` and `V_ij(k)`;
- one- and multicomponent QOZ/HNC results `g_ij(r)` and `S_ij(k)`.

## Install

Otter requires CPython 3.12 or newer, Git, and Poetry 2.1.3.

macOS, Linux, or WSL:

```bash
curl -sSL https://install.python-poetry.org | python3 - --version 2.1.3
```

Windows PowerShell:

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py - --version 2.1.3
```

Verify the installation:

```console
poetry --version
```

Clone and install:

```bash
git clone https://github.com/quchongbing/otter.git
cd otter
poetry install
```

Dependencies are locked by `poetry.lock`; Otter is installed in editable mode.

```console
poetry run python -c "import otter; print(otter.__version__)"
```

An unlocked fallback is `python -m pip install -e .`. See the
[installation guide](https://quchongbing.github.io/otter/installing.html).

## Quick start

Run the complete [single-species workflow](examples/single_species_workflow.py)
from the repository root:

```bash
poetry run python examples/single_species_workflow.py
```

The default state is Al at `rho=8.1 g/cm^3` and `Te=Ti=15 eV`. Edit the input
block to change the state or output controls. The script plots the electronic
density, effective potential, `g_ii(r)`, and `S_ii(k)`, and saves PNG, PDF, and
NPZ files.

For mixtures, run [mixture_workflow.py](examples/mixture_workflow.py).

### Saved `q/f/g/S`

The versioned NPZ schema stores:

- `q_k == n_scr_k` and `f_k == n_ion_k`;
- `gij_r`, `sij_k`, and (when available) `vij_k`;
- species ordering, units, model settings, and convergence metadata.

The default windows are `r < 20 Bohr` and `k < 20 Bohr^-1`. Archives load with
`allow_pickle=False` and are written atomically.

Quantum continuum calculations can be slow near pressure ionization.
`continue_plasma_workflow_from_electronic_result` reuses a validated
electronic result for subsequent QOZ/HNC calculations.

## Validation and documentation

Build the documentation and cached benchmark gallery with:

```bash
poetry install
poetry run make -C docs strict
```

Open `docs/build/html/index.html` after the build. Start with:

- [documentation source](docs/source/index.rst);
- [capability example gallery](docs/examples/README.rst);
- [scientific benchmark gallery](docs/source/benchmarks/index.rst);
- [validation policy](docs/source/benchmarks/validation_policy.rst);
- [portable state schema](docs/source/user_guide/state_exports.rst);
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
poetry install
poetry run pytest -q
poetry run make -C docs strict
poetry run python -m build
poetry run python -m twine check dist/*
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
