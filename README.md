# otter

`otter` is a Python code for calculating the electronic and ionic structure of
dense plasmas. Given a composition, mass density, electron temperature `Te`,
and optional ion temperature `Ti`, it solves an average-atom electronic
structure problem and computes ion-ion pair correlation functions `g_ij(r)` and
structure factors `S_ij(k)`.

The code is based on the average-atom, pseudoatom, and ionic-structure
framework developed by Starrett and Saumon. It solves the full/external
average-atom problem, builds the pseudoatom screening density `n_scr(r)`,
constructs effective ion-ion potentials `V_ij(k)`, and solves the QOZ/HNC
equations for the ionic structure. The electronic outputs also provide the
density components needed to construct XRTS-related form-factor quantities such
as `q(k) + f(k)`.

## Layout

```text
src/otter/
  workflows.py                  # high-level composition -> AA -> QOZ/HNC API
  data/                         # element data and density helpers
  numerics/                     # constants, radial grids, interpolation, transforms
  electronic/
    ks_dft.py                   # electronic AA overview/orchestration layer
    densities.py                # bound density, n_ion, cutoff, electron counts
    full_external.py            # full -> external workflow driver
    mixture.py                  # multicomponent electronic closure
    potential.py                # Hartree + Starrett full/external potentials
    xc.py                       # exchange-correlation models
    continuum/                  # continuum density and tail machinery
    solvers/
      bound.py                  # bound-state Numerov solver
      free.py                   # free-state Numerov propagation kernels
  ionic/
    qoz.py                      # QOZ effective potentials and HNC solvers
    response.py                 # jellium response / local field corrections
    correlation.py              # ion-sphere correlation model
  io/
    results.py                  # core NPZ result writers
```

## Install

`otter` requires Python `>=3.12`. Clone the repository and enter the project
directory first:

```bash
git clone <repository-url> otter
cd otter
```

For development, install the package in editable mode with `pip install -e .`.
The `-e` flag makes local source-code changes immediately visible to Python
without reinstalling the package.

### Conda

```bash
conda create -n otter python=3.12 -y
conda activate otter

pip install -e .
```

Verify the install:

```bash
python - <<'PY'
from otter import PlasmaWorkflowConfig

cfg = PlasmaWorkflowConfig(elements=["C"], temperature_ev=10.0, rho_g_cc=1.0)
print("otter import ok")
print(cfg)
PY
```

### Python venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

### Poetry

The current project uses standard `pyproject.toml` metadata with a setuptools
backend. With Poetry, use Poetry to create and manage the virtual environment,
then install the local package in editable mode inside that environment:

```bash
poetry env use python3.12
poetry run python -m pip install --upgrade pip
poetry run pip install -e .
```

### Run Tests

To run the lightweight test suite, install the optional development
dependencies and call `pytest`:

```bash
pip install -e ".[dev]"
pytest -q
```

## Minimal Use

```python
from otter import PlasmaWorkflowConfig, solve_plasma_workflow

result = solve_plasma_workflow(
    PlasmaWorkflowConfig(
        elements=["C"],
        temperature_ev=50.0,
        rho_g_cc=1.0,
        ion_temperature_ev=50.0,
    )
)
```

## References

- C. E. Starrett and D. Saumon, *A simple method for determining the ionic structure of warm dense matter*, High Energy Density Physics 10, 35-42 (2014), https://doi.org/10.1016/j.hedp.2013.12.001

- C. E. Starrett and D. Saumon, *Electronic and ionic structures of warm and hot dense matter*, Phys. Rev. E 87, 013104 (2013), https://doi.org/10.1103/PhysRevE.87.013104
