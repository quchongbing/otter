Installation
============

Requirements
------------

Otter requires Python 3.12 or newer together with NumPy, SciPy, and Numba.
Matplotlib is an optional dependency used by examples and benchmark plots.  A
dedicated virtual environment is recommended.

Development installation
------------------------

Clone the repository, create an environment, and install the current checkout
in editable mode:

.. code-block:: console

   $ git clone https://github.com/quchongbing/otter.git
   $ cd otter
   $ python3.12 -m venv .venv
   $ source .venv/bin/activate
   $ python -m pip install --upgrade pip
   $ python -m pip install -e ".[dev,docs]"

An editable installation makes changes under ``src/otter`` immediately
available without reinstalling the package.

Verify the installation
-----------------------

.. code-block:: console

   $ python -c "import otter; print(otter.__file__)"
   $ pytest -q

Build the documentation
-----------------------

Install the documentation dependencies and run a strict local build:

.. code-block:: console

   $ python -m pip install -e ".[docs]"
   $ make -C docs strict

The resulting site is written to ``docs/build/html``.  It can be served only
on the local machine with:

.. code-block:: console

   $ make -C docs serve

Otter's source repository does not track generated HTML, autosummary pages, or
Sphinx-Gallery output.
