Installation
============

Requirements
------------

Otter requires CPython 3.12 or newer.  Core dependencies are NumPy, SciPy,
Numba, and Matplotlib.

Install Python and Poetry
-------------------------

Install `Python <https://www.python.org/downloads/>`_ and
`Git <https://git-scm.com/downloads/>`_, then install Poetry 2.1.3.

macOS, Linux, or WSL:

.. code-block:: console

   $ curl -sSL https://install.python-poetry.org | python3 - --version 2.1.3

Windows PowerShell:

.. code-block:: powershell

   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py - --version 2.1.3

If ``py`` is unavailable, use ``python``.  Restart the terminal if necessary,
then verify:

.. code-block:: console

   $ poetry --version

If the command is unavailable, add Poetry to ``PATH`` as reported by the
installer.

Install Otter
-------------

Clone the repository and install from the lock file:

.. code-block:: console

   $ git clone https://github.com/otter-hed/otter.git
   $ cd otter
   $ poetry install

``poetry install`` installs the locked dependencies and Otter in editable
mode.

The default local-density Dirac exchange model requires no optional package.
Libxc is needed only for additional LDA correlation and GGA functionals. To
enable them, install the ``libxc`` extra:

.. code-block:: console

   $ poetry install --extras libxc

PyPI currently distributes ``pylibxc7`` as source rather than as platform
wheels. This optional installation therefore requires CMake and a C compiler;
Conda is not required. Poetry downloads, builds, and installs the bindings in
Otter's environment. Platform-specific prerequisites are listed in
:doc:`user_guide/xc_functionals`.

For a manual source installation in Otter's Poetry environment:

.. code-block:: console

   $ git clone --branch 7.0.0 --depth 1 https://gitlab.com/libxc/libxc.git
   $ poetry run python -m pip install ./libxc

The manual command above requires the same CMake/C-compiler toolchain. The
`official Libxc installation guide <https://libxc.gitlab.io/installation/>`_
documents the prerequisites. The detailed Otter XC guide explains model
selection and installation:
:doc:`user_guide/xc_functionals`.

Run commands inside the managed environment with ``poetry run``:

.. code-block:: console

   $ poetry run python -c "import otter; print(otter.__version__)"

Run the tests
-------------

.. code-block:: console

   $ poetry run pytest -q

Build the documentation
-----------------------

Run a strict local build:

.. code-block:: console

   $ poetry run make -C docs strict

Without ``make`` (for example, on Windows), run:

.. code-block:: console

   $ poetry run python -m sphinx -E -a -W --keep-going -b html docs/source docs/build/html

Output is written to ``docs/build/html``.  To serve it locally:

.. code-block:: console

   $ poetry run make -C docs serve

Unlocked pip fallback
---------------------

Without Poetry, use ``python -m pip install -e .``.  This does not use
:file:`poetry.lock`.

Otter's source repository does not track generated HTML, autosummary pages, or
Sphinx-Gallery output.
