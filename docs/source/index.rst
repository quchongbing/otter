Otter
=====

**Electronic and ionic structure for dense plasmas**

Otter is a Python package for average-atom, pseudoatom, and integral-equation
calculations in warm and hot dense matter.  Starting from a composition, mass
density, and temperature, it can calculate electronic density components,
pseudoatom screening clouds, effective ion--ion potentials, pair distribution
functions :math:`g_{ij}(r)`, and static structure factors
:math:`S_{ij}(k)`.

The implementation follows the framework developed by Starrett and
collaborators :cite:p:`StarrettSaumon2013,StarrettSaumon2014,StarrettEtAl2014`.
Otter keeps model assumptions and numerical diagnostics visible so that
results can be assessed rather than treated as a black box.

.. important::

   Otter is under active development.  Validate new thermodynamic regimes
   against convergence checks and literature benchmarks before using results
   in production research.

Where to begin
--------------

* :doc:`installing` describes a source and development installation.
* :doc:`quickstart` introduces the unified plasma workflow.
* :doc:`user_guide/index` explains portable state files and analysis workflows.
* :doc:`benchmarks/index` records reproducible comparisons with published data.
* :doc:`api/index` documents the stable public Python interface.

Capabilities
------------

* quantum average-atom full/external calculations;
* finite-temperature Thomas--Fermi electronic structure;
* one- and multicomponent pseudoatom construction;
* finite-temperature electron response and local-field corrections;
* one- and multicomponent QOZ/HNC ionic structure;
* portable, pickle-free :math:`q/f/g/S` state files;
* explicitly isolated experimental models for research and validation.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installing
   quickstart
   user_guide/index
   gen_examples/index

.. toctree::
   :maxdepth: 2
   :caption: Physics and validation

   benchmarks/index
   experimental/index

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   development/index
   citations
   bibliography

Indices
-------

* :ref:`genindex`
* :ref:`search`
