Otter
=====

**Electronic and ionic structure for dense plasmas**

Otter is a Python package for average-atom, pseudoatom, and integral-equation
calculations in warm and hot dense matter.  Starting from a composition, mass
density, and temperature, it can calculate electronic density components,
pseudoatom screening clouds, effective ion--ion potentials, pair distribution
functions :math:`g_{ij}(r)`, and static structure factors
:math:`S_{ij}(k)`.

Otter is based primarily on the pseudoatom model of
`Starrett and Saumon (2014)
<https://doi.org/10.1016/j.hedp.2013.12.001>`_.

Where to begin
--------------

* :doc:`installing` describes the locked source installation.
* :doc:`quickstart` introduces the unified plasma workflow.
* :doc:`user_guide/index` explains portable state files and analysis workflows.
* :doc:`benchmarks/index` records reproducible comparisons with published data.
* :doc:`api/index` documents the stable public Python interface.

Capabilities
------------

* finite-temperature quantum and Thomas--Fermi electronic structure; the
  quantum model provides orbital levels, occupations, and density components;
* pseudoatom densities :math:`n_{\mathrm{ion}}(r)` and
  :math:`n_{\mathrm{scr}}(r)`, with form factors
  :math:`f(k)=n_{\mathrm{ion}}(k)` and :math:`q(k)=n_{\mathrm{scr}}(k)`;
* effective ion--ion potentials :math:`V_{ab}(r)` and :math:`V_{ab}(k)`;
* one- and multicomponent QOZ/HNC results :math:`g_{ij}(r)` and
  :math:`S_{ij}(k)`.

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
