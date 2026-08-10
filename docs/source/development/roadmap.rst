Development roadmap
===================

This page records planned work.  It is not a claim about current production
capabilities.

Hybrid complex-energy Green-function continuum
----------------------------------------------

The current quantum backend integrates real-energy scattering states, adds
near-zero logarithmic energy anchors, and scouts phase-shift roots.  That
method is testable and efficient, but a resonance narrower than the adaptive
real-axis resolution can still be missed.

A future backend will combine explicitly resolved core orbitals with a
complex-energy radial Green function.  The intended work packages are:

1. implement the regular and outgoing/Coulomb radial solutions and their
   Wronskian-normalized Green function for the same effective potential and
   boundary convention used by the orbital solver;
2. evaluate density and density-of-states contributions on a complex contour,
   including a controlled treatment of Fermi poles and a low-temperature
   contour refinement;
3. define an explicit projector or energy partition between orbital poles and
   the Green-function continuum so a pressure-ionizing state cannot be counted
   twice or disappear from :math:`n_{\rm bound}+n_{\rm cont}`;
4. adapt the contour using error estimates from paired quadratures, pole
   proximity, and sum-rule residuals rather than a fixed real-energy mesh;
5. retain the current phase-root integrator as an independent fallback and
   cross-check.

Required validation includes:

* free-electron and analytic Coulomb reference problems;
* spectral-density positivity and particle-number closure;
* Levinson/phase-density-of-states sum rules;
* continuity of :math:`n_{\rm bound}+n_{\rm cont}` through a threshold
  crossing;
* independence from bound-box radius, radial grid, contour height, and
  quadrature order;
* resolved artificial Breit--Wigner resonances spanning several widths;
* Al, C, and H pressure-ionization scans compared with the real-axis backend.

No finite implementation can promise to resolve an ``arbitrarily`` narrow
feature without an error tolerance.  The goal is instead a posteriori error
control and stable integrated observables.

Benchmark maturation
--------------------

Imported precursor results should be replaced incrementally by Otter
recomputations.  Each replacement receives a new schema, exact commit,
configuration, convergence record, and quantitative comparison; provenance is
never rewritten in place.

Stable public API
-----------------

Before version 1.0, the project will define typed result objects above the
current dictionary payloads while preserving the ``otter`` high-level entry
points and the versioned NPZ state schema.
