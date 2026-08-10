Migration ledger
================

Otter is a curated implementation, not a history-preserving copy of its
development precursor.  Migration is performed by physical capability, with
tests and references added before a feature is accepted.

Accepted production capabilities
--------------------------------

* orbital Kohn--Sham full/external average atoms;
* finite-temperature Thomas--Fermi full/external average atoms;
* physical Fermi--Dirac bound occupations;
* threshold-bound-state and real-axis continuum-resonance diagnostics;
* charge-constrained full B3/Friedel continuum tails;
* single- and multicomponent common-chemical-potential construction;
* pseudoatom-partition charge closure on the actual QOZ/DST lattice;
* finite-temperature Chabrier-1990 local-field correction;
* stable low-:math:`k` effective-potential decomposition;
* scalar and multicomponent HNC solvers with physical-root gates;
* portable, pickle-free :math:`q/f/g/S` state files.

Experimental capability
-----------------------

The AA--QOZ/HNC self-consistent feedback loop lives under
:mod:`otter.experimental`.  It is kept out of the production workflow because
the published mixture benchmark uses the ion-sphere construction and because
the feedback closure still needs independent validation.

Deliberately not migrated
-------------------------

* one-off parameter scans and investigation scripts;
* machine-specific environments, paths, caches, and serialized Python
  objects;
* plots without a reproducible data source;
* best-effort solver branches that hide failed charge, common-:math:`\mu`, or
  HNC convergence;
* large raw calculation archives when a compact numerical reference result is
  sufficient.
* precursor free-energy branch scores, which omit continuum thermodynamic
  terms and are not a production EOS;
* specialized dataset-generation and pickle-cache tooling, which belongs in a
  separate application layer rather than the physics library.

Every cached benchmark records whether it is an Otter result, an
imported precursor result, or digitized literature data.  These categories
must never be silently interchanged.

Reference-data release policy
-----------------------------

Imported publication curves require an explicit, machine-readable
distribution decision and complete source attribution.  The current bundled
sets use the maintainer decision ``published_by_maintainer_with_attribution``
with license status ``NOASSERTION``.  Otter recomputations must use a new
result schema rather than overwriting imported provenance.
