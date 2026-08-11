Development roadmap
===================

This page records planned work.  It is not a claim about current production
capabilities.

Pseudoatom molecular dynamics
-----------------------------

The next major capability is pseudoatom molecular dynamics (PAMD), following
:cite:t:`StarrettDaligaultSaumon2015`.  Otter already supplies the principal
electronic inputs: pseudoatom electron densities, screening clouds, and
effective ion--ion pair potentials.  The planned MD layer will propagate
classical ions with these potentials, beginning with single-component plasmas
and comparison against published PAMD structure and transport results.

Equation of state
-----------------

The following stage is an EOS layer based on the PAMD thermodynamic
construction of :cite:t:`StarrettSaumon2016`.  It will reconstruct the plasma
electron density by superposing pseudoatoms on MD configurations and combine
ionic, electrostatic, exchange--correlation, and electronic kinetic
contributions to obtain internal energy and pressure.  Development will start
with the Thomas--Fermi single-component formulation and retain the individual
thermodynamic terms for validation against the published EOS calculations.
