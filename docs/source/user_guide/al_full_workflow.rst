Complete Al 1 eV workflow
=========================

The gallery page :doc:`../gen_examples/plot_al_full_workflow` is a complete,
single-state walk-through for Al at ``rho=8.1 g/cc`` and
``Te=Ti=1 eV``.  It includes:

* the finite-temperature bound-level table;
* full, continuum/free, external, ionic, pseudoatom, and screening densities;
* full/external effective potentials and their nuclear, Hartree, and
  exchange-correlation components;
* :math:`q(k)=n_{\rm scr}(k)` and
  :math:`f(k)=n_{\rm ion}(k)`;
* :math:`V_{ii}(k)` and its inverse transform :math:`V_{ii}(r)`;
* the final :math:`g_{ii}(r)` and :math:`S_{ii}(k)`.

A code-level ``RECOMPUTE_WITH_OTTER`` switch selects the checksummed,
precomputed Otter result or an expensive fresh calculation.  The recorded HNC
residual and the independent finite-DST :math:`g\leftrightarrow S` closure
error are both checked when the precomputed result is loaded.

For slides, the same script also writes three single-purpose figures (PNG and
vector PDF) to ``benchmarks/outputs/al_full_workflow_1ev/figures``:
``al_full_workflow_electronic_densities``, ``al_full_workflow_gii``, and
``al_full_workflow_sii``.  They are generated directly from the state used by
the composite gallery figures, so no second calculation or data-export script
is required.  These slide-only exports are intentionally hidden from the HTML
gallery; the page displays only the two overview figures above.
