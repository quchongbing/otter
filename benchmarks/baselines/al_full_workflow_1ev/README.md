# Complete Al 1 eV Otter workflow result

This reviewed Otter archive powers the full electronic-to-ionic
Sphinx-Gallery example for Al at `rho=8.1 g/cc` and `Te=Ti=1 eV`. It contains
the converged KS levels; full/external/pseudoatom densities; nuclear, Hartree,
XC, and effective potentials; `q(k)=n_scr(k)`; `f(k)=n_ion(k)`; the effective
pair potential; and final `g_ii(r)` and `S_ii(k)`.

The physical workflow follows C. E. Starrett and D. Saumon,
*High Energy Density Physics* **10**, 35–42 (2014),
[doi:10.1016/j.hedp.2013.12.001](https://doi.org/10.1016/j.hedp.2013.12.001),
with the Chabrier-1990 finite-temperature jellium LFC. The manifest records
the standalone producer script and data hashes, common AA-domain settings,
strict HNC and finite-DST closure tolerances, units, and the development
worktree fingerprint used for this reviewed calculation.

The gallery defaults to this checksummed file.  Set
`RECOMPUTE_WITH_OTTER = True` in
`docs/examples/plot_al_full_workflow.py` to stage a fresh result under
`benchmarks/outputs/al_full_workflow_1ev/recomputed`.
