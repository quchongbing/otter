Carbon local-field-correction sensitivity
=========================================

This example isolates the static electron local-field correction (LFC) for
carbon at :math:`\rho=5\ {\rm g\,cm^{-3}}` and
:math:`T_e=T_i=2,100` eV.  At each temperature all model branches reuse the
same converged KS average atom, :math:`\bar Z`, screening cloud
:math:`q(k)=n_{\rm scr}(k)`, and finite-temperature Lindhard response
:math:`\chi^0_{ee}(k)`.  Only the LFC and downstream effective potential and
HNC calculation change.

The compared models are RPA (:math:`G_{ee}=0`), Hubbard
:cite:p:`Hubbard1958`, Utsumi--Ichimaru
:cite:p:`UtsumiIchimaru1982`, Chabrier-1990
:cite:p:`Chabrier1990`, and the Gregori finite-temperature interpolation
:cite:p:`GeldartVosko1966,GregoriEtAl2007`.  Chabrier-1990 is the comparison
reference, not an asserted exact result.  The pseudoatom/QOZ construction
follows :cite:t:`StarrettSaumon2014`.

Why similar responses can give different potentials
----------------------------------------------------

The Starrett--Saumon reduction contains the inverse response
:cite:p:`StarrettSaumon2014`,

.. math::

   V_{ii}(k)=\frac{4\pi\bar Z^2}{k^2}
             +\frac{q(k)^2}{\chi_{ee}(k)}.

For the LFC models used here,
:math:`G_{ee}(k)=a k^2+O(k^4)`, so their interacting responses have the same
leading Coulomb limit,

.. math::

   \chi_{ee}(k)=-\frac{k^2}{4\pi}+O(k^4).

The LFC contribution to the effective potential nevertheless approaches the
finite, model-dependent limit

.. math::

   V_{\rm LFC}(k)
   =\frac{4\pi}{k^2}G_{ee}(k)q(k)^2
   \longrightarrow 4\pi a\bar Z^2.

This is why response curves that nearly overlap at small :math:`k` can produce
visibly different :math:`V_{ii}(k)`.  The example plots the stable
decomposition

.. math::

   V_{ii}=V_{\rm charge}+V_{\rm LFC}+V_{\chi_0}

where, with :math:`d(k)=q(k)-\bar Z`,

.. math::

   V_{\rm charge} &= \frac{4\pi}{k^2}
      \left[-2\bar Z d(k)-d(k)^2\right],\\
   V_{\rm LFC} &= \frac{4\pi}{k^2}G_{ee}(k)q(k)^2,\\
   V_{\chi_0} &= \frac{q(k)^2}{\chi^0_{ee}(k)}.

All branches share :math:`V_{\rm charge}+V_{\chi_0}` because the average
atom, :math:`q(k)`, and :math:`\chi^0_{ee}(k)` are held fixed.  This sum is
used only to verify the decomposition and is not plotted as a model curve.
Only :math:`V_{\rm LFC}` changes; the complete three-term sum is passed to
QOZ/HNC.

Run the example
---------------

:ref:`sphx_glr_gen_examples_plot_carbon_lfc_sensitivity.py`

The source contains one user-facing switch:

.. code-block:: python

   RECOMPUTE_WITH_OTTER = False

``False`` verifies and loads checksummed Otter results.  ``True`` runs the
shared electronic calculation and all five LFC/QOZ/HNC branches, then writes
new files under
``benchmarks/outputs/carbon_lfc_sensitivity/gallery_recomputed``.  Existing
accepted results are not overwritten.

The pickle-free NPZ files retain arrays through :math:`r,k\leq20` in Bohr
units.  The manifest under
``benchmarks/baselines/carbon_lfc_sensitivity`` records the resolved settings
and checksums.
