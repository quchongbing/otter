"""
otter/ionic/lfc.py

Electron local-field correction models used by the QOZ response chain.

Scope
-----
This module collects jellium quantities and static LFC models so that:
  - `qoz.py` keeps the response assembly logic readable
  - different `G_ee(k)` approximations can be swapped cleanly
  - non-equilibrium QOZ calculations can use an explicit electron temperature
    `T_e` while the ion-ion HNC part continues to use `T_i`

Model references
----------------
The short labels below are cited in the individual model docstrings.
The canonical keys in ``otter.literature.bib`` are given in parentheses;
runtime configuration reports use those canonical keys rather than the
legacy labels.

[RPA52 / PinesBohm1952] D. Pines and D. Bohm, Phys. Rev. 85, 338--353 (1952),
        DOI 10.1103/PhysRev.85.338.
[H58 / Hubbard1958]   J. Hubbard, Proc. R. Soc. A 243, 336--352 (1958),
        DOI 10.1098/rspa.1958.0003.
[UI82 / UtsumiIchimaru1982]  K. Utsumi and S. Ichimaru, Phys. Rev. A 26, 603--610 (1982),
        DOI 10.1103/PhysRevA.26.603.
[VS72 / VashishtaSingwi1972]  P. Vashishta and K. S. Singwi, Phys. Rev. B 6, 875--887 (1972),
        DOI 10.1103/PhysRevB.6.875.
[GV66 / GeldartVosko1966]  D. J. W. Geldart and S. H. Vosko, Can. J. Phys. 44, 2137--2171
        (1966), DOI 10.1139/p66-174.
[IIT87 / IchimaruEtAl1987] S. Ichimaru, H. Iyetomi, and S. Tanaka, Phys. Rep. 149, 91--205
        (1987), DOI 10.1016/0370-1573(87)90125-6.
[C90 / Chabrier1990]   G. Chabrier, J. Phys. France 51, 1607--1632 (1990),
        DOI 10.1051/jphys:0199000510150160700.
[G07 / GregoriEtAl2007]   G. Gregori, A. Ravasio, A. Hoell, S. H. Glenzer, and S. J. Rose,
        High Energy Density Phys. 3, 99--108 (2007),
        DOI 10.1016/j.hedp.2007.02.006.
[JAXRTS] J. Lütgert, S. Schumacher, J. Rips, C. Qu, T. Döppner, and
        D. Kraus, Comput. Phys. Commun. 325, 110173 (2026),
        DOI 10.1016/j.cpc.2026.110173.

Implementation provenance
-------------------------
The finite-temperature Geldart--Vosko and Gregori-2007 routines below were
copied from ``jaxrts.ee_localfieldcorrections`` at JaXRTS commit
``de309018194a036cf513b4156aee389501308703``:
https://github.com/JaXRTS/jaxrts.  The port translates JAX quantities to NumPy
atomic units and adapts the API to Otter.  JaXRTS is BSD-3-Clause software;
the retained upstream notice
and disclaimer are in ``THIRD_PARTY_NOTICES.md``.  This software provenance
does not replace the primary physical references [UI82], [GV66], and [G07].
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import i1

from otter.literature import citation_keys_for_lfc_model

_TINY = np.finfo(float).tiny

# Canonical dispatcher name -> references in the module docstring.  This map
# is intentionally kept beside the implementation so a model choice is
# traceable without consulting an example or an external report.
LFC_MODEL_REFERENCE_KEYS = {
    "none": ("RPA52",),
    "hubbard": ("H58",),
    "utsumiichimaru": ("UI82",),
    "chabrier1990": ("UI82", "IIT87", "C90"),
    "vashistasingwi": ("VS72", "C90"),
    "chabrier_hubbard": ("H58", "C90"),
    "geldartvosko": ("GV66", "G07"),
    "gregori2007": ("UI82", "GV66", "G07"),
}

# Canonical bibliography keys used by the runtime citation API.  The legacy
# short labels above remain public for compatibility with older diagnostics;
# new code should use this map or ``citation_keys_for_lfc_model``.
LFC_MODEL_CITATION_KEYS = {
    name: citation_keys_for_lfc_model(name)
    for name in LFC_MODEL_REFERENCE_KEYS
}


def _as_array(x):
    arr = np.asarray(x, dtype=float)
    return arr, np.isscalar(x)


def _return_scalar_if_needed(arr, was_scalar):
    if was_scalar:
        return float(np.asarray(arr))
    return np.asarray(arr)


def fermi_wavenumber_au(nbar_e0: float) -> float:
    """
    Fermi wave number in atomic units for a uniform electron gas.
    """
    nbar = float(nbar_e0)
    if nbar <= 0.0:
        raise ValueError("nbar_e0 must be positive.")
    return (3.0 * np.pi**2 * nbar) ** (1.0 / 3.0)


def fermi_energy_au(nbar_e0: float) -> float:
    """
    Fermi energy in Hartree for a uniform electron gas.
    """
    kf = fermi_wavenumber_au(nbar_e0)
    return 0.5 * kf * kf


def wigner_seitz_radius_au(nbar_e0: float) -> float:
    """
    Electron Wigner-Seitz radius in atomic units.
    """
    nbar = float(nbar_e0)
    if nbar <= 0.0:
        raise ValueError("nbar_e0 must be positive.")
    return (3.0 / (4.0 * np.pi * nbar)) ** (1.0 / 3.0)


def coupling_parameter_ee_au(nbar_e0: float, t_ee_ha: float) -> float:
    """
    Electron-electron coupling parameter Gamma_ee in atomic units.

    Here `t_ee_ha` means the thermal energy `k_B T_ee` in Hartree.
    """
    t_ee = float(t_ee_ha)
    if t_ee <= 0.0:
        raise ValueError("t_ee_ha must be positive.")
    rs = wigner_seitz_radius_au(nbar_e0)
    return 1.0 / (rs * t_ee)


def _ui_log_factor(q_over_kf: np.ndarray) -> np.ndarray:
    """
    Stable evaluation of the Utsumi-Ichimaru logarithmic kernel factor.

    Reference: Utsumi and Ichimaru [UI82].
    """
    q = np.asarray(q_over_kf, dtype=float)
    out = np.empty_like(q)

    small = np.abs(q) < 1.0e-8
    near_two = np.abs(q - 2.0) < 1.0e-8
    regular = ~(small | near_two)

    out[small] = 1.0
    out[near_two] = 0.0

    qq = q[regular]
    out[regular] = ((4.0 - qq * qq) / (4.0 * qq)) * np.log(
        np.abs((2.0 + qq) / (2.0 - qq))
    )
    return out


def _ui_zero_t_constraints(nbar_e0: float) -> tuple[float, float, float]:
    """
    Return `(k_F, g_ee(0), gamma_0)` used by Utsumi-Ichimaru-style LFCs.

    Chabrier 1990 Sec. 3.3 discusses finite-T extensions of local-field
    corrections whose free parameters are fixed by the small-k
    compressibility sum rule and the large-k pair-correlation limit. We use
    the same zero-temperature Utsumi-Ichimaru constraints already present in
    this module as diagnostic approximations for the explicit Eq. (32)/(33)
    forms in that paper.

    References: Utsumi and Ichimaru [UI82]; Chabrier [C90], Sec. 3.3.
    """
    nbar = float(nbar_e0)
    if nbar <= 0.0:
        raise ValueError("nbar_e0 must be positive.")

    rs = wigner_seitz_radius_au(nbar)
    z = 4.0 * (4.0 / (9.0 * np.pi)) ** (1.0 / 6.0) * np.sqrt(rs / np.pi)
    g0_ee_0 = 0.125 * (z / i1(z)) ** 2

    rs2_dEc_drs = (0.0621814 + 0.61024 * np.sqrt(rs)) / (
        1.0 + 9.81379 * np.sqrt(rs) + 2.82224 * rs + 0.736411 * rs ** 1.5
    )
    rs3_d2Ec_d2rs = (
        -8.23505 * rs ** 1.5
        - 2.4860 * rs**2
        - 23.0573 * rs
        - 4.50109 * np.sqrt(rs)
        - 0.229324
    ) / (rs ** 1.5 + 3.83243 * rs + 13.3265 * np.sqrt(rs) + 1.35794) ** 2

    gamma_0 = 0.25 - (
        np.pi * (4.0 / (9.0 * np.pi)) ** (1.0 / 3.0) / 24.0
    ) * (rs3_d2Ec_d2rs - 2.0 * rs2_dEc_drs)
    return fermi_wavenumber_au(nbar), float(g0_ee_0), float(gamma_0)


def _gee_utsumi_ichimaru_from_constraints(
    k: np.ndarray,
    kf: float,
    g0_ee_0: float,
    gamma_0: float,
) -> np.ndarray:
    """
    Evaluate the UI functional form for supplied exact-limit coefficients.

    Reference: Utsumi and Ichimaru [UI82].
    """
    a_fit = 0.029
    b_fit = (
        (9.0 / 16.0) * gamma_0
        - (3.0 / 64.0) * (1.0 - g0_ee_0)
        - (16.0 / 15.0) * a_fit
    )
    c_fit = (
        -(3.0 / 4.0) * gamma_0
        + (9.0 / 16.0) * (1.0 - g0_ee_0)
        - (16.0 / 5.0) * a_fit
    )

    q = np.asarray(k, dtype=float) / float(kf)
    lq = _ui_log_factor(q)
    return (
        a_fit * q**4
        + b_fit * q**2
        + c_fit
        + (a_fit * q**4 + (b_fit + (8.0 / 3.0) * a_fit) * q**2 - c_fit)
        * lq
    )


def _iit_fex_over_gamma(gamma_e: float, theta: float) -> float:
    """
    IIT Eq. (3.83) coupling-integrated XC free energy.

    Reference: Ichimaru, Iyetomi, and Tanaka [IIT87], Eq. (3.83).
    """
    gamma_e = float(gamma_e)
    theta = float(theta)
    if not (np.isfinite(gamma_e) and gamma_e > 0.0):
        raise ValueError("gamma_e must be positive and finite.")
    if not (np.isfinite(theta) and theta > 0.0):
        raise ValueError("theta must be positive and finite.")

    lam = (4.0 / (9.0 * np.pi)) ** (1.0 / 3.0)
    tanh_theta = np.tanh(1.0 / theta)
    tanh_sqrt_theta = np.tanh(1.0 / np.sqrt(theta))
    theta2 = theta * theta
    theta4 = theta2 * theta2

    a = (
        (0.75 + 3.04363 * theta2 - 0.09227 * theta2 * theta + 1.7035 * theta4)
        / (1.0 + 8.31051 * theta2 + 5.1105 * theta4)
        * tanh_theta
        / (np.pi * lam)
    )
    b = (
        (0.341308 + 12.070873 * theta2 + 1.148889 * theta4)
        / (1.0 + 10.495346 * theta2 + 1.326623 * theta4)
        * np.sqrt(theta)
        * tanh_sqrt_theta
    )
    e = (
        (0.539409 + 2.522206 * theta2 + 0.178484 * theta4)
        / (1.0 + 2.555501 * theta2 + 0.146319 * theta4)
        * theta
        * tanh_theta
    )
    c = (0.872496 + 0.025248 * np.exp(-1.0 / theta)) * e
    d = (
        (0.614925 + 16.996055 * theta2 + 1.489056 * theta4)
        / (1.0 + 10.10935 * theta2 + 1.22184 * theta4)
        * np.sqrt(theta)
        * tanh_sqrt_theta
    )

    sqrt_gamma = np.sqrt(gamma_e)
    b_minus_cd_over_e = b - c * d / e
    a_minus_c_over_e = a - c / e
    root_arg = 4.0 * e - d * d
    if root_arg <= 0.0:
        raise ValueError(
            "IIT finite-temperature XC fit encountered 4*e-d^2 <= 0."
        )
    root = np.sqrt(root_arg)

    numerator = (
        (c / e) * gamma_e
        + (2.0 / e) * b_minus_cd_over_e * sqrt_gamma
        + (
            a_minus_c_over_e - (d / e) * b_minus_cd_over_e
        )
        / e
        * np.log1p(e * gamma_e + d * sqrt_gamma)
        - 2.0
        / (e * root)
        * (
            d * a_minus_c_over_e
            + (2.0 - d * d / e) * b_minus_cd_over_e
        )
        * (
            np.arctan((2.0 * e * sqrt_gamma + d) / root)
            - np.arctan(d / root)
        )
    )
    return float(-numerator / gamma_e)


def iit_exchange_correlation_free_energy_per_electron_au(
    nbar_e0: float,
    t_e_ha: float,
) -> float:
    """
    IIT finite-temperature XC free energy per electron in Hartree.

    This is the unpolarized parametrization of Ichimaru, Iyetomi, and
    Tanaka [IIT87], Eq. (3.83), used by Chabrier [C90], Sec. 3.3. The
    coefficient transcription was cross-checked against ABINIT's independent
    ``xciit`` implementation.
    """
    nbar = float(nbar_e0)
    t_e = float(t_e_ha)
    if not (np.isfinite(nbar) and nbar > 0.0):
        raise ValueError("nbar_e0 must be positive and finite.")
    if not (np.isfinite(t_e) and t_e >= 0.0):
        raise ValueError("t_e_ha must be non-negative and finite.")

    rs = wigner_seitz_radius_au(nbar)
    ef = fermi_energy_au(nbar)
    # The analytic fit has a regular theta -> 0 limit but is written with
    # explicit 1/theta factors.  A tiny reduced-temperature floor evaluates
    # that limit without introducing a separate, inconsistent XC fit.
    theta = max(t_e / ef, 1.0e-8)
    gamma_e = 1.0 / (rs * theta * ef)
    return _iit_fex_over_gamma(gamma_e, theta) / rs


def chabrier1990_gamma0(
    nbar_e0: float,
    t_e_ha: float,
    *,
    log_density_step: float = 2.0e-3,
) -> float:
    """
    Finite-temperature UI small-k coefficient from Chabrier [C90], Eq. (34).

    For ``q=k/k_F``, ``G(q) = gamma_0 q^2 + O(q^4)``.  The exact
    compressibility relation is evaluated as

    ``gamma_0 = -k_F^2/(4*pi) * d^2[f_xc(n,T)]/dn^2``

    in atomic units, with ``f_xc`` the IIT XC free-energy density.  A
    five-point fourth-order stencil in log density keeps the derivative at
    fixed physical temperature and avoids subtracting densities of very
    different scale.

    References: Chabrier [C90], Eq. (34), and the IIT free energy [IIT87].
    """
    nbar = float(nbar_e0)
    t_e = float(t_e_ha)
    h = float(log_density_step)
    if not (np.isfinite(nbar) and nbar > 0.0):
        raise ValueError("nbar_e0 must be positive and finite.")
    if not (np.isfinite(t_e) and t_e >= 0.0):
        raise ValueError("t_e_ha must be non-negative and finite.")
    if not (np.isfinite(h) and 1.0e-5 <= h <= 2.0e-2):
        raise ValueError("log_density_step must lie in [1e-5, 2e-2].")

    def scaled_free_energy_density(log_scale: float) -> float:
        scale = np.exp(log_scale)
        density = nbar * scale
        return scale * iit_exchange_correlation_free_energy_per_electron_au(
            density,
            t_e,
        )

    y_m2 = scaled_free_energy_density(-2.0 * h)
    y_m1 = scaled_free_energy_density(-h)
    y_0 = scaled_free_energy_density(0.0)
    y_p1 = scaled_free_energy_density(h)
    y_p2 = scaled_free_energy_density(2.0 * h)
    first_log_derivative = (y_m2 - 8.0 * y_m1 + 8.0 * y_p1 - y_p2) / (
        12.0 * h
    )
    second_log_derivative = (
        -y_p2 + 16.0 * y_p1 - 30.0 * y_0 + 16.0 * y_m1 - y_m2
    ) / (12.0 * h * h)

    kf = fermi_wavenumber_au(nbar)
    gamma_0 = (
        -kf
        * kf
        / (4.0 * np.pi * nbar)
        * (second_log_derivative - first_log_derivative)
    )
    if not np.isfinite(gamma_0):
        raise ValueError("Chabrier-1990 compressibility coefficient is not finite.")
    return float(gamma_0)


def gee_jellium_hubbard(k: np.ndarray, nbar_e0: float) -> np.ndarray:
    """
    Static Hubbard local-field correction.

      G_ee(k) = 0.5 * k^2 / (k^2 + k_F^2)

    Reference: the conventional Hubbard electron-gas approximation originating
    with Hubbard [H58]. This is the simple baseline, not Chabrier's separate
    Eq. (33) constrained Hubbard-form diagnostic.
    """
    k_arr, was_scalar = _as_array(k)
    kf = fermi_wavenumber_au(nbar_e0)
    k2 = k_arr * k_arr
    gee = 0.5 * k2 / (k2 + kf * kf)
    return _return_scalar_if_needed(gee, was_scalar)


def gee_jellium_utsumiichimaru(k: np.ndarray, nbar_e0: float) -> np.ndarray:
    """
    Static Utsumi-Ichimaru local-field correction for 3D jellium at T = 0.

    Reference: Utsumi and Ichimaru [UI82], especially the analytic static
    local-field form constrained by the compressibility and large-k limits.
    """
    k_arr, was_scalar = _as_array(k)
    nbar = float(nbar_e0)
    if nbar <= 0.0:
        raise ValueError("nbar_e0 must be positive.")

    kf, g0_ee_0, gamma_0 = _ui_zero_t_constraints(nbar)

    gee = _gee_utsumi_ichimaru_from_constraints(
        k_arr,
        kf,
        g0_ee_0,
        gamma_0,
    )
    return _return_scalar_if_needed(gee, was_scalar)


def gee_jellium_chabrier1990(
    k: np.ndarray,
    nbar_e0: float,
    t_e_ha: float,
) -> np.ndarray:
    """
    Chabrier-1990 finite-temperature jellium LFC ``G'_UI``.

    Chabrier [C90], Sec. 3.3, retains the complete Utsumi--Ichimaru [UI82]
    wave-number dependence (including its feature near ``2*k_F``), replaces
    its small-k coefficient by the finite-temperature IIT [IIT87]
    compressibility derivative [C90, Eq. (34)], and retains the
    zero-temperature contact value ``g_ee(0)`` because no finite-T fit was
    then available. This is the LFC selected in that paper; its Eq. (32) VS
    and Eq. (33) Hubbard forms are alternatives discussed for comparison.
    """
    k_arr, was_scalar = _as_array(k)
    nbar = float(nbar_e0)
    t_e = float(t_e_ha)
    if not (np.isfinite(nbar) and nbar > 0.0):
        raise ValueError("nbar_e0 must be positive and finite.")
    if not (np.isfinite(t_e) and t_e >= 0.0):
        raise ValueError("t_e_ha must be non-negative and finite.")

    kf, g0_ee_0, _ = _ui_zero_t_constraints(nbar)
    gamma_0 = chabrier1990_gamma0(nbar, t_e)
    gee = _gee_utsumi_ichimaru_from_constraints(
        k_arr,
        kf,
        g0_ee_0,
        gamma_0,
    )
    return _return_scalar_if_needed(gee, was_scalar)


def gee_jellium_vashistasingwi(k: np.ndarray, nbar_e0: float) -> np.ndarray:
    """
    Vashishta-Singwi LFC form used by Chabrier 1990 Eq. (32).

      G_VS(q) = A * [1 - exp(-q^2 / B)],  q = k / k_F

    `A` and `B` are fixed by the large-k limit `A = 1 - g(0)` and the
    small-k limit `A/B = gamma_0`. The present implementation uses the same
    zero-temperature Utsumi-Ichimaru constraints as a diagnostic model.

    References: the Vashishta--Singwi construction [VS72] and the finite-T
    comparison form in Chabrier [C90], Eq. (32).
    """
    k_arr, was_scalar = _as_array(k)
    kf, g0_ee_0, gamma_0 = _ui_zero_t_constraints(float(nbar_e0))
    a = max(1.0 - float(g0_ee_0), _TINY)
    b = max(a / max(float(gamma_0), _TINY), _TINY)
    q = k_arr / kf
    gee = a * (1.0 - np.exp(-(q * q) / b))
    return _return_scalar_if_needed(gee, was_scalar)


def gee_jellium_chabrier_hubbard(k: np.ndarray, nbar_e0: float) -> np.ndarray:
    """
    Chabrier 1990 finite-T Hubbard-form diagnostic, Eq. (33).

      G_H(q) = A q^2 / (q^2 + B),  q = k / k_F

    `A` and `B` are the same coefficients as in the Vashishta-Singwi form.
    This differs from the simple Hubbard model kept as `lfc_model="hubbard"`.

    Reference: Chabrier [C90], Eq. (33); compare the original Hubbard
    approximation [H58].
    """
    k_arr, was_scalar = _as_array(k)
    kf, g0_ee_0, gamma_0 = _ui_zero_t_constraints(float(nbar_e0))
    a = max(1.0 - float(g0_ee_0), _TINY)
    b = max(a / max(float(gamma_0), _TINY), _TINY)
    q2 = (k_arr / kf) ** 2
    gee = a * q2 / (q2 + b)
    return _return_scalar_if_needed(gee, was_scalar)


def _gbin_0_finite_t(t_ee_ha: float) -> float:
    """
    Finite-temperature binary part g_bin(0) used in the Gregori/Geldart-Vosko fit.

    Physical reference: the finite-temperature construction used by Gregori
    et al. [G07].  Implementation adapted from JaXRTS [JAXRTS]; see the module
    provenance notice and ``THIRD_PARTY_NOTICES.md``.
    """
    t_ee = float(t_ee_ha)
    if t_ee <= 0.0:
        raise ValueError("t_ee_ha must be positive.")

    xi = 1.0 / t_ee
    prefactor = np.sqrt(2.0 * np.pi) * xi ** 1.5

    def integrand_u(u: float) -> float:
        if u <= 0.0:
            return 0.0
        arg = np.pi / u
        if arg > 700.0:
            return 0.0
        return u * np.exp(-xi * u * u) / np.expm1(arg)

    val, _ = quad(
        integrand_u,
        0.0,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-12,
        limit=400,
    )
    return prefactor * val


def gee_jellium_geldartvosko(
    k: np.ndarray,
    nbar_e0: float,
    t_e_ha: float,
) -> np.ndarray:
    """
    Static finite-temperature Geldart--Vosko branch used by Gregori et al.

    The original interacting-electron-gas screening calculation is [GV66].
    The effective temperature, contact-value, and fit formulas implemented
    here follow the finite-temperature construction in Gregori et al. [G07];
    this function is therefore not a verbatim zero-temperature GV formula.
    The numerical implementation is adapted from JaXRTS [JAXRTS], translated
    from JAX quantities to NumPy atomic units.
    """
    k_arr, was_scalar = _as_array(k)

    nbar = float(nbar_e0)
    t_e = float(t_e_ha)
    if nbar <= 0.0:
        raise ValueError("nbar_e0 must be positive.")
    if t_e < 0.0:
        raise ValueError("t_e_ha must be non-negative.")

    kf = fermi_wavenumber_au(nbar)
    ef = 0.5 * kf * kf
    rs = wigner_seitz_radius_au(nbar)

    denom_tq = 1.3251 - 0.1779 * np.sqrt(rs)
    if denom_tq <= 0.0:
        raise ValueError("Gregori/Geldart-Vosko T_q fit became non-positive.")
    t_q = ef / denom_tq
    t_ee = np.sqrt(t_e * t_e + t_q * t_q)

    gamma_ee = max(coupling_parameter_ee_au(nbar, t_ee), _TINY)
    gamma_t = (12.0 * np.pi**2) ** (1.0 / 3.0) * (
        0.0999305
        + 0.0187058 / gamma_ee
        + 0.0013240 / (gamma_ee ** (4.0 / 3.0))
        - 0.0479236 / (gamma_ee ** (2.0 / 3.0))
    )

    c_sc = 1.0754
    h_0 = (c_sc * gamma_ee ** 1.5) / (
        ((c_sc / np.sqrt(3.0)) ** 4 + gamma_ee**4) ** 0.25
    )

    g_bin_0 = _gbin_0_finite_t(t_ee)
    g_t_ee_0 = g_bin_0 * np.exp(h_0)
    one_minus_g = 1.0 - g_t_ee_0
    if one_minus_g <= 0.0:
        raise ValueError("Encountered unphysical g_T,ee(0) >= 1.")

    denom = (kf * kf) / gamma_t + (k_arr * k_arr) / one_minus_g
    gee = (k_arr * k_arr) / denom
    return _return_scalar_if_needed(gee, was_scalar)


def gee_jellium_gregori2007(
    k: np.ndarray,
    nbar_e0: float,
    t_e_ha: float,
) -> np.ndarray:
    """
    Static Gregori-2007 interpolation:

      G_ee(k) = [G0(k) + Theta * GT(k)] / (1 + Theta),
      Theta   = T_e / T_F.

    Here `G0` is the zero-temperature Utsumi--Ichimaru form [UI82] and `GT`
    is the finite-temperature Geldart--Vosko branch [GV66] as assembled by
    Gregori et al. [G07].

    Physical references: Utsumi--Ichimaru [UI82], Geldart--Vosko [GV66], and
    Gregori et al. [G07].  Implementation adapted from JaXRTS [JAXRTS]; see
    ``THIRD_PARTY_NOTICES.md``.
    """
    nbar = float(nbar_e0)
    t_e = float(t_e_ha)
    if nbar <= 0.0:
        raise ValueError("nbar_e0 must be positive.")
    if t_e < 0.0:
        raise ValueError("t_e_ha must be non-negative.")

    ef = max(fermi_energy_au(nbar), _TINY)
    theta = t_e / ef
    g0 = gee_jellium_utsumiichimaru(k, nbar)
    gt = gee_jellium_geldartvosko(k, nbar, t_e)
    return (g0 + theta * gt) / (1.0 + theta)


def gee_jellium(
    k: np.ndarray,
    nbar_e0: float,
    *,
    model: str = "chabrier1990",
    t_e_ha: float | None = None,
) -> np.ndarray:
    """
    Unified LFC dispatcher used by the QOZ response chain.

    Canonical choices and their primary references are recorded in
    ``LFC_MODEL_REFERENCE_KEYS``:

    - ``none``: RPA, `G=0` [RPA52];
    - ``hubbard``: simple static Hubbard form [H58];
    - ``utsumiichimaru``: zero-temperature UI [UI82];
    - ``chabrier1990``: finite-temperature `G'_UI` [UI82, IIT87, C90];
    - ``vashistasingwi``: [VS72] / Chabrier Eq. (32) [C90];
    - ``chabrier_hubbard``: Chabrier Eq. (33) [C90];
    - ``geldartvosko``: finite-T GV branch [GV66, G07];
    - ``gregori2007``: UI/GV interpolation [UI82, GV66, G07].
    """
    key = str(model).lower().strip()
    if key in ("none", "zero"):
        return np.zeros_like(np.asarray(k, dtype=float))
    if key == "hubbard":
        return gee_jellium_hubbard(k, nbar_e0)
    if key in ("utsumiichimaru", "utsumi_ichimaru", "ui"):
        return gee_jellium_utsumiichimaru(k, nbar_e0)
    if key in (
        "chabrier1990",
        "chabrier_1990",
        "chabrier-1990",
        "chabrier_ui",
        "chabrier-ui",
        "chabrier",
    ):
        if t_e_ha is None:
            raise ValueError("chabrier1990 requires t_e_ha.")
        return gee_jellium_chabrier1990(k, nbar_e0, t_e_ha)
    if key in ("vashistasingwi", "vashista_singwi", "vashishta_singwi", "vs"):
        return gee_jellium_vashistasingwi(k, nbar_e0)
    if key in ("chabrier_hubbard", "chabrier-hubbard", "ch_hubbard", "ch"):
        return gee_jellium_chabrier_hubbard(k, nbar_e0)
    if key in ("geldartvosko", "geldart_vosko", "gv"):
        if t_e_ha is None:
            raise ValueError("geldartvosko requires t_e_ha.")
        return gee_jellium_geldartvosko(k, nbar_e0, t_e_ha)
    if key in ("gregori2007", "gregori"):
        if t_e_ha is None:
            raise ValueError("gregori2007 requires t_e_ha.")
        return gee_jellium_gregori2007(k, nbar_e0, t_e_ha)
    raise ValueError(f"Unsupported lfc model: {model!r}")


def cee_from_gee(
    k: np.ndarray,
    gee: np.ndarray,
    t_e_ha: float,
    *,
    k0_value: float | None = None,
) -> np.ndarray:
    """
    Build the electron-electron direct correlation function from a supplied
    local-field correction.

      C_ee(k) = - beta_e * (4*pi/k^2) * [1 - G_ee(k)]
      beta_e  = 1 / T_e

    Parameters
    ----------
    k
        Wave-number grid in Bohr^-1.
    gee
        Local-field correction G_ee(k), dimensionless.
    t_e_ha
        Electron thermal energy k_B T_e in Hartree.
    k0_value
        Optional value used exactly at k=0. The QOZ DST lattice starts at the
        first non-zero mode, so this is mainly for defensive standalone use.

    Returns
    -------
    ndarray
        C_ee(k) in atomic units.
    """
    k = np.asarray(k, dtype=float)
    gee = np.asarray(gee, dtype=float)
    if k.shape != gee.shape:
        raise ValueError("k and gee must have the same shape.")
    beta_e = 1.0 / max(float(t_e_ha), _TINY)

    cee = np.empty_like(k, dtype=float)
    mask = np.abs(k) > 0.0
    cee[mask] = -beta_e * (4.0 * np.pi / (k[mask] ** 2)) * (1.0 - gee[mask])
    fill = np.nan if k0_value is None else float(k0_value)
    cee[~mask] = fill
    return cee


def cee_jellium_from_lfc(
    k: np.ndarray,
    nbar_e0: float,
    *,
    model: str = "chabrier1990",
    t_e_ha: float,
    k0_value: float | None = None,
) -> np.ndarray:
    """
    Convenience wrapper:

      model -> G_ee(k) -> C_ee(k)

    This keeps the algebra explicit in one place when debugging electron
    response models.
    """
    gee = gee_jellium(k, nbar_e0, model=model, t_e_ha=t_e_ha)
    return cee_from_gee(k, gee, t_e_ha, k0_value=k0_value)
