"""
otter/ionic/response.py

Electron local-field correction models used by the QOZ response chain.

Scope
-----
This module collects jellium quantities and static LFC models so that:
  - `qoz.py` keeps the response assembly logic readable
  - different `G_ee(k)` approximations can be swapped cleanly
  - non-equilibrium QOZ calculations can use an explicit electron temperature
    `T_e` while the ion-ion HNC part continues to use `T_i`
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import i1

_TINY = np.finfo(float).tiny


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


def gee_jellium_hubbard(k: np.ndarray, nbar_e0: float) -> np.ndarray:
    """
    Static Hubbard local-field correction.

      G_ee(k) = 0.5 * k^2 / (k^2 + k_F^2)
    """
    k_arr, was_scalar = _as_array(k)
    kf = fermi_wavenumber_au(nbar_e0)
    k2 = k_arr * k_arr
    gee = 0.5 * k2 / (k2 + kf * kf)
    return _return_scalar_if_needed(gee, was_scalar)


def gee_jellium_utsumiichimaru(k: np.ndarray, nbar_e0: float) -> np.ndarray:
    """
    Static Utsumi-Ichimaru local-field correction for 3D jellium at T = 0.
    """
    k_arr, was_scalar = _as_array(k)
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

    a_fit = 0.029
    b_fit = (9.0 / 16.0) * gamma_0 - (3.0 / 64.0) * (1.0 - g0_ee_0) - (16.0 / 15.0) * a_fit
    c_fit = -(3.0 / 4.0) * gamma_0 + (9.0 / 16.0) * (1.0 - g0_ee_0) - (16.0 / 5.0) * a_fit

    kf = fermi_wavenumber_au(nbar)
    q = k_arr / kf
    lq = _ui_log_factor(q)

    gee = (
        a_fit * q**4
        + b_fit * q**2
        + c_fit
        + (a_fit * q**4 + (b_fit + (8.0 / 3.0) * a_fit) * q**2 - c_fit) * lq
    )
    return _return_scalar_if_needed(gee, was_scalar)


def _gbin_0_finite_t(t_ee_ha: float) -> float:
    """
    Finite-temperature binary part g_bin(0) used in the Gregori/Geldart-Vosko fit.
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
    Static Geldart-Vosko local-field correction for finite-T jellium.
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
    model: str = "hubbard",
    t_e_ha: float | None = None,
) -> np.ndarray:
    """
    Unified LFC dispatcher used by the QOZ response chain.
    """
    key = str(model).lower().strip()
    if key in ("none", "zero"):
        return np.zeros_like(np.asarray(k, dtype=float))
    if key == "hubbard":
        return gee_jellium_hubbard(k, nbar_e0)
    if key in ("utsumiichimaru", "utsumi_ichimaru", "ui"):
        return gee_jellium_utsumiichimaru(k, nbar_e0)
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
    model: str = "hubbard",
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
