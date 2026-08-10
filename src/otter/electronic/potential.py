"""
otter/electronic/potential.py

Purpose
-------
Assemble effective electron potentials for the AA full/external systems.

Methods
-------
- Combine nucleus term, Hartree term, and XC difference.
- Use g_II(r) to include the ion-background subtraction term.

Equations
---------
Full system (Starrett2014 Eq. 4):
  V_eff(r) = -Z/r + integral (n_full - n0 g_II) / |r-r'| + V_xc[n_full] - V_xc[n0]

External system (Starrett2014 Eq. 7):
  V_eff_ext(r) = integral (n_ext - n0 g_II) / |r-r'| + V_xc[n_ext] - V_xc[n0]

References
----------
- :cite:`StarrettSaumon2014`, Eqs. (4) and (7), DOI
  10.1016/j.hedp.2013.12.001.

The spherical quadrature, origin handling, and XC finite-core regularization
are Otter numerical implementation choices; they are not attributed to the
reference paper.
"""
from __future__ import annotations

import numpy as np

from otter.electronic.xc import resolve_gga_core_radius, xc_potential


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoid integral with values on one monotonic 1D grid."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 1:
        raise ValueError("y and x must be 1D arrays.")
    if y.size != x.size:
        raise ValueError("y and x must have the same length.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")

    out = np.zeros_like(y, dtype=float)
    if y.size == 1:
        return out

    dx = np.diff(x)
    avg = 0.5 * (y[1:] + y[:-1])
    out[1:] = np.cumsum(avg * dx)
    return out


def spherical_hartree_potential(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Compute the spherical Hartree potential from a radial charge density.

    The expression is

        V_H(r) = 4 pi [ 1/r int_0^r rho(r') r'^2 dr'
                        + int_r^R rho(r') r' dr' ].
    """
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if r.ndim != 1 or rho.ndim != 1:
        raise ValueError("r and rho must be 1D arrays.")
    if r.size != rho.size:
        raise ValueError("r and rho must have the same length.")
    if np.any(r <= 0):
        raise ValueError("r must be > 0 for spherical potential.")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r must be strictly increasing.")

    int_f = _cumtrapz(rho * r**2, r)
    int_g = _cumtrapz(rho * r, r)
    return 4.0 * np.pi * (int_f / r + (int_g[-1] - int_g))


def spherical_hartree_potential_screened(
    r: np.ndarray,
    rho: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Compute a screened Poisson-Helmholtz Hartree potential."""
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if kappa <= 0.0:
        return spherical_hartree_potential(r, rho)

    if r.ndim != 1 or rho.ndim != 1:
        raise ValueError("r and rho must be 1D arrays.")
    if r.size != rho.size:
        raise ValueError("r and rho must have the same length.")
    if np.any(r <= 0):
        raise ValueError("r must be > 0 for spherical potential.")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r must be strictly increasing.")

    kr = kappa * r
    kr_clip = np.clip(kr, 0.0, 50.0)
    exp_pkr = np.exp(kr_clip)
    exp_mkr = np.exp(-kr)
    sinh_kr = 0.5 * (exp_pkr - exp_mkr)

    inner = _cumtrapz(rho * r * sinh_kr, r)
    outer = _cumtrapz(rho * r * exp_mkr, r)
    return (4.0 * np.pi / (kappa * r)) * (
        exp_mkr * inner + sinh_kr * (outer[-1] - outer)
    )


def _as_array_like(x: float | np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Broadcast a scalar or validate an array against ref.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return np.full_like(ref, float(arr), dtype=float)
    if arr.shape != ref.shape:
        raise ValueError("Input array must match reference shape.")
    return arr


def _ion_sphere_background_hartree(
    r: np.ndarray,
    *,
    n0: float,
    r_ws: float,
) -> np.ndarray:
    """Analytic Hartree potential of ``-n0 Theta(r-r_ws)`` in the box.

    Sampling the ion-sphere discontinuity on a nonuniform radial grid creates
    a grid-dependent background charge.  The analytic expression keeps the
    full and external Thomas--Fermi maps neutral to the requested tolerance.
    """
    r_arr = np.asarray(r, dtype=float)
    if r_arr.ndim != 1 or r_arr.size == 0:
        raise ValueError("r must be a non-empty one-dimensional array.")
    if np.any(r_arr <= 0.0) or np.any(np.diff(r_arr) <= 0.0):
        raise ValueError("r must be positive and strictly increasing.")
    r_box = float(r_arr[-1])
    radius = float(np.clip(float(r_ws), 0.0, r_box))
    density = -float(n0)
    out = np.empty_like(r_arr)
    inside = r_arr < radius
    out[inside] = 2.0 * np.pi * density * (r_box**2 - radius**2)
    outside = ~inside
    r_out = r_arr[outside]
    out[outside] = 4.0 * np.pi * density * (
        (r_out**3 - radius**3) / (3.0 * r_out)
        + 0.5 * (r_box**2 - r_out**2)
    )
    return out


def _require_sharp_ion_sphere_profile(
    r: np.ndarray,
    g_ii: np.ndarray,
    radius: float,
) -> None:
    """Fail closed if the analytic IS shortcut is paired with a non-IS g(r).

    The shortcut integrates ``-n0 Theta(r-R_ws)`` analytically.  It is
    therefore mathematically equivalent to the sampled source only when the
    supplied profile is that same sharp step.  In particular it must never be
    used for the self-consistent ``g_II`` in Starrett--Saumon (2014),
    Sec. 2.4, because doing so would silently discard the feedback in Eqs.
    (4) and (7).
    """
    expected = (np.asarray(r, dtype=float) >= float(radius)).astype(float)
    if not np.array_equal(np.asarray(g_ii, dtype=float), expected):
        raise ValueError(
            "ion_sphere_radius analytic background requires the literal "
            "sharp ion-sphere g_ii step; use ion_sphere_radius=None for a "
            "tabulated QOZ/HNC background."
        )


def effective_potential_full(
    r: np.ndarray,
    n_full: np.ndarray,
    n0: float | np.ndarray,
    g_ii: np.ndarray,
    Z: float,
    xc_model: str = "dirac",
    kappa: float = 0.0,
    ion_sphere_radius: float | None = None,
    gga_core_mode: str = "finite",
    gga_core_zr: float = 0.05,
) -> np.ndarray:
    """
    Effective potential for the full AA system (with nucleus).

    Parameters
    ----------
    r : ndarray
        Radial grid (Bohr).
    n_full : ndarray
        Full electron density (Bohr^-3).
    n0 : float or ndarray
        Field-free electron density (Bohr^-3).
    g_ii : ndarray
        Ion-ion pair distribution function.
    Z : float
        Nuclear charge.
    xc_model : str
        XC model name; GGA models use radial density gradients.
    kappa : float
        Poisson-Helmholtz screening parameter (Bohr^-1).
    ion_sphere_radius : float or None
        If supplied with scalar ``n0`` and ``kappa=0``, integrate the sharp
        ion-sphere background analytically.
    gga_core_mode : {"finite", "strict"}
        Nuclear-core treatment for GGA models.
    gga_core_zr : float
        Dimensionless finite-core radius ``Z*r_c``.

    Returns
    -------
    ndarray
        Effective potential V_eff(r) (Ha).
    """
    r = np.asarray(r, dtype=float)
    n_full = np.asarray(n_full, dtype=float)
    g_ii = np.asarray(g_ii, dtype=float)

    if r.ndim != 1:
        raise ValueError("r must be 1D.")
    if n_full.shape != r.shape or g_ii.shape != r.shape:
        raise ValueError("n_full and g_ii must match r shape.")

    n0_arr = _as_array_like(n0, r)
    use_analytic_background = (
        ion_sphere_radius is not None and np.asarray(n0).ndim == 0 and kappa <= 0.0
    )
    if use_analytic_background:
        _require_sharp_ion_sphere_profile(
            r,
            g_ii,
            float(ion_sphere_radius),
        )
        v_h = spherical_hartree_potential(r, n_full)
        v_h = v_h + _ion_sphere_background_hartree(
            r,
            n0=float(np.asarray(n0)),
            r_ws=float(ion_sphere_radius),
        )
    elif kappa > 0.0:
        rho = n_full - n0_arr * g_ii
        v_h = spherical_hartree_potential_screened(r, rho, kappa)
    else:
        rho = n_full - n0_arr * g_ii
        v_h = spherical_hartree_potential(r, rho)
    gga_core_radius = resolve_gga_core_radius(
        xc_model,
        nuclear_charge=float(Z),
        mode=gga_core_mode,
        core_zr=gga_core_zr,
        r=r,
    )
    v_xc = xc_potential(
        n_full,
        model=xc_model,
        r=r,
        gga_core_radius_bohr=gga_core_radius,
    ) - xc_potential(
        n0_arr,
        model=xc_model,
        r=r,
        gga_core_radius_bohr=gga_core_radius,
    )

    return -Z / r + v_h + v_xc


def effective_potential_external(
    r: np.ndarray,
    n_ext: np.ndarray,
    n0: float | np.ndarray,
    g_ii: np.ndarray,
    xc_model: str = "dirac",
    kappa: float = 0.0,
    ion_sphere_radius: float | None = None,
    *,
    nuclear_charge: float | None = None,
    gga_core_mode: str = "strict",
    gga_core_zr: float = 0.05,
) -> np.ndarray:
    """
    Effective potential for the external AA system (no nucleus).

    Parameters
    ----------
    r : ndarray
        Radial grid (Bohr).
    n_ext : ndarray
        External electron density (Bohr^-3).
    n0 : float or ndarray
        Field-free electron density (Bohr^-3).
    g_ii : ndarray
        Ion-ion pair distribution function.
    xc_model : str
        XC model name; GGA models use radial density gradients.
    kappa : float
        Poisson-Helmholtz screening parameter (Bohr^-1).
    ion_sphere_radius : float or None
        Optional analytic sharp-background radius; see
        :func:`effective_potential_full`.
    nuclear_charge : float or None
        Species nuclear charge used to resolve a finite GGA core.
    gga_core_mode : {"finite", "strict"}
        Nuclear-core treatment for GGA models.
    gga_core_zr : float
        Dimensionless finite-core radius ``Z*r_c``.

    Returns
    -------
    ndarray
        Effective potential V_eff_ext(r) (Ha).
    """
    r = np.asarray(r, dtype=float)
    n_ext = np.asarray(n_ext, dtype=float)
    g_ii = np.asarray(g_ii, dtype=float)

    if r.ndim != 1:
        raise ValueError("r must be 1D.")
    if n_ext.shape != r.shape or g_ii.shape != r.shape:
        raise ValueError("n_ext and g_ii must match r shape.")

    n0_arr = _as_array_like(n0, r)
    use_analytic_background = (
        ion_sphere_radius is not None and np.asarray(n0).ndim == 0 and kappa <= 0.0
    )
    if use_analytic_background:
        _require_sharp_ion_sphere_profile(
            r,
            g_ii,
            float(ion_sphere_radius),
        )
        v_h = spherical_hartree_potential(r, n_ext)
        v_h = v_h + _ion_sphere_background_hartree(
            r,
            n0=float(np.asarray(n0)),
            r_ws=float(ion_sphere_radius),
        )
    elif kappa > 0.0:
        rho = n_ext - n0_arr * g_ii
        v_h = spherical_hartree_potential_screened(r, rho, kappa)
    else:
        rho = n_ext - n0_arr * g_ii
        v_h = spherical_hartree_potential(r, rho)
    mode_key = str(gga_core_mode).strip().lower()
    if mode_key == "finite" and nuclear_charge is None:
        raise ValueError(
            "nuclear_charge is required for finite-core GGA external potentials."
        )
    gga_core_radius = resolve_gga_core_radius(
        xc_model,
        nuclear_charge=(1.0 if nuclear_charge is None else float(nuclear_charge)),
        mode=mode_key,
        core_zr=gga_core_zr,
        r=r,
    )
    v_xc = xc_potential(
        n_ext,
        model=xc_model,
        r=r,
        gga_core_radius_bohr=gga_core_radius,
    ) - xc_potential(
        n0_arr,
        model=xc_model,
        r=r,
        gga_core_radius_bohr=gga_core_radius,
    )

    return v_h + v_xc
