import numpy as np
import pytest

from otter.ionic import LFC_MODEL_REFERENCE_KEYS
from otter.ionic.lfc import (
    chabrier1990_gamma0,
    fermi_energy_au,
    fermi_wavenumber_au,
    gee_jellium,
    gee_jellium_chabrier1990,
    iit_exchange_correlation_free_energy_per_electron_au,
)
from otter.ionic.qoz import QOZResponseOptions, chi_ee_from_eq17


def test_lfc_reference_map_covers_every_canonical_dispatch_model():
    assert set(LFC_MODEL_REFERENCE_KEYS) == {
        "none",
        "hubbard",
        "utsumiichimaru",
        "chabrier1990",
        "vashistasingwi",
        "chabrier_hubbard",
        "geldartvosko",
        "gregori2007",
    }
    assert all(LFC_MODEL_REFERENCE_KEYS.values())


def _density_from_rs(rs):
    return 3.0 / (4.0 * np.pi * float(rs) ** 3)


def test_iit_reference_point_rs1_theta1():
    nbar = _density_from_rs(1.0)
    t_e = fermi_energy_au(nbar)

    # Independent regression values obtained by evaluating IIT Eq. (3.83)
    # with the published decimal coefficients.
    assert iit_exchange_correlation_free_energy_per_electron_au(
        nbar, t_e
    ) == pytest.approx(-0.4027754001328252, rel=2.0e-13)
    assert chabrier1990_gamma0(nbar, t_e) == pytest.approx(
        0.2931154078682697, rel=2.0e-10
    )


def test_chabrier_gamma_matches_fixed_temperature_density_curvature():
    nbar = _density_from_rs(1.0)
    t_e = 0.7 * fermi_energy_au(nbar)
    dn_fraction = 1.0e-3

    def free_energy_density(scale):
        density = nbar * scale
        return density * iit_exchange_correlation_free_energy_per_electron_au(
            density, t_e
        )

    f_m2 = free_energy_density(1.0 - 2.0 * dn_fraction)
    f_m1 = free_energy_density(1.0 - dn_fraction)
    f_0 = free_energy_density(1.0)
    f_p1 = free_energy_density(1.0 + dn_fraction)
    f_p2 = free_energy_density(1.0 + 2.0 * dn_fraction)
    second_density_derivative = (
        -f_p2 + 16.0 * f_p1 - 30.0 * f_0 + 16.0 * f_m1 - f_m2
    ) / (12.0 * (nbar * dn_fraction) ** 2)
    expected = (
        -fermi_wavenumber_au(nbar) ** 2
        / (4.0 * np.pi)
        * second_density_derivative
    )

    assert chabrier1990_gamma0(nbar, t_e) == pytest.approx(
        expected, rel=2.0e-7, abs=2.0e-9
    )


def test_chabrier_lfc_obeys_small_k_compressibility_coefficient():
    nbar = _density_from_rs(2.0)
    t_e = 0.3 * fermi_energy_au(nbar)
    kf = fermi_wavenumber_au(nbar)
    q = np.array([1.0e-3, 2.0e-3])

    gee = gee_jellium_chabrier1990(q * kf, nbar, t_e)
    gamma = chabrier1990_gamma0(nbar, t_e)

    assert gee / q**2 == pytest.approx(
        np.full_like(q, gamma), rel=8.0e-6, abs=2.0e-9
    )


def test_chabrier_dispatcher_requires_temperature_and_preserves_shape():
    nbar = _density_from_rs(1.0)
    k = np.linspace(0.0, 5.0, 101)

    with pytest.raises(ValueError, match="requires t_e_ha"):
        gee_jellium(k, nbar, model="chabrier1990")

    direct = gee_jellium_chabrier1990(k, nbar, 0.5)
    dispatched = gee_jellium(
        k,
        nbar,
        model="chabrier1990",
        t_e_ha=0.5,
    )
    assert dispatched.shape == k.shape
    assert np.all(np.isfinite(dispatched))
    assert dispatched == pytest.approx(direct, rel=0.0, abs=0.0)
    assert dispatched[0] == pytest.approx(0.0, abs=1.0e-15)


def test_chabrier_lfc_runs_through_production_eq17_response():
    nbar = _density_from_rs(1.5)
    t_e = 0.4 * fermi_energy_au(nbar)
    kf = fermi_wavenumber_au(nbar)
    k = np.linspace(0.02, 5.0, 201) * kf

    chi_ee, chi0, gee = chi_ee_from_eq17(
        k,
        nbar_e0=nbar,
        electron_temperature_ha=t_e,
        mu_jellium_ha=fermi_energy_au(nbar),
        response_options=QOZResponseOptions(
            chi0_model="lindhard_fd",
            lfc_model="chabrier1990",
            electron_temperature_ha=t_e,
            lindhard_p_points=1024,
        ),
    )

    assert chi_ee.shape == k.shape
    assert np.all(np.isfinite(chi_ee))
    assert np.all(chi_ee < 0.0)
    assert np.all(np.isfinite(chi0))
    assert np.all(chi0 < 0.0)
    assert gee == pytest.approx(
        gee_jellium_chabrier1990(k, nbar, t_e),
        rel=0.0,
        abs=0.0,
    )


@pytest.mark.parametrize("rs", [0.05, 0.2, 1.0, 5.0, 20.0])
@pytest.mark.parametrize("theta", [0.0, 1.0e-4, 0.1, 1.0, 10.0, 1.0e3])
def test_chabrier_lfc_is_finite_over_supported_jellium_grid(rs, theta):
    nbar = _density_from_rs(rs)
    t_e = theta * fermi_energy_au(nbar)
    kf = fermi_wavenumber_au(nbar)
    k = np.linspace(0.0, 6.0 * kf, 301)

    gamma = chabrier1990_gamma0(nbar, t_e)
    gee = gee_jellium_chabrier1990(k, nbar, t_e)

    assert np.isfinite(gamma)
    assert gamma >= 0.0
    assert np.all(np.isfinite(gee))
