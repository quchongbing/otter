from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import otter.electronic.xc as xc
import otter.workflows as workflows
from otter.electronic.full_external import FullExternalConfig, _build_ks_config


class _FakeFunctional:
    def __init__(self, name: str):
        self.name = name

    def compute(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rho = np.asarray(inputs["rho"], dtype=float).reshape(-1)
        constants = {
            "lda_x": (1.0, 2.0, None),
            "gga_x_pbe": (1.0, 2.0, 3.0),
            "gga_c_pbe": (4.0, 5.0, 6.0),
        }
        zk, vrho, vsigma = constants[self.name]
        output = {
            "zk": np.full((rho.size, 1), zk),
            "vrho": np.full((rho.size, 1), vrho),
        }
        if vsigma is not None:
            output["vsigma"] = np.full((rho.size, 1), vsigma)
        return output


class _VariationalFakeFunctional:
    """Two identical halves of e(n,sigma)=a*n^2/2+b*sigma."""

    def __init__(self, name: str):
        self.name = name

    def compute(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rho = np.asarray(inputs["rho"], dtype=float).reshape(-1)
        sigma = np.asarray(inputs["sigma"], dtype=float).reshape(-1)
        a = 1.7
        b = 0.35
        return {
            "zk": (0.25 * a * rho + 0.5 * b * sigma / rho)[:, None],
            "vrho": (0.5 * a * rho)[:, None],
            "vsigma": np.full((rho.size, 1), 0.5 * b),
        }


class _FakeCitationFunctional:
    """Small stand-in for pylibxc's public provenance API."""

    def __init__(self, name: str):
        self.name = name

    def get_number(self) -> int:
        return {"gga_x_pbe": 101, "gga_c_pbe": 130}[self.name]

    def get_name(self) -> str:
        return f"display name for {self.name}"

    def get_references(self) -> list[str]:
        return [f"primary reference for {self.name}"]

    def get_doi(self) -> list[str]:
        return ["10.1103/PhysRevLett.77.3865"]


def test_dirac_and_none_models_remain_dependency_free() -> None:
    density = np.array([-1.0, 0.0, 0.1, np.nan])
    expected = xc.dirac_exchange_potential(density)

    np.testing.assert_allclose(xc.xc_potential(density), expected)
    np.testing.assert_array_equal(
        xc.xc_potential(density, model="none"),
        np.zeros_like(density),
    )


def test_dirac_provenance_is_dependency_free_and_cites_primary_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_import() -> object:
        raise AssertionError("built-in provenance must not import pylibxc")

    monkeypatch.setattr(xc, "_load_pylibxc", unexpected_import)
    provenance = xc.xc_provenance("dirac")

    assert provenance["provider"] == "otter_builtin"
    assert provenance["software_doi"] is None
    assert provenance["components"][0]["dois"] == [
        "10.1017/S0305004100016108"
    ]


def test_libxc_provenance_reports_version_ids_and_runtime_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePylibxc:
        __version__ = "9.8.7-test"

    monkeypatch.setattr(xc, "_load_pylibxc", lambda: FakePylibxc())
    monkeypatch.setattr(xc, "_make_libxc_functional", _FakeCitationFunctional)

    provenance = xc.xc_provenance("pbe")

    assert provenance["provider"] == "libxc"
    assert provenance["provider_version"] == "9.8.7-test"
    assert provenance["software_doi"] == "10.1016/j.softx.2017.11.002"
    assert [item["id"] for item in provenance["components"]] == [
        "gga_x_pbe",
        "gga_c_pbe",
    ]
    assert [item["number"] for item in provenance["components"]] == [101, 130]
    assert provenance["components"][0]["references"] == [
        "primary reference for gga_x_pbe"
    ]


def test_unknown_model_explains_explicit_libxc_syntax() -> None:
    with pytest.raises(ValueError, match="libxc:<functional>"):
        xc.xc_potential(np.array([0.1]), model="not-a-functional")


def test_gga_requires_radial_grid_before_loading_libxc() -> None:
    with pytest.raises(ValueError, match="radial grid"):
        xc.xc_potential(np.array([0.1, 0.2]), model="pbe")


def test_meta_gga_is_rejected_with_missing_input_explanation() -> None:
    r = np.linspace(0.1, 1.0, 8)
    with pytest.raises(ValueError, match="kinetic-energy density"):
        xc.xc_potential(
            np.exp(-r),
            model="libxc:mgga_x_scan",
            r=r,
        )


def test_libxc_lda_dispatch_preserves_density_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xc, "_make_libxc_functional", _FakeFunctional)
    density = np.array([[0.1, 0.2], [0.3, 0.4]])

    potential = xc.xc_potential(density, model="libxc:lda_x")
    energy_density = xc.xc_energy_density(density, model="libxc:lda_x")

    np.testing.assert_array_equal(potential, np.full_like(density, 2.0))
    np.testing.assert_allclose(energy_density, density)


def test_pbe_alias_adds_radial_gga_functional_derivative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xc, "_make_libxc_functional", _FakeFunctional)
    radius = np.linspace(0.5, 2.0, 31)
    density = 1.0 + radius

    potential = xc.xc_potential(density, model="pbe", r=radius)
    explicit = xc.xc_potential(
        density,
        model="libxc:gga_x_pbe+gga_c_pbe",
        r=radius,
    )
    energy_density = xc.xc_energy_density(density, model="pbe", r=radius)

    # The fake pair gives vrho=7, vsigma=9, and dn/dr=1. In the continuum
    # limit v_xc = 7 - 2/r^2 d(9 r^2)/dr = 7 - 36/r. The production operator
    # is instead the exact derivative of a shell-weighted discrete energy, so
    # compare away from its natural-boundary rows.
    np.testing.assert_allclose(
        potential[3:-3],
        (7.0 - 36.0 / radius)[3:-3],
        rtol=1.2e-3,
        atol=3.0e-2,
    )
    np.testing.assert_array_equal(explicit, potential)
    np.testing.assert_allclose(energy_density, 5.0 * density)


def test_gga_potential_is_discrete_energy_derivative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xc, "_make_libxc_functional", _VariationalFakeFunctional)
    xi = np.linspace(np.sqrt(1.0e-4), np.sqrt(3.0), 240)
    radius = xi**2
    density = 0.4 + 0.3 * np.exp(-radius)
    perturbation = np.sin(np.pi * radius / radius[-1]) ** 2
    core_radius = 0.08

    potential = xc.xc_potential(
        density,
        model="pbe",
        r=radius,
        gga_core_radius_bohr=core_radius,
    )
    weights = xc._radial_shell_weights(radius)
    epsilon = 2.0e-6

    def total_energy(values: np.ndarray) -> float:
        energy_density = xc.xc_energy_density(
            values,
            model="pbe",
            r=radius,
            gga_core_radius_bohr=core_radius,
        )
        return float(np.sum(weights * energy_density))

    finite_difference = (
        total_energy(density + epsilon * perturbation)
        - total_energy(density - epsilon * perturbation)
    ) / (2.0 * epsilon)
    potential_derivative = float(np.sum(weights * potential * perturbation))

    np.testing.assert_allclose(
        potential_derivative,
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-10,
    )


def test_finite_gga_core_is_smooth_and_matches_strict_outside_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xc, "_make_libxc_functional", _VariationalFakeFunctional)
    xi = np.linspace(np.sqrt(1.0e-5), np.sqrt(2.0), 512)
    radius = xi**2
    density = 120.0 * np.exp(-12.0 * radius)
    core_radius = 0.05 / 6.0

    finite = xc.xc_potential(
        density,
        model="pbe",
        r=radius,
        gga_core_radius_bohr=core_radius,
    )
    strict = xc.xc_potential(density, model="pbe", r=radius)

    assert np.all(np.isfinite(finite))
    outside = radius >= core_radius + 3.0 * np.max(np.diff(radius[radius <= 1.2 * core_radius]))
    np.testing.assert_allclose(finite[outside], strict[outside], rtol=0.0, atol=0.0)
    innermost = radius <= 0.05 * core_radius
    assert np.max(np.abs(finite[innermost])) < 5.0e2
    assert np.max(np.abs(finite[innermost])) < 0.01 * np.max(
        np.abs(strict[innermost])
    )


def test_resolve_gga_core_radius_validates_resolution() -> None:
    radius = np.geomspace(1.0e-5, 2.0, 64)

    assert xc.resolve_gga_core_radius(
        "lda_pw", nuclear_charge=6.0, r=radius
    ) is None
    assert xc.resolve_gga_core_radius(
        "pbe", nuclear_charge=6.0, mode="strict", r=radius
    ) is None
    assert xc.resolve_gga_core_radius(
        "pbe", nuclear_charge=6.0, core_zr=0.05, r=radius
    ) == pytest.approx(0.05 / 6.0)

    with pytest.raises(ValueError, match="at least 8"):
        xc.resolve_gga_core_radius(
            "pbe",
            nuclear_charge=6.0,
            core_zr=1.0e-5,
            r=radius,
        )


def test_missing_pylibxc_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        raise ModuleNotFoundError("No module named pylibxc", name=name)

    xc._make_libxc_functional.cache_clear()
    monkeypatch.setattr(xc, "import_module", missing)
    try:
        with pytest.raises(xc.LibXCUnavailableError, match="conda-forge pylibxc"):
            xc.xc_potential(np.array([0.1]), model="libxc:lda_x")
    finally:
        xc._make_libxc_functional.cache_clear()


@pytest.mark.skipif(
    importlib.util.find_spec("pylibxc") is None,
    reason="optional pylibxc bindings are not installed",
)
def test_real_libxc_lda_x_matches_builtin_dirac() -> None:
    density = np.geomspace(1.0e-8, 10.0, 64)
    actual = xc.xc_potential(density, model="libxc:lda_x")
    expected = xc.dirac_exchange_potential(density)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=1.0e-14)


@pytest.mark.skipif(
    importlib.util.find_spec("pylibxc") is None,
    reason="optional pylibxc bindings are not installed",
)
def test_real_pbe_alias_matches_explicit_components() -> None:
    radius = np.geomspace(1.0e-3, 8.0, 256)
    density = 0.01 + 0.2 * np.exp(-radius)

    alias = xc.xc_potential(density, model="pbe", r=radius)
    explicit = xc.xc_potential(
        density,
        model="libxc:gga_x_pbe+gga_c_pbe",
        r=radius,
    )

    assert np.all(np.isfinite(alias))
    np.testing.assert_allclose(alias, explicit, rtol=2.0e-13, atol=1.0e-13)


def test_full_external_config_forwards_xc_model_to_ks_solver() -> None:
    cfg = FullExternalConfig(
        element="C",
        temperature_ev=100.0,
        rho_g_cc=2.0,
        xc_model="pbe",
        gga_core_mode="finite",
        gga_core_zr=0.04,
    )
    ks_cfg = _build_ks_config(
        cfg,
        z_nuc=6,
        temperature_ha=100.0 / 27.211386245988,
        n_i=0.01,
        r_ws=2.0,
        rmax=20.0,
        mu_guess=0.1,
        mu_bounds=(-5.0, 5.0),
        max_iter=4,
        cont_params={"tail_mode": "off"},
        compute_external=True,
    )

    assert ks_cfg.xc_model == "pbe"
    assert ks_cfg.gga_core_mode == "finite"
    assert ks_cfg.gga_core_zr == pytest.approx(0.04)


def test_plasma_workflow_forwards_xc_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflows,
        "solve_full_then_external",
        lambda cfg: {
            "xc_model": cfg.xc_model,
            "gga_core_mode": cfg.gga_core_mode,
            "gga_core_zr": cfg.gga_core_zr,
        },
    )
    cfg = workflows.PlasmaWorkflowConfig(
        elements=["C"],
        temperature_ev=100.0,
        rho_g_cc=2.0,
        xc_model="pbe",
        gga_core_mode="finite",
        gga_core_zr=0.04,
    )

    kind, result = workflows._solve_electronic_structure(
        cfg,
        symbols=["C"],
        counts=[1.0],
    )

    assert kind == "single_species"
    assert result["xc_model"] == "pbe"
    assert result["gga_core_mode"] == "finite"
    assert result["gga_core_zr"] == pytest.approx(0.04)


def test_plasma_workflow_rejects_conflicting_xc_override() -> None:
    cfg = workflows.PlasmaWorkflowConfig(
        elements=["C"],
        temperature_ev=100.0,
        rho_g_cc=2.0,
        xc_model="dirac",
        aa_overrides={"xc_model": "pbe"},
    )

    with pytest.raises(ValueError, match="Conflicting xc_model"):
        workflows._solve_electronic_structure(
            cfg,
            symbols=["C"],
            counts=[1.0],
        )


def test_plasma_workflow_rejects_conflicting_gga_core_override() -> None:
    cfg = workflows.PlasmaWorkflowConfig(
        elements=["C"],
        temperature_ev=100.0,
        rho_g_cc=2.0,
        xc_model="pbe",
        gga_core_mode="finite",
        aa_overrides={"gga_core_mode": "strict"},
    )

    with pytest.raises(ValueError, match="Conflicting gga_core_mode"):
        workflows._solve_electronic_structure(
            cfg,
            symbols=["C"],
            counts=[1.0],
        )


def test_public_configs_validate_gga_core_controls() -> None:
    with pytest.raises(ValueError, match="gga_core_mode"):
        FullExternalConfig(
            element="C",
            temperature_ev=2.0,
            rho_g_cc=1.0,
            gga_core_mode="clip",
        )
    with pytest.raises(ValueError, match="gga_core_zr"):
        workflows.PlasmaWorkflowConfig(
            elements=["C"],
            temperature_ev=2.0,
            rho_g_cc=1.0,
            gga_core_zr=0.0,
        )
