"""Exchange-correlation models for spherical electronic-structure solvers.

The historical Dirac model is implemented directly and remains the
dependency-free default. LDA and GGA functionals from Libxc are available
when its Python bindings (pylibxc) are installed. GGA potentials include the
radial functional-derivative term, rather than returning Libxc's partial
derivative with respect to density alone.

References
----------
Libxc should be cited as S. Lehtola, C. Steigemann, M. J. T. Oliveira, and
M. A. L. Marques, SoftwareX 7, 1--5 (2018),
DOI 10.1016/j.softx.2017.11.002.  Libxc also supplies the primary references
for every functional through its runtime API; :func:`xc_provenance` preserves
those references and the exact Libxc version/functional IDs in Otter results.

The PBE alias follows J. P. Perdew, K. Burke, and M. Ernzerhof, Physical
Review Letters 77, 3865 (1996), DOI 10.1103/PhysRevLett.77.3865, including
its erratum, DOI 10.1103/PhysRevLett.78.1396.  The finite-core switch used by
Otter is an explicitly documented Otter numerical regularization; it is not
part of the cited PBE functional or of Libxc.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any

import numpy as np


# These are exact Libxc API identifiers, not informal functional labels.
# Primary references are queried from Libxc by ``xc_provenance`` so explicit
# ``libxc:...`` combinations receive the same attribution as built-in aliases.
_LIBXC_ALIASES: dict[str, tuple[str, ...]] = {
    "pbe": ("gga_x_pbe", "gga_c_pbe"),
    "lda_pw": ("lda_x", "lda_c_pw"),
    "lda_pz": ("lda_x", "lda_c_pz"),
    "lda_vwn": ("lda_x", "lda_c_vwn"),
}

LIBXC_SOFTWARE_DOI = "10.1016/j.softx.2017.11.002"
LIBXC_SOFTWARE_REFERENCE = (
    "S. Lehtola, C. Steigemann, M. J. T. Oliveira, and M. A. L. Marques, "
    "Recent developments in Libxc: A comprehensive library of functionals "
    "for density functional theory, SoftwareX 7, 1-5 (2018), "
    f"doi:{LIBXC_SOFTWARE_DOI}."
)
LIBXC_CITATION_GUIDANCE_URL = "https://libxc.gitlab.io/#citing-libxc"

_DIRAC_REFERENCE = (
    "P. A. M. Dirac, Note on Exchange Phenomena in the Thomas Atom, "
    "Math. Proc. Cambridge Philos. Soc. 26, 376 (1930), "
    "doi:10.1017/S0305004100016108."
)
_DIRAC_DOI = "10.1017/S0305004100016108"


class LibXCUnavailableError(ImportError):
    """Raised when a requested Libxc model cannot import pylibxc."""


def _clean_density(n: np.ndarray) -> np.ndarray:
    density = np.asarray(n, dtype=float)
    density = np.where(np.isfinite(density), density, 0.0)
    return np.clip(density, 0.0, None)


def dirac_exchange_energy_density(n: np.ndarray) -> np.ndarray:
    """Return the spin-unpolarized Dirac exchange energy per volume.

    Parameters
    ----------
    n
        Electron density in inverse cubic Bohr.

    Returns
    -------
    ndarray
        Exchange energy density in Ha per cubic Bohr.
    """
    density = _clean_density(n)
    prefactor = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    return prefactor * np.power(density, 4.0 / 3.0)


def dirac_exchange_potential(n: np.ndarray) -> np.ndarray:
    """Return the spin-unpolarized Dirac exchange potential in Hartree."""
    density = _clean_density(n)
    prefactor = -((3.0 / np.pi) ** (1.0 / 3.0))
    return prefactor * np.power(density, 1.0 / 3.0)


def _libxc_components(model: str) -> tuple[str, ...] | None:
    key = str(model).strip().lower()
    if key in {"dirac", "none"}:
        return None
    if key in _LIBXC_ALIASES:
        return _LIBXC_ALIASES[key]
    if not key.startswith("libxc:"):
        choices = ", ".join(("dirac", "none", *_LIBXC_ALIASES))
        raise ValueError(
            f"Unknown XC model {model!r}. Use one of {choices}, or "
            "'libxc:<functional>[+<functional>...]'."
        )
    components = tuple(part.strip() for part in key[6:].split("+") if part.strip())
    if not components:
        raise ValueError("A libxc model must name at least one functional.")
    return components


def _functional_family(name: str) -> str:
    """Return the supported semilocal family encoded in a Libxc name."""
    if name.startswith("lda_"):
        return "lda"
    if name.startswith("gga_"):
        return "gga"
    if name.startswith(("mgga_", "hyb_")):
        raise ValueError(
            f"Libxc functional {name!r} is not supported: Otter currently "
            "provides density and density-gradient inputs, but not the orbital "
            "kinetic-energy density or exact-exchange operator required by "
            "meta-GGA and hybrid functionals."
        )
    raise ValueError(
        f"Cannot determine the Libxc family from {name!r}; use a canonical "
        "LDA or GGA functional name such as 'lda_c_pw' or 'gga_x_pbe'."
    )


def resolve_gga_core_radius(
    model: str,
    *,
    nuclear_charge: float,
    mode: str = "finite",
    core_zr: float = 0.05,
    r: np.ndarray | None = None,
    min_points: int = 8,
) -> float | None:
    """Resolve and validate the finite-core radius for a GGA model."""
    components = _libxc_components(str(model).strip().lower())
    if components is None or not any(
        _functional_family(name) == "gga" for name in components
    ):
        return None
    mode_key = str(mode).strip().lower()
    if mode_key == "strict":
        return None
    if mode_key != "finite":
        raise ValueError("gga_core_mode must be 'finite' or 'strict'.")
    charge = float(nuclear_charge)
    zr = float(core_zr)
    if not np.isfinite(charge) or charge <= 0.0:
        raise ValueError("A finite GGA core requires a positive nuclear charge.")
    if not np.isfinite(zr) or zr <= 0.0:
        raise ValueError("gga_core_zr must be finite and positive.")
    core_radius = zr / charge
    if r is not None:
        radius = np.asarray(r, dtype=float)
        points = int(np.count_nonzero(radius <= core_radius))
        if points < int(min_points):
            raise ValueError(
                "The finite GGA core contains "
                f"{points} radial points, but at least {int(min_points)} are "
                "required. Increase n_points, reduce rmin, select a larger "
                "gga_core_zr, or use gga_core_mode='strict'."
            )
    return core_radius


def _load_pylibxc() -> Any:
    try:
        return import_module("pylibxc")
    except ModuleNotFoundError as exc:
        if exc.name != "pylibxc":
            raise
        raise LibXCUnavailableError(
            "XC model requires the optional Libxc Python bindings. Install "
            "them with 'poetry install --extras libxc'. This source build "
            "requires CMake and a C compiler; see "
            "https://libxc.gitlab.io/installation/. Otherwise select the "
            "built-in xc_model='dirac'."
        ) from exc


@lru_cache(maxsize=None)
def _make_libxc_functional(name: str) -> Any:
    pylibxc = _load_pylibxc()
    try:
        return pylibxc.LibXCFunctional(name, "unpolarized")
    except KeyError as exc:
        raise ValueError(f"Unknown Libxc functional: {name!r}.") from exc


def _metadata_sequence(value: Any) -> list[str]:
    """Normalize one optional Libxc metadata field to JSON-safe strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _functional_metadata(functional: Any, functional_id: str) -> dict[str, Any]:
    """Read citation metadata through the public pylibxc functional API."""

    def optional_call(method_name: str, default: Any) -> Any:
        method = getattr(functional, method_name, None)
        if not callable(method):
            return default
        try:
            return method()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return default

    number = optional_call("get_number", None)
    try:
        number = int(number) if number is not None else None
    except (TypeError, ValueError):
        number = None
    return {
        "id": str(functional_id),
        "number": number,
        "name": str(optional_call("get_name", functional_id)),
        "references": _metadata_sequence(
            optional_call("get_references", [])
        ),
        "dois": _metadata_sequence(optional_call("get_doi", [])),
    }


def xc_provenance(model: str = "dirac") -> dict[str, Any]:
    """Return software/version/functional citations for one XC selection.

    The returned dictionary is JSON serializable.  For Libxc-backed models,
    functional names, numeric IDs, references, and DOIs are read from the
    installed Libxc runtime rather than maintained as a second hand-written
    catalogue in Otter.  This follows Libxc's official citation guidance:
    report the used version, cite Libxc itself, and cite every selected
    functional.

    Notes
    -----
    ``gga_core_mode='finite'`` changes how Otter evaluates a GGA at the radial
    origin.  It does not change the identity or citation of the selected
    Libxc functional; that Otter-specific regularization is reported
    separately in solver metadata.
    """
    key = str(model).strip().lower()
    if key == "none":
        return {
            "model": key,
            "provider": "otter_builtin",
            "provider_version": None,
            "software_reference": None,
            "software_doi": None,
            "citation_guidance_url": None,
            "components": [],
        }
    if key == "dirac":
        return {
            "model": key,
            "provider": "otter_builtin",
            "provider_version": None,
            "software_reference": None,
            "software_doi": None,
            "citation_guidance_url": None,
            "components": [
                {
                    "id": "dirac_exchange",
                    "number": None,
                    "name": "Dirac exchange",
                    "references": [_DIRAC_REFERENCE],
                    "dois": [_DIRAC_DOI],
                }
            ],
        }

    components = _libxc_components(key)
    assert components is not None
    pylibxc = _load_pylibxc()
    version = getattr(pylibxc, "__version__", "unknown")
    return {
        "model": key,
        "provider": "libxc",
        "provider_version": str(version),
        "software_reference": LIBXC_SOFTWARE_REFERENCE,
        "software_doi": LIBXC_SOFTWARE_DOI,
        "citation_guidance_url": LIBXC_CITATION_GUIDANCE_URL,
        "components": [
            _functional_metadata(_make_libxc_functional(name), name)
            for name in components
        ],
    }


def _radial_inputs(
    n: np.ndarray,
    r: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    density = _clean_density(n)
    if density.ndim != 1:
        raise ValueError("GGA functionals require a one-dimensional radial density.")
    if r is None:
        raise ValueError("GGA functionals require the radial grid argument r.")
    radius = np.asarray(r, dtype=float)
    if radius.shape != density.shape:
        raise ValueError("The radial grid r must match the density shape.")
    if radius.size < 2:
        raise ValueError("GGA functionals require at least two radial grid points.")
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0.0):
        raise ValueError("The radial grid r must contain finite positive values.")
    if np.any(np.diff(radius) <= 0.0):
        raise ValueError("The radial grid r must be strictly increasing.")
    return density, radius


@dataclass(frozen=True)
class _RadialDerivative:
    """Matrix-free first derivative and its Euclidean transpose."""

    indices: np.ndarray
    coefficients: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        return np.sum(self.coefficients * data[self.indices], axis=1)

    def transpose_apply(self, values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float)
        out = np.zeros(self.indices.shape[0], dtype=float)
        for column in range(self.indices.shape[1]):
            np.add.at(
                out,
                self.indices[:, column],
                self.coefficients[:, column] * data,
            )
        return out


@dataclass(frozen=True)
class _RadialGGAContext:
    """Grid quantities shared by GGA energy and potential evaluation."""

    derivative: _RadialDerivative
    shell_weights: np.ndarray
    raw_gradient: np.ndarray
    core_switch: np.ndarray


def _three_point_derivative_coefficients(
    coordinate: np.ndarray,
    row: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a second-order three-point derivative row."""
    size = coordinate.size
    if row == 0:
        indices = np.array([0, 1, 2], dtype=int)
    elif row == size - 1:
        indices = np.array([size - 3, size - 2, size - 1], dtype=int)
    else:
        indices = np.array([row - 1, row, row + 1], dtype=int)
    nodes = coordinate[indices]
    point = float(coordinate[row])
    coefficients = np.empty(3, dtype=float)
    for j in range(3):
        others = [k for k in range(3) if k != j]
        numerator = 2.0 * point - nodes[others[0]] - nodes[others[1]]
        denominator = (nodes[j] - nodes[others[0]]) * (
            nodes[j] - nodes[others[1]]
        )
        coefficients[j] = numerator / denominator
    return indices, coefficients


def _radial_derivative_operator(r: np.ndarray) -> _RadialDerivative:
    """Build the grid-native radial derivative used by a spherical GGA."""
    radius = np.asarray(r, dtype=float)
    size = radius.size
    if size == 2:
        spacing = float(radius[1] - radius[0])
        indices = np.array([[0, 1], [0, 1]], dtype=int)
        coefficients = np.array(
            [[-1.0 / spacing, 1.0 / spacing]] * 2,
            dtype=float,
        )
        return _RadialDerivative(indices=indices, coefficients=coefficients)

    sqrt_radius = np.sqrt(radius)
    sqrt_spacing = np.diff(sqrt_radius)
    sqrt_uniform = np.allclose(
        sqrt_spacing,
        float(np.mean(sqrt_spacing)),
        rtol=2.0e-10,
        atol=2.0e-13,
    )
    coordinate = sqrt_radius if sqrt_uniform else radius
    chain = 1.0 / (2.0 * sqrt_radius) if sqrt_uniform else np.ones_like(radius)
    indices = np.empty((size, 3), dtype=int)
    coefficients = np.empty((size, 3), dtype=float)
    for row in range(size):
        row_indices, row_coefficients = _three_point_derivative_coefficients(
            coordinate,
            row,
        )
        indices[row] = row_indices
        coefficients[row] = chain[row] * row_coefficients
    return _RadialDerivative(indices=indices, coefficients=coefficients)


def _radial_shell_weights(r: np.ndarray) -> np.ndarray:
    """Return positive spherical control-volume weights without the 4*pi."""
    radius = np.asarray(r, dtype=float)
    faces = np.empty(radius.size + 1, dtype=float)
    faces[0] = 0.0
    faces[1:-1] = 0.5 * (radius[:-1] + radius[1:])
    faces[-1] = radius[-1]
    weights = (faces[1:] ** 3 - faces[:-1] ** 3) / 3.0
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("The radial grid does not define positive shell weights.")
    return weights


def _gga_core_switch(
    r: np.ndarray,
    core_radius_bohr: float | None,
) -> np.ndarray:
    """Return Otter's C2 switch from the finite LDA core to the full GGA.

    This switch is an Otter numerical regularization, not a term in PBE and
    not behavior supplied by Libxc.  The selected Libxc functional and this
    radial-core treatment are consequently recorded as separate provenance
    fields.
    """
    if core_radius_bohr is None:
        return np.ones_like(r, dtype=float)
    core_radius = float(core_radius_bohr)
    if not np.isfinite(core_radius) or core_radius <= 0.0:
        raise ValueError("gga_core_radius_bohr must be finite and positive.")
    scaled = np.clip(np.asarray(r, dtype=float) / core_radius, 0.0, 1.0)
    return scaled**3 * (10.0 - 15.0 * scaled + 6.0 * scaled**2)


def radial_core_diagnostics(
    r: np.ndarray,
    density: np.ndarray,
    potential: np.ndarray,
    *,
    nuclear_charge: float,
    core_radius_bohr: float | None,
) -> dict[str, float | int]:
    """Measure density-cusp error and point-to-point core turning."""
    radius = np.asarray(r, dtype=float)
    density_arr = np.asarray(density, dtype=float)
    potential_arr = np.asarray(potential, dtype=float)
    if radius.shape != density_arr.shape or radius.shape != potential_arr.shape:
        raise ValueError("Core diagnostic arrays must have identical shapes.")
    if core_radius_bohr is None:
        return {
            "core_points": 0,
            "density_cusp_rel_error": float("nan"),
            "potential_turn_count": 0,
            "max_abs_potential_ha": float("nan"),
        }
    core_radius = float(core_radius_bohr)
    core_mask = radius <= core_radius
    fit_mask = radius <= 0.5 * core_radius
    if np.count_nonzero(fit_mask) < 3:
        fit_mask = core_mask
    fit_r = radius[fit_mask]
    fit_n = density_arr[fit_mask]
    cusp_error = float("nan")
    if fit_r.size >= 3 and np.all(np.isfinite(fit_n)):
        degree = min(3, int(fit_r.size) - 1)
        coefficients = np.polyfit(fit_r, fit_n, deg=degree)
        slope = float(coefficients[-2])
        intercept = float(coefficients[-1])
        target = -2.0 * float(nuclear_charge) * intercept
        cusp_error = float(abs(slope - target) / max(abs(target), 1.0e-30))
    core_v = potential_arr[core_mask]
    differences = np.diff(core_v)
    scale = max(float(np.max(np.abs(core_v))) if core_v.size else 0.0, 1.0)
    significant = differences[np.abs(differences) > 1.0e-12 * scale]
    turn_count = 0
    if significant.size >= 2:
        turn_count = int(np.count_nonzero(np.diff(np.signbit(significant))))
    return {
        "core_points": int(np.count_nonzero(core_mask)),
        "density_cusp_rel_error": cusp_error,
        "potential_turn_count": turn_count,
        "max_abs_potential_ha": (
            float(np.max(np.abs(core_v))) if core_v.size else float("nan")
        ),
    }


def _extract_output(
    output: dict[str, Any],
    key: str,
    size: int,
    functional_name: str,
) -> np.ndarray:
    if key not in output:
        raise RuntimeError(
            f"Libxc functional {functional_name!r} did not return {key!r}."
        )
    values = np.asarray(output[key], dtype=float).reshape(-1)
    if values.size != size:
        raise RuntimeError(
            f"Libxc functional {functional_name!r} returned {values.size} "
            f"values for {size} density points."
        )
    if np.any(~np.isfinite(values)):
        raise RuntimeError(
            f"Libxc functional {functional_name!r} returned non-finite {key}."
        )
    return values


def _evaluate_libxc(
    density: np.ndarray,
    components: tuple[str, ...],
    *,
    r: np.ndarray | None,
    gga_core_radius_bohr: float | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    _RadialGGAContext | None,
]:
    """Return total zk, vrho, vsigma, and the radial GGA context."""
    families = tuple(_functional_family(name) for name in components)
    has_gga = "gga" in families
    if has_gga:
        density, radius = _radial_inputs(density, r)
        derivative = _radial_derivative_operator(radius)
        raw_gradient = derivative.apply(density)
        core_switch = _gga_core_switch(radius, gga_core_radius_bohr)
        sigma = (core_switch * raw_gradient) ** 2
        gga_context = _RadialGGAContext(
            derivative=derivative,
            shell_weights=_radial_shell_weights(radius),
            raw_gradient=raw_gradient,
            core_switch=core_switch,
        )
    else:
        density = _clean_density(density)
        sigma = None
        gga_context = None

    shape = density.shape
    density_flat = density.reshape(-1)
    positive = density_flat > 0.0
    zk = np.zeros_like(density_flat)
    vrho = np.zeros_like(density_flat)
    vsigma = np.zeros_like(density_flat) if has_gga else None

    if np.any(positive):
        rho_eval = density_flat[positive]
        for name, family in zip(components, families, strict=True):
            inputs: dict[str, np.ndarray] = {"rho": rho_eval}
            if family == "gga":
                assert sigma is not None
                inputs["sigma"] = sigma.reshape(-1)[positive]
            try:
                output = _make_libxc_functional(name).compute(inputs)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Libxc could not evaluate functional {name!r}: {exc}"
                ) from exc
            count = int(np.count_nonzero(positive))
            zk[positive] += _extract_output(output, "zk", count, name)
            vrho[positive] += _extract_output(output, "vrho", count, name)
            if family == "gga":
                assert vsigma is not None
                vsigma[positive] += _extract_output(output, "vsigma", count, name)

    return (
        zk.reshape(shape),
        vrho.reshape(shape),
        None if vsigma is None else vsigma.reshape(shape),
        gga_context,
    )


def xc_energy_density(
    n: np.ndarray,
    model: str = "dirac",
    *,
    r: np.ndarray | None = None,
    gga_core_radius_bohr: float | None = None,
) -> np.ndarray:
    """Return XC energy per volume for a built-in or Libxc model.

    pbe is shorthand for libxc:gga_x_pbe+gga_c_pbe. Other LDA and GGA
    combinations use the explicit libxc: form. A radial grid is required
    whenever any selected component is a GGA.
    """
    key = str(model).strip().lower()
    if key == "dirac":
        return dirac_exchange_energy_density(n)
    if key == "none":
        return np.zeros_like(np.asarray(n, dtype=float))
    components = _libxc_components(key)
    assert components is not None
    density = _clean_density(n)
    zk, _, _, _ = _evaluate_libxc(
        density,
        components,
        r=r,
        gga_core_radius_bohr=gga_core_radius_bohr,
    )
    return density * zk


def xc_potential(
    n: np.ndarray,
    model: str = "dirac",
    *,
    r: np.ndarray | None = None,
    gga_core_radius_bohr: float | None = None,
) -> np.ndarray:
    r"""Return the spin-unpolarized XC potential in Hartree.

    For a spherical GGA, Libxc returns the partial derivatives vrho and
    vsigma of the energy density. Otter evaluates the discrete functional
    derivative of the radial shell-integrated XC energy. With derivative
    matrix ``D`` and shell weights ``W``, this is

    .. math::

       v_{xc} = v_\rho + W^{-1}D^T
       \left[2 W v_\sigma s^2 Dn\right].

    ``s`` is one unless ``gga_core_radius_bohr`` requests the finite-core
    regularization. In that case it is a C2 smoothstep from zero at the
    nucleus to one at the requested radius. The same switch is used in the
    energy through ``sigma = (s Dn)^2``, so the potential remains the exact
    derivative of the regularized discrete functional.  The switch and the
    discrete radial adjoint are Otter numerical methods, not parts of the
    underlying Libxc functional; use :func:`xc_provenance` for the Libxc
    version and primary functional references.
    """
    key = str(model).strip().lower()
    if key == "dirac":
        return dirac_exchange_potential(n)
    if key == "none":
        return np.zeros_like(np.asarray(n, dtype=float))
    components = _libxc_components(key)
    assert components is not None
    density = _clean_density(n)
    _, vrho, vsigma, gga_context = _evaluate_libxc(
        density,
        components,
        r=r,
        gga_core_radius_bohr=gga_core_radius_bohr,
    )
    if vsigma is None:
        return vrho

    assert gga_context is not None
    weighted_flux = (
        2.0
        * gga_context.shell_weights
        * vsigma
        * gga_context.core_switch**2
        * gga_context.raw_gradient
    )
    gradient_term = gga_context.derivative.transpose_apply(weighted_flux)
    return vrho + gradient_term / gga_context.shell_weights


def lda_xc_potential(
    n: np.ndarray,
    model: str = "dirac",
    *,
    r: np.ndarray | None = None,
    gga_core_radius_bohr: float | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper for xc_potential.

    The historical name is retained for callers using the Dirac LDA. It also
    accepts Libxc models; GGA models require r.
    """
    return xc_potential(
        n,
        model=model,
        r=r,
        gga_core_radius_bohr=gga_core_radius_bohr,
    )


__all__ = [
    "LIBXC_CITATION_GUIDANCE_URL",
    "LIBXC_SOFTWARE_DOI",
    "LIBXC_SOFTWARE_REFERENCE",
    "LibXCUnavailableError",
    "dirac_exchange_energy_density",
    "dirac_exchange_potential",
    "lda_xc_potential",
    "radial_core_diagnostics",
    "resolve_gga_core_radius",
    "xc_energy_density",
    "xc_potential",
    "xc_provenance",
]
