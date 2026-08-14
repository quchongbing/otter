"""Carbon XC comparison over density and temperature.

Run a QM average-atom plus QOZ/HNC calculation for carbon on a small state
grid and compare Dirac exchange, LDA-PW correlation, and Libxc PBE::

    poetry run python tools/studies/carbon_xc_comparison.py

The default grid is ``rho=[1, 3, 5] g/cc`` and ``Te=Ti=[2, 15, 50] eV``.
Each XC model gets its own calculation; no electronic result is reused across
models.  Results are written to ``outputs/carbon_xc_comparison`` (or the path
given with ``--output-dir``):

* ``summary.csv`` contains ``mu``, ``Zbar``, bound levels, and scalar
  diagnostics from ``g_ii`` and ``S_ii``;
* one ``.npz`` archive per state contains radial electron profiles, bound
  energies/occupations, ``g_ii(r)``, and ``S_ii(k)``;
* matching PNG/PDF summary, bound-level, profile, and nuclear-core figures
  compare the requested XC models.
* ``xc_provenance.json`` and ``CITATIONS.md`` record the actual Libxc version,
  exact functional IDs, and the references returned by Libxc.

The PBE calculation uses the ground-state PBE energy-density/potential
interface exposed by Libxc and Otter's default finite GGA core regularization.
The regularization is an Otter numerical method, not part of PBE or Libxc. Each
archive also stores the strict potential evaluated on the final density for a
non-self-consistent nuclear-core diagnostic.  This remains an XC-model
sensitivity study, not a finite-temperature XC free-energy benchmark.

References
----------
Libxc: S. Lehtola et al., SoftwareX 7, 1--5 (2018),
doi:10.1016/j.softx.2017.11.002.
PBE: J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. 77,
3865 (1996), doi:10.1103/PhysRevLett.77.3865, and erratum
doi:10.1103/PhysRevLett.78.1396.
PW92 correlation: J. P. Perdew and Y. Wang, Phys. Rev. B 45, 13244 (1992),
doi:10.1103/PhysRevB.45.13244.
Dirac exchange: P. A. M. Dirac, Math. Proc. Cambridge Philos. Soc. 26,
376 (1930), doi:10.1017/S0305004100016108.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from otter import (
    PlasmaWorkflowConfig,
    continue_plasma_workflow_from_electronic_result,
)
from otter.workflows import (
    _solve_electronic_structure,
    resolve_plasma_composition,
)
from otter.electronic.xc import (
    LIBXC_CITATION_GUIDANCE_URL,
    xc_potential,
    xc_provenance,
)
from otter.plotting import grid_figsize, save_figure, set_style


HARTREE_TO_EV = 27.211386245988
DEFAULT_XC_MODELS = ("dirac", "lda_pw", "pbe")
DEFAULT_RHO = (1.0, 3.0, 5.0)
DEFAULT_TEMPERATURE = (2.0, 15.0, 50.0)
DEFAULT_GII_R_MAX_BOHR = 10.0
DEFAULT_SII_K_MAX_BOHR_INV = 10.0

set_style("docs", palette="deep_science")


def _safe_tag(value: float) -> str:
    """Make a stable filename token for a floating-point state value."""
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _unique_strings(values: list[str]) -> list[str]:
    """Deduplicate citation fields without changing Libxc's order."""
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _write_xc_provenance(
    output_dir: Path,
    xc_models: list[str] | tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Write machine- and human-readable XC citation manifests."""
    records = {
        str(model): xc_provenance(str(model))
        for model in dict.fromkeys(str(value) for value in xc_models)
    }
    payload = {
        "schema": "otter-xc-provenance-v1",
        "libxc_citation_guidance": LIBXC_CITATION_GUIDANCE_URL,
        "models": records,
        "otter_core_regularization": (
            "The finite GGA core is an Otter numerical regularization; "
            "it is not part of PBE or Libxc. See gga_core_* run metadata."
        ),
    }
    (output_dir / "xc_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# XC citations",
        "",
        "This file records the XC implementation used by this comparison. ",
        "The finite GGA core is an Otter numerical regularization and is not ",
        "part of PBE or Libxc.",
        "",
    ]
    software_seen: set[tuple[str, str]] = set()
    for model, record in records.items():
        provider = str(record["provider"])
        version = record.get("provider_version")
        lines.extend(
            [
                f"## `{model}`",
                "",
                f"Provider: {provider}"
                + (f" {version}" if version is not None else "")
                + ".",
                "",
            ]
        )
        software_reference = record.get("software_reference")
        software_key = (provider, str(software_reference))
        if software_reference and software_key not in software_seen:
            lines.extend([f"Software citation: {software_reference}", ""])
            software_seen.add(software_key)
        for component in record.get("components", []):
            number = component.get("number")
            identifier = str(component["id"])
            id_text = identifier + (f" (id={number})" if number is not None else "")
            lines.extend([f"Functional: {id_text}, {component['name']}.", ""])
            for reference in component.get("references", []):
                lines.append(f"- {reference}")
            for doi in component.get("dois", []):
                lines.append(f"- DOI: https://doi.org/{doi}")
            if component.get("references"):
                lines.append("")
    (output_dir / "CITATIONS.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return records


def _print_xc_provenance(records: dict[str, dict[str, Any]]) -> None:
    """Print the concise provenance notice requested by Libxc."""
    print("XC provenance (full references: CITATIONS.md):", flush=True)
    for model, record in records.items():
        provider = str(record["provider"])
        version = record.get("provider_version")
        components = []
        dois: list[str] = []
        for component in record.get("components", []):
            number = component.get("number")
            components.append(
                str(component["id"])
                + (f"(id={number})" if number is not None else "")
            )
            dois.extend(str(value) for value in component.get("dois", []))
        provider_text = provider + (f" {version}" if version is not None else "")
        print(
            f"  {model}: {provider_text}; components={'+'.join(components) or 'none'}",
            flush=True,
        )
        software_doi = record.get("software_doi")
        if software_doi:
            print(f"    Libxc DOI: {software_doi}", flush=True)
        if dois:
            print(
                f"    XC reference DOI(s): {'; '.join(_unique_strings(dois))}",
                flush=True,
            )


def _bound_level_records(electronic: dict[str, Any]) -> list[dict[str, float]]:
    """Return finite negative bound levels and their FD occupations."""
    energies = np.asarray(electronic.get("bound_energy_ha", []), dtype=float)
    l_values = np.asarray(electronic.get("bound_l_list", []), dtype=int)
    fd = np.asarray(electronic.get("bound_fd", np.full_like(energies, np.nan)))
    occupation = np.asarray(
        electronic.get("bound_occ_deg_fd", np.full_like(energies, np.nan))
    )
    if energies.ndim != 2 or not l_values.size:
        return []
    records: list[dict[str, float]] = []
    for l_index, l_value in enumerate(l_values):
        for n_index in range(energies.shape[1]):
            energy = float(energies[l_index, n_index])
            if np.isfinite(energy) and energy < 0.0:
                records.append(
                    {
                        "l": float(l_value),
                        "n": float(n_index + 1),
                        "energy_ha": energy,
                        "energy_ev": energy * HARTREE_TO_EV,
                        "fd": float(fd[l_index, n_index]),
                        "occupation": float(occupation[l_index, n_index]),
                    }
                )
    return records


def _profile_value(x: np.ndarray, y: np.ndarray, target: float) -> float:
    return float(y[int(np.argmin(np.abs(x - target)))])


def _scalar_diagnostics(
    workflow: dict[str, Any],
    *,
    xc_provenance_record: dict[str, Any],
    ion_error: str = "",
) -> dict[str, Any]:
    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"] or {})
    levels = _bound_level_records(electronic)
    g = np.asarray(ion.get("gii_r", []), dtype=float)
    r_ion = np.asarray(ion.get("r", []), dtype=float)
    s = np.asarray(ion.get("sii_k", []), dtype=float)
    k = np.asarray(ion.get("k", []), dtype=float)
    r_ws = float(electronic["r_ws"])
    g_peak = int(np.argmax(g)) if g.size else -1
    s_peak = int(np.argmax(s)) if s.size else -1
    components = list(xc_provenance_record.get("components", []))
    functional_dois = _unique_strings(
        [
            str(doi)
            for component in components
            for doi in component.get("dois", [])
        ]
    )
    return {
        "mu_ha": float(electronic["mu"]),
        "mu_ev": float(electronic["mu"]) * HARTREE_TO_EV,
        "zbar": float(electronic["zbar"]),
        "zbar_partition": float(electronic.get("zbar_partition", np.nan)),
        "r_ws_bohr": r_ws,
        "n0_bohr3": float(electronic["n0"]),
        "n_bound_levels": len(levels),
        "bound_levels_json": json.dumps(levels, separators=(",", ":")),
        "stage2_converged": bool(electronic.get("stage2_converged", False)),
        "external_converged": bool(
            dict(electronic.get("ext_status", {})).get("converged", False)
        ),
        "xc_provider": str(xc_provenance_record.get("provider", "unknown")),
        "libxc_version": str(
            xc_provenance_record.get("provider_version") or "not_applicable"
        ),
        "xc_functional_ids": "+".join(
            str(component["id"]) for component in components
        ),
        "xc_functional_numbers": "+".join(
            str(component["number"])
            for component in components
            if component.get("number") is not None
        ),
        "xc_software_doi": str(
            xc_provenance_record.get("software_doi") or "not_applicable"
        ),
        "xc_functional_dois": ";".join(functional_dois),
        "gga_core_mode": str(electronic.get("gga_core_mode", "not_applicable")),
        "gga_core_zr": float(electronic.get("gga_core_zr", np.nan)),
        "gga_core_radius_bohr": float(
            electronic.get("gga_core_radius_bohr", np.nan)
        ),
        "gga_core_points": int(electronic.get("gga_core_points", 0)),
        "density_cusp_rel_error": float(
            electronic.get("density_cusp_rel_error", np.nan)
        ),
        "v_xc_core_turn_count": int(
            electronic.get("v_xc_core_turn_count", 0)
        ),
        "v_xc_core_max_abs_ev": float(
            electronic.get("v_xc_core_max_abs_ha", np.nan)
        )
        * HARTREE_TO_EV,
        "ion_status": "converged" if ion else "failed",
        "ion_error": str(ion_error),
        "hnc_converged": bool(ion.get("hnc_converged", False)),
        "hnc_residual": float(ion.get("hnc_output_residual", np.nan)),
        "gii_at_rws": (
            _profile_value(r_ion, g, r_ws) if g.size else float("nan")
        ),
        "gii_peak": float(g[g_peak]) if g.size else float("nan"),
        "gii_peak_r_over_rws": (
            float(r_ion[g_peak] / r_ws) if g.size else float("nan")
        ),
        "sii_peak": float(s[s_peak]) if s.size else float("nan"),
        "sii_peak_k_bohr_inv": float(k[s_peak]) if s.size else float("nan"),
    }


def _save_state(
    output_dir: Path,
    *,
    rho: float,
    temperature: float,
    xc_model: str,
    xc_provenance_record: dict[str, Any],
    workflow: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Path:
    """Save the profiles needed for later plotting without rerunning Otter."""
    electronic = dict(workflow["electronic"]["result"])
    ion = dict(workflow["ion"] or {})
    r_electronic = np.asarray(electronic["r"], dtype=float)
    n_xc_source = np.asarray(
        electronic.get("n_full_source", electronic["n_full"]),
        dtype=float,
    )
    n0_profile = np.full_like(r_electronic, float(electronic["n0"]))
    v_xc_strict = xc_potential(
        n_xc_source,
        model=xc_model,
        r=r_electronic,
    ) - xc_potential(
        n0_profile,
        model=xc_model,
        r=r_electronic,
    )
    arrays: dict[str, np.ndarray] = {
        "rho_g_cc": np.asarray(rho),
        "temperature_ev": np.asarray(temperature),
        "ion_temperature_ev": np.asarray(temperature),
        "xc_model": np.asarray(xc_model),
        "xc_provenance_json": np.asarray(
            json.dumps(xc_provenance_record, sort_keys=True)
        ),
        "r_bohr": np.asarray(electronic["r"], dtype=float),
        "n_full_bohr3": np.asarray(electronic["n_full"], dtype=float),
        "n_ext_bohr3": np.asarray(electronic["n_ext"], dtype=float),
        "n_bound_bohr3": np.asarray(electronic["n_bound"], dtype=float),
        "v_xc_ha": np.asarray(electronic["v_xc"], dtype=float),
        "v_xc_strict_ha": np.asarray(v_xc_strict, dtype=float),
        "gga_core_mode": np.asarray(
            electronic.get("gga_core_mode", "not_applicable")
        ),
        "gga_core_zr": np.asarray(electronic.get("gga_core_zr", np.nan)),
        "gga_core_radius_bohr": np.asarray(
            electronic.get("gga_core_radius_bohr", np.nan)
        ),
        "bound_energy_ha": np.asarray(electronic["bound_energy_ha"], dtype=float),
        "bound_l_list": np.asarray(electronic["bound_l_list"], dtype=int),
        "bound_fd": np.asarray(electronic["bound_fd"], dtype=float),
        "bound_occ_deg_fd": np.asarray(
            electronic["bound_occ_deg_fd"], dtype=float
        ),
        "r_ion_bohr": np.asarray(ion.get("r", []), dtype=float),
        "gii_r": np.asarray(ion.get("gii_r", []), dtype=float),
        "k_bohr_inv": np.asarray(ion.get("k", []), dtype=float),
        "sii_k": np.asarray(ion.get("sii_k", []), dtype=float),
    }
    arrays["diagnostics_json"] = np.asarray(
        json.dumps(diagnostics, sort_keys=True)
    )
    filename = (
        f"C_rho{_safe_tag(rho)}_Te{_safe_tag(temperature)}_"
        f"xc-{xc_model.replace(':', '_').replace('+', '_')}.npz"
    )
    path = output_dir / filename
    np.savez_compressed(path, **arrays)
    return path


def _plot_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot scalar electronic and ionic diagnostics for the full scan."""
    if not rows:
        return
    models = list(dict.fromkeys(str(row["xc_model"]) for row in rows))
    densities = sorted({float(row["rho_g_cc"]) for row in rows})
    fig, axes = plt.subplots(
        2,
        2,
        figsize=grid_figsize(2, 2),
        constrained_layout=True,
    )
    panels = (
        ("zbar", "Zbar", "Average ionization"),
        ("mu_ev", "mu (eV)", "Chemical potential"),
        ("gii_peak", "max gii", "Ion pair-correlation peak"),
        ("sii_peak", "max Sii", "Ion structure-factor peak"),
    )
    for ax, (field, ylabel, title) in zip(axes.flat, panels):
        for rho in densities:
            for model in models:
                selected = [
                    row
                    for row in rows
                    if float(row["rho_g_cc"]) == rho
                    and str(row["xc_model"]) == model
                ]
                selected.sort(key=lambda row: float(row["temperature_ev"]))
                if not selected:
                    continue
                label = f"rho={rho:g}, {model}"
                ax.plot(
                    [float(row["temperature_ev"]) for row in selected],
                    [float(row[field]) for row in selected],
                    marker="o",
                    label=label,
                )
        ax.set_xlabel("Te = Ti (eV)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    save_figure(fig, output_dir / "summary", dpi=180)
    plt.close(fig)

def _write_bound_level_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Write one long-form row per bound level for cross-state analysis."""
    records: list[dict[str, Any]] = []
    for row in rows:
        levels = json.loads(str(row.get("bound_levels_json", "[]")))
        for level in levels:
            records.append(
                {
                    "rho_g_cc": row["rho_g_cc"],
                    "temperature_ev": row["temperature_ev"],
                    "xc_model": row["xc_model"],
                    **level,
                }
            )
    if not records:
        return
    fields = [
        "rho_g_cc", "temperature_ev", "xc_model", "n", "l",
        "energy_ha", "energy_ev", "fd", "occupation",
    ]
    with (output_dir / "bound_levels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _plot_bound_levels(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot each bound orbital versus temperature for every density and XC."""
    densities = sorted({float(row["rho_g_cc"]) for row in rows})
    level_keys = sorted(
        {
            (int(level["n"]), int(level["l"]))
            for row in rows
            for level in json.loads(str(row.get("bound_levels_json", "[]")))
        }
    )
    models = list(dict.fromkeys(str(row["xc_model"]) for row in rows))
    if not densities or not level_keys:
        return
    nrows = len(densities)
    ncols = len(level_keys)
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=grid_figsize(nrows, ncols),
        constrained_layout=True,
    )
    colors = dict(zip(models, plt.cm.tab10.colors[: len(models)]))
    for row_index, rho in enumerate(densities):
        for col_index, (n_level, l_value) in enumerate(level_keys):
            ax = axes[row_index, col_index]
            for model in models:
                selected = [
                    row for row in rows
                    if float(row["rho_g_cc"]) == rho
                    and str(row["xc_model"]) == model
                ]
                selected.sort(key=lambda row: float(row["temperature_ev"]))
                temperatures: list[float] = []
                energies: list[float] = []
                for row in selected:
                    levels = json.loads(str(row.get("bound_levels_json", "[]")))
                    match = [
                        level for level in levels
                        if int(level["n"]) == n_level
                        and int(level["l"]) == l_value
                    ]
                    if match:
                        temperatures.append(float(row["temperature_ev"]))
                        energies.append(float(match[0]["energy_ev"]))
                if energies:
                    ax.plot(
                        temperatures, energies, marker="o",
                        color=colors[model], label=model,
                    )
            ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
            ax.set_title(f"rho={rho:g}, n={n_level}, l={l_value}")
            ax.set_xlabel("Te = Ti (eV)")
            ax.set_ylabel("bound energy (eV)")
            ax.grid(alpha=0.25)
            if row_index == 0 and col_index == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Carbon bound levels: XC and temperature comparison")
    save_figure(fig, output_dir / "bound_levels", dpi=180)

    plt.close(fig)

def _plot_profiles(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    rho: float,
    temperature: float,
    gii_r_max_bohr: float,
    sii_k_max_bohr_inv: float,
) -> None:
    """Plot electron density, bound levels, gii, and Sii for one state."""
    selected = [
        row
        for row in rows
        if np.isclose(float(row["rho_g_cc"]), rho)
        and np.isclose(float(row["temperature_ev"]), temperature)
    ]
    if not selected:
        return
    fig, axes = plt.subplots(
        2,
        2,
        figsize=grid_figsize(2, 2),
        constrained_layout=True,
    )
    for row in selected:
        with np.load(output_dir / str(row["archive"]), allow_pickle=False) as data:
            r = np.asarray(data["r_bohr"], dtype=float)
            r_ws = float(row["r_ws_bohr"])
            axes[0, 0].plot(
                r / r_ws,
                np.asarray(data["n_ext_bohr3"]),
                label=str(row["xc_model"]),
            )
            axes[0, 1].plot(
                r / r_ws,
                np.asarray(data["v_xc_ha"]) * HARTREE_TO_EV,
                label=str(row["xc_model"]),
            )
            r_ion = np.asarray(data["r_ion_bohr"])
            g = np.asarray(data["gii_r"])
            k = np.asarray(data["k_bohr_inv"])
            s = np.asarray(data["sii_k"])
            r_mask = r_ion <= float(gii_r_max_bohr)
            k_mask = k <= float(sii_k_max_bohr_inv)
            axes[1, 0].plot(
                r_ion[r_mask], g[r_mask], label=str(row["xc_model"])
            )
            axes[1, 1].plot(
                k[k_mask], s[k_mask], label=str(row["xc_model"])
            )
    axes[0, 0].set(xlabel="r / R_WS", ylabel="n_ext (bohr^-3)", yscale="log")
    axes[0, 1].set(xlabel="radial level index", ylabel="bound energy (eV)")
    axes[1, 0].set(
        xlabel="r (bohr)", ylabel="g_ii(r)", xlim=(0.0, gii_r_max_bohr)
    )
    axes[1, 1].set(
        xlabel="k (bohr^-1)", ylabel="S_ii(k)", xlim=(0.0, sii_k_max_bohr_inv)
    )
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0, 1].set(xlabel="r / R_WS", ylabel="v_xc (eV)")
    name = f"profiles_rho{_safe_tag(rho)}_Te{_safe_tag(temperature)}.png"
    save_figure(fig, output_dir / name, dpi=180)
    plt.close(fig)

    fig_core, ax_core = plt.subplots(
        figsize=grid_figsize(1, 1),
        constrained_layout=True,
    )
    plotted = False
    for row in selected:
        with np.load(output_dir / str(row["archive"]), allow_pickle=False) as data:
            if "gga_core_radius_bohr" not in data.files:
                continue
            core_radius = float(np.asarray(data["gga_core_radius_bohr"]))
            if not np.isfinite(core_radius) or core_radius <= 0.0:
                continue
            r = np.asarray(data["r_bohr"], dtype=float)
            mask = r <= 2.0 * core_radius
            if not np.any(mask):
                continue
            line = ax_core.plot(
                r[mask] / core_radius,
                np.asarray(data["v_xc_ha"])[mask] * HARTREE_TO_EV,
                label=f"{row['xc_model']} finite (SCF)",
            )[0]
            if "v_xc_strict_ha" in data.files:
                ax_core.plot(
                    r[mask] / core_radius,
                    np.asarray(data["v_xc_strict_ha"])[mask] * HARTREE_TO_EV,
                    linestyle="--",
                    color=line.get_color(),
                    alpha=0.75,
                    label=f"{row['xc_model']} strict (diagnostic)",
                )
            plotted = True
    if plotted:
        ax_core.set(
            xlabel=r"$r/r_c$",
            ylabel=r"$V_{xc}$ (eV)",
            xlim=(0.0, 2.0),
            title=f"Carbon XC nuclear core: rho={rho:g}, Te={temperature:g} eV",
        )
        ax_core.set_yscale("symlog", linthresh=10.0)
        ax_core.grid(alpha=0.25)
        ax_core.legend(fontsize=8)
        core_name = (
            f"vxc_core_rho{_safe_tag(rho)}_Te{_safe_tag(temperature)}.png"
        )
        save_figure(fig_core, output_dir / core_name, dpi=180)
    plt.close(fig_core)


def build_config(
    *,
    rho: float,
    temperature: float,
    xc_model: str,
    n_points: int,
    qoz_n_points: int,
    continuum_workers: int,
    hnc_max_iter: int,
    show_progress: bool,
    gga_core_mode: str,
    gga_core_zr: float,
) -> PlasmaWorkflowConfig:
    return PlasmaWorkflowConfig(
        elements=["C"],
        counts=[1],
        temperature_ev=temperature,
        ion_temperature_ev=temperature,
        rho_g_cc=rho,
        xc_model=xc_model,
        gga_core_mode=gga_core_mode,
        gga_core_zr=gga_core_zr,
        aa_overrides={
            "n_points": n_points,
            "cont_n_jobs": continuum_workers,
            "cont_shards": max(1, 2 * continuum_workers),
            "bound_zero_tail_refine": True,
            "bound_zero_tail_max_binding_ha": 1.0e-2,
            "bound_zero_tail_scan_points": 64,
            "bound_zero_tail_edge_rel_tol": 0.1,
            # Track several extra radial and angular channels for the level plots.
            "bound_auto_n_pad": 3,
            "bound_auto_l_pad": 2,
        },
        qoz_linear_n_points=qoz_n_points,
        hnc_closure_transform_tol=1.0e-3,
        hnc_max_iter=hnc_max_iter,
        show_progress=show_progress,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", nargs="+", type=float, default=list(DEFAULT_RHO))
    parser.add_argument(
        "--temperature",
        nargs="+",
        type=float,
        default=list(DEFAULT_TEMPERATURE),
        help="Electron and ion temperatures in eV.",
    )
    parser.add_argument(
        "--xc-models",
        nargs="+",
        default=list(DEFAULT_XC_MODELS),
        help="XC names accepted by PlasmaWorkflowConfig.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/carbon_xc_comparison")
    )
    parser.add_argument(
        "--n-points", type=int, default=2**12,
        help="AA radial points; 4096 is the production default.",
    )
    parser.add_argument(
        "--qoz-n-points", type=int, default=2**12,
        help="QOZ linear-grid points; 4096 is the production default.",
    )
    parser.add_argument(
        "--gii-r-max-bohr", type=float, default=DEFAULT_GII_R_MAX_BOHR,
        help="Maximum plotted g_ii radius in bohr (default: 10).",
    )
    parser.add_argument(
        "--sii-k-max-bohr-inv", type=float, default=DEFAULT_SII_K_MAX_BOHR_INV,
        help="Maximum plotted S_ii wave number in bohr^-1 (default: 10).",
    )
    parser.add_argument("--continuum-workers", type=int, default=1)
    parser.add_argument(
        "--gga-core-mode",
        choices=("finite", "strict"),
        default="finite",
        help="Nuclear-core treatment used by GGA models (default: finite).",
    )
    parser.add_argument(
        "--gga-core-zr",
        type=float,
        default=0.05,
        help="Dimensionless finite-core transition radius Z*r_c.",
    )
    parser.add_argument("--hnc-max-iter", type=int, default=500)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate plots from an existing summary.csv and archives.",
    )
    return parser.parse_args()


def _solve_case(config: PlasmaWorkflowConfig) -> tuple[dict[str, Any], str]:
    """Solve electronic structure, then retain it if ion HNC fails."""
    symbols, counts = resolve_plasma_composition(
        formula=None,
        elements=["C"],
        counts=[1],
        number_fraction=None,
    )
    electronic_kind, electronic_result = _solve_electronic_structure(
        config,
        symbols=symbols,
        counts=counts,
    )
    try:
        workflow = continue_plasma_workflow_from_electronic_result(
            config,
            electronic_kind=electronic_kind,
            electronic_result=electronic_result,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        workflow = {
            "electronic": {
                "kind": electronic_kind,
                "result": electronic_result,
            },
            "ion": None,
        }
        return workflow, message
    return workflow, ""


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    if args.plot_only:
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing {summary_path}.")
        with summary_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        archived_models = list(
            dict.fromkeys(str(row["xc_model"]) for row in rows)
        )
        _write_xc_provenance(output_dir, archived_models)
        _plot_summary(rows, output_dir)
        _write_bound_level_table(rows, output_dir)
        _plot_bound_levels(rows, output_dir)
        for rho in sorted({float(row["rho_g_cc"]) for row in rows}):
            for temperature in sorted(
                {float(row["temperature_ev"]) for row in rows}
            ):
                _plot_profiles(
                    rows,
                    output_dir,
                    rho=rho,
                    temperature=temperature,
                    gii_r_max_bohr=args.gii_r_max_bohr,
                    sii_k_max_bohr_inv=args.sii_k_max_bohr_inv,
                )
        return

    provenance_by_model = _write_xc_provenance(
        output_dir,
        list(args.xc_models),
    )
    _print_xc_provenance(provenance_by_model)
    rows: list[dict[str, Any]] = []
    total = len(args.rho) * len(args.temperature) * len(args.xc_models)
    case_index = 0
    for rho in args.rho:
        for temperature in args.temperature:
            for xc_model in args.xc_models:
                case_index += 1
                print(
                    f"[{case_index}/{total}] C rho={rho:g} Te=Ti={temperature:g} "
                    f"XC={xc_model}",
                    flush=True,
                )
                started = time.perf_counter()
                config = build_config(
                    rho=rho,
                    temperature=temperature,
                    xc_model=xc_model,
                    n_points=args.n_points,
                    qoz_n_points=args.qoz_n_points,
                    continuum_workers=args.continuum_workers,
                    hnc_max_iter=args.hnc_max_iter,
                    show_progress=args.show_progress,
                    gga_core_mode=args.gga_core_mode,
                    gga_core_zr=args.gga_core_zr,
                )
                workflow, ion_error = _solve_case(config)
                provenance_record = provenance_by_model[str(xc_model)]
                diagnostics = _scalar_diagnostics(
                    workflow,
                    xc_provenance_record=provenance_record,
                    ion_error=ion_error,
                )
                archive = _save_state(
                    output_dir,
                    rho=rho,
                    temperature=temperature,
                    xc_model=xc_model,
                    xc_provenance_record=provenance_record,
                    workflow=workflow,
                    diagnostics=diagnostics,
                )
                row: dict[str, Any] = {
                    "element": "C",
                    "rho_g_cc": rho,
                    "temperature_ev": temperature,
                    "ion_temperature_ev": temperature,
                    "xc_model": xc_model,
                    "elapsed_s": time.perf_counter() - started,
                    "archive": archive.name,
                    **diagnostics,
                }
                rows.append(row)
                print(
                    f"  Zbar={row['zbar']:.6g} mu={row['mu_ev']:.6g} eV "
                    f"g_peak={row['gii_peak']:.6g} S_peak={row['sii_peak']:.6g} "
                    f"({row['elapsed_s']:.1f}s)",
                    flush=True,
                )

    fields = list(rows[0])
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _plot_summary(rows, output_dir)
    _write_bound_level_table(rows, output_dir)
    _plot_bound_levels(rows, output_dir)
    for rho in args.rho:
        for temperature in args.temperature:
            _plot_profiles(
                rows,
                output_dir,
                rho=rho,
                temperature=temperature,
                gii_r_max_bohr=args.gii_r_max_bohr,
                sii_k_max_bohr_inv=args.sii_k_max_bohr_inv,
            )
    print(f"Wrote {summary_path} and {len(rows)} state archives.")


if __name__ == "__main__":
    main()
