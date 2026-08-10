"""Compare zero and local-potential bound/free partitions in a real AA state.

The default carbon state (100 eV, 0.60 g/cc) contains a shallow 3s level. Three
complete, self-consistent full-average-atom calculations differ only in the
electronic partition convention:

* ``zero`` uses the physical asymptotic reference ``E_cut = 0``;
* ``v_frac`` uses ``E_cut = V_eff(0.70 R_max)``;
* ``fixed_m1e3`` is a deliberate stress test with ``E_cut = -1e-3 Ha``;
  it excludes a negative-energy shell and exposes the numerical sensitivity.

The script prints the levels and integrated density changes, then saves NPZ,
PNG, and PDF diagnostics under
``benchmarks/outputs/bound_energy_partition_sensitivity``.  It is deliberately
not a normal pytest because three orbital continuum SCF calculations are too
expensive for routine CI.

The positive-energy continuum remains the Appendix-A3 integral in every run.
Consequently, a deliberately negative fixed cut leaves ``E_cut < E < 0`` in
neither density; it is a sensitivity diagnostic, not a recommended physical
partition.
See Starrett and Saumon, HEDP 10, 35--42 (2014), Appendix A.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from otter.electronic.full_external import (
    FullExternalConfig,
    solve_full_only,
)
from otter.plotting import PALETTES, save_figure, style_context


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
ELEMENT = "C"
TEMPERATURE_EV = 100.0
RHO_G_CC = 0.60
AA_N_POINTS = 2**12
CONTINUUM_WORKERS = 8
RECOMPUTE = False  # Missing cache files are calculated regardless.
R_PLOT_MAX_BOHR = 8.0
SHOW_FIGURES = True

PARTITIONS = {
    "zero": ("zero", 0.0),
    "v_frac_0p70": ("v_frac", 0.70),
    "fixed_m1e3": ("fixed", -1.0e-3),
}

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = (
    ROOT / "benchmarks" / "outputs" / "bound_energy_partition_sensitivity"
)
HARTREE_TO_EV = 27.211386245988
SHELLS = "spdfgh"


def configuration(mode: str, value: float) -> FullExternalConfig:
    """Return one production-resolution full-AA sensitivity configuration."""
    return FullExternalConfig(
        element=ELEMENT,
        temperature_ev=TEMPERATURE_EV,
        rho_g_cc=RHO_G_CC,
        run_mode="full",
        n_points=AA_N_POINTS,
        stage2_max_iter=180,
        cont_n_jobs=CONTINUUM_WORKERS,
        cont_shards=2 * CONTINUUM_WORKERS,
        show_scf_progress=False,
        save_data=False,
        bound_occ_mode="fd",
        bound_rmax_mult=None,
        bound_energy_cut_mode=mode,
        bound_energy_cut=value,
        bound_zero_tail_refine=True,
        bound_zero_tail_max_binding_ha=1.0e-2,
        bound_zero_tail_scan_points=64,
        bound_zero_tail_l_max=1,
        bound_zero_tail_edge_rel_tol=0.1,
        b3_tail_model="full",
    )


def cache_path(label: str) -> Path:
    element = "".join(character for character in ELEMENT if character.isalnum())
    temperature = f"{TEMPERATURE_EV:g}".replace(".", "p")
    density = f"{RHO_G_CC:g}".replace(".", "p")
    return OUTPUT_DIR / (
        f"{element}_Te{temperature}eV_rho{density}_n{AA_N_POINTS}_{label}.npz"
    )


def pack_result(result: dict[str, Any], elapsed_s: float) -> dict[str, np.ndarray]:
    """Retain only portable arrays needed by the comparison."""
    meta = dict(result["meta"])
    history = list(result.get("history", []))
    final_error = (
        float(history[-1].get("err", np.nan)) if history else float("nan")
    )
    return {
        "r_bohr": np.asarray(result["r"], dtype=float),
        "n_bound_bohr3": np.asarray(result["n_bound"], dtype=float),
        "n_free_a3_bohr3": np.asarray(result["n_free"], dtype=float),
        "n_cont_bohr3": np.asarray(result["n_cont"], dtype=float),
        "n_full_bohr3": np.asarray(result["n_full"], dtype=float),
        "v_full_ha": np.asarray(result["v_full"], dtype=float),
        "bound_energy_ha": np.asarray(result["bound_energy_ha"], dtype=float),
        "bound_l_list": np.asarray(result["bound_l_list"], dtype=int),
        "mu_ha": np.asarray(float(result["mu"])),
        "zbar": np.asarray(float(result["zbar"])),
        "r_ws_bohr": np.asarray(float(result["r_ws"])),
        "energy_cut_ha": np.asarray(float(meta["bound_energy_cut_ha"])),
        "stage2_error": np.asarray(final_error),
        "stage2_converged": np.asarray(bool(result["stage2_converged"])),
        "threshold_status": np.asarray(
            str(result.get("threshold_state_status", "none"))
        ),
        "threshold_representation": np.asarray(
            str(result.get("threshold_state_representation", "none"))
        ),
        "elapsed_s": np.asarray(float(elapsed_s)),
        "element": np.asarray(ELEMENT),
        "temperature_ev": np.asarray(float(TEMPERATURE_EV)),
        "rho_g_cc": np.asarray(float(RHO_G_CC)),
        "n_points": np.asarray(int(AA_N_POINTS)),
    }


def solve_or_load(label: str, mode: str, value: float) -> dict[str, np.ndarray]:
    """Calculate one partition, or load its local diagnostic cache."""
    path = cache_path(label)
    if path.is_file() and not RECOMPUTE:
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
        expected = (
            str(payload.get("element", "")) == ELEMENT
            and float(payload.get("temperature_ev", np.nan)) == TEMPERATURE_EV
            and float(payload.get("rho_g_cc", np.nan)) == RHO_G_CC
            and int(payload.get("n_points", -1)) == AA_N_POINTS
        )
        if expected:
            print(f"[cache] {label}: {path}", flush=True)
            return payload
        print(f"[cache] ignoring incompatible file: {path}", flush=True)

    print(f"[compute] {label}: mode={mode}, value={value:g}", flush=True)
    started = time.perf_counter()
    result = solve_full_only(configuration(mode, value))
    if result.get("stage2_converged") is not True:
        raise RuntimeError(f"{label}: full-AA stage 2 did not converge.")
    payload = pack_result(result, time.perf_counter() - started)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return payload


def radial_charge(
    r: np.ndarray,
    density: np.ndarray,
    *,
    r_max: float | None = None,
) -> float:
    finite = np.isfinite(r) & np.isfinite(density)
    if r_max is not None:
        finite &= r <= float(r_max)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    return float(
        4.0 * np.pi * np.trapezoid(r[finite] ** 2 * density[finite], r[finite])
    )


def cumulative_charge(r: np.ndarray, density: np.ndarray) -> np.ndarray:
    integrand = 4.0 * np.pi * r * r * density
    dr = np.diff(r)
    panels = 0.5 * (integrand[1:] + integrand[:-1]) * dr
    return np.concatenate(([0.0], np.cumsum(panels)))


def level_rows(state: dict[str, np.ndarray]) -> list[tuple[str, float, bool]]:
    """Return finite negative box/matched levels and density inclusion flags."""
    energies = np.asarray(state["bound_energy_ha"], dtype=float)
    angular = np.asarray(state["bound_l_list"], dtype=int)
    edge = float(state["energy_cut_ha"])
    rows: list[tuple[str, float, bool]] = []
    for l_index, l_value in enumerate(angular):
        for radial_index, energy in enumerate(energies[l_index]):
            if not np.isfinite(energy):
                continue
            principal = radial_index + 1 + int(l_value)
            shell = f"{principal}{SHELLS[int(l_value)]}"
            rows.append((shell, float(energy), bool(energy < edge)))
    return rows


def print_report(states: dict[str, dict[str, np.ndarray]]) -> None:
    print(
        "\npartition          Ecut[Ha]       mu[Ha]      Zbar   "
        "Qb(all)    Qb(WS)  QfreeA3(WS)  threshold"
    )
    for label, state in states.items():
        r = np.asarray(state["r_bohr"], dtype=float)
        r_ws = float(state["r_ws_bohr"])
        print(
            f"{label:16s} {float(state['energy_cut_ha']): .6e} "
            f"{float(state['mu_ha']): .6e} {float(state['zbar']):7.4f} "
            f"{radial_charge(r, state['n_bound_bohr3']):9.6f} "
            f"{radial_charge(r, state['n_bound_bohr3'], r_max=r_ws):9.6f} "
            f"{radial_charge(r, state['n_free_a3_bohr3'], r_max=r_ws):12.6f} "
            f"{str(state['threshold_status'].item())}"
        )
        for shell, energy, included in level_rows(state):
            print(
                f"    {shell:3s} E={energy: .8e} Ha "
                f"E-Ecut={(energy-float(state['energy_cut_ha'])): .3e} Ha "
                f"included={included}"
            )

    reference = states["zero"]
    r = np.asarray(reference["r_bohr"], dtype=float)
    r_ws = float(reference["r_ws_bohr"])
    for label, state in states.items():
        if label == "zero":
            continue
        print(f"\nself-consistent {label} - zero density changes")
        for key in (
            "n_bound_bohr3",
            "n_free_a3_bohr3",
            "n_cont_bohr3",
            "n_full_bohr3",
        ):
            delta = np.asarray(state[key], float) - np.asarray(reference[key], float)
            l1 = radial_charge(r, np.abs(delta), r_max=r_ws)
            dq = radial_charge(r, delta, r_max=r_ws)
            print(
                f"  {key:18s} L1(WS)={l1:.6e} e   "
                f"signed(WS)={dq:+.6e} e"
            )


def plot_report(states: dict[str, dict[str, np.ndarray]]) -> None:
    colors = PALETTES["bing"]
    styles = {
        "zero": (colors[1], "-"),
        "v_frac_0p70": (colors[2], "--"),
        "fixed_m1e3": (colors[3], "-."),
    }
    zero = states["zero"]
    r = np.asarray(zero["r_bohr"], dtype=float)
    mask = r <= min(float(R_PLOT_MAX_BOHR), float(r[-1]))

    with style_context("thesis", palette="bing"):
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
        components = (
            ("n_bound_bohr3", r"$4\pi r^2n_{\rm bound}$"),
            ("n_free_a3_bohr3", r"$4\pi r^2n_{\rm free}^{A3}$"),
            ("n_full_bohr3", r"$4\pi r^2n_{\rm full}$"),
        )
        for axis, (key, ylabel) in zip(axes.flat[:3], components, strict=True):
            for label, state in states.items():
                color, linestyle = styles[label]
                density = np.asarray(state[key], dtype=float)
                axis.plot(r[mask], 4.0 * np.pi * r[mask] ** 2 * density[mask],
                          color=color, ls=linestyle, label=label)
            axis.set(xlim=(-0.5, r[mask][-1]), xlabel=r"$r$ [Bohr]", ylabel=ylabel)
            axis.legend()

        for label, state in states.items():
            if label == "zero":
                continue
            color, linestyle = styles[label]
            delta_full = np.asarray(state["n_full_bohr3"], float) - np.asarray(
                zero["n_full_bohr3"], float
            )
            axes[1, 1].plot(
                r[mask], cumulative_charge(r, delta_full)[mask],
                color=color, ls=linestyle, label=f"{label} - zero",
            )
        axes[1, 1].axhline(0.0, color="0.35", ls=":", lw=1.0)
        axes[1, 1].set(
            xlim=(-0.5, r[mask][-1]),
            xlabel=r"$r$ [Bohr]",
            ylabel=r"$4\pi\int_0^r r'^2\Delta n_{\rm full}(r')\,dr'$ [e]",
        )
        axes[1, 1].legend()
        fig.suptitle(
            rf"Bound/free partition sensitivity: C, $T_e={TEMPERATURE_EV:g}$ eV, "
            rf"$\rho={RHO_G_CC:g}$ g cm$^{{-3}}$",
            y=0.995,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        save_figure(fig, OUTPUT_DIR / "carbon_partition_density_sensitivity")

        fig_levels, (ax_all, ax_shallow) = plt.subplots(1, 2, figsize=(10.0, 4.2))
        for mode_index, (label, state) in enumerate(states.items()):
            color, _ = styles[label]
            edge_ev = float(state["energy_cut_ha"]) * HARTREE_TO_EV
            ax_all.scatter(mode_index, edge_ev, marker="x", s=55, color="black")
            ax_shallow.scatter(mode_index, edge_ev, marker="x", s=55,
                               color="black", label="Ecut" if mode_index == 0 else None)
            for shell, energy, included in level_rows(state):
                energy_ev = energy * HARTREE_TO_EV
                marker = "o" if included else "s"
                ax_all.scatter(mode_index, energy_ev, marker=marker, color=color)
                if energy_ev > -1.0:
                    ax_shallow.scatter(mode_index, energy_ev, marker=marker,
                                       color=color)
                    ax_shallow.annotate(shell, (mode_index, energy_ev),
                                        xytext=(5, 2), textcoords="offset points")
        for axis in (ax_all, ax_shallow):
            axis.axhline(0.0, color="0.35", ls=":", lw=1.0)
            axis.set_xticks(range(len(states)), labels=list(states))
            axis.set_ylabel("absolute level energy [eV]")
        ax_all.set_title("All finite negative levels")
        ax_shallow.set_title("Near-threshold levels")
        ax_shallow.set_ylim(-1.0, max(0.02, 1.2 * max(
            float(state["energy_cut_ha"]) * HARTREE_TO_EV for state in states.values()
        )))
        fig_levels.tight_layout()
        save_figure(fig_levels, OUTPUT_DIR / "carbon_partition_level_sensitivity")
        if SHOW_FIGURES and "agg" not in plt.get_backend().lower():
            plt.show()
        else:
            plt.close("all")


def main() -> None:
    states = {
        label: solve_or_load(label, mode, value)
        for label, (mode, value) in PARTITIONS.items()
    }
    print_report(states)
    plot_report(states)
    print(f"\nsaved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
