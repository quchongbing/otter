r"""
Carbon ionization and pressure-ionization levels
================================================

This example scans carbon at :math:`T_e=100` eV with Otter's orbital
finite-temperature average atom.  The same full-AA solutions provide

* :math:`\bar{Z}=Z-Q_{\mathrm{ion}}(R_{\mathrm{WS}})`;
* :math:`Z^*=n_e^0/n_i`; and
* the carbon 1s, 2s, 2p, 3s, 3p, and 3d energies when localized, relative
  to Otter's local numerical
  continuum edge.

The level panels use a second (right-hand) y-axis for each shell's direct
contribution to the ionic charge,

.. math::

   Q^{\rm ion}_{nl}(R_{\rm WS}) = 2(2l+1)f_{\rm FD}(E_{nl})M(E_{nl})
   \int_0^{R_{\rm WS}} f_{\rm cut}(r)|P_{nl}(r)|^2\,dr.

Solid curves with circle markers are level energies; dashed curves with cross
markers are the shell charges.  Unlike the total orbital occupation
:math:`\mathrm{OCC}_{nl}=2(2l+1)f_{\rm FD}`, this quantity contains the
pressure-ionization weight and radial partition that enter
:math:`\bar Z=Z-Q_{\rm ion}(R_{\rm WS})`, following
:cite:t:`StarrettSaumon2013`.

Mean ionization is not unique.  Section 4.2 and Eq. (64) of
:cite:t:`StarrettEtAl2019` discuss these two definitions: :math:`\bar Z`
has an intuitive bound-state count but can jump at pressure ionization,
whereas :math:`Z^*` is normally smooth but need not reproduce an intuitive
chemical valence.  Neither definition changes the self-consistent AA
solution.

A shallow level is plotted only while it lies below that edge and its
threshold classification is resolved; Otter does not infer a precise
disappearance density from such points.  The bound/continuum construction
and ionic-density partition follow :cite:t:`StarrettSaumon2014`; the
negative-energy exterior matching used for shallow states follows the
boundary-matching construction discussed by :cite:t:`StarrettEtAl2019`.
For this scan the displayed edge is explicitly
:math:`E_{\mathrm{cut}}=V_{\mathrm{eff}}(0.70R_{\mathrm{max}})`.  It is the numerical
partition used by this orbital AA calculation, not the strict
:math:`\epsilon=0` lower limit in the Appendix-A continuum integral.

For context, the ionization figure overlays the model-dependent
:math:`Z^{\mathrm{free}}` curves digitized from Fig. 3(a) of
:cite:t:`BethkenhagenEtAl2020`.  Those quantities are not identified with
either Otter :math:`\bar{Z}` or :math:`Z^*`; they are displayed only as a
definition-aware comparison.

The page uses a checksummed, pressure-ionization-refined Otter result by
default.  Set
``RECOMPUTE_WITH_OTTER = True`` below to extend the accepted scan.  Every
newly calculated state uses
``2**12 = 4096`` radial points.  The installed accepted archive uses the same
resolution and contains the shell-resolved :math:`Q^{\rm ion}_{nl}`
diagnostic, so its requested densities can be reused as exact-method seeds.
The script is complete and directly executable; recomputed data are staged under
``benchmarks/outputs`` rather than replacing the accepted gallery data.
Both figures are saved as PNG and vector PDF files.

Requested states that do not reach a physical SCF fixed point are never
stored as electronic results.  They are retained separately as failed-point
audit records and marked by a vertical dotted line in recomputed plots.  A
diagnostic ``"fd_m"`` result is never substituted for a failed production
``bound_occ_mode="fd"`` state because it changes the self-consistent bound
density.  Set
``OTTER_RETRY_NONCONVERGED_CARBON_IONIZATION=1`` to explicitly retry a cached
failed point after solver changes.

"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from otter.electronic import FullExternalConfig, solve_full_only
from otter.plotting import PALETTES, grid_figsize, save_figure, style_context


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
# Set this one switch in the script, then run the file directly.  Incremental
# reuse below ensures that only densities absent from the accepted scan/cache
# are calculated.
RECOMPUTE_WITH_OTTER = False
USE_PRECOMPUTED_DATA = not RECOMPUTE_WITH_OTTER
RETRY_NONCONVERGED_POINTS = (
    os.environ.get("OTTER_RETRY_NONCONVERGED_CARBON_IONIZATION", "0") == "1"
)

ELEMENT = "C"
TEMPERATURE_EV = 100.0
# This explicit grid is identical to the checksummed accepted 72-state scan.
# It is deliberately denser from 0.1--6 g cm^-3, where shallow-state branches
# change rapidly.  The points diagnose finite-grid level disappearance; they
# do not assert exact pressure-ionization densities.
DENSITIES_G_CC = np.asarray(
    (
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        1.00,
        1.05,
        1.10,
        1.15,
        1.20,
        1.25,
        1.30,
        1.40,
        1.50,
        1.60,
        1.70,
        1.80,
        1.90,
        2.00,
        2.10,
        2.20,
        2.30,
        2.35,
        2.40,
        2.50,
        2.55,
        2.60,
        2.70,
        2.80,
        2.90,
        3.00,
        3.20,
        3.50,
        3.80,
        4.00,
        4.10,
        4.20,
        4.30,
        4.40,
        4.50,
        4.60,
        4.70,
        4.80,
        4.90,
        5.00,
        5.20,
        5.40,
        5.50,
        5.60,
        5.80,
        6.00,
        8.00,
        10.0,
        15.0,
        20.0,
        30.0,
        50.0,
        75.0,
        100.0,
        200.0,
        300.0,
        400.0,
        450.0,
    ),
    dtype=float,
)

# No release-only density is requested beyond the accepted scan.  A developer
# can add points here and enable RECOMPUTE_WITH_OTTER to stage new candidates.
NEW_DENSITIES_G_CC = np.asarray((), dtype=float)

# Two independent AA states, each with four continuum workers, use eight
# explicit workers.  This is faster than a sequential scan without the large
# memory peak caused by nesting 3 x 6 processes on typical workstations.
MAX_STATE_WORKERS = 2
CONTINUUM_WORKERS_PER_STATE = 2
# Incremental extension is the normal workflow: reuse every requested point
# already present in the checksummed accepted scan, then calculate only new
# densities.  Set this to False only to force an independent full scan.
REUSE_ACCEPTED_POINTS_WHEN_RECOMPUTING = True
AA_N_POINTS = 2**12
BOUND_ENERGY_CUT_MODE = "v_frac"
BOUND_ENERGY_CUT_VALUE = 0.70

SCHEMA = "otter_carbon_ionization_levels_v3"
LEGACY_BASELINE_SCHEMAS = {
    "otter_carbon_ionization_levels_v1",
    "otter_carbon_ionization_levels_v2",
}
HARTREE_TO_EV = 27.211386245988
ORBITAL_LETTERS = ("s", "p", "d", "f", "g", "h")
DISPLAYED_SHELLS = ("1s", "2s", "2p", "3s", "3p", "3d")


def _repository_root() -> Path:
    """Locate the source tree from either the gallery source or generated copy."""
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    source_file = globals().get("__file__")
    if source_file is not None:
        source = Path(str(source_file)).resolve()
        candidates.extend([source.parent, *source.parents])
    for candidate in candidates:
        if (candidate / "src" / "otter").is_dir() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
    raise FileNotFoundError("Cannot locate the Otter repository root.")


ROOT = _repository_root()
BASELINE_DIR = ROOT / "benchmarks" / "baselines" / "carbon_ionization_levels"
BASELINE_PATH = BASELINE_DIR / "C_Te100eV_density_scan.npz"
BASELINE_MANIFEST = BASELINE_DIR / "manifest.json"
OUTPUT_DIR = ROOT / "benchmarks" / "outputs" / "carbon_ionization_levels"
CANDIDATE_PATH = OUTPUT_DIR / "C_Te100eV_density_scan.npz"
POINT_CACHE_DIR = OUTPUT_DIR / "point_cache"
FIGURE_DIR = OUTPUT_DIR / "figures"
REFERENCE_DIR = (
    ROOT
    / "benchmarks"
    / "reference_data"
    / "bethkenhagen_et_al_2020_carbon_ionization"
)
REFERENCE_MANIFEST = REFERENCE_DIR / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bethkenhagen_reference() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load checksummed Fig. 3(a) ``Z_free`` model curves for context."""
    manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "otter_reference_manifest_v1"
        or manifest.get("dataset_id")
        != "bethkenhagen_et_al_2020_carbon_ionization"
        or str(manifest["source"]["doi"])
        != "10.1103/PhysRevResearch.2.023260"
    ):
        raise ValueError("Unexpected Bethkenhagen et al. reference manifest.")
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for record in manifest["files"]:
        path = REFERENCE_DIR / str(record["file"])
        if _sha256(path) != str(record["sha256"]):
            raise RuntimeError(f"Reference checksum mismatch: {path.name}.")
        values = np.loadtxt(path, delimiter=",", comments="#")
        if (
            values.ndim != 2
            or values.shape[1] != 2
            or np.any(~np.isfinite(values))
            or np.any(values[:, 0] <= 0.0)
            or np.any(np.diff(values[:, 0]) <= 0.0)
        ):
            raise ValueError(f"Malformed reference curve: {path.name}.")
        curves[str(record["label"])] = (values[:, 0], values[:, 1])
    return curves


def _level_label(l_value: int, radial_index: int) -> str:
    principal_n = int(radial_index) + int(l_value)
    if 0 <= int(l_value) < len(ORBITAL_LETTERS):
        return f"{principal_n}{ORBITAL_LETTERS[int(l_value)]}"
    return f"n={principal_n},l={int(l_value)}"


def _configuration(rho_g_cc: float) -> FullExternalConfig:
    return FullExternalConfig(
        element=ELEMENT,
        temperature_ev=float(TEMPERATURE_EV),
        rho_g_cc=float(rho_g_cc),
        run_mode="full",
        n_points=int(AA_N_POINTS),
        # Retain headroom on the steep high-density ionization branch.  The
        # convergence criterion itself is unchanged.
        stage2_max_iter=180,
        cont_n_jobs=int(CONTINUUM_WORKERS_PER_STATE),
        cont_shards=int(2 * CONTINUUM_WORKERS_PER_STATE),
        show_scf_progress=False,
        save_data=False,
        bound_occ_mode="fd",
        bound_rmax_mult=None,
        bound_energy_cut_mode=BOUND_ENERGY_CUT_MODE,
        bound_energy_cut=BOUND_ENERGY_CUT_VALUE,
        # Match a shallow negative-energy orbital at the common outer SCF
        # boundary, with no separate enlarged bound-only box.  This optional
        # numerical refinement is motivated by the exterior matching in
        # Starrett et al. (2019), Eqs. (21)-(22), but is not identical to the
        # ion-sphere-boundary implementation in that work.
        bound_zero_tail_refine=True,
        bound_zero_tail_max_binding_ha=1.0e-2,
        bound_zero_tail_scan_points=64,
        bound_zero_tail_l_max=1,
        bound_zero_tail_edge_rel_tol=0.1,
        b3_tail_model="full",
    )


def _finite_levels(
    result: dict[str, Any],
    *,
    continuum_edge_ha: float,
) -> dict[str, float]:
    """Return levels below the same continuum edge used by the AA density."""
    energies = np.asarray(
        result.get("bound_energy_ha", np.empty((0, 0))),
        dtype=float,
    )
    l_values = np.asarray(
        result.get("bound_l_list", np.arange(energies.shape[0])),
        dtype=int,
    )
    levels: dict[str, float] = {}
    for l_index, l_value in enumerate(l_values):
        for state_index in range(energies.shape[1]):
            energy_ha = float(energies[l_index, state_index])
            if np.isfinite(energy_ha) and energy_ha < continuum_edge_ha:
                label = _level_label(int(l_value), int(state_index + 1))
                levels[label] = (
                    energy_ha - float(continuum_edge_ha)
                ) * HARTREE_TO_EV
    return levels


def _finite_level_ion_charges(
    result: dict[str, Any],
) -> dict[str, float]:
    r"""Return shell contributions to ``Q_ion(R_WS)`` in electrons.

    ``bound_q_ion_ws`` is assembled from the exact final orbitals used by the
    electronic solver and includes ``2(2l+1) f_FD M(E) f_cut(r)``.  Keeping
    this reduction in the producer avoids reconstructing a spatially weighted
    quantity from energies alone.
    """
    charges = np.asarray(
        result.get("bound_q_ion_ws", np.empty((0, 0))),
        dtype=float,
    )
    l_values = np.asarray(
        result.get("bound_l_list", np.arange(charges.shape[0])),
        dtype=int,
    )
    if charges.ndim != 2 or l_values.shape != (charges.shape[0],):
        raise RuntimeError("Malformed shell-resolved Q_ion table.")
    total = float(np.nansum(charges))
    q_ion_ws = float(result.get("q_ion_ws", np.nan))
    if (
        not np.isfinite(q_ion_ws)
        or abs(total - q_ion_ws) > 2.0e-8 * max(1.0, abs(q_ion_ws))
    ):
        raise RuntimeError(
            "Shell-resolved Q_ion does not close the total WS ionic charge: "
            f"sum={total:.12g}, total={q_ion_ws:.12g}."
        )

    shell_charges: dict[str, float] = {}
    for l_index, l_value in enumerate(l_values):
        for state_index in range(charges.shape[1]):
            charge = float(charges[l_index, state_index])
            if not np.isfinite(charge):
                continue
            if charge < -1.0e-12:
                raise RuntimeError("A shell-resolved ionic charge is negative.")
            label = _level_label(int(l_value), int(state_index + 1))
            shell_charges[label] = max(charge, 0.0)
    return shell_charges


def _solve_density(rho_g_cc: float) -> dict[str, Any]:
    """Compute and reduce one independent full-AA state."""
    started = time.perf_counter()
    result = solve_full_only(_configuration(float(rho_g_cc)))
    elapsed_s = time.perf_counter() - started

    stage2_history = list(result.get("history", ()))
    stage2_error = (
        float(stage2_history[-1].get("err", np.nan))
        if stage2_history
        else np.nan
    )
    if not bool(result.get("stage2_converged", False)):
        bound_charge = np.asarray(
            [entry.get("charge_bound", np.nan) for entry in stage2_history],
            dtype=float,
        )
        finite_bound_charge = bound_charge[np.isfinite(bound_charge)]
        return {
            "record_type": "stage2_nonconvergence",
            "rho_g_cc": float(rho_g_cc),
            "elapsed_s": float(elapsed_s),
            "stage": "full_aa_stage2",
            "stage2_converged": False,
            "stage2_error": stage2_error,
            "stage2_iters": int(
                result.get("stage2_iters", len(stage2_history))
            ),
            "bound_charge_min_e": (
                float(np.min(finite_bound_charge))
                if finite_bound_charge.size
                else np.nan
            ),
            "bound_charge_max_e": (
                float(np.max(finite_bound_charge))
                if finite_bound_charge.size
                else np.nan
            ),
            "message": (
                f"C rho={rho_g_cc:g} g/cc: production full-AA stage 2 "
                "did not reach a physical fixed point."
            ),
            "point_source": "fresh_otter_solve",
        }
    threshold_status = str(result.get("threshold_state_status", "none")).lower()
    if not np.isfinite(stage2_error):
        raise RuntimeError(
            f"C rho={rho_g_cc:g} g/cc: final stage-2 error is not finite."
        )

    meta = dict(result.get("meta", {}))
    continuum_edge_ha = float(meta["bound_energy_cut_ha"])
    n_i_bohr3 = float(meta["n_i_bohr3"])
    n0_bohr3 = float(meta["n0_final_bohr3"])
    zbar = float(result["zbar"])
    zstar = n0_bohr3 / n_i_bohr3
    if not np.all(
        np.isfinite(
            (
                continuum_edge_ha,
                n_i_bohr3,
                n0_bohr3,
                zbar,
                zstar,
                float(result["mu"]),
            )
        )
    ):
        raise RuntimeError(f"C rho={rho_g_cc:g} g/cc produced non-finite output.")

    raw_levels = _finite_levels(
        result,
        continuum_edge_ha=continuum_edge_ha,
    )
    level_q_ion_ws = _finite_level_ion_charges(result)
    levels = raw_levels
    if threshold_status in {"marginal", "unresolved"}:
        # The flagged state concerns the shallow outer branch.  Retain the
        # deeply bound 1s diagnostic, but never turn an unreliable shallow
        # eigenvalue into a pressure-ionization datum.
        levels = {
            label: energy for label, energy in levels.items() if label == "1s"
        }

    return {
        "rho_g_cc": float(rho_g_cc),
        "zbar": zbar,
        "zstar": zstar,
        "mu_ha": float(result["mu"]),
        "continuum_edge_ha": continuum_edge_ha,
        "threshold_status": threshold_status,
        "threshold_representation": str(
            result.get("threshold_state_representation", "none")
        ),
        "levels_ev": levels,
        # Keep every shell charge that actually entered Q_ion, including a
        # shallow branch whose energy is hidden by a marginal/unresolved
        # threshold flag.  This makes the smooth M(E) partition auditable.
        "level_q_ion_ws": level_q_ion_ws,
        "elapsed_s": float(elapsed_s),
        "stage2_converged": True,
        "stage2_error": stage2_error,
        "stage2_iters": int(result.get("stage2_iters", len(stage2_history))),
        "point_source": "fresh_otter_solve",
    }


def _point_cache_path(rho_g_cc: float) -> Path:
    token = f"{float(rho_g_cc):012.6f}".replace(".", "p")
    return POINT_CACHE_DIR / f"C_Te100eV_rho{token}.json"


def _point_failure_cache_path(rho_g_cc: float) -> Path:
    return _point_cache_path(rho_g_cc).with_suffix(".failure.json")


def _cache_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return method metadata shared by converged and failed point caches."""
    return {
        "schema_version": SCHEMA,
        "temperature_ev": TEMPERATURE_EV,
        "aa_n_points": AA_N_POINTS,
        "continuum_workers": CONTINUUM_WORKERS_PER_STATE,
        "bound_occ_mode": "fd",
        "bound_energy_cut_mode": BOUND_ENERGY_CUT_MODE,
        "bound_energy_cut_value": BOUND_ENERGY_CUT_VALUE,
        "record": record,
    }


def _cache_metadata_matches(payload: dict[str, Any]) -> bool:
    """Check the immutable scientific controls represented by a point cache."""
    return bool(
        payload.get("schema_version") == SCHEMA
        and np.isclose(
            float(payload.get("temperature_ev", np.nan)),
            TEMPERATURE_EV,
        )
        and int(payload.get("aa_n_points", -1)) == AA_N_POINTS
        # The first 61 v3 caches predate this explicit field; v3 always used
        # the production FD state sum, so a missing value is unambiguous.
        and payload.get("bound_occ_mode", "fd") == "fd"
        and payload.get("bound_energy_cut_mode") == BOUND_ENERGY_CUT_MODE
        and np.isclose(
            float(payload.get("bound_energy_cut_value", np.nan)),
            BOUND_ENERGY_CUT_VALUE,
        )
    )


def _save_point_cache(row: dict[str, Any]) -> None:
    """Save one completed AA point so interrupted scans can resume safely."""
    POINT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = _cache_payload(row)
    # Preserve the row key used by existing v3 caches.
    payload["row"] = payload.pop("record")
    path = _point_cache_path(float(row["rho_g_cc"]))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_point_cache(rho_g_cc: float) -> dict[str, Any] | None:
    path = _point_cache_path(rho_g_cc)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _cache_metadata_matches(payload):
        return None
    row = dict(payload["row"])
    if not np.isclose(float(row["rho_g_cc"]), float(rho_g_cc)):
        return None
    row["levels_ev"] = {
        str(key): float(value)
        for key, value in dict(row.get("levels_ev", {})).items()
    }
    row["level_q_ion_ws"] = {
        str(key): float(value)
        for key, value in dict(row.get("level_q_ion_ws", {})).items()
    }
    if not row["level_q_ion_ws"]:
        return None
    return row


def _save_point_failure(record: dict[str, Any]) -> None:
    """Checkpoint a scientifically unusable state without storing AA output."""
    POINT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = _cache_payload(record)
    path = _point_failure_cache_path(float(record["rho_g_cc"]))
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_point_failure(rho_g_cc: float) -> dict[str, Any] | None:
    """Load a same-method failed-point audit unless an explicit retry is set."""
    if RETRY_NONCONVERGED_POINTS:
        return None
    path = _point_failure_cache_path(rho_g_cc)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _cache_metadata_matches(payload):
        return None
    record = dict(payload.get("record", {}))
    if (
        record.get("record_type") != "stage2_nonconvergence"
        or not np.isclose(float(record.get("rho_g_cc", np.nan)), rho_g_cc)
        or bool(record.get("stage2_converged", True))
    ):
        return None
    return record


def _accepted_seed_rows(densities: np.ndarray) -> list[dict[str, Any]]:
    """Reuse accepted points while computing only newly requested states."""
    if not REUSE_ACCEPTED_POINTS_WHEN_RECOMPUTING:
        return []
    if not BASELINE_PATH.is_file() or not BASELINE_MANIFEST.is_file():
        return []
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "otter_example_manifest_v1"
        or manifest.get("example_id") != "carbon_ionization_levels"
        or not str(manifest.get("status", "")).startswith("accepted")
        or dict(manifest.get("state", {})).get("data_sha256")
        != _sha256(BASELINE_PATH)
    ):
        raise RuntimeError("Cannot reuse an unreviewed carbon-ionization state.")

    with np.load(BASELINE_PATH, allow_pickle=False) as archive:
        old = {key: np.asarray(archive[key]) for key in archive.files}
    # Never mix a coarse archive into a production 4096-point regeneration,
    # and never reconstruct spatial Q_ion data from energies.
    if (
        int(np.asarray(old.get("aa_n_points", -1)).item()) != AA_N_POINTS
        or "level_q_ion_ws" not in old
        or "level_q_ion_ws_is_available" not in old
    ):
        return []
    old = _upgrade_accepted_state(old, manifest)
    if not bool(np.asarray(old["level_q_ion_ws_available"]).item()):
        return []
    old_rho = np.asarray(old["rho_g_cc"], dtype=float)
    requested = {float(value) for value in np.asarray(densities, dtype=float)}
    labels = tuple(np.asarray(old["level_labels"], dtype=str))
    rows: list[dict[str, Any]] = []
    for index, rho_g_cc in enumerate(old_rho):
        if float(rho_g_cc) not in requested:
            continue
        levels = {
            label: float(old["level_energy_relative_edge_ev"][index, level_index])
            for level_index, label in enumerate(labels)
            if bool(old["level_is_bound"][index, level_index])
        }
        level_q_ion_ws = {
            label: float(old["level_q_ion_ws"][index, level_index])
            for level_index, label in enumerate(labels)
            if bool(old["level_q_ion_ws_is_available"][index, level_index])
        }
        rows.append(
            {
                "rho_g_cc": float(rho_g_cc),
                "zbar": float(old["zbar"][index]),
                "zstar": float(old["zstar"][index]),
                "mu_ha": float(old["mu_ha"][index]),
                "continuum_edge_ha": float(old["continuum_edge_ha"][index]),
                "threshold_status": str(old["threshold_status"][index]),
                "threshold_representation": str(
                    old["threshold_representation"][index]
                ),
                "levels_ev": levels,
                "level_q_ion_ws": level_q_ion_ws,
                "elapsed_s": float(old["elapsed_s"][index]),
                "stage2_converged": True,
                "stage2_error": float(old["stage2_error"][index]),
                "stage2_iters": int(old["stage2_iters"][index]),
                "point_source": "accepted_baseline_seed",
            }
        )
    return rows


def _compute_scan() -> dict[str, np.ndarray]:
    densities = np.unique(np.asarray(DENSITIES_G_CC, dtype=float))
    if densities.size < 2 or np.any(densities <= 0.0):
        raise ValueError("DENSITIES_G_CC must contain at least two positive values.")

    rows = _accepted_seed_rows(densities)
    failures: list[dict[str, Any]] = []
    seeded_rho = {float(row["rho_g_cc"]) for row in rows}
    pending: list[float] = []
    for rho in densities:
        if float(rho) in seeded_rho:
            print(f"[accepted baseline] C, rho={float(rho):g} g/cc")
            continue
        cached = _load_point_cache(float(rho))
        if cached is not None:
            rows.append(cached)
            print(f"[cached] C, rho={float(rho):g} g/cc")
            continue
        failed = _load_point_failure(float(rho))
        if failed is not None:
            failures.append(failed)
            print(f"[cached nonconverged] C, rho={float(rho):g} g/cc")
            continue
        pending.append(float(rho))

    worker_count = min(max(int(MAX_STATE_WORKERS), 1), int(densities.size))
    if worker_count == 1:
        for rho in pending:
            record = _solve_density(float(rho))
            if bool(record.get("stage2_converged", False)):
                _save_point_cache(record)
                rows.append(record)
                print(f"[computed] C, rho={rho:g} g/cc")
            else:
                _save_point_failure(record)
                failures.append(record)
                print(f"[not converged] C, rho={rho:g} g/cc")
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_solve_density, rho): rho for rho in pending
            }
            for future in as_completed(futures):
                rho = futures[future]
                record = future.result()
                if bool(record.get("stage2_converged", False)):
                    _save_point_cache(record)
                    rows.append(record)
                    print(f"[computed] C, rho={rho:g} g/cc")
                else:
                    _save_point_failure(record)
                    failures.append(record)
                    print(f"[not converged] C, rho={rho:g} g/cc")
    if len(rows) + len(failures) != densities.size:
        raise RuntimeError(
            f"Expected {densities.size} density attempts, obtained "
            f"{len(rows)} converged and {len(failures)} failed records."
        )
    rows.sort(key=lambda item: float(item["rho_g_cc"]))
    failures.sort(key=lambda item: float(item["rho_g_cc"]))

    # Track the K/L branches and the short-lived localized 3s branch.  Higher
    # finite-box Rydberg roots remain internal diagnostics rather than ionic
    # levels in this gallery.
    labels = sorted(
        {
            label
            for row in rows
            for label in (
                set(row["levels_ev"]) | set(row["level_q_ion_ws"])
            )
            if label in DISPLAYED_SHELLS
        },
        key=lambda label: (
            int(label[:-1]) if label[-1:] in ORBITAL_LETTERS else 99,
            ORBITAL_LETTERS.index(label[-1])
            if label[-1:] in ORBITAL_LETTERS
            else 99,
        ),
    )
    level_energy_ev = np.zeros((len(rows), len(labels)), dtype=float)
    level_is_bound = np.zeros_like(level_energy_ev, dtype=bool)
    level_q_ion_ws = np.zeros_like(level_energy_ev, dtype=float)
    level_q_ion_ws_is_available = np.zeros_like(level_energy_ev, dtype=bool)
    for density_index, row in enumerate(rows):
        for level_index, label in enumerate(labels):
            if label in row["levels_ev"]:
                level_energy_ev[density_index, level_index] = float(
                    row["levels_ev"][label]
                )
                level_is_bound[density_index, level_index] = True
            if label in row["level_q_ion_ws"]:
                level_q_ion_ws[density_index, level_index] = float(
                    row["level_q_ion_ws"][label]
                )
                level_q_ion_ws_is_available[density_index, level_index] = True

    return {
        "schema_version": np.asarray(SCHEMA),
        "element_symbol": np.asarray(ELEMENT),
        "temperature_ev": np.asarray(TEMPERATURE_EV),
        "rho_g_cc": np.asarray([row["rho_g_cc"] for row in rows]),
        "zbar": np.asarray([row["zbar"] for row in rows]),
        "zstar": np.asarray([row["zstar"] for row in rows]),
        "mu_ha": np.asarray([row["mu_ha"] for row in rows]),
        "continuum_edge_ha": np.asarray(
            [row["continuum_edge_ha"] for row in rows]
        ),
        "threshold_status": np.asarray(
            [row["threshold_status"] for row in rows]
        ),
        "threshold_representation": np.asarray(
            [row["threshold_representation"] for row in rows]
        ),
        "level_labels": np.asarray(labels),
        "level_energy_relative_edge_ev": level_energy_ev,
        "level_is_bound": level_is_bound,
        "level_q_ion_ws": level_q_ion_ws,
        "level_q_ion_ws_is_available": level_q_ion_ws_is_available,
        "level_q_ion_ws_available": np.asarray(True),
        "elapsed_s": np.asarray([row["elapsed_s"] for row in rows]),
        "stage2_converged": np.asarray(
            [row["stage2_converged"] for row in rows],
            dtype=bool,
        ),
        "stage2_error": np.asarray([row["stage2_error"] for row in rows]),
        "stage2_iters": np.asarray([row["stage2_iters"] for row in rows]),
        "point_source": np.asarray([row["point_source"] for row in rows]),
        "failed_rho_g_cc": np.asarray(
            [record["rho_g_cc"] for record in failures], dtype=float
        ),
        "failed_stage": np.asarray(
            [record["stage"] for record in failures], dtype=str
        ),
        "failed_message": np.asarray(
            [record["message"] for record in failures], dtype=str
        ),
        "failed_stage2_iters": np.asarray(
            [record["stage2_iters"] for record in failures], dtype=int
        ),
        "failed_stage2_error": np.asarray(
            [record["stage2_error"] for record in failures], dtype=float
        ),
        "failed_bound_charge_min_e": np.asarray(
            [record["bound_charge_min_e"] for record in failures], dtype=float
        ),
        "failed_bound_charge_max_e": np.asarray(
            [record["bound_charge_max_e"] for record in failures], dtype=float
        ),
        "failed_point_source": np.asarray(
            [record["point_source"] for record in failures], dtype=str
        ),
        "bound_occ_mode": np.asarray("fd"),
        "bound_rmax_mult": np.asarray("none"),
        "bound_zero_tail_refine": np.asarray(True),
        "bound_energy_cut_mode": np.asarray(BOUND_ENERGY_CUT_MODE),
        "bound_energy_cut_value": np.asarray(BOUND_ENERGY_CUT_VALUE),
        "b3_tail_model": np.asarray("full"),
        "aa_n_points": np.asarray(AA_N_POINTS),
    }


def _upgrade_accepted_state(
    state: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Expose a legacy accepted archive through the current plotting schema.

    v1 predates per-point SCF errors; v1 and v2 both predate shell-resolved
    ``Q_ion``.  Missing scientific data are marked unavailable and are never
    reconstructed from energies or OCC values.
    """
    schema = str(state.get("schema_version", np.asarray("")).item())
    if schema == SCHEMA:
        upgraded = dict(state)
        upgraded.setdefault("failed_rho_g_cc", np.asarray([], dtype=float))
        upgraded.setdefault("failed_stage", np.asarray([], dtype=str))
        upgraded.setdefault("failed_message", np.asarray([], dtype=str))
        upgraded.setdefault("failed_stage2_iters", np.asarray([], dtype=int))
        upgraded.setdefault("failed_stage2_error", np.asarray([], dtype=float))
        upgraded.setdefault(
            "failed_bound_charge_min_e", np.asarray([], dtype=float)
        )
        upgraded.setdefault(
            "failed_bound_charge_max_e", np.asarray([], dtype=float)
        )
        upgraded.setdefault("failed_point_source", np.asarray([], dtype=str))
        return upgraded
    if schema not in LEGACY_BASELINE_SCHEMAS:
        raise ValueError(f"Unsupported legacy carbon schema {schema!r}.")
    rho = np.asarray(state.get("rho_g_cc", ()), dtype=float)
    upgraded = dict(state)
    if schema == "otter_carbon_ionization_levels_v1":
        converged_count = int(
            dict(manifest.get("scientific_audit", {})).get(
                "full_aa_converged_states",
                -1,
            )
        )
        if rho.ndim != 1 or converged_count != rho.size:
            raise RuntimeError(
                "The accepted v1 carbon baseline lacks a complete "
                "convergence audit."
            )
        upgraded.update({
            "stage2_converged": np.ones(rho.shape, dtype=bool),
            "stage2_error": np.full(rho.shape, np.nan, dtype=float),
            "stage2_iters": np.full(rho.shape, -1, dtype=int),
            "point_source": np.full(
                rho.shape,
                "accepted_v1_baseline",
                dtype="<U20",
            ),
        })
    level_shape = np.asarray(
        upgraded["level_energy_relative_edge_ev"],
        dtype=float,
    ).shape
    upgraded.update({
        "schema_version": np.asarray(SCHEMA),
        "level_q_ion_ws": np.zeros(level_shape, dtype=float),
        "level_q_ion_ws_is_available": np.zeros(level_shape, dtype=bool),
        "level_q_ion_ws_available": np.asarray(False),
        "failed_rho_g_cc": np.asarray([], dtype=float),
        "failed_stage": np.asarray([], dtype=str),
        "failed_message": np.asarray([], dtype=str),
        "failed_stage2_iters": np.asarray([], dtype=int),
        "failed_stage2_error": np.asarray([], dtype=float),
        "failed_bound_charge_min_e": np.asarray([], dtype=float),
        "failed_bound_charge_max_e": np.asarray([], dtype=float),
        "failed_point_source": np.asarray([], dtype=str),
    })
    return upgraded


def _validate_state(state: dict[str, np.ndarray]) -> None:
    required = {
        "schema_version",
        "element_symbol",
        "temperature_ev",
        "rho_g_cc",
        "zbar",
        "zstar",
        "mu_ha",
        "continuum_edge_ha",
        "level_labels",
        "level_energy_relative_edge_ev",
        "level_is_bound",
        "level_q_ion_ws",
        "level_q_ion_ws_is_available",
        "level_q_ion_ws_available",
        "bound_energy_cut_mode",
        "bound_energy_cut_value",
        "aa_n_points",
        "stage2_converged",
        "stage2_error",
        "point_source",
        "failed_rho_g_cc",
        "failed_stage",
        "failed_message",
        "failed_stage2_iters",
        "failed_stage2_error",
        "failed_bound_charge_min_e",
        "failed_bound_charge_max_e",
        "failed_point_source",
    }
    missing = required.difference(state)
    if missing:
        raise KeyError(f"Carbon ionization state is missing {sorted(missing)}.")
    if str(state["schema_version"].item()) != SCHEMA:
        raise ValueError("Unsupported carbon ionization state schema.")
    if str(state["element_symbol"].item()) != ELEMENT:
        raise ValueError("Unexpected element in carbon ionization state.")
    if not np.isclose(float(state["temperature_ev"]), TEMPERATURE_EV):
        raise ValueError("Unexpected temperature in carbon ionization state.")
    if str(state["bound_energy_cut_mode"].item()) != BOUND_ENERGY_CUT_MODE:
        raise ValueError("Unexpected bound/continuum edge convention.")
    if not np.isclose(
        float(state["bound_energy_cut_value"]),
        BOUND_ENERGY_CUT_VALUE,
    ):
        raise ValueError("Unexpected bound/continuum edge parameter.")

    rho = np.asarray(state["rho_g_cc"], dtype=float)
    if rho.ndim != 1 or rho.size < 2 or np.any(np.diff(rho) <= 0.0):
        raise ValueError("Density grid must be a strictly increasing vector.")
    for key in ("zbar", "zstar", "mu_ha"):
        values = np.asarray(state[key], dtype=float)
        if values.shape != rho.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{key} must be finite on the density grid.")
    continuum_edge = np.asarray(state["continuum_edge_ha"], dtype=float)
    if continuum_edge.shape != rho.shape or not np.all(np.isfinite(continuum_edge)):
        raise ValueError("continuum_edge_ha must be finite on the density grid.")
    level_energy = np.asarray(
        state["level_energy_relative_edge_ev"],
        dtype=float,
    )
    level_mask = np.asarray(state["level_is_bound"], dtype=bool)
    labels = np.asarray(state["level_labels"])
    if level_energy.shape != level_mask.shape or level_energy.shape != (
        rho.size,
        labels.size,
    ):
        raise ValueError("Bound-level arrays are not aligned.")
    if not np.all(np.isfinite(level_energy)):
        raise ValueError("Bound-level storage must be finite; mask absent levels.")
    if np.any(level_energy[level_mask] >= 0.0):
        raise ValueError("A stored bound level lies above the continuum edge.")
    level_q_ion_ws = np.asarray(state["level_q_ion_ws"], dtype=float)
    q_ion_mask = np.asarray(
        state["level_q_ion_ws_is_available"],
        dtype=bool,
    )
    q_ion_available = bool(np.asarray(state["level_q_ion_ws_available"]).item())
    if level_q_ion_ws.shape != level_energy.shape or q_ion_mask.shape != (
        level_energy.shape
    ):
        raise ValueError("Shell Q_ion arrays are not aligned with the levels.")
    if not np.all(np.isfinite(level_q_ion_ws)):
        raise ValueError("Shell Q_ion storage must be finite; mask absent values.")
    if np.any(level_q_ion_ws[q_ion_mask] < 0.0):
        raise ValueError("Shell Q_ion contributions must be non-negative.")
    if np.any(level_q_ion_ws[~q_ion_mask] != 0.0):
        raise ValueError("Unavailable shell Q_ion entries must use zero storage.")
    if q_ion_available != bool(np.any(q_ion_mask)):
        raise ValueError("Shell Q_ion availability metadata is inconsistent.")
    n_points = int(np.asarray(state["aa_n_points"]).item())
    if q_ion_available and n_points != AA_N_POINTS:
        raise ValueError(
            f"Production shell Q_ion data require {AA_N_POINTS} radial points."
        )
    statuses = np.asarray(state.get("threshold_status", ()), dtype=str)
    if statuses.shape != rho.shape:
        raise ValueError("threshold_status must align with the density grid.")
    if not set(statuses).issubset(
        {"none", "resolved", "marginal", "unresolved"}
    ):
        raise ValueError("Unknown threshold-state classification.")
    converged = np.asarray(state["stage2_converged"], dtype=bool)
    errors = np.asarray(state["stage2_error"], dtype=float)
    point_source = np.asarray(state["point_source"], dtype=str)
    if converged.shape != rho.shape or not np.all(converged):
        raise ValueError("Every full-AA point must have a reviewed convergence flag.")
    if errors.shape != rho.shape or point_source.shape != rho.shape:
        raise ValueError("Convergence audit arrays must align with density.")
    fresh = point_source == "fresh_otter_solve"
    if np.any(~np.isfinite(errors[fresh])):
        raise ValueError("Fresh full-AA points require a finite stage-2 error.")
    if np.any(fresh) and n_points == AA_N_POINTS and not q_ion_available:
        raise ValueError("Fresh states require shell-resolved Q_ion diagnostics.")
    failed_rho = np.asarray(state["failed_rho_g_cc"], dtype=float)
    failed_size = failed_rho.size
    if (
        failed_rho.ndim != 1
        or np.any(~np.isfinite(failed_rho))
        or np.any(failed_rho <= 0.0)
        or np.any(np.diff(failed_rho) <= 0.0)
    ):
        raise ValueError("Failed-point densities must be positive and increasing.")
    failed_fields = (
        "failed_stage",
        "failed_message",
        "failed_stage2_iters",
        "failed_stage2_error",
        "failed_bound_charge_min_e",
        "failed_bound_charge_max_e",
        "failed_point_source",
    )
    if any(np.asarray(state[key]).shape != (failed_size,) for key in failed_fields):
        raise ValueError("Failed-point audit arrays must be aligned.")
    if failed_size and np.intersect1d(rho, failed_rho).size:
        raise ValueError("A density cannot be both converged and failed.")
    if np.any(np.asarray(state["failed_stage2_iters"], dtype=int) <= 0):
        raise ValueError("Failed-point iteration counts must be positive.")
    if np.any(np.asarray(state["failed_stage"], dtype=str) == "") or np.any(
        np.asarray(state["failed_message"], dtype=str) == ""
    ):
        raise ValueError("Failed-point stage and message must be recorded.")


def _save_candidate(state: dict[str, np.ndarray]) -> Path:
    _validate_state(state)
    point_source = np.asarray(state["point_source"], dtype=str)
    fresh = point_source == "fresh_otter_solve"
    fresh_errors = np.asarray(state["stage2_error"], dtype=float)[fresh]
    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = CANDIDATE_PATH.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **state)
    temporary.replace(CANDIDATE_PATH)
    manifest = {
        "schema_version": "otter_gallery_manifest_v1",
        "example_id": "carbon_ionization_levels",
        "status": "candidate_not_accepted",
        "state": {
            "data_file": CANDIDATE_PATH.name,
            "data_sha256": _sha256(CANDIDATE_PATH),
        },
        "configuration": {
            "element": ELEMENT,
            "temperature_ev": TEMPERATURE_EV,
            "densities_g_cc": np.asarray(DENSITIES_G_CC, dtype=float).tolist(),
            "reuse_accepted_points_when_recomputing": bool(
                REUSE_ACCEPTED_POINTS_WHEN_RECOMPUTING
            ),
            "bound_occ_mode": "fd",
            "bound_rmax_mult": None,
            "bound_zero_tail_refine": True,
            "bound_energy_cut_mode": BOUND_ENERGY_CUT_MODE,
            "bound_energy_cut_value": BOUND_ENERGY_CUT_VALUE,
            "b3_tail_model": "full",
            "aa_n_points": AA_N_POINTS,
            "radial_resolution_policy": "production_2**12_no_coarse_seed_reuse",
        },
        "method_references": [
            {
                "citation_key": "StarrettSaumon2013",
                "doi": "10.1103/PhysRevE.87.013104",
                "scope": "M(E) pressure-ionization weight and radial cutoff",
            },
            {
                "citation_key": "StarrettSaumon2014",
                "doi": "10.1016/j.hedp.2013.12.001",
                "scope": "average-atom and ionic-density partition",
            },
        ],
        "shell_charge_diagnostic": {
            "field": "level_q_ion_ws",
            "units": "electrons",
            "definition": (
                "2(2l+1) f_FD(E_nl) M(E_nl) integral_0^Rws "
                "f_cut(r) |P_nl(r)|^2 dr"
            ),
        },
        "scientific_audit": {
            "requested_states": int(
                np.asarray(DENSITIES_G_CC, dtype=float).size
            ),
            "stage2_converged_states": int(
                np.count_nonzero(state["stage2_converged"])
            ),
            "stage2_nonconverged_states": int(
                np.asarray(state["failed_rho_g_cc"]).size
            ),
            "stage2_nonconverged_densities_g_cc": np.asarray(
                state["failed_rho_g_cc"], dtype=float
            ).tolist(),
            "nonconverged_policy": (
                "Failed states are audit metadata only and are excluded from "
                "all electronic and shell-level arrays; diagnostic fd_m "
                "solutions are never substituted for production fd states."
            ),
            "fresh_states": int(
                np.count_nonzero(fresh)
            ),
            "accepted_seed_states": int(
                np.count_nonzero(point_source == "accepted_baseline_seed")
            ),
            "fresh_max_stage2_error": (
                float(np.max(fresh_errors)) if fresh_errors.size else None
            ),
        },
    }
    manifest_path = CANDIDATE_PATH.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CANDIDATE_PATH


def _load_precomputed() -> dict[str, np.ndarray]:
    if not BASELINE_PATH.is_file() or not BASELINE_MANIFEST.is_file():
        raise FileNotFoundError(
            "The checksummed carbon-ionization gallery state is not installed. "
            "Set USE_PRECOMPUTED_DATA=False and RECOMPUTE_WITH_OTTER=True."
        )
    manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "otter_example_manifest_v1"
        or manifest.get("example_id") != "carbon_ionization_levels"
        or not str(manifest.get("status", "")).startswith("accepted")
    ):
        raise ValueError("The installed carbon-ionization manifest is not accepted.")
    state_entry = dict(manifest.get("state", {}))
    if state_entry.get("data_file") != BASELINE_PATH.name:
        raise ValueError("Carbon ionization manifest names the wrong data file.")
    if state_entry.get("data_sha256") != _sha256(BASELINE_PATH):
        raise RuntimeError("Carbon ionization baseline checksum mismatch.")
    with np.load(BASELINE_PATH, allow_pickle=False) as archive:
        state = {key: np.asarray(archive[key]) for key in archive.files}
    state = _upgrade_accepted_state(state, manifest)
    requested_rho = np.asarray(DENSITIES_G_CC, dtype=float)
    stored_rho = np.asarray(state.get("rho_g_cc", ()), dtype=float)
    if stored_rho.shape != requested_rho.shape or not np.allclose(
        stored_rho,
        requested_rho,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(
            "The accepted carbon-ionization baseline uses an older density "
            f"grid ({stored_rho.size} states), while this producer requests "
            f"{requested_rho.size} states. Set "
            "OTTER_RECOMPUTE_CARBON_IONIZATION=1 to extend it incrementally "
            "from the accepted 4096-point states."
        )
    _validate_state(state)
    return state


def _print_state_table(state: dict[str, np.ndarray]) -> None:
    print("\nC full-AA density scan at Te=100 eV")
    print(
        f"{'rho [g/cc]':>12} {'Zbar':>10} {'Zstar':>10} "
        f"{'mu [Ha]':>12} {'threshold':>12}"
    )
    statuses = np.asarray(state.get("threshold_status", []), dtype=str)
    for index, rho in enumerate(np.asarray(state["rho_g_cc"], dtype=float)):
        status = statuses[index] if statuses.size else "not recorded"
        print(
            f"{rho:12.5g} {float(state['zbar'][index]):10.6f} "
            f"{float(state['zstar'][index]):10.6f} "
            f"{float(state['mu_ha'][index]):12.6f} {status:>12}"
        )
    failed_rho = np.asarray(state["failed_rho_g_cc"], dtype=float)
    if failed_rho.size:
        print("\nRequested states excluded because full-AA did not converge")
        for index, rho in enumerate(failed_rho):
            iterations = int(state["failed_stage2_iters"][index])
            error = float(state["failed_stage2_error"][index])
            error_text = f"{error:.3e}" if np.isfinite(error) else "not recorded"
            print(
                f"  rho={rho:g} g/cc: stage2_iters={iterations}, "
                f"last_error={error_text}; {state['failed_message'][index]}"
            )


def _plot(state: dict[str, np.ndarray]) -> None:
    rho = np.asarray(state["rho_g_cc"], dtype=float)
    zbar = np.asarray(state["zbar"], dtype=float)
    zstar = np.asarray(state["zstar"], dtype=float)
    mu = np.asarray(state["mu_ha"], dtype=float)
    all_labels = tuple(np.asarray(state["level_labels"], dtype=str))
    displayed_indices = [
        index for index, label in enumerate(all_labels)
        if label in DISPLAYED_SHELLS
    ]
    labels = tuple(all_labels[index] for index in displayed_indices)
    energies = np.asarray(
        state["level_energy_relative_edge_ev"][:, displayed_indices],
        dtype=float,
    )
    is_bound = np.asarray(
        state["level_is_bound"][:, displayed_indices],
        dtype=bool,
    )
    level_q_ion_ws = np.asarray(
        state["level_q_ion_ws"][:, displayed_indices],
        dtype=float,
    )
    q_ion_is_available = np.asarray(
        state["level_q_ion_ws_is_available"][:, displayed_indices],
        dtype=bool,
    )
    has_shell_q_ion = bool(np.asarray(state["level_q_ion_ws_available"]).item())
    failed_rho = np.asarray(state["failed_rho_g_cc"], dtype=float)
    colors = PALETTES["bing"]
    reference_curves = _load_bethkenhagen_reference()

    with style_context("thesis", palette="bing"):
        fig_ionization, (ax_z, ax_mu) = plt.subplots(
            1,
            2,
            figsize=grid_figsize(1, 2, cell_width=4.6, cell_height=3.6),
        )
        published_styles = {
            "DFT-MD": (colors[2], ":", "o"),
            "Purgatorio": (colors[3], "--", None),
            "OPAL": (colors[4], "--", None),
            "ATOMIC": (colors[5], ":", None),
            "BU-EK": (colors[6], ":", None),
            "BU-SP": (colors[7], "-.", None),
            "BU-SP + Pauli blocking": ("0.48", "--", None),
        }
        for label, (rho_reference, z_reference) in reference_curves.items():
            color, line_style, marker = published_styles[label]
            ax_z.plot(
                rho_reference,
                z_reference,
                color=color,
                ls=line_style,
                lw=2.3,
                marker=marker,
                ms=3.8 if marker else None,
                markerfacecolor="white" if marker else None,
                markeredgewidth=0.9 if marker else None,
                alpha=0.95,
                label=rf"{label}",
                zorder=2,
            )
        ax_z.plot(
            rho,
            zbar,
            color="0.08",
            lw=2.3,
            ls="-",
            marker="o",
            ms=3.8,
            label=r"Otter $\bar Z=Z-Q_{\rm ion}(R_{\rm WS})$",
            zorder=6,
            alpha=0.4,
        )
        ax_z.plot(
            rho,
            zstar,
            color=colors[1],
            lw=2.3,
            ls="-",
            marker="o",
            ms=3.8,
            label=r"Otter $Z^*=n_e^0/n_i$",
            zorder=5,
            alpha=0.4,
        )
        ax_z.set(
            xscale="log",
            xlabel=r"$\rho$ [g cm$^{-3}$]",
            ylabel="mean ionization",
            title="Mean ionization",
            xlim=(0.09, 500.0),
        )
        ax_z.legend(ncol=2, fontsize=7.0, loc="best")

        ax_mu.plot(
            rho,
            mu,
            color=colors[2],
            lw=2.3,
            ls="-",
            marker="o",
            ms=3.8,
            alpha=0.45,
        )
        ax_mu.axhline(0.0, color="0.35", ls=":", lw=0.9)
        ax_mu.set(
            xscale="log",
            xlabel=r"$\rho$ [g cm$^{-3}$]",
            ylabel=r"$\mu$ [Ha]",
            title="Electron chemical potential",
            xlim=(0.09, 500.0),
        )
        fig_ionization.suptitle(
            r"Carbon ionization at $T_e=100$ eV",
            y=0.985,
        )
        fig_ionization.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
        save_figure(
            fig_ionization,
            FIGURE_DIR / "carbon_ionization_100ev",
        )

        fig_levels, (ax_core, ax_outer) = plt.subplots(
            1,
            2,
            figsize=grid_figsize(1, 2, cell_width=4.8, cell_height=3.8),
        )
        core_indices = [
            index for index, label in enumerate(labels) if label == "1s"
        ]
        outer_indices = [
            index for index, label in enumerate(labels) if label != "1s"
        ]
        ax_core_qion = ax_core.twinx()
        ax_outer_qion = ax_outer.twinx()
        ax_core_qion.patch.set_visible(False)
        ax_outer_qion.patch.set_visible(False)

        def draw_levels(
            axis: Any,
            ion_charge_axis: Any,
            indices: list[int],
        ) -> None:
            for index in indices:
                energy_mask = is_bound[:, index]
                axis.plot(
                    rho[energy_mask],
                    energies[energy_mask, index],
                    color=colors[index % len(colors)],
                    lw=2.3,
                    ls="-",
                    marker="o",
                    ms=3.8,
                    alpha=0.65,
                    label=labels[index],
                )
                charge_mask = q_ion_is_available[:, index]
                ion_charge_axis.plot(
                    rho[charge_mask],
                    level_q_ion_ws[charge_mask, index],
                    color=colors[index % len(colors)],
                    lw=1.5,
                    ls=(0, (4, 2)),
                    marker="x",
                    ms=4.2,
                    alpha=0.80,
                    label=rf"$Q^{{\rm ion}}_{{{labels[index]}}}$",
                )
            axis.axhline(
                0.0,
                color="0.25",
                ls=":",
                lw=1.0,
                label="continuum edge" if not indices else None,
            )
            axis.set_xscale("log")
            axis.set_xlim(0.09, 500.0)
            axis.set_xlabel(r"$\rho$ [g cm$^{-3}$]")

            if axis is ax_outer:
                ion_charge_axis.set_ylim(0.0, 0.55)
            else:
                ion_charge_axis.set_ylim(bottom=0.0)

            axis.set_ylabel("level energy [eV]", labelpad=2)
            ion_charge_axis.set_ylabel(
                r"$Q^{\rm ion}_{nl}$ [e]",
                color="0.25",
                labelpad=2,
            )
            ion_charge_axis.tick_params(axis="y", colors="0.25")
            if indices:
                energy_handles, energy_labels = axis.get_legend_handles_labels()
                charge_handles, charge_labels = (
                    ion_charge_axis.get_legend_handles_labels()
                )
                axis.legend(
                    energy_handles + charge_handles,
                    energy_labels + charge_labels,
                    ncol=2,
                    fontsize=8.0,
                    loc="best",
                )

        draw_levels(ax_core, ax_core_qion, core_indices)
        draw_levels(ax_outer, ax_outer_qion, outer_indices)
        ax_core.set(title="Core level")
        ax_outer.set(title="Outer levels")
        if not outer_indices:
            ax_outer.text(
                0.5,
                0.5,
                "No outer bound level in this scan",
                transform=ax_outer.transAxes,
                ha="center",
                va="center",
            )
        if not has_shell_q_ion:
            ax_core_qion.set_visible(False)
            ax_outer_qion.set_visible(False)
            ax_outer.text(
                0.98,
                0.04,
                "shell $Q^{\\rm ion}_{nl}$ requires\n4096-point regeneration",
                transform=ax_outer.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.0,
                color="0.35",
            )
        fig_levels.suptitle(
            r"Carbon orbital levels at $T_e=100$ eV"
            "\nsolid: energy relative to continuum; dashed: "
            r"$Q^{\rm ion}_{nl}(R_{\rm WS})$",
            y=0.985,
        )
        fig_levels.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
        save_figure(
            fig_levels,
            FIGURE_DIR / "carbon_bound_levels_100ev",
        )
    #if "agg" not in plt.get_backend().lower():
        plt.show()


def main() -> None:
    if bool(USE_PRECOMPUTED_DATA) == bool(RECOMPUTE_WITH_OTTER):
        raise ValueError(
            "Select exactly one data path: set one of USE_PRECOMPUTED_DATA "
            "and RECOMPUTE_WITH_OTTER to True."
        )
    if RECOMPUTE_WITH_OTTER:
        state = _compute_scan()
        path = _save_candidate(state)
        point_source = np.asarray(state["point_source"], dtype=str)
        accepted_count = int(
            np.count_nonzero(point_source == "accepted_baseline_seed")
        )
        fresh_count = int(
            np.count_nonzero(point_source == "fresh_otter_solve")
        )
        failed_count = int(np.asarray(state["failed_rho_g_cc"]).size)
        print(
            f"Using incrementally assembled Otter data staged at {path}: "
            f"reused {accepted_count} accepted states; calculated "
            f"{fresh_count} new states; retained {failed_count} "
            "nonconverged audit record(s)."
        )
    else:
        state = _load_precomputed()
        print(
            "Using checksummed Otter data from "
            f"{BASELINE_PATH.relative_to(ROOT)}."
        )
    _print_state_table(state)
    _plot(state)


if __name__ == "__main__":
    main()
