"""Regression tests for pickle-free, atomic electronic-result archives."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from otter.io.results import save_full_external_data, save_mixture_data


def _assert_pickle_free(path: Path) -> None:
    with np.load(path, allow_pickle=False) as payload:
        assert payload.files
        assert all(payload[key].dtype.kind != "O" for key in payload.files)


def test_full_external_archive_is_pickle_free_with_none_metadata(tmp_path) -> None:
    r = np.linspace(0.01, 2.0, 16)
    written = save_full_external_data(
        output_dir=tmp_path,
        element_symbol="C",
        z=6,
        temperature_ev=10.0,
        rho_g_cc=3.0,
        result={
            "r": r,
            "n_full": np.exp(-r),
            "bound_q_ion_ws": np.asarray(((1.2, 0.1), (0.2, np.nan))),
        },
        metadata={"optional": None, "converged": True},
    )
    path = Path(written["data_npz"])
    _assert_pickle_free(path)
    with np.load(path, allow_pickle=False) as payload:
        assert payload["meta_optional"].item() == "null"
        np.testing.assert_allclose(
            payload["bound_q_ion_ws"],
            np.asarray(((1.2, 0.1), (0.2, np.nan))),
            equal_nan=True,
        )
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_mixture_archive_encodes_nonnumeric_history_without_objects(tmp_path) -> None:
    r = np.linspace(0.01, 2.0, 16)

    def species(symbol: str, count: float) -> dict:
        return {
            "element": symbol,
            "count": count,
            "x": count / 3.0,
            "volume_bohr3": 10.0,
            "r_ws_bohr": 1.0,
            "result": {
                "r": r,
                "n_full": np.exp(-r),
                "n0": 0.01,
                "mu": 0.1,
                "r_ws": 1.0,
            },
        }

    written = save_mixture_data(
        output_dir=tmp_path,
        mixture_label="CH2",
        temperature_ev=10.0,
        rho_g_cc=1.0,
        result={
            "species": [species("C", 1.0), species("H", 2.0)],
            "history": [
                {"method": "bracket", "diagnostic": {"eligible": True}},
                {"method": "brentq", "diagnostic": None},
            ],
            "meta": {"root_detail": None},
        },
    )
    path = Path(written["data_npz"])
    _assert_pickle_free(path)
    with np.load(path, allow_pickle=False) as payload:
        assert payload["history_method"].tolist() == [
            '"bracket"',
            '"brentq"',
        ]
        assert payload["history_diagnostic"].dtype.kind == "U"
        assert payload["meta_root_detail"].item() == "null"
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))
