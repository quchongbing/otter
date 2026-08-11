"""Tests for Otter's publication-figure styling helpers."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from otter.plotting import (
    FIGURE_SIZES,
    MODEL_STYLES,
    PAIR_COLORS,
    PALETTES,
    add_panel_label,
    grid_figsize,
    save_figure,
    set_style,
    style_context,
    style_rcparams,
)


def test_importing_core_otter_does_not_import_matplotlib() -> None:
    """Importing the numerical API should not initialize Matplotlib."""
    repository_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import otter; "
                "assert 'matplotlib' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_public_style_constants_are_stable() -> None:
    assert PALETTES["nature"][0] == "#E64B35"
    assert PALETTES["bing"][:4] == (
        "#4B4747",
        "#2E5EAA",
        "#E76F51",
        "#23BB62",
    )
    assert FIGURE_SIZES["paper_1col"] == (3.54, 2.76)
    assert FIGURE_SIZES["slides"] == (10.0, 5.625)
    assert PAIR_COLORS == {
        "CC": "#131313",
        "CH": "#E64B35",
        "HH": "#00A087",
    }
    assert MODEL_STYLES["ks_dft"]["linestyle"] == "-"
    assert MODEL_STYLES["tf"]["linestyle"] == "--"


def test_style_rcparams_resolves_profile_palette_and_size() -> None:
    params = style_rcparams(
        profile="paper",
        palette="science",
        figsize="paper_2col",
    )
    assert params["font.size"] == 8.0
    assert params["axes.labelsize"] == 9.0
    assert params["figure.figsize"] == FIGURE_SIZES["paper_2col"]
    assert params["axes.prop_cycle"].by_key()["color"] == list(
        PALETTES["science"]
    )
    assert params["pdf.fonttype"] == 42
    assert params["ps.fonttype"] == 42
    assert params["svg.fonttype"] == "none"
    assert params["text.usetex"] is False
    assert params["axes.grid"] is False


def test_thesis_profile_matches_gallery_typography() -> None:
    params = style_rcparams(profile="thesis", palette="bing")
    assert params["font.size"] == 11.0
    assert params["axes.labelsize"] == 13.0
    assert params["axes.titlesize"] == 13.0
    assert params["legend.fontsize"] == 10.0
    assert params["lines.linewidth"] == 1.8
    assert params["axes.prop_cycle"].by_key()["color"] == list(
        PALETTES["bing"]
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("profile", "poster"),
        ("palette", "rainbow"),
        ("figsize", "unknown"),
        ("figsize", (3.0,)),
        ("figsize", (-1.0, 2.0)),
    ),
)
def test_style_rcparams_rejects_invalid_inputs(
    keyword: str,
    value: object,
) -> None:
    arguments = {keyword: value}
    with pytest.raises(ValueError):
        style_rcparams(**arguments)


def test_style_context_restores_previous_rcparams() -> None:
    with matplotlib.rc_context({"font.size": 23.0}):
        assert matplotlib.rcParams["font.size"] == 23.0
        with style_context("paper", palette="nature", figsize="paper_1col"):
            assert matplotlib.rcParams["font.size"] == 8.0
            assert tuple(matplotlib.rcParams["figure.figsize"]) == (3.54, 2.76)
        assert matplotlib.rcParams["font.size"] == 23.0


def test_set_style_updates_current_context() -> None:
    with matplotlib.rc_context():
        set_style("docs", palette="deep_science", figsize=(6.0, 4.0))
        assert matplotlib.rcParams["font.size"] == 10.0
        assert tuple(matplotlib.rcParams["figure.figsize"]) == (6.0, 4.0)
        assert matplotlib.rcParams["axes.prop_cycle"].by_key()["color"] == list(
            PALETTES["deep_science"]
        )


def test_grid_figsize() -> None:
    assert grid_figsize(3, 2) == pytest.approx((7.2, 8.4))
    assert grid_figsize(
        2,
        4,
        cell_width=2.5,
        cell_height=3.0,
    ) == pytest.approx((10.0, 6.0))
    for arguments in ((0, 1), (1, 0), (1.5, 2), (True, 2)):
        with pytest.raises(ValueError):
            grid_figsize(*arguments)
    with pytest.raises(ValueError):
        grid_figsize(1, 1, cell_width=-1.0)


def test_add_panel_label_uses_style_title_size() -> None:
    with style_context("docs"):
        fig, ax = plt.subplots()
        artist = add_panel_label(ax, "(a)")
        assert artist.get_text() == "(a)"
        assert artist.get_fontweight() == "bold"
        assert artist.get_fontsize() == matplotlib.rcParams["axes.titlesize"]
        plt.close(fig)


def test_save_figure_writes_png_and_vector_pdf_without_closing(
    tmp_path: Path,
) -> None:
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [1.0, 0.0])
    figure_number = fig.number

    paths = save_figure(fig, tmp_path / "nested" / "result")

    assert paths == {
        "png": tmp_path / "nested" / "result.png",
        "pdf": tmp_path / "nested" / "result.pdf",
    }
    assert paths["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths["pdf"].read_bytes().startswith(b"%PDF")
    assert paths["png"].stat().st_size > 0
    assert paths["pdf"].stat().st_size > 0
    assert plt.fignum_exists(figure_number)
    plt.close(fig)


def test_save_figure_can_close_and_validates_options(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    figure_number = fig.number
    paths = save_figure(
        fig,
        tmp_path / "one.pdf",
        formats="pdf",
        tight=False,
        close=True,
    )
    assert paths == {"pdf": tmp_path / "one.pdf"}
    assert not plt.fignum_exists(figure_number)

    fig, _ = plt.subplots()
    with pytest.raises(ValueError):
        save_figure(fig, tmp_path / "bad", formats=())
    with pytest.raises(ValueError):
        save_figure(fig, tmp_path / "bad", formats=("png", "png"))
    with pytest.raises(ValueError):
        save_figure(fig, tmp_path / "bad", dpi=0)
    with pytest.raises(ValueError):
        save_figure(fig, tmp_path / "bad", pad_inches=-0.1)
    plt.close(fig)
