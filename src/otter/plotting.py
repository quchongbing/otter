"""Consistent Matplotlib styling for Otter figures.

Matplotlib is an optional Otter dependency.  This module therefore imports it
only inside functions, and :mod:`otter.__init__` deliberately does not import
this module.  Install ``otter[plot]`` before using these helpers.

The ``thesis`` profile is the project-wide default for the example and
benchmark galleries, academic slides, and publication-ready figures.  The
``docs`` and ``paper`` profiles provide progressively more compact typography.
All profiles use the same semantic colours and line conventions.

Examples
--------
Apply a temporary style without leaking global ``rcParams``:

>>> import matplotlib.pyplot as plt
>>> from otter.plotting import style_context
>>> with style_context("docs", figsize="docs"):
...     fig, ax = plt.subplots()
...     line = ax.plot([0.0, 1.0], [0.0, 1.0])

Save matching raster and vector copies:

>>> from otter.plotting import save_figure
>>> paths = save_figure(fig, "otter-result")  # doctest: +SKIP
>>> sorted(paths)  # doctest: +SKIP
['pdf', 'png']
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


# Colours are tuples rather than lists so callers cannot accidentally mutate
# the package defaults while customising one figure.
PALETTES: Mapping[str, tuple[str, ...]] = {
    "nature": (
        "#E64B35",
        "#4DBBD5",
        "#00A087",
        "#3C5488",
        "#F39B7F",
        "#8491B4",
        "#91D1C2",
        "#DC0000",
        "#7E6148",
    ),
    "science": (
        "#332288",
        "#117733",
        "#44AA99",
        "#88CCEE",
        "#DDCC77",
        "#CC6677",
        "#AA4499",
        "#882255",
    ),
    "deep_science": (
        "#131313",
        "#1F3A5F",
        "#2E5EAA",
        "#4C78A8",
        "#2A9D8F",
        "#3A7D44",
        "#6BAF92",
        "#E9C46A",
        "#F4A261",
        "#E76F51",
        "#C44E52",
        "#A05195",
        "#5C4D7D",
    ),
    # Muted thesis/slide palette adapted from the maintainer's standalone
    # ``figstyle`` package.  Keeping the values here avoids adding a plotting
    # dependency to Otter.
    "bing": (
        "#4B4747",
        "#2E5EAA",
        "#E76F51",
        "#23BB62",
        "#8B7355",
        "#6A5ACD",
        "#C0392B",
        "#3B6BC9",
    ),
}


# Physical sizes are in inches.  The paper widths follow common 90 mm and
# 180 mm one-/two-column layouts; the slides preset is a 16:9 canvas.
FIGURE_SIZES: Mapping[str, tuple[float, float]] = {
    "paper_1col": (3.54, 2.76),
    "paper_1col_square": (3.54, 3.54),
    "paper_2col": (7.09, 3.94),
    "paper_2col_square": (7.09, 5.00),
    "docs": (7.20, 4.50),
    "docs_wide": (10.50, 4.50),
    "slides": (10.00, 5.625),
    "square": (4.80, 4.80),
}


# Stable semantic colours make the same species/model recognisable across
# benchmarks.  These mappings are intentionally independent of the active
# palette and may be passed directly to ``Axes.plot``.
PAIR_COLORS: Mapping[str, str] = {
    "CC": "#131313",
    "CH": "#E64B35",
    "HH": "#00A087",
}

MODEL_STYLES: Mapping[str, Mapping[str, Any]] = {
    "otter": {"color": "#131313", "linestyle": "-", "linewidth": 1.8},
    "reference": {
        "color": "#6B6B6B",
        "linestyle": "none",
        "marker": "o",
        "markerfacecolor": "none",
    },
    "ks_dft": {"color": "#2E5EAA", "linestyle": "-", "linewidth": 1.6},
    "qm": {"color": "#2E5EAA", "linestyle": "-", "linewidth": 1.6},
    "tf": {"color": "#E76F51", "linestyle": "--", "linewidth": 1.6},
    "is": {"color": "#2E5EAA", "linestyle": "-", "linewidth": 1.6},
    "sc": {"color": "#E76F51", "linestyle": "--", "linewidth": 1.6},
}


_PROFILES: Mapping[str, Mapping[str, float]] = {
    "thesis": {
        "font_size": 14.0,
        "label_size": 16.0,
        "title_size": 16.0,
        "tick_size": 13.0,
        "legend_size": 13.0,
        "axes_linewidth": 1.0,
        "line_width": 1.8,
        "marker_size": 5.0,
        "tick_major": 5.0,
        "tick_minor": 2.5,
    },
    "docs": {
        "font_size": 12.0,
        "label_size": 14.0,
        "title_size": 14.0,
        "tick_size": 11.0,
        "legend_size": 11.0,
        "axes_linewidth": 0.9,
        "line_width": 1.6,
        "marker_size": 5.5,
        "tick_major": 4.5,
        "tick_minor": 2.25,
    },
    "paper": {
        "font_size": 9.0,
        "label_size": 10.0,
        "title_size": 10.0,
        "tick_size": 9.0,
        "legend_size": 9.0,
        "axes_linewidth": 0.8,
        "line_width": 1.3,
        "marker_size": 4.5,
        "tick_major": 4.0,
        "tick_minor": 2.0,
    },
}


def _matplotlib():
    """Import Matplotlib with an actionable optional-dependency error."""
    try:
        import matplotlib as mpl
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise ModuleNotFoundError(
            "Otter plotting helpers require Matplotlib. "
            "Install them with `pip install 'otter[plot]'`."
        ) from exc
    return mpl


def _resolve_figsize(
    figsize: str | Sequence[float] | None,
) -> tuple[float, float] | None:
    if figsize is None:
        return None
    if isinstance(figsize, str):
        try:
            return FIGURE_SIZES[figsize]
        except KeyError as exc:
            choices = ", ".join(sorted(FIGURE_SIZES))
            raise ValueError(
                f"Unknown figure-size preset {figsize!r}; choose from {choices}."
            ) from exc
    if len(figsize) != 2:
        raise ValueError("A custom figure size must contain exactly two values.")
    width, height = (float(value) for value in figsize)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Figure width and height must be positive.")
    return width, height


def style_rcparams(
    profile: str = "docs",
    palette: str = "nature",
    figsize: str | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Return Otter's Matplotlib ``rcParams`` without applying them.

    Parameters
    ----------
    profile
        ``"thesis"`` for publication-quality gallery/slide figures,
        ``"docs"`` for compact screen figures, or ``"paper"`` for journal
        figures.
    palette
        One of the keys in :data:`PALETTES`.
    figsize
        A key in :data:`FIGURE_SIZES`, a ``(width, height)`` pair in inches,
        or ``None`` to leave the current default figure size unchanged.
    """
    try:
        profile_values = _PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"Unknown plotting profile {profile!r}; choose from {choices}."
        ) from exc
    try:
        colors = PALETTES[palette]
    except KeyError as exc:
        choices = ", ".join(sorted(PALETTES))
        raise ValueError(
            f"Unknown colour palette {palette!r}; choose from {choices}."
        ) from exc

    mpl = _matplotlib()
    axes_linewidth = profile_values["axes_linewidth"]
    params: dict[str, Any] = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": profile_values["font_size"],
        "axes.labelsize": profile_values["label_size"],
        "axes.titlesize": profile_values["title_size"],
        "xtick.labelsize": profile_values["tick_size"],
        "ytick.labelsize": profile_values["tick_size"],
        "legend.fontsize": profile_values["legend_size"],
        "axes.linewidth": axes_linewidth,
        "axes.edgecolor": "#333333",
        "axes.labelpad": 4.0,
        "axes.axisbelow": True,
        "axes.grid": False,
        "axes.prop_cycle": mpl.cycler(color=list(colors)),
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": profile_values["tick_major"],
        "ytick.major.size": profile_values["tick_major"],
        "xtick.minor.size": profile_values["tick_minor"],
        "ytick.minor.size": profile_values["tick_minor"],
        "xtick.major.width": axes_linewidth,
        "ytick.major.width": axes_linewidth,
        "xtick.minor.width": 0.75 * axes_linewidth,
        "ytick.minor.width": 0.75 * axes_linewidth,
        "lines.linewidth": profile_values["line_width"],
        "lines.markersize": profile_values["marker_size"],
        "patch.linewidth": 0.6,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.35,
        "figure.dpi": 110.0,
        "savefig.dpi": 300.0,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
    }
    resolved_figsize = _resolve_figsize(figsize)
    if resolved_figsize is not None:
        params["figure.figsize"] = resolved_figsize
    return params


@contextmanager
def style_context(
    profile: str = "docs",
    palette: str = "nature",
    figsize: str | Sequence[float] | None = None,
) -> Iterator[None]:
    """Temporarily apply the Otter style and restore prior ``rcParams``."""
    mpl = _matplotlib()
    with mpl.rc_context(rc=style_rcparams(profile, palette, figsize)):
        yield


def set_style(
    profile: str = "docs",
    palette: str = "nature",
    figsize: str | Sequence[float] | None = None,
) -> None:
    """Apply the Otter style globally to subsequent Matplotlib figures.

    Library and Sphinx-Gallery code should generally prefer
    :func:`style_context`, which cannot leak settings to unrelated figures.
    """
    mpl = _matplotlib()
    mpl.rcParams.update(style_rcparams(profile, palette, figsize))


def grid_figsize(
    nrows: int,
    ncols: int,
    *,
    cell_width: float | None = None,
    cell_height: float | None = None,
) -> tuple[float, float]:
    """Return the project-standard figure size for a panel grid.

    Standard panel sizes depend on the grid shape.  A one-panel figure is 6.6
    by 4.2 inches; two-column layouts use 5.1 by 3.8 inches per panel; wider
    grids use progressively narrower panels.  Grids with three or more rows
    use compact row heights.  Thus every public figure with the same grid
    shape has the same canvas size.

    ``cell_width`` and ``cell_height`` may override either standard dimension
    for a diagnostic whose aspect ratio carries scientific information.
    """
    if (
        not isinstance(nrows, int)
        or isinstance(nrows, bool)
        or nrows <= 0
        or not isinstance(ncols, int)
        or isinstance(ncols, bool)
        or ncols <= 0
    ):
        raise ValueError("nrows and ncols must be positive integers.")
    standard_cells = {
        1: (6.6, 4.2),
        2: (5.1, 3.8),
        3: (4.0, 3.5),
        4: (3.5, 3.2),
    }
    standard_width, standard_height = standard_cells.get(ncols, (3.2, 3.0))
    if nrows >= 3:
        # Repeated rows normally share titles, labels, and legends.  Compact
        # rows keep temperature/density scans readable on a gallery page.
        compact_height = 2.4 if ncols <= 2 else 2.8
        standard_height = min(standard_height, compact_height)
    width = standard_width if cell_width is None else float(cell_width)
    height = standard_height if cell_height is None else float(cell_height)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Panel width and height must be positive.")
    return float(ncols * width), float(nrows * height)


def add_panel_label(
    ax: Any,
    label: str,
    *,
    x: float = -0.12,
    y: float = 1.03,
    fontsize: float | None = None,
    **kwargs: Any,
) -> Any:
    """Add a bold panel label such as ``"(a)"`` and return the text artist."""
    mpl = _matplotlib()
    return ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize or mpl.rcParams["axes.titlesize"],
        fontweight="bold",
        verticalalignment="bottom",
        horizontalalignment="left",
        **kwargs,
    )


def save_figure(
    fig: Any,
    output_stem: str | Path,
    *,
    formats: Sequence[str] | str = ("png", "pdf"),
    dpi: int = 300,
    tight: bool = True,
    pad_inches: float = 0.03,
    close: bool = False,
    **kwargs: Any,
) -> dict[str, Path]:
    """Save matching raster and vector copies of a figure.

    Parameters
    ----------
    fig
        A Matplotlib figure.
    output_stem
        Destination without a suffix.  A trailing ``.png`` or ``.pdf`` is
        accepted and removed before all requested formats are written.
    formats
        Output suffixes.  The default writes a 300 dpi PNG and a vector PDF.
    dpi
        Raster resolution.  Matplotlib also uses this value for any rasterized
        artists embedded in the PDF; vector lines and text remain vector.
    tight
        Use Matplotlib's tight bounding-box calculation.  Set this to ``False``
        when exact physical dimensions are more important than trimming.
    pad_inches
        Padding used with a tight bounding box.
    close
        Close the figure after every requested file has been written.

    Returns
    -------
    dict
        Mapping from lowercase format name to the written path.
    """
    if isinstance(formats, str):
        format_values = (formats,)
    else:
        format_values = tuple(formats)
    normalized_formats = tuple(
        str(value).lower().removeprefix(".") for value in format_values
    )
    if not normalized_formats or any(not value for value in normalized_formats):
        raise ValueError("At least one non-empty output format is required.")
    if len(set(normalized_formats)) != len(normalized_formats):
        raise ValueError("Output formats must be unique.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    if pad_inches < 0.0:
        raise ValueError("pad_inches cannot be negative.")

    stem = Path(output_stem).expanduser()
    if stem.suffix.lower() in {".png", ".pdf"}:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    save_options = dict(kwargs)
    save_options.setdefault("dpi", dpi)
    save_options.setdefault("facecolor", "white")
    save_options.setdefault("transparent", False)
    if tight:
        save_options.setdefault("bbox_inches", "tight")
        save_options.setdefault("pad_inches", pad_inches)

    paths: dict[str, Path] = {}
    for output_format in normalized_formats:
        path = stem.with_suffix(f".{output_format}")
        fig.savefig(path, format=output_format, **save_options)
        paths[output_format] = path

    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return paths


__all__ = [
    "FIGURE_SIZES",
    "MODEL_STYLES",
    "PAIR_COLORS",
    "PALETTES",
    "add_panel_label",
    "grid_figsize",
    "save_figure",
    "set_style",
    "style_context",
    "style_rcparams",
]
