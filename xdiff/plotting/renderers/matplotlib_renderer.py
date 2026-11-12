"""Static image renderer: one full-size difference map (or line) per variable.

Headless by construction — the matplotlib ``Agg`` backend is selected before ``pyplot``
is imported, so PNG/PDF/SVG rendering needs no display (CI, remote login nodes). The
heavy import is lazy and raises a clear install hint when the ``plot`` extra is missing.

There is no map projection or external coastline layer (no cartopy): land is drawn from
the data's own sea mask (where either file has data), together with a coastline traced
from that mask, so it matches the model grid exactly and needs no download. The axes use
a latitude-corrected aspect so the domain is not distorted.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from xdiff.plotting.renderers import DIFF_CMAP

if TYPE_CHECKING:
    from xdiff.plotting.spec import PlotSpec, VariablePlot

SUPPORTED_EXTENSIONS = (".png", ".pdf", ".svg")


def validate_output_extension(output: Path) -> None:
    """Raise ``ValueError`` unless ``output`` has a supported image extension."""
    if output.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValueError(f"unsupported output extension {output.suffix or '(none)'!r}; use one of {supported}")


def output_paths(output: Path, labels: list[str]) -> list[Path]:
    """Map a base ``-o`` path to one path per variable.

    A single variable writes exactly ``output``; N variables insert a filesystem-safe
    form of each label into the stem (``diff.png`` -> ``diff_thetao.png``, ...).
    """
    if len(labels) == 1:
        return [output]
    return [output.with_name(f"{output.stem}_{_slugify(label)}{output.suffix}") for label in labels]


SCALES = ("auto", "linear", "log")

# A signed log scale is used when the typical difference is this many times smaller than
# the colour limit a linear scale would need (99th percentile over the median of the
# non-zero |difference|). On MedFS fields: sea-surface height 60 (linear reads well),
# temperature, salinity, fluxes 400 to 150000 (linear hides almost every difference).
_LOG_SCALE_RATIO = 100.0
_LOG_DECADES = 6  # a signed log scale spans this many decades below the largest difference

_LAND = "#d9d6cf"
_COAST = "#4a4a4a"
_MUTED_TEXT = "0.3"


def render_to_files(spec: PlotSpec, output: Path, *, scale: str = "auto") -> list[Path]:
    """Write one difference map (or line) per variable in ``spec``; return the written paths.

    ``scale`` is ``"linear"``, ``"log"`` (signed log) or ``"auto"`` (chosen per variable by
    :func:`choose_scale`).
    """
    validate_output_extension(output)
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; use one of {', '.join(SCALES)}")
    if not spec.variables:
        detail = ""
        if spec.skipped:
            detail = "; skipped: " + ", ".join(f"{item.label} ({item.reason})" for item in spec.skipped)
        raise ValueError(f"no plottable variables found{detail}")

    plt = _load_pyplot()
    targets = output_paths(output, [variable.label for variable in spec.variables])
    files = f"{Path(spec.reference_path).name}  vs  {Path(spec.comparison_path).name}"

    written: list[Path] = []
    for variable, target in zip(spec.variables, targets, strict=True):
        figure = _render_variable(plt, variable, scale=scale, files=files)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, bbox_inches="tight", pad_inches=0.12)
        plt.close(figure)
        written.append(target)
    return written


def choose_scale(difference) -> str:
    """``"log"`` when the differences span too many orders of magnitude for a linear scale.

    A linear colour limit is set by the largest differences; when the typical (median)
    non-zero difference is more than ``_LOG_SCALE_RATIO`` times smaller than that limit,
    almost every differing point would be drawn as zero, so a signed log scale is used.
    """
    nonzero = _nonzero_magnitudes(difference)
    if nonzero.size == 0:
        return "linear"
    median = float(np.median(nonzero))
    return "log" if float(np.percentile(nonzero, 99)) > _LOG_SCALE_RATIO * median else "linear"


def scaled_label(limit: float, units: str | None) -> tuple[float, str]:
    """A power of ten to divide values by, and the colorbar label that names it.

    Keeps tick labels short (``2``, ``4``, ``6``) instead of ``0.000002`` or a detached
    ``1e-6`` multiplier: the factor goes into the label, ``difference [×10⁻⁶ m]``.
    """
    exponent = int(np.floor(np.log10(limit))) if limit > 0 else 0
    if -2 <= exponent <= 3:
        return 1.0, f"difference [{units}]" if units else "difference"
    unit_text = f" {units}" if units else ""
    return 10.0**exponent, f"difference [×10$^{{{exponent}}}${unit_text}]"


def _load_pyplot():
    """Import matplotlib with the headless Agg backend, lazily and with a clear hint."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "matplotlib is required for static plots. Install the plot extra with "
            '`uv sync --extra plot` or `uv tool install "xdiffly[plot]"`.'
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _render_variable(plt, variable: VariablePlot, *, scale: str = "auto", files: str = ""):
    """Render only the difference — a full-size map (2-D) or line (1-D).

    Static output (``-o``) is for reports: the difference is the point, so it gets the
    whole figure rather than one third of a reference|comparison triptych.
    """
    if len(variable.dims) == 2:
        return _render_diff_map(plt, variable, scale=scale, files=files)
    return _render_diff_line(plt, variable, files=files)


def _render_diff_map(plt, variable: VariablePlot, *, scale: str = "auto", files: str = ""):
    """A full-size difference map: flat cells, land and coastline from the sea mask.

    Land is wherever neither file has data: those cells are left undrawn over a land-coloured
    background, and a coastline is traced along the edge of the sea mask. The colour scale is
    linear (limit from the points that differ) or signed log; see :func:`choose_scale`.
    """
    import matplotlib

    from xdiff.plotting.spec import valid_extent

    difference = np.asarray(variable.difference, dtype=float)
    nonzero = _nonzero_magnitudes(difference)
    chosen = choose_scale(difference) if scale == "auto" else scale
    colour = _colour_scale(difference, nonzero, chosen, variable.units)
    extent = valid_extent(variable)

    figure = plt.figure(figsize=_figure_size(extent), dpi=150, layout="constrained")
    axis = figure.add_subplot()
    axis.set_facecolor(_LAND)
    cmap = matplotlib.colormaps[DIFF_CMAP].with_extremes(bad=(0.0, 0.0, 0.0, 0.0))  # land shows through
    mesh = _mesh(axis, colour["values"], variable.lon, variable.lat, cmap=cmap, norm=colour["norm"])
    sea = np.isfinite(np.asarray(variable.reference, dtype=float)) | np.isfinite(
        np.asarray(variable.comparison, dtype=float)
    )
    _coastline(axis, sea, variable.lon, variable.lat)
    _frame(axis, variable, extent)

    _heading(figure, axis, variable)

    colorbar = figure.colorbar(
        mesh,
        ax=axis,
        orientation="horizontal",
        location="bottom",
        fraction=0.05,
        pad=0.02,
        aspect=45,
        shrink=0.75,
        extend=colour["extend"],
        label=colour["label"],
    )
    colorbar.ax.tick_params(labelsize=9)
    colorbar.outline.set_linewidth(0.6)
    if colour["ticks"] is not None:
        colorbar.set_ticks(colour["ticks"])
        colorbar.set_ticklabels(colour["ticklabels"])
    figure.text(
        0.5,
        -0.005,
        _footer(difference, nonzero, variable.units, files),
        ha="center",
        va="top",
        fontsize=8.5,
        color=_MUTED_TEXT,
    )
    return figure


def _heading(figure, axis, variable: VariablePlot) -> None:
    """Two left-aligned lines above the axes: the variable in bold, then what is shown and when.

    Stacked rather than split left/right, so a long name never runs into the date and depth.
    """
    from matplotlib.transforms import offset_copy

    name = _base_name(variable.label)
    title = f"{variable.long_name} ({name})" if variable.long_name else name
    subtitle = "   ·   ".join(("reference minus comparison", *variable.context))
    axis.set_title(subtitle, loc="left", fontsize=10, color=_MUTED_TEXT, pad=6)
    above_subtitle = offset_copy(axis.transAxes, fig=figure, y=22, units="points")
    axis.text(0.0, 1.0, title, transform=above_subtitle, ha="left", va="bottom", fontsize=12.5, fontweight="bold")


def _nonzero_magnitudes(difference) -> np.ndarray:
    magnitudes = np.abs(np.asarray(difference, dtype=float))
    magnitudes = magnitudes[np.isfinite(magnitudes)]
    return magnitudes[magnitudes > 0]


def _colour_scale(difference: np.ndarray, nonzero: np.ndarray, scale: str, units: str | None) -> dict:
    """Norm, values to draw, colorbar label/ticks and extension for a linear or signed log scale."""
    import matplotlib.colors as mcolors

    if scale == "log" and nonzero.size:
        top = int(np.ceil(np.log10(nonzero.max())))
        low = max(top - _LOG_DECADES, int(np.floor(np.log10(nonzero.min()))))
        low = min(low, top - 1)
        norm = mcolors.SymLogNorm(linthresh=10.0**low, linscale=0.4, vmin=-(10.0**top), vmax=10.0**top, base=10)
        step = 2 if top - low > 4 else 1
        decades = list(range(top, low, -step))
        ticks = [-(10.0**k) for k in decades] + [0.0] + [10.0**k for k in reversed(decades)]
        ticklabels = ["0" if t == 0 else ("−" if t < 0 else "") + f"10$^{{{round(np.log10(abs(t)))}}}$" for t in ticks]
        unit_text = f" [{units}]" if units else ""
        label = f"difference{unit_text}, signed log scale (|d| < 1e{low} shown as 0)"
        return {
            "norm": norm,
            "values": difference,
            "label": label,
            "extend": "neither",
            "ticks": ticks,
            "ticklabels": ticklabels,
        }

    limit = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0
    factor, label = scaled_label(limit, units)
    norm = mcolors.Normalize(vmin=-limit / factor, vmax=limit / factor)
    extend = "both" if nonzero.size and nonzero.max() > limit else "neither"
    return {
        "norm": norm,
        "values": difference / factor,
        "label": label,
        "extend": extend,
        "ticks": None,
        "ticklabels": None,
    }


def _mesh(axis, values, lon, lat, **kwargs):
    """Draw a 2-D field as flat cells, on lon/lat axes when their shapes line up.

    Masked (NaN) cells are not drawn, so the land-coloured background shows through. Falls
    back to an index-axis image when coordinates are absent or do not match the field.
    """
    masked = np.ma.masked_invalid(np.asarray(values, dtype=float))
    if lon is not None and lat is not None:
        try:
            if lon.ndim == 1 and lat.ndim == 1:
                if masked.shape == (lat.size, lon.size):
                    return axis.pcolormesh(lon, lat, masked, shading="nearest", rasterized=True, **kwargs)
                if masked.shape == (lon.size, lat.size):
                    return axis.pcolormesh(lon, lat, masked.T, shading="nearest", rasterized=True, **kwargs)
            elif lon.shape == masked.shape and lat.shape == masked.shape:
                return axis.pcolormesh(lon, lat, masked, shading="nearest", rasterized=True, **kwargs)
        except Exception:  # noqa: BLE001 - fall back to a plain image on any plotting quirk
            pass
    return axis.imshow(masked, origin="lower", aspect="auto", interpolation="nearest", **kwargs)


def _coastline(axis, sea: np.ndarray, lon, lat) -> None:
    """Trace the edge of the sea mask as a coastline (1-D axes or index axes only).

    2-D coordinates can carry fill values on land, which would drag the contour across the
    map, so a curvilinear grid gets the land colour without the traced line.
    """
    if sea.all() or not sea.any():
        return
    mask = sea.astype(float)
    if lon is None or lat is None:
        axis.contour(mask, levels=[0.5], colors=_COAST, linewidths=0.45)
    elif lon.ndim == 1 and lat.ndim == 1:
        if mask.shape == (lon.size, lat.size) and mask.shape != (lat.size, lon.size):
            mask = mask.T
        if mask.shape == (lat.size, lon.size):
            axis.contour(lon, lat, mask, levels=[0.5], colors=_COAST, linewidths=0.45)


def _frame(axis, variable: VariablePlot, extent) -> None:
    """Crop to the data, set the latitude-corrected aspect, degree ticks and gridlines."""
    import matplotlib.ticker as mticker

    if extent is None:
        axis.set_xlabel(variable.dims[1])
        axis.set_ylabel(variable.dims[0])
        return
    lon_min, lon_max, lat_min, lat_max = extent
    axis.set_xlim(lon_min, lon_max)
    axis.set_ylim(lat_min, lat_max)
    axis.set_aspect(_latitude_aspect((lat_min + lat_max) / 2.0))
    axis.xaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: _degrees(value, "E", "W")))
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _pos: _degrees(value, "N", "S")))
    axis.grid(color="0.35", linewidth=0.3, linestyle=(0, (2, 3)), alpha=0.6)
    axis.tick_params(labelsize=9, direction="out", length=3)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def _degrees(value: float, positive: str, negative: str) -> str:
    hemisphere = positive if value > 0 else (negative if value < 0 else "")
    return f"{abs(value):g}°{hemisphere}"


def _footer(difference: np.ndarray, nonzero: np.ndarray, units: str | None, files: str) -> str:
    finite = difference[np.isfinite(difference)]
    parts = []
    if nonzero.size:
        unit_text = f" {units}" if units else ""
        rms = float(np.sqrt(np.mean(finite**2)))
        parts.append(f"min {finite.min():.3g}   max {finite.max():.3g}   RMS {rms:.3g}{unit_text}")
        parts.append(f"{nonzero.size / max(finite.size, 1):.1%} of sea points differ")
    else:
        parts.append("identical: no point differs")
    if files:
        parts.append(files)
    return "   ·   ".join(parts)


def _base_name(label: str) -> str:
    """The variable part of a label, without the ``[dim=index]`` decoration."""
    return label.split(" [", 1)[0]


def _latitude_aspect(mean_lat: float) -> float:
    """Axes y/x aspect so 1° lon and 1° lat are drawn to physical scale (equirectangular)."""
    return 1.0 / max(np.cos(np.deg2rad(mean_lat)), 0.1)


def _figure_size(extent) -> tuple[float, float]:
    """Figure size matching the domain's latitude-corrected aspect, plus room for the text."""
    if extent is None:
        return (10.0, 6.0)
    lon_min, lon_max, lat_min, lat_max = extent
    span_x = max(lon_max - lon_min, 1e-6) * np.cos(np.deg2rad((lat_min + lat_max) / 2.0))
    span_y = max(lat_max - lat_min, 1e-6)
    width = 10.0
    height = min(max(width * span_y / span_x + 2.1, 3.5), 12.0)  # +room for title/colorbar/footer
    return (width, height)


def _render_diff_line(plt, variable: VariablePlot, *, files: str = ""):
    figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    axis.axhline(0.0, color="0.7", linewidth=0.8)
    axis.plot(_axis_1d(variable), variable.difference, color="tab:red")
    _heading(figure, axis, variable)
    axis.set_ylabel(f"difference [{variable.units}]" if variable.units else "difference")
    axis.grid(color="0.35", linewidth=0.3, linestyle=(0, (2, 3)), alpha=0.6)
    if files:
        figure.text(0.5, -0.005, files, ha="center", va="top", fontsize=8.5, color=_MUTED_TEXT)
    return figure


def _axis_1d(variable: VariablePlot) -> np.ndarray:
    length = variable.reference.shape[0]
    for coordinate in (variable.lon, variable.lat):
        if coordinate is not None and coordinate.ndim == 1 and coordinate.size == length:
            return coordinate
    return np.arange(length)


def _slugify(label: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")
    return slug or "var"
