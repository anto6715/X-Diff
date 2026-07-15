"""Static image renderer: one triptych (reference | comparison | difference) per variable.

Headless by construction — the matplotlib ``Agg`` backend is selected before ``pyplot``
is imported, so PNG/PDF/SVG rendering needs no display (CI, remote login nodes). The
heavy import is lazy and raises a clear install hint when the ``plot`` extra is missing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xdiff.plotting.spec import PlotSpec, VariablePlot

logger = logging.getLogger("xdiff")

SUPPORTED_EXTENSIONS = (".png", ".pdf", ".svg")

_DIFF_CMAP = "RdBu_r"
# NaN cells (land / masked) are painted this neutral grey when there is no coastline map.
# On the diverging colormap white already means "no difference", so leaving NaN white would
# be ambiguous. Matches the interactive server's land colour.
_LAND_COLOR = "#b0b0b0"
# When cartopy is available we instead draw a filled land feature in this colour (matching
# the mtplot look) and let the data NaNs fall through transparently.
_LAND_FILL = "#e8e6d8"
_COASTLINE_RESOLUTION = "10m"


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


def render_to_files(spec: PlotSpec, output: Path) -> list[Path]:
    """Write one static triptych per variable in ``spec``; return the written paths."""
    validate_output_extension(output)
    if not spec.variables:
        detail = ""
        if spec.skipped:
            detail = "; skipped: " + ", ".join(f"{item.label} ({item.reason})" for item in spec.skipped)
        raise ValueError(f"no plottable variables found{detail}")

    plt = _load_pyplot()
    targets = output_paths(output, [variable.label for variable in spec.variables])

    written: list[Path] = []
    for variable, target in zip(spec.variables, targets, strict=True):
        figure = _render_variable(plt, variable)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, bbox_inches="tight", dpi=150)
        plt.close(figure)
        written.append(target)
    return written


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


def _render_variable(plt, variable: VariablePlot):
    """Render only the difference — a full-size map (2-D) or line (1-D).

    Static output (``-o``) is for reports: the difference is the point, so we give it the
    whole figure rather than cramming it into one third of a reference|comparison triptych.
    """
    if len(variable.dims) == 2:
        return _render_diff_map(plt, variable)
    return _render_diff_line(plt, variable)


def _render_diff_map(plt, variable: VariablePlot):
    """Difference map: a cartopy map (coastlines + filled land) when available, else plain.

    The cartopy path is wrapped in a fallback so a missing coastline dataset (e.g. an
    offline login node that never downloaded Natural Earth) degrades to the plain
    lon/lat pcolormesh instead of failing the whole plot.
    """
    from xdiff.plotting.spec import valid_extent

    geo = _try_cartopy()
    if geo is not None and variable.lon is not None and variable.lat is not None:
        try:
            return _render_diff_map_geo(plt, variable, *geo, extent=valid_extent(variable))
        except Exception as exc:  # noqa: BLE001 - coastline data missing, etc.
            logger.warning("Falling back to a plain map (cartopy rendering failed): %s", exc)
    return _render_diff_map_plain(plt, variable)


def _render_diff_map_geo(plt, variable: VariablePlot, ccrs, cfeature, *, extent):
    figure = plt.figure(figsize=(11, 6))
    axis = figure.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    if extent is not None:
        axis.set_extent(list(extent), crs=ccrs.PlateCarree())

    mappable = _pcolor(
        axis,
        variable.difference,
        variable.lon,
        variable.lat,
        cmap=_diverging_cmap_transparent(),
        vmin=-variable.diff_limit,
        vmax=variable.diff_limit,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    axis.add_feature(cfeature.LAND, facecolor=_LAND_FILL, zorder=2)
    axis.coastlines(resolution=_COASTLINE_RESOLUTION, color="black", linewidth=0.6, zorder=3)
    gridlines = axis.gridlines(draw_labels=True, linewidth=1, color="grey", alpha=0.5, linestyle="--")
    gridlines.top_labels = False
    gridlines.right_labels = False

    colorbar = figure.colorbar(mappable, ax=axis, orientation="vertical", aspect=30, fraction=0.046, extend="both")
    colorbar.ax.set_title(_value_label(variable), fontsize=10)
    figure.suptitle(f"{variable.label} — difference", fontsize=15, y=0.98)
    axis.set_title(f"clipped ±{variable.diff_limit:.3g}; true ±{variable.diff_extreme:.3g}", fontsize=10)
    figure.text(0.5, 0.02, _minmax_caption(variable), ha="center", fontsize=9)
    return figure


def _render_diff_map_plain(plt, variable: VariablePlot):
    figure, axis = plt.subplots(figsize=(8, 6.5))
    mappable = _pcolor(
        axis,
        variable.difference,
        variable.lon,
        variable.lat,
        cmap=_DIFF_CMAP,
        vmin=-variable.diff_limit,
        vmax=variable.diff_limit,
    )
    axis.set_title(
        f"{variable.label} — difference\nclipped ±{variable.diff_limit:.3g}; true ±{variable.diff_extreme:.3g}"
    )
    figure.colorbar(mappable, ax=axis, label=_value_label(variable))
    figure.tight_layout()
    return figure


def _try_cartopy():
    """Return (cartopy.crs, cartopy.feature) if importable, else None."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        return ccrs, cfeature
    except Exception:  # noqa: BLE001 - cartopy absent or broken -> plain map
        return None


def _diverging_cmap_transparent():
    """The diff colormap with NaN rendered transparent (so the land feature shows through)."""
    import matplotlib

    return matplotlib.colormaps[_DIFF_CMAP].with_extremes(bad=(0.0, 0.0, 0.0, 0.0))


def _minmax_caption(variable: VariablePlot) -> str:
    difference = np.asarray(variable.difference, dtype=float)
    return f"min: {np.nanmin(difference):.2e}   max: {np.nanmax(difference):.2e}"


def _render_diff_line(plt, variable: VariablePlot):
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.axhline(0.0, color="0.7", linewidth=0.8)
    axis.plot(_axis_1d(variable), variable.difference, color="tab:red")
    axis.set_title(f"{variable.label} — difference (true ±{variable.diff_extreme:.3g})")
    axis.set_ylabel(_value_label(variable))
    figure.tight_layout()
    return figure


def _pcolor(axis, values, lon, lat, **kwargs):
    """Draw a 2-D field, using lon/lat for axes when their shapes line up.

    Falls back to an index-axis image when coordinates are absent or their shapes do
    not match the field (so an orientation quirk never aborts the render). NaN cells
    (land / masked) render as neutral grey in both paths.
    """
    values = np.asarray(values, dtype=float)
    if isinstance(kwargs.get("cmap"), str):
        import matplotlib

        kwargs["cmap"] = matplotlib.colormaps[kwargs["cmap"]].with_extremes(bad=_LAND_COLOR)
    if lon is not None and lat is not None:
        try:
            if lon.ndim == 1 and lat.ndim == 1:
                if values.shape == (lat.size, lon.size):
                    return axis.pcolormesh(lon, lat, values, **kwargs)
                if values.shape == (lon.size, lat.size):
                    return axis.pcolormesh(lon, lat, values.T, **kwargs)
            elif lon.shape == values.shape and lat.shape == values.shape:
                return axis.pcolormesh(lon, lat, values, **kwargs)
        except Exception:  # noqa: BLE001 - fall back to a plain image on any plotting quirk
            pass
    return axis.imshow(values, origin="lower", aspect="auto", **kwargs)


def _axis_1d(variable: VariablePlot) -> np.ndarray:
    length = variable.reference.shape[0]
    for coordinate in (variable.lon, variable.lat):
        if coordinate is not None and coordinate.ndim == 1 and coordinate.size == length:
            return coordinate
    return np.arange(length)


def _value_label(variable: VariablePlot) -> str:
    return variable.units if variable.units else "value"


def _slugify(label: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")
    return slug or "var"
