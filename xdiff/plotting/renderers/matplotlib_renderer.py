"""Static image renderer: one full-size difference map (or line) per variable.

Headless by construction — the matplotlib ``Agg`` backend is selected before ``pyplot``
is imported, so PNG/PDF/SVG rendering needs no display (CI, remote login nodes). The
heavy import is lazy and raises a clear install hint when the ``plot`` extra is missing.
There is no map projection or coastline layer (no cartopy): the data's own NaN mask draws
the land, and the axes use a latitude-corrected aspect so the domain is not distorted.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xdiff.plotting.spec import PlotSpec, VariablePlot

SUPPORTED_EXTENSIONS = (".png", ".pdf", ".svg")

_DIFF_CMAP = "RdBu_r"
# NaN cells (land / masked) are painted this neutral grey. On the diverging colormap white
# already means "no difference", so leaving NaN white would be ambiguous. Matches the server.
_LAND_COLOR = "#b0b0b0"


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
    """A full-size difference map: smooth (gouraud) pcolormesh, cropped to the data domain.

    Plain lon/lat axes (no projection). The figure is sized to the domain's latitude-corrected
    aspect and the axes use that aspect, so a wide-short basin (e.g. the Mediterranean) is drawn
    in proportion instead of stretched into a square. ``bbox_inches='tight'`` on save trims the
    surrounding whitespace.
    """
    from xdiff.plotting.spec import valid_extent

    extent = valid_extent(variable)
    figure, axis = plt.subplots(figsize=_figure_size(extent))
    mappable = _pcolor(
        axis,
        variable.difference,
        variable.lon,
        variable.lat,
        cmap=_DIFF_CMAP,
        vmin=-variable.diff_limit,
        vmax=variable.diff_limit,
    )
    if extent is not None:
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect(_latitude_aspect((extent[2] + extent[3]) / 2.0))
        axis.set_xlabel("lon")
        axis.set_ylabel("lat")
    axis.set_title(
        f"{variable.label} — difference\nclipped ±{variable.diff_limit:.3g}; true ±{variable.diff_extreme:.3g}"
    )
    figure.colorbar(mappable, ax=axis, label=_value_label(variable), extend="both", fraction=0.046, pad=0.02)
    figure.text(0.5, 0.01, _minmax_caption(variable), ha="center", fontsize=9)
    figure.tight_layout()
    return figure


def _latitude_aspect(mean_lat: float) -> float:
    """Axes y/x aspect so 1° lon and 1° lat are drawn to physical scale (equirectangular)."""
    return 1.0 / max(np.cos(np.deg2rad(mean_lat)), 0.1)


def _figure_size(extent) -> tuple[float, float]:
    """Figure size matching the domain's latitude-corrected aspect (extra height for labels)."""
    if extent is None:
        return (9.0, 6.0)
    lon_min, lon_max, lat_min, lat_max = extent
    span_x = max(lon_max - lon_min, 1e-6) * np.cos(np.deg2rad((lat_min + lat_max) / 2.0))
    span_y = max(lat_max - lat_min, 1e-6)
    width = 9.0
    height = min(max(width * span_y / span_x + 1.6, 3.5), 11.0)  # +room for title/colorbar/caption
    return (width, height)


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
    """Draw a 2-D field smoothly, using lon/lat for axes when their shapes line up.

    ``shading='gouraud'`` interpolates between cell centres (the smooth look, matching the
    server's linear-interpolated maps). Falls back to a bilinear index-axis image when
    coordinates are absent or their shapes do not match the field (so an orientation quirk
    never aborts the render). NaN cells (land / masked) render as neutral grey in both paths.
    """
    values = np.asarray(values, dtype=float)
    if isinstance(kwargs.get("cmap"), str):
        import matplotlib

        kwargs["cmap"] = matplotlib.colormaps[kwargs["cmap"]].with_extremes(bad=_LAND_COLOR)
    if lon is not None and lat is not None:
        try:
            if lon.ndim == 1 and lat.ndim == 1:
                if values.shape == (lat.size, lon.size):
                    return axis.pcolormesh(lon, lat, values, shading="gouraud", **kwargs)
                if values.shape == (lon.size, lat.size):
                    return axis.pcolormesh(lon, lat, values.T, shading="gouraud", **kwargs)
            elif lon.shape == values.shape and lat.shape == values.shape:
                return axis.pcolormesh(lon, lat, values, shading="gouraud", **kwargs)
        except Exception:  # noqa: BLE001 - fall back to a plain image on any plotting quirk
            pass
    return axis.imshow(values, origin="lower", aspect="auto", interpolation="bilinear", **kwargs)


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
