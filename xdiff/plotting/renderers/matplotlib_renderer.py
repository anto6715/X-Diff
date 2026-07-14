"""Static image renderer: one triptych (reference | comparison | difference) per variable.

Headless by construction — the matplotlib ``Agg`` backend is selected before ``pyplot``
is imported, so PNG/PDF/SVG rendering needs no display (CI, remote login nodes). The
heavy import is lazy and raises a clear install hint when the ``plot`` extra is missing.
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
    not match the field (so an orientation quirk never aborts the render). NaNs render
    blank in both paths.
    """
    values = np.asarray(values, dtype=float)
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
