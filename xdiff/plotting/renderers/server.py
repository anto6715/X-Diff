"""Live interactive renderer: a Panel/Bokeh server serving one page per file pair.

Lifecycle differs from the static renderer — ``serve`` starts a Bokeh server bound to
``localhost``, (optionally) opens a browser, and **blocks** until Ctrl-C. Data stays in
the running process; the browser talks to it over a websocket, so the diff colour limit
can be adjusted live with no recompute. Nothing is written to disk. Heavy imports
(holoviews/panel/bokeh) are lazy, mirroring ``load_xarray`` and the Dask runtime.
"""

from __future__ import annotations

import functools
import socket
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xdiff.plotting.spec import PlotSpec, VariablePlot

DEFAULT_PORT = 5006  # fixed & predictable for a one-time SSH tunnel; not the Dask :8787

_REFERENCE_CMAP = "viridis"
_DIFF_CMAP = "RdBu_r"
_PANEL_WIDTH = 340
_PANEL_HEIGHT = 300


def ensure_port_available(address: str, port: int) -> None:
    """Fail fast (before any datasets are opened) if ``port`` is already in use.

    Never auto-increments: a user who set up ``ssh -L PORT:localhost:PORT`` needs the
    server on exactly ``PORT``; silently moving to another would point the tunnel at
    nothing. Raises ``ValueError`` naming the port so the CLI surfaces a clear message.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((address, port))
        except OSError as exc:
            raise ValueError(
                f"port {port} on {address} is already in use; choose another with --port "
                f"(note the Dask dashboard uses :8787)"
            ) from exc


def serve(spec: PlotSpec, *, port: int, open_browser: bool, address: str = "localhost") -> None:
    """Serve the dashboard at ``http://{address}:{port}`` and block until Ctrl-C."""
    _, panel = _load_viz()
    application = functools.partial(build_dashboard, spec)
    try:
        panel.serve(
            application,
            port=port,
            address=address,
            show=open_browser,
            title="xdiff plot",
            websocket_origin=[f"localhost:{port}", f"127.0.0.1:{port}", f"{address}:{port}"],
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive Ctrl-C path
        pass


def build_dashboard(spec: PlotSpec):
    """Compose one row per variable into a single Panel layout (no server started)."""
    holoviews, panel = _load_viz()
    rows = [_variable_row(holoviews, panel, variable) for variable in spec.variables]
    header = panel.pane.Markdown(f"# xdiff plot\n`{spec.reference_path.name}` vs `{spec.comparison_path.name}`")
    return panel.Column(header, *rows)


def _load_viz():
    """Import holoviews + panel lazily, with the Bokeh backend and a clear install hint."""
    try:
        import holoviews
        import panel
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "holoviews/panel are required for the interactive server. Install the plot extra "
            'with `uv sync --extra plot` or `uv tool install "xdiffly[plot]"`, '
            "or pass -o FILE.png for a static image instead."
        ) from exc
    holoviews.extension("bokeh")
    return holoviews, panel


def _variable_row(holoviews, panel, variable: VariablePlot):
    title = panel.pane.Markdown(f"### {variable.label}")
    if len(variable.dims) == 2:
        body = _map_row(holoviews, panel, variable)
    else:
        body = _line_row(holoviews, panel, variable)
    return panel.Column(title, body)


def _map_row(holoviews, panel, variable: VariablePlot):
    low, high = _shared_clim(variable.reference, variable.comparison)
    reference = _map_element(
        holoviews, variable, variable.reference, cmap=_REFERENCE_CMAP, clim=(low, high), title="reference"
    )
    comparison = _map_element(
        holoviews, variable, variable.comparison, cmap=_REFERENCE_CMAP, clim=(low, high), title="comparison"
    )
    return panel.Row(reference, comparison, _diff_panel(holoviews, panel, variable))


def _diff_panel(holoviews, panel, variable: VariablePlot):
    """The difference map plus a live symmetric colour-limit slider (no recompute)."""
    slider = panel.widgets.FloatSlider(
        name="colour limit (±)",
        start=0.0,
        end=_slider_end(variable),
        value=variable.diff_limit,
        step=_slider_end(variable) / 100.0,
    )

    def render(limit):
        element = _hv_element(holoviews, variable, variable.difference)
        return element.opts(cmap=_DIFF_CMAP, clim=(-limit, limit), colorbar=True, title="difference")

    return panel.Column(slider, panel.bind(render, slider))


def _line_row(holoviews, panel, variable: VariablePlot):
    axis = _axis_1d(variable)
    reference = holoviews.Curve((axis, variable.reference), label="reference")
    comparison = holoviews.Curve((axis, variable.comparison), label="comparison")
    overlay = (reference * comparison).opts(
        holoviews.opts.Curve(width=_PANEL_WIDTH, height=_PANEL_HEIGHT, tools=["hover"]),
        holoviews.opts.Overlay(legend_position="top_right", title="reference vs comparison"),
    )
    difference = holoviews.Curve((axis, variable.difference)).opts(
        width=_PANEL_WIDTH, height=_PANEL_HEIGHT, color="red", tools=["hover"], title="difference"
    )
    return panel.Row(overlay, difference)


def _map_element(holoviews, variable: VariablePlot, values, *, cmap, clim, title):
    element = _hv_element(holoviews, variable, values)
    return element.opts(cmap=cmap, clim=clim, colorbar=True, title=title)


def _hv_element(holoviews, variable: VariablePlot, values):
    """Build a QuadMesh (honours 1-D or 2-D lon/lat), Image, or Curve from a slice."""
    values = np.asarray(values, dtype=float)
    if len(variable.dims) == 2:
        quadmesh = _quadmesh(holoviews, variable.lon, variable.lat, values)
        element = quadmesh if quadmesh is not None else holoviews.Image(values, vdims=["value"])
        return element.opts(width=_PANEL_WIDTH, height=_PANEL_HEIGHT, tools=["hover"])
    return holoviews.Curve((_axis_1d(variable), values), vdims=["value"]).opts(
        width=_PANEL_WIDTH, height=_PANEL_HEIGHT, tools=["hover"]
    )


def _quadmesh(holoviews, lon, lat, values):
    """A QuadMesh using lon/lat for axes when their shapes line up, else None."""
    if lon is None or lat is None:
        return None
    kdims = ["lon", "lat"]
    if lon.ndim == 1 and lat.ndim == 1:
        if values.shape == (lat.size, lon.size):
            return holoviews.QuadMesh((lon, lat, values), kdims=kdims, vdims=["value"])
        if values.shape == (lon.size, lat.size):
            return holoviews.QuadMesh((lon, lat, values.T), kdims=kdims, vdims=["value"])
    elif lon.shape == values.shape and lat.shape == values.shape:
        return holoviews.QuadMesh((lon, lat, values), kdims=kdims, vdims=["value"])
    return None


def _axis_1d(variable: VariablePlot) -> np.ndarray:
    length = variable.reference.shape[0]
    for coordinate in (variable.lon, variable.lat):
        if coordinate is not None and coordinate.ndim == 1 and coordinate.size == length:
            return coordinate
    return np.arange(length)


def _shared_clim(reference: np.ndarray, comparison: np.ndarray) -> tuple[float | None, float | None]:
    stacked = np.concatenate([np.asarray(reference, dtype=float).ravel(), np.asarray(comparison, dtype=float).ravel()])
    if not np.any(np.isfinite(stacked)):
        return None, None
    return float(np.nanmin(stacked)), float(np.nanmax(stacked))


def _slider_end(variable: VariablePlot) -> float:
    return max(variable.diff_extreme, variable.diff_limit)
