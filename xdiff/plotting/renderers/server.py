"""Live interactive renderer: a Panel/Bokeh server serving one page per file pair.

Lifecycle differs from the static renderer — ``serve`` starts a Bokeh server bound to
``localhost``, (optionally) opens a browser, and **blocks** until Ctrl-C. Data stays in
the running process; the browser talks to it over a websocket, so the diff colour limit
can be adjusted live with no recompute. Nothing is written to disk. Heavy imports
(holoviews/panel/bokeh) are lazy, mirroring ``load_xarray`` and the Dask runtime.
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xdiff.plotting.spec import PlotSpec, VariablePlot

DEFAULT_PORT = 5006  # fixed & predictable for a one-time SSH tunnel; not the Dask :8787

_REFERENCE_CMAP = "viridis"
_DIFF_CMAP = "RdBu_r"
# The difference is the hero: give it a large frame. Reference/comparison are secondary
# (revealed on demand in a collapsed card), so they get a smaller frame.
_DIFF_FRAME = {"frame_width": 760, "frame_height": 520}
_REFERENCE_FRAME = {"frame_width": 340, "frame_height": 260}


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


def build_application(spec: PlotSpec):
    """Return the no-arg callable Panel serves (a fresh dashboard per browser session).

    Deliberately a named closure, **not** ``functools.partial``: Bokeh's server
    introspects the app callable's signature to decide how to invoke it and mishandles a
    partial, serving an empty document (the page loads but the plots are blank). Building
    per session also gives each browser its own widget (slider) state.
    """

    def application():
        return build_dashboard(spec)

    return application


def serve(spec: PlotSpec, *, port: int, open_browser: bool, address: str = "localhost") -> None:
    """Serve the dashboard at ``http://{address}:{port}`` and block until Ctrl-C."""
    _, panel = _load_viz()
    try:
        panel.serve(
            build_application(spec),
            port=port,
            address=address,
            show=open_browser,
            title="xdiff plot",
            websocket_origin=[f"localhost:{port}", f"127.0.0.1:{port}", f"{address}:{port}"],
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive Ctrl-C path
        pass


def build_dashboard(spec: PlotSpec):
    """Compose the page (no server started): difference maps are the hero.

    Layout: a header, then one large **difference** plot per variable (2-D map with a live
    colour-limit slider, or 1-D line). The reference and comparison maps — secondary — go
    into a single collapsed card at the bottom, revealed on demand.
    """
    holoviews, panel = _load_viz()
    header = panel.pane.Markdown(f"# xdiff plot\n`{spec.reference_path.name}` vs `{spec.comparison_path.name}`")
    diff_sections = [_diff_section(holoviews, panel, variable) for variable in spec.variables]
    reference_card = panel.Card(
        *(_reference_block(holoviews, panel, variable) for variable in spec.variables),
        title="Reference & comparison maps",
        collapsed=True,
        sizing_mode="stretch_width",
    )
    return panel.Column(header, *diff_sections, reference_card)


def _diff_section(holoviews, panel, variable: VariablePlot):
    """The hero difference plot for one variable (with a live colour slider when 2-D)."""
    title = panel.pane.Markdown(f"## {variable.label}")
    if len(variable.dims) != 2:
        difference = holoviews.Curve((_axis_1d(variable), variable.difference)).opts(
            color="red", title="difference", tools=["hover"], **_DIFF_FRAME
        )
        return panel.Column(title, difference)

    slider = panel.widgets.FloatSlider(
        name="colour limit (±)",
        start=0.0,
        end=_slider_end(variable),
        value=variable.diff_limit,
        step=_slider_end(variable) / 100.0,
    )

    def render(limit):
        return _map(holoviews, variable, variable.difference).opts(
            cmap=_DIFF_CMAP, clim=(-limit, limit), colorbar=True, title="difference", tools=["hover"], **_DIFF_FRAME
        )

    return panel.Column(title, slider, panel.bind(render, slider))


def _reference_block(holoviews, panel, variable: VariablePlot):
    """The secondary reference/comparison view for one variable (goes in the bottom card)."""
    heading = panel.pane.Markdown(f"**{variable.label}**")
    if len(variable.dims) != 2:
        axis = _axis_1d(variable)
        overlay = (
            holoviews.Curve((axis, variable.reference), label="reference")
            * holoviews.Curve((axis, variable.comparison), label="comparison")
        ).opts(
            holoviews.opts.Curve(tools=["hover"], **_REFERENCE_FRAME),
            holoviews.opts.Overlay(legend_position="top_right", title="reference vs comparison"),
        )
        return panel.Column(heading, overlay)

    low, high = _shared_clim(variable.reference, variable.comparison)
    reference = _map(holoviews, variable, variable.reference).opts(
        cmap=_REFERENCE_CMAP, clim=(low, high), colorbar=True, title="reference", tools=["hover"], **_REFERENCE_FRAME
    )
    comparison = _map(holoviews, variable, variable.comparison).opts(
        cmap=_REFERENCE_CMAP, clim=(low, high), colorbar=True, title="comparison", tools=["hover"], **_REFERENCE_FRAME
    )
    return panel.Column(heading, panel.Row(reference, comparison))


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
    # Both are needed for a served page to actually render the HoloViews plots:
    # holoviews.extension registers the Bokeh backend; panel.extension makes Panel
    # inject the HoloViews/Bokeh JS resources into the served document (without it the
    # layout and Markdown render but the plot panes stay blank). Both are idempotent.
    holoviews.extension("bokeh")
    panel.extension()

    # Our plots use fixed frame sizes (frame_width/frame_height) inside auto-sizing
    # layout containers, which Bokeh flags with W-1005 (FIXED_SIZING_MODE) on every
    # render — cosmetic here and it floods the server log. Silence just that check.
    from bokeh.core.validation import silence
    from bokeh.core.validation.warnings import FIXED_SIZING_MODE

    silence(FIXED_SIZING_MODE, True)

    return holoviews, panel


def _map(holoviews, variable: VariablePlot, values):
    """Build a 2-D map element: QuadMesh (honours 1-D or 2-D lon/lat), else Image.

    The caller applies cmap/clim/frame opts, since the difference and reference maps use
    different colormaps, limits, and sizes.
    """
    values = np.asarray(values, dtype=float)
    quadmesh = _quadmesh(holoviews, variable.lon, variable.lat, values)
    return quadmesh if quadmesh is not None else holoviews.Image(values, vdims=["value"])


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
