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
# When geoviews draws a coastline map, land is a filled feature in this colour (matching
# the static/mtplot look) and the data NaNs fall through transparently. Without geoviews
# (no coastline data), NaN cells are painted this neutral grey instead.
_LAND_FILL = "#e8e6d8"
_LAND_COLOR = "#b0b0b0"
_COASTLINE_SCALE = "50m"  # Natural Earth resolution; 110m (geoviews default) is too coarse
# The difference is the hero, so it gets more height; reference/comparison are secondary
# (revealed on demand in a collapsed card). Both fill the width at a fixed height (see
# _finalize for why bare `responsive` is avoided).
_DIFF_HEIGHT = 460
_REFERENCE_HEIGHT = 300


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
            color="red", title="difference", tools=["hover"], responsive=True, height=_DIFF_HEIGHT
        )
        return panel.Column(title, difference)

    geoviews, features = _geoviews()
    field, is_geo, rasterize_ok = _build_field(holoviews, geoviews, variable, variable.difference)
    base = _rasterize(field) if rasterize_ok else field
    slider = panel.widgets.FloatSlider(
        name="colour limit (±)",
        start=0.0,
        end=_slider_end(variable),
        value=variable.diff_limit,
        step=_slider_end(variable) / 100.0,
    )

    # `apply.opts` binds the colour limit to the slider on the *same* map, so dragging it
    # re-clims in place without rebuilding the plot (zoom/pan preserved).
    styled = base.apply.opts(
        cmap=_DIFF_CMAP,
        clim=panel.bind(lambda limit: (-limit, limit), slider),
        clipping_colors=_clipping(is_geo),
        colorbar=True,
        tools=["hover"],
        title="difference",
    )
    view = _finalize(holoviews, features, styled, variable, _DIFF_HEIGHT, is_geo)
    return panel.Column(title, slider, panel.pane.HoloViews(view))


def _reference_block(holoviews, panel, variable: VariablePlot):
    """The secondary reference/comparison view for one variable (goes in the bottom card)."""
    heading = panel.pane.Markdown(f"**{variable.label}**")
    if len(variable.dims) != 2:
        axis = _axis_1d(variable)
        overlay = (
            holoviews.Curve((axis, variable.reference), label="reference")
            * holoviews.Curve((axis, variable.comparison), label="comparison")
        ).opts(
            holoviews.opts.Curve(tools=["hover"]),
            holoviews.opts.Overlay(
                legend_position="top_right",
                title="reference vs comparison",
                responsive=True,
                height=_REFERENCE_HEIGHT,
            ),
        )
        return panel.Column(heading, overlay)

    geoviews, features = _geoviews()
    low, high = _shared_clim(variable.reference, variable.comparison)
    maps = []
    for values, name in ((variable.reference, "reference"), (variable.comparison, "comparison")):
        field, is_geo, rasterize_ok = _build_field(holoviews, geoviews, variable, values)
        base = _rasterize(field) if rasterize_ok else field
        styled = base.opts(
            cmap=_REFERENCE_CMAP,
            clim=(low, high),
            clipping_colors=_clipping(is_geo),
            colorbar=True,
            tools=["hover"],
            title=name,
        )
        maps.append(_finalize(holoviews, features, styled, variable, _REFERENCE_HEIGHT, is_geo))
    return panel.Column(heading, panel.Row(*maps))


def _geoviews():
    """Return (geoviews, geoviews.feature) if importable, else (None, None).

    geoviews gives the maps a PlateCarree projection with coastlines and a filled land
    feature (the mtplot look). Absent or unable to reach its coastline data, the maps fall
    back to plain lon/lat rasters with grey land.
    """
    try:
        import geoviews
        import geoviews.feature as features

        geoviews.extension("bokeh")
        return geoviews, features
    except Exception:  # noqa: BLE001 - geoviews/cartopy absent or broken -> plain maps
        return None, None


def _build_field(holoviews, geoviews, variable: VariablePlot, values):
    """Return ``(element, is_geo, rasterize_ok)`` for the field.

    - **Regular lat/lon grid + geoviews** → a geoviews ``Image`` (a smooth raster, no
      visible cells) that datashades cleanly → ``(Image, True, True)``.
    - **Curvilinear grid + geoviews** → a geoviews ``QuadMesh`` drawn directly. datashader's
      ``rasterize`` of a *projected* curvilinear QuadMesh renders blank in the served
      pipeline, so it is skipped here (the vector cells stay crisp on zoom) →
      ``(QuadMesh, True, False)``.
    - **No geoviews** → a plain holoviews raster with grey land, datashaded →
      ``(element, False, True)``.
    """
    values = np.asarray(values, dtype=float)
    if geoviews is not None and variable.lon is not None and variable.lat is not None:
        regular = _regular_1d(variable.lon, variable.lat)
        if regular is not None:
            lon1d, lat1d = regular
            oriented = _orient_to_grid(lon1d, lat1d, values)
            if oriented is not None:
                return geoviews.Image((lon1d, lat1d, oriented), vdims=["value"]), True, True
        quadmesh = _geo_quadmesh(geoviews, variable.lon, variable.lat, values)
        if quadmesh is not None:
            return quadmesh, True, False
    return _map(holoviews, variable, values), False, True


def _regular_1d(lon, lat):
    """1-D (lon, lat) if the grid is rectilinear (already 1-D, or 2-D with identical rows/
    columns), else None. Fill values on masked cells break the identity check, so such a
    grid is treated as curvilinear — which is handled correctly, just without datashader.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.ndim == 1 and lat.ndim == 1:
        return lon, lat
    if lon.ndim == 2 and lat.ndim == 2:
        if np.allclose(lon, lon[0:1, :], atol=1e-4) and np.allclose(lat, lat[:, 0:1], atol=1e-4):
            return lon[0, :], lat[:, 0]
    return None


def _orient_to_grid(lon1d, lat1d, values):
    """``values`` transposed so its shape is ``(lat, lon)``, or None if it cannot line up."""
    if values.shape == (lat1d.size, lon1d.size):
        return values
    if values.shape == (lon1d.size, lat1d.size):
        return values.T
    return None


def _geo_quadmesh(geoviews, lon, lat, values):
    """A geoviews QuadMesh (default Longitude/Latitude dims) for a curvilinear grid.

    Coordinates on masked cells are blanked to NaN so NEMO ``nav_lon``/``nav_lat`` fill
    values neither draw spurious cells nor stretch the auto-ranged extent (e.g. over the
    Sahara). This replaces the xlim/ylim cropping, which blanks a datashaded geoviews map.
    """
    if lon is None or lat is None:
        return None
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.ndim == 1 and lat.ndim == 1:
        if values.shape == (lat.size, lon.size):
            return geoviews.QuadMesh((lon, lat, values), vdims=["value"])
        if values.shape == (lon.size, lat.size):
            return geoviews.QuadMesh((lon, lat, values.T), vdims=["value"])
    elif lon.shape == values.shape and lat.shape == values.shape:
        masked = ~np.isfinite(values)
        lon = np.where(masked, np.nan, lon)
        lat = np.where(masked, np.nan, lat)
        return geoviews.QuadMesh((lon, lat, values), vdims=["value"])
    return None


def _rasterize(element):
    from holoviews.operation.datashader import rasterize

    return rasterize(element)


def _clipping(is_geo: bool) -> dict:
    # Geographic maps draw a filled land feature, so NaN cells fall through transparently;
    # plain maps have no land feature, so NaN is painted grey.
    return {"NaN": (0.0, 0.0, 0.0, 0.0)} if is_geo else {"NaN": _LAND_COLOR}


def _finalize(holoviews, features, styled, variable: VariablePlot, height: int, is_geo: bool):
    """Add coastlines + land (geo), or just size a plain raster.

    Sizing is ``responsive=True`` **with a fixed ``height``**, NOT bare ``responsive``:
    holoviews ``responsive=True`` alone maps to Bokeh ``stretch_both``, which collapses to
    zero height inside a vertically-stacked Column (the whole map goes blank). Pinning the
    height makes it stretch in width only.

    The extent is left to auto-range from the data (fill-value cells are already masked out
    of the coordinates in ``_build_field``). Do NOT set xlim/ylim here: on a datashaded
    geoviews overlay they collapse the axis to a degenerate range and blank the map.
    """
    if not is_geo or features is None:
        return styled.opts(responsive=True, height=height, active_tools=["wheel_zoom"])

    # geoviews features default to the coarse 110m Natural Earth data; use 50m so coastlines
    # are precise (matching the static map) without the weight of the full 10m dataset.
    land = features.land.opts(fill_color=_LAND_FILL, scale=_COASTLINE_SCALE)
    coastline = features.coastline.opts(line_color="black", line_width=0.5, scale=_COASTLINE_SCALE)
    view = land * styled * coastline
    return view.opts(
        holoviews.opts.Overlay(responsive=True, height=height, active_tools=["wheel_zoom"])
    )


def _load_viz():
    """Import holoviews + panel lazily and configure the Bokeh backend once.

    Returns the two modules, raising a clear install hint when the plot extra is absent.
    """
    try:
        import holoviews
        import panel
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "holoviews/panel are required for the interactive server. Install the plot extra "
            'with `uv sync --extra plot` or `uv tool install "xdiffly[plot]"`, '
            "or pass -o FILE.png for a static image instead."
        ) from exc
    _configure_bokeh_backend(holoviews, panel)
    return holoviews, panel


def _configure_bokeh_backend(holoviews, panel) -> None:
    """Register the Bokeh backend and quiet a benign layout warning (all idempotent)."""
    # Both are needed for a served page to actually render the HoloViews plots:
    # holoviews.extension registers the Bokeh backend; panel.extension makes Panel inject
    # the HoloViews/Bokeh JS resources into the served document (without it the layout and
    # Markdown render but the plot panes stay blank).
    holoviews.extension("bokeh")
    panel.extension()

    # Some layout combinations trip Bokeh's W-1005 (FIXED_SIZING_MODE) validation on every
    # render — cosmetic here and it floods the server log. Silence just that check.
    from bokeh.core.validation import silence
    from bokeh.core.validation.warnings import FIXED_SIZING_MODE

    silence(FIXED_SIZING_MODE, True)

    # datashader imports dask.dataframe, which warns once about query planning; irrelevant
    # to us (we never build a dask dataframe) and noisy in the server log. Match by module,
    # not message — the warning text starts with a newline, which defeats a message regex.
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module=r"dask\.dataframe")


def _map(module, variable: VariablePlot, values):
    """Build a 2-D map element (non-geographic): QuadMesh if lon/lat line up, else Image.

    ``module`` is holoviews here; ``rasterize`` (applied by the caller) re-aggregates it on
    zoom. The geographic path uses a geoviews QuadMesh instead (see ``_build_field``).
    """
    values = np.asarray(values, dtype=float)
    quadmesh = _quadmesh(module, variable.lon, variable.lat, values)
    return quadmesh if quadmesh is not None else module.Image(values, vdims=["value"])


def _quadmesh(module, lon, lat, values):
    """A QuadMesh (holoviews or geoviews) using lon/lat when their shapes line up, else None."""
    if lon is None or lat is None:
        return None
    kdims = ["lon", "lat"]
    if lon.ndim == 1 and lat.ndim == 1:
        if values.shape == (lat.size, lon.size):
            return module.QuadMesh((lon, lat, values), kdims=kdims, vdims=["value"])
        if values.shape == (lon.size, lat.size):
            return module.QuadMesh((lon, lat, values.T), kdims=kdims, vdims=["value"])
    elif lon.shape == values.shape and lat.shape == values.shape:
        return module.QuadMesh((lon, lat, values), kdims=kdims, vdims=["value"])
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
