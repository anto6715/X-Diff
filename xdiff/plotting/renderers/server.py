"""Live interactive renderer: a Panel/Bokeh server serving one page per file pair.

Lifecycle differs from the static renderer — ``serve`` starts a Bokeh server bound to
``localhost``, (optionally) opens a browser, and **blocks** until Ctrl-C. Data stays in
the running process; the browser talks to it over a websocket, so the diff colour limit
can be adjusted live with no recompute. Nothing is written to disk. Heavy imports
(holoviews/panel/bokeh) are lazy, mirroring ``load_xarray`` and the Dask runtime.

Fields are datashaded (``regrid``/``rasterize``) so they re-aggregate server-side on
zoom and scale to large grids. There is no map projection or coastline layer: the data's
own NaN mask draws the land implicitly, which keeps loading fast and the dependency set
small (no cartopy/geoviews).
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xdiff.plotting.spec import DimControl, PlotSource, VariablePlot

DEFAULT_PORT = 5006  # fixed & predictable for a one-time SSH tunnel; not the Dask :8787

_REFERENCE_CMAP = "viridis"
_DIFF_CMAP = "RdBu_r"
# Diverging colormaps offered for the difference (centred at 0). RdBu_r is the default.
_DIFF_CMAPS = ("RdBu_r", "coolwarm", "seismic", "bwr", "PuOr_r")
# NaN cells (land / masked) are painted this neutral grey. On the diverging diff colormap
# white already means "no difference", so leaving NaN white would be ambiguous.
_LAND_COLOR = "#b0b0b0"
# The difference is the hero, so it gets more height; reference/comparison are secondary
# (revealed on demand in a collapsed card). Both stretch to the page width at a fixed
# height (a bare responsive/stretch_both collapses to zero height in a Column). The height
# is generous so the map is not a thin band; `data_aspect` is deliberately NOT set — it
# fights responsive width (holoviews disables responsive when height+aspect are both fixed).
_DIFF_HEIGHT = 620
_REFERENCE_HEIGHT = 320

# Rendering styles offered by the difference toggle: "smooth" interpolates between cell
# centres (linear), "blocks" shows the faithful grid cells (nearest). The reference and
# comparison maps are always smooth.
_SMOOTH = "smooth"
_BLOCKS = "blocks"
_METHODS = (_SMOOTH, _BLOCKS)

# Basemaps offered by the sidebar selector: display label -> holoviews tile-source class name
# (all keyless, fetched over the internet at view time). None = no basemap (offline-safe grey).
_BASEMAP_OFF = "None (offline)"
_BASEMAPS = {
    _BASEMAP_OFF: None,
    "Carto Light": "CartoLight",
    "Carto Dark": "CartoDark",
    "OpenStreetMap": "OSM",
    "Esri Imagery (satellite)": "EsriImagery",
    "Esri NatGeo": "EsriNatGeo",
    "Esri Terrain": "EsriTerrain",
    "OpenTopoMap": "OpenTopoMap",
}


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


def build_application(source: PlotSource, *, default_index: int = 0):
    """Return the no-arg callable Panel serves (a fresh dashboard per browser session).

    Deliberately a named closure, **not** ``functools.partial``: Bokeh's server
    introspects the app callable's signature to decide how to invoke it and mishandles a
    partial, serving an empty document (the page loads but the plots are blank). Building
    per session also gives each browser its own widget (slider) state.
    """

    def application():
        return build_dashboard(source, default_index=default_index)

    return application


def serve(
    source: PlotSource, *, port: int, open_browser: bool, address: str = "localhost", default_index: int = 0
) -> None:
    """Serve the dashboard at ``http://{address}:{port}`` and block until Ctrl-C."""
    _, panel = _load_viz()
    try:
        panel.serve(
            build_application(source, default_index=default_index),
            port=port,
            address=address,
            show=open_browser,
            title="xdiff plot",
            websocket_origin=[f"localhost:{port}", f"127.0.0.1:{port}", f"{address}:{port}"],
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive Ctrl-C path
        pass


def build_dashboard(source: PlotSource, *, default_index: int = 0):
    """Compose the page (no server started) as a sidebar-driven, one-variable app.

    A ``FastListTemplate`` holds the controls in the sidebar — variable selector, colour-limit
    slider, colormap, smooth/blocks toggle — and the main area shows the *selected* variable:
    a min/max readout, the time/depth controls (when the variable has extra dims), the hero
    difference map, and a collapsed reference/comparison card. Slices are produced on demand
    from ``source`` so time/depth can be changed live. Switching variable re-targets the
    controls; colour limit and colormap re-style the hero in place (zoom kept), the toggle
    re-runs the interpolation, and a time/depth change re-slices server-side.
    """
    holoviews, panel = _load_viz()
    handles = source.variables
    subtitle = panel.pane.Markdown(f"`{source.reference_path.name}` vs `{source.comparison_path.name}`")
    if not handles:
        return panel.template.FastListTemplate(
            title="xdiff plot", main=[subtitle, panel.pane.Markdown("No plottable variables.")]
        )

    var_select = panel.widgets.Select(
        name="Variable",
        options={handle.label: index for index, handle in enumerate(handles)},
        value=min(default_index, len(handles) - 1),
    )
    climit = panel.widgets.FloatSlider(name="colour limit (±)", start=0.0, end=1.0, value=0.5, step=0.01)
    cmap_select = panel.widgets.Select(name="Colormap", options=list(_DIFF_CMAPS), value=_DIFF_CMAP)
    # color="primary" highlights the active mode (the selected button is filled), so it is
    # clear at a glance which of smooth/blocks is applied without toggling to compare.
    render_toggle = panel.widgets.RadioButtonGroup(
        name="Rendering", options=list(_METHODS), value=_SMOOTH, color="primary"
    )
    # Off by default: tiles need internet (blank on an offline node), so the plain grey path
    # stays the safe default; picking a basemap overlays the data on those web-map tiles.
    basemap_select = panel.widgets.Select(name="Basemap", options=_BASEMAPS, value=None)
    map_controls = (climit, cmap_select, render_toggle, basemap_select)
    dims_box = panel.Column()  # sidebar "Dimensions" section, repopulated per variable
    main_holder = panel.Column(sizing_mode="stretch_width")  # main content, replaced on redraw
    state: dict = {"names": [], "selects": []}

    def redraw(*_events) -> None:
        # Re-slice for the current variable + dim selection and replace the main content. A
        # dim change / variable switch / render-mode / basemap change lands here (rebuilds the
        # plot); colour limit and colormap are bound in place and never trigger this.
        index = var_select.value
        selection = {name: select.value for name, select in zip(state["names"], state["selects"], strict=True)}
        variable = source.slice(index, selection)
        main_holder[:] = [
            _rendered_variable(
                holoviews,
                panel,
                variable,
                method=render_toggle.value,
                basemap=basemap_select.value,
                cmap_widget=cmap_select,
                climit_widget=climit,
            )
        ]

    def rebuild_for_variable(index: int) -> None:
        # Rebuild the per-variable dimension selectors (Select, not slider: fires once on pick,
        # not on every step of a drag) and re-target the colour limit from the default slice.
        handle = handles[index]
        for widget in map_controls:
            widget.disabled = not handle.is_map
        selects = [_dim_select(panel, dim) for dim in handle.extra_dims]
        for select in selects:
            select.param.watch(redraw, "value")
        state["names"] = [dim.name for dim in handle.extra_dims]
        state["selects"] = selects
        dims_box[:] = selects or [panel.pane.Markdown("_no time / depth dimensions_", margin=(0, 10))]
        if handle.is_map:
            variable = source.slice(index, {dim.name: dim.default for dim in handle.extra_dims})
            end = _slider_end(variable)
            climit.param.update(start=0.0, end=end, value=min(variable.diff_limit, end), step=end / 100.0)
        redraw()

    rebuild_for_variable(var_select.value)
    var_select.param.watch(lambda event: rebuild_for_variable(event.new), "value")
    render_toggle.param.watch(redraw, "value")
    basemap_select.param.watch(redraw, "value")

    return panel.template.FastListTemplate(
        title="xdiff plot",
        sidebar=_sidebar(panel, var_select, dims_box, climit, cmap_select, render_toggle, basemap_select),
        main=[subtitle, main_holder],
    )


def _sidebar(panel, var_select, dims_box, climit, cmap_select, render_toggle, basemap_select) -> list:
    """Group the controls into labelled, divider-separated sections."""

    def heading(text: str):
        return panel.pane.Markdown(f"#### {text}", margin=(4, 10, -6, 10))

    return [
        heading("Variable"),
        var_select,
        panel.layout.Divider(),
        heading("Dimensions"),
        dims_box,
        panel.layout.Divider(),
        heading("Colour"),
        climit,
        cmap_select,
        panel.layout.Divider(),
        heading("Rendering"),
        render_toggle,
        basemap_select,
    ]


def _dim_select(panel, dim: DimControl):
    """A dropdown selecting one index of a dimension (time, depth, …) by its coordinate label.

    A ``Select`` (not a slider) so a choice re-slices exactly once, instead of firing for every
    value swept through while dragging a slider.
    """
    options = {label: index for index, label in enumerate(dim.labels)}
    return panel.widgets.Select(name=dim.name, options=options, value=dim.default)


def _rendered_variable(holoviews, panel, variable: VariablePlot, *, method, basemap, cmap_widget, climit_widget):
    """Render one already-sliced variable: min/max readout, hero difference, reference card.

    ``basemap`` is a holoviews tile-source class name, or ``None`` for the plain grey path.
    """
    metadata = _metadata(panel, variable)
    if len(variable.dims) != 2:
        hero = holoviews.Curve((_axis_1d(variable), variable.difference)).opts(
            color="red", title="difference", tools=["hover"], responsive=True, height=_DIFF_HEIGHT
        )
        reference = _reference_overlay_1d(holoviews, variable)
    else:
        hero = _hero_map(
            holoviews,
            panel,
            variable,
            method=method,
            basemap=basemap,
            cmap_widget=cmap_widget,
            climit_widget=climit_widget,
        )
        reference = panel.Row(*_reference_maps(holoviews, variable, basemap=basemap), sizing_mode="stretch_width")
    card = panel.Card(reference, title="Reference & comparison", collapsed=True, sizing_mode="stretch_width")
    return panel.Column(metadata, hero, card, sizing_mode="stretch_width")


def _metadata(panel, variable: VariablePlot):
    """The min/max readout — the true magnitude of the difference, not the slider's clip."""
    difference = np.asarray(variable.difference, dtype=float)
    low = float(np.nanmin(difference))
    high = float(np.nanmax(difference))
    units = variable.units or "—"
    return panel.pane.Markdown(f"## {variable.label}\nunits **{units}**  ·  min **{low:.3g}**  ·  max **{high:.3g}**")


def _hero_map(holoviews, panel, variable: VariablePlot, *, method, basemap, cmap_widget, climit_widget):
    """The hero difference map: datashaded, colormap + colour limit bound in place, zoom kept."""
    element = _field_element(holoviews, variable, variable.difference, web_mercator=basemap is not None)
    base = _datashaded(holoviews, element, method=method)
    styled = base.apply.opts(
        cmap=panel.bind(lambda name: name, cmap_widget),
        clim=panel.bind(lambda limit: (-limit, limit), climit_widget),
        clipping_colors=_clipping(basemap),
        colorbar=True,
        tools=["hover"],
        title="difference",
    )
    return panel.pane.HoloViews(_with_basemap(holoviews, styled, variable, _DIFF_HEIGHT, basemap))


def _reference_maps(holoviews, variable: VariablePlot, *, basemap):
    """The secondary reference/comparison maps for one variable (viridis, shared clim)."""
    low, high = _shared_clim(variable.reference, variable.comparison)
    maps = []
    for values, name in ((variable.reference, "reference"), (variable.comparison, "comparison")):
        element = _field_element(holoviews, variable, values, web_mercator=basemap is not None)
        styled = _datashaded(holoviews, element, method=_SMOOTH).opts(
            cmap=_REFERENCE_CMAP,
            clim=(low, high),
            clipping_colors=_clipping(basemap),
            colorbar=True,
            tools=["hover"],
            title=name,
        )
        maps.append(_with_basemap(holoviews, styled, variable, _REFERENCE_HEIGHT, basemap))
    return maps


def _clipping(basemap) -> dict:
    # With a basemap, masked (NaN) cells are transparent so the tiles show through; without
    # one, they are painted grey (land) since there is nothing underneath.
    return {"NaN": (0.0, 0.0, 0.0, 0.0)} if basemap is not None else {"NaN": _LAND_COLOR}


def _with_basemap(holoviews, styled, variable: VariablePlot, height: int, basemap):
    """Size the map, cropping to the data; overlay it on the ``basemap`` tiles when set.

    ``basemap`` is a holoviews tile-source class name (or None). Tiles are Web Mercator, so
    the field's coordinates were already reprojected in ``_field_element`` and the extent is
    converted to metres here.
    """
    sizing = {
        "responsive": True,
        "height": height,
        "active_tools": ["wheel_zoom"],
        **_extent(variable, basemap is not None),
    }
    if basemap is None:
        return styled.opts(**sizing)
    return (_basemap_tiles(holoviews, basemap) * styled).opts(holoviews.opts.Overlay(**sizing))


def _basemap_tiles(holoviews, name: str):
    """The named holoviews web-map basemap (needs internet; blank tiles if offline)."""
    import holoviews.element.tiles as tiles

    return getattr(tiles, name)()


def _reference_overlay_1d(holoviews, variable: VariablePlot):
    """The 1-D reference-vs-comparison overlay (goes in the collapsed card)."""
    axis = _axis_1d(variable)
    return (
        holoviews.Curve((axis, variable.reference), label="reference")
        * holoviews.Curve((axis, variable.comparison), label="comparison")
    ).opts(
        holoviews.opts.Curve(tools=["hover"]),
        holoviews.opts.Overlay(
            legend_position="top_right", title="reference vs comparison", responsive=True, height=_REFERENCE_HEIGHT
        ),
    )


def _field_element(holoviews, variable: VariablePlot, values, *, web_mercator: bool = False):
    """A plain holoviews element for a 2-D field, ready to datashade.

    - **Rectilinear grid** (1-D lon/lat) → ``Image`` (datashades and interpolates cleanly).
    - **Curvilinear grid** (2-D lon/lat) → ``QuadMesh``, coordinates passed through untouched.
    - **No usable coordinates** → an index-axis ``Image``.

    When ``web_mercator`` (basemap on), lon/lat are reprojected to Web Mercator metres and the
    field is always a ``QuadMesh`` — Web Mercator northing is non-linear in latitude, so a
    regular ``Image`` would misalign with the tiles.
    """
    values = np.asarray(values, dtype=float)
    lon, lat = variable.lon, variable.lat
    if web_mercator and lon is not None and lat is not None:
        mercator = _mercator_quadmesh(holoviews, np.asarray(lon, dtype=float), np.asarray(lat, dtype=float), values)
        if mercator is not None:
            return mercator
    if lon is not None and lat is not None:
        lon_a = np.asarray(lon, dtype=float)
        lat_a = np.asarray(lat, dtype=float)
        if lon_a.ndim == 1 and lat_a.ndim == 1:
            oriented = _orient(values, lat_a.size, lon_a.size)
            if oriented is not None:
                try:
                    return holoviews.Image((lon_a, lat_a, oriented), kdims=["lon", "lat"], vdims=["value"])
                except Exception:  # noqa: BLE001 - irregular 1-D sampling -> QuadMesh handles it
                    return holoviews.QuadMesh((lon_a, lat_a, oriented), kdims=["lon", "lat"], vdims=["value"])
        quadmesh = _curvilinear_quadmesh(holoviews, lon_a, lat_a, values)
        if quadmesh is not None:
            return quadmesh
    return holoviews.Image(values, vdims=["value"])


def _mercator_quadmesh(holoviews, lon, lat, values):
    """A QuadMesh with lon/lat reprojected to Web Mercator metres (to align with tiles), or None.

    Rectilinear 1-D coordinates are meshed to 2-D first; datashader's ``lnglat_to_meters`` then
    converts them elementwise. Returns None if the coordinates and values cannot be lined up.
    """
    from datashader.utils import lnglat_to_meters

    if lon.ndim == 1 and lat.ndim == 1:
        oriented = _orient(values, lat.size, lon.size)
        if oriented is None:
            return None
        lon, lat, values = *np.meshgrid(lon, lat), oriented
    elif not (lon.ndim == 2 and lat.ndim == 2 and lon.shape == values.shape and lat.shape == values.shape):
        return None
    easting, northing = lnglat_to_meters(lon, lat)
    return holoviews.QuadMesh((easting, northing, values), kdims=["x", "y"], vdims=["value"])


def _orient(values, n_rows: int, n_cols: int):
    """``values`` shaped ``(n_rows, n_cols)`` (transposing if needed), or None if it cannot."""
    if values.shape == (n_rows, n_cols):
        return values
    if values.shape == (n_cols, n_rows):
        return values.T
    return None


def _curvilinear_quadmesh(holoviews, lon, lat, values):
    """A QuadMesh for a 2-D (curvilinear) lon/lat grid, or None if the shapes disagree.

    The coordinates are passed through UNTOUCHED. NaN-ing the vertices of masked cells
    corrupts a *structured* QuadMesh: datashader draws quads spanning the holes, producing
    diagonal streaks across the domain on real grids (e.g. NEMO's 2-D ``nav_lon``/``nav_lat``
    with interior land). Land is masked by the NaN *values* instead, which datashader renders
    as NaN pixels — painted grey via ``clipping_colors``.
    """
    if lon.ndim == 2 and lat.ndim == 2 and lon.shape == values.shape and lat.shape == values.shape:
        return holoviews.QuadMesh((lon, lat, values), kdims=["lon", "lat"], vdims=["value"])
    return None


def _extent(variable: VariablePlot, basemap: bool = False) -> dict:
    """``{'xlim': ..., 'ylim': ...}`` cropping the map to where data actually is, else ``{}``.

    Frames the plot tightly on the valid domain (e.g. the Mediterranean) instead of the raw
    coordinate bounding box, which fill values on masked cells would stretch. With a basemap
    the bounds are converted to Web Mercator metres to match the reprojected field and tiles.
    """
    from xdiff.plotting.spec import valid_extent

    extent = valid_extent(variable)
    if extent is None:
        return {}
    lon_min, lon_max, lat_min, lat_max = extent
    if basemap:
        from datashader.utils import lnglat_to_meters

        eastings, northings = lnglat_to_meters([lon_min, lon_max], [lat_min, lat_max])
        return {"xlim": (float(eastings[0]), float(eastings[1])), "ylim": (float(northings[0]), float(northings[1]))}
    return {"xlim": (lon_min, lon_max), "ylim": (lat_min, lat_max)}


def _datashaded(holoviews, element, *, method: str):
    """Datashade ``element`` so it re-aggregates on zoom, at the given interpolation.

    ``smooth`` interpolates between cell centres (linear); ``blocks`` keeps the faithful
    grid cells (nearest). A QuadMesh is first rasterized to a regular grid, then regridded
    so the interpolation choice applies uniformly to rectilinear and curvilinear data.
    """
    from holoviews.operation.datashader import rasterize, regrid

    interpolation = "linear" if method == _SMOOTH else "nearest"
    base = rasterize(element) if isinstance(element, holoviews.QuadMesh) else element
    return regrid(base, interpolation=interpolation, upsample=True)


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

    # If a grid stores non-finite fill values in its 2-D coordinates, datashader casts them
    # to int while aggregating the QuadMesh and emits this benign RuntimeWarning on every
    # re-aggregation (i.e. every zoom). Those cells still render correctly; silence the noise.
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"datashader\.glyphs\.quadmesh")


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
