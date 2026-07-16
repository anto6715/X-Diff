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
    from xdiff.plotting.spec import PlotSpec, VariablePlot

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
    """Compose the page (no server started) as a sidebar-driven, one-variable app.

    A ``FastListTemplate`` holds the controls in the sidebar — variable selector, colour-limit
    slider, colormap, smooth/blocks toggle — and the main area shows the *selected* variable:
    a min/max readout, the hero difference map, and a collapsed reference/comparison card.
    Building one variable at a time keeps loading fast on many-variable files. Switching
    variable re-targets the controls; colour limit and colormap re-style the hero in place
    (zoom kept), and the toggle re-runs the datashader interpolation.
    """
    holoviews, panel = _load_viz()
    variables = spec.variables
    subtitle = panel.pane.Markdown(f"`{spec.reference_path.name}` vs `{spec.comparison_path.name}`")
    if not variables:
        return panel.template.FastListTemplate(
            title="xdiff plot", main=[subtitle, panel.pane.Markdown("No plottable variables.")]
        )

    var_select = panel.widgets.Select(
        name="Variable", options={variable.label: index for index, variable in enumerate(variables)}, value=0
    )
    climit = panel.widgets.FloatSlider(name="colour limit (±)", start=0.0, end=1.0, value=0.5, step=0.01)
    cmap_select = panel.widgets.Select(name="Colormap", options=list(_DIFF_CMAPS), value=_DIFF_CMAP)
    # color="primary" highlights the active mode (the selected button is filled), so it is
    # clear at a glance which of smooth/blocks is applied without toggling to compare.
    render_toggle = panel.widgets.RadioButtonGroup(
        name="Rendering", options=list(_METHODS), value=_SMOOTH, color="primary"
    )
    map_controls = (climit, cmap_select, render_toggle)

    def sync_controls(index: int) -> None:
        # Colour limit is per-variable (own units/scale); re-target its bounds on switch. The
        # map-only controls are disabled for a 1-D variable (a line has no colour map).
        variable = variables[index]
        is_map = len(variable.dims) == 2
        if is_map:
            end = _slider_end(variable)
            climit.param.update(start=0.0, end=end, value=min(variable.diff_limit, end), step=end / 100.0)
        for widget in map_controls:
            widget.disabled = not is_map

    sync_controls(0)
    var_select.param.watch(lambda event: sync_controls(event.new), "value")

    def render_main(index, method):
        return _variable_view(
            holoviews, panel, variables[index], method=method, cmap_widget=cmap_select, climit_widget=climit
        )

    return panel.template.FastListTemplate(
        title="xdiff plot",
        sidebar=_sidebar(panel, var_select, climit, cmap_select, render_toggle),
        main=[subtitle, panel.bind(render_main, var_select, render_toggle)],
    )


def _sidebar(panel, var_select, climit, cmap_select, render_toggle) -> list:
    """Group the controls into labelled, divider-separated sections (Variable / Colour / Rendering)."""

    def heading(text: str):
        return panel.pane.Markdown(f"#### {text}", margin=(4, 10, -6, 10))

    return [
        heading("Variable"),
        var_select,
        panel.layout.Divider(),
        heading("Colour"),
        climit,
        cmap_select,
        panel.layout.Divider(),
        heading("Rendering"),
        render_toggle,
    ]


def _variable_view(holoviews, panel, variable: VariablePlot, *, method, cmap_widget, climit_widget):
    """The main-area view for one variable: min/max readout, hero difference, reference card."""
    metadata = _metadata(panel, variable)
    if len(variable.dims) != 2:
        hero = holoviews.Curve((_axis_1d(variable), variable.difference)).opts(
            color="red", title="difference", tools=["hover"], responsive=True, height=_DIFF_HEIGHT
        )
        reference = _reference_overlay_1d(holoviews, variable)
    else:
        hero = _hero_map(
            holoviews, panel, variable, method=method, cmap_widget=cmap_widget, climit_widget=climit_widget
        )
        reference = panel.Row(*_reference_maps(holoviews, variable), sizing_mode="stretch_width")
    card = panel.Card(reference, title="Reference & comparison", collapsed=True, sizing_mode="stretch_width")
    return panel.Column(metadata, hero, card, sizing_mode="stretch_width")


def _metadata(panel, variable: VariablePlot):
    """The min/max readout — the true magnitude of the difference, not the slider's clip."""
    difference = np.asarray(variable.difference, dtype=float)
    low = float(np.nanmin(difference))
    high = float(np.nanmax(difference))
    units = variable.units or "—"
    return panel.pane.Markdown(f"## {variable.label}\nunits **{units}**  ·  min **{low:.3g}**  ·  max **{high:.3g}**")


def _hero_map(holoviews, panel, variable: VariablePlot, *, method, cmap_widget, climit_widget):
    """The hero difference map: datashaded, colormap + colour limit bound in place, zoom kept."""
    element = _field_element(holoviews, variable, variable.difference)
    base = _datashaded(holoviews, element, method=method)
    view = base.apply.opts(
        cmap=panel.bind(lambda name: name, cmap_widget),
        clim=panel.bind(lambda limit: (-limit, limit), climit_widget),
        clipping_colors={"NaN": _LAND_COLOR},
        colorbar=True,
        tools=["hover"],
        title="difference",
        responsive=True,
        height=_DIFF_HEIGHT,
        active_tools=["wheel_zoom"],
        **_extent(variable),
    )
    return panel.pane.HoloViews(view)


def _reference_maps(holoviews, variable: VariablePlot):
    """The secondary reference/comparison maps for one variable (viridis, shared clim)."""
    low, high = _shared_clim(variable.reference, variable.comparison)
    extent = _extent(variable)
    maps = []
    for values, name in ((variable.reference, "reference"), (variable.comparison, "comparison")):
        element = _field_element(holoviews, variable, values)
        maps.append(
            _datashaded(holoviews, element, method=_SMOOTH).opts(
                cmap=_REFERENCE_CMAP,
                clim=(low, high),
                clipping_colors={"NaN": _LAND_COLOR},
                colorbar=True,
                tools=["hover"],
                title=name,
                responsive=True,
                height=_REFERENCE_HEIGHT,
                active_tools=["wheel_zoom"],
                **extent,
            )
        )
    return maps


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


def _field_element(holoviews, variable: VariablePlot, values):
    """A plain holoviews element for a 2-D field, ready to datashade.

    - **Rectilinear grid** (1-D lon/lat) → ``Image`` (datashades and interpolates cleanly).
    - **Curvilinear grid** (2-D lon/lat) → ``QuadMesh``, with fill-value cells blanked to
      NaN so NEMO ``nav_lon``/``nav_lat`` fills neither draw spurious cells nor stretch the
      auto-ranged extent.
    - **No usable coordinates** → an index-axis ``Image``.
    """
    values = np.asarray(values, dtype=float)
    lon, lat = variable.lon, variable.lat
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


def _extent(variable: VariablePlot) -> dict:
    """``{'xlim': ..., 'ylim': ...}`` cropping the map to where data actually is, else ``{}``.

    Frames the plot tightly on the valid domain (e.g. the Mediterranean) instead of the raw
    coordinate bounding box, which fill values on masked cells would stretch. Without geoviews
    on this path, xlim/ylim frame the plain raster cleanly (they blanked the old geo overlay).
    """
    from xdiff.plotting.spec import valid_extent

    extent = valid_extent(variable)
    if extent is None:
        return {}
    return {"xlim": (extent[0], extent[1]), "ylim": (extent[2], extent[3])}


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
