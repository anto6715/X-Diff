"""Tests for the plot feature: PlotSpec building (all the logic) and renderer smoke."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xdiff.model import BoundingBox
from xdiff.plotting.renderers.matplotlib_renderer import (
    SUPPORTED_EXTENSIONS,
    choose_scale,
    output_paths,
    render_to_files,
    scaled_label,
    validate_output_extension,
)
from xdiff.plotting.spec import (
    PlotSource,
    PlotSpec,
    SkippedVariable,
    VariablePlot,
    build_plot_spec,
    find_regular_grid,
    open_plot_source,
    reduce_to_plottable,
    regular_axes,
)


def _static_source(spec: PlotSpec) -> PlotSource:
    """Wrap a ready PlotSpec as a (non-slicing) PlotSource for the server functions."""
    return PlotSource.static(spec.reference_path, spec.comparison_path, spec.variables)


def _rectilinear_dataset(field, *, name="sst", lon=None, lat=None, extra_dims=None, attrs=None):
    """Build a rectilinear dataset with lon/lat coords and one variable.

    ``field`` is the variable's data; its trailing two axes are (lat, lon), with any
    leading axes named by ``extra_dims`` (e.g. ``("time",)`` or ``("depth",)``).
    """
    lon = np.arange(field.shape[-1], dtype=float) if lon is None else np.asarray(lon, dtype=float)
    lat = np.arange(field.shape[-2], dtype=float) if lat is None else np.asarray(lat, dtype=float)
    dims = (*(extra_dims or ()), "lat", "lon")
    return xr.Dataset(
        {name: (dims, field, attrs or {})},
        coords={
            "lon": ("lon", lon, {"units": "degrees_east"}),
            "lat": ("lat", lat, {"units": "degrees_north"}),
        },
    )


def _write(path: Path, dataset: xr.Dataset) -> Path:
    dataset.to_netcdf(path)
    return path


def _spec_for(tmp_path, reference, comparison, variables=None, **kwargs) -> PlotSpec:
    reference_path = _write(tmp_path / "ref.nc", reference)
    comparison_path = _write(tmp_path / "cmp.nc", comparison)
    return build_plot_spec(
        reference_path,
        comparison_path,
        variables,
        last_time_step=kwargs.get("last_time_step", False),
        bbox=kwargs.get("bbox"),
    )


# --------------------------------------------------------------------------- build_plot_spec


def test_build_plot_spec_basic_2d(tmp_path):
    reference = _rectilinear_dataset(np.full((5, 5), 3.0), attrs={"units": "degC"})
    comparison = _rectilinear_dataset(np.full((5, 5), 1.0))

    spec = _spec_for(tmp_path, reference, comparison, variables=(("sst", "sst"),))

    assert spec.skipped == []
    assert len(spec.variables) == 1
    variable = spec.variables[0]
    assert variable.label == "sst"
    assert variable.units == "degC"
    assert variable.dims == ("lat", "lon")
    np.testing.assert_allclose(variable.difference, 2.0)
    assert variable.diff_limit == pytest.approx(2.0)
    assert variable.diff_extreme == pytest.approx(2.0)
    assert variable.lon is not None and variable.lat is not None
    assert variable.lon.shape == (5,)


def test_build_plot_spec_integer_safe_difference(tmp_path):
    # uint8 10 - 12 wraps to 254; the plot must show -2, computed in float.
    reference = _rectilinear_dataset(np.full((3, 3), 10, dtype=np.uint8))
    comparison = _rectilinear_dataset(np.full((3, 3), 12, dtype=np.uint8))

    spec = _spec_for(tmp_path, reference, comparison, variables=(("sst", "sst"),))

    variable = spec.variables[0]
    assert np.issubdtype(variable.difference.dtype, np.floating)
    np.testing.assert_allclose(variable.difference, -2.0)
    assert variable.diff_extreme == pytest.approx(2.0)


def test_build_plot_spec_variable_mapping(tmp_path):
    reference = _rectilinear_dataset(np.full((4, 4), 5.0), name="thetao")
    comparison = _rectilinear_dataset(np.full((4, 4), 4.0), name="votemper")

    spec = _spec_for(tmp_path, reference, comparison, variables=(("thetao", "votemper"),))

    assert spec.variables[0].label == "thetao -> votemper"
    np.testing.assert_allclose(spec.variables[0].difference, 1.0)


def test_build_plot_spec_picks_up_lon_lat_coordinates(tmp_path):
    lon = np.linspace(-10.0, 10.0, 6)
    lat = np.linspace(30.0, 45.0, 5)
    reference = _rectilinear_dataset(np.zeros((5, 6)), lon=lon, lat=lat)
    comparison = _rectilinear_dataset(np.ones((5, 6)), lon=lon, lat=lat)

    variable = _spec_for(tmp_path, reference, comparison, variables=(("sst", "sst"),)).variables[0]

    np.testing.assert_allclose(variable.lon, lon)
    np.testing.assert_allclose(variable.lat, lat)


def test_build_plot_spec_applies_bbox(tmp_path):
    lon = np.arange(-10.0, 11.0)  # 21 points
    lat = np.arange(-10.0, 11.0)
    reference = _rectilinear_dataset(np.zeros((21, 21)), lon=lon, lat=lat)
    comparison = _rectilinear_dataset(np.ones((21, 21)), lon=lon, lat=lat)

    variable = _spec_for(
        tmp_path,
        reference,
        comparison,
        variables=(("sst", "sst"),),
        bbox=BoundingBox(lon_min=-2, lon_max=2, lat_min=-2, lat_max=2),
    ).variables[0]

    # 5 points inside [-2, 2] on each axis.
    assert variable.reference.shape == (5, 5)


def test_build_plot_spec_last_time_step_reduction(tmp_path):
    field_ref = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 9.0)])  # (time, lat, lon)
    field_cmp = np.stack([np.full((3, 3), 0.0), np.full((3, 3), 4.0)])
    reference = _rectilinear_dataset(field_ref, extra_dims=("time",))
    comparison = _rectilinear_dataset(field_cmp, extra_dims=("time",))

    variable = _spec_for(tmp_path, reference, comparison, variables=(("sst", "sst"),), last_time_step=True).variables[0]

    assert variable.label == "sst [time=1]"
    assert variable.dims == ("lat", "lon")
    np.testing.assert_allclose(variable.difference, 5.0)  # 9 - 4 at the last step


def test_build_plot_spec_first_time_step_by_default(tmp_path):
    field_ref = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 9.0)])
    field_cmp = np.stack([np.full((3, 3), 0.0), np.full((3, 3), 4.0)])
    reference = _rectilinear_dataset(field_ref, extra_dims=("time",))
    comparison = _rectilinear_dataset(field_cmp, extra_dims=("time",))

    variable = _spec_for(tmp_path, reference, comparison, variables=(("sst", "sst"),)).variables[0]

    assert variable.label == "sst [time=0]"
    np.testing.assert_allclose(variable.difference, 1.0)  # 1 - 0 at the first step


def test_build_plot_spec_extra_dim_reduction_visible_in_label(tmp_path):
    reference = _rectilinear_dataset(np.ones((3, 4, 4)), name="thetao", extra_dims=("depth",))
    comparison = _rectilinear_dataset(np.zeros((3, 4, 4)), name="thetao", extra_dims=("depth",))

    variable = _spec_for(tmp_path, reference, comparison, variables=(("thetao", "thetao"),)).variables[0]

    assert variable.label == "thetao [depth=0]"
    assert variable.dims == ("lat", "lon")


def test_build_plot_spec_skips_shape_mismatch(tmp_path):
    reference = _rectilinear_dataset(np.zeros((5, 5)))
    comparison = _rectilinear_dataset(np.zeros((4, 5)))  # different lat size

    spec = _spec_for(tmp_path, reference, comparison, variables=(("sst", "sst"),))

    assert spec.variables == []
    assert len(spec.skipped) == 1
    assert spec.skipped[0].label == "sst"
    assert "(5, 5)" in spec.skipped[0].reason and "(4, 5)" in spec.skipped[0].reason


def test_build_plot_spec_skips_scalar_variable(tmp_path):
    reference = xr.Dataset({"scalar": ((), 5.0)})
    comparison = xr.Dataset({"scalar": ((), 4.0)})

    spec = _spec_for(tmp_path, reference, comparison, variables=(("scalar", "scalar"),))

    assert spec.variables == []
    assert len(spec.skipped) == 1
    assert "0-D" in spec.skipped[0].reason


def test_build_plot_spec_robust_diff_limit_clips_outlier(tmp_path):
    difference = np.ones((40, 40))
    difference[0, 0] = 100.0  # a single large outlier
    reference = _rectilinear_dataset(difference)
    comparison = _rectilinear_dataset(np.zeros((40, 40)))

    variable = _spec_for(tmp_path, reference, comparison, variables=(("sst", "sst"),)).variables[0]

    assert variable.diff_extreme == pytest.approx(100.0)
    # The 99.5th-percentile clip ignores the lone outlier, staying near the bulk value.
    assert variable.diff_limit == pytest.approx(1.0)


def test_build_plot_spec_skips_all_nan_difference(tmp_path):
    reference = _rectilinear_dataset(np.full((3, 3), np.nan))
    comparison = _rectilinear_dataset(np.zeros((3, 3)))

    spec = _spec_for(tmp_path, reference, comparison, variables=(("sst", "sst"),))

    assert spec.variables == []
    assert "all-NaN" in spec.skipped[0].reason


# --------------------------------------------------------------------------- reduce_to_plottable


def test_reduce_to_plottable_collapses_extra_dims():
    field = xr.DataArray(np.ones((2, 3, 4, 4)), dims=("time", "depth", "lat", "lon"))

    reduced, collapsed = reduce_to_plottable(field, {"lat", "lon"}, last_time_step=True)

    assert reduced.dims == ("lat", "lon")
    assert collapsed == {"time": 1, "depth": 0}


# --------------------------------------------------------------------------- naming / extension helpers


def test_output_paths_single_variable_is_exact(tmp_path):
    target = tmp_path / "diff.png"
    assert output_paths(target, ["thetao"]) == [target]


def test_output_paths_multiple_variables_suffix_the_stem(tmp_path):
    target = tmp_path / "diff.png"
    paths = output_paths(target, ["thetao", "sst -> analysed_sst"])
    assert paths[0].name == "diff_thetao.png"
    assert paths[1].name == "diff_sst_analysed_sst.png"


@pytest.mark.parametrize("extension", SUPPORTED_EXTENSIONS)
def test_validate_output_extension_accepts_supported(extension):
    validate_output_extension(Path(f"out{extension}"))


@pytest.mark.parametrize("bad", ["out.jpg", "out.html", "out"])
def test_validate_output_extension_rejects_unsupported(bad):
    with pytest.raises(ValueError, match="unsupported output extension"):
        validate_output_extension(Path(bad))


# --------------------------------------------------------------------------- renderer smoke


def _variable_plot(difference, **overrides):
    difference = np.asarray(difference, dtype=float)
    defaults = dict(
        label="sst",
        reference=np.zeros_like(difference),
        comparison=-difference,
        difference=difference,
        lon=None,
        lat=None,
        diff_limit=float(np.nanmax(np.abs(difference))) or 1.0,
        diff_extreme=float(np.nanmax(np.abs(difference))) or 1.0,
        units="degC",
        dims=("lat", "lon") if difference.ndim == 2 else ("lon",),
    )
    defaults.update(overrides)
    return VariablePlot(**defaults)


def test_render_to_files_writes_2d_triptych(tmp_path):
    spec = PlotSpec(tmp_path / "ref.nc", tmp_path / "cmp.nc", [_variable_plot(np.ones((5, 5)))], [])
    output = tmp_path / "diff.png"

    written = render_to_files(spec, output)

    assert written == [output]
    assert output.exists() and output.stat().st_size > 0


def test_render_to_files_writes_1d_lines(tmp_path):
    spec = PlotSpec(tmp_path / "ref.nc", tmp_path / "cmp.nc", [_variable_plot(np.arange(6.0))], [])
    output = tmp_path / "profile.pdf"

    written = render_to_files(spec, output)

    assert written == [output]
    assert output.exists() and output.stat().st_size > 0


@pytest.mark.parametrize("difference", [np.ones((5, 5)), np.arange(6.0)], ids=["map", "line"])
def test_render_to_files_saves_at_300_dpi(tmp_path, difference):
    from PIL import Image

    spec = PlotSpec(tmp_path / "ref.nc", tmp_path / "cmp.nc", [_variable_plot(difference)], [])
    output = tmp_path / "diff.png"

    render_to_files(spec, output)

    with Image.open(output) as image:
        assert tuple(round(value) for value in image.info["dpi"]) == (300, 300)
        assert image.width > 2500  # a 10-inch-wide figure, not 1000-1500 px


def test_render_to_files_multi_variable_suffixes(tmp_path):
    spec = PlotSpec(
        tmp_path / "ref.nc",
        tmp_path / "cmp.nc",
        [_variable_plot(np.ones((4, 4)), label="thetao"), _variable_plot(np.ones((4, 4)), label="sst")],
        [],
    )

    written = render_to_files(spec, tmp_path / "diff.png")

    assert {path.name for path in written} == {"diff_thetao.png", "diff_sst.png"}
    assert all(path.exists() for path in written)


def test_render_to_files_raises_when_nothing_plottable(tmp_path):
    spec = PlotSpec(tmp_path / "ref.nc", tmp_path / "cmp.nc", [], [SkippedVariable("sst", "0-D")])
    with pytest.raises(ValueError, match="no plottable variables"):
        render_to_files(spec, tmp_path / "diff.png")


# --------------------------------------------------------------------------- interactive server


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("localhost", 0))
        return probe.getsockname()[1]


def test_open_plot_source_exposes_extra_dims_and_slices(tmp_path):
    from xdiff.plotting.spec import open_plot_source

    nt, nz, ny, nx = 2, 3, 4, 5
    depth = np.array([1.0, 10.0, 100.0])
    coords = {
        "lon": ("lon", np.arange(nx, dtype=float)),
        "lat": ("lat", np.arange(ny, dtype=float)),
        "depth": ("depth", depth),
    }

    def dataset(per_depth):
        field = np.zeros((nt, nz, ny, nx))
        for level in range(nz):
            field[:, level] = per_depth * level  # difference will scale with depth level
        return xr.Dataset({"thetao": (("time", "depth", "lat", "lon"), field)}, coords=coords)

    ref = _write(tmp_path / "ref.nc", dataset(0.0))
    cmp = _write(tmp_path / "cmp.nc", dataset(1.0))  # ref - cmp = -level at each depth

    source = open_plot_source(ref, cmp, None, last_time_step=False, bbox=None)
    try:
        handle = source.variables[0]
        assert handle.is_map
        dims = {control.name: control for control in handle.extra_dims}
        assert set(dims) == {"time", "depth"}
        assert dims["depth"].size == nz
        assert dims["depth"].labels == ("1", "10", "100")  # coordinate values, not indices

        shallow = source.slice(0, {"time": 0, "depth": 0})
        deep = source.slice(0, {"time": 0, "depth": 2})
        assert shallow.label == "thetao"  # no [time=…, depth=…] decoration in server mode
        assert shallow.dims == ("lat", "lon")
        assert np.nanmean(shallow.difference) == pytest.approx(0.0)
        assert np.nanmean(deep.difference) == pytest.approx(-2.0)
    finally:
        source.close()


def test_open_plot_source_skips_bounds_and_coordinate_variables(tmp_path):
    from xdiff.plotting.spec import open_plot_source

    def dataset():
        return xr.Dataset(
            {
                "thetao": (("lat", "lon"), np.ones((4, 5))),
                "deptht_bounds": (("deptht", "axis_nbounds"), np.zeros((3, 2))),
            },
            coords={
                "lon": ("lon", np.arange(5.0)),
                "lat": ("lat", np.arange(4.0)),
                "deptht": ("deptht", np.arange(3.0)),
            },
        )

    ref = _write(tmp_path / "ref.nc", dataset())
    cmp = _write(tmp_path / "cmp.nc", dataset())

    source = open_plot_source(ref, cmp, None, last_time_step=False, bbox=None)
    try:
        assert [handle.label for handle in source.variables] == ["thetao"]  # no bounds, no lon/lat
    finally:
        source.close()


def test_build_dashboard_sidebar_variable_selector_lists_all_variables():
    import panel as pn

    from xdiff.plotting.renderers.server import build_dashboard

    spec = PlotSpec(
        Path("ref.nc"),
        Path("cmp.nc"),
        [_variable_plot(np.ones((5, 5)), label="thetao"), _variable_plot(np.arange(6.0), label="profile")],
        [],
    )

    dashboard = build_dashboard(_static_source(spec))

    assert isinstance(dashboard, pn.template.FastListTemplate)
    variable_select = next(
        w for w in dashboard.sidebar.objects if isinstance(w, pn.widgets.Select) and w.name == "Variable"
    )
    assert set(variable_select.options) == {"thetao", "profile"}


def test_build_dashboard_disables_map_controls_for_1d_variable():
    import panel as pn

    from xdiff.plotting.renderers.server import build_dashboard

    spec = PlotSpec(Path("ref.nc"), Path("cmp.nc"), [_variable_plot(np.arange(6.0), label="profile")], [])
    dashboard = build_dashboard(_static_source(spec))

    toggles = [w for w in dashboard.sidebar.objects if isinstance(w, pn.widgets.RadioButtonGroup)]
    assert toggles and all(w.disabled for w in toggles)


def test_build_dashboard_survives_unplottable_variable():
    """Regression: a variable discovery lists but slicing cannot difference (shape mismatch,
    all-NaN level) must not blank the dashboard.

    ``open_plot_source`` lists a variable from its dims alone; ``_build_variable_plot`` only
    raises later, inside ``source.slice`` (during a Panel callback). Unlike the static renderer
    it has no chance to record it as skipped up front, so the server must surface the reason in
    place instead of letting the exception escape the callback and serve an empty page.
    """
    import panel as pn

    from xdiff.plotting.renderers.server import build_dashboard
    from xdiff.plotting.spec import PlotSource, VariableHandle

    def slicer(index, selection):
        raise ValueError("shape (5, 5) vs (4, 5)")

    source = PlotSource(
        Path("ref.nc"),
        Path("cmp.nc"),
        [VariableHandle(label="thetao", is_map=True, extra_dims=())],
        slicer,
    )

    dashboard = build_dashboard(source)  # previously raised straight out of rebuild_for_variable

    assert isinstance(dashboard, pn.template.FastListTemplate)
    main_holder = dashboard.main.objects[1]
    assert "cannot plot this slice" in main_holder.objects[0].object


def test_build_dashboard_constructs_curvilinear_quadmesh():
    import panel as pn

    from xdiff.plotting.renderers.server import build_dashboard

    lon2d = np.tile(np.arange(4.0), (3, 1))
    lat2d = np.tile(np.arange(3.0)[:, None], (1, 4))
    variable = _variable_plot(np.ones((3, 4)), lon=lon2d, lat=lat2d)
    dashboard = build_dashboard(_static_source(PlotSpec(Path("ref.nc"), Path("cmp.nc"), [variable], [])))

    assert isinstance(dashboard, pn.template.FastListTemplate)


def test_curvilinear_land_is_not_filled_by_streaks():
    """Regression: NaN-value land on a 2-D (curvilinear) grid must stay empty.

    Blanking the *coordinates* of masked cells corrupts a structured QuadMesh — datashader
    draws quads spanning the holes, streaking across the domain (seen on real NEMO grids).
    Keeping coordinates intact leaves land as NaN pixels (rendered grey), so the centre of
    an interior land block must aggregate to NaN, not spurious data.
    """
    import holoviews as hv

    from xdiff.plotting.renderers.server import _datashaded, _field_element, _load_viz

    _load_viz()
    ny, nx = 60, 120
    lon2d = np.tile(np.linspace(-18.0, 37.0, nx), (ny, 1))
    lat2d = np.tile(np.linspace(30.0, 46.0, ny)[:, None], (1, nx))
    difference = np.random.default_rng(0).normal(size=(ny, nx))
    difference[20:45, 30:90] = np.nan  # interior "land" block
    variable = _variable_plot(difference, lon=lon2d, lat=lat2d)

    element = _field_element(hv, variable, variable.difference)
    assert isinstance(element, hv.QuadMesh)
    figure = hv.render(_datashaded(hv, element, method="smooth").opts(width=200, height=120))
    images = [
        r for r in figure.renderers if getattr(r, "glyph", None) is not None and r.glyph.__class__.__name__ == "Image"
    ]
    raster = np.asarray(images[0].data_source.data["image"][0])
    centre = raster[raster.shape[0] // 2, raster.shape[1] // 2]
    assert np.isnan(centre), "interior land aggregated to data — the streak corruption is back"


def test_basemap_reprojects_to_web_mercator_and_overlays_tiles():
    import holoviews as hv
    import panel as pn

    from xdiff.plotting.renderers.server import _field_element, _hero_map, _load_viz

    _load_viz()
    variable = _variable_plot(np.ones((5, 6)), lon=np.linspace(-10.0, 20.0, 6), lat=np.linspace(30.0, 45.0, 5))

    plain = _field_element(hv, variable, variable.difference, web_mercator=False)
    mercator = _field_element(hv, variable, variable.difference, web_mercator=True)
    assert isinstance(plain, hv.Image)  # lon/lat path unchanged
    assert isinstance(mercator, hv.QuadMesh)
    assert mercator.dimension_values("x").max() > 1e5  # metres, not degrees

    hero = _hero_map(
        hv,
        pn,
        variable,
        method="smooth",
        basemap="EsriImagery",
        cmap_widget=pn.widgets.Select(options=["RdBu_r"], value="RdBu_r"),
        climit_widget=pn.widgets.FloatSlider(start=0.0, end=3.0, value=1.0),
    )
    figure = hv.render(hero.object)
    assert any("Tile" in type(renderer).__name__ for renderer in figure.renderers), "basemap tiles missing"


def test_metadata_reports_min_and_max_of_difference():
    import panel as pn

    from xdiff.plotting.renderers.server import _metadata

    variable = _variable_plot(np.array([[-2.0, 0.0], [1.0, 3.0]]), label="thetao", units="degC")
    markdown = _metadata(pn, variable).object

    assert "min" in markdown and "max" in markdown
    assert "-2" in markdown and "3" in markdown


def test_ensure_port_available_passes_for_free_port():
    from xdiff.plotting.renderers.server import ensure_port_available

    ensure_port_available("localhost", _free_port())


def test_ensure_port_available_raises_for_busy_port():
    import socket

    from xdiff.plotting.renderers.server import ensure_port_available

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("localhost", 0))
    holder.listen()
    busy_port = holder.getsockname()[1]
    try:
        with pytest.raises(ValueError, match="already in use"):
            ensure_port_available("localhost", busy_port)
    finally:
        holder.close()


def test_serving_delivers_plot_glyphs_to_a_session():
    """Regression: serve the dashboard for real and assert the plots actually arrive.

    The first cut passed ``functools.partial(build_dashboard, spec)`` as the Panel app
    factory; Bokeh mishandled the partial and served an empty document (page loaded,
    plots blank). Building the bootstrap page returns HTTP 200 either way, so only a real
    served session — asserting glyph renderers are delivered — catches this.
    """
    import time

    import panel as pn
    from bokeh.client import pull_session
    from bokeh.models import GlyphRenderer
    from bokeh.models import Image as BkImage

    from xdiff.plotting.renderers.server import build_application

    # Real lon/lat so the raster (datashade) path is exercised, not the coordinate-less fallback.
    lon = np.linspace(0.0, 10.0, 6)
    lat = np.linspace(0.0, 8.0, 5)
    spec = PlotSpec(
        Path("ref.nc"),
        Path("cmp.nc"),
        [_variable_plot(np.ones((5, 6)), reference=np.arange(30.0).reshape(5, 6), lon=lon, lat=lat)],
        [],
    )
    port = _free_port()
    server = pn.serve(
        build_application(_static_source(spec)),
        port=port,
        address="localhost",
        show=False,
        threaded=True,
        websocket_origin=[f"localhost:{port}"],
    )
    try:
        session = None
        for _ in range(20):
            try:
                session = pull_session(url=f"http://localhost:{port}/")
                break
            except Exception:
                time.sleep(0.5)
        assert session is not None, "server did not become ready"
        images = [
            model
            for model in session.document.select({})
            if isinstance(model, GlyphRenderer) and isinstance(model.glyph, BkImage)
        ]
        finite = [int(np.isfinite(np.asarray(model.data_source.data["image"][0])).sum()) for model in images]
        session.close()
        # The difference map is delivered (the reference/comparison maps are built only when
        # their card is opened, see test_reference_card_is_built_only_when_opened)...
        assert len(images) >= 1
        # ...and each image carries real (non-blank) aggregated data, not an empty raster.
        assert finite and all(count > 0 for count in finite)
    finally:
        server.stop()


# --------------------------------------------------------------------------- shared discovery
@pytest.mark.parametrize(
    ("variables", "bbox"),
    [
        (None, None),
        ((("thetao", "votemper"),), None),
        (None, BoundingBox(lon_min=0.5, lon_max=2.5, lat_min=0.5, lat_max=2.5)),
        (
            (("thetao", "votemper"), ("missing", "missing")),
            BoundingBox(lon_min=0.5, lon_max=2.5, lat_min=0.5, lat_max=2.5),
        ),
    ],
)
def test_static_and_live_entry_points_agree_on_what_is_plottable(tmp_path, variables, bbox):
    field = np.arange(12.0).reshape(3, 4)
    reference = _rectilinear_dataset(field, name="thetao")
    reference["sst"] = (("lat", "lon"), field)
    comparison = _rectilinear_dataset(field + 1.0, name="thetao")
    comparison["votemper"] = (("lat", "lon"), field + 2.0)
    comparison["sst"] = (("lat", "lon"), field + 3.0)
    reference_path = _write(tmp_path / "ref.nc", reference)
    comparison_path = _write(tmp_path / "cmp.nc", comparison)

    spec = build_plot_spec(reference_path, comparison_path, variables, last_time_step=False, bbox=bbox)
    source = open_plot_source(reference_path, comparison_path, variables, last_time_step=False, bbox=bbox)
    try:
        assert [handle.label for handle in source.variables] == [plot.label for plot in spec.variables]
        assert [skipped.label for skipped in source.skipped] == [skipped.label for skipped in spec.skipped]
        assert source.variables, "the case must plot at least one variable to be meaningful"
    finally:
        source.close()


def test_open_plot_source_closes_reference_when_comparison_cannot_be_opened(tmp_path, monkeypatch):
    reference_path = _write(tmp_path / "ref.nc", _rectilinear_dataset(np.ones((2, 2))))
    opened = []
    real_open = xr.open_dataset

    def tracking_open(path, *args, **kwargs):
        dataset = real_open(path, *args, **kwargs)
        opened.append(dataset)
        return dataset

    monkeypatch.setattr(xr, "open_dataset", tracking_open)

    with pytest.raises(Exception):  # noqa: B017 - the exact error type is xarray's/the OS's
        open_plot_source(reference_path, tmp_path / "missing.nc", None, last_time_step=False, bbox=None)

    assert len(opened) == 1
    # A closed dataset has released its file handle.
    assert opened[0]._close is None


# --------------------------------------------------------------------------- regular grids in 2-D coordinates
def _nemo_style_grid(ny=12, nx=20, fill_block=(slice(0, 4), slice(0, 6))):
    """2-D lon/lat of a regular 0.25° grid with a (-1, -1) fill block, as NEMO writes them."""
    lon_axis = np.linspace(-6.0, -6.0 + 0.25 * (nx - 1), nx)
    lat_axis = np.linspace(30.0, 30.0 + 0.25 * (ny - 1), ny)
    lon2d, lat2d = np.meshgrid(lon_axis, lat_axis)
    lon2d[fill_block] = -1.0
    lat2d[fill_block] = -1.0
    return lon_axis, lat_axis, lon2d, lat2d


def test_find_regular_grid_recovers_axes_behind_fill_values():
    lon_axis, lat_axis, lon2d, lat2d = _nemo_style_grid()

    grid = find_regular_grid(lon2d, lat2d)

    assert grid is not None
    np.testing.assert_allclose(grid.lon, lon_axis)
    np.testing.assert_allclose(grid.lat, lat_axis)
    assert grid.fill[:4, :6].all() and grid.fill.sum() == 24


def test_find_regular_grid_rejects_a_curvilinear_grid():
    _, _, lon2d, lat2d = _nemo_style_grid(fill_block=(slice(0, 0), slice(0, 0)))
    rows = np.arange(lon2d.shape[0])[:, None]
    lon2d = lon2d + 0.05 * rows  # longitude now drifts with the row: a sheared grid

    assert find_regular_grid(lon2d, lat2d) is None


def test_find_regular_grid_rejects_a_single_displaced_point():
    _, _, lon2d, lat2d = _nemo_style_grid()
    lon2d[8, 10] += 0.1  # 40% of a cell: not a fill value, not on the axis

    assert find_regular_grid(lon2d, lat2d) is None


def test_regular_axes_keep_2d_coordinates_when_a_fill_point_holds_data():
    _, _, lon2d, lat2d = _nemo_style_grid()
    has_data = np.zeros(lon2d.shape, dtype=bool)
    assert np.ndim(regular_axes(lon2d, lat2d, has_data)[0]) == 1

    has_data[1, 1] = True  # inside the fill block: its position is unknown
    lon, lat = regular_axes(lon2d, lat2d, has_data)

    assert lon is lon2d and lat is lat2d


def test_build_plot_spec_gives_1d_axes_for_a_regular_grid_stored_in_2d(tmp_path):
    lon_axis, lat_axis, lon2d, lat2d = _nemo_style_grid()
    field = np.random.default_rng(1).normal(size=lon2d.shape)
    field[:4, :6] = np.nan  # no data where the coordinates are fill values

    def dataset(values):
        return xr.Dataset(
            {"sst": (("y", "x"), values)},
            coords={
                "nav_lon": (("y", "x"), lon2d, {"standard_name": "longitude"}),
                "nav_lat": (("y", "x"), lat2d, {"standard_name": "latitude"}),
            },
        )

    variable = _spec_for(tmp_path, dataset(field), dataset(field + 1.0), variables=(("sst", "sst"),)).variables[0]

    assert variable.lon.ndim == 1 and variable.lat.ndim == 1
    np.testing.assert_allclose(variable.lon, lon_axis)
    np.testing.assert_allclose(variable.lat, lat_axis)


def test_regular_grid_renders_as_image_and_as_a_rectilinear_mercator_mesh():
    import holoviews as hv

    from xdiff.plotting.renderers.server import _field_element, _load_viz

    _load_viz()
    lon_axis, lat_axis, _, _ = _nemo_style_grid()
    variable = _variable_plot(np.ones((lat_axis.size, lon_axis.size)), lon=lon_axis, lat=lat_axis)

    assert isinstance(_field_element(hv, variable, variable.difference), hv.Image)
    mercator = _field_element(hv, variable, variable.difference, web_mercator=True)
    assert isinstance(mercator, hv.QuadMesh)
    # 1-D coordinates stay 1-D in metres: datashader's fast rectilinear path.
    assert not mercator.interface.irregular(mercator, "x")
    assert not mercator.interface.irregular(mercator, "y")


# --------------------------------------------------------------------------- live server: redraw cost
def _map_variable_plot():
    lon = np.linspace(0.0, 10.0, 6)
    lat = np.linspace(0.0, 8.0, 5)
    return _variable_plot(np.ones((5, 6)), reference=np.arange(30.0).reshape(5, 6), lon=lon, lat=lat)


def test_reference_card_is_built_only_when_opened():
    import holoviews as hv
    import panel as pn

    from xdiff.plotting.renderers.server import _load_viz, _rendered_variable

    _load_viz()
    toggles = []
    column = _rendered_variable(
        hv,
        pn,
        _map_variable_plot(),
        method="smooth",
        basemap=None,
        cmap_widget=pn.widgets.Select(options=["RdBu_r"], value="RdBu_r"),
        climit_widget=pn.widgets.FloatSlider(start=0.0, end=1.0, value=0.5),
        on_card_toggle=toggles.append,
    )
    card = column.objects[-1]
    assert card.collapsed and len(card.objects) == 0  # nothing built while collapsed

    card.collapsed = False

    assert len(card.objects) == 1 and len(card.objects[0].objects) == 2  # reference + comparison
    assert toggles == [True]


def test_reference_card_reopens_filled_when_left_open():
    import holoviews as hv
    import panel as pn

    from xdiff.plotting.renderers.server import _load_viz, _rendered_variable

    _load_viz()
    column = _rendered_variable(
        hv,
        pn,
        _map_variable_plot(),
        method="smooth",
        basemap=None,
        cmap_widget=pn.widgets.Select(options=["RdBu_r"], value="RdBu_r"),
        climit_widget=pn.widgets.FloatSlider(start=0.0, end=1.0, value=0.5),
        card_open=True,
    )

    card = column.objects[-1]
    assert not card.collapsed and len(card.objects) == 1


def test_dashboard_slices_once_per_change_and_not_on_render_toggles():
    from xdiff.plotting.renderers.server import build_dashboard

    plots = [_map_variable_plot(), _map_variable_plot()]
    calls = []
    static = PlotSource.static(Path("ref.nc"), Path("cmp.nc"), plots)
    source = PlotSource(
        Path("ref.nc"),
        Path("cmp.nc"),
        static.variables,
        lambda index, selection: calls.append(index) or plots[index],
    )

    dashboard = build_dashboard(source)
    assert calls == [0]  # one slice for the first variable, shared by slider bounds and drawing

    sidebar = dashboard.sidebar.objects
    variable_select = next(widget for widget in sidebar if getattr(widget, "name", "") == "Variable")
    render_toggle = next(widget for widget in sidebar if getattr(widget, "name", "") == "Rendering")
    render_toggle.value = "blocks"
    assert calls == [0]  # re-rendering the same slice does not read it again

    variable_select.value = 1
    assert calls == [0, 1]


def test_fields_are_sent_as_32_bit_floats():
    import holoviews as hv

    from xdiff.plotting.renderers.server import _field_element, _load_viz

    _load_viz()
    variable = _map_variable_plot()

    for web_mercator in (False, True):
        element = _field_element(hv, variable, variable.difference, web_mercator=web_mercator)
        assert element.dimension_values("value").dtype == np.float32


def test_live_source_enlarges_the_chunk_cache_while_open(tmp_path):
    import netCDF4

    before = netCDF4.get_chunk_cache()
    reference = _write(tmp_path / "ref.nc", _rectilinear_dataset(np.ones((3, 4))))
    comparison = _write(tmp_path / "cmp.nc", _rectilinear_dataset(np.ones((3, 4))))

    source = open_plot_source(reference, comparison, None, last_time_step=False, bbox=None)
    try:
        assert netCDF4.get_chunk_cache()[0] >= 1024**3
    finally:
        source.close()

    assert netCDF4.get_chunk_cache() == before


def test_datashader_warm_up_runs_and_serve_starts_it_first(monkeypatch, caplog):
    import logging

    from xdiff.plotting.renderers import server

    with caplog.at_level(logging.DEBUG, logger="xdiff"):
        server._warm_up_datashader()
        server._warm_up_views()
    # They swallow their own errors (an optimisation only), so check that none was logged.
    assert "warm-up failed" not in caplog.text

    events = []

    class FakePanel:
        @staticmethod
        def serve(*args, **kwargs):
            events.append("serve")

    monkeypatch.setattr(server, "_load_viz", lambda: (None, FakePanel))
    monkeypatch.setattr(server, "_start_warm_up", lambda: events.append("warm-up"))

    server.serve(PlotSource.static(Path("ref.nc"), Path("cmp.nc"), []), port=_free_port(), open_browser=False)

    assert events == ["warm-up", "serve"]


def test_loaded_pages_start_the_views_warm_up_once_per_server(monkeypatch):
    """The second warm-up waits for a loaded page (so it does not slow the first one) and runs once."""
    import panel as pn

    from xdiff.plotting.renderers import server

    runs = []
    monkeypatch.setattr(server, "_warm_up_views", lambda: runs.append("views"))
    monkeypatch.setattr(server, "_views_warm_up_threads", [])
    on_load = []
    monkeypatch.setattr(pn.state, "onload", on_load.append)
    application = server.build_application(_static_source(PlotSpec(Path("ref.nc"), Path("cmp.nc"), [], [])))

    application()  # two browser sessions
    application()

    assert on_load == [server._start_views_warm_up] * 2
    for callback in on_load:
        callback().join()
    assert runs == ["views"]


# --------------------------------------------------------------------------- live server: view kept across redraws
_ZOOM = ((2.0, 6.0), (1.0, 5.0))  # a lon/lat region inside the test grid


def _depth_source():
    """Two map variables on one lon/lat grid, with three depth levels.

    Level 2 holds data only west of 5°E, so its own data extent is narrower than level 0's.
    """
    from xdiff.plotting.spec import DimControl, VariableHandle

    lon = np.linspace(0.0, 10.0, 21)
    lat = np.linspace(0.0, 8.0, 17)

    def slicer(index, selection):
        difference = np.random.default_rng(index).normal(size=(lat.size, lon.size))
        if selection["depth"] == 2:
            difference[:, lon > 5.0] = np.nan
        return _variable_plot(difference, label=("thetao", "so")[index], lon=lon, lat=lat)

    depth = DimControl(name="depth", size=3, labels=("0 m", "10 m", "20 m"), default=0)
    handles = [VariableHandle(label=label, is_map=True, extra_dims=(depth,)) for label in ("thetao", "so")]
    return PlotSource(Path("ref.nc"), Path("cmp.nc"), handles, slicer)


def _live(source):
    """The dashboard, its main area, and that area rendered to Bokeh models as a served page has it."""
    from xdiff.plotting.renderers.server import build_dashboard

    dashboard = build_dashboard(source)
    main = dashboard.main.objects[1]
    return dashboard, main, main.get_root()


def _widget(dashboard, name):
    for item in dashboard.sidebar.objects:
        for widget in getattr(item, "objects", [item]):  # the time/depth selectors sit in a Column
            if getattr(widget, "name", None) == name:
                return widget
    raise LookupError(name)


def _hero_figure(root):
    """The Bokeh figure of the difference map (the reference card stays collapsed in these tests)."""
    from bokeh.models import GlyphRenderer, Plot
    from bokeh.models import Image as BkImage

    for figure in root.select({"type": Plot}):
        if any(isinstance(r, GlyphRenderer) and isinstance(r.glyph, BkImage) for r in figure.renderers):
            return figure
    raise AssertionError("no difference map rendered")


def _shown(root):
    """``(x_range, y_range)`` the difference map shows, and the ``(x_range, y_range)`` Reset returns to."""
    figure = _hero_figure(root)
    x, y = figure.x_range, figure.y_range
    return ((x.start, x.end), (y.start, y.end)), ((x.reset_start, x.reset_end), (y.reset_start, y.reset_end))


def _report_zoom(main, x_range, y_range):
    """Report a pan/zoom as a browser does: through the RangeXY callback HoloViews puts on the map."""
    import asyncio

    import holoviews as hv
    from holoviews.plotting.bokeh.callbacks import RangeXYCallback

    hero = main.objects[0].objects[1]
    plot = hv.renderer("bokeh").get_plot(hero.object)
    callbacks = [
        callback
        for subplot in plot.traverse()
        for callback in getattr(subplot, "callbacks", [])
        if isinstance(callback, RangeXYCallback)
    ]
    assert callbacks, "the difference map does not listen to pans/zooms"
    message = {"x0": x_range[0], "x1": x_range[1], "y0": y_range[0], "y1": y_range[1]}
    for callback in callbacks:
        asyncio.run(callback.on_msg(message))


def test_live_map_keeps_the_zoom_across_depth_toggle_and_variable_changes():
    dashboard, main, root = _live(_depth_source())
    _, whole_map = _shown(root)
    _report_zoom(main, *_ZOOM)

    _widget(dashboard, "depth").value = 1
    assert _shown(root) == (_ZOOM, whole_map)  # Reset still returns to the whole map

    _widget(dashboard, "Rendering").value = "blocks"
    assert _shown(root)[0] == _ZOOM

    _widget(dashboard, "Variable").value = 1  # same lon/lat grid
    assert _shown(root)[0] == _ZOOM

    # The colour controls still restyle the same figure in place.
    figure = _hero_figure(root)
    _widget(dashboard, "colour limit (±)").value = 0.3
    _widget(dashboard, "Colormap").value = "coolwarm"
    assert _hero_figure(root) is figure
    assert _shown(root)[0] == _ZOOM


def test_live_map_keeps_its_frame_when_not_zoomed():
    dashboard, _, root = _live(_depth_source())
    level_0, _ = _shown(root)

    _widget(dashboard, "depth").value = 2  # its data stops at 5°E

    shown, reset_to = _shown(root)
    assert shown == level_0  # the map does not jump to this level's narrower extent...
    assert reset_to[0][1] <= 5.0  # ...but Reset goes to it


def test_live_map_converts_the_zoom_when_a_basemap_is_toggled():
    from datashader.utils import lnglat_to_meters
    from holoviews.util.transform import easting_northing_to_lon_lat

    dashboard, main, root = _live(_depth_source())
    _report_zoom(main, *_ZOOM)

    _widget(dashboard, "Basemap").value = "CartoLight"
    (x_range, y_range), _ = _shown(root)
    eastings, northings = lnglat_to_meters(np.array(_ZOOM[0]), np.array(_ZOOM[1]))
    assert np.allclose(x_range, eastings) and np.allclose(y_range, northings)

    # A pan with the basemap on is recorded too (in metres) and converted back when it goes off.
    panned = (eastings + 50_000.0, northings)
    _report_zoom(main, *panned)
    _widget(dashboard, "Basemap").value = None
    (x_range, y_range), _ = _shown(root)
    lon, lat = easting_northing_to_lon_lat(*panned)
    assert np.allclose(x_range, lon) and np.allclose(y_range, lat)


@pytest.mark.parametrize("basemap", [None, "CartoLight"])
def test_restored_view_is_rasterized_at_full_resolution(basemap):
    """The first image covers the restored view, not the whole map stretched over it.

    With a basemap the limits must also sit on the data layer: HoloViews takes the first
    region to rasterize, and the first view it reports, from that layer, not from the overlay.
    """
    import holoviews as hv
    import panel as pn
    from bokeh.models import GlyphRenderer, Plot
    from bokeh.models import Image as BkImage
    from datashader.utils import lnglat_to_meters

    from xdiff.plotting.renderers.server import _LONLAT, _hero_map, _load_viz, _View

    _load_viz()
    variable = _variable_plot(
        np.random.default_rng(0).normal(size=(40, 80)),
        lon=np.linspace(-10.0, 20.0, 80),
        lat=np.linspace(30.0, 45.0, 40),
    )
    view = _View(_LONLAT, (0.0, 5.0), (35.0, 38.0))
    reported = []
    hero = _hero_map(
        hv,
        pn,
        variable,
        method="smooth",
        basemap=basemap,
        cmap_widget=pn.widgets.Select(options=["RdBu_r"], value="RdBu_r"),
        climit_widget=pn.widgets.FloatSlider(start=0.0, end=3.0, value=1.0),
        view=view,
        on_view=reported.append,
    )
    root = pn.Column(hero).get_root()

    expected = view.x_range if basemap is None else tuple(lnglat_to_meters(np.array(view.x_range), np.zeros(2))[0])
    images = [
        renderer.data_source.data
        for figure in root.select({"type": Plot})
        for renderer in figure.renderers
        if isinstance(renderer, GlyphRenderer) and isinstance(renderer.glyph, BkImage)
    ]
    image_x = (images[0]["x"][0], images[0]["x"][0] + images[0]["dw"][0])
    assert np.allclose(image_x, expected)
    assert np.allclose(reported[-1].x_range, expected)  # the stored view is the one shown


# The field's own rows at ±90° still project to infinity (datashader/numpy warn); only the view is clamped.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_global_map_with_a_basemap_keeps_a_finite_view_and_reset():
    """Regression: data at ±90° latitude must not send Reset (or the stored view) to infinity.

    Web Mercator puts the poles infinitely far north and south, so the extent is clamped.
    """
    import holoviews as hv
    import panel as pn

    from xdiff.plotting.renderers.server import _hero_map, _load_viz

    _load_viz()
    variable = _variable_plot(np.ones((19, 37)), lon=np.linspace(-180.0, 180.0, 37), lat=np.linspace(-90.0, 90.0, 19))
    reported = []
    hero = _hero_map(
        hv,
        pn,
        variable,
        method="smooth",
        basemap="CartoLight",
        cmap_widget=pn.widgets.Select(options=["RdBu_r"], value="RdBu_r"),
        climit_widget=pn.widgets.FloatSlider(start=0.0, end=3.0, value=1.0),
        on_view=reported.append,
    )
    shown, reset_to = _shown(pn.Column(hero).get_root())

    assert np.all(np.isfinite(shown)) and np.all(np.isfinite(reset_to))
    assert np.all(np.isfinite(reported[-1].y_range))


def test_view_carries_over_only_between_matching_coordinates():
    from xdiff.plotting.renderers.server import _INDEX, _LONLAT, _MERCATOR, _ranges_in, _View

    view = _View(_LONLAT, (0.0, 10.0), (30.0, 40.0))

    assert _ranges_in(view, _LONLAT) == (view.x_range, view.y_range)
    assert _ranges_in(view, _INDEX) is None  # a field without coordinates opens whole
    assert _ranges_in(None, _LONLAT) is None
    metres = _ranges_in(view, _MERCATOR)
    assert np.allclose(_ranges_in(_View(_MERCATOR, *metres), _LONLAT), (view.x_range, view.y_range))
    polar = _ranges_in(_View(_LONLAT, (0.0, 10.0), (80.0, 90.0)), _MERCATOR)
    assert np.all(np.isfinite(polar))  # clamped to Web Mercator's latitude limit


# --------------------------------------------------------------------------- static map style
def test_choose_scale_picks_log_when_most_differences_are_far_below_the_limit():
    sparse = np.zeros((40, 40))
    sparse[0, :20] = np.logspace(-7, 0, 20)  # a few points, differing by 1e-7 .. 1
    dense = np.random.default_rng(2).normal(scale=1e-6, size=(40, 40))

    assert choose_scale(sparse) == "log"
    assert choose_scale(dense) == "linear"
    assert choose_scale(np.zeros((4, 4))) == "linear"  # identical fields


def test_scaled_label_moves_the_power_of_ten_into_the_label():
    assert scaled_label(7.8e-6, "m") == (1e-6, "difference [×10$^{-6}$ m]")
    assert scaled_label(0.5, "degC") == (1.0, "difference [degC]")
    assert scaled_label(2e5, None) == (1e5, "difference [×10$^{5}$]")


def _map_with_land(**overrides):
    lon = np.linspace(-6.0, 4.0, 20)
    lat = np.linspace(35.0, 40.0, 12)
    difference = np.random.default_rng(3).normal(size=(12, 20))
    reference = np.zeros((12, 20))
    for array in (difference, reference):
        array[:5, :8] = np.nan  # land: no data in either file
    defaults = dict(reference=reference, comparison=reference - difference, lon=lon, lat=lat)
    defaults.update(overrides)
    return _variable_plot(difference, **defaults)


def test_static_map_draws_land_and_a_coastline_from_the_sea_mask():
    import matplotlib.colors as mcolors
    from matplotlib.contour import ContourSet

    from xdiff.plotting.renderers import matplotlib_renderer

    plt = matplotlib_renderer._load_pyplot()
    figure = matplotlib_renderer._render_diff_map(plt, _map_with_land())
    axis = figure.axes[0]

    assert mcolors.same_color(axis.get_facecolor(), matplotlib_renderer._LAND)  # undrawn cells show land
    assert any(isinstance(artist, ContourSet) for artist in axis.collections)  # traced coastline
    assert axis.xaxis.get_major_formatter()(10.0, 0) == "10°E"
    plt.close(figure)


def test_static_map_heading_and_log_colorbar():
    from xdiff.plotting.renderers import matplotlib_renderer

    plt = matplotlib_renderer._load_pyplot()
    variable = _map_with_land(
        long_name="sea water potential temperature", context=("2020-01-01 12:00", "deptht 1.02 m")
    )
    figure = matplotlib_renderer._render_diff_map(plt, variable, scale="log")
    axis, colorbar_axis = figure.axes[0], figure.axes[1]

    texts = [text.get_text() for text in axis.texts]
    assert "sea water potential temperature (sst)" in texts
    assert axis.get_title(loc="left") == "reference minus comparison   ·   2020-01-01 12:00   ·   deptht 1.02 m"
    assert "signed log scale" in colorbar_axis.get_xlabel()
    assert 0.0 in list(colorbar_axis.get_xticks())
    plt.close(figure)


def test_render_to_files_rejects_an_unknown_scale(tmp_path):
    spec = PlotSpec(tmp_path / "ref.nc", tmp_path / "cmp.nc", [_map_with_land()], [])
    with pytest.raises(ValueError, match="unknown scale"):
        render_to_files(spec, tmp_path / "diff.png", scale="cubic")


def test_build_plot_spec_describes_the_slice_for_titles(tmp_path):
    field = np.ones((2, 3, 4, 5))

    def dataset(values):
        return xr.Dataset(
            {"thetao": (("time", "depth", "lat", "lon"), values, {"long_name": "sea_water_potential_temperature"})},
            coords={
                "time": np.array(["2020-01-01T12:00", "2020-01-02T12:00"], dtype="datetime64[ns]"),
                "depth": ("depth", [1.5, 10.0, 50.0], {"units": "m"}),
                "lon": ("lon", np.arange(5.0), {"units": "degrees_east"}),
                "lat": ("lat", np.arange(4.0), {"units": "degrees_north"}),
            },
        )

    variable = _spec_for(tmp_path, dataset(field), dataset(field), variables=(("thetao", "thetao"),)).variables[0]

    assert variable.long_name == "sea water potential temperature"
    assert variable.context == ("2020-01-01 12:00", "depth 1.5 m")
