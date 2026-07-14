"""Tests for the plot feature: PlotSpec building (all the logic) and renderer smoke."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xdiff.model import BoundingBox
from xdiff.plotting.renderers.matplotlib_renderer import (
    SUPPORTED_EXTENSIONS,
    output_paths,
    render_to_files,
    validate_output_extension,
)
from xdiff.plotting.spec import (
    PlotSpec,
    SkippedVariable,
    VariablePlot,
    build_plot_spec,
    reduce_to_plottable,
)


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


def test_build_dashboard_has_one_row_per_variable():
    from xdiff.plotting.renderers.server import build_dashboard

    spec = PlotSpec(
        Path("ref.nc"),
        Path("cmp.nc"),
        [_variable_plot(np.ones((5, 5)), label="thetao"), _variable_plot(np.arange(6.0), label="profile")],
        [],
    )

    dashboard = build_dashboard(spec)

    # A header pane plus one column per variable (2-D map row and 1-D line row).
    assert len(dashboard) == 1 + len(spec.variables)


def test_build_dashboard_constructs_curvilinear_quadmesh():
    from xdiff.plotting.renderers.server import build_dashboard

    lon2d = np.tile(np.arange(4.0), (3, 1))
    lat2d = np.tile(np.arange(3.0)[:, None], (1, 4))
    variable = _variable_plot(np.ones((3, 4)), lon=lon2d, lat=lat2d)
    dashboard = build_dashboard(PlotSpec(Path("ref.nc"), Path("cmp.nc"), [variable], []))

    assert len(dashboard) == 2  # header + one row


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
