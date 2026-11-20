import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xdiff.conf as settings
from xdiff.comparators.netcdf import (
    compare_datasets,
    compare_files,
    compare_variables,
    compute_relative_error,
    crop_to_bbox,
    find_time_dims_name,
    get_dataset_variables,
    locate_horizontal_coords,
    select_last_time_step,
)
from xdiff.compare import compare
from xdiff.exceptions import LastTimestepTimeCheckException
from xdiff.model import BoundingBox, CompareResult


def make_data_array(values, dims=("x",), dtype=None):
    array = np.array(values, dtype=dtype) if dtype is not None else np.array(values)
    return xr.DataArray(array, dims=dims)


def write_dataset(tmp_path, name, dataset):
    path = tmp_path / name
    dataset.to_netcdf(path)
    return path


def test_get_dataset_variables_defaults_keep_text_variables():
    dataset = xr.Dataset(
        data_vars={
            "temp": ("time", [1.0, 2.0]),
            "label": ("time", np.array(["a", "b"], dtype=str)),
        },
        coords={"time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")},
    )

    variables = get_dataset_variables(dataset, settings.DEFAULT_VARIABLES_TO_CHECK)

    assert variables == [("temp", "temp"), ("label", "label"), ("time", "time")]


def test_get_dataset_variables_respects_explicit_variable_selection():
    dataset = xr.Dataset(
        data_vars={
            "temp": ("x", [1.0, 2.0]),
            "label": ("x", np.array(["a", "b"], dtype=str)),
        }
    )

    variables = get_dataset_variables(dataset, ["label", "temp", "missing"])

    assert variables == [("label", "label"), ("temp", "temp")]


def test_find_time_dims_name_returns_single_time_dimension():
    assert find_time_dims_name(("time_counter", "depth")) == "time_counter"


def test_find_time_dims_name_raises_for_multiple_time_dimensions():
    with pytest.raises(ValueError, match="Found more than 1 time dimension"):
        find_time_dims_name(("time", "time_counter"))


def test_select_last_time_step_keeps_only_last_entry():
    field = make_data_array([1.0, 2.0, 3.0], dims=("time",))

    result = select_last_time_step(field)

    assert result.shape == (1,)
    assert result.values.tolist() == [3.0]


def test_select_last_time_step_uses_named_time_dimension_when_not_first():
    field = make_data_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dims=("depth", "time"))

    result = select_last_time_step(field)

    assert result.shape == (2, 1)
    assert result.values.tolist() == [[3.0], [6.0]]


def test_compute_relative_error_returns_zero_for_identical_arrays():
    diff = np.array([0.0, 0.0])
    reference = np.array([2.0, 4.0])

    assert compute_relative_error(diff, reference) == 0.0


def test_compute_relative_error_divides_by_the_reference():
    # |diff| / |reference| is 2/4 and 1/8; dividing by the comparison (2 and 7) would give 1.0.
    reference = np.array([4.0, 8.0])
    comparison = np.array([2.0, 7.0])

    assert compute_relative_error(reference - comparison, reference) == 0.5


def test_compute_relative_error_ignores_points_that_do_not_differ():
    # 0 - 0 and NaN (missing on either side) are left out instead of making the result NaN.
    diff = np.array([0.0, np.nan, 1.0, np.nan])
    reference = np.array([0.0, np.nan, 4.0, 2.0])

    assert compute_relative_error(diff, reference) == 0.25


@pytest.mark.parametrize(
    "diff",
    [np.array([0.0, 0.0, np.nan]), np.array([np.nan, np.nan, np.nan])],
)
def test_compute_relative_error_is_zero_when_no_point_differs(diff):
    reference = np.array([0.0, 0.0, np.nan])

    assert compute_relative_error(diff, reference) == 0.0


def test_compute_relative_error_is_inf_for_a_difference_over_a_zero_reference():
    diff = np.array([1.0, 0.5])
    reference = np.array([0.0, 2.0])

    assert compute_relative_error(diff, reference) == np.inf


@pytest.mark.parametrize(("comparison_seconds", "expected"), [(1, np.nan), (2, 0.0)])
def test_compute_relative_error_has_none_for_dates(comparison_seconds, expected):
    reference = np.array(["2024-01-01T00:00:02", "NaT"], dtype="datetime64[ns]")
    comparison = np.array([f"2024-01-01T00:00:0{comparison_seconds}", "NaT"], dtype="datetime64[ns]")

    result = compute_relative_error(reference - comparison, reference)

    np.testing.assert_equal(result, expected)


def test_compute_relative_error_divides_durations_into_a_plain_ratio():
    reference = np.array([4, 10, 0], dtype="timedelta64[h]")
    comparison = np.array([3, 10, 0], dtype="timedelta64[h]")

    result = compute_relative_error(reference - comparison, reference)

    assert result == 0.25
    assert isinstance(result, float)


def test_compute_relative_error_is_inf_for_a_duration_over_a_zero_reference():
    reference = np.array([0, 4], dtype="timedelta64[s]")
    comparison = np.array([1, 4], dtype="timedelta64[s]")

    assert compute_relative_error(reference - comparison, reference) == np.inf


def test_compute_relative_error_takes_the_absolute_value_of_an_integer_reference_in_float():
    # abs(int8(-128)) overflows back to -128 in int8.
    reference = np.array([-128, 10], dtype="int8")
    diff = np.array([-64.0, 0.0])

    assert compute_relative_error(diff, reference) == 0.5


@pytest.mark.parametrize(
    ("diff", "reference"),
    [
        (np.array([1.0, 0.0, np.nan]), np.array([0.0, 0.0, np.nan])),
        (np.array([np.nan, np.nan]), np.array([np.nan, np.nan])),
        (np.array([1e300]), np.array([1e-300])),
        (np.array([1, 0], dtype="timedelta64[s]"), np.array([0, 0], dtype="timedelta64[s]")),
        (np.array([1, 0], dtype="timedelta64[s]"), np.array(["2024-01-01", "NaT"], dtype="datetime64[s]")),
    ],
)
def test_compute_relative_error_emits_no_runtime_warning(diff, reference):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compute_relative_error(diff, reference)


@pytest.mark.parametrize(
    ("reference", "comparison", "expected"),
    [
        # Nothing differs in the first block; the largest ratio is in the second, and the
        # smaller one in the last block must not replace it.
        ([1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 10.0], [1.0, 2.0, 3.0, 2.0, 4.0, 4.0, 9.0], 0.5),
        # A zero reference in the last, partial block.
        ([1.0, 1.0, 1.0, 2.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0], np.inf),
        # inf/inf in a later block wins over an earlier finite ratio, as without blocks.
        ([2.0, 1.0, 1.0, np.inf], [1.0, 1.0, 1.0, 1.0], np.nan),
        # Points that differ only through NaN, in every block.
        ([np.nan, 1.0, 1.0, np.nan, 2.0], [1.0, 1.0, 1.0, 1.0, 2.0], 0.0),
    ],
)
def test_compute_relative_error_gives_the_same_maximum_block_by_block(monkeypatch, reference, comparison, expected):
    reference, comparison = np.array(reference), np.array(comparison)
    with np.errstate(invalid="ignore"):
        diff = reference - comparison
    whole = compute_relative_error(diff, reference)

    monkeypatch.setattr("xdiff.comparators.netcdf.RELATIVE_ERROR_BLOCK_SIZE", 3)
    blocked = compute_relative_error(diff.reshape(-1, 1), reference.reshape(-1, 1))

    np.testing.assert_equal([whole, blocked], [expected, expected])


def test_compute_relative_error_broadcasts_the_reference_like_a_division():
    diff = np.array([[1.0, 2.0], [0.0, 3.0]])

    assert compute_relative_error(diff, np.array([4.0, 6.0])) == 0.5


def test_compare_variables_gives_durations_their_ratio_and_differences(tmp_path):
    hours = np.array([6, 1, 2], dtype="timedelta64[h]").astype("timedelta64[ns]")
    xr.Dataset({"duration": ("x", hours)}).to_netcdf(tmp_path / "ref.nc")
    xr.Dataset({"duration": ("x", hours - np.array([3, 0, 0], dtype="timedelta64[h]"))}).to_netcdf(tmp_path / "cmp.nc")
    # Read back through netCDF, as in a real run: xarray decodes the variable as timedelta64.
    with xr.open_dataset(tmp_path / "ref.nc") as reference, xr.open_dataset(tmp_path / "cmp.nc") as comparison:
        assert reference["duration"].dtype.kind == "m"
        result = compare_variables(reference["duration"], comparison["duration"], "duration", last_time_step=False)
        identical = compare_variables(reference["duration"], reference["duration"], "duration", last_time_step=False)

    # numpy counts timedelta64 as an integer type; converting it to float like an integer
    # made the ratio fail with "ufunc 'divide' cannot use operands with types float64 and m8[ns]".
    assert result.description == "-"
    assert result.relative_error == 0.5
    assert result.max_diff == np.timedelta64(3, "h")
    assert result.min_diff == np.timedelta64(0, "h")
    assert not result.passed
    assert identical.passed


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_compare_variables_reports_durations_in_any_unit(unit):
    # xarray reads durations back in the unit they were written in, e.g. pandas' [s].
    hours = np.array([6, 1, 2], dtype="timedelta64[h]").astype(f"timedelta64[{unit}]")
    reference = xr.DataArray(hours, dims="x")
    comparison = xr.DataArray(hours - np.array([3, 0, 0], dtype="timedelta64[h]"), dims="x")

    result = compare_variables(reference, comparison, "duration", last_time_step=False)
    identical = compare_variables(reference, reference.copy(), "duration", last_time_step=False)

    # float() of a timedelta64 coarser than ns gives a datetime.timedelta, so reading
    # .passed used to raise and abort the run.
    assert identical.passed
    assert not result.passed
    assert result.relative_error == 0.5
    assert result.max_diff == np.timedelta64(3, "h")


def test_compare_variables_divides_the_relative_error_by_the_reference():
    reference = make_data_array([4.0, 8.0])
    comparison = make_data_array([2.0, 7.0])

    result = compare_variables(reference, comparison, "temp", last_time_step=False)

    assert result.relative_error == 0.5


def test_compare_variables_passes_identical_fields_of_zeros_and_nan():
    # 0/0 at the zeros used to make the relative error NaN, which failed identical fields.
    reference = make_data_array([0.0, 0.0, np.nan])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = compare_variables(reference, reference.copy(), "sea_ice", last_time_step=False)

    assert result.passed
    assert result.relative_error == 0.0


def test_compare_variables_reports_inf_where_the_reference_is_zero():
    reference = make_data_array([0.0, 2.0])
    comparison = make_data_array([1.0, 2.0])

    result = compare_variables(reference, comparison, "Cd", last_time_step=False)

    assert result.relative_error == np.inf
    assert not result.passed


def test_compare_variables_returns_zeroed_result_for_identical_fields():
    reference = make_data_array([1.0, 2.0, 3.0])
    comparison = make_data_array([1.0, 2.0, 3.0])

    result = compare_variables(reference, comparison, "temp", last_time_step=False)

    assert result == CompareResult(
        relative_error=0.0,
        min_diff=0.0,
        max_diff=0.0,
        mask_equal=True,
        variable="temp",
        description="-",
    )


def test_compare_variables_raises_on_dimension_size_mismatch():
    reference = make_data_array([1.0, 2.0], dims=("x",))
    comparison = make_data_array([[1.0, 2.0]], dims=("x", "y"))

    with pytest.raises(ValueError, match="Dimension size mismatch"):
        compare_variables(reference, comparison, "temp", last_time_step=False)


def test_compare_variables_allows_dimension_name_mismatch_when_shape_matches():
    reference = make_data_array([1.0, 2.0], dims=("x",))
    comparison = make_data_array([3.0, 2.0], dims=("y",))

    result = compare_variables(reference, comparison, "temp", last_time_step=False)

    assert result.min_diff == -2.0
    assert result.max_diff == 0.0
    assert result.variable == "temp"


def test_compare_variables_allows_coordinate_value_mismatches():
    reference = xr.DataArray([1.0, 3.0], dims=("time",), coords={"time": [0, 1]})
    comparison = xr.DataArray([1.0, 2.0], dims=("time",), coords={"time": [1, 2]})

    result = compare_variables(reference, comparison, "temp", last_time_step=False)

    assert result.min_diff == 0.0
    assert result.max_diff == 1.0
    assert result.relative_error == pytest.approx(1 / 3)


def test_compare_variables_treats_matching_all_nan_fields_as_identical():
    reference = make_data_array([np.nan, np.nan])
    comparison = make_data_array([np.nan, np.nan])

    result = compare_variables(reference, comparison, "temp", last_time_step=False)

    assert result.passed
    assert result.min_diff == 0.0
    assert result.max_diff == 0.0
    assert result.mask_equal is True
    assert "only NaN" in result.note


def test_compare_variables_fails_when_all_nan_but_masks_differ():
    reference = make_data_array([1.0, np.nan])
    comparison = make_data_array([np.nan, 2.0])

    result = compare_variables(reference, comparison, "temp", last_time_step=False)

    assert not result.passed
    assert result.mask_equal is False
    assert "no overlapping valid points" in result.description


def test_compare_variables_treats_empty_fields_as_identical():
    reference = make_data_array([], dtype="float64")
    comparison = make_data_array([], dtype="float64")

    result = compare_variables(reference, comparison, "temp", last_time_step=False)

    assert result.passed
    assert "empty" in result.note


def test_compare_variables_detects_mask_mismatch():
    reference = make_data_array([1.0, np.nan])
    comparison = make_data_array([1.0, 2.0])

    result = compare_variables(reference, comparison, "temp", last_time_step=False)

    assert result.mask_equal is False


def test_compare_variables_does_not_wrap_around_on_unsigned_integers():
    # Regression: uint8 subtraction wraps (10 - 12 -> 254), which used to
    # corrupt the diff metrics. The diff must reflect the true signed values.
    reference = make_data_array([10, 20, 30], dtype="uint8")
    comparison = make_data_array([12, 18, 35], dtype="uint8")

    result = compare_variables(reference, comparison, "counts", last_time_step=False)

    assert result.min_diff == -5.0
    assert result.max_diff == 2.0
    assert result.relative_error == pytest.approx(2 / 10)


@pytest.mark.parametrize(
    ("comparison_values", "passed", "description", "note"),
    [
        ([True, False, True], True, "-", "equality only"),
        ([True, True, True], False, "1 of 3 values differ", ""),
    ],
)
def test_compare_variables_compares_booleans_for_equality(comparison_values, passed, description, note):
    # numpy refuses to subtract booleans, so identical flags used to fail with its error.
    reference = make_data_array([True, False, True], dtype=bool)
    comparison = make_data_array(comparison_values, dtype=bool)

    result = compare_variables(reference, comparison, "flag", last_time_step=False)

    assert (result.passed, result.description, result.note) == (passed, description, note)
    assert result.mask_equal


@pytest.mark.parametrize(("value", "passed"), [(True, True), (False, False)])
def test_compare_variables_compares_a_scalar_boolean(value, passed):
    result = compare_variables(
        make_data_array(True, dims=(), dtype=bool),
        make_data_array(value, dims=(), dtype=bool),
        "flag",
        last_time_step=False,
    )

    assert result.passed is passed
    assert result.description == ("-" if passed else "1 of 1 values differ")


def test_compare_variables_fails_on_booleans_against_integers():
    reference = make_data_array([True, False], dtype=bool)
    comparison = make_data_array([1, 0], dtype="int8")

    with pytest.raises(ValueError, match="bool"):
        compare_variables(reference, comparison, "flag", last_time_step=False)


def test_compare_variables_rejects_time_variables_when_last_time_step_is_enabled():
    reference = make_data_array(
        ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
        dims=("time_counter",),
        dtype="datetime64[ns]",
    )
    comparison = make_data_array(
        ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
        dims=("time_counter",),
        dtype="datetime64[ns]",
    )

    with pytest.raises(
        LastTimestepTimeCheckException,
        match="Can't compare time if last time step is enabled",
    ):
        compare_variables(reference, comparison, "time_counter", last_time_step=True)


def test_compare_variables_allows_non_time_variables_with_time_in_the_name():
    reference = make_data_array([1.0, 2.0], dims=("x",))
    comparison = make_data_array([1.0, 2.0], dims=("x",))

    result = compare_variables(reference, comparison, "runtime_bias", last_time_step=True)

    assert result.passed is True


def test_compare_variables_rejects_mapped_time_coordinate_with_last_time_step():
    # A mapped time axis has differently-named dims on each side; the guard must
    # still fire (detection is by dtype/shape, not by name).
    values = ["2024-01-01T00:00:00", "2024-01-02T00:00:00"]
    reference = make_data_array(values, dims=("time",), dtype="datetime64[ns]")
    comparison = make_data_array(values, dims=("time_counter",), dtype="datetime64[ns]")

    with pytest.raises(LastTimestepTimeCheckException, match="Can't compare time"):
        compare_variables(reference, comparison, "time -> time_counter", last_time_step=True)


def test_compare_datasets_records_variable_level_errors():
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0])})
    comparison = xr.Dataset()

    results = compare_datasets(reference, comparison, ["temp"], last_time_step=False)

    assert len(results) == 1
    assert results[0].variable == "temp"
    assert results[0].description != "-"


def test_compare_datasets_maps_differently_named_variables():
    reference = xr.Dataset({"thetao": ("x", [1.0, 2.0, 3.0])})
    comparison = xr.Dataset({"votemper": ("x", [1.0, 2.0, 3.0])})

    results = compare_datasets(reference, comparison, [("thetao", "votemper")], last_time_step=False)

    assert len(results) == 1
    assert results[0].passed
    assert results[0].variable == "thetao -> votemper"


def test_compare_datasets_reports_missing_mapped_comparison_variable():
    reference = xr.Dataset({"thetao": ("x", [1.0, 2.0])})
    comparison = xr.Dataset({"other": ("x", [1.0, 2.0])})

    results = compare_datasets(reference, comparison, [("thetao", "votemper")], last_time_step=False)

    assert len(results) == 1
    assert not results[0].passed
    assert results[0].variable == "thetao -> votemper"
    assert results[0].description == "Not in the comparison file"


def test_compare_files_reads_netcdf_inputs(tmp_path):
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0])})
    comparison = xr.Dataset({"temp": ("x", [1.0, 2.0])})
    reference_path = write_dataset(tmp_path, "reference.nc", reference)
    comparison_path = write_dataset(tmp_path, "comparison.nc", comparison)

    results = compare_files(reference_path, comparison_path, ["temp"], last_time_step=False)

    assert len(results) == 1
    assert results[0].variable == "temp"
    assert results[0].relative_error == 0.0


def _compare_default(tmp_path, reference, comparison, *, last_time_step=False):
    """Write both datasets and compare them with no variable selection (no -v)."""
    reference_path = write_dataset(tmp_path, "reference.nc", reference)
    comparison_path = write_dataset(tmp_path, "comparison.nc", comparison)
    return compare_files(reference_path, comparison_path, None, last_time_step=last_time_step)


def _by_variable(results):
    return {result.variable: result for result in results}


def _nemo_grid(longitude_shift=0.0):
    """A 2-D field on a NEMO-like grid: nav_lon/nav_lat are auxiliary coordinates on y, x."""
    longitudes, latitudes = np.meshgrid(np.arange(4.0), np.arange(3.0))
    return xr.Dataset(
        {"votemper": (("y", "x"), np.ones((3, 4)))},
        coords={"nav_lon": (("y", "x"), longitudes + longitude_shift), "nav_lat": (("y", "x"), latitudes)},
    )


def test_compare_files_default_selection_covers_both_files_in_a_fixed_order(tmp_path):
    reference = _nemo_grid().assign(gone=("y", np.zeros(3)))
    comparison = _nemo_grid().assign(extra=("x", np.zeros(4)), level=("z", [1.0]))

    results = _compare_default(tmp_path, reference, comparison)

    # Shared variables (data variables, then coordinates), dimensions without a
    # coordinate variable, then what is in one file only: the reference's, then the
    # comparison's, each with variables before dimensions.
    assert [result.variable for result in results] == [
        "votemper",
        "nav_lon",
        "nav_lat",
        "y",
        "x",
        "gone",
        "extra",
        "level",
        "z",
    ]


def test_compare_files_fails_on_a_variable_only_in_the_comparison_file(tmp_path):
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0])})
    comparison = xr.Dataset({"temp": ("x", [1.0, 2.0]), "salt": ("x", [3.0, 4.0])})

    results = _by_variable(_compare_default(tmp_path, reference, comparison))

    assert results["temp"].passed
    assert not results["salt"].passed
    assert results["salt"].description == "Only in the comparison file"


def test_compare_files_fails_on_a_variable_only_in_the_reference_file(tmp_path):
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0]), "salt": ("x", [3.0, 4.0])})
    comparison = xr.Dataset({"temp": ("x", [1.0, 2.0])})

    results = _by_variable(_compare_default(tmp_path, reference, comparison))

    assert results["temp"].passed
    assert not results["salt"].passed
    assert results["salt"].description == "Only in the reference file"


def test_compare_files_compares_text_variables_and_reports_a_one_sided_one(tmp_path):
    labels = np.array(["a", "b"], dtype=str)
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0]), "label": ("x", labels), "name": ("x", labels)})
    comparison = xr.Dataset({"temp": ("x", [1.0, 2.0]), "label": ("x", labels)})

    results = _by_variable(_compare_default(tmp_path, reference, comparison))

    assert results["label"].passed
    assert results["label"].note == "equality only"
    assert results["name"].description == "Only in the reference file"


@pytest.mark.parametrize("dtype", [str, bytes], ids=["str", "bytes"])
def test_compare_files_counts_the_text_values_that_differ(tmp_path, dtype):
    # Different longest values give different widths (<U2 vs <U3): still compared by value.
    # bytes are written as a netCDF char array and read back as |S.
    reference = xr.Dataset({"label": ("x", np.array(["a", "bb", "c"], dtype=dtype))})
    comparison = xr.Dataset({"label": ("x", np.array(["a", "bbb", "d"], dtype=dtype))})

    results = _by_variable(_compare_default(tmp_path, reference, comparison))

    assert not results["label"].passed
    assert results["label"].description == "2 of 3 values differ"
    assert results["label"].mask_equal  # nothing is missing on either side


def test_compare_files_compares_text_read_back_as_object(tmp_path):
    # A _FillValue makes xarray read text back as an object array, the same text without
    # one as <U: the stored form differs, the values do not.
    dataset = xr.Dataset({"label": ("x", np.array(["a", "b"], dtype=str))})
    reference_path = tmp_path / "reference.nc"
    dataset.to_netcdf(reference_path, encoding={"label": {"_FillValue": "zz"}})
    comparison_path = write_dataset(tmp_path, "comparison.nc", dataset)

    results = compare_files(reference_path, comparison_path, None, last_time_step=False)

    assert xr.open_dataset(reference_path)["label"].dtype == object
    assert _by_variable(results)["label"].passed


@pytest.mark.parametrize(
    ("comparison_values", "description", "mask_equal"),
    [
        (["a", "zz", "b"], "-", True),  # missing at the same place: equal
        (["a", "zz", "c"], "1 of 3 values differ", True),
        (["a", "x", "b"], "1 of 3 values differ", False),  # missing on one side only
    ],
)
def test_compare_files_text_missing_values(tmp_path, comparison_values, description, mask_equal):
    reference_path, comparison_path = tmp_path / "reference.nc", tmp_path / "comparison.nc"
    for path, values in ((reference_path, ["a", "zz", "b"]), (comparison_path, comparison_values)):
        dataset = xr.Dataset({"label": ("x", np.array(values, dtype=str))})
        dataset.to_netcdf(path, encoding={"label": {"_FillValue": "zz"}})  # "zz" is read back as missing

    result = _by_variable(compare_files(reference_path, comparison_path, None, last_time_step=False))["label"]

    assert result.description == description
    assert result.mask_equal is mask_equal


def test_compare_files_classifies_object_text_by_its_first_present_value(tmp_path):
    # The reference's first value is missing (NaN in an object array): its kind (str) comes
    # from the next value, so bytes on the other side still fail as a data type mismatch.
    reference_path = tmp_path / "reference.nc"
    xr.Dataset({"label": ("x", np.array(["zz", "a"], dtype=str))}).to_netcdf(
        reference_path, encoding={"label": {"_FillValue": "zz"}}
    )
    comparison_path = write_dataset(tmp_path, "comparison.nc", xr.Dataset({"label": ("x", np.array([b"x", b"a"]))}))

    result = _by_variable(compare_files(reference_path, comparison_path, None, last_time_step=False))["label"]

    assert result.description == "Data type mismatch: 'object (str)' - '|S1'"


def test_compare_files_compares_all_missing_text_with_present_text(tmp_path):
    # Nothing to tell the kind of an all-missing object array: it is compared by value.
    reference_path = tmp_path / "reference.nc"
    xr.Dataset({"label": ("x", np.array(["zz", "zz"], dtype=str))}).to_netcdf(
        reference_path, encoding={"label": {"_FillValue": "zz"}}
    )
    comparison_path = write_dataset(
        tmp_path, "comparison.nc", xr.Dataset({"label": ("x", np.array(["a", "b"], dtype=str))})
    )

    result = _by_variable(compare_files(reference_path, comparison_path, None, last_time_step=False))["label"]

    assert result.description == "2 of 2 values differ"
    assert not result.mask_equal


def test_compare_variables_compares_other_objects_for_equality():
    reference = make_data_array([1, (2, 3)], dtype=object)

    assert compare_variables(reference, make_data_array([1, (2, 3)], dtype=object), "v", last_time_step=False).passed
    different = compare_variables(reference, make_data_array([1, (2, 4)], dtype=object), "v", last_time_step=False)
    assert different.description == "1 of 2 values differ"


@pytest.mark.parametrize(("comparison_dtype", "passed"), [(str, True), (bytes, False)], ids=["str", "bytes"])
def test_compare_files_text_with_a_bounding_box(tmp_path, comparison_dtype, passed):
    # A curvilinear crop leaves text as it is (masking would turn it into object arrays):
    # str must still match str, and bytes must still fail against str.
    labels = np.full((3, 4), "a")
    reference = _nemo_grid().assign(label=(("y", "x"), labels.astype(str)))
    comparison = _nemo_grid().assign(label=(("y", "x"), labels.astype(comparison_dtype)))
    reference_path = write_dataset(tmp_path, "reference.nc", reference)
    comparison_path = write_dataset(tmp_path, "comparison.nc", comparison)
    box = BoundingBox(lon_min=0.5, lon_max=2.5, lat_min=0.5, lat_max=1.5)

    result = _by_variable(compare_files(reference_path, comparison_path, None, last_time_step=False, bbox=box))["label"]

    assert result.passed is passed
    if not passed:
        assert result.description == "Data type mismatch: '<U1' - '|S1'"


def test_compare_files_compares_a_text_variable_named_with_v(tmp_path):
    reference = xr.Dataset({"label": ("x", np.array(["a", "b"], dtype=str))})
    comparison = xr.Dataset({"label": ("x", np.array(["a", "c"], dtype=str))})
    reference_path = write_dataset(tmp_path, "reference.nc", reference)
    comparison_path = write_dataset(tmp_path, "comparison.nc", comparison)

    results = compare_files(reference_path, comparison_path, [("label", "label")], last_time_step=False)

    assert [(result.variable, result.description) for result in results] == [("label", "1 of 2 values differ")]


@pytest.mark.parametrize("dtype", [None, object], ids=["text", "object"])
def test_compare_variables_fails_on_bytes_against_str(dtype):
    # Also when both are held in object arrays, as xarray reads text with a _FillValue.
    reference = make_data_array([b"a", b"b"], dtype=dtype)
    comparison = make_data_array(["a", "b"], dtype=dtype)

    with pytest.raises(ValueError, match="Data type mismatch") as error:
        compare_variables(reference, comparison, "label", last_time_step=False)

    if dtype is object:
        assert "'object (bytes)' - 'object (str)'" in str(error.value)


def test_compare_variables_counts_a_value_missing_on_both_sides_as_equal():
    reference = make_data_array(["a", None, np.nan], dtype=object)
    same_values = make_data_array(["a", None, None], dtype=object)
    other_values = make_data_array(["a", "b", None], dtype=object)

    same = compare_variables(reference, same_values, "label", last_time_step=False)
    different = compare_variables(reference, other_values, "label", last_time_step=False)

    assert same.passed
    assert not different.passed
    assert different.description == "1 of 3 values differ"
    assert not different.mask_equal


def _noleap_run(start="2000-01-01", steps=2, calendar="noleap"):
    """A run whose time axis uses a non-standard calendar: xarray decodes it to cftime dates."""
    times = xr.date_range(start, periods=steps, freq="D", calendar=calendar, use_cftime=True)
    return xr.Dataset(
        {"temp": (("time", "x"), np.arange(2.0 * steps).reshape(steps, 2) - 2.0 * steps)},
        coords={"time": times, "time_centered": ("time", times + pd.Timedelta(hours=12))},
    )


def test_compare_files_compares_cftime_dates(tmp_path):
    same = _by_variable(_compare_default(tmp_path, _noleap_run(), _noleap_run()))
    later = _by_variable(_compare_default(tmp_path, _noleap_run(), _noleap_run(start="2000-01-05")))

    assert same["time"].passed and same["time_centered"].passed
    assert same["time"].note == "equality only"
    assert later["time"].description == "2 of 2 values differ"
    assert later["time_centered"].description == "2 of 2 values differ"


def test_compare_files_fails_on_cftime_dates_in_different_calendars(tmp_path):
    results = _by_variable(_compare_default(tmp_path, _noleap_run(), _noleap_run(calendar="360_day")))

    assert results["time"].description == "Data type mismatch: 'object (cftime noleap)' - 'object (cftime 360_day)'"


@pytest.mark.parametrize("last_time_step", [False, True])
def test_compare_files_fails_on_cftime_dates_against_datetime64(tmp_path, last_time_step):
    # A calendar change between the files (noleap -> standard) is reported, also under
    # --last-time-step, which only leaves out a time axis of the same kind on both sides.
    standard = _noleap_run(calendar="standard").assign_coords(
        time=pd.to_datetime(["2000-01-01", "2000-01-02"]),
        time_centered=("time", pd.to_datetime(["2000-01-01 12:00", "2000-01-02 12:00"])),
    )

    results = _by_variable(_compare_default(tmp_path, _noleap_run(), standard, last_time_step=last_time_step))

    assert results["time"].description == "Data type mismatch: 'object (cftime noleap)' - 'datetime64[ns]'"


def test_compare_files_last_time_step_skips_cftime_time_axes(tmp_path):
    # Runs of different lengths with the same last step, on a noleap calendar.
    reference_path = write_dataset(tmp_path, "reference.nc", _noleap_run(start="2000-01-02", steps=2))
    comparison_path = write_dataset(tmp_path, "comparison.nc", _noleap_run(start="2000-01-01", steps=3))

    by_default = compare_files(reference_path, comparison_path, None, last_time_step=True)
    explicit = compare_files(reference_path, comparison_path, [("time", "time")], last_time_step=True)

    assert [result.variable for result in by_default] == ["temp", "x"]
    assert all(result.passed for result in by_default)
    assert "last time step" in explicit[0].description


def test_compare_files_last_time_step_compares_text_along_time(tmp_path):
    # Text along time, read back as an object array (it has a _FillValue): it is not a
    # time axis, so --last-time-step compares its last step instead of leaving it out.
    reference_path, comparison_path = tmp_path / "reference.nc", tmp_path / "comparison.nc"
    for path, values in ((reference_path, ["a", "b"]), (comparison_path, ["a", "c"])):
        dataset = xr.Dataset({"event": ("time", np.array(values, dtype=str))})
        dataset.to_netcdf(path, encoding={"event": {"_FillValue": "zz"}})

    results = _by_variable(compare_files(reference_path, comparison_path, None, last_time_step=True))

    assert results["event"].description == "1 of 1 values differ"


@pytest.mark.parametrize("text_side", ["reference", "comparison"])
def test_compare_files_fails_on_a_variable_that_is_text_in_one_file_only(tmp_path, text_side):
    text = xr.Dataset({"flag": ("x", np.array(["a", "b"], dtype=str))})
    numbers = xr.Dataset({"flag": ("x", [1.0, 2.0])})
    reference, comparison = (text, numbers) if text_side == "reference" else (numbers, text)

    results = _by_variable(_compare_default(tmp_path, reference, comparison))

    # Skipped only when it is text in both files; swapping the files gives the same verdict.
    assert not results["flag"].passed
    assert "Data type mismatch" in results["flag"].description


def test_compare_files_matches_a_data_variable_with_a_coordinate_of_the_same_name(tmp_path):
    # The same values stored as a data variable in one file and as a coordinate in the
    # other: one row, compared by value, not two one-sided rows.
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0]), "lon": ("x", [10.0, 20.0])})
    comparison = xr.Dataset({"temp": ("x", [1.0, 2.0])}, coords={"lon": ("x", [10.0, 20.0])})

    results = _compare_default(tmp_path, reference, comparison)

    assert [result.variable for result in results] == ["temp", "lon", "x"]
    assert all(result.passed for result in results)


def test_compare_files_compares_a_scalar_coordinate(tmp_path):
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0])}, coords={"depth": 5.0})
    comparison = xr.Dataset({"temp": ("x", [1.0, 2.0])}, coords={"depth": 6.0})

    results = _by_variable(_compare_default(tmp_path, reference, comparison))

    assert results["temp"].passed
    assert not results["depth"].passed
    assert results["depth"].max_diff == -1.0


def test_compare_files_detects_a_shifted_auxiliary_coordinate(tmp_path):
    # NEMO layout: xarray decodes 2-D nav_lon/nav_lat as coordinates, not data variables.
    results = _by_variable(_compare_default(tmp_path, _nemo_grid(), _nemo_grid(longitude_shift=5.0)))

    assert results["votemper"].passed
    assert results["nav_lat"].passed
    assert not results["nav_lon"].passed
    assert results["nav_lon"].min_diff == -5.0


def test_compare_files_compares_the_size_of_a_dimension_without_coordinate(tmp_path):
    results = _by_variable(_compare_default(tmp_path, _nemo_grid(), _nemo_grid()))

    for dimension in ("y", "x"):
        assert results[dimension].passed
        assert results[dimension].note == "size only"


def test_compare_files_fails_on_a_dimension_size_mismatch(tmp_path):
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0, 3.0])})
    comparison = xr.Dataset({"temp": ("x", [1.0, 2.0, 3.0, 4.0])})

    results = _by_variable(_compare_default(tmp_path, reference, comparison))

    assert not results["x"].passed
    assert results["x"].description == "Dimension size mismatch: 3 - 4"


def test_compare_files_fails_on_a_renamed_dimension_without_coordinate(tmp_path):
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0])})
    comparison = xr.Dataset({"temp": ("i", [1.0, 2.0])})

    results = _by_variable(_compare_default(tmp_path, reference, comparison))

    # The values match (a dimension name mismatch alone does not fail a variable), but
    # the files are laid out differently.
    assert results["temp"].passed
    assert results["x"].description == "Dimension only in the reference file"
    assert results["i"].description == "Dimension only in the comparison file"


@pytest.mark.parametrize("dimension_side", ["reference", "comparison"])
def test_compare_files_tells_a_variable_from_a_dimension_of_the_same_name(tmp_path, dimension_side):
    # A scalar variable "depth" in one file, a dimension "depth" without a coordinate
    # variable in the other: each row says what it is, so the two do not read as a
    # contradiction.
    with_variable = xr.Dataset({"temp": ("x", [1.0, 2.0]), "depth": 5.0})
    with_dimension = xr.Dataset({"temp": ("x", [1.0, 2.0]), "column": ("depth", [1.0])})
    if dimension_side == "reference":
        reference, comparison, variable_side = with_dimension, with_variable, "comparison"
    else:
        reference, comparison, variable_side = with_variable, with_dimension, "reference"

    results = _compare_default(tmp_path, reference, comparison)

    depth_rows = sorted(result.description for result in results if result.variable == "depth")
    assert depth_rows == sorted([f"Only in the {variable_side} file", f"Dimension only in the {dimension_side} file"])


def test_compare_files_reports_a_coordinate_variable_only_in_the_comparison_file(tmp_path):
    # Both files have dimension x, but only the comparison has a coordinate variable for
    # it: one row for that variable, no size row.
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0])})
    comparison = xr.Dataset({"temp": ("x", [1.0, 2.0])}, coords={"x": [10.0, 20.0]})

    results = _compare_default(tmp_path, reference, comparison)

    assert [(result.variable, result.description) for result in results] == [
        ("temp", "-"),
        ("x", "Only in the comparison file"),
    ]


def test_compare_files_explicit_selection_ignores_names_outside_it(tmp_path):
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0])})
    comparison = xr.Dataset({"temp": ("x", [1.0, 2.0]), "salt": ("x", [3.0, 4.0])})
    reference_path = write_dataset(tmp_path, "reference.nc", reference)
    comparison_path = write_dataset(tmp_path, "comparison.nc", comparison)

    results = compare_files(reference_path, comparison_path, [("temp", "temp")], last_time_step=False)

    assert [result.variable for result in results] == ["temp"]
    assert results[0].passed


@pytest.mark.parametrize("variables", [None, [("flag", "flag")]])
@pytest.mark.parametrize("last_time_step", [False, True])
def test_compare_files_compares_boolean_variables(tmp_path, variables, last_time_step):
    # netCDF has no bool type: xarray stores it as int8 with a dtype attribute and decodes it back.
    def dataset(last_flags):
        return xr.Dataset({"flag": (("time", "x"), [[True, True, False], last_flags])})

    reference_path = write_dataset(tmp_path, "reference.nc", dataset([True, False, True]))
    same_path = write_dataset(tmp_path, "same.nc", dataset([True, False, True]))
    changed_path = write_dataset(tmp_path, "changed.nc", dataset([False, False, True]))

    same = _by_variable(compare_files(reference_path, same_path, variables, last_time_step=last_time_step))
    changed = _by_variable(compare_files(reference_path, changed_path, variables, last_time_step=last_time_step))

    assert same["flag"].passed and same["flag"].note == "equality only"
    assert changed["flag"].description == ("1 of 3 values differ" if last_time_step else "1 of 6 values differ")


def _compare_selection(tmp_path, reference, comparison, variables, *, last_time_step=False):
    """Write both datasets and compare the given -v selection."""
    reference_path = write_dataset(tmp_path, "reference.nc", reference)
    comparison_path = write_dataset(tmp_path, "comparison.nc", comparison)
    return compare_files(reference_path, comparison_path, variables, last_time_step=last_time_step)


@pytest.mark.parametrize(
    ("reference_names", "comparison_names", "description"),
    [
        (["temp"], ["temp", "salt"], "Not in the reference file"),
        (["temp", "salt"], ["temp"], "Not in the comparison file"),
        (["temp"], ["temp"], "Not in either file"),
    ],
)
def test_compare_files_fails_on_a_v_name_missing_from_a_file(tmp_path, reference_names, comparison_names, description):
    def dataset(names):
        return xr.Dataset({name: ("x", [1.0, 2.0]) for name in names})

    results = _compare_selection(
        tmp_path, dataset(reference_names), dataset(comparison_names), [("temp", "temp"), ("salt", "salt")]
    )

    # Each requested name keeps its row; a missing one used to be dropped (or, missing
    # from the comparison file only, failed with xarray's KeyError text).
    assert [(result.variable, result.description) for result in results] == [("temp", "-"), ("salt", description)]
    assert results[0].passed
    assert not results[1].passed


def test_compare_files_fails_on_a_single_v_name_missing_from_both_files(tmp_path):
    dataset = xr.Dataset({"temp": ("x", [1.0, 2.0])})

    results = _compare_selection(tmp_path, dataset, dataset, [("tmep", "tmep")])

    # A failed row naming it, rather than no row at all.
    assert [(result.variable, result.description) for result in results] == [("tmep", "Not in either file")]


@pytest.mark.parametrize(
    ("reference_name", "comparison_name", "description"),
    [
        ("thetao", "votemper", "-"),
        ("thetao", "missing", "Not in the comparison file"),
        ("missing", "votemper", "Not in the reference file"),
    ],
)
def test_compare_files_names_the_side_of_a_missing_mapped_v_name(
    tmp_path, reference_name, comparison_name, description
):
    reference = xr.Dataset({"thetao": ("x", [1.0, 2.0])})
    comparison = xr.Dataset({"votemper": ("x", [1.0, 2.0])})

    results = _compare_selection(tmp_path, reference, comparison, [(reference_name, comparison_name)])

    assert [(result.variable, result.description) for result in results] == [
        (f"{reference_name} -> {comparison_name}", description)
    ]


@pytest.mark.parametrize(("comparison_size", "description"), [(3, "-"), (4, "Dimension size mismatch: 3 - 4")])
def test_compare_files_compares_the_size_of_a_v_named_dimension_without_coordinate(
    tmp_path, comparison_size, description
):
    reference = xr.Dataset({"temp": ("k", [1.0, 2.0, 3.0])})
    comparison = xr.Dataset({"temp": ("k", np.arange(comparison_size, dtype=float))})

    results = _compare_selection(tmp_path, reference, comparison, [("k", "k")])

    # As without -v; it used to be dropped because it is not a variable.
    assert [(result.variable, result.description) for result in results] == [("k", description)]
    assert results[0].note == ("size only" if description == "-" else "")


@pytest.mark.parametrize("variable_side", ["reference", "comparison"])
def test_compare_files_fails_on_a_v_name_that_is_only_a_dimension_in_one_file(tmp_path, variable_side):
    with_variable = xr.Dataset({"temp": ("k", [1.0, 2.0])}, coords={"k": [10.0, 20.0]})
    with_dimension = xr.Dataset({"temp": ("k", [1.0, 2.0])})
    if variable_side == "reference":
        reference, comparison = with_variable, with_dimension
    else:
        reference, comparison = with_dimension, with_variable

    named = _compare_selection(tmp_path, reference, comparison, [("k", "k")])
    unnamed = _compare_default(tmp_path, reference, comparison)

    # The same row as without -v: the variable is in one file only.
    assert [(result.variable, result.description) for result in named] == [("k", f"Only in the {variable_side} file")]
    assert [(result.variable, result.description) for result in unnamed if result.variable == "k"] == [
        ("k", f"Only in the {variable_side} file")
    ]


def test_compare_files_fails_on_a_v_named_dimension_missing_from_a_file(tmp_path):
    reference = xr.Dataset({"temp": ("k", [1.0, 2.0])})
    comparison = xr.Dataset({"temp": ("x", [1.0, 2.0])})

    results = _compare_selection(tmp_path, reference, comparison, [("k", "k")])

    assert [(result.variable, result.description) for result in results] == [("k", "Not in the comparison file")]


def test_compare_files_compares_a_mapped_v_named_dimension_without_coordinate(tmp_path):
    reference = xr.Dataset({"temp": ("k", [1.0, 2.0, 3.0])})
    comparison = xr.Dataset({"temp": ("m", [1.0, 2.0, 3.0, 4.0])})

    results = _compare_selection(tmp_path, reference, comparison, [("k", "m")])

    assert [(result.variable, result.description) for result in results] == [
        ("k -> m", "Dimension size mismatch: 3 - 4")
    ]


@pytest.mark.parametrize(("reference_name", "comparison_name"), [("time", "time"), ("x", "time_counter")])
def test_compare_files_last_time_step_rejects_a_v_named_time_dimension_without_coordinate(
    tmp_path, reference_name, comparison_name
):
    reference = xr.Dataset({"temp": ((reference_name, "y"), [[1.0, 2.0], [3.0, 4.0]])})
    comparison = xr.Dataset({"temp": ((comparison_name, "y"), [[1.0, 2.0], [3.0, 4.0]])})

    results = _compare_selection(
        tmp_path, reference, comparison, [(reference_name, comparison_name)], last_time_step=True
    )

    # Like asking for a time coordinate: only the last step is compared, so not its size.
    # Either name being a time dimension is enough.
    assert [result.description for result in results] == ["Can't compare time if last time step is enabled"]


@pytest.mark.parametrize(("name", "last_time_step"), [("time", False), ("x", True)])
def test_compare_files_compares_the_size_of_a_v_named_dimension_otherwise(tmp_path, name, last_time_step):
    # A time dimension without --last-time-step, any other dimension with it.
    dataset = xr.Dataset({"temp": (("time", "x"), [[1.0, 2.0], [3.0, 4.0]])})

    results = _compare_selection(tmp_path, dataset, dataset, [(name, name)], last_time_step=last_time_step)

    assert [(result.variable, result.description, result.note) for result in results] == [(name, "-", "size only")]


def _run_with_time_axes(steps):
    """The last ``steps`` days of a run: the same last step whatever ``steps`` is."""
    times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")[-steps:]
    return xr.Dataset(
        {"temp": (("time", "x"), np.arange(6.0).reshape(3, 2)[-steps:])},
        coords={
            "time": times,
            "time_centered": ("time", times + np.timedelta64(12, "h")),
            "nav_lon": ("x", [10.0, 20.0]),
        },
    )


def test_compare_files_last_time_step_skips_auxiliary_time_coordinates_by_default(tmp_path):
    # Runs of different lengths: the time axes differ, the last step does not.
    results = _compare_default(tmp_path, _run_with_time_axes(2), _run_with_time_axes(3), last_time_step=True)

    # Only the datetime time axes are left out; other coordinates are still compared.
    assert [result.variable for result in results] == ["temp", "nav_lon", "x"]
    assert all(result.passed for result in results)


def test_compare_files_default_selection_with_a_bounding_box(tmp_path):
    longitudes, latitudes = np.meshgrid(np.arange(-10.0, 11.0), np.arange(-10.0, 11.0))

    def dataset(latitude_shift):
        return xr.Dataset(
            {"sst": (("y", "x"), longitudes + latitudes)},
            coords={"nav_lon": (("y", "x"), longitudes), "nav_lat": (("y", "x"), latitudes + latitude_shift)},
        )

    # Edges between grid points, so the small shift below keeps the same points inside.
    box = BoundingBox(lon_min=-5.5, lon_max=5.5, lat_min=-5.5, lat_max=5.5)
    reference_path = write_dataset(tmp_path, "reference.nc", dataset(0.0))
    same_path = write_dataset(tmp_path, "same.nc", dataset(0.0))
    shifted_path = write_dataset(tmp_path, "shifted.nc", dataset(1e-3))

    same = _by_variable(compare_files(reference_path, same_path, None, last_time_step=False, bbox=box))
    shifted = _by_variable(compare_files(reference_path, shifted_path, None, last_time_step=False, bbox=box))

    # The coordinates are cropped with the fields, and a small latitude shift that keeps
    # the same points inside the box is still detected.
    assert list(same) == ["sst", "nav_lon", "nav_lat", "y", "x"]
    assert all(result.passed for result in same.values())
    assert not shifted["nav_lat"].passed
    assert shifted["nav_lon"].passed


def test_compare_files_last_time_step_ignores_the_size_of_a_time_dimension_without_coordinate(tmp_path):
    # Same last step, different number of steps: what --last-time-step is for.
    reference = xr.Dataset({"temp": (("time", "x"), [[0.0, 0.0], [1.0, 2.0]])})
    comparison = xr.Dataset({"temp": (("time", "x"), [[5.0, 5.0], [0.0, 0.0], [1.0, 2.0]])})

    with_last_step = _by_variable(_compare_default(tmp_path, reference, comparison, last_time_step=True))
    without_last_step = _by_variable(_compare_default(tmp_path, reference, comparison))

    assert "time" not in with_last_step
    assert all(result.passed for result in with_last_step.values())
    assert without_last_step["time"].description == "Dimension size mismatch: 2 - 3"


def _dataset_with_time_coordinate():
    return xr.Dataset(
        {"temp": (("time", "x"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")},
    )


def test_compare_files_last_time_step_skips_time_coordinate_by_default(tmp_path):
    reference_path = write_dataset(tmp_path, "reference.nc", _dataset_with_time_coordinate())
    comparison_path = write_dataset(tmp_path, "comparison.nc", _dataset_with_time_coordinate())

    results = compare_files(reference_path, comparison_path, None, last_time_step=True)

    assert "time" not in [result.variable for result in results]
    assert "temp" in [result.variable for result in results]
    assert all(result.passed for result in results)


def test_compare_files_last_time_step_still_rejects_explicit_time_variable(tmp_path):
    reference_path = write_dataset(tmp_path, "reference.nc", _dataset_with_time_coordinate())
    comparison_path = write_dataset(tmp_path, "comparison.nc", _dataset_with_time_coordinate())

    results = compare_files(reference_path, comparison_path, ["time"], last_time_step=True)

    assert len(results) == 1
    assert not results[0].passed
    assert "last time step" in results[0].description


def test_compare_files_last_time_step_skips_time_coordinate_for_the_default_sentinel(tmp_path):
    reference_path = write_dataset(tmp_path, "reference.nc", _dataset_with_time_coordinate())
    comparison_path = write_dataset(tmp_path, "comparison.nc", _dataset_with_time_coordinate())

    results = compare_files(reference_path, comparison_path, settings.DEFAULT_VARIABLES_TO_CHECK, last_time_step=True)

    assert "time" not in [result.variable for result in results]
    assert all(result.passed for result in results)


def test_compare_files_last_time_step_still_reports_a_mismatching_time_axis(tmp_path):
    # Reference time is a datetime axis, comparison time is plain numbers: the two axes
    # differ, so the time variable must stay in the selection and fail.
    comparison = xr.Dataset(
        {"temp": (("time", "x"), [[1.0, 2.0], [3.0, 4.0]])},
        coords={"time": [0.0, 1.0]},
    )
    reference_path = write_dataset(tmp_path, "reference.nc", _dataset_with_time_coordinate())
    comparison_path = write_dataset(tmp_path, "comparison.nc", comparison)

    results = {
        result.variable: result for result in compare_files(reference_path, comparison_path, None, last_time_step=True)
    }

    assert "time" in results
    assert not results["time"].passed
    assert "Data type mismatch" in results["time"].description


def test_compare_files_last_time_step_keeps_errors_per_variable(tmp_path):
    # A variable with two time-like dimensions cannot be reduced to a last step. That is
    # one failed variable; it must not abort the whole file.
    dataset = xr.Dataset(
        {
            "forecast": (("time", "lead_time", "x"), np.ones((2, 3, 2))),
            "sst": (("time", "x"), np.ones((2, 2))),
        },
        coords={"time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")},
    )
    reference_path = write_dataset(tmp_path, "reference.nc", dataset)
    comparison_path = write_dataset(tmp_path, "comparison.nc", dataset)

    results = {
        result.variable: result for result in compare_files(reference_path, comparison_path, None, last_time_step=True)
    }

    assert results["sst"].passed
    assert not results["forecast"].passed
    assert "more than 1 time dimension" in results["forecast"].description


def _rectilinear_dataset(lon_values, lat_values):
    longitudes, latitudes = np.meshgrid(lon_values, lat_values)
    return xr.Dataset(
        {"sst": (("lat", "lon"), longitudes + latitudes)},
        coords={
            "lon": ("lon", lon_values, {"units": "degrees_east"}),
            "lat": ("lat", lat_values, {"units": "degrees_north"}),
        },
    )


def test_locate_horizontal_coords_prefers_standard_name():
    dataset = xr.Dataset(
        coords={
            "xc": ("xc", [1.0, 2.0], {"standard_name": "longitude"}),
            "yc": ("yc", [1.0, 2.0], {"standard_name": "latitude"}),
        }
    )
    assert locate_horizontal_coords(dataset) == ("xc", "yc")


def test_locate_horizontal_coords_falls_back_to_common_names():
    dataset = xr.Dataset(coords={"nav_lon": ("x", [1.0, 2.0]), "nav_lat": ("x", [1.0, 2.0])})
    assert locate_horizontal_coords(dataset) == ("nav_lon", "nav_lat")


def test_crop_to_bbox_curvilinear_grid():
    ys, xs = np.mgrid[0:10, 0:10]
    dataset = xr.Dataset(
        {"thetao": (("y", "x"), np.ones((10, 10)))},
        coords={"nav_lon": (("y", "x"), xs * 1.0), "nav_lat": (("y", "x"), ys * 1.0)},
    )

    cropped = crop_to_bbox(dataset, BoundingBox(lon_min=2, lon_max=5, lat_min=2, lat_max=5))

    assert dict(cropped.sizes) == {"y": 4, "x": 4}


def _rotated_grid():
    """A 7x7 curvilinear grid turned so that a lon/lat box does not fill its index rectangle."""
    xs, ys = np.meshgrid(np.arange(-3.0, 4.0), np.arange(-3.0, 4.0))
    return xr.Dataset(
        {
            "sst": (("y", "x"), np.ones((7, 7), dtype="float32")),
            "count": (("y", "x"), np.ones((7, 7), dtype="int32")),
            "flag": (("y", "x"), np.zeros((7, 7), dtype=bool)),
            "label": ("y", np.array(list("abcdefg"))),
            "bounds": (("t", "nb"), np.zeros((2, 2))),
        },
        coords={"nav_lon": (("y", "x"), xs + 0.5 * ys), "nav_lat": (("y", "x"), ys)},
    )


ROTATED_BOX = BoundingBox(lon_min=-1.2, lon_max=1.2, lat_min=-1.2, lat_max=1.2)


def test_crop_to_bbox_curvilinear_keeps_data_types_and_layout():
    dataset = _rotated_grid()

    cropped = crop_to_bbox(dataset, ROTATED_BOX)

    # Every variable keeps its data type and dimensions; Dataset.where used to turn
    # integers and booleans into floats, text into objects, and to broadcast a variable
    # that is not on the grid ("bounds", "label") onto it.
    assert {name: (variable.dims, variable.dtype) for name, variable in cropped.variables.items()} == {
        name: (variable.dims, variable.dtype) for name, variable in dataset.variables.items()
    }
    assert dict(cropped.sizes) == {"y": 3, "x": 3, "t": 2, "nb": 2}
    # Only the float field is masked outside the box; the others keep the whole rectangle.
    inside = (np.abs(cropped["nav_lon"]) <= 1.2) & (np.abs(cropped["nav_lat"]) <= 1.2)
    assert not bool(inside.all())
    np.testing.assert_array_equal(cropped["sst"].notnull().values, inside.values)
    assert int(cropped["count"].sum()) == 9
    assert list(cropped["label"].values) == ["c", "d", "e"]


@pytest.mark.parametrize(
    ("comparison_flag", "description"),
    [
        (np.zeros((7, 7), dtype=bool), "-"),
        (np.eye(7, dtype=bool), "3 of 9 values differ"),
        (np.zeros((7, 7)), "Data type mismatch: 'bool' - 'float64'"),
        (np.zeros((7, 7), dtype="int8"), "Data type mismatch: 'bool' - 'int8'"),
    ],
    ids=["same", "changed", "float", "int8"],
)
def test_compare_files_booleans_with_a_curvilinear_bounding_box(tmp_path, comparison_flag, description):
    # The crop used to turn booleans into floats: no "equality only" or count of the
    # values that differ, and booleans against floats of the same values passed.
    reference_path = write_dataset(tmp_path, "reference.nc", _rotated_grid())
    comparison_path = write_dataset(
        tmp_path, "comparison.nc", _rotated_grid().assign(flag=(("y", "x"), comparison_flag))
    )

    flag = _by_variable(compare_files(reference_path, comparison_path, None, last_time_step=False, bbox=ROTATED_BOX))[
        "flag"
    ]

    assert flag.description == description
    assert flag.note == ("equality only" if description == "-" else "")


def test_compare_files_integers_against_floats_with_a_curvilinear_bounding_box(tmp_path):
    reference_path = write_dataset(tmp_path, "reference.nc", _rotated_grid())
    comparison = _rotated_grid().assign(count=(("y", "x"), np.ones((7, 7))))
    comparison_path = write_dataset(tmp_path, "comparison.nc", comparison)

    results = _by_variable(compare_files(reference_path, comparison_path, None, last_time_step=False, bbox=ROTATED_BOX))

    # It used to pass: the crop turned both into float64.
    assert results["count"].description == "Data type mismatch: 'int32' - 'float64'"


def test_crop_to_bbox_one_dimensional_auxiliary_coordinates():
    # lon/lat are 1-D but auxiliary (their dims are x/y, not lon/lat), so they
    # are not indexable by .sel and must be cropped by masking instead.
    dataset = xr.Dataset(
        {"sst": (("y", "x"), np.zeros((5, 5)))},
        coords={
            "lon": ("x", np.arange(-2.0, 3.0), {"units": "degrees_east"}),
            "lat": ("y", np.arange(-2.0, 3.0), {"units": "degrees_north"}),
        },
    )

    cropped = crop_to_bbox(dataset, BoundingBox(lon_min=-1, lon_max=1, lat_min=-1, lat_max=1))

    assert dict(cropped.sizes) == {"y": 3, "x": 3}


def test_crop_to_bbox_raises_when_box_is_outside_extent():
    dataset = _rectilinear_dataset(np.arange(-10.0, 11.0), np.arange(-10.0, 11.0))

    with pytest.raises(ValueError, match="selects no data"):
        crop_to_bbox(dataset, BoundingBox(lon_min=100, lon_max=110, lat_min=80, lat_max=89))


def test_crop_to_bbox_raises_when_coordinates_absent():
    dataset = xr.Dataset({"v": ("a", [1.0, 2.0, 3.0])})

    with pytest.raises(ValueError, match="no longitude/latitude coordinates"):
        crop_to_bbox(dataset, BoundingBox(lon_min=-5, lon_max=5, lat_min=-5, lat_max=5))


def test_compare_files_bbox_aligns_different_extents_on_the_same_grid(tmp_path):
    reference = _rectilinear_dataset(np.arange(-10.0, 11.0), np.arange(-10.0, 11.0))
    comparison = _rectilinear_dataset(np.arange(-5.0, 16.0), np.arange(-10.0, 11.0))
    reference_path = write_dataset(tmp_path, "reference.nc", reference)
    comparison_path = write_dataset(tmp_path, "comparison.nc", comparison)

    # Same shape but shifted extent: without a box the fields differ...
    without_box = compare_files(reference_path, comparison_path, ["sst"], last_time_step=False)
    assert not without_box[0].passed

    # ...cropped to a common window they are identical.
    with_box = compare_files(
        reference_path,
        comparison_path,
        ["sst"],
        last_time_step=False,
        bbox=BoundingBox(lon_min=-5, lon_max=5, lat_min=-5, lat_max=5),
    )
    assert with_box[0].passed


def test_compare_yields_no_match_comparison():
    comparisons = list(compare({Path("reference.nc"): []}, ["temp"], False))

    assert len(comparisons) == 1
    assert comparisons[0].comparison_file is None
    assert type(comparisons[0].exception).__name__ == "NoMatchFound"


def test_compare_yields_one_comparison_for_each_match(monkeypatch):
    def fake_compare_files(file1, file2, variables, **kwargs):
        assert variables == ["temp"]
        assert kwargs["last_time_step"] is False
        return [CompareResult(variable=f"{file1.name}:{file2.name}")]

    monkeypatch.setattr("xdiff.compare.ncdiff.compare_files", fake_compare_files)

    reference = Path("reference.nc")
    comparisons = list(compare({reference: [Path("a.nc"), Path("b.nc")]}, ["temp"], False))

    assert [comparison.comparison_file for comparison in comparisons] == [
        Path("a.nc"),
        Path("b.nc"),
    ]
    assert [comparison[0].variable for comparison in comparisons] == [
        "reference.nc:a.nc",
        "reference.nc:b.nc",
    ]
