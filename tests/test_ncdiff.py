from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import xdiff.conf as settings
from xdiff.comparators.netcdf import (
    compare_datasets,
    compare_files,
    compare_variables,
    compute_relative_error,
    find_time_dims_name,
    get_dataset_variables,
    select_last_time_step,
)
from xdiff.compare import compare
from xdiff.exceptions import AllNaN, LastTimestepTimeCheckException
from xdiff.model import CompareResult


def make_data_array(values, dims=("x",), dtype=None):
    array = np.array(values, dtype=dtype) if dtype is not None else np.array(values)
    return xr.DataArray(array, dims=dims)


def write_dataset(tmp_path, name, dataset):
    path = tmp_path / name
    dataset.to_netcdf(path)
    return path


def test_get_dataset_variables_defaults_skip_string_like_values():
    dataset = xr.Dataset(
        data_vars={
            "temp": ("time", [1.0, 2.0]),
            "label": ("time", np.array(["a", "b"], dtype=str)),
        },
        coords={"time": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")},
    )

    variables = get_dataset_variables(dataset, settings.DEFAULT_VARIABLES_TO_CHECK)

    assert variables == ["temp", "time"]


def test_get_dataset_variables_respects_explicit_variable_selection():
    dataset = xr.Dataset(
        data_vars={
            "temp": ("x", [1.0, 2.0]),
            "label": ("x", np.array(["a", "b"], dtype=str)),
        }
    )

    variables = get_dataset_variables(dataset, ["label", "temp", "missing"])

    assert variables == ["temp"]


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
    field = np.array([2.0, 4.0])

    assert compute_relative_error(diff, field) == 0.0


def test_compute_relative_error_handles_division_by_zero():
    diff = np.array([1.0, 0.0])
    field = np.array([0.0, 2.0])

    assert compute_relative_error(diff, field) == 0.0


def test_compute_relative_error_handles_datetime_arrays():
    diff = np.array([np.timedelta64(1, "s")])
    field = np.array([np.datetime64("2024-01-01T00:00:02.000000000")])

    assert compute_relative_error(diff, field) == 0.0


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
    assert result.relative_error == 0.5


def test_compare_variables_raises_on_all_nan_values():
    reference = make_data_array([np.nan, np.nan])
    comparison = make_data_array([np.nan, np.nan])

    with pytest.raises(AllNaN, match="All nan values found"):
        compare_variables(reference, comparison, "temp", last_time_step=False)


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
    assert result.relative_error == pytest.approx(2 / 12)


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


def test_compare_datasets_records_variable_level_errors():
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0])})
    comparison = xr.Dataset()

    results = compare_datasets(reference, comparison, ["temp"], last_time_step=False)

    assert len(results) == 1
    assert results[0].variable == "temp"
    assert results[0].description != "-"


def test_compare_files_reads_netcdf_inputs(tmp_path):
    reference = xr.Dataset({"temp": ("x", [1.0, 2.0])})
    comparison = xr.Dataset({"temp": ("x", [1.0, 2.0])})
    reference_path = write_dataset(tmp_path, "reference.nc", reference)
    comparison_path = write_dataset(tmp_path, "comparison.nc", comparison)

    results = compare_files(reference_path, comparison_path, ["temp"], last_time_step=False)

    assert len(results) == 1
    assert results[0].variable == "temp"
    assert results[0].relative_error == 0.0


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
