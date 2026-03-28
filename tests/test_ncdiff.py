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
from xdiff.model import CompareResult, ExecutionMode


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


def test_compare_variables_raises_on_dimension_mismatch():
    reference = make_data_array([1.0, 2.0], dims=("x",))
    comparison = make_data_array([[1.0, 2.0]], dims=("x", "y"))

    with pytest.raises(ValueError, match="Dimension mismatch"):
        compare_variables(reference, comparison, "temp", last_time_step=False)


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


def test_compare_variables_rejects_time_variables_when_last_time_step_is_enabled():
    reference = make_data_array([1.0, 2.0], dims=("time",))
    comparison = make_data_array([1.0, 2.0], dims=("time",))

    with pytest.raises(
        LastTimestepTimeCheckException,
        match="Can't compare time if last time step is enabled",
    ):
        compare_variables(reference, comparison, "time_counter", last_time_step=True)


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


def test_compare_files_uses_chunked_opening_for_arrays_mode(monkeypatch):
    calls = []

    class FakeDataset(dict):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeXarrayModule:
        def open_dataset(self, path, **kwargs):
            calls.append(kwargs)
            return FakeDataset({"temp": make_data_array([1.0, 2.0])})

    monkeypatch.setattr("xdiff.comparators.netcdf.load_xarray", lambda: FakeXarrayModule())
    monkeypatch.setattr(
        "xdiff.comparators.netcdf.compare_datasets",
        lambda *args, **kwargs: [CompareResult(variable="temp")],
    )

    results = compare_files(
        Path("reference.nc"),
        Path("comparison.nc"),
        ["temp"],
        last_time_step=False,
        execution_mode=ExecutionMode.ARRAYS,
    )

    assert len(results) == 1
    assert calls == [{"chunks": "auto"}, {"chunks": "auto"}]


def test_compare_variables_supports_arrays_mode_with_same_results():
    reference = make_data_array([1.0, np.nan, 3.0])
    comparison = make_data_array([1.0, np.nan, 2.0])

    result = compare_variables(
        reference,
        comparison,
        "temp",
        last_time_step=False,
        execution_mode=ExecutionMode.ARRAYS,
    )

    assert result.relative_error == 0.5
    assert result.min_diff == 0.0
    assert result.max_diff == 1.0
    assert result.mask_equal is True


def test_compare_variables_arrays_mode_preserves_mask_comparison():
    reference = make_data_array([1.0, np.nan])
    comparison = make_data_array([1.0, 2.0])

    result = compare_variables(
        reference,
        comparison,
        "temp",
        last_time_step=False,
        execution_mode=ExecutionMode.ARRAYS,
    )

    assert result.mask_equal is False


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
