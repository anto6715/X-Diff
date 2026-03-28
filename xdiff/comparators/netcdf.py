"""Comparator for netCDF artifacts and related numeric helpers."""

from __future__ import annotations

import logging
import warnings

from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

import xdiff.conf as settings

from xdiff.comparators.base import ArtifactComparator
from xdiff.exceptions import AllNaN, LastTimestepTimeCheckException
from xdiff.model import CompareResult
from xdiff.model.artifact import ArtifactKind
from xdiff.model.comparison import Comparison
from xdiff.model.match import ArtifactMatch
from xdiff.model.request import CompareRequest, ExecutionMode

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

logger = logging.getLogger("xdiff")

if TYPE_CHECKING:
    import xarray as xr


@lru_cache(maxsize=1)
def load_xarray():
    """Import xarray only when a netCDF comparison is actually requested."""
    try:
        return import_module("xarray")
    except ImportError as exc:
        raise RuntimeError(
            "xarray is required to compare netCDF files. Install the package dependencies before using this feature."
        ) from exc


class NetcdfComparator(ArtifactComparator):
    """Compare two netCDF files using xarray-backed numeric checks."""

    artifact_kind = ArtifactKind.NETCDF

    def compare(self, match: ArtifactMatch, request: CompareRequest) -> Comparison:
        comparison = Comparison(
            reference_artifact=match.reference,
            comparison_artifact=match.comparison,
        )

        if match.comparison is None:
            raise ValueError("A comparison artifact is required for netCDF comparison")

        comparison.extend(
            compare_files(
                match.reference.path,
                match.comparison.path,
                request.variables,
                last_time_step=request.last_time_step,
                execution_mode=request.execution_mode,
            )
        )
        return comparison


def compare_files(
    file1,
    file2,
    variables: tuple[str, ...] | list[str] | object | None,
    *,
    last_time_step: bool,
    execution_mode: ExecutionMode = ExecutionMode.SERIAL,
) -> list[CompareResult]:
    xr = load_xarray()
    open_dataset_kwargs = build_open_dataset_kwargs(execution_mode)
    with xr.open_dataset(file1, **open_dataset_kwargs) as dataset1, xr.open_dataset(
        file2,
        **open_dataset_kwargs,
    ) as dataset2:
        variables_to_compare = get_dataset_variables(dataset1, variables)
        return compare_datasets(
            dataset1,
            dataset2,
            variables_to_compare,
            last_time_step=last_time_step,
            execution_mode=execution_mode,
        )


def compare_datasets(
    reference: xr.Dataset,
    comparison: xr.Dataset,
    variables: list[str],
    *,
    last_time_step: bool,
    execution_mode: ExecutionMode = ExecutionMode.SERIAL,
) -> list[CompareResult]:
    results: list[CompareResult] = []

    for variable in variables:
        logger.info("Comparing %s", variable)
        try:
            reference_field = reference[variable]
            comparison_field = comparison[variable]
            results.append(
                compare_variables(
                    reference_field,
                    comparison_field,
                    variable,
                    last_time_step=last_time_step,
                    execution_mode=execution_mode,
                )
            )
        except Exception as exc:
            results.append(CompareResult(variable=variable, description=str(exc)))

    return results


def compare_variables(
    ref_da: xr.DataArray,
    cmp_da: xr.DataArray,
    variable: str,
    *,
    last_time_step: bool,
    execution_mode: ExecutionMode = ExecutionMode.SERIAL,
) -> CompareResult:
    if last_time_step:
        if is_time_coordinate_variable(variable, ref_da, cmp_da):
            raise LastTimestepTimeCheckException("Can't compare time if last time step is enabled")
        ref_da = select_last_time_step(ref_da)
        cmp_da = select_last_time_step(cmp_da)

    validate_matching_metadata(ref_da, cmp_da)

    if execution_mode is ExecutionMode.ARRAYS:
        return compare_variables_with_chunks(ref_da, cmp_da, variable)

    reference_values = ref_da.values
    comparison_values = cmp_da.values
    reference_masked = ref_da.to_masked_array()
    comparison_masked = cmp_da.to_masked_array()

    difference_field = reference_values - comparison_values

    if np.isnan(difference_field).all():
        raise AllNaN("All nan values found")

    mask_is_equal = np.array_equal(reference_masked.mask, comparison_masked.mask)

    return CompareResult(
        relative_error=compute_relative_error(difference_field, comparison_values),
        min_diff=np.nanmin(difference_field),
        max_diff=np.nanmax(difference_field),
        mask_equal=mask_is_equal,
        variable=variable,
    )


def build_open_dataset_kwargs(execution_mode: ExecutionMode) -> dict[str, object]:
    """Return dataset opening options for the selected execution strategy."""
    if execution_mode is ExecutionMode.ARRAYS:
        # Let xarray ask Dask for a reasonable per-variable chunk layout.
        return {"chunks": "auto"}
    return {}


def compare_variables_with_chunks(
    ref_da: xr.DataArray,
    cmp_da: xr.DataArray,
    variable: str,
) -> CompareResult:
    """Reduce a possibly Dask-backed variable comparison down to scalar metrics."""
    metrics = build_array_comparison_metrics(ref_da, cmp_da).compute()
    all_missing = bool(extract_scalar(metrics["all_missing"]))
    if all_missing:
        raise AllNaN("All nan values found")

    all_zero = bool(extract_scalar(metrics["all_zero"]))
    relative_error = 0.0 if all_zero else extract_scalar(metrics["relative_error"])
    if is_time_dtype(cmp_da.dtype) and not all_zero:
        relative_error = relative_error / np.timedelta64(1, "s")

    return CompareResult(
        relative_error=relative_error,
        min_diff=extract_scalar(metrics["min_diff"]),
        max_diff=extract_scalar(metrics["max_diff"]),
        mask_equal=bool(extract_scalar(metrics["mask_equal"])),
        variable=variable,
    )


def build_array_comparison_metrics(ref_da: xr.DataArray, cmp_da: xr.DataArray):
    """Build lazy scalar reductions for chunked comparisons."""
    xr = load_xarray()
    difference = ref_da - cmp_da

    return xr.Dataset(
        data_vars={
            "all_missing": difference.isnull().all(),
            "all_zero": (difference == 0).all(),
            "mask_equal": (ref_da.isnull() == cmp_da.isnull()).all(),
            "min_diff": difference.min(skipna=True),
            "max_diff": difference.max(skipna=True),
            "relative_error": build_relative_error_metric(difference, cmp_da),
        }
    )


def build_relative_error_metric(diff_da: xr.DataArray, cmp_da: xr.DataArray):
    """Build the lazy relative-error reduction for a variable comparison."""
    xr = load_xarray()

    if is_time_dtype(cmp_da.dtype):
        abs_diff = abs(diff_da)
        abs_cmp = abs(cmp_da.astype("int64"))
    else:
        abs_diff = abs(diff_da)
        abs_cmp = abs(cmp_da)

    rel_err_array = abs_diff / abs_cmp
    rel_err_array = xr.where(np.isinf(rel_err_array), np.nan, rel_err_array)
    return rel_err_array.max(skipna=True)


def extract_scalar(data_array: xr.DataArray):
    """Return a scalar while preserving NumPy datetime/timedelta scalar types."""
    return np.asarray(data_array.data)[()]


def select_last_time_step(field: xr.DataArray) -> xr.DataArray:
    time_dimension = find_time_dims_name(field.dims)
    if time_dimension is None:
        return field

    if field.sizes[time_dimension] > 1:
        return field.isel({time_dimension: slice(-1, None)})
    return field


def find_time_dims_name(dims: Iterable) -> Any | None:
    time_dimensions = [dimension for dimension in dims if "time" in dimension]
    if len(time_dimensions) == 0:
        return None
    if len(time_dimensions) > 1:
        raise ValueError(f"Found more than 1 time dimension: {', '.join(time_dimensions)}")
    return time_dimensions.pop()


def compute_relative_error(diff: np.ndarray, field2: np.ndarray):
    if np.all(diff == 0.0):
        return 0.0

    if is_time_dtype(field2.dtype):
        field2_values = field2.view("int64")
    else:
        field2_values = field2

    abs_diff = np.abs(diff)
    abs_field2 = np.abs(field2_values)

    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err_array = abs_diff / abs_field2
            if np.isinf(rel_err_array).any():
                rel_err_array[np.isinf(rel_err_array)] = np.nan
            rel_err = np.nanmax(rel_err_array)
    except Exception as exc:
        logger.debug("An error occurred when computing relative error: %s", exc)
        rel_err = np.nan

    if is_time_dtype(field2.dtype):
        return rel_err / np.timedelta64(1, "s")
    return rel_err


def is_time_dtype(dtype) -> bool:
    normalized_dtype = np.dtype(dtype)
    return np.issubdtype(normalized_dtype, np.datetime64) or np.issubdtype(normalized_dtype, np.timedelta64)


def is_time_coordinate_variable(variable: str, ref_da: xr.DataArray, cmp_da: xr.DataArray) -> bool:
    time_dimension = find_time_dims_name(ref_da.dims)
    comparison_time_dimension = find_time_dims_name(cmp_da.dims)

    if time_dimension != comparison_time_dimension or time_dimension is None:
        return False

    return (
        variable == time_dimension
        and ref_da.dims == (time_dimension,)
        and cmp_da.dims == (time_dimension,)
        and is_time_dtype(ref_da.dtype)
        and is_time_dtype(cmp_da.dtype)
    )


def validate_matching_metadata(ref_da: xr.DataArray, cmp_da: xr.DataArray) -> None:
    if ref_da.dims != cmp_da.dims:
        raise ValueError(f"Dimension mismatch: '{ref_da.dims}' - '{cmp_da.dims}'")

    reference_sizes = tuple(ref_da.sizes[dimension] for dimension in ref_da.dims)
    comparison_sizes = tuple(cmp_da.sizes[dimension] for dimension in cmp_da.dims)
    if reference_sizes != comparison_sizes:
        raise ValueError(f"Dimension size mismatch: '{reference_sizes}' - '{comparison_sizes}'")

    if np.dtype(ref_da.dtype) != np.dtype(cmp_da.dtype):
        raise ValueError(f"Data type mismatch: '{ref_da.dtype}' - '{cmp_da.dtype}'")

    reference_coordinates = set(ref_da.coords)
    comparison_coordinates = set(cmp_da.coords)
    if reference_coordinates != comparison_coordinates:
        raise ValueError(
            "Coordinate mismatch: "
            f"'{', '.join(sorted(reference_coordinates)) or '-'}' - "
            f"'{', '.join(sorted(comparison_coordinates)) or '-'}'"
        )

    for coordinate_name in sorted(reference_coordinates):
        reference_coordinate = ref_da.coords[coordinate_name]
        comparison_coordinate = cmp_da.coords[coordinate_name]

        if reference_coordinate.dims != comparison_coordinate.dims:
            raise ValueError(
                f"Coordinate dimension mismatch for '{coordinate_name}': "
                f"'{reference_coordinate.dims}' - '{comparison_coordinate.dims}'"
            )

        if np.dtype(reference_coordinate.dtype) != np.dtype(comparison_coordinate.dtype):
            raise ValueError(
                f"Coordinate type mismatch for '{coordinate_name}': "
                f"'{reference_coordinate.dtype}' - '{comparison_coordinate.dtype}'"
            )

        if not reference_coordinate.equals(comparison_coordinate):
            raise ValueError(f"Coordinate values mismatch for '{coordinate_name}'")


def get_dataset_variables(dataset: xr.Dataset, variables: tuple[str, ...] | list[str] | object | None) -> list[str]:
    """Extract comparable variables and dimensions from a dataset."""
    selected_variables: list[str] = []

    if variables in (None, settings.DEFAULT_VARIABLES_TO_CHECK):
        variables_to_check = list(dataset.data_vars) + list(dataset.dims)
    else:
        variables_to_check = list(variables)

    for variable in variables_to_check:
        if variable not in dataset:
            continue

        dtype_kind = dataset[variable].dtype.kind
        if dtype_kind in ("U", "S", "O", "a"):
            logger.debug("Skipping variable %s due to datatype %s", variable, dtype_kind)
            continue

        selected_variables.append(variable)

    return selected_variables
