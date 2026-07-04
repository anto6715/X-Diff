"""Comparator for netCDF artifacts and related numeric helpers."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any

import numpy as np

import xdiff.conf as settings
from xdiff.comparators.base import ArtifactComparator
from xdiff.exceptions import AllNaN, LastTimestepTimeCheckException
from xdiff.model import CompareResult
from xdiff.model.artifact import ArtifactKind
from xdiff.model.comparison import Comparison
from xdiff.model.match import ArtifactMatch
from xdiff.model.request import CompareRequest

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
            )
        )
        return comparison


def compare_files(
    file1,
    file2,
    variables: tuple[str, ...] | list[str] | object | None,
    *,
    last_time_step: bool,
) -> list[CompareResult]:
    xr = load_xarray()
    with xr.open_dataset(file1) as dataset1, xr.open_dataset(file2) as dataset2:
        variables_to_compare = get_dataset_variables(dataset1, variables)
        return compare_datasets(
            dataset1,
            dataset2,
            variables_to_compare,
            last_time_step=last_time_step,
        )


def compare_datasets(
    reference: xr.Dataset,
    comparison: xr.Dataset,
    variables: list[str],
    *,
    last_time_step: bool,
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
) -> CompareResult:
    if last_time_step:
        if is_time_coordinate_variable(variable, ref_da, cmp_da):
            raise LastTimestepTimeCheckException("Can't compare time if last time step is enabled")
        ref_da = select_last_time_step(ref_da)
        cmp_da = select_last_time_step(cmp_da)

    validate_matching_metadata(ref_da, cmp_da)

    reference_values = ref_da.values
    comparison_values = cmp_da.values
    reference_masked = ref_da.to_masked_array()
    comparison_masked = cmp_da.to_masked_array()

    # Integer dtypes wrap around on subtraction (e.g. uint8: 10 - 12 -> 254),
    # silently corrupting min/max/relative-error. Promote integer operands to
    # float before subtracting; float and datetime/timedelta dtypes already
    # subtract correctly and are left untouched (the time path relies on the
    # resulting timedelta64).
    if np.issubdtype(reference_values.dtype, np.integer):
        difference_field = reference_values.astype(np.float64) - comparison_values.astype(np.float64)
    else:
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
        logger.debug(
            "Dimension name mismatch: '%s' - '%s'",
            ref_da.dims,
            cmp_da.dims,
        )

    if ref_da.shape != cmp_da.shape:
        raise ValueError(f"Dimension size mismatch: '{ref_da.shape}' - '{cmp_da.shape}'")

    if np.dtype(ref_da.dtype) != np.dtype(cmp_da.dtype):
        raise ValueError(f"Data type mismatch: '{ref_da.dtype}' - '{cmp_da.dtype}'")

    reference_coordinates = set(ref_da.coords)
    comparison_coordinates = set(cmp_da.coords)
    if reference_coordinates != comparison_coordinates:
        logger.debug(
            "Coordinate mismatch: '%s' - '%s'",
            ", ".join(sorted(reference_coordinates)) or "-",
            ", ".join(sorted(comparison_coordinates)) or "-",
        )

    for coordinate_name in sorted(reference_coordinates & comparison_coordinates):
        reference_coordinate = ref_da.coords[coordinate_name]
        comparison_coordinate = cmp_da.coords[coordinate_name]

        if reference_coordinate.dims != comparison_coordinate.dims:
            logger.debug(
                "Coordinate dimension mismatch for '%s': '%s' - '%s'",
                coordinate_name,
                reference_coordinate.dims,
                comparison_coordinate.dims,
            )
            continue

        if np.dtype(reference_coordinate.dtype) != np.dtype(comparison_coordinate.dtype):
            logger.debug(
                "Coordinate type mismatch for '%s': '%s' - '%s'",
                coordinate_name,
                reference_coordinate.dtype,
                comparison_coordinate.dtype,
            )
            continue

        if not reference_coordinate.equals(comparison_coordinate):
            logger.debug("Coordinate values mismatch for '%s'", coordinate_name)


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
