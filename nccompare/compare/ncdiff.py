import logging
import warnings
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import xarray as xr

import nccompare.conf as settings
from nccompare.exceptions import AllNaN, LastTimestepTimeCheckException
from nccompare.model import CompareResult
from nccompare.model.comparison import Comparison

warnings.filterwarnings("ignore", message="All-NaN slice encountered")

logger = logging.getLogger("nccompare")


class NoMatchFound(Exception):
    pass


def compare(
    compare_match: dict[Path, list[Path]], variables: List[str], last_time_step: bool
):
    for reference, to_compares in compare_match.items():
        if len(to_compares) == 0:
            yield Comparison(
                reference, None, NoMatchFound(f"No match found for {reference}")
            )

        try:
            for to_compare in to_compares:
                comparison = Comparison(reference, to_compare)
                comparison.extend(
                    compare_files(
                        reference, to_compare, variables, last_time_step=last_time_step
                    )
                )
                yield comparison

        except Exception as e:
            yield Comparison(reference, None, e)


def compare_files(
    file1: Path, file2: Path, variables: List[str], **kwargs
) -> list[CompareResult]:
    dataset1 = xr.open_dataset(file1)
    dataset2 = xr.open_dataset(file2)
    variables_to_compare = get_dataset_variables(dataset1, variables)
    return compare_datasets(dataset1, dataset2, variables_to_compare, **kwargs)


def compare_datasets(
    reference: xr.Dataset,
    comparison: xr.Dataset,
    variables: List[str],
    last_time_step: bool,
) -> List[CompareResult]:
    results = []
    for var in variables:
        logger.info(f"Comparing {var}")
        try:
            field1 = reference[var]
            field2 = comparison[var]
            results.append(compare_variables(field1, field2, var, last_time_step))
        except Exception as e:
            results.append(CompareResult(variable=var, description=str(e)))
    return results


def compare_variables(
    ref_da: xr.DataArray,
    cmp_da: xr.DataArray,
    var,
    last_time_step: bool,
):
    # - drop all time steps except last one
    # - do not compare time variables
    if last_time_step:
        if "time" in var:
            raise LastTimestepTimeCheckException(
                "Can't compare time if last time step is enabled"
            )
        ref_da = select_last_time_step(ref_da)
        cmp_da = select_last_time_step(cmp_da)

    # dimensions mismatch
    if ref_da.shape != cmp_da.shape:
        raise ValueError(f"Dimension mismatch: '{ref_da.shape}' - '{cmp_da.shape}'")

    # array views
    ref_array = ref_da.values
    cmp_array = cmp_da.values
    ref_masked = ref_da.to_masked_array()
    cmp_masked = cmp_da.to_masked_array()

    # try computing difference
    difference_field = ref_array - cmp_array

    # get statistics
    max_difference = np.nanmax(difference_field)
    min_difference = np.nanmin(difference_field)
    if np.isnan(max_difference) and np.isnan(min_difference):
        raise AllNaN("All nan values found")

    mask_is_equal = np.array_equal(ref_masked.mask, cmp_masked.mask)
    rel_err = compute_relative_error(difference_field, cmp_array)

    return CompareResult(
        relative_error=rel_err,
        min_diff=min_difference,
        max_diff=max_difference,
        mask_equal=mask_is_equal,
        variable=var,
    )


def select_last_time_step(field: xr.DataArray) -> xr.DataArray:
    time_dims_name = find_time_dims_name(field.dims)
    if time_dims_name and field.shape[0] > 1:
        return field.drop_isel({time_dims_name: [t for t in range(field.shape[0] - 1)]})
    else:
        return field


def find_time_dims_name(dims: Iterable) -> Any | None:
    time_dims_name = [dim for dim in dims if "time" in dim]
    if len(time_dims_name) == 0:
        return None
    if len(time_dims_name) > 1:
        raise ValueError(
            f"Found more than 1 time dimension: {', '.join(time_dims_name)}"
        )
    return time_dims_name.pop()


def compute_relative_error(diff: np.ndarray, field2: np.ndarray):
    if np.all(diff == 0.0):
        return 0.0

    if field2.dtype in settings.TIME_DTYPE:
        field2_values = field2.view("int64")
    else:
        field2_values = field2

    abs_diff = np.abs(diff)
    abs_field2 = np.abs(field2_values)

    try:
        # Suppress division by zero and invalid value warnings
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err_array = abs_diff / abs_field2
            if np.isinf(rel_err_array).any():
                rel_err_array[np.isinf(rel_err_array)] = np.nan
            rel_err = np.nanmax(rel_err_array)
    except Exception as e:
        logger.debug(f"An error occurred when computing relative error: {e}")
        rel_err = np.nan

    if field2.dtype in settings.TIME_DTYPE:
        return rel_err / np.timedelta64(1, "s")
    return rel_err


def get_dataset_variables(dataset: xr.Dataset, variables: List[str]) -> List[str]:
    """Extract all non char/str variables included dimension from a dataset"""
    ds_variables = []

    if variables is settings.DEFAULT_VARIABLES_TO_CHECK:
        variables_to_check = list(dataset.data_vars) + list(dataset.dims)
    else:
        variables_to_check = variables

    for v in variables_to_check:
        if v in dataset:
            dtype_kind = dataset[v].dtype.kind
            # Skip variables with string/unicode/byte string datatypes
            # U: Unicode, S: byte string, O: object, a: fixed-length string
            if dtype_kind in ("U", "S", "O", "a"):
                logger.debug(f"Skipping variable {v} due to datatype {dtype_kind}")
                continue
            ds_variables.append(v)

    return ds_variables
