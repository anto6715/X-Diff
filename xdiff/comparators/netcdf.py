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
from xdiff.exceptions import LastTimestepTimeCheckException
from xdiff.model import BoundingBox, CompareResult
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
                bbox=request.bbox,
            )
        )
        return comparison


def compare_files(
    file1,
    file2,
    variables: tuple[str, ...] | list[str] | object | None,
    *,
    last_time_step: bool,
    bbox: BoundingBox | None = None,
) -> list[CompareResult]:
    xr = load_xarray()
    with xr.open_dataset(file1) as dataset1, xr.open_dataset(file2) as dataset2:
        if bbox is not None:
            dataset1 = crop_to_bbox(dataset1, bbox)
            dataset2 = crop_to_bbox(dataset2, bbox)
        variables_to_compare = get_dataset_variables(dataset1, variables)
        return compare_datasets(
            dataset1,
            dataset2,
            variables_to_compare,
            last_time_step=last_time_step,
        )


# CF/attribute and common-name heuristics for locating horizontal coordinates.
_LONGITUDE_STANDARD_NAMES = {"longitude"}
_LATITUDE_STANDARD_NAMES = {"latitude"}
_LONGITUDE_UNITS = {"degrees_east", "degree_east", "degreese", "degree_e", "degrees_e"}
_LATITUDE_UNITS = {"degrees_north", "degree_north", "degreesn", "degree_n", "degrees_n"}
_LONGITUDE_NAMES = ("longitude", "lon", "nav_lon", "glamt")
_LATITUDE_NAMES = ("latitude", "lat", "nav_lat", "gphit")


def locate_horizontal_coords(dataset: xr.Dataset) -> tuple[Any | None, Any | None]:
    """Return the (longitude, latitude) coordinate names, or None where absent.

    Detection prefers the CF ``standard_name`` attribute, then CF ``units``
    (degrees_east/north), then a small set of common names (lon/nav_lon, ...).
    """
    return (
        _find_coordinate(dataset, _LONGITUDE_STANDARD_NAMES, _LONGITUDE_UNITS, _LONGITUDE_NAMES),
        _find_coordinate(dataset, _LATITUDE_STANDARD_NAMES, _LATITUDE_UNITS, _LATITUDE_NAMES),
    )


def _find_coordinate(dataset: xr.Dataset, standard_names: set[str], units: set[str], common_names: tuple[str, ...]):
    for name, variable in dataset.variables.items():
        if str(variable.attrs.get("standard_name", "")).lower() in standard_names:
            return name
    for name, variable in dataset.variables.items():
        if str(variable.attrs.get("units", "")).lower() in units:
            return name
    for name in common_names:
        if name in dataset.variables:
            return name
    return None


def crop_to_bbox(dataset: xr.Dataset, bbox: BoundingBox) -> xr.Dataset:
    """Crop a dataset to a lon/lat box, dispatching on the coordinate layout.

    Raises ``ValueError`` if lon/lat coordinates cannot be located or if the box
    selects no data, so a mis-specified box fails loudly rather than yielding
    empty, trivially-identical fields.
    """
    longitude_name, latitude_name = locate_horizontal_coords(dataset)
    if longitude_name is None or latitude_name is None:
        raise ValueError(f"Cannot apply bounding box ({bbox}): no longitude/latitude coordinates found in the dataset.")

    # Label-based `.sel` only works when lon/lat are dimension coordinates (have
    # an index). 1-D auxiliary coordinates (e.g. lon(x)) and 2-D curvilinear grids
    # are cropped by boolean masking instead.
    if _is_rectilinear_axis(dataset, longitude_name) and _is_rectilinear_axis(dataset, latitude_name):
        cropped = _crop_rectilinear(dataset, longitude_name, latitude_name, bbox)
    else:
        cropped = _crop_curvilinear(dataset, dataset[longitude_name], dataset[latitude_name], bbox)

    if cropped[longitude_name].size == 0 or cropped[latitude_name].size == 0:
        raise ValueError(f"Bounding box ({bbox}) selects no data; it is empty or lies outside the dataset extent.")

    logger.info("Cropped to bounding box %s", bbox)
    return cropped


def _is_rectilinear_axis(dataset: xr.Dataset, name) -> bool:
    """True when ``name`` is a 1-D dimension coordinate (indexable by ``.sel``)."""
    coordinate = dataset[name]
    return coordinate.ndim == 1 and coordinate.dims == (name,)


def _crop_rectilinear(dataset: xr.Dataset, longitude_name, latitude_name, bbox: BoundingBox) -> xr.Dataset:
    """Crop a 1-D (rectilinear) grid with label-based selection, honouring axis order."""
    return dataset.sel(
        {
            longitude_name: _oriented_slice(dataset[longitude_name], bbox.lon_min, bbox.lon_max),
            latitude_name: _oriented_slice(dataset[latitude_name], bbox.lat_min, bbox.lat_max),
        }
    )


def _oriented_slice(coordinate: xr.DataArray, low: float, high: float) -> slice:
    values = coordinate.values
    if values.size >= 2 and values[0] > values[-1]:
        return slice(high, low)  # descending axis
    return slice(low, high)


def _crop_curvilinear(dataset: xr.Dataset, longitude: xr.DataArray, latitude: xr.DataArray, bbox: BoundingBox):
    """Crop a 2-D (curvilinear) grid, e.g. NEMO nav_lon/nav_lat, by masking and dropping."""
    inside_box = (
        (longitude >= bbox.lon_min)
        & (longitude <= bbox.lon_max)
        & (latitude >= bbox.lat_min)
        & (latitude <= bbox.lat_max)
    )
    return dataset.where(inside_box, drop=True)


def _as_variable_pair(item: str | tuple[str, str]) -> tuple[str, str]:
    """Coerce a variable spec to a (reference_name, comparison_name) pair."""
    if isinstance(item, str):
        return (item, item)
    reference_name, comparison_name = item  # unpacking also asserts a 2-element pair
    return reference_name, comparison_name


def compare_datasets(
    reference: xr.Dataset,
    comparison: xr.Dataset,
    variables: list[str] | list[tuple[str, str]],
    *,
    last_time_step: bool,
) -> list[CompareResult]:
    results: list[CompareResult] = []

    for item in variables:
        reference_name, comparison_name = _as_variable_pair(item)
        label = reference_name if reference_name == comparison_name else f"{reference_name} -> {comparison_name}"
        logger.info("Comparing %s", label)
        try:
            reference_field = reference[reference_name]
            comparison_field = comparison[comparison_name]
            results.append(
                compare_variables(
                    reference_field,
                    comparison_field,
                    label,
                    last_time_step=last_time_step,
                )
            )
        except Exception as exc:
            results.append(CompareResult(variable=label, description=str(exc)))

    return results


def compare_variables(
    ref_da: xr.DataArray,
    cmp_da: xr.DataArray,
    variable: str,
    *,
    last_time_step: bool,
) -> CompareResult:
    if last_time_step:
        if is_time_coordinate_variable(ref_da, cmp_da):
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

    mask_is_equal = np.array_equal(reference_masked.mask, comparison_masked.mask)

    if difference_field.size == 0:
        # Two empty (zero-size) fields are trivially identical. Guard this before
        # the all-NaN check below, whose `np.isnan(...).all()` is vacuously True
        # for an empty array and would otherwise mislabel it as "only NaN".
        return CompareResult(
            relative_error=0.0,
            min_diff=0.0,
            max_diff=0.0,
            mask_equal=True,
            variable=variable,
            note="Both fields are empty (zero-size); treated as identical.",
        )

    if np.isnan(difference_field).all():
        # Every difference is NaN. If both fields are NaN in the same positions
        # they are identical (a passing result with a note); otherwise their
        # valid regions do not overlap and the fields genuinely differ, which is
        # a failure and so carries its reason on `description`, not `note`.
        if mask_is_equal:
            return CompareResult(
                relative_error=0.0,
                min_diff=0.0,
                max_diff=0.0,
                mask_equal=True,
                variable=variable,
                note="Both fields contain only NaN values; treated as identical.",
            )
        return CompareResult(
            mask_equal=False,
            variable=variable,
            description="All differences are NaN (no overlapping valid points).",
        )

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


def is_time_coordinate_variable(ref_da: xr.DataArray, cmp_da: xr.DataArray) -> bool:
    """True when both sides are a 1-D datetime time axis (the time coordinate).

    Detection is by dimension shape and dtype rather than by name, so it also
    holds for a mapped pair whose time axes are named differently (e.g. a
    ``time=time2`` mapping), where a name-based check would miss it.
    """
    return _is_time_axis(ref_da) and _is_time_axis(cmp_da)


def _is_time_axis(da: xr.DataArray) -> bool:
    time_dimension = find_time_dims_name(da.dims)
    return time_dimension is not None and da.dims == (time_dimension,) and is_time_dtype(da.dtype)


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


def get_dataset_variables(
    dataset: xr.Dataset,
    variables: tuple[tuple[str, str], ...] | list | object | None,
) -> list[tuple[str, str]]:
    """Select comparable (reference_name, comparison_name) variable pairs.

    Filtering (presence + non-string dtype) is done against the reference
    dataset; the comparison-side name is validated later in ``compare_datasets``
    against the comparison dataset. When no selection is given, defaults to every
    data variable and dimension, compared under the same name on both sides.
    """
    if variables in (None, settings.DEFAULT_VARIABLES_TO_CHECK):
        pairs = [(name, name) for name in list(dataset.data_vars) + list(dataset.dims)]
    else:
        pairs = [_as_variable_pair(item) for item in variables]

    selected_variables: list[tuple[str, str]] = []
    for reference_name, comparison_name in pairs:
        if reference_name not in dataset:
            continue

        dtype_kind = dataset[reference_name].dtype.kind
        if dtype_kind in ("U", "S", "O", "a"):
            logger.debug("Skipping variable %s due to datatype %s", reference_name, dtype_kind)
            continue

        selected_variables.append((reference_name, comparison_name))

    return selected_variables
