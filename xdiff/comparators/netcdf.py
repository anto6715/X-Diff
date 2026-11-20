"""Comparator for netCDF artifacts and related numeric helpers."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Hashable, Iterable
from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal

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
        if variables in (None, settings.DEFAULT_VARIABLES_TO_CHECK):
            return compare_all_variables(dataset1, dataset2, last_time_step=last_time_step)
        # Every -v name gets a row, so one missing from a file fails instead of vanishing.
        return compare_datasets(dataset1, dataset2, list(variables), last_time_step=last_time_step)


ONLY_IN_REFERENCE = "Only in the reference file"
ONLY_IN_COMPARISON = "Only in the comparison file"
# A dimension says so: the same name can be a variable in one file and a dimension in the other.
DIMENSION_ONLY_IN_REFERENCE = "Dimension only in the reference file"
DIMENSION_ONLY_IN_COMPARISON = "Dimension only in the comparison file"
# Kept short: it shows in the DESCR column of every passing row of such a dimension.
DIMENSION_SIZE_ONLY = "size only"
# A name given with -v that a file lacks. The row label shows both names of a REF=CMP pair.
NOT_IN_REFERENCE = "Not in the reference file"
NOT_IN_COMPARISON = "Not in the comparison file"
NOT_IN_EITHER = "Not in either file"


def compare_all_variables(
    reference: xr.Dataset,
    comparison: xr.Dataset,
    *,
    last_time_step: bool,
) -> list[CompareResult]:
    """Compare everything in either dataset: the selection used when no variable is given.

    Rows come in this order:

    1. every variable of the reference (data variables, then dimension and auxiliary
       coordinates) that the comparison also has, compared by value (text, cftime
       dates and booleans for equality, see ``compare_variables``);
    2. every dimension without a coordinate variable in both datasets, compared by size;
    3. every variable, then every dimension without a coordinate variable, found in only
       one of the datasets, as a failure that names the side it is in.

    With ``last_time_step``, the 1-D time axes (datetime64 or cftime) and the size of a
    time dimension without a coordinate variable are left out: each file is reduced to
    its last step, whatever its length.
    """
    pairs: list[tuple[str, str]] = []
    one_sided: list[CompareResult] = []
    for name in _variable_names(reference):
        if name in comparison.variables:
            pairs.append((name, name))
        else:
            one_sided.append(CompareResult(variable=name, description=ONLY_IN_REFERENCE))
    if last_time_step:
        # The time coordinates cannot be compared on a single step. Leave them out so the
        # option works without -v; asking for one explicitly (-v time) still reports the error.
        pairs = _without_time_axes(reference, comparison, pairs)

    results = compare_datasets(reference, comparison, pairs, last_time_step=last_time_step)

    for name in _dimensions_without_coordinate(reference):
        if name not in comparison.sizes:
            one_sided.append(CompareResult(variable=name, description=DIMENSION_ONLY_IN_REFERENCE))
        elif name in comparison.variables:
            continue  # the comparison has a coordinate variable for it, reported below as one-sided
        elif last_time_step and find_time_dims_name([name]) is not None:
            # --last-time-step compares one step of each file, whatever their lengths.
            logger.debug("Skipping the size of %s: only the last time step is compared", name)
        else:
            results.append(compare_dimension_sizes(name, reference.sizes[name], comparison.sizes[name]))

    for name in _variable_names(comparison):
        if name not in reference.variables:
            one_sided.append(CompareResult(variable=name, description=ONLY_IN_COMPARISON))
    for name in _dimensions_without_coordinate(comparison):
        if name not in reference.sizes:
            one_sided.append(CompareResult(variable=name, description=DIMENSION_ONLY_IN_COMPARISON))

    return results + one_sided


def _variable_names(dataset: xr.Dataset) -> list:
    """Data variables, then coordinates (dimension and auxiliary), in file order."""
    return list(dataset.data_vars) + list(dataset.coords)


def _dimensions_without_coordinate(dataset: xr.Dataset) -> list:
    return [name for name in dataset.sizes if name not in dataset.variables]


def compare_dimension_sizes(name: str, reference_size: int, comparison_size: int) -> CompareResult:
    """Compare a dimension that has no coordinate variable: its size is all there is."""
    if reference_size != comparison_size:
        return CompareResult(
            variable=name,
            description=f"Dimension size mismatch: {reference_size} - {comparison_size}",
        )
    return CompareResult(
        relative_error=0.0,
        min_diff=0.0,
        max_diff=0.0,
        mask_equal=True,
        variable=name,
        note=DIMENSION_SIZE_ONLY,
    )


def is_text_or_object(field: xr.DataArray) -> bool:
    """Strings, bytes or Python objects (such as cftime dates): values with no numeric difference."""
    return field.dtype.kind in ("U", "S", "O")


def _value_kind(field: xr.DataArray) -> str | None:
    """What a text or object array holds: ``str``, ``bytes``, ``cftime <calendar>`` or ``object``.

    None for any other dtype, without loading the values. xarray reads text as an object
    array when it has a ``_FillValue``, so an object array is classified by its first
    value that is not missing; ``object`` when all are missing or the value is of another
    type.
    """
    kind = field.dtype.kind
    if kind == "U":
        return "str"
    if kind == "S":
        return "bytes"
    if kind != "O":
        return None
    from pandas import isnull  # an xarray dependency; imported here to keep startup light

    values = np.asarray(field.values).ravel()
    present = np.flatnonzero(~isnull(values))
    if present.size == 0:
        return "object"
    sample = values[present[0]]
    if isinstance(sample, str):
        return "str"
    if isinstance(sample, bytes):
        return "bytes"
    # Recognized by module rather than isinstance: cftime is not a declared dependency (it
    # comes with netCDF4), and xarray only produces these objects when it is installed.
    if type(sample).__module__.split(".")[0] == "cftime":
        return f"cftime {sample.calendar}"
    return "object"


# Kept short, like DIMENSION_SIZE_ONLY: no numeric difference was computed for the row.
EQUALITY_ONLY = "equality only"


def compare_for_equality(reference_values: np.ndarray, comparison_values: np.ndarray, variable: str) -> CompareResult:
    """Compare values that have no numeric difference (text, cftime dates, booleans) one by one.

    A value missing on both sides (None or NaN in an object array) counts as equal. The
    two arrays must have the same shape.
    """
    from pandas import isnull  # an xarray dependency; imported here to keep startup light

    reference_missing = isnull(reference_values)
    comparison_missing = isnull(comparison_values)
    equal = np.asarray((reference_values == comparison_values) | (reference_missing & comparison_missing))
    differing = int(np.count_nonzero(~equal))
    if differing:
        return CompareResult(
            mask_equal=bool(np.array_equal(reference_missing, comparison_missing)),
            variable=variable,
            description=f"{differing} of {equal.size} values differ",
        )
    return CompareResult(
        relative_error=0.0,
        min_diff=0.0,
        max_diff=0.0,
        mask_equal=True,
        variable=variable,
        note=EQUALITY_ONLY,
    )


def _without_time_axes(
    reference: xr.Dataset,
    comparison: xr.Dataset,
    pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Drop the pairs that ``compare_variables`` would reject under ``--last-time-step``.

    Only a pair that is a time axis of the same kind on *both* sides (datetime64, or cftime
    dates in one calendar) is dropped: that is exactly the case that raises
    ``LastTimestepTimeCheckException``. A time axis that differs between the files (other
    calendar, undecoded units, missing) stays in the selection so the mismatch is still
    reported.
    """
    kept: list[tuple[str, str]] = []
    for reference_name, comparison_name in pairs:
        if comparison_name in comparison and is_time_coordinate_variable(
            reference[reference_name], comparison[comparison_name]
        ):
            logger.debug("Skipping time axis %s: it cannot be compared on a single time step", reference_name)
            continue
        kept.append((reference_name, comparison_name))
    return kept


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


# Data types that hold a missing value (NaN, NaT) without changing: float, complex, dates, durations.
_MASKABLE_KINDS = "fcmM"


def _crop_curvilinear(dataset: xr.Dataset, longitude: xr.DataArray, latitude: xr.DataArray, bbox: BoundingBox):
    """Crop a 2-D (curvilinear) grid, e.g. NEMO nav_lon/nav_lat, or 1-D auxiliary lon/lat.

    Keeps the smallest index rectangle that holds the box, then masks the points of that
    rectangle outside the box, on the data variables that use the grid's dimensions and
    can hold a missing value. Other variables keep their data type and are compared over
    the whole rectangle: masking would turn integers and booleans into floats and text
    into objects. Variables that do not use the grid's dimensions are left as they are
    (``Dataset.where`` would broadcast them onto the grid).
    """
    inside_box = (
        (longitude >= bbox.lon_min)
        & (longitude <= bbox.lon_max)
        & (latitude >= bbox.lat_min)
        & (latitude <= bbox.lat_max)
    )
    rectangle = {}
    for dim in inside_box.dims:
        other_dims = [other for other in inside_box.dims if other != dim]
        holds_box = inside_box.any(dim=other_dims) if other_dims else inside_box
        rectangle[dim] = np.flatnonzero(holds_box.values)
    cropped = dataset.isel(rectangle)
    inside_rectangle = inside_box.isel(rectangle)
    masked = {
        name: variable.where(inside_rectangle)
        for name, variable in cropped.data_vars.items()
        if variable.dtype.kind in _MASKABLE_KINDS and set(inside_rectangle.dims) <= set(variable.dims)
    }
    return cropped.assign(masked)


def _as_variable_pair(item: str | tuple[str, str]) -> tuple[str, str]:
    """Coerce a variable spec to a (reference_name, comparison_name) pair."""
    if isinstance(item, str):
        return (item, item)
    reference_name, comparison_name = item  # unpacking also asserts a 2-element pair
    return reference_name, comparison_name


def variable_label(reference_name: str, comparison_name: str) -> str:
    """Display label for a variable pair: the name, or ``ref -> cmp`` when they differ."""
    if reference_name == comparison_name:
        return reference_name
    return f"{reference_name} -> {comparison_name}"


def compare_datasets(
    reference: xr.Dataset,
    comparison: xr.Dataset,
    variables: list[str] | list[tuple[str, str]],
    *,
    last_time_step: bool,
) -> list[CompareResult]:
    """Compare each name, or (reference_name, comparison_name) pair, on its own row.

    A variable is compared by value (see ``compare_variables``) and a dimension without a
    coordinate variable by size. A name missing from a file fails and says which file
    lacks it. A name that is a variable in one file but only a dimension in the other
    fails as a variable found in one file only, the row ``compare_all_variables`` gives
    it. An error while comparing a pair is recorded on its row instead of raised.
    """
    results: list[CompareResult] = []

    for item in variables:
        reference_name, comparison_name = _as_variable_pair(item)
        label = variable_label(reference_name, comparison_name)
        logger.info("Comparing %s", label)
        try:
            results.append(
                _compare_names(
                    reference,
                    comparison,
                    reference_name,
                    comparison_name,
                    label,
                    last_time_step=last_time_step,
                )
            )
        except Exception as exc:
            results.append(CompareResult(variable=label, description=str(exc)))

    return results


def _compare_names(
    reference: xr.Dataset,
    comparison: xr.Dataset,
    reference_name: Hashable,
    comparison_name: Hashable,
    label: str,
    *,
    last_time_step: bool,
) -> CompareResult:
    reference_kind = _name_kind(reference, reference_name)
    comparison_kind = _name_kind(comparison, comparison_name)

    if reference_kind is None or comparison_kind is None:
        if reference_kind is None and comparison_kind is None:
            description = NOT_IN_EITHER
        elif reference_kind is None:
            description = NOT_IN_REFERENCE
        else:
            description = NOT_IN_COMPARISON
        return CompareResult(variable=label, description=description)

    if reference_kind == comparison_kind == "variable":
        return compare_variables(
            reference[reference_name],
            comparison[comparison_name],
            label,
            last_time_step=last_time_step,
        )

    if reference_kind == comparison_kind == "dimension":
        names = (reference_name, comparison_name)
        if last_time_step and any(find_time_dims_name([name]) is not None for name in names):
            # Same as asking for a time coordinate (see compare_variables): only one step is compared.
            raise LastTimestepTimeCheckException("Can't compare time if last time step is enabled")
        return compare_dimension_sizes(label, reference.sizes[reference_name], comparison.sizes[comparison_name])

    # A variable in one file, only a dimension in the other: the same row as without -v.
    description = ONLY_IN_REFERENCE if reference_kind == "variable" else ONLY_IN_COMPARISON
    return CompareResult(variable=label, description=description)


def _name_kind(dataset: xr.Dataset, name: Hashable) -> Literal["variable", "dimension"] | None:
    """``"variable"``, ``"dimension"`` (a dimension without coordinate variable) or None if absent."""
    if name in dataset.variables:
        return "variable"
    if name in dataset.sizes:
        return "dimension"
    return None


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

    if is_text_or_object(ref_da) or ref_da.dtype == np.bool_:
        # Both sides hold text or objects, or both booleans (the data type check above makes
        # sure). Neither has a numeric difference, and numpy refuses to subtract booleans.
        return compare_for_equality(ref_da.values, cmp_da.values, variable)

    reference_values = ref_da.values
    comparison_values = cmp_da.values
    reference_masked = ref_da.to_masked_array()
    comparison_masked = cmp_da.to_masked_array()

    # Integer dtypes wrap around on subtraction (e.g. uint8: 10 - 12 -> 254),
    # silently corrupting min/max/relative-error. Promote integer operands to
    # float before subtracting; float and datetime/timedelta dtypes already
    # subtract correctly and are left untouched (the time path relies on the
    # resulting timedelta64). Test the kind, not np.integer: numpy counts timedelta64
    # as an integer type, and converting durations to float broke their ratio.
    if reference_values.dtype.kind in "iu":
        difference_field = reference_values.astype(np.float64) - comparison_values.astype(np.float64)
    else:
        difference_field = reference_values - comparison_values
    if difference_field.dtype.kind == "m":
        # A date or duration difference: keep it in nanoseconds. float() of a timedelta64 in
        # a coarser unit (s, ms, us) returns a datetime.timedelta, which CompareResult.passed
        # cannot turn into a number, and the run would abort.
        difference_field = difference_field.astype("timedelta64[ns]")

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
        relative_error=compute_relative_error(difference_field, reference_values),
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


# Points per block in compute_relative_error. Its float64 temporaries stay at about 2 MB
# whatever the size of the field, and small enough to stay in the CPU cache: measured on
# 50M points, 2**16 was faster than 2**20 and 2**14.
RELATIVE_ERROR_BLOCK_SIZE = 1 << 16


def compute_relative_error(diff: np.ndarray, field2: np.ndarray) -> float:
    """The maximum relative error ``|diff| / |field2|``, where ``field2`` is the reference field.

    ``diff`` is the difference (reference minus comparison) and ``field2`` the reference,
    broadcast to the shape of ``diff``. Only points that differ count: a point whose
    difference is 0 (equal values, two zeros included) or NaN (missing on either side) is
    left out, and the result is 0.0 when no point differs. A difference where the reference
    is 0 gives ``inf``. Dates (datetime64) have no relative error: NaN when they differ, 0.0
    when they do not. Durations (timedelta64) give the plain ratio of the two durations.
    """
    diff = np.asarray(diff)
    reference = np.broadcast_to(np.asarray(field2), diff.shape)
    if np.issubdtype(reference.dtype, np.datetime64):
        # A date has no zero to measure an error against.
        return np.nan if _differs(diff).any() else 0.0

    flat_diff = diff.reshape(-1)
    flat_reference = reference.reshape(-1)
    result = 0.0
    # The errstate silences a difference over a zero reference (inf), inf/inf and overflow.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for start in range(0, flat_diff.size, RELATIVE_ERROR_BLOCK_SIZE):
            block = slice(start, start + RELATIVE_ERROR_BLOCK_SIZE)
            block_diff = flat_diff[block]
            differs = _differs(block_diff)
            if not differs.any():
                continue
            ratios = _magnitude(block_diff) / _magnitude(flat_reference[block])
            if block_diff.dtype.kind != "m" and np.isinf(block_diff).any():
                # inf/inf gives NaN at a point that differs, so only a masked maximum
                # keeps it apart from the NaN of points that do not.
                block_max = float(ratios.max(where=differs, initial=0.0))
            else:
                # A point that does not differ gives 0 (0/|ref|) or NaN (0/0, a missing
                # value), and fmax skips NaN: the same maximum as one masked to the points
                # that differ, without the masked reduction's cost on scattered points.
                block_max = float(np.fmax.reduce(ratios, initial=0.0))
            if np.isnan(block_max):
                return np.nan  # inf/inf: an infinite reference against a finite comparison
            result = max(result, block_max)
    return result


def _differs(diff: np.ndarray) -> np.ndarray:
    """True where the difference is neither 0 nor NaN (a value missing on either side)."""
    differs = diff != 0
    differs &= ~np.isnan(diff)
    return differs


def _magnitude(values: np.ndarray) -> np.ndarray:
    """``|values|`` as float64, or as durations for timedelta64.

    Integers are converted before taking the absolute value, which overflows for them
    (``abs(int8(-128))`` is ``-128``), and complex values are widened to complex128 first.
    Dividing two durations already gives a float64 ratio.
    """
    if np.issubdtype(values.dtype, np.timedelta64):
        return np.abs(values)
    if np.iscomplexobj(values):
        return np.abs(values.astype(np.complex128, copy=False))
    return np.abs(values, dtype=np.float64)


def is_time_dtype(dtype) -> bool:
    normalized_dtype = np.dtype(dtype)
    return np.issubdtype(normalized_dtype, np.datetime64) or np.issubdtype(normalized_dtype, np.timedelta64)


def is_time_coordinate_variable(ref_da: xr.DataArray, cmp_da: xr.DataArray) -> bool:
    """True when both sides are a 1-D time axis (the time coordinate) of the same kind.

    The same kind is datetime64 on both sides, or cftime dates (what a non-standard
    calendar decodes to) in the same calendar. Detection is by dimension shape and
    values rather than by name, so it also holds for a mapped pair whose time axes are
    named differently (e.g. a ``time=time2`` mapping), where a name-based check would
    miss it.
    """
    reference_kind = _time_axis_kind(ref_da)
    return reference_kind is not None and reference_kind == _time_axis_kind(cmp_da)


def _time_axis_kind(da: xr.DataArray) -> str | None:
    """The kind of a 1-D time axis: its datetime64/timedelta64 dtype kind, or ``cftime <calendar>``."""
    if da.ndim != 1:
        # Only a 1-D variable can be an axis. Returning early also keeps a variable with
        # several time-like dimensions from raising here, outside the per-variable handling.
        return None
    time_dimension = find_time_dims_name(da.dims)
    if time_dimension is None or da.dims != (time_dimension,):
        return None
    if is_time_dtype(da.dtype):
        return da.dtype.kind
    value_kind = _value_kind(da)
    return value_kind if value_kind is not None and value_kind.startswith("cftime") else None


def validate_matching_metadata(ref_da: xr.DataArray, cmp_da: xr.DataArray) -> None:
    if ref_da.dims != cmp_da.dims:
        logger.debug(
            "Dimension name mismatch: '%s' - '%s'",
            ref_da.dims,
            cmp_da.dims,
        )

    if ref_da.shape != cmp_da.shape:
        raise ValueError(f"Dimension size mismatch: '{ref_da.shape}' - '{cmp_da.shape}'")

    reference_kind, comparison_kind = _value_kind(ref_da), _value_kind(cmp_da)
    if not _same_data_type(ref_da.dtype, cmp_da.dtype, reference_kind, comparison_kind):
        reference_type = _describe_data_type(ref_da.dtype, reference_kind)
        comparison_type = _describe_data_type(cmp_da.dtype, comparison_kind)
        raise ValueError(f"Data type mismatch: '{reference_type}' - '{comparison_type}'")

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


def _same_data_type(reference_dtype, comparison_dtype, reference_kind, comparison_kind) -> bool:
    """Equal numeric dtypes, or text and objects holding the same kind of value (``_value_kind``).

    For text the dtype says how values are stored, not what they are: xarray sizes vlen
    strings to their longest value and char arrays to their string-length dimension, and
    reads text with a ``_FillValue`` as an object array. So ``<U3``, ``<U5`` and an object
    array of str all compare as str, and ``compare_for_equality`` reports what differs.
    Bytes against str, and cftime dates in two calendars, still differ. An object array
    with no value to classify (all missing) is compared whatever the other side holds.
    """
    if reference_kind is None or comparison_kind is None:
        return np.dtype(reference_dtype) == np.dtype(comparison_dtype)
    return reference_kind == comparison_kind or "object" in (reference_kind, comparison_kind)


def _describe_data_type(dtype, value_kind: str | None) -> str:
    """The dtype, plus what an object array holds: ``object (cftime noleap)``."""
    if np.dtype(dtype).kind == "O" and value_kind is not None:
        return f"object ({value_kind})"
    return str(dtype)


def get_dataset_variables(
    dataset: xr.Dataset,
    variables: tuple[tuple[str, str], ...] | list | object | None,
) -> list[tuple[str, str]]:
    """Select the (reference_name, comparison_name) pairs that ``xdiff plot`` considers.

    A name that is not a variable of the reference dataset is dropped (``xdiff plot``
    reports a dropped ``-v`` name itself); the comparison-side name is checked later.
    When no selection is given, defaults to every data variable and dimension coordinate
    (a dimension without a coordinate variable is dropped like any other name), under
    the same name on both sides. ``xdiff plot`` then leaves out text and object
    variables, which have no difference to plot.

    ``compare_files`` does not use this: it hands a ``-v`` selection to ``compare_datasets``
    as given, and with no selection calls ``compare_all_variables``.
    """
    if variables in (None, settings.DEFAULT_VARIABLES_TO_CHECK):
        pairs = [(name, name) for name in list(dataset.data_vars) + list(dataset.dims)]
    else:
        pairs = [_as_variable_pair(item) for item in variables]

    selected_variables: list[tuple[str, str]] = []
    for reference_name, comparison_name in pairs:
        if reference_name not in dataset:
            continue

        selected_variables.append((reference_name, comparison_name))

    return selected_variables
