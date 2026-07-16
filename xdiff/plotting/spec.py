"""Backend-agnostic description of what to draw, and the builder that produces it.

All the *logic* of the plot feature lives here — reduction to a plottable slice,
integer-safe differencing, axis pickup, colour limits, and the skip policy. Both
renderers (static matplotlib, interactive server) consume the resulting
:class:`PlotSpec` and stay dumb. See ``docs/plot-diff-plan.md`` §2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from xdiff.comparators.netcdf import (
    crop_to_bbox,
    find_time_dims_name,
    get_dataset_variables,
    load_xarray,
    locate_horizontal_coords,
)

if TYPE_CHECKING:
    import xarray as xr

    from xdiff.model import BoundingBox


@dataclass(frozen=True)
class VariablePlot:
    """One variable reduced to a ≤2-D slice, ready for any renderer."""

    label: str  # "thetao", "thetao -> votemper", plus "[depth=0]" for collapsed dims
    reference: np.ndarray  # reduced to <=2-D
    comparison: np.ndarray  # same shape as reference (invariant, enforced by the builder)
    difference: np.ndarray  # reference - comparison, in float (integer-underflow-safe)
    lon: np.ndarray | None  # 1-D or 2-D axis coordinates, or None if undetectable
    lat: np.ndarray | None
    diff_limit: float  # DEFAULT symmetric colour limit: robust 99.5th-pct clip of |difference|
    diff_extreme: float  # TRUE symmetric extreme: max(|nanmin|, |nanmax|) of difference
    units: str | None  # from ref_da.attrs.get("units")
    dims: tuple[str, ...]  # reduced dims, for axis labels / 1-D vs 2-D dispatch


@dataclass(frozen=True)
class SkippedVariable:
    """A variable that could not be plotted, recorded rather than raised."""

    label: str
    reason: str


@dataclass(frozen=True)
class PlotSpec:
    """Everything the renderers need for one file pair."""

    reference_path: Path
    comparison_path: Path
    variables: list[VariablePlot]
    skipped: list[SkippedVariable]


def build_plot_spec(
    reference_path: Path,
    comparison_path: Path,
    variables,  # normalized (ref, cmp) pairs or None, same shape as CompareRequest.variables
    *,
    last_time_step: bool,
    bbox: BoundingBox | None,
) -> PlotSpec:
    """Open both files, reduce each selected variable to a plottable slice, diff them.

    Variables that cannot be plotted (shape mismatch after reduction, no horizontal
    dims, all-NaN difference, or any per-variable error) are recorded on
    ``PlotSpec.skipped`` instead of raising, so one bad variable never aborts the
    whole plot.
    """
    xr = load_xarray()
    plots: list[VariablePlot] = []
    skipped: list[SkippedVariable] = []

    with xr.open_dataset(reference_path) as reference_raw, xr.open_dataset(comparison_path) as comparison_raw:
        reference_ds = crop_to_bbox(reference_raw, bbox) if bbox is not None else reference_raw
        comparison_ds = crop_to_bbox(comparison_raw, bbox) if bbox is not None else comparison_raw

        longitude_name, latitude_name = locate_horizontal_coords(reference_ds)
        horizontal_dims = _horizontal_dims(reference_ds, longitude_name, latitude_name)

        candidate_pairs = get_dataset_variables(reference_ds, variables)
        if variables is None:
            # "All variables" also returns dims/coords (lon/lat/time); plotting those as
            # difference fields is meaningless, so keep only the actual data variables.
            candidate_pairs = [pair for pair in candidate_pairs if pair[0] not in reference_ds.coords]

        for reference_name, comparison_name in candidate_pairs:
            label = reference_name if reference_name == comparison_name else f"{reference_name} -> {comparison_name}"
            try:
                plots.append(
                    _build_variable_plot(
                        reference_ds,
                        comparison_ds,
                        reference_name,
                        comparison_name,
                        label,
                        longitude_name,
                        latitude_name,
                        horizontal_dims,
                        last_time_step=last_time_step,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — errors are data past this seam
                skipped.append(SkippedVariable(label=label, reason=str(exc)))

    return PlotSpec(
        reference_path=Path(reference_path),
        comparison_path=Path(comparison_path),
        variables=plots,
        skipped=skipped,
    )


def reduce_to_plottable(
    field: xr.DataArray,
    horizontal_dims: set,
    *,
    last_time_step: bool,
    selection_override: dict[str, int] | None = None,
) -> tuple[xr.DataArray, dict[str, int]]:
    """Collapse a field to ≤2-D, keeping the horizontal dims.

    Every non-horizontal dim is selected down to a single index. ``selection_override``
    (used by the live server's time/depth controls) pins specific indices; otherwise the
    time dim (per ``find_time_dims_name``) goes to its last step when ``last_time_step``
    else its first, and any other extra dim (e.g. depth) to index 0. Returns the reduced
    array and a ``{dim: selected_index}`` map of what was collapsed.
    """
    time_name = find_time_dims_name(field.dims)
    selection: dict[str, int] = {}
    for dim in field.dims:
        if dim in horizontal_dims:
            continue
        if selection_override is not None and dim in selection_override:
            index = int(selection_override[dim])
        elif dim == time_name and last_time_step:
            index = int(field.sizes[dim]) - 1
        else:
            index = 0
        selection[dim] = max(0, min(index, int(field.sizes[dim]) - 1))  # clamp to valid range
    reduced = field.isel(selection) if selection else field
    return reduced, selection


def _build_variable_plot(
    reference_ds: xr.Dataset,
    comparison_ds: xr.Dataset,
    reference_name: str,
    comparison_name: str,
    label: str,
    longitude_name,
    latitude_name,
    horizontal_dims: set,
    *,
    last_time_step: bool,
    selection_override: dict[str, int] | None = None,
    decorate_label: bool = True,
) -> VariablePlot:
    if comparison_name not in comparison_ds:
        raise ValueError(f"{comparison_name!r} not present in comparison dataset")

    reference_field = reference_ds[reference_name]
    comparison_field = comparison_ds[comparison_name]

    reference_reduced, collapsed = reduce_to_plottable(
        reference_field, horizontal_dims, last_time_step=last_time_step, selection_override=selection_override
    )
    comparison_reduced, _ = reduce_to_plottable(
        comparison_field, horizontal_dims, last_time_step=last_time_step, selection_override=selection_override
    )

    if reference_reduced.ndim == 0 or reference_reduced.ndim > 2:
        raise ValueError(f"no horizontal dims to plot (reduced to {reference_reduced.ndim}-D)")

    reference_values = np.asarray(reference_reduced.values)
    comparison_values = np.asarray(comparison_reduced.values)
    if reference_values.shape != comparison_values.shape:
        raise ValueError(f"shape {reference_values.shape} vs {comparison_values.shape}")

    # Integer dtypes wrap on subtraction (uint8: 10 - 12 -> 254); promote to float
    # first, mirroring compare_variables so the picture matches the numbers.
    if np.issubdtype(reference_values.dtype, np.integer):
        difference = reference_values.astype(np.float64) - comparison_values.astype(np.float64)
    else:
        difference = reference_values - comparison_values

    absolute_difference = np.abs(np.asarray(difference, dtype=np.float64))
    if not np.any(np.isfinite(absolute_difference)):
        raise ValueError("all-NaN difference (no overlapping valid points)")

    diff_extreme = float(np.nanmax(absolute_difference))
    diff_limit = float(np.nanpercentile(absolute_difference, 99.5))
    if not np.isfinite(diff_limit) or diff_limit == 0.0:
        # A degenerate limit (all-equal or single outlier at 0) would collapse the
        # diverging colormap; fall back to the true extreme, then to a unit range.
        diff_limit = diff_extreme if diff_extreme > 0.0 else 1.0

    if collapsed and decorate_label:
        label = f"{label} [{', '.join(f'{dim}={index}' for dim, index in collapsed.items())}]"

    units = reference_field.attrs.get("units")

    return VariablePlot(
        label=label,
        reference=reference_values,
        comparison=comparison_values,
        difference=difference,
        lon=_coordinate_values(reference_reduced, longitude_name),
        lat=_coordinate_values(reference_reduced, latitude_name),
        diff_limit=diff_limit,
        diff_extreme=diff_extreme,
        units=units,
        dims=tuple(str(dim) for dim in reference_reduced.dims),
    )


def valid_extent(variable: VariablePlot) -> tuple[float, float, float, float] | None:
    """``(lon_min, lon_max, lat_min, lat_max)`` over the cells that actually hold data.

    NEMO ``nav_lon``/``nav_lat`` carry fill values on masked cells, so the raw coordinate
    min/max would stretch a map far past the real domain (e.g. all of the Sahara under a
    Mediterranean field). Restrict the box to where the difference is finite. Returns
    ``None`` when coordinates are absent or nothing is finite.
    """
    if variable.lon is None or variable.lat is None:
        return None
    valid = np.isfinite(np.asarray(variable.difference, dtype=float))
    if not valid.any():
        return None
    lon = np.asarray(variable.lon, dtype=float)
    lat = np.asarray(variable.lat, dtype=float)
    if lon.ndim == 1 and lat.ndim == 1:
        columns = valid.any(axis=0)
        rows = valid.any(axis=1)
        return (float(lon[columns].min()), float(lon[columns].max()), float(lat[rows].min()), float(lat[rows].max()))
    return (float(lon[valid].min()), float(lon[valid].max()), float(lat[valid].min()), float(lat[valid].max()))


def _horizontal_dims(dataset: xr.Dataset, longitude_name, latitude_name) -> set:
    """The set of dims spanned by the lon/lat coordinates (the ones to keep)."""
    dims: set = set()
    for name in (longitude_name, latitude_name):
        if name is not None and name in dataset.variables:
            dims.update(dataset[name].dims)
    return dims


def _coordinate_values(reduced_field: xr.DataArray, name) -> np.ndarray | None:
    if name is None or name not in reduced_field.coords:
        return None
    return np.asarray(reduced_field.coords[name].values)


# --------------------------------------------------------------------------- live (sliceable) source


@dataclass(frozen=True)
class DimControl:
    """One selectable non-horizontal dimension (e.g. time or depth) of a variable."""

    name: str
    size: int
    labels: tuple[str, ...]  # human labels per index (coordinate values, or "0".."n-1")
    default: int  # initial index (last step for time under --last-time-step, else 0)


@dataclass(frozen=True)
class VariableHandle:
    """A plottable variable in a :class:`PlotSource`, sliced on demand (no data loaded yet)."""

    label: str  # base name / "ref -> cmp" (no [time=…] decoration; the controls show it)
    is_map: bool  # reduces to 2-D (vs a 1-D profile)
    extra_dims: tuple[DimControl, ...]  # the dims the server exposes as controls


class PlotSource:
    """A live source of plottable slices: keeps both datasets open and re-slices on demand.

    The static renderer consumes an eager :class:`PlotSpec`; the interactive server consumes
    this instead, so time/depth can be chosen without reducing/loading up front. Call
    :meth:`close` when the server stops. :meth:`static` wraps an already-reduced list of
    :class:`VariablePlot` (no open files, no extra dims) for callers/tests that don't slice.
    """

    def __init__(self, reference_path, comparison_path, variables, slicer, skipped=(), closer=None):
        self.reference_path = Path(reference_path)
        self.comparison_path = Path(comparison_path)
        self.variables: list[VariableHandle] = list(variables)
        self.skipped: list[SkippedVariable] = list(skipped)
        self._slicer = slicer
        self._closer = closer

    def slice(self, index: int, selection: dict[str, int]) -> VariablePlot:
        return self._slicer(index, selection)

    def close(self) -> None:
        if self._closer is not None:
            self._closer()

    @classmethod
    def static(cls, reference_path, comparison_path, variables: list[VariablePlot]) -> PlotSource:
        handles = [VariableHandle(label=plot.label, is_map=len(plot.dims) == 2, extra_dims=()) for plot in variables]
        return cls(reference_path, comparison_path, handles, lambda index, _selection: variables[index])


def open_plot_source(
    reference_path: Path,
    comparison_path: Path,
    variables,
    *,
    last_time_step: bool,
    bbox: BoundingBox | None,
) -> PlotSource:
    """Open both files (kept open) and expose their variables for on-demand slicing.

    Mirrors ``build_plot_spec``'s discovery (bbox crop, horizontal-coord location, coord/0-D
    filtering) but does NOT reduce or load data: each variable becomes a :class:`VariableHandle`
    carrying its selectable extra dims. ``PlotSource.slice`` then produces a 2-D VariablePlot for
    a chosen ``{dim: index}``. The caller must ``close()`` the returned source.
    """
    xr = load_xarray()
    reference_raw = xr.open_dataset(reference_path)
    comparison_raw = xr.open_dataset(comparison_path)
    try:
        reference_ds = crop_to_bbox(reference_raw, bbox) if bbox is not None else reference_raw
        comparison_ds = crop_to_bbox(comparison_raw, bbox) if bbox is not None else comparison_raw
        longitude_name, latitude_name = locate_horizontal_coords(reference_ds)
        horizontal_dims = _horizontal_dims(reference_ds, longitude_name, latitude_name)

        handles: list[VariableHandle] = []
        specs: list[tuple[str, str]] = []
        skipped: list[SkippedVariable] = []
        candidate_pairs = get_dataset_variables(reference_ds, variables)
        if variables is None:
            candidate_pairs = [pair for pair in candidate_pairs if pair[0] not in reference_ds.coords]

        for reference_name, comparison_name in candidate_pairs:
            label = reference_name if reference_name == comparison_name else f"{reference_name} -> {comparison_name}"
            field = reference_ds[reference_name]
            map_dims = [dim for dim in field.dims if dim in horizontal_dims]
            if len(map_dims) not in (1, 2):
                skipped.append(SkippedVariable(label=label, reason=f"no horizontal dims to plot ({len(map_dims)}-D)"))
                continue
            if comparison_name not in comparison_ds:
                reason = f"{comparison_name!r} not present in comparison dataset"
                skipped.append(SkippedVariable(label=label, reason=reason))
                continue
            handles.append(
                VariableHandle(
                    label=label,
                    is_map=len(map_dims) == 2,
                    extra_dims=_extra_dim_controls(reference_ds, field, horizontal_dims, last_time_step=last_time_step),
                )
            )
            specs.append((reference_name, comparison_name))
    except Exception:
        reference_raw.close()
        comparison_raw.close()
        raise

    def slicer(index: int, selection: dict[str, int]) -> VariablePlot:
        reference_name, comparison_name = specs[index]
        return _build_variable_plot(
            reference_ds,
            comparison_ds,
            reference_name,
            comparison_name,
            handles[index].label,
            longitude_name,
            latitude_name,
            horizontal_dims,
            last_time_step=last_time_step,
            selection_override=selection,
            decorate_label=False,
        )

    def closer() -> None:
        reference_raw.close()
        comparison_raw.close()

    return PlotSource(reference_path, comparison_path, handles, slicer, skipped, closer)


def _extra_dim_controls(dataset, field, horizontal_dims, *, last_time_step) -> tuple[DimControl, ...]:
    """The non-horizontal dims of ``field`` (time/depth/…) as selectable controls."""
    time_name = find_time_dims_name(field.dims)
    controls: list[DimControl] = []
    for dim in field.dims:
        if dim in horizontal_dims:
            continue
        size = int(field.sizes[dim])
        is_time = dim == time_name
        default = size - 1 if (is_time and last_time_step) else 0
        controls.append(DimControl(name=str(dim), size=size, labels=_dim_labels(dataset, dim, size), default=default))
    return tuple(controls)


def _dim_labels(dataset, dim, size) -> tuple[str, ...]:
    """Per-index labels for a dim: its coordinate values (depths, timestamps) when present, else indices."""
    if dim in dataset.coords:
        values = np.asarray(dataset[dim].values)
        if values.ndim == 1 and values.size == size:
            return tuple(_format_coord(value) for value in values)
    return tuple(str(index) for index in range(size))


def _format_coord(value) -> str:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.datetime64):
        return str(np.datetime_as_string(array, unit="m"))
    try:
        return f"{float(array):g}"
    except (TypeError, ValueError):
        return str(value)
