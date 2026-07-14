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

        for reference_name, comparison_name in get_dataset_variables(reference_ds, variables):
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
) -> tuple[xr.DataArray, dict[str, int]]:
    """Collapse a field to ≤2-D, keeping the horizontal dims.

    Every non-horizontal dim is selected down to a single index: the time dim (per
    ``find_time_dims_name``) to its last step when ``last_time_step`` else its first,
    any other extra dim (e.g. depth) to index 0. Returns the reduced array and a
    ``{dim: selected_index}`` map of what was collapsed, so the caller can make the
    selection visible in the label (an info log alone hides that only one layer is shown).
    """
    time_name = find_time_dims_name(field.dims)
    selection: dict[str, int] = {}
    for dim in field.dims:
        if dim in horizontal_dims:
            continue
        if dim == time_name and last_time_step:
            selection[dim] = int(field.sizes[dim]) - 1
        else:
            selection[dim] = 0
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
) -> VariablePlot:
    if comparison_name not in comparison_ds:
        raise ValueError(f"{comparison_name!r} not present in comparison dataset")

    reference_field = reference_ds[reference_name]
    comparison_field = comparison_ds[comparison_name]

    reference_reduced, collapsed = reduce_to_plottable(reference_field, horizontal_dims, last_time_step=last_time_step)
    comparison_reduced, _ = reduce_to_plottable(comparison_field, horizontal_dims, last_time_step=last_time_step)

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

    if collapsed:
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
