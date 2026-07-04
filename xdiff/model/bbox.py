"""Geographic bounding box for spatial subsetting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A lon/lat window used to crop datasets before comparison.

    Only same-grid inputs of different extent are supported: cropping selects the
    grid cells inside the box on each side, so the two croppings only line up when
    the underlying grids match. Antimeridian-crossing boxes (``lon_min > lon_max``)
    are not supported.
    """

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def __post_init__(self) -> None:
        if self.lon_min >= self.lon_max:
            raise ValueError(f"--bbox longitude min ({self.lon_min}) must be less than max ({self.lon_max}).")
        if self.lat_min >= self.lat_max:
            raise ValueError(f"--bbox latitude min ({self.lat_min}) must be less than max ({self.lat_max}).")
        if not (-90.0 <= self.lat_min <= 90.0 and -90.0 <= self.lat_max <= 90.0):
            raise ValueError(f"--bbox latitude values must be within [-90, 90], got {self.lat_min}..{self.lat_max}.")

    def __str__(self) -> str:
        return f"lon [{self.lon_min}, {self.lon_max}], lat [{self.lat_min}, {self.lat_max}]"
