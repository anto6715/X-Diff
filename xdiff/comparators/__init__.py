"""Comparator implementations for each artifact type."""

from importlib import import_module

from xdiff.comparators.base import ArtifactComparator

__all__ = ["ArtifactComparator", "NetcdfComparator"]


def __getattr__(name: str):
    if name == "NetcdfComparator":
        comparator = getattr(import_module("xdiff.comparators.netcdf"), name)
        globals()[name] = comparator
        return comparator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
