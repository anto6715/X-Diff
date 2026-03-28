"""Comparator implementations for each artifact type."""

from xdiff.comparators.base import ArtifactComparator
from xdiff.comparators.netcdf import NetcdfComparator

__all__ = ["ArtifactComparator", "NetcdfComparator"]
