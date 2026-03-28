"""Comparator implementations for each artifact type."""

from nccompare.comparators.base import ArtifactComparator
from nccompare.comparators.netcdf import NetcdfComparator

__all__ = ["ArtifactComparator", "NetcdfComparator"]
