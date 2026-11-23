"""Helpers for regex-based artifact and filename matching."""

from __future__ import annotations

import re


def common_pattern_exists(first_str: str, second_str: str, pattern: str) -> bool:
    """
    Check if pattern exists in first string and second string and if they match exactly the same string

    Example:
        first_str = "mfs-eas8_20150101_grid_T.nc"
        second_str = "my-simu_20150101_grid_T.nc"
        pattern = "\\d{8}"

    Args:
        first_str: First string
        second_str: Second string
        pattern: regex pattern

    Returns:
        True if pattern match both first_str and second_str and the matched string is the same.
    """
    first_key = common_pattern_key(first_str, pattern)
    return first_key is not None and first_key == common_pattern_key(second_str, pattern)


def common_pattern_key(name: str, pattern: str | None) -> tuple | None:
    """Return what ``pattern`` finds in ``name``, sorted, or ``None`` when it finds nothing.

    Two names share a common pattern exactly when their keys are equal and not ``None``.
    """
    if pattern is None:
        return None

    found = re.findall(pattern, name)
    return tuple(sorted(found)) if found else None
