"""Helpers for regex-based artifact and filename matching."""

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
    if pattern is None:
        return False

    regex = re.compile(pattern)

    match1 = regex.findall(first_str)
    match2 = regex.findall(second_str)

    if match1 and match2:
        return sorted(match1) == sorted(match2)

    return False
