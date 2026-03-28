import re
from pathlib import Path
from typing import List, Dict


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


def find_file_matches(
    reference_input_files: List[Path],
    comparison_input_files: List[Path],
    common_pattern: str = None,
) -> Dict[Path, List[Path]]:
    """
    For each file in reference_input_files,
    return a list of file with the same filename or with the same substring matching common_pattern.
    If no match is found, an empty list is associated to that file.

    Args:
        reference_input_files: List of reference input files
        comparison_input_files: List of files to compare with reference input files
        common_pattern: regex expression to identify a common substring between two files

    Returns:

    """
    to_compare = dict()
    for ref in reference_input_files:
        to_compare[ref] = []
        for cmp in comparison_input_files:
            if ref.name == cmp.name or common_pattern_exists(
                ref.name, cmp.name, common_pattern
            ):
                to_compare[ref].append(cmp)

    return to_compare
