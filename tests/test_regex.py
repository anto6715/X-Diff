from pathlib import Path

from nccompare.utils.regex import common_pattern_exists, find_file_matches


def test_common_pattern_exists_returns_false_when_pattern_is_none():
    assert common_pattern_exists("a.nc", "b.nc", None) is False


def test_find_file_matches_matches_same_filename():
    reference = Path("reference/file.nc")
    comparison = Path("comparison/file.nc")

    matches = find_file_matches([reference], [comparison])

    assert matches == {reference: [comparison]}


def test_find_file_matches_matches_on_common_pattern():
    reference = Path("reference/my-simu_19820101_grid_T.nc")
    comparison = Path("comparison/another-exp_19820101_grid_T.nc")

    matches = find_file_matches(
        [reference], [comparison], r"\d{8}_grid_T\.nc"
    )

    assert matches == {reference: [comparison]}


def test_find_file_matches_returns_empty_list_when_no_match_exists():
    reference = Path("reference/file_a.nc")
    comparison = Path("comparison/file_b.nc")

    matches = find_file_matches([reference], [comparison], r"\d{8}_grid_T\.nc")

    assert matches == {reference: []}
