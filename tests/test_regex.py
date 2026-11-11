from xdiff.utils.regex import common_pattern_exists


def test_common_pattern_exists_returns_false_when_pattern_is_none():
    assert common_pattern_exists("a.nc", "b.nc", None) is False


def test_common_pattern_exists_matches_the_same_substring_in_both_names():
    assert common_pattern_exists(
        "my-simu_19820101_grid_T.nc",
        "another-exp_19820101_grid_T.nc",
        r"\d{8}_grid_T\.nc",
    )


def test_common_pattern_exists_rejects_different_matched_substrings():
    assert not common_pattern_exists(
        "my-simu_19820101_grid_T.nc",
        "another-exp_19820102_grid_T.nc",
        r"\d{8}_grid_T\.nc",
    )


def test_common_pattern_exists_rejects_names_where_the_pattern_is_absent():
    assert not common_pattern_exists("file_a.nc", "file_b.nc", r"\d{8}_grid_T\.nc")
