from pathlib import Path

import pytest

from nccompare.conf import settings
from nccompare.management.cli import get_args


def test_get_args_parses_filter_common_pattern_variables_and_last_time_step():
    args = get_args(
        [
            "ref-dir",
            "cmp-dir",
            "--filter",
            "*_grid_T.nc",
            "--common-pattern",
            r"\d{8}_grid_T\.nc",
            "--variables",
            "votemper",
            "vosaline",
            "--last_time_step",
        ]
    )

    assert args.folder1 == Path("ref-dir")
    assert args.folder2 == Path("cmp-dir")
    assert args.filter_name == "*_grid_T.nc"
    assert args.common_pattern == r"\d{8}_grid_T\.nc"
    assert args.variables == ["votemper", "vosaline"]
    assert args.last_time_step is True


def test_get_args_supports_short_filter_option():
    args = get_args(["ref-dir", "cmp-dir", "-f", "*.nc"])

    assert args.filter_name == "*.nc"


def test_get_args_uses_defaults_when_optional_args_are_omitted():
    args = get_args(["ref-dir", "cmp-dir"])

    assert args.filter_name == settings.DEFAULT_NAME_TO_COMPARE
    assert args.common_pattern is settings.DEFAULT_COMMON_PATTERN
    assert args.variables is settings.DEFAULT_VARIABLES_TO_CHECK
    assert args.last_time_step is False


def test_get_args_prints_version_and_exits(capsys):
    with pytest.raises(SystemExit, match="0"):
        get_args(["--version"])

    captured = capsys.readouterr()
    assert captured.out.strip()
