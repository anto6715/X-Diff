"""CLI argument parsing and direct script execution."""

import argparse
import importlib.metadata
import time

from pathlib import Path

from nccompare import core
from nccompare.conf import settings
from nccompare.printlib import formatter


def get_args(raw_args=None):
    parse = argparse.ArgumentParser(description="netCDF Comparison Tool")
    # General args
    parse.add_argument("folder1", type=Path, help="Path of first folder to compare")
    parse.add_argument("folder2", type=Path, help="Path of second folder to compare")
    parse.add_argument(
        "-f",
        "--filter",
        dest="filter_name",
        type=str,
        default=settings.DEFAULT_NAME_TO_COMPARE,
        help="Filter to select files to compare. Examples: *.nc, *_grid_*",
    )
    parse.add_argument(
        "--common-pattern",
        type=str,
        default=settings.DEFAULT_COMMON_PATTERN,
        help="Common file pattern in two files to compare"
        "Es mfsX_date.nc and expX_date.nc -> date.nc is the common part",
    )
    parse.add_argument(
        "-v",
        "--variables",
        nargs="+",
        default=settings.DEFAULT_VARIABLES_TO_CHECK,
        help="Variable to compare",
    )
    parse.add_argument(
        "--last_time_step",
        dest="last_time_step",
        action="store_true",
        default=False,
        help="If True, compare only the last time step available in each file",
    )
    parse.add_argument(
        "-V",
        "--version",
        action="version",
        version=importlib.metadata.version("nccompare"),
        help="Print version and exit",
    )
    return parse.parse_args(raw_args)


if __name__ == "__main__":
    start = time.perf_counter()
    args: argparse.Namespace = get_args()
    formatter.print_report(core.execute(**vars(args)))
    print(f"Run time: {time.perf_counter() - start}Sec")
