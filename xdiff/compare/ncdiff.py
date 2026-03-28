"""NetCDF comparison orchestration for matched file pairs."""

from pathlib import Path
from typing import Iterable

from xdiff.comparators.netcdf import compare_files
from xdiff.exceptions import NoMatchFound
from xdiff.model.comparison import Comparison


def compare(
    compare_match: dict[Path, list[Path]],
    variables: Iterable[str] | tuple[str, ...] | list[str] | object | None,
    last_time_step: bool,
):
    for reference, to_compares in compare_match.items():
        if len(to_compares) == 0:
            yield Comparison.from_paths(reference, None, NoMatchFound(f"No match found for {reference}"))
            continue

        try:
            for to_compare in to_compares:
                comparison = Comparison.from_paths(reference, to_compare)
                comparison.extend(compare_files(reference, to_compare, variables, last_time_step=last_time_step))
                yield comparison

        except Exception as e:
            yield Comparison.from_paths(reference, None, e)
