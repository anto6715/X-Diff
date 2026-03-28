"""Click-based CLI for directory and file comparison."""

from __future__ import annotations

import importlib.metadata

from pathlib import Path

import click

from xdiff import core
from xdiff.conf import settings
from xdiff.model import CompareMode
from xdiff.printlib import formatter

def _validate_netcdf_file(ctx, param, value: Path | None) -> Path | None:
    """Ensure the explicit files command only accepts netCDF inputs for now."""
    if value is None:
        return value

    if value.suffix.lower() != ".nc":
        raise click.BadParameter("only .nc files are supported by the files command")

    return value


def _render_report(**kwargs) -> None:
    """Execute a comparison request and print the resulting report."""
    formatter.print_report(core.execute(**kwargs))


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(
    version=importlib.metadata.version("xdiff"),
    prog_name="xdiff",
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """netCDF comparison tool."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command("dirs")
@click.argument(
    "reference_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "comparison_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "-f",
    "--filter",
    "filter_name",
    default=settings.DEFAULT_NAME_TO_COMPARE,
    show_default=True,
    help="Filter to select files to compare. Examples: *.nc, *_grid_*.",
)
@click.option(
    "--common-pattern",
    default=settings.DEFAULT_COMMON_PATTERN,
    help=(
        "Common file pattern in two files to compare. "
        "Example: mfsX_date.nc and expX_date.nc -> date.nc is the common part."
    ),
)
@click.option(
    "-v",
    "--variables",
    multiple=True,
    help="Variables to compare. Repeat the option to compare multiple variables.",
)
@click.option(
    "--last-time-step",
    is_flag=True,
    default=False,
    help="If enabled, compare only the last time step available in each file.",
)
def compare_directories(
    reference_path: Path,
    comparison_path: Path,
    filter_name: str,
    common_pattern: str | None,
    variables: tuple[str, ...],
    last_time_step: bool,
) -> None:
    """Compare two directories of netCDF files."""
    _render_report(
        reference_path=reference_path,
        comparison_path=comparison_path,
        input_mode=CompareMode.DIRECTORIES,
        filter_name=filter_name,
        common_pattern=common_pattern,
        variables=variables or settings.DEFAULT_VARIABLES_TO_CHECK,
        last_time_step=last_time_step,
    )


@cli.command("files")
@click.argument(
    "reference_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    callback=_validate_netcdf_file,
)
@click.argument(
    "comparison_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    callback=_validate_netcdf_file,
)
@click.option(
    "-v",
    "--variables",
    multiple=True,
    help="Variables to compare. Repeat the option to compare multiple variables.",
)
@click.option(
    "--last-time-step",
    is_flag=True,
    default=False,
    help="If enabled, compare only the last time step available in each file.",
)
def compare_files(
    reference_path: Path,
    comparison_path: Path,
    variables: tuple[str, ...],
    last_time_step: bool,
) -> None:
    """Compare two netCDF files directly, even if their filenames differ."""
    _render_report(
        reference_path=reference_path,
        comparison_path=comparison_path,
        input_mode=CompareMode.FILES,
        filter_name=settings.DEFAULT_NAME_TO_COMPARE,
        common_pattern=settings.DEFAULT_COMMON_PATTERN,
        variables=variables or settings.DEFAULT_VARIABLES_TO_CHECK,
        last_time_step=last_time_step,
    )


if __name__ == "__main__":
    cli()
