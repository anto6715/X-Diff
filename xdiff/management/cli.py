"""Click-based CLI for directory and file comparison."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import click

from xdiff import core
from xdiff.conf import settings
from xdiff.model import CompareMode
from xdiff.model.request import validate_dask_options
from xdiff.printlib import formatter, progress


def _validate_netcdf_file(ctx, param, value: Path | None) -> Path | None:
    """Ensure the explicit files command only accepts netCDF inputs for now."""
    if value is None:
        return value

    if value.suffix.lower() != ".nc":
        raise click.BadParameter("only .nc files are supported by the files command")

    return value


def _render_report(*, progress_enabled: bool, **kwargs) -> None:
    """Execute a comparison request and print the resulting report."""
    try:
        progress_reporter = progress.create_progress_reporter(disabled=not progress_enabled)
        with progress_reporter:
            report = core.execute(progress_reporter=progress_reporter, **kwargs)
        formatter.print_report(report)
        if getattr(report, "has_failures", False):
            raise click.exceptions.Exit(1)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _validate_runtime_options(
    dask_scheduler: str | None,
    dask_scheduler_file: Path | None,
    dask_workers: int | None,
) -> None:
    try:
        validate_dask_options(
            dask_scheduler=dask_scheduler,
            dask_scheduler_file=dask_scheduler_file,
            dask_workers=dask_workers,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc


#
# Common execution options
#
def _execution_options(command):
    command = click.option(
        "--no-progress",
        is_flag=True,
        default=False,
        help="Disable live progress reporting during comparison.",
    )(command)
    command = click.option(
        "-w",
        "--dask-workers",
        type=click.IntRange(min=1),
        metavar="N",
        help="Run comparisons in parallel on a local Dask cluster with N worker processes.",
    )(command)
    command = click.option(
        "--dask-scheduler-file",
        type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
        help="Attach to an existing Dask cluster using a scheduler file.",
    )(command)
    command = click.option(
        "--dask-scheduler",
        help="Attach to an existing Dask cluster using its scheduler address.",
    )(command)
    return command


def _bbox_option(command):
    return click.option(
        "--bbox",
        nargs=4,
        type=float,
        default=None,
        metavar="LON_MIN LON_MAX LAT_MIN LAT_MAX",
        help=(
            "Crop both inputs to a lon/lat box before comparing (for same-grid inputs "
            "of different extent). Example: --bbox -6 36 30 46."
        ),
    )(command)


#
# xdiff entry point
#
@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option(
    version=importlib.metadata.version("xdiffly"),
    prog_name="xdiff",
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Explore differences between datasets."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


#
# xdiff dirs
#
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
    help=(
        "Variables to compare. Repeat the option to compare multiple variables. "
        "Use REF=CMP to compare differently-named variables (e.g. thetao=votemper)."
    ),
)
@click.option(
    "--last-time-step",
    is_flag=True,
    default=False,
    help="If enabled, compare only the last time step available in each file.",
)
@_bbox_option
@_execution_options
def compare_directories(
    reference_path: Path,
    comparison_path: Path,
    filter_name: str,
    common_pattern: str | None,
    variables: tuple[str, ...],
    last_time_step: bool,
    bbox: tuple[float, float, float, float] | None,
    dask_scheduler: str | None,
    dask_scheduler_file: Path | None,
    dask_workers: int | None,
    no_progress: bool,
) -> None:
    """Compare two directories of datasets."""
    _validate_runtime_options(dask_scheduler, dask_scheduler_file, dask_workers)
    _render_report(
        progress_enabled=not no_progress,
        reference_path=reference_path,
        comparison_path=comparison_path,
        input_mode=CompareMode.DIRECTORIES,
        filter_name=filter_name,
        common_pattern=common_pattern,
        variables=variables or settings.DEFAULT_VARIABLES_TO_CHECK,
        last_time_step=last_time_step,
        bbox=bbox,
        dask_scheduler=dask_scheduler,
        dask_scheduler_file=dask_scheduler_file,
        dask_workers=dask_workers,
    )


#
# xdiff files
#
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
    help=(
        "Variables to compare. Repeat the option to compare multiple variables. "
        "Use REF=CMP to compare differently-named variables (e.g. thetao=votemper)."
    ),
)
@click.option(
    "--last-time-step",
    is_flag=True,
    default=False,
    help="If enabled, compare only the last time step available in each file.",
)
@_bbox_option
@_execution_options
def compare_files(
    reference_path: Path,
    comparison_path: Path,
    variables: tuple[str, ...],
    last_time_step: bool,
    bbox: tuple[float, float, float, float] | None,
    dask_scheduler: str | None,
    dask_scheduler_file: Path | None,
    dask_workers: int | None,
    no_progress: bool,
) -> None:
    """Compare two dataset files directly, even if their filenames differ."""
    _validate_runtime_options(dask_scheduler, dask_scheduler_file, dask_workers)
    _render_report(
        progress_enabled=not no_progress,
        reference_path=reference_path,
        comparison_path=comparison_path,
        input_mode=CompareMode.FILES,
        filter_name=settings.DEFAULT_NAME_TO_COMPARE,
        common_pattern=settings.DEFAULT_COMMON_PATTERN,
        variables=variables or settings.DEFAULT_VARIABLES_TO_CHECK,
        last_time_step=last_time_step,
        bbox=bbox,
        dask_scheduler=dask_scheduler,
        dask_scheduler_file=dask_scheduler_file,
        dask_workers=dask_workers,
    )


if __name__ == "__main__":
    cli()
