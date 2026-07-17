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


#
# xdiff plot
#
@cli.command("plot")
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
        "Variables to plot. Repeat the option to plot multiple variables. "
        "Use REF=CMP to plot differently-named variables (e.g. thetao=votemper)."
    ),
)
@click.option(
    "--last-time-step",
    is_flag=True,
    default=False,
    help="If enabled, plot only the last time step available in each file.",
)
@_bbox_option
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    metavar="FILE",
    help="Write a static image; the extension picks the format (.png/.pdf/.svg). Omit for the live server.",
)
@click.option(
    "--port",
    type=click.IntRange(min=1, max=65535),
    default=None,
    metavar="N",
    # Keep the quoted default in sync with server.DEFAULT_PORT (not imported here to keep
    # CLI startup free of the plotting stack).
    help="Port for the live interactive server (default: 5006). Ignored with -o.",
)
@click.option(
    "--no-open",
    is_flag=True,
    default=False,
    help="Do not auto-open the browser; just print the URL (server mode, e.g. for headless/SSH).",
)
def plot(
    reference_path: Path,
    comparison_path: Path,
    variables: tuple[str, ...],
    last_time_step: bool,
    bbox: tuple[float, float, float, float] | None,
    output: Path | None,
    port: int | None,
    no_open: bool,
) -> None:
    """Plot where two netCDF files differ, focusing on the difference field.

    With -o, render the difference (one full-size image per variable) and exit. Without
    -o, start a live interactive server on localhost — each difference shown large with a
    colour-limit slider, reference/comparison behind a collapsed card — and block until
    Ctrl-C.
    """
    # Lazy imports keep CLI startup fast: the plotting stack loads only for `plot`.
    from xdiff.core.main import normalize_bbox

    static = output is not None
    address = "localhost"

    if static:
        _plot_static(reference_path, comparison_path, variables, output, last_time_step, bbox)
        return

    from xdiff.plotting.renderers.server import DEFAULT_PORT, ensure_port_available, serve
    from xdiff.plotting.spec import open_plot_source

    server_port = port if port is not None else DEFAULT_PORT
    try:
        # Fail fast on a busy port before opening any datasets.
        ensure_port_available(address, server_port)
        # The live server lets you browse every variable, so open them all; -v only picks
        # which one is shown first (see _default_variable_index).
        source = open_plot_source(
            reference_path,
            comparison_path,
            None,
            last_time_step=last_time_step,
            bbox=normalize_bbox(bbox),
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        for skipped in source.skipped:
            click.echo(f"skipped {skipped.label}: {skipped.reason}", err=True)
        if not source.variables:
            raise click.ClickException("no plottable variables found")

        default_index = _default_variable_index(source, variables)
        url = f"http://{address}:{server_port}"
        click.echo(f"Serving xdiff plot at {url} (Ctrl-C to stop)")
        if no_open:
            click.echo("--no-open: browser not launched; open the URL above (e.g. over an ssh -L tunnel).")
        serve(source, port=server_port, open_browser=not no_open, address=address, default_index=default_index)
    finally:
        source.close()


def _plot_static(reference_path, comparison_path, variables, output, last_time_step, bbox) -> None:
    """Render the difference to a static image (one per requested variable) and exit."""
    from xdiff.core.main import normalize_bbox, normalize_variables
    from xdiff.plotting.renderers.matplotlib_renderer import render_to_files, validate_output_extension
    from xdiff.plotting.spec import build_plot_spec

    try:
        validate_output_extension(output)
        # Static writes one image per requested variable, so keep the -v filter.
        spec = build_plot_spec(
            reference_path,
            comparison_path,
            normalize_variables(variables or settings.DEFAULT_VARIABLES_TO_CHECK),
            last_time_step=last_time_step,
            bbox=normalize_bbox(bbox),
        )
        for skipped in spec.skipped:
            click.echo(f"skipped {skipped.label}: {skipped.reason}", err=True)
        written = render_to_files(spec, output)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    for path in written:
        click.echo(f"wrote {path}")


def _default_variable_index(source, requested: tuple[str, ...]) -> int:
    """Index of the first ``-v`` requested variable in ``source`` (the default shown), else 0.

    All variables stay available in the live server's selector; this only sets which one is
    shown first. Matches the base variable name, ignoring any ``REF=CMP`` mapping.
    """
    if not requested:
        return 0
    wanted_names = {name.split("=")[0] for name in requested}
    for index, handle in enumerate(source.variables):
        if handle.label.split(" -> ")[0] in wanted_names:
            return index
    return 0


if __name__ == "__main__":
    cli()
