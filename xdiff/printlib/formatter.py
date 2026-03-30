"""Rich-based rendering for reports and comparisons."""

from typing import Any

import numpy as np

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from xdiff.model import CompareResult, ComparisonReport
from xdiff.model.comparison import Comparison

COLUMNS = ["RESULT", "MIN DIFF", "MAX DIFF", "REL ERR", "MASK", "VAR", "DESCR"]
FAILED_COMPARISON_COLUMNS = ["REFERENCE", "COMPARISON", "FAILED", "TYPE", "DETAILS"]

FAILED = "[red] :x: FAILED"
PASSED = "[green] :heavy_check_mark: PASSED"


def get_result(cr: CompareResult) -> str:
    if cr.passed:
        return PASSED

    return FAILED


def print_report(report: ComparisonReport) -> None:
    """Render a full report to the console."""
    console = Console()
    comparisons = list(report)
    failed_comparisons = [comparison for comparison in comparisons if not comparison.passed]
    passed_comparisons = [comparison for comparison in comparisons if comparison.passed]

    console.print(render_summary(report))
    if failed_comparisons:
        console.print(render_failed_comparisons_table(failed_comparisons))

    if failed_comparisons:
        console.print("[bold red]Failed Comparison Details[/bold red]")
        for comparison in failed_comparisons:
            print_comparison(comparison, console=console)

    if passed_comparisons:
        heading = "[bold green]Passed Comparison Details[/bold green]"
        if failed_comparisons:
            console.print(heading)
        for comparison in passed_comparisons:
            print_comparison(comparison, console=console)


def print_comparison(comparison: Comparison, console: Console | None = None) -> None:
    console = console or Console()

    table = Table(
        show_header=True,
        header_style="bold blue",
        title=build_comparison_title(comparison),
        box=box.SIMPLE,
    )
    for column in COLUMNS:
        table.add_column(column)

    if comparison.exception is not None:
        table.add_row(FAILED, "-", "-", "-", "-", "-", str(comparison.exception))
    elif len(comparison) == 0:
        table.add_row(FAILED, "-", "-", "-", "-", "-", "No comparable variables were selected")
    else:
        for c in comparison:
            result = get_result(c)
            table.add_row(
                f"{result}",
                render(c.min_diff),
                render(c.max_diff),
                render(c.relative_error),
                render(c.mask_equal),
                render(c.variable),
                render(c.description),
            )

    console.print(table)


def render_summary(report: ComparisonReport) -> Panel:
    """Render the top-level run summary."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()

    total_comparisons = len(report)
    failed_comparisons = report.failed_count
    total_checks = sum(comparison_total_check_count(comparison) for comparison in report)
    failed_checks = sum(comparison_failed_check_count(comparison) for comparison in report)
    passed_checks = total_checks - failed_checks
    status = "[bold green]PASSED[/bold green]" if failed_comparisons == 0 else "[bold red]FAILED[/bold red]"

    grid.add_row("Status", status)
    grid.add_row("Compared pairs", str(total_comparisons))
    grid.add_row(
        "Comparison results",
        f"[green]{report.passed_count} passed[/green], [red]{failed_comparisons} failed[/red]",
    )
    grid.add_row(
        "Checks",
        f"[green]{passed_checks} passed[/green], [red]{failed_checks} failed[/red] ({total_checks} total)",
    )

    return Panel.fit(grid, title="Comparison Summary", border_style="cyan")


def render_failed_comparisons_table(failed_comparisons: list[Comparison]) -> Table:
    """Render a compact table that highlights which comparisons failed."""
    table = Table(
        show_header=True,
        header_style="bold red",
        title="Failed Comparisons",
        box=box.SIMPLE,
    )
    for column in FAILED_COMPARISON_COLUMNS:
        table.add_column(column)

    for comparison in failed_comparisons:
        table.add_row(
            str(comparison.reference_file),
            str(comparison.comparison_file or "-"),
            f"{comparison_failed_check_count(comparison)}/{comparison_total_check_count(comparison)}",
            comparison_failure_type(comparison),
            comparison_failure_details(comparison),
        )

    return table


def build_comparison_title(comparison: Comparison) -> str:
    """Return a status-rich title for one comparison table."""
    if comparison.passed:
        check_summary = f"{comparison_total_check_count(comparison)} checks passed"
    else:
        check_summary = (
            f"{comparison_failed_check_count(comparison)}/{comparison_total_check_count(comparison)} failed checks"
        )

    return (
        f"{comparison_status_label(comparison)} {comparison.reference_file} vs {comparison.comparison_file or '-'} "
        f"[dim]({check_summary})[/dim]"
    )


def comparison_status_label(comparison: Comparison) -> str:
    """Return the status label used in comparison titles."""
    if comparison.passed:
        return "[green]PASSED[/green]"
    if comparison.exception is not None and comparison.comparison_file is None:
        return "[red]NO MATCH[/red]"
    if comparison.exception is not None:
        return "[red]ERROR[/red]"
    if len(comparison) == 0:
        return "[red]NO CHECKS[/red]"
    return "[red]FAILED[/red]"


def comparison_total_check_count(comparison: Comparison) -> int:
    """Count the checks represented by one comparison."""
    if comparison.exception is not None or len(comparison) == 0:
        return 1
    return len(comparison)


def comparison_failed_check_count(comparison: Comparison) -> int:
    """Count failed checks for one comparison."""
    if comparison.exception is not None or len(comparison) == 0:
        return 1
    return sum(not result.passed for result in comparison)


def comparison_failure_type(comparison: Comparison) -> str:
    """Return a short failure classification."""
    if comparison.exception is not None and comparison.comparison_file is None:
        return "No match"
    if comparison.exception is not None:
        return "Error"
    if len(comparison) == 0:
        return "No checks"
    return "Variable differences"


def comparison_failure_details(comparison: Comparison) -> str:
    """Return a compact failure summary for one comparison."""
    if comparison.exception is not None:
        return str(comparison.exception)
    if len(comparison) == 0:
        return "No comparable variables were selected"

    failed_results = [summarize_failed_result(result) for result in comparison if not result.passed]
    if len(failed_results) <= 3:
        return ", ".join(failed_results)

    remaining = len(failed_results) - 3
    return f"{', '.join(failed_results[:3])} (+{remaining} more)"


def summarize_failed_result(result: CompareResult) -> str:
    """Return a compact label for one failed variable check."""
    if result.description not in {"", "-"}:
        return f"{result.variable} ({result.description})"
    return str(result.variable)


def render(value: Any):
    if isinstance(value, bool):
        return str(value)

    if isinstance(value, np.timedelta64):
        return f"{value.view('int64'):.2e}"
    try:
        return f"{value:.2e}"
    except Exception:
        return str(value)
