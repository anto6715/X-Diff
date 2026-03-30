"""Progress reporting helpers for long-running comparison runs."""

from __future__ import annotations

import math
import sys
import time

from typing import TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from xdiff.model.comparison import Comparison
from xdiff.model.report import ComparisonReport
from xdiff.model.request import CompareRequest


class ProgressReporter:
    """Minimal lifecycle for CLI progress reporting."""

    def __enter__(self) -> "ProgressReporter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def start(self, request: CompareRequest) -> None:
        """Notify the reporter that a run is starting."""

    def on_discovery_complete(self, reference_count: int, comparison_count: int) -> None:
        """Report discovered input counts."""

    def on_matching_complete(self, total_matches: int) -> None:
        """Report how many comparison tasks were scheduled."""

    def on_comparisons_started(self, total_matches: int) -> None:
        """Start live comparison progress for the scheduled tasks."""

    def on_comparison_complete(
        self,
        comparison: Comparison,
        completed: int,
        total_matches: int,
    ) -> None:
        """Advance progress for one completed comparison."""

    def finish(self, report: ComparisonReport) -> None:
        """Report the final summary for the run."""

    def close(self) -> None:
        """Release any live-rendering resources."""


class NullProgressReporter(ProgressReporter):
    """No-op reporter used for library calls and explicit opt-out."""


class RichProgressReporter(ProgressReporter):
    """Render comparison progress with a Rich progress bar."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console(file=sys.stderr)
        self._progress: Progress | None = None
        self._task_id: int | None = None
        self._started_at: float | None = None
        self._total_matches = 0

    def on_discovery_complete(self, reference_count: int, comparison_count: int) -> None:
        self.console.print(f"[bold]Reference files discovered:[/bold] {reference_count}")
        self.console.print(f"[bold]Comparison files discovered:[/bold] {comparison_count}")

    def on_matching_complete(self, total_matches: int) -> None:
        self.console.print(f"[bold]Comparison tasks scheduled:[/bold] {total_matches}")

    def on_comparisons_started(self, total_matches: int) -> None:
        self._total_matches = total_matches
        if total_matches == 0:
            self.console.print("[yellow]No comparison tasks were scheduled.[/yellow]")
            return

        self._started_at = time.monotonic()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("{task.fields[remaining]} left"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            "Comparing file pairs",
            total=total_matches,
            remaining=total_matches,
        )

    def on_comparison_complete(
        self,
        comparison: Comparison,
        completed: int,
        total_matches: int,
    ) -> None:
        del comparison
        if self._progress is None or self._task_id is None:
            return

        self._progress.update(
            self._task_id,
            completed=completed,
            remaining=max(total_matches - completed, 0),
        )

    def finish(self, report: ComparisonReport) -> None:
        self._stop_progress()
        if self._total_matches == 0:
            return

        self.console.print(_format_summary(report, self._total_matches, self._started_at))

    def close(self) -> None:
        self._stop_progress()

    def _stop_progress(self) -> None:
        if self._progress is None:
            return

        if self._task_id is not None:
            self._progress.update(self._task_id, completed=self._total_matches, remaining=0)
        self._progress.stop()
        self._progress = None
        self._task_id = None


class TextProgressReporter(ProgressReporter):
    """Render static checkpoints for non-interactive output."""

    def __init__(self, stream: TextIO | None = None):
        self.stream = stream or sys.stderr
        self._started_at: float | None = None
        self._total_matches = 0
        self._checkpoints: set[int] = set()
        self._last_reported = 0

    def on_discovery_complete(self, reference_count: int, comparison_count: int) -> None:
        self._write(f"Reference files discovered: {reference_count}")
        self._write(f"Comparison files discovered: {comparison_count}")

    def on_matching_complete(self, total_matches: int) -> None:
        self._write(f"Comparison tasks scheduled: {total_matches}")

    def on_comparisons_started(self, total_matches: int) -> None:
        self._total_matches = total_matches
        self._started_at = time.monotonic()
        self._checkpoints = build_progress_checkpoints(total_matches)
        self._last_reported = 0

        if total_matches == 0:
            self._write("No comparison tasks were scheduled.")
            return

        self._write("Comparing file pairs...")

    def on_comparison_complete(
        self,
        comparison: Comparison,
        completed: int,
        total_matches: int,
    ) -> None:
        del comparison
        if completed not in self._checkpoints or completed == self._last_reported:
            return

        remaining = max(total_matches - completed, 0)
        percentage = int((completed / total_matches) * 100)
        self._write(
            f"Progress: {completed}/{total_matches} comparison tasks completed "
            f"({percentage}%, {remaining} left)"
        )
        self._last_reported = completed

    def finish(self, report: ComparisonReport) -> None:
        if self._total_matches == 0:
            return

        self._write(_format_summary(report, self._total_matches, self._started_at, rich_markup=False))

    def close(self) -> None:
        return None

    def _write(self, message: str) -> None:
        print(message, file=self.stream)


def create_progress_reporter(
    *,
    disabled: bool = False,
    stream: TextIO | None = None,
) -> ProgressReporter:
    """Create the appropriate progress reporter for the current stderr stream."""
    if disabled:
        return NullProgressReporter()

    output_stream = stream or sys.stderr
    if is_interactive_stream(output_stream):
        return RichProgressReporter(console=Console(file=output_stream))
    return TextProgressReporter(stream=output_stream)


def build_progress_checkpoints(total_matches: int) -> set[int]:
    """Return progress checkpoints for the text reporter."""
    if total_matches <= 0:
        return set()

    checkpoints = {1, total_matches}
    checkpoints.update(math.ceil(total_matches * fraction / 10) for fraction in range(1, 10))
    return checkpoints


def is_interactive_stream(stream: TextIO) -> bool:
    """Return True when the output stream is attached to a terminal."""
    isatty = getattr(stream, "isatty", None)
    if isatty is None:
        return False
    return bool(isatty())


def _format_summary(
    report: ComparisonReport,
    total_matches: int,
    started_at: float | None,
    *,
    rich_markup: bool = True,
) -> str:
    duration = 0.0 if started_at is None else max(time.monotonic() - started_at, 0.0)
    summary = (
        f"Completed {len(report)}/{total_matches} comparison tasks in {duration:.1f}s: "
        f"{report.passed_count} passed, {report.failed_count} failed."
    )
    if not rich_markup:
        return summary
    return f"[bold]{summary}[/bold]"
