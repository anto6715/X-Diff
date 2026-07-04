from pathlib import Path

from xdiff.model.artifact import Artifact
from xdiff.model.compare_result import CompareResult
from xdiff.model.comparison import Comparison
from xdiff.model.report import ComparisonReport
from xdiff.printlib import formatter


def test_print_report_renders_summary_failed_table_and_details(monkeypatch):
    printed = []
    rendered = []

    class FakeConsole:
        def print(self, value):
            printed.append(value)

    passing = Comparison(
        reference_artifact=Artifact.from_path(Path("ok-ref.nc")),
        comparison_artifact=Artifact.from_path(Path("ok-cmp.nc")),
        compare_results=[
            CompareResult(
                relative_error=0.0,
                min_diff=0.0,
                max_diff=0.0,
                mask_equal=True,
                variable="temp",
            )
        ],
    )
    failing = Comparison(
        reference_artifact=Artifact.from_path(Path("bad-ref.nc")),
        comparison_artifact=Artifact.from_path(Path("bad-cmp.nc")),
        compare_results=[CompareResult(variable="salt", description="mask mismatch")],
    )
    report = ComparisonReport(request=object(), comparisons=[passing, failing])

    monkeypatch.setattr(formatter, "print_comparison", lambda comparison, console=None: rendered.append(comparison))
    monkeypatch.setattr(formatter, "Console", FakeConsole)
    monkeypatch.setattr(formatter, "render_summary", lambda report: "SUMMARY")
    monkeypatch.setattr(formatter, "render_failed_comparisons_table", lambda comparisons: "FAILED TABLE")

    formatter.print_report(report)

    assert printed == [
        "[bold green]Passed Comparison Details[/bold green]",
        "[bold red]Failed Comparison Details[/bold red]",
        "FAILED TABLE",
        "SUMMARY",
    ]
    assert rendered == [passing, failing]


def test_print_comparison_renders_exception_row_in_description_column(monkeypatch):
    captured = {}

    class FakeTable:
        def __init__(self, *args, **kwargs):
            captured["rows"] = []

        def add_column(self, *_args, **_kwargs):
            return None

        def add_row(self, *values):
            captured["rows"].append(values)

    class FakeConsole:
        def print(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(formatter, "Table", FakeTable)
    monkeypatch.setattr(formatter, "Console", FakeConsole)

    comparison = Comparison(
        reference_artifact=Artifact.from_path(Path("a.nc")),
        comparison_artifact=Artifact.from_path(Path("b.nc")),
        exception=RuntimeError("boom"),
    )

    formatter.print_comparison(comparison)

    assert captured["rows"] == [
        (formatter.FAILED, "-", "-", "-", "-", "-", "boom"),
    ]


def test_print_comparison_renders_empty_comparison_as_failure(monkeypatch):
    captured = {}

    class FakeTable:
        def __init__(self, *args, **kwargs):
            captured["rows"] = []

        def add_column(self, *_args, **_kwargs):
            return None

        def add_row(self, *values):
            captured["rows"].append(values)

    class FakeConsole:
        def print(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(formatter, "Table", FakeTable)
    monkeypatch.setattr(formatter, "Console", FakeConsole)

    comparison = Comparison(
        reference_artifact=Artifact.from_path(Path("a.nc")),
        comparison_artifact=Artifact.from_path(Path("b.nc")),
    )

    formatter.print_comparison(comparison)

    assert captured["rows"] == [
        (formatter.FAILED, "-", "-", "-", "-", "-", "No comparable variables were selected"),
    ]


def test_render_failed_comparisons_table_highlights_failed_checks():
    exception = Comparison(
        reference_artifact=Artifact.from_path(Path("missing.nc")),
        comparison_artifact=None,
        exception=RuntimeError("No match found"),
    )
    failed_variables = Comparison(
        reference_artifact=Artifact.from_path(Path("ref.nc")),
        comparison_artifact=Artifact.from_path(Path("cmp.nc")),
        compare_results=[
            CompareResult(variable="temp", description="shape mismatch"),
            CompareResult(
                relative_error=0.0,
                min_diff=0.0,
                max_diff=0.0,
                mask_equal=True,
                variable="salt",
            ),
        ],
    )

    table = formatter.render_failed_comparisons_table([exception, failed_variables])

    assert table.row_count == 2
    assert formatter.comparison_failed_check_count(exception) == 1
    assert formatter.comparison_failed_check_count(failed_variables) == 1
    assert formatter.comparison_failure_type(exception) == "No match"
    assert formatter.comparison_failure_details(failed_variables) == "temp (shape mismatch)"
