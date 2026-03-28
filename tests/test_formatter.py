from pathlib import Path

from xdiff.model.artifact import Artifact
from xdiff.model.comparison import Comparison
from xdiff.printlib import formatter


def test_print_report_renders_each_comparison(monkeypatch):
    rendered = []
    report = ["first", "second"]

    monkeypatch.setattr(formatter, "print_comparison", rendered.append)

    formatter.print_report(report)

    assert rendered == report


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
