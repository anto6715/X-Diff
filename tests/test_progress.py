from io import StringIO
from pathlib import Path

from xdiff.model import CompareResult, ComparisonReport
from xdiff.model.artifact import Artifact
from xdiff.model.comparison import Comparison
from xdiff.printlib import progress


class TtyStream(StringIO):
    def isatty(self):
        return True


class NonTtyStream(StringIO):
    def isatty(self):
        return False


def test_create_progress_reporter_uses_rich_for_interactive_stream():
    reporter = progress.create_progress_reporter(stream=TtyStream())

    assert isinstance(reporter, progress.RichProgressReporter)


def test_create_progress_reporter_uses_text_for_non_interactive_stream():
    reporter = progress.create_progress_reporter(stream=NonTtyStream())

    assert isinstance(reporter, progress.TextProgressReporter)


def test_create_progress_reporter_returns_null_when_disabled():
    reporter = progress.create_progress_reporter(disabled=True, stream=TtyStream())

    assert isinstance(reporter, progress.NullProgressReporter)


def test_text_progress_reporter_emits_checkpoints_and_summary():
    stream = NonTtyStream()
    reporter = progress.TextProgressReporter(stream=stream)
    comparison = Comparison(
        reference_artifact=Artifact.from_path(Path("a.nc")),
        comparison_artifact=Artifact.from_path(Path("b.nc")),
    )
    comparison.append(
        CompareResult(
            relative_error=0.0,
            min_diff=0.0,
            max_diff=0.0,
            mask_equal=True,
            variable="temp",
        )
    )
    failing = Comparison(
        reference_artifact=Artifact.from_path(Path("c.nc")),
        comparison_artifact=Artifact.from_path(Path("d.nc")),
    )
    failing.append(CompareResult(variable="salt", description="mask mismatch"))

    reporter.on_discovery_complete(2, 2)
    reporter.on_matching_complete(2)
    reporter.on_comparisons_started(2)
    reporter.on_comparison_complete(comparison, 1, 2)
    reporter.on_comparison_complete(failing, 2, 2)
    reporter.finish(ComparisonReport(request=object(), comparisons=[comparison, failing]))

    output = stream.getvalue()

    assert "Reference files discovered: 2" in output
    assert "Comparison files discovered: 2" in output
    assert "Comparison tasks scheduled: 2" in output
    assert "PASSED a.nc vs b.nc (1 check passed)" in output
    assert "FAILED c.nc vs d.nc (1/1 failed: salt (mask mismatch))" in output
    assert "Progress: 1/2 comparison tasks completed (50%, 1 left)" in output
    assert "Progress: 2/2 comparison tasks completed (100%, 0 left)" in output
    assert "Completed 2/2 comparison tasks" in output


def test_rich_progress_reporter_updates_progress_and_prints_summary(monkeypatch):
    captured = {}

    class FakeConsole:
        def print(self, message):
            captured.setdefault("prints", []).append(message)

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            captured["progress_kwargs"] = kwargs
            captured["updates"] = []
            self.console = kwargs["console"]

        def start(self):
            captured["started"] = True

        def add_task(self, description, total, remaining):
            captured["task"] = (description, total, remaining)
            return 7

        def update(self, task_id, **kwargs):
            captured["updates"].append((task_id, kwargs))

        def stop(self):
            captured["stopped"] = True

    monkeypatch.setattr(progress, "Progress", FakeProgress)

    reporter = progress.RichProgressReporter(console=FakeConsole())
    comparison = Comparison(
        reference_artifact=Artifact.from_path(Path("a.nc")),
        comparison_artifact=Artifact.from_path(Path("b.nc")),
    )
    comparison.append(
        CompareResult(
            relative_error=0.0,
            min_diff=0.0,
            max_diff=0.0,
            mask_equal=True,
            variable="temp",
        )
    )

    reporter.on_matching_complete(3)
    reporter.on_comparisons_started(3)
    reporter.on_comparison_complete(comparison, 1, 3)
    reporter.finish(ComparisonReport(request=object(), comparisons=[comparison]))

    assert captured["task"] == ("Comparing file pairs", 3, 3)
    assert captured["updates"] == [
        (7, {"completed": 1, "remaining": 2}),
        (7, {"completed": 3, "remaining": 0}),
    ]
    assert captured["started"] is True
    assert captured["stopped"] is True
    assert "Comparison tasks scheduled" in captured["prints"][0]
    assert "PASSED" in captured["prints"][1]
    assert "a.nc vs b.nc" in captured["prints"][1]
    assert "1 check passed" in captured["prints"][1]
    assert "Completed 1/3 comparison tasks" in captured["prints"][2]
