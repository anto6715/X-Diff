from pathlib import Path

from xdiff.model import Artifact, CompareMode, CompareRequest, CompareResult, ComparisonReport
from xdiff.model.comparison import Comparison


def make_request():
    return CompareRequest(
        input_mode=CompareMode.DIRECTORIES,
        reference_path=Path("ref"),
        comparison_path=Path("cmp"),
        filter_name="*.nc",
        common_pattern=None,
        variables=None,
    )


def make_comparison(*results, exception=None):
    return Comparison(
        reference_artifact=Artifact.from_path(Path("ref/a.nc"), root=Path("ref")),
        comparison_artifact=Artifact.from_path(Path("cmp/a.nc"), root=Path("cmp")),
        compare_results=list(results),
        exception=exception,
    )


def test_comparison_requires_results_and_no_errors_to_pass():
    passing_result = CompareResult(
        relative_error=0.0,
        min_diff=0.0,
        max_diff=0.0,
        mask_equal=True,
        variable="temp",
    )

    assert make_comparison(passing_result).passed is True
    assert make_comparison().passed is False
    assert make_comparison(passing_result, exception=RuntimeError("boom")).passed is False


def test_report_tracks_passed_and_failed_comparisons():
    passing_result = CompareResult(
        relative_error=0.0,
        min_diff=0.0,
        max_diff=0.0,
        mask_equal=True,
        variable="temp",
    )
    failing_result = CompareResult(variable="temp", description="mismatch")

    report = ComparisonReport(
        request=make_request(),
        comparisons=[
            make_comparison(passing_result),
            make_comparison(failing_result),
        ],
    )

    assert report.passed is False
    assert report.has_failures is True
    assert report.passed_count == 1
    assert report.failed_count == 1
