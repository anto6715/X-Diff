from pathlib import Path

from xdiff.core.service import ComparisonService
from xdiff.matching import DefaultArtifactMatcher
from xdiff.model import Artifact, ArtifactKind, CompareMode, CompareRequest, CompareResult, ExecutionMode
from xdiff.model.comparison import Comparison


class StaticDiscovery:
    def __init__(self, artifacts_by_directory):
        self.artifacts_by_directory = artifacts_by_directory

    def discover(self, directory, filter_name):
        return self.artifacts_by_directory[directory]


def make_request():
    return CompareRequest(
        input_mode=CompareMode.DIRECTORIES,
        reference_path=Path("ref"),
        comparison_path=Path("cmp"),
        filter_name="*",
        common_pattern=None,
        variables=None,
        last_time_step=False,
    )


def test_default_artifact_matcher_matches_relative_paths():
    matcher = DefaultArtifactMatcher()
    reference = Artifact.from_path(Path("ref/nemo/output/mesh_mask.nc"), root=Path("ref"), kind=ArtifactKind.NETCDF)
    comparison = Artifact.from_path(Path("cmp/nemo/output/mesh_mask.nc"), root=Path("cmp"), kind=ArtifactKind.NETCDF)

    matches = matcher.match([reference], [comparison])

    assert len(matches) == 1
    assert matches[0].reference == reference
    assert matches[0].comparison == comparison


def test_default_artifact_matcher_uses_common_pattern_when_paths_differ():
    matcher = DefaultArtifactMatcher()
    reference = Artifact.from_path(
        Path("ref/my-simu_19820101_grid_T.nc"),
        root=Path("ref"),
        kind=ArtifactKind.NETCDF,
    )
    comparison = Artifact.from_path(
        Path("cmp/another-exp_19820101_grid_T.nc"),
        root=Path("cmp"),
        kind=ArtifactKind.NETCDF,
    )

    matches = matcher.match([reference], [comparison], r"\d{8}_grid_T\.nc")

    assert len(matches) == 1
    assert matches[0].comparison == comparison


def test_comparison_service_records_missing_matches():
    reference_artifact = Artifact.from_path(Path("ref/ocean.nc"), root=Path("ref"), kind=ArtifactKind.NETCDF)
    service = ComparisonService(
        discovery=StaticDiscovery({Path("ref"): [reference_artifact], Path("cmp"): []}),
        matcher=DefaultArtifactMatcher(),
        comparators=[],
    )

    report = service.run(make_request())

    assert len(report) == 1
    assert report.comparisons[0].comparison_file is None
    assert type(report.comparisons[0].exception).__name__ == "NoMatchFound"


def test_comparison_service_reports_unregistered_artifact_types():
    reference_artifact = Artifact.from_path(
        Path("ref/namelist_cfg"),
        root=Path("ref"),
        kind=ArtifactKind.NAMELIST,
    )
    comparison_artifact = Artifact.from_path(
        Path("cmp/namelist_cfg"),
        root=Path("cmp"),
        kind=ArtifactKind.NAMELIST,
    )
    service = ComparisonService(
        discovery=StaticDiscovery(
            {
                Path("ref"): [reference_artifact],
                Path("cmp"): [comparison_artifact],
            }
        ),
        matcher=DefaultArtifactMatcher(),
        comparators=[],
    )

    report = service.run(make_request())

    assert len(report) == 1
    assert type(report.comparisons[0].exception).__name__ == "UnsupportedArtifactTypeError"


def test_comparison_service_bypasses_matching_for_explicit_file_mode():
    request = CompareRequest(
        input_mode=CompareMode.FILES,
        reference_path=Path("reference.nc"),
        comparison_path=Path("different-name.nc"),
        filter_name="*.nc",
        common_pattern=None,
        variables=None,
        last_time_step=False,
    )

    class FailIfCalled:
        def discover(self, directory, filter_name):
            raise AssertionError("discovery should not be used in file mode")

    class FailMatcher:
        def match(self, reference_artifacts, comparison_artifacts, common_pattern):
            raise AssertionError("matcher should not be used in file mode")

    service = ComparisonService(
        discovery=FailIfCalled(),
        matcher=FailMatcher(),
        comparators=[],
    )

    report = service.run(request)

    assert len(report) == 1
    assert report.comparisons[0].reference_file == Path("reference.nc")
    assert report.comparisons[0].comparison_file == Path("different-name.nc")


def test_comparison_service_reports_progress_during_serial_execution():
    reference_a = Artifact.from_path(Path("ref/a.nc"), root=Path("ref"), kind=ArtifactKind.NETCDF)
    reference_b = Artifact.from_path(Path("ref/b.nc"), root=Path("ref"), kind=ArtifactKind.NETCDF)
    comparison_a = Artifact.from_path(Path("cmp/a.nc"), root=Path("cmp"), kind=ArtifactKind.NETCDF)
    events = []

    class FakeComparator:
        artifact_kind = ArtifactKind.NETCDF

        def compare(self, match, request):
            comparison = Comparison(reference_artifact=match.reference, comparison_artifact=match.comparison)
            comparison.append(
                CompareResult(
                    relative_error=0.0,
                    min_diff=0.0,
                    max_diff=0.0,
                    mask_equal=True,
                    variable=match.reference.path.name,
                )
            )
            return comparison

    class RecordingReporter:
        def start(self, request):
            events.append(("start", request.input_mode))

        def on_discovery_complete(self, reference_count, comparison_count):
            events.append(("discovery", reference_count, comparison_count))

        def on_matching_complete(self, total_matches):
            events.append(("matching", total_matches))

        def on_comparisons_started(self, total_matches):
            events.append(("comparisons_started", total_matches))

        def on_comparison_complete(self, comparison, completed, total_matches):
            events.append(("comparison_complete", comparison.reference_file.name, completed, total_matches))

        def finish(self, report):
            events.append(("finish", len(report), report.passed_count, report.failed_count))

    service = ComparisonService(
        discovery=StaticDiscovery(
            {
                Path("ref"): [reference_a, reference_b],
                Path("cmp"): [comparison_a],
            }
        ),
        matcher=DefaultArtifactMatcher(),
        comparators=[FakeComparator()],
    )

    report = service.run(make_request(), progress_reporter=RecordingReporter())

    assert [comparison.reference_file.name for comparison in report.comparisons] == ["a.nc", "b.nc"]
    assert events == [
        ("start", CompareMode.DIRECTORIES),
        ("discovery", 2, 1),
        ("matching", 2),
        ("comparisons_started", 2),
        ("comparison_complete", "a.nc", 1, 2),
        ("comparison_complete", "b.nc", 2, 2),
        ("finish", 2, 1, 1),
    ]


def test_comparison_service_runs_file_pairs_through_dask(monkeypatch):
    reference_a = Artifact.from_path(Path("ref/a.nc"), root=Path("ref"), kind=ArtifactKind.NETCDF)
    comparison_a = Artifact.from_path(Path("cmp/a.nc"), root=Path("cmp"), kind=ArtifactKind.NETCDF)
    reference_b = Artifact.from_path(Path("ref/b.nc"), root=Path("ref"), kind=ArtifactKind.NETCDF)
    comparison_b = Artifact.from_path(Path("cmp/b.nc"), root=Path("cmp"), kind=ArtifactKind.NETCDF)

    class FakeComparator:
        artifact_kind = ArtifactKind.NETCDF

        def compare(self, match, request):
            comparison = Comparison(reference_artifact=match.reference, comparison_artifact=match.comparison)
            comparison.append(CompareResult(variable=match.reference.path.name))
            return comparison

    class FakeFuture:
        def __init__(self, result):
            self.result = result

    class FakeClient:
        def __init__(self):
            self.submissions = []

        def submit(self, func, *args, **kwargs):
            self.submissions.append((func, args, kwargs))
            return FakeFuture(func(*args))

    fake_client = FakeClient()

    class FakeContextManager:
        def __enter__(self):
            return fake_client

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("xdiff.core.service.dask_runtime.client_from_request", lambda request: FakeContextManager())
    monkeypatch.setattr(
        "xdiff.core.service.dask_runtime.iterate_results_as_completed",
        lambda futures: ((future, future.result) for future in futures),
    )

    request = CompareRequest(
        input_mode=CompareMode.DIRECTORIES,
        reference_path=Path("ref"),
        comparison_path=Path("cmp"),
        filter_name="*.nc",
        common_pattern=None,
        variables=None,
        last_time_step=False,
        execution_mode=ExecutionMode.FILES,
        dask_workers=2,
    )

    service = ComparisonService(
        discovery=StaticDiscovery({Path("ref"): [reference_a, reference_b], Path("cmp"): [comparison_a, comparison_b]}),
        matcher=DefaultArtifactMatcher(),
        comparators=[FakeComparator()],
    )

    report = service.run(request)

    assert len(fake_client.submissions) == 2
    assert [comparison.reference_file.name for comparison in report.comparisons] == ["a.nc", "b.nc"]


def test_comparison_service_reports_dask_progress_in_completion_order(monkeypatch):
    reference_a = Artifact.from_path(Path("ref/a.nc"), root=Path("ref"), kind=ArtifactKind.NETCDF)
    comparison_a = Artifact.from_path(Path("cmp/a.nc"), root=Path("cmp"), kind=ArtifactKind.NETCDF)
    reference_b = Artifact.from_path(Path("ref/b.nc"), root=Path("ref"), kind=ArtifactKind.NETCDF)
    comparison_b = Artifact.from_path(Path("cmp/b.nc"), root=Path("cmp"), kind=ArtifactKind.NETCDF)
    events = []

    class FakeComparator:
        artifact_kind = ArtifactKind.NETCDF

        def compare(self, match, request):
            comparison = Comparison(reference_artifact=match.reference, comparison_artifact=match.comparison)
            comparison.append(CompareResult(variable=match.reference.path.name))
            return comparison

    class FakeFuture:
        def __init__(self, result):
            self.result = result

    class FakeClient:
        def submit(self, func, *args, **kwargs):
            return FakeFuture(func(*args))

    class FakeContextManager:
        def __enter__(self):
            return FakeClient()

        def __exit__(self, exc_type, exc, tb):
            return False

    class RecordingReporter:
        def start(self, request):
            events.append(("start", request.execution_mode))

        def on_discovery_complete(self, reference_count, comparison_count):
            events.append(("discovery", reference_count, comparison_count))

        def on_matching_complete(self, total_matches):
            events.append(("matching", total_matches))

        def on_comparisons_started(self, total_matches):
            events.append(("comparisons_started", total_matches))

        def on_comparison_complete(self, comparison, completed, total_matches):
            events.append(("comparison_complete", comparison.reference_file.name, completed, total_matches))

        def finish(self, report):
            events.append(("finish", len(report)))

    monkeypatch.setattr("xdiff.core.service.dask_runtime.client_from_request", lambda request: FakeContextManager())
    monkeypatch.setattr(
        "xdiff.core.service.dask_runtime.iterate_results_as_completed",
        lambda futures: ((future, future.result) for future in reversed(futures)),
    )

    request = CompareRequest(
        input_mode=CompareMode.DIRECTORIES,
        reference_path=Path("ref"),
        comparison_path=Path("cmp"),
        filter_name="*.nc",
        common_pattern=None,
        variables=None,
        last_time_step=False,
        execution_mode=ExecutionMode.FILES,
        dask_workers=2,
    )

    service = ComparisonService(
        discovery=StaticDiscovery({Path("ref"): [reference_a, reference_b], Path("cmp"): [comparison_a, comparison_b]}),
        matcher=DefaultArtifactMatcher(),
        comparators=[FakeComparator()],
    )

    report = service.run(request, progress_reporter=RecordingReporter())

    assert [comparison.reference_file.name for comparison in report.comparisons] == ["a.nc", "b.nc"]
    assert events == [
        ("start", ExecutionMode.FILES),
        ("discovery", 2, 2),
        ("matching", 2),
        ("comparisons_started", 2),
        ("comparison_complete", "b.nc", 1, 2),
        ("comparison_complete", "a.nc", 2, 2),
        ("finish", 2),
    ]


def test_comparison_service_warns_when_more_workers_than_file_pairs(monkeypatch, caplog):
    reference = Artifact.from_path(Path("ref/a.nc"), root=Path("ref"), kind=ArtifactKind.NETCDF)
    comparison = Artifact.from_path(Path("cmp/a.nc"), root=Path("cmp"), kind=ArtifactKind.NETCDF)

    class FakeComparator:
        artifact_kind = ArtifactKind.NETCDF

        def compare(self, match, request):
            return Comparison(reference_artifact=match.reference, comparison_artifact=match.comparison)

    class FakeFuture:
        def __init__(self, result):
            self.result = result

    class FakeClient:
        def submit(self, func, *args, **kwargs):
            return FakeFuture(func(*args))

    class FakeContextManager:
        def __enter__(self):
            return FakeClient()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("xdiff.core.service.dask_runtime.client_from_request", lambda request: FakeContextManager())
    monkeypatch.setattr(
        "xdiff.core.service.dask_runtime.iterate_results_as_completed",
        lambda futures: ((future, future.result) for future in futures),
    )

    request = CompareRequest(
        input_mode=CompareMode.DIRECTORIES,
        reference_path=Path("ref"),
        comparison_path=Path("cmp"),
        filter_name="*.nc",
        common_pattern=None,
        variables=None,
        last_time_step=False,
        execution_mode=ExecutionMode.FILES,
        dask_workers=4,
    )

    service = ComparisonService(
        discovery=StaticDiscovery({Path("ref"): [reference], Path("cmp"): [comparison]}),
        matcher=DefaultArtifactMatcher(),
        comparators=[FakeComparator()],
    )

    with caplog.at_level("WARNING", logger="xdiff"):
        service.run(request)

    assert "some workers will remain idle" in caplog.text
