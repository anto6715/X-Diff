from pathlib import Path

import pytest

from xdiff.core.service import ComparisonService, compare_match
from xdiff.discovery import FileSystemArtifactDiscovery
from xdiff.exceptions import AmbiguousMatch, NoMatchFound
from xdiff.matching import DefaultArtifactMatcher
from xdiff.model import Artifact, ArtifactKind, CompareMode, CompareRequest, CompareResult
from xdiff.model.comparison import Comparison
from xdiff.model.match import ArtifactMatch


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
    assert str(report.comparisons[0].exception) == "Only in the reference directory"


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
            events.append(("start", request.uses_dask))

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
        ("start", True),
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


def _netcdf(path: str) -> Artifact:
    return Artifact.from_path(Path(path), root=Path(path.split("/")[0]), kind=ArtifactKind.NETCDF)


def test_default_artifact_matcher_reports_comparison_files_without_a_reference():
    matcher = DefaultArtifactMatcher()
    reference = _netcdf("ref/a.nc")
    paired, extra_b, extra_c = _netcdf("cmp/a.nc"), _netcdf("cmp/b.nc"), _netcdf("cmp/c.nc")

    matches = matcher.match([reference], [paired, extra_b, extra_c])

    # The pairs first, then each comparison file no reference file paired with, in discovery order.
    assert [(match.reference, match.comparison) for match in matches] == [
        (reference, paired),
        (None, extra_b),
        (None, extra_c),
    ]


def test_artifact_match_needs_at_least_one_artifact():
    with pytest.raises(ValueError, match="at least one artifact"):
        ArtifactMatch(reference=None, comparison=None)


def test_artifact_match_rejects_candidates_on_a_pair():
    with pytest.raises(ValueError, match="Only an unpaired artifact"):
        ArtifactMatch(reference=_netcdf("ref/a.nc"), comparison=_netcdf("cmp/a.nc"), candidates=(_netcdf("cmp/b.nc"),))


def _pairs(matches):
    """Each match as (reference path, comparison path, candidate paths), with ``None`` for a missing side."""
    return [
        (
            str(match.reference.path) if match.reference else None,
            str(match.comparison.path) if match.comparison else None,
            [str(candidate.path) for candidate in match.candidates],
        )
        for match in matches
    ]


def test_filesystem_discovery_lists_each_file_once(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "x.nc").touch()

    artifacts = FileSystemArtifactDiscovery().discover(tmp_path, "**/*/**/*.nc")

    # Python 3.13+ globs a/b/x.nc twice with this pattern; the matcher would then report it as ambiguous.
    assert [artifact.relative_path for artifact in artifacts] == [Path("a/b/x.nc")]


def test_default_artifact_matcher_does_not_cross_pair_same_named_files_in_different_subdirectories():
    matcher = DefaultArtifactMatcher()
    references = [_netcdf("ref/x.nc"), _netcdf("ref/sub/x.nc")]
    comparisons = [_netcdf("cmp/x.nc"), _netcdf("cmp/sub/x.nc")]

    matches = matcher.match(references, comparisons)

    # Each file used to pair with both same-named files: 4 pairs instead of 2.
    assert _pairs(matches) == [("ref/x.nc", "cmp/x.nc", []), ("ref/sub/x.nc", "cmp/sub/x.nc", [])]


def test_default_artifact_matcher_reports_a_file_whose_relative_path_is_missing():
    matcher = DefaultArtifactMatcher()

    matches = matcher.match([_netcdf("ref/x.nc"), _netcdf("ref/sub/x.nc")], [_netcdf("cmp/x.nc")])

    # sub/x.nc used to pair with cmp/x.nc by file name, so the run passed when the two
    # files held the same data although the comparison directory has no sub/x.nc.
    assert _pairs(matches) == [("ref/x.nc", "cmp/x.nc", []), ("ref/sub/x.nc", None, [])]


def test_default_artifact_matcher_falls_back_to_the_file_name_among_unpaired_files():
    matcher = DefaultArtifactMatcher()
    references = [_netcdf("ref/a/x.nc"), _netcdf("ref/b/x.nc")]
    comparisons = [_netcdf("cmp/a/x.nc"), _netcdf("cmp/c/x.nc")]

    matches = matcher.match(references, comparisons)

    # a/x.nc pairs by relative path; b/x.nc and c/x.nc are then each other's only
    # unpaired file with that name.
    assert _pairs(matches) == [("ref/a/x.nc", "cmp/a/x.nc", []), ("ref/b/x.nc", "cmp/c/x.nc", [])]


def test_default_artifact_matcher_reports_an_ambiguous_file_name_on_both_sides():
    matcher = DefaultArtifactMatcher()
    references = [_netcdf("ref/x.nc"), _netcdf("ref/y.nc")]
    comparisons = [_netcdf("cmp/a/x.nc"), _netcdf("cmp/y.nc"), _netcdf("cmp/b/x.nc")]

    matches = matcher.match(references, comparisons)

    # Reference files in order, then the unpaired comparison files in order.
    assert _pairs(matches) == [
        ("ref/x.nc", None, ["cmp/a/x.nc", "cmp/b/x.nc"]),
        ("ref/y.nc", "cmp/y.nc", []),
        (None, "cmp/a/x.nc", ["ref/x.nc"]),
        (None, "cmp/b/x.nc", ["ref/x.nc"]),
    ]


def test_default_artifact_matcher_reports_a_comparison_file_matching_several_reference_files():
    matcher = DefaultArtifactMatcher()

    matches = matcher.match([_netcdf("ref/a/x.nc"), _netcdf("ref/b/x.nc")], [_netcdf("cmp/x.nc")])

    # Pairing cmp/x.nc with the first reference file would be arbitrary.
    assert _pairs(matches) == [
        ("ref/a/x.nc", None, ["cmp/x.nc"]),
        ("ref/b/x.nc", None, ["cmp/x.nc"]),
        (None, "cmp/x.nc", ["ref/a/x.nc", "ref/b/x.nc"]),
    ]


def test_default_artifact_matcher_reports_an_ambiguous_common_pattern():
    matcher = DefaultArtifactMatcher()
    references = [_netcdf("ref/exp1_19820101_grid_T.nc"), _netcdf("ref/exp1_19820101_grid_U.nc")]
    comparisons = [_netcdf("cmp/exp2_19820101_grid_T.nc"), _netcdf("cmp/exp2_19820101_grid_U.nc")]

    ambiguous = matcher.match(references, comparisons, r"\d{8}")
    paired = matcher.match(references, comparisons, r"\d{8}_grid_[TU]")

    # The date alone does not tell grid_T from grid_U: each file used to pair with both.
    assert [match.candidates for match in ambiguous] == [tuple(comparisons)] * 2 + [tuple(references)] * 2
    assert _pairs(paired) == [
        ("ref/exp1_19820101_grid_T.nc", "cmp/exp2_19820101_grid_T.nc", []),
        ("ref/exp1_19820101_grid_U.nc", "cmp/exp2_19820101_grid_U.nc", []),
    ]


def test_default_artifact_matcher_tries_the_common_pattern_only_on_unpaired_files():
    matcher = DefaultArtifactMatcher()
    references = [_netcdf("ref/exp1_1982.nc"), _netcdf("ref/exp1_1983.nc")]
    comparisons = [_netcdf("cmp/exp1_1982.nc"), _netcdf("cmp/exp2_1982.nc"), _netcdf("cmp/exp2_1983.nc")]

    matches = matcher.match(references, comparisons, r"\d{4}")

    # exp1_1982.nc pairs by relative path, so exp2_1982.nc is no longer one of its matches.
    assert _pairs(matches) == [
        ("ref/exp1_1982.nc", "cmp/exp1_1982.nc", []),
        ("ref/exp1_1983.nc", "cmp/exp2_1983.nc", []),
        (None, "cmp/exp2_1982.nc", []),
    ]


def test_default_artifact_matcher_leaves_files_reported_ambiguous_out_of_the_later_steps():
    matcher = DefaultArtifactMatcher()
    references = [_netcdf("ref/a/f_1.nc"), _netcdf("ref/b/f_1.nc"), _netcdf("ref/h_1.nc")]
    comparisons = [_netcdf("cmp/f_1.nc"), _netcdf("cmp/g_1.nc")]

    matches = matcher.match(references, comparisons, r"_\d")

    # The f_1.nc files are settled as ambiguous by the file-name step, so the pattern step
    # sees only h_1.nc and g_1.nc.
    assert _pairs(matches) == [
        ("ref/a/f_1.nc", None, ["cmp/f_1.nc"]),
        ("ref/b/f_1.nc", None, ["cmp/f_1.nc"]),
        ("ref/h_1.nc", "cmp/g_1.nc", []),
        (None, "cmp/f_1.nc", ["ref/a/f_1.nc", "ref/b/f_1.nc"]),
    ]


@pytest.mark.parametrize(
    ("match", "message"),
    [
        (
            ArtifactMatch(_netcdf("ref/x.nc"), None, candidates=(_netcdf("cmp/a/x.nc"), _netcdf("cmp/b/x.nc"))),
            "Same file name as 2 files in the comparison directory: a/x.nc, b/x.nc",
        ),
        (
            ArtifactMatch(None, _netcdf("cmp/x.nc"), candidates=(_netcdf("ref/a/x.nc"), _netcdf("ref/b/x.nc"))),
            "Same file name as 2 files in the reference directory: a/x.nc, b/x.nc",
        ),
        (
            ArtifactMatch(_netcdf("ref/a/x.nc"), None, candidates=(_netcdf("cmp/x.nc"),)),
            "Same file name as x.nc in the comparison directory, which also matches other files in the reference "
            "directory",
        ),
        (
            ArtifactMatch(_netcdf("ref/e1_0101.nc"), None, candidates=(_netcdf("cmp/e2_0101.nc"),)),
            "Same --common-pattern match as e2_0101.nc in the comparison directory, which also matches other files "
            "in the reference directory",
        ),
        (
            ArtifactMatch(None, _netcdf("cmp/x.nc"), candidates=tuple(_netcdf(f"ref/{d}/x.nc") for d in "abcde")),
            "Same file name as 5 files in the reference directory: a/x.nc, b/x.nc, c/x.nc (+2 more)",
        ),
    ],
)
def test_compare_match_fails_an_ambiguous_match_naming_the_candidates(match, message):
    comparison = compare_match(match, make_request(), comparators={})

    assert isinstance(comparison.exception, AmbiguousMatch)
    assert str(comparison.exception) == message
    assert not comparison.passed


def test_comparison_service_fails_a_file_only_in_the_comparison_directory():
    class PassingComparator:
        artifact_kind = ArtifactKind.NETCDF

        def compare(self, match, request):
            comparison = Comparison(reference_artifact=match.reference, comparison_artifact=match.comparison)
            comparison.append(
                CompareResult(relative_error=0.0, min_diff=0.0, max_diff=0.0, mask_equal=True, variable="temp")
            )
            return comparison

    service = ComparisonService(
        discovery=StaticDiscovery(
            {Path("ref"): [_netcdf("ref/a.nc")], Path("cmp"): [_netcdf("cmp/a.nc"), _netcdf("cmp/b.nc")]}
        ),
        matcher=DefaultArtifactMatcher(),
        comparators=[PassingComparator()],
    )

    report = service.run(make_request())

    # The pair that exists on both sides passes, but the extra comparison file used to be
    # ignored and the run passed; now it fails the run.
    assert [comparison.passed for comparison in report.comparisons] == [True, False]
    only_in_comparison = report.comparisons[1]
    assert only_in_comparison.reference_file is None
    assert only_in_comparison.comparison_file == Path("cmp/b.nc")
    assert isinstance(only_in_comparison.exception, NoMatchFound)
    assert str(only_in_comparison.exception) == "Only in the comparison directory"
    assert report.has_failures


def test_comparison_service_runs_a_file_only_in_the_comparison_directory_outside_dask(monkeypatch):
    class FakeComparator:
        artifact_kind = ArtifactKind.NETCDF

        def compare(self, match, request):
            return Comparison(reference_artifact=match.reference, comparison_artifact=match.comparison)

    submissions = []

    class FakeClient:
        def submit(self, func, *args, **kwargs):
            submissions.append(args[0])
            return func(*args)

    class FakeContextManager:
        def __enter__(self):
            return FakeClient()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("xdiff.core.service.dask_runtime.client_from_request", lambda request: FakeContextManager())
    monkeypatch.setattr(
        "xdiff.core.service.dask_runtime.iterate_results_as_completed",
        lambda futures: ((future, future) for future in futures),
    )
    request = CompareRequest(
        input_mode=CompareMode.DIRECTORIES,
        reference_path=Path("ref"),
        comparison_path=Path("cmp"),
        filter_name="*.nc",
        common_pattern=None,
        variables=None,
        last_time_step=False,
        dask_workers=2,
    )
    service = ComparisonService(
        discovery=StaticDiscovery(
            {Path("ref"): [_netcdf("ref/a.nc")], Path("cmp"): [_netcdf("cmp/a.nc"), _netcdf("cmp/b.nc")]}
        ),
        matcher=DefaultArtifactMatcher(),
        comparators=[FakeComparator()],
    )

    report = service.run(request)

    # Only the real pair goes to a worker; the one-sided match is settled inline.
    assert [match.comparison.path.name for match in submissions] == ["a.nc"]
    assert [comparison.comparison_file.name for comparison in report.comparisons] == ["a.nc", "b.nc"]
    assert str(report.comparisons[1].exception) == "Only in the comparison directory"
