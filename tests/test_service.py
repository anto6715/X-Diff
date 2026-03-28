from pathlib import Path

from xdiff.core.service import ComparisonService
from xdiff.matching import DefaultArtifactMatcher
from xdiff.model import Artifact, ArtifactKind, CompareMode, CompareRequest


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
