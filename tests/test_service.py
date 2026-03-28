from pathlib import Path

from nccompare.core.service import ComparisonService
from nccompare.matching import DefaultArtifactMatcher
from nccompare.model import Artifact, ArtifactKind, CompareRequest


class StaticDiscovery:
    def __init__(self, artifacts_by_directory):
        self.artifacts_by_directory = artifacts_by_directory

    def discover(self, directory, filter_name):
        return self.artifacts_by_directory[directory]


def make_request():
    return CompareRequest(
        reference_root=Path("ref"),
        comparison_root=Path("cmp"),
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
