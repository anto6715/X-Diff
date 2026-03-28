"""Application service for end-to-end artifact comparison."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

from xdiff.discovery import FileSystemArtifactDiscovery
from xdiff.exceptions import NoMatchFound, UnsupportedArtifactTypeError
from xdiff.matching import DefaultArtifactMatcher
from xdiff.model.artifact import Artifact
from xdiff.model.comparison import Comparison
from xdiff.model.match import ArtifactMatch
from xdiff.model.report import ComparisonReport
from xdiff.model.request import CompareMode, CompareRequest, ExecutionMode

from . import dask_runtime

if TYPE_CHECKING:
    from xdiff.comparators.base import ArtifactComparator


def load_default_comparators() -> list["ArtifactComparator"]:
    """Build the default comparator registry lazily."""
    from xdiff.comparators import NetcdfComparator

    return [NetcdfComparator()]


def compare_match(
    match: ArtifactMatch,
    request: CompareRequest,
    comparators: Mapping,
) -> Comparison:
    """Compare a single match while preserving the current error-reporting contract."""
    if match.comparison is None:
        return Comparison(
            reference_artifact=match.reference,
            comparison_artifact=None,
            exception=NoMatchFound(f"No match found for {match.reference.path}"),
        )

    if match.reference.kind != match.comparison.kind:
        return Comparison(
            reference_artifact=match.reference,
            comparison_artifact=match.comparison,
            exception=UnsupportedArtifactTypeError(
                f"Artifact kind mismatch: {match.reference.kind.value} vs {match.comparison.kind.value}"
            ),
        )

    comparator = comparators.get(match.reference.kind)
    if comparator is None:
        return Comparison(
            reference_artifact=match.reference,
            comparison_artifact=match.comparison,
            exception=UnsupportedArtifactTypeError(f"No comparator registered for {match.reference.kind.value}"),
        )

    try:
        return comparator.compare(match, request)
    except Exception as exc:
        return Comparison(
            reference_artifact=match.reference,
            comparison_artifact=match.comparison,
            exception=exc,
        )


class ComparisonService:
    """Coordinate discovery, matching, and comparison for one request."""

    def __init__(self, discovery, matcher, comparators: Iterable["ArtifactComparator"]):
        self.discovery = discovery
        self.matcher = matcher
        self.comparators = {comparator.artifact_kind: comparator for comparator in comparators}

    @classmethod
    def default(cls) -> "ComparisonService":
        return cls(
            discovery=FileSystemArtifactDiscovery(),
            matcher=DefaultArtifactMatcher(),
            comparators=load_default_comparators(),
        )

    def run(self, request: CompareRequest) -> ComparisonReport:
        if request.input_mode is CompareMode.FILES:
            matches = [self._build_explicit_file_match(request)]
        else:
            reference_artifacts = self.discovery.discover(
                request.reference_path,
                request.filter_name,
            )
            comparison_artifacts = self.discovery.discover(
                request.comparison_path,
                request.filter_name,
            )
            matches = self.matcher.match(
                reference_artifacts,
                comparison_artifacts,
                request.common_pattern,
            )

        report = ComparisonReport(request=request)
        for comparison in self._compare_matches(matches, request):
            report.append(comparison)

        return report

    @staticmethod
    def _build_explicit_file_match(request: CompareRequest) -> ArtifactMatch:
        """Create a direct file-to-file match without discovery or name matching."""
        return ArtifactMatch(
            reference=Artifact.from_path(request.reference_path),
            comparison=Artifact.from_path(request.comparison_path),
        )

    def _compare_matches(self, matches: list[ArtifactMatch], request: CompareRequest) -> list[Comparison]:
        if request.execution_mode is ExecutionMode.FILES:
            return self._compare_matches_with_dask(matches, request)
        if request.execution_mode is ExecutionMode.ARRAYS:
            return self._compare_matches_with_chunked_arrays(matches, request)
        return [self._compare_match(match, request) for match in matches]

    def _compare_matches_with_dask(self, matches: list[ArtifactMatch], request: CompareRequest) -> list[Comparison]:
        comparisons: list[Comparison | None] = [None] * len(matches)
        parallel_indices: list[int] = []
        parallel_matches: list[ArtifactMatch] = []

        for index, match in enumerate(matches):
            if self._can_parallelize_match(match):
                parallel_indices.append(index)
                parallel_matches.append(match)
            else:
                comparisons[index] = self._compare_match(match, request)

        if len(parallel_matches) == 0:
            return [comparison for comparison in comparisons if comparison is not None]

        dask_runtime.log_file_mode_advisories(request, len(parallel_matches))
        with dask_runtime.client_from_request(request) as client:
            futures = [
                client.submit(compare_match, match, request, self.comparators, pure=False)
                for match in parallel_matches
            ]
            for index, comparison in zip(parallel_indices, client.gather(futures)):
                comparisons[index] = comparison

        return [comparison for comparison in comparisons if comparison is not None]

    def _compare_matches_with_chunked_arrays(
        self,
        matches: list[ArtifactMatch],
        request: CompareRequest,
    ) -> list[Comparison]:
        if not any(self._can_parallelize_match(match) for match in matches):
            return [self._compare_match(match, request) for match in matches]

        dask_runtime.log_local_worker_advisories(request)
        with dask_runtime.client_from_request(request):
            return [self._compare_match(match, request) for match in matches]

    def _can_parallelize_match(self, match: ArtifactMatch) -> bool:
        return (
            match.comparison is not None
            and match.reference.kind == match.comparison.kind
            and self.comparators.get(match.reference.kind) is not None
        )

    def _compare_match(self, match: ArtifactMatch, request: CompareRequest) -> Comparison:
        return compare_match(match, request, self.comparators)
