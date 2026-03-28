"""Application service for end-to-end artifact comparison."""

from __future__ import annotations

from typing import Iterable

from nccompare.comparators import NetcdfComparator
from nccompare.discovery import FileSystemArtifactDiscovery
from nccompare.exceptions import NoMatchFound, UnsupportedArtifactTypeError
from nccompare.matching import DefaultArtifactMatcher
from nccompare.model.artifact import Artifact
from nccompare.model.comparison import Comparison
from nccompare.model.match import ArtifactMatch
from nccompare.model.report import ComparisonReport
from nccompare.model.request import CompareMode, CompareRequest

class ComparisonService:
    """Coordinate discovery, matching, and comparison for one request."""

    def __init__(self, discovery, matcher, comparators: Iterable[NetcdfComparator]):
        self.discovery = discovery
        self.matcher = matcher
        self.comparators = {comparator.artifact_kind: comparator for comparator in comparators}

    @classmethod
    def default(cls) -> "ComparisonService":
        return cls(
            discovery=FileSystemArtifactDiscovery(),
            matcher=DefaultArtifactMatcher(),
            comparators=[NetcdfComparator()],
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
        for match in matches:
            report.append(self._compare_match(match, request))

        return report

    @staticmethod
    def _build_explicit_file_match(request: CompareRequest) -> ArtifactMatch:
        """Create a direct file-to-file match without discovery or name matching."""
        return ArtifactMatch(
            reference=Artifact.from_path(request.reference_path),
            comparison=Artifact.from_path(request.comparison_path),
        )

    def _compare_match(self, match: ArtifactMatch, request: CompareRequest) -> Comparison:
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

        comparator = self.comparators.get(match.reference.kind)
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
