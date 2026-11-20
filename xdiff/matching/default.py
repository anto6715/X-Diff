"""Default matcher for pairing artifacts across two roots."""

from __future__ import annotations

from xdiff.model.artifact import Artifact
from xdiff.model.match import ArtifactMatch
from xdiff.utils.regex import common_pattern_exists


class DefaultArtifactMatcher:
    """Pair artifacts by relative path, by file name, or by an optional common pattern.

    A file found under only one root gets a match whose other side is ``None``.
    """

    def match(
        self,
        reference_artifacts: list[Artifact],
        comparison_artifacts: list[Artifact],
        common_pattern: str | None = None,
    ) -> list[ArtifactMatch]:
        matches: list[ArtifactMatch] = []
        paired: set[Artifact] = set()

        for reference in reference_artifacts:
            matched_artifacts: list[Artifact] = []
            for comparison in comparison_artifacts:
                if self._is_match(reference, comparison, common_pattern):
                    matched_artifacts.append(comparison)

            if not matched_artifacts:
                matches.append(ArtifactMatch(reference=reference, comparison=None))
                continue

            for comparison in matched_artifacts:
                matches.append(ArtifactMatch(reference=reference, comparison=comparison))
            paired.update(matched_artifacts)

        # A comparison file no reference file paired with is a difference between the two
        # trees too: report it (after the pairs, in discovery order) so the run cannot pass.
        matches.extend(
            ArtifactMatch(reference=None, comparison=comparison)
            for comparison in comparison_artifacts
            if comparison not in paired
        )
        return matches

    @staticmethod
    def _is_match(reference: Artifact, comparison: Artifact, common_pattern: str | None) -> bool:
        return (
            reference.relative_path == comparison.relative_path
            or reference.path.name == comparison.path.name
            or common_pattern_exists(reference.path.name, comparison.path.name, common_pattern)
        )
