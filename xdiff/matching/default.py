"""Default matcher for pairing artifacts across two roots."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable

from xdiff.model.artifact import Artifact
from xdiff.model.match import ArtifactMatch
from xdiff.utils.regex import common_pattern_key


class DefaultArtifactMatcher:
    """Pair each file with at most one file under the other root.

    Three steps, each keyed on one property: the relative path, the file name, then what
    ``common_pattern`` finds in the name. Each step looks only at the files that no earlier step
    settled, and settles every group of files sharing a key across both roots: one file per root
    makes a pair; more than one under either root makes each of them a one-sided match listing
    its ``candidates`` under the other root, and they leave the later steps too. A file no step
    settled gets a one-sided match without candidates.
    """

    def match(
        self,
        reference_artifacts: list[Artifact],
        comparison_artifacts: list[Artifact],
        common_pattern: str | None = None,
    ) -> list[ArtifactMatch]:
        reference_matches: dict[Artifact, ArtifactMatch] = {}
        comparison_matches: dict[Artifact, ArtifactMatch] = {}
        keys: list[Callable[[Artifact], Hashable | None]] = [
            lambda artifact: artifact.relative_path,
            lambda artifact: artifact.path.name,
            lambda artifact: common_pattern_key(artifact.path.name, common_pattern),
        ]

        for key in keys:
            references_left = [artifact for artifact in reference_artifacts if artifact not in reference_matches]
            comparisons_left = [artifact for artifact in comparison_artifacts if artifact not in comparison_matches]
            if not references_left or not comparisons_left:
                break

            comparisons_by_key = _group_by(comparisons_left, key)
            for value, references in _group_by(references_left, key).items():
                comparisons = comparisons_by_key.get(value)
                if comparisons is None:
                    continue
                if len(references) == 1 and len(comparisons) == 1:
                    pair = ArtifactMatch(reference=references[0], comparison=comparisons[0])
                    reference_matches[references[0]] = comparison_matches[comparisons[0]] = pair
                    continue
                for reference in references:
                    reference_matches[reference] = ArtifactMatch(
                        reference=reference, comparison=None, candidates=tuple(comparisons)
                    )
                for comparison in comparisons:
                    comparison_matches[comparison] = ArtifactMatch(
                        reference=None, comparison=comparison, candidates=tuple(references)
                    )

        # Reference files in discovery order, each paired or not; then the comparison files left
        # unpaired, so a file found only under the comparison root cannot let the run pass.
        matches = [
            reference_matches.get(reference) or ArtifactMatch(reference=reference, comparison=None)
            for reference in reference_artifacts
        ]
        for comparison in comparison_artifacts:
            match = comparison_matches.get(comparison) or ArtifactMatch(reference=None, comparison=comparison)
            if match.reference is None:
                matches.append(match)
        return matches


def _group_by(artifacts: list[Artifact], key: Callable[[Artifact], Hashable | None]) -> dict[Hashable, list[Artifact]]:
    """Group artifacts by ``key``, in their order, leaving out those whose key is ``None``."""
    groups: dict[Hashable, list[Artifact]] = defaultdict(list)
    for artifact in artifacts:
        value = key(artifact)
        if value is not None:
            groups[value].append(artifact)
    return groups
