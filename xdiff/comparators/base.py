"""Base interfaces for artifact comparators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from xdiff.model.artifact import ArtifactKind
from xdiff.model.comparison import Comparison
from xdiff.model.match import ArtifactMatch
from xdiff.model.request import CompareRequest

class ArtifactComparator(ABC):
    """Compares a matched pair of artifacts."""

    artifact_kind: ArtifactKind

    @abstractmethod
    def compare(self, match: ArtifactMatch, request: CompareRequest) -> Comparison:
        """Return the comparison outcome for a single matched pair."""
