"""Base interfaces for artifact comparators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from nccompare.model.artifact import ArtifactKind
from nccompare.model.comparison import Comparison
from nccompare.model.match import ArtifactMatch
from nccompare.model.request import CompareRequest

class ArtifactComparator(ABC):
    """Compares a matched pair of artifacts."""

    artifact_kind: ArtifactKind

    @abstractmethod
    def compare(self, match: ArtifactMatch, request: CompareRequest) -> Comparison:
        """Return the comparison outcome for a single matched pair."""
