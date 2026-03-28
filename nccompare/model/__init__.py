"""Public model exports for the comparison domain."""

from nccompare.model.artifact import Artifact, ArtifactKind
from nccompare.model.compare_result import CompareResult
from nccompare.model.comparison import Comparison
from nccompare.model.match import ArtifactMatch
from nccompare.model.report import ComparisonReport
from nccompare.model.request import CompareMode, CompareRequest

__all__ = [
    "Artifact",
    "ArtifactKind",
    "ArtifactMatch",
    "CompareMode",
    "CompareRequest",
    "CompareResult",
    "Comparison",
    "ComparisonReport",
]
