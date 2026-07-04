"""Public model exports for the comparison domain."""

from xdiff.model.artifact import Artifact, ArtifactKind
from xdiff.model.bbox import BoundingBox
from xdiff.model.compare_result import CompareResult
from xdiff.model.comparison import Comparison
from xdiff.model.match import ArtifactMatch
from xdiff.model.report import ComparisonReport
from xdiff.model.request import CompareMode, CompareRequest

__all__ = [
    "Artifact",
    "ArtifactKind",
    "ArtifactMatch",
    "BoundingBox",
    "CompareMode",
    "CompareRequest",
    "CompareResult",
    "Comparison",
    "ComparisonReport",
]
