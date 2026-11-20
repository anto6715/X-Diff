"""Models for matching artifacts between two roots."""

from __future__ import annotations

from dataclasses import dataclass

from xdiff.model.artifact import Artifact


@dataclass(frozen=True, slots=True)
class ArtifactMatch:
    """A candidate pair of artifacts to compare.

    One side is ``None`` when the file exists under only one root: ``comparison`` for a
    reference file nothing matched, ``reference`` for a comparison file no reference file
    matched. Both sides cannot be missing.
    """

    reference: Artifact | None
    comparison: Artifact | None

    def __post_init__(self) -> None:
        if self.reference is None and self.comparison is None:
            raise ValueError("An ArtifactMatch needs at least one artifact")
