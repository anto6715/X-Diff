"""Models for matching artifacts between two roots."""

from __future__ import annotations

from dataclasses import dataclass

from xdiff.model.artifact import Artifact


@dataclass(frozen=True, slots=True)
class ArtifactMatch:
    """A candidate pair of artifacts to compare."""

    reference: Artifact
    comparison: Artifact | None
