"""Models for matching artifacts between two roots."""

from __future__ import annotations

from dataclasses import dataclass

from xdiff.model.artifact import Artifact


@dataclass(frozen=True, slots=True)
class ArtifactMatch:
    """A candidate pair of artifacts to compare.

    One side is ``None`` when the file was left unpaired: ``comparison`` for a reference file,
    ``reference`` for a comparison file. Both sides cannot be missing. Without ``candidates``,
    the file has no counterpart under the other root. With them, the pairing was ambiguous:
    ``candidates`` are the files under the other root that match it, and either there are
    several of them or they also match other files under its own root (see
    ``DefaultArtifactMatcher``).
    """

    reference: Artifact | None
    comparison: Artifact | None
    candidates: tuple[Artifact, ...] = ()

    def __post_init__(self) -> None:
        if self.reference is None and self.comparison is None:
            raise ValueError("An ArtifactMatch needs at least one artifact")
        if self.candidates and self.reference is not None and self.comparison is not None:
            raise ValueError("Only an unpaired artifact can have candidates")
