"""Domain model for discovered artifacts within a comparison run."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ArtifactKind(str, Enum):
    """Supported artifact categories."""

    NETCDF = "netcdf"
    TEXT = "text"
    NAMELIST = "namelist"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Artifact:
    """A file discovered in a comparison root."""

    path: Path
    root: Path
    kind: ArtifactKind
    relative_path: Path

    @classmethod
    def from_path(cls, path: Path, root: Path | None = None, kind: ArtifactKind | None = None) -> "Artifact":
        """Create an artifact while normalizing its root-relative location."""
        normalized_root = root or path.parent
        normalized_path = path
        artifact_kind = kind or infer_artifact_kind(normalized_path)

        try:
            relative_path = normalized_path.relative_to(normalized_root)
        except ValueError:
            relative_path = normalized_path.name

        return cls(
            path=normalized_path,
            root=normalized_root,
            kind=artifact_kind,
            relative_path=Path(relative_path),
        )


def infer_artifact_kind(path: Path) -> ArtifactKind:
    """Infer an artifact category from the file path."""
    suffix = path.suffix.lower()

    if suffix == ".nc":
        return ArtifactKind.NETCDF
    if suffix in {".txt", ".cfg", ".conf", ".def", ".xml"}:
        return ArtifactKind.TEXT
    if path.name.startswith("namelist"):
        return ArtifactKind.NAMELIST
    return ArtifactKind.UNKNOWN
