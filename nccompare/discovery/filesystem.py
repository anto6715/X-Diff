"""Filesystem-based artifact discovery."""

from __future__ import annotations

from pathlib import Path

from nccompare.model.artifact import Artifact

class FileSystemArtifactDiscovery:
    """Discover artifacts from a directory using a glob filter."""

    def discover(self, directory: Path, filter_name: str) -> list[Artifact]:
        return [
            Artifact.from_path(path, root=directory) for path in sorted(directory.glob(filter_name)) if path.is_file()
        ]

    def list_paths(self, directory: Path, filter_name: str) -> list[Path]:
        """Compatibility helper for legacy callers that only need paths."""
        return [artifact.path for artifact in self.discover(directory, filter_name)]
