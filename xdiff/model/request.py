"""Request model for a comparison execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class CompareMode(str, Enum):
    """Supported high-level comparison modes."""

    DIRECTORIES = "dirs"
    FILES = "files"


@dataclass(frozen=True, slots=True)
class CompareRequest:
    """User-supplied settings normalized for the application layer."""

    input_mode: CompareMode
    reference_path: Path
    comparison_path: Path
    filter_name: str
    common_pattern: str | None
    variables: tuple[str, ...] | None
    last_time_step: bool = False
