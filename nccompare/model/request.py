"""Request model for a comparison execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CompareRequest:
    """User-supplied settings normalized for the application layer."""

    reference_root: Path
    comparison_root: Path
    filter_name: str
    common_pattern: str | None
    variables: tuple[str, ...] | None
    last_time_step: bool = False
