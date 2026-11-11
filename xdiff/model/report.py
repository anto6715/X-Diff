"""Aggregate report returned by the comparison service."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from xdiff.model.comparison import Comparison
from xdiff.model.request import CompareRequest


@dataclass(slots=True)
class ComparisonReport:
    """Top-level comparison outcome for a request."""

    request: CompareRequest
    comparisons: list[Comparison] = field(default_factory=list)

    def __iter__(self) -> Iterator[Comparison]:
        return iter(self.comparisons)

    def __len__(self) -> int:
        return len(self.comparisons)

    def append(self, comparison: Comparison) -> None:
        self.comparisons.append(comparison)

    @property
    def is_empty(self) -> bool:
        """True when the run compared nothing (no file pair was found or matched)."""
        return len(self.comparisons) == 0

    @property
    def passed(self) -> bool:
        # An empty run is not a success: nothing was verified.
        return not self.is_empty and all(comparison.passed for comparison in self.comparisons)

    @property
    def has_failures(self) -> bool:
        return not self.passed

    @property
    def passed_count(self) -> int:
        return sum(comparison.passed for comparison in self.comparisons)

    @property
    def failed_count(self) -> int:
        return len(self.comparisons) - self.passed_count
