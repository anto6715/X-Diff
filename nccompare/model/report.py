"""Aggregate report returned by the comparison service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from nccompare.model.comparison import Comparison
from nccompare.model.request import CompareRequest


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
