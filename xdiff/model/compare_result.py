"""Variable-level comparison result model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PASSED = True
FAILED = False


@dataclass(frozen=True, slots=True)
class CompareResult:
    """Result of comparing a single variable."""

    relative_error: float = np.nan
    min_diff: float = np.nan
    max_diff: float = np.nan
    mask_equal: bool = False
    variable: str = ""
    description: str = "-"

    @property
    def passed(self) -> bool:
        return (
            self.description == "-"
            and float(self.min_diff) == 0.0
            and float(self.max_diff) == 0.0
            and self.mask_equal
            and float(self.relative_error) == 0.0
        )
