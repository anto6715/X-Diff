"""File-level comparison result model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from nccompare.model.artifact import Artifact
from nccompare.model.compare_result import CompareResult

@dataclass(slots=True)
class Comparison:
    """Comparison outcome for a matched artifact pair."""

    reference_artifact: Artifact
    comparison_artifact: Artifact | None
    compare_results: list[CompareResult] = field(default_factory=list)
    exception: Exception | None = None

    @classmethod
    def from_paths(
        cls,
        reference_file: Path,
        comparison_file: Path | None,
        exception: Exception | None = None,
    ) -> "Comparison":
        return cls(
            reference_artifact=Artifact.from_path(reference_file),
            comparison_artifact=(Artifact.from_path(comparison_file) if comparison_file is not None else None),
            exception=exception,
        )

    @property
    def reference_file(self) -> Path:
        return self.reference_artifact.path

    @property
    def comparison_file(self) -> Path | None:
        if self.comparison_artifact is None:
            return None
        return self.comparison_artifact.path

    def __iter__(self) -> Iterator[CompareResult]:
        return iter(self.compare_results)

    def __len__(self) -> int:
        return len(self.compare_results)

    def __getitem__(self, position) -> CompareResult:
        return self.compare_results[position]

    def append(self, result: CompareResult) -> None:
        self.compare_results.append(result)

    def extend(self, results: list[CompareResult]) -> None:
        self.compare_results.extend(results)

    def set_exception(self, exception: Exception) -> None:
        self.exception = exception

    def __str__(self) -> str:
        result_count = len(self.compare_results)

        title = f"Comparison between {self.reference_file} and {self.comparison_file}"
        subtitle = f"\n\t- Variables checked: {result_count}"
        if result_count > 0:
            result_info = "\n\t- Results:"
            for result in self.compare_results:
                result_info += f"\n\t\t- {result}"
        else:
            result_info = ""
        exception_info = f"\n\t- Exception: {self.exception}" if self.exception else ""

        return f"{title}{subtitle}{result_info}{exception_info}\n"
