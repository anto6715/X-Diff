"""Request model for a comparison execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class CompareMode(str, Enum):
    """Supported high-level comparison modes."""

    DIRECTORIES = "dirs"
    FILES = "files"


class ExecutionMode(str, Enum):
    """Supported execution strategies for running comparisons."""

    SERIAL = "serial"
    FILES = "files"


def validate_execution_options(
    execution_mode: ExecutionMode,
    dask_scheduler: str | None,
    dask_scheduler_file: Path | None,
    dask_workers: int | None,
) -> None:
    """Validate the Dask-related options shared by the CLI and request model."""
    has_external_scheduler = dask_scheduler is not None or dask_scheduler_file is not None

    if dask_scheduler is not None and dask_scheduler_file is not None:
        raise ValueError("Use either '--dask-scheduler' or '--dask-scheduler-file', not both.")

    if dask_workers is not None and dask_workers < 1:
        raise ValueError("'--dask-workers' must be greater than zero.")

    if execution_mode is ExecutionMode.SERIAL:
        if has_external_scheduler or dask_workers is not None:
            raise ValueError("Dask options require '--execution-mode files'.")
        return

    if dask_workers is not None and has_external_scheduler:
        raise ValueError("Use either an external Dask scheduler or '--dask-workers', not both.")

    if not has_external_scheduler and dask_workers is None:
        raise ValueError(
            f"'--execution-mode {execution_mode.value}' requires either '--dask-workers' or an external Dask scheduler."
        )


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
    execution_mode: ExecutionMode = ExecutionMode.SERIAL
    dask_scheduler: str | None = None
    dask_scheduler_file: Path | None = None
    dask_workers: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution_mode, ExecutionMode):
            object.__setattr__(self, "execution_mode", ExecutionMode(self.execution_mode))
        if self.dask_scheduler_file is not None and not isinstance(self.dask_scheduler_file, Path):
            object.__setattr__(self, "dask_scheduler_file", Path(self.dask_scheduler_file))

        validate_execution_options(
            execution_mode=self.execution_mode,
            dask_scheduler=self.dask_scheduler,
            dask_scheduler_file=self.dask_scheduler_file,
            dask_workers=self.dask_workers,
        )

    @property
    def uses_dask(self) -> bool:
        return self.execution_mode is not ExecutionMode.SERIAL

    @property
    def uses_external_dask_scheduler(self) -> bool:
        return self.dask_scheduler is not None or self.dask_scheduler_file is not None
