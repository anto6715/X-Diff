"""Compatibility layer for the application service entrypoint."""

from pathlib import Path
from typing import Iterable

import xdiff.conf as settings

from xdiff.core.service import ComparisonService
from xdiff.discovery import FileSystemArtifactDiscovery
from xdiff.model import CompareMode, CompareRequest, ComparisonReport

def execute(
    reference_path: Path,
    comparison_path: Path,
    filter_name: str = settings.DEFAULT_NAME_TO_COMPARE,
    common_pattern: str | None = settings.DEFAULT_COMMON_PATTERN,
    variables: Iterable[str] | object = settings.DEFAULT_VARIABLES_TO_CHECK,
    last_time_step: bool = False,
    input_mode: CompareMode = CompareMode.DIRECTORIES,
) -> ComparisonReport:
    request = build_request(
        reference_path=reference_path,
        comparison_path=comparison_path,
        input_mode=input_mode,
        filter_name=filter_name,
        common_pattern=common_pattern,
        variables=variables,
        last_time_step=last_time_step,
    )
    return ComparisonService.default().run(request)


def build_request(
    reference_path: Path,
    comparison_path: Path,
    input_mode: CompareMode = CompareMode.DIRECTORIES,
    filter_name: str = settings.DEFAULT_NAME_TO_COMPARE,
    common_pattern: str | None = settings.DEFAULT_COMMON_PATTERN,
    variables: Iterable[str] | object = settings.DEFAULT_VARIABLES_TO_CHECK,
    last_time_step: bool = False,
) -> CompareRequest:
    """Normalize legacy execute arguments into a service request."""
    return CompareRequest(
        input_mode=input_mode,
        reference_path=reference_path,
        comparison_path=comparison_path,
        filter_name=filter_name,
        common_pattern=common_pattern,
        variables=normalize_variables(variables),
        last_time_step=last_time_step,
    )


def normalize_variables(variables: Iterable[str] | object) -> tuple[str, ...] | None:
    if variables in (None, settings.DEFAULT_VARIABLES_TO_CHECK):
        return None
    return tuple(variables)


def load_files(directory: Path, filter_name: str) -> list[Path]:
    """Compatibility helper preserved for callers and tests."""
    return FileSystemArtifactDiscovery().list_paths(directory, filter_name)
