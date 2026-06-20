from pathlib import Path

import pytest

from xdiff.model.request import CompareMode, CompareRequest, ExecutionMode


def test_serial_mode_rejects_dask_options():
    with pytest.raises(ValueError, match="Dask options require '--execution-mode files'"):
        CompareRequest(
            input_mode=CompareMode.DIRECTORIES,
            reference_path=Path("ref"),
            comparison_path=Path("cmp"),
            filter_name="*.nc",
            common_pattern=None,
            variables=None,
            dask_workers=2,
        )


def test_files_mode_requires_a_dask_backend():
    with pytest.raises(ValueError, match="requires either '--dask-workers'"):
        CompareRequest(
            input_mode=CompareMode.DIRECTORIES,
            reference_path=Path("ref"),
            comparison_path=Path("cmp"),
            filter_name="*.nc",
            common_pattern=None,
            variables=None,
            execution_mode=ExecutionMode.FILES,
        )


def test_removed_arrays_mode_is_rejected():
    with pytest.raises(ValueError, match="arrays"):
        CompareRequest(
            input_mode=CompareMode.DIRECTORIES,
            reference_path=Path("ref"),
            comparison_path=Path("cmp"),
            filter_name="*.nc",
            common_pattern=None,
            variables=None,
            execution_mode="arrays",
            dask_workers=2,
        )


def test_files_mode_rejects_mixed_scheduler_and_local_workers():
    with pytest.raises(ValueError, match="external Dask scheduler"):
        CompareRequest(
            input_mode=CompareMode.DIRECTORIES,
            reference_path=Path("ref"),
            comparison_path=Path("cmp"),
            filter_name="*.nc",
            common_pattern=None,
            variables=None,
            execution_mode=ExecutionMode.FILES,
            dask_scheduler="tcp://scheduler:8786",
            dask_workers=2,
        )


def test_files_mode_accepts_scheduler_file():
    request = CompareRequest(
        input_mode=CompareMode.DIRECTORIES,
        reference_path=Path("ref"),
        comparison_path=Path("cmp"),
        filter_name="*.nc",
        common_pattern=None,
        variables=None,
        execution_mode=ExecutionMode.FILES,
        dask_scheduler_file=Path("scheduler.json"),
    )

    assert request.uses_dask is True
    assert request.uses_external_dask_scheduler is True
