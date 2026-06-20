from pathlib import Path

import pytest

from xdiff.model.request import CompareMode, CompareRequest


def _request(**overrides) -> CompareRequest:
    kwargs = dict(
        input_mode=CompareMode.DIRECTORIES,
        reference_path=Path("ref"),
        comparison_path=Path("cmp"),
        filter_name="*.nc",
        common_pattern=None,
        variables=None,
    )
    kwargs.update(overrides)
    return CompareRequest(**kwargs)


def test_no_dask_options_runs_serial():
    request = _request()

    assert request.uses_dask is False
    assert request.uses_external_dask_scheduler is False


def test_dask_workers_enable_dask():
    request = _request(dask_workers=4)

    assert request.uses_dask is True
    assert request.uses_external_dask_scheduler is False


def test_scheduler_file_enables_dask():
    request = _request(dask_scheduler_file=Path("scheduler.json"))

    assert request.uses_dask is True
    assert request.uses_external_dask_scheduler is True


def test_rejects_both_scheduler_address_and_file():
    with pytest.raises(ValueError, match="not both"):
        _request(
            dask_scheduler="tcp://scheduler:8786",
            dask_scheduler_file=Path("scheduler.json"),
        )


def test_rejects_mixed_scheduler_and_local_workers():
    with pytest.raises(ValueError, match="external Dask scheduler"):
        _request(dask_scheduler="tcp://scheduler:8786", dask_workers=2)


def test_rejects_non_positive_workers():
    with pytest.raises(ValueError, match="greater than zero"):
        _request(dask_workers=0)
