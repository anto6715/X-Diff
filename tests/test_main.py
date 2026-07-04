from pathlib import Path

import pytest

from xdiff.core import main
from xdiff.model import CompareMode


def test_load_files_returns_only_matching_files(tmp_path):
    matching_file = tmp_path / "kept.nc"
    matching_file.touch()
    (tmp_path / "ignored.txt").touch()
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "nested.nc").touch()

    files = main.load_files(tmp_path, "*.nc")

    assert files == [matching_file]


def test_execute_builds_request_and_delegates_to_service(monkeypatch):
    class FakeService:
        def __init__(self):
            self.request = None
            self.progress_reporter = None
            self.report = object()

        def run(self, request, progress_reporter=None):
            self.request = request
            self.progress_reporter = progress_reporter
            return self.report

    fake_service = FakeService()
    progress_reporter = object()

    monkeypatch.setattr(main.ComparisonService, "default", lambda: fake_service)

    report = main.execute(
        reference_path=Path("ref-dir"),
        comparison_path=Path("cmp-dir"),
        filter_name="*.nc",
        common_pattern=r"\d{8}\.nc",
        variables=["temp"],
        last_time_step=True,
        progress_reporter=progress_reporter,
    )

    assert report is fake_service.report
    assert fake_service.request.reference_path == Path("ref-dir")
    assert fake_service.request.comparison_path == Path("cmp-dir")
    assert fake_service.request.input_mode is CompareMode.DIRECTORIES
    assert fake_service.request.filter_name == "*.nc"
    assert fake_service.request.common_pattern == r"\d{8}\.nc"
    assert fake_service.request.variables == (("temp", "temp"),)
    assert fake_service.request.last_time_step is True
    assert fake_service.request.uses_dask is False
    assert fake_service.request.dask_scheduler is None
    assert fake_service.request.dask_scheduler_file is None
    assert fake_service.request.dask_workers is None
    assert fake_service.progress_reporter is progress_reporter


def test_build_request_normalizes_default_variable_selection_to_none():
    request = main.build_request(
        reference_path=Path("ref-dir"),
        comparison_path=Path("cmp-dir"),
    )

    assert request.variables is None
    assert request.input_mode is CompareMode.DIRECTORIES
    assert request.uses_dask is False


def test_normalize_variables_parses_plain_and_mapped_specs():
    assert main.normalize_variables(["thetao=votemper", "lon=longitude", "temp"]) == (
        ("thetao", "votemper"),
        ("lon", "longitude"),
        ("temp", "temp"),
    )


def test_normalize_variables_strips_whitespace_around_mapping():
    assert main.normalize_variables(["thetao = votemper"]) == (("thetao", "votemper"),)


@pytest.mark.parametrize("spec", ["=votemper", "thetao=", "=", ""])
def test_normalize_variables_rejects_specs_with_an_empty_side(spec):
    with pytest.raises(ValueError, match="Invalid variable specification"):
        main.normalize_variables([spec])


def test_build_request_enables_dask_from_worker_option():
    request = main.build_request(
        reference_path=Path("ref-dir"),
        comparison_path=Path("cmp-dir"),
        dask_workers=8,
    )

    assert request.uses_dask is True
    assert request.dask_workers == 8
