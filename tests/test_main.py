from pathlib import Path

from nccompare.core import main

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
            self.report = object()

        def run(self, request):
            self.request = request
            return self.report

    fake_service = FakeService()

    monkeypatch.setattr(main.ComparisonService, "default", lambda: fake_service)

    report = main.execute(
        Path("ref-dir"),
        Path("cmp-dir"),
        "*.nc",
        r"\d{8}\.nc",
        ["temp"],
        True,
    )

    assert report is fake_service.report
    assert fake_service.request.reference_root == Path("ref-dir")
    assert fake_service.request.comparison_root == Path("cmp-dir")
    assert fake_service.request.filter_name == "*.nc"
    assert fake_service.request.common_pattern == r"\d{8}\.nc"
    assert fake_service.request.variables == ("temp",)
    assert fake_service.request.last_time_step is True


def test_build_request_normalizes_default_variable_selection_to_none():
    request = main.build_request(
        folder1=Path("ref-dir"),
        folder2=Path("cmp-dir"),
    )

    assert request.variables is None
