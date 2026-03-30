import importlib

from pathlib import Path

from click.testing import CliRunner

from xdiff.conf import settings
from xdiff.management.cli import cli
from xdiff.model.report import ComparisonReport
from xdiff.model import CompareMode, ExecutionMode

cli_module = importlib.import_module("xdiff.management.cli")


def test_root_help_lists_subcommands():
    runner = CliRunner()

    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "dirs" in result.output
    assert "files" in result.output


def test_root_command_without_subcommand_shows_help():
    runner = CliRunner()

    result = runner.invoke(cli, [])

    assert result.exit_code == 0
    assert "Commands:" in result.output


def test_dirs_command_builds_directory_request(monkeypatch):
    runner = CliRunner()
    report = object()
    captured = {}

    def fake_execute(**kwargs):
        captured["kwargs"] = kwargs
        return report

    monkeypatch.setattr(cli_module.formatter, "print_report", lambda value: captured.setdefault("rendered", value))
    monkeypatch.setattr(cli_module.core, "execute", fake_execute)

    with runner.isolated_filesystem():
        ref_dir = Path("ref")
        cmp_dir = Path("cmp")
        ref_dir.mkdir()
        cmp_dir.mkdir()

        result = runner.invoke(
            cli,
            [
                "dirs",
                str(ref_dir),
                str(cmp_dir),
                "--filter",
                "*_grid_T.nc",
                "--common-pattern",
                r"\d{8}_grid_T\.nc",
                "-v",
                "votemper",
                "-v",
                "vosaline",
                "--last-time-step",
            ],
        )

    assert result.exit_code == 0
    assert captured["rendered"] is report
    assert captured["kwargs"]["reference_path"] == Path("ref")
    assert captured["kwargs"]["comparison_path"] == Path("cmp")
    assert captured["kwargs"]["input_mode"] is CompareMode.DIRECTORIES
    assert captured["kwargs"]["filter_name"] == "*_grid_T.nc"
    assert captured["kwargs"]["common_pattern"] == r"\d{8}_grid_T\.nc"
    assert captured["kwargs"]["variables"] == ("votemper", "vosaline")
    assert captured["kwargs"]["last_time_step"] is True
    assert captured["kwargs"]["execution_mode"] is ExecutionMode.SERIAL
    assert captured["kwargs"]["dask_scheduler"] is None
    assert captured["kwargs"]["dask_scheduler_file"] is None
    assert captured["kwargs"]["dask_workers"] is None


def test_files_command_builds_file_request_for_different_filenames(monkeypatch):
    runner = CliRunner()
    report = object()
    captured = {}

    def fake_execute(**kwargs):
        captured["kwargs"] = kwargs
        return report

    monkeypatch.setattr(cli_module.formatter, "print_report", lambda value: captured.setdefault("rendered", value))
    monkeypatch.setattr(cli_module.core, "execute", fake_execute)

    with runner.isolated_filesystem():
        ref_file = Path("reference.nc")
        cmp_file = Path("another-name.nc")
        ref_file.write_text("placeholder")
        cmp_file.write_text("placeholder")

        result = runner.invoke(
            cli,
            [
                "files",
                str(ref_file),
                str(cmp_file),
                "-v",
                "thetao",
                "--last-time-step",
            ],
        )

    assert result.exit_code == 0
    assert captured["rendered"] is report
    assert captured["kwargs"]["reference_path"] == Path("reference.nc")
    assert captured["kwargs"]["comparison_path"] == Path("another-name.nc")
    assert captured["kwargs"]["input_mode"] is CompareMode.FILES
    assert captured["kwargs"]["filter_name"] == settings.DEFAULT_NAME_TO_COMPARE
    assert captured["kwargs"]["common_pattern"] is settings.DEFAULT_COMMON_PATTERN
    assert captured["kwargs"]["variables"] == ("thetao",)
    assert captured["kwargs"]["last_time_step"] is True
    assert captured["kwargs"]["execution_mode"] is ExecutionMode.SERIAL
    assert captured["kwargs"]["dask_scheduler"] is None
    assert captured["kwargs"]["dask_scheduler_file"] is None
    assert captured["kwargs"]["dask_workers"] is None


def test_files_command_rejects_non_netcdf_inputs():
    runner = CliRunner()

    with runner.isolated_filesystem():
        ref_file = Path("reference.txt")
        cmp_file = Path("comparison.txt")
        ref_file.write_text("placeholder")
        cmp_file.write_text("placeholder")

        result = runner.invoke(cli, ["files", str(ref_file), str(cmp_file)])

    assert result.exit_code != 0
    assert "only .nc files are supported" in result.output


def test_dirs_command_accepts_dask_files_mode(monkeypatch):
    runner = CliRunner()
    report = object()
    captured = {}

    def fake_execute(**kwargs):
        captured["kwargs"] = kwargs
        return report

    monkeypatch.setattr(cli_module.formatter, "print_report", lambda value: captured.setdefault("rendered", value))
    monkeypatch.setattr(cli_module.core, "execute", fake_execute)

    with runner.isolated_filesystem():
        ref_dir = Path("ref")
        cmp_dir = Path("cmp")
        ref_dir.mkdir()
        cmp_dir.mkdir()

        result = runner.invoke(
            cli,
            [
                "dirs",
                str(ref_dir),
                str(cmp_dir),
                "--execution-mode",
                "files",
                "--dask-workers",
                "4",
            ],
        )

    assert result.exit_code == 0
    assert captured["rendered"] is report
    assert captured["kwargs"]["execution_mode"] is ExecutionMode.FILES
    assert captured["kwargs"]["dask_workers"] == 4


def test_dirs_command_rejects_parallel_mode_without_dask_backend():
    runner = CliRunner()

    with runner.isolated_filesystem():
        ref_dir = Path("ref")
        cmp_dir = Path("cmp")
        ref_dir.mkdir()
        cmp_dir.mkdir()

        result = runner.invoke(
            cli,
            [
                "dirs",
                str(ref_dir),
                str(cmp_dir),
                "--execution-mode",
                "files",
            ],
        )

    assert result.exit_code != 0
    assert "--dask-workers" in result.output


def test_files_command_accepts_dask_arrays_mode(monkeypatch):
    runner = CliRunner()
    report = object()
    captured = {}

    def fake_execute(**kwargs):
        captured["kwargs"] = kwargs
        return report

    monkeypatch.setattr(cli_module.formatter, "print_report", lambda value: captured.setdefault("rendered", value))
    monkeypatch.setattr(cli_module.core, "execute", fake_execute)

    with runner.isolated_filesystem():
        ref_file = Path("reference.nc")
        cmp_file = Path("comparison.nc")
        ref_file.write_text("placeholder")
        cmp_file.write_text("placeholder")

        result = runner.invoke(
            cli,
            [
                "files",
                str(ref_file),
                str(cmp_file),
                "--execution-mode",
                "arrays",
                "--dask-workers",
                "2",
            ],
        )

    assert result.exit_code == 0
    assert captured["kwargs"]["execution_mode"] is ExecutionMode.ARRAYS
    assert captured["kwargs"]["dask_workers"] == 2


def test_dirs_command_returns_non_zero_when_report_has_failures(monkeypatch):
    runner = CliRunner()

    failing_report = ComparisonReport(request=object(), comparisons=[])

    monkeypatch.setattr(cli_module.formatter, "print_report", lambda value: None)
    monkeypatch.setattr(cli_module.core, "execute", lambda **kwargs: failing_report)

    with runner.isolated_filesystem():
        ref_dir = Path("ref")
        cmp_dir = Path("cmp")
        ref_dir.mkdir()
        cmp_dir.mkdir()

        result = runner.invoke(
            cli,
            [
                "dirs",
                str(ref_dir),
                str(cmp_dir),
            ],
        )

    assert result.exit_code == 1


def test_dirs_command_builds_progress_reporter(monkeypatch):
    runner = CliRunner()
    report = object()
    captured = {}

    class FakeReporter:
        def __enter__(self):
            captured["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["exited"] = True
            return False

    fake_reporter = FakeReporter()

    def fake_create_progress_reporter(*, disabled=False):
        captured["disabled"] = disabled
        return fake_reporter

    def fake_execute(**kwargs):
        captured["kwargs"] = kwargs
        return report

    monkeypatch.setattr(cli_module.progress, "create_progress_reporter", fake_create_progress_reporter)
    monkeypatch.setattr(cli_module.formatter, "print_report", lambda value: captured.setdefault("rendered", value))
    monkeypatch.setattr(cli_module.core, "execute", fake_execute)

    with runner.isolated_filesystem():
        ref_dir = Path("ref")
        cmp_dir = Path("cmp")
        ref_dir.mkdir()
        cmp_dir.mkdir()

        result = runner.invoke(cli, ["dirs", str(ref_dir), str(cmp_dir)])

    assert result.exit_code == 0
    assert captured["disabled"] is False
    assert captured["kwargs"]["progress_reporter"] is fake_reporter
    assert captured["entered"] is True
    assert captured["exited"] is True
    assert captured["rendered"] is report


def test_dirs_command_supports_no_progress(monkeypatch):
    runner = CliRunner()
    report = object()
    captured = {}

    class FakeReporter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_create_progress_reporter(*, disabled=False):
        captured["disabled"] = disabled
        return FakeReporter()

    monkeypatch.setattr(cli_module.progress, "create_progress_reporter", fake_create_progress_reporter)
    monkeypatch.setattr(cli_module.formatter, "print_report", lambda value: None)
    monkeypatch.setattr(cli_module.core, "execute", lambda **kwargs: report)

    with runner.isolated_filesystem():
        ref_dir = Path("ref")
        cmp_dir = Path("cmp")
        ref_dir.mkdir()
        cmp_dir.mkdir()

        result = runner.invoke(cli, ["dirs", str(ref_dir), str(cmp_dir), "--no-progress"])

    assert result.exit_code == 0
    assert captured["disabled"] is True
