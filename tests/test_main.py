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


def test_execute_wires_file_loading_matching_and_rendering(monkeypatch):
    calls = {}
    reference_files = [Path("reference.nc")]
    comparison_files = [Path("comparison.nc")]
    matches = {reference_files[0]: comparison_files}
    rendered = []
    expected_output = object()

    def fake_load_files(directory, filter_name):
        calls.setdefault("load_files", []).append((directory, filter_name))
        if directory == Path("ref-dir"):
            return reference_files
        return comparison_files

    def fake_find_file_matches(reference_input_files, comparison_input_files, pattern):
        calls["find_file_matches"] = (
            reference_input_files,
            comparison_input_files,
            pattern,
        )
        return matches

    def fake_compare(compare_match, variables, last_time_step):
        calls["compare"] = (compare_match, variables, last_time_step)
        yield expected_output

    monkeypatch.setattr(main, "load_files", fake_load_files)
    monkeypatch.setattr(main, "find_file_matches", fake_find_file_matches)
    monkeypatch.setattr(main.compare, "compare", fake_compare)
    monkeypatch.setattr(main.formatter, "print_comparison", rendered.append)

    main.execute(
        Path("ref-dir"),
        Path("cmp-dir"),
        "*.nc",
        r"\d{8}\.nc",
        ["temp"],
        True,
    )

    assert calls["load_files"] == [
        (Path("ref-dir"), "*.nc"),
        (Path("cmp-dir"), "*.nc"),
    ]
    assert calls["find_file_matches"] == (reference_files, comparison_files, r"\d{8}\.nc")
    assert calls["compare"] == (matches, ["temp"], True)
    assert rendered == [expected_output]
