import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_python_snippet(snippet: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        check=True,
        text=True,
        cwd=ROOT,
    )
    return result.stdout.strip()


def test_cli_import_does_not_load_xarray():
    loaded = run_python_snippet("import sys; import xdiff.management.cli; print('xarray' in sys.modules)")

    assert loaded == "False"


def test_compare_package_import_does_not_load_xarray():
    loaded = run_python_snippet("import sys; import xdiff.compare; print('xarray' in sys.modules)")

    assert loaded == "False"
