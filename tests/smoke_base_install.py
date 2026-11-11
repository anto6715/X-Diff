"""Smoke test for a base install (no ``dask`` / ``plot`` extras, no dev tools).

Not a pytest module: the base install has no pytest. CI runs it with plain Python after
``uv sync --no-default-groups`` (see the ``base-install`` job in
``.github/workflows/tests.yml``); locally:

    uv sync --no-default-groups
    uv run --no-sync python tests/smoke_base_install.py
    uv sync    # restore the dev environment afterwards

It checks that the comparison commands work, and that every feature needing an extra
fails with its install hint rather than an ``ImportError`` traceback.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

OPTIONAL_MODULES = ("distributed", "dask", "matplotlib", "holoviews", "panel", "datashader")


def run(*args: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("xdiff")
    if executable is None:
        sys.exit("the `xdiff` command is not on PATH; run this inside the project environment")
    return subprocess.run([executable, *args], capture_output=True, text=True, timeout=300)


def check(condition: bool, message: str, result: subprocess.CompletedProcess[str] | None = None) -> None:
    if condition:
        print(f"ok: {message}")
        return
    print(f"FAILED: {message}", file=sys.stderr)
    if result is not None:
        print(f"exit code: {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}", file=sys.stderr)
    sys.exit(1)


def expect_install_hint(result: subprocess.CompletedProcess[str], extra: str, what: str) -> None:
    output = result.stdout + result.stderr
    check(result.returncode != 0, f"{what} fails without the {extra} extra", result)
    check("Traceback" not in output, f"{what} fails without a traceback", result)
    check(f"xdiffly[{extra}]" in output, f"{what} points to the {extra} extra", result)


def write_dataset(path: Path, offset: float) -> None:
    import numpy as np
    import xarray as xr

    xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.ones((2, 3, 4)) + offset)},
        coords={
            "time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
            "lat": [1.0, 2.0, 3.0],
            "lon": [1.0, 2.0, 3.0, 4.0],
        },
    ).to_netcdf(path)


def main() -> None:
    installed = [name for name in OPTIONAL_MODULES if importlib.util.find_spec(name) is not None]
    check(not installed, f"no optional backend is installed (found: {installed or 'none'})")

    with tempfile.TemporaryDirectory() as directory:
        reference = Path(directory) / "reference.nc"
        same = Path(directory) / "same.nc"
        different = Path(directory) / "different.nc"
        write_dataset(reference, 0.0)
        write_dataset(same, 0.0)
        write_dataset(different, 0.5)

        result = run("--help")
        check(result.returncode == 0, "`xdiff --help` works", result)

        result = run("files", str(reference), str(same), "-v", "sst", "--no-progress")
        check(result.returncode == 0, "identical files compare as passed (exit 0)", result)

        result = run("files", str(reference), str(different), "-v", "sst", "--no-progress")
        check(result.returncode == 1, "different files compare as failed (exit 1)", result)
        check("Traceback" not in result.stdout + result.stderr, "a failed comparison prints no traceback", result)

        result = run("dirs", directory, directory, "-f", "same.nc", "--no-progress")
        check(result.returncode == 0, "`xdiff dirs` works", result)

        expect_install_hint(
            run("files", str(reference), str(same), "-w", "2", "--no-progress"), "dask", "`--dask-workers`"
        )
        expect_install_hint(
            run("plot", str(reference), str(different), "-o", str(Path(directory) / "diff.png")), "plot", "`plot -o`"
        )
        expect_install_hint(
            run("plot", str(reference), str(different), "--no-open", "--port", "5997"), "plot", "the live `plot` server"
        )

    print("base install smoke test passed")


if __name__ == "__main__":
    main()
