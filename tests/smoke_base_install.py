"""Smoke test for a base install (no ``dask`` / ``plot`` extras, no dev tools).

Not a pytest module: the base install has no pytest. CI runs it with plain Python after
``uv sync --no-default-groups --no-editable`` (see the ``base-install`` job in
``.github/workflows/tests.yml``). To run it locally without touching your dev
environment, build the base install somewhere else:

    UV_PROJECT_ENVIRONMENT=/tmp/xdiff-base uv sync --no-default-groups --no-editable
    /tmp/xdiff-base/bin/python tests/smoke_base_install.py

It checks that the comparison commands work, and that every feature needing an extra
fails with its install hint rather than an ``ImportError`` traceback. The ``xdiff``
command it runs is always the one installed next to the interpreter running this script.
"""

from __future__ import annotations

import importlib.metadata
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

DISTRIBUTION = "xdiffly"


def optional_distributions() -> list[str]:
    """Distributions that only the extras require, read from the package metadata."""
    names = set()
    for requirement in importlib.metadata.requires(DISTRIBUTION) or []:
        if "extra ==" not in requirement:
            continue
        name = re.match(r"[A-Za-z0-9._-]+", requirement).group(0)
        if name.lower() != DISTRIBUTION:  # the `all` extra refers back to the package itself
            names.add(name)
    return sorted(names)


def is_installed(distribution: str) -> bool:
    try:
        importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def xdiff_executable() -> Path:
    """The ``xdiff`` script of this interpreter's environment, never another one on PATH."""
    executable = Path(sys.executable).with_name("xdiff")
    if not executable.exists():
        sys.exit(f"no `xdiff` command next to {sys.executable}; run this with the project environment's Python")
    return executable


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(xdiff_executable()), *args], capture_output=True, text=True, timeout=300)


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
    check(f"{DISTRIBUTION}[{extra}]" in output, f"{what} points to the {extra} extra", result)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("localhost", 0))
        return probe.getsockname()[1]


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
    optional = optional_distributions()
    check(bool(optional), f"the package metadata lists the optional backends ({', '.join(optional) or 'none'})")
    installed = [name for name in optional if is_installed(name)]
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
        check(result.returncode == 0, "identical files exit 0", result)
        check("PASSED" in result.stdout, "identical files are reported as PASSED", result)

        result = run("files", str(reference), str(different), "-v", "sst", "--no-progress")
        check(result.returncode == 1, "different files exit 1", result)
        # The two files differ by exactly 0.5, so the report must show that number: this
        # proves a numeric difference was computed, not merely that something exited 1.
        check("5.00e-01" in result.stdout, "different files report the size of the difference", result)
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
            run("plot", str(reference), str(different), "--no-open", "--port", str(free_port())),
            "plot",
            "the live `plot` server",
        )

    print("base install smoke test passed")


if __name__ == "__main__":
    main()
