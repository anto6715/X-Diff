# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> A detailed contributor guide already exists in **`AGENTS.md`** (project structure, full CLI option table, changelog/towncrier workflow, CI, commit conventions). Read it for those topics; this file focuses on the architecture and the few things worth knowing before editing code.

## Commands

All tooling runs through `uv`:

- `uv sync --group dev` — install runtime + dev dependencies.
- `uv run xdiff dirs a b` — run the CLI against bundled sample folders (`a/`, `b/`).
- `uv run pytest --cov --cov-report=term-missing` — full test suite with coverage.
- `uv run pytest tests/test_service.py` — run one test file.
- `uv run pytest tests/test_service.py::test_name -x` — run a single test, stop on first failure.
- `uv run python -m compileall xdiff` — quick syntax check before a PR.

Every non-release PR needs a towncrier fragment in `changes.d/` (`<issue>.<feature|bugfix|doc|misc>.md`) or CI's `towncrier check` fails. See `AGENTS.md` for the release flow.

## Architecture

The package name is `xdiff`; the installed CLI command is also `xdiff` (entrypoint `xdiff.management:start_from_command_line_interface`).

**Request flow** (CLI → core → service → report):

1. `management/cli.py` — Click commands `dirs` and `files`. Parses options, validates Dask/execution options via `model.request.validate_execution_options`, builds keyword args, and calls `core.execute(...)`. It owns the **exit code** (1 on `report.has_failures`) and wraps the run in a `ProgressReporter` context.
2. `core/main.py` — thin facade. `execute()` and `build_request()` normalize loose CLI args into a single immutable `CompareRequest`, then call `ComparisonService.default().run(request, progress_reporter=...)`. (`load_files` / `normalize_variables` are compatibility helpers kept for tests/callers.)
3. `core/service.py` — `ComparisonService` orchestrates the pipeline: **discovery → matching → comparison**.
   - `DIRECTORIES` mode: `FileSystemArtifactDiscovery` globs each tree into `Artifact`s, then `DefaultArtifactMatcher` pairs them by relative path or `--common-pattern`.
   - `FILES` mode: skips discovery/matching and builds one explicit `ArtifactMatch`.
   - Comparators are held in a registry keyed by `ArtifactKind`. `compare_match()` enforces the contract: missing pair → `NoMatchFound`, kind mismatch / no comparator → `UnsupportedArtifactTypeError`, and any comparator exception is captured onto the `Comparison` rather than raised.
4. Returns a `ComparisonReport` (request + list of `Comparison`); `printlib/formatter.py` renders it with Rich.

**Comparators** (`comparators/`): `ArtifactComparator` ABC (`artifact_kind` + `compare()`). `NetcdfComparator` is the only implementation. The actual numeric work lives in module-level functions in `comparators/netcdf.py` (`compare_files` → `compare_datasets` → `compare_variables`), producing `CompareResult`s with relative error, min/max diff, and mask equality. Note: by default it compares `data_vars` **plus** dims/coords; string/object dtypes are skipped; dimension-name and coordinate mismatches are logged at debug, not failed — only a true shape (size) mismatch raises (see `validate_matching_metadata`).

**Errors are data, not exceptions.** Past the comparator boundary, failures are attached to `Comparison.exception` / `CompareResult.description` so one bad variable or file never aborts the whole report. Preserve this when editing the service or comparators.

**Lazy imports are deliberate.** `xarray` (`load_xarray()` in `netcdf.py`) and the Dask runtime (`core/dask_runtime.py`) are imported only when actually needed, to keep CLI startup fast. Keep heavy imports lazy and inside functions / `TYPE_CHECKING` blocks.

**Execution modes** (`-m/--execution-mode`): `serial` (default) runs comparisons in a loop; `files`/`arrays` route through `_compare_matches_with_dask`, which submits parallelizable matches to a Dask client (`dask_runtime.client_from_request`) and runs non-parallelizable ones inline. See `docs/dask.md`.

## Adding a new artifact type

1. Create `comparators/<type>.py` implementing `ArtifactComparator`.
2. Add an `ArtifactKind` value in `model/artifact.py` and update `infer_artifact_kind()`.
3. Register the comparator in `service.load_default_comparators()`.

## Conventions

- Python 3.10–3.14; local env is 3.14. Start new modules with `from __future__ import annotations`.
- `snake_case` modules/functions, `PascalCase` classes; prefer `pathlib.Path`; keep explicit type hints on public paths.
- Ruff is committed (`[tool.ruff]` in `pyproject.toml`) and enforced in CI: `uv run ruff check` + `uv run ruff format`. Match surrounding style, avoid unrelated reformatting.
- `compare/ncdiff.py` is a legacy compatibility proxy to `comparators.netcdf` — keep it working but don't extend it.
