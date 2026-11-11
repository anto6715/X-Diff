# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> A detailed contributor guide already exists in **`AGENTS.md`** (project structure, full CLI option table, changelog/towncrier workflow, CI, commit conventions). Read it for those topics; this file focuses on the architecture and the few things worth knowing before editing code.

## Commands

All tooling runs through `uv`:

- `uv sync` — install runtime + the default `dev` group (which pulls in the `all` extra, i.e. `dask` + `plot`, so every test can run).
- `uv run pytest --cov --cov-report=term-missing` — full test suite with coverage.
- `uv run pytest tests/test_service.py` — run one test file.
- `uv run pytest tests/test_service.py::test_name -x` — run a single test, stop on first failure.
- `uv run ruff check xdiff tests` and `uv run ruff format --check xdiff tests` — the exact lint/format checks CI runs.
- `uv run xdiff files <ref.nc> <cmp.nc>` / `uv run xdiff dirs <ref_dir> <cmp_dir>` — run the CLI. No sample data is tracked (`*.nc` is gitignored), so bring your own files.

Every non-release PR needs a towncrier fragment in `changes.d/` (`<PR number>.<feature|bugfix|doc|misc>.md`) or CI's `towncrier check` fails.

**Tests** have no `conftest.py` and no sample data: each test builds small xarray datasets and writes them to `tmp_path`. Dask is never started for real; tests replace the client or `dask_runtime._load_distributed` with fakes via `monkeypatch`.

**Packaging:** a new optional extra must also be listed in the `all` extra (`tests/test_packaging.py` enforces it). The uv version is pinned in all three workflows and constrained by `[tool.uv] required-version` and the `uv_build` range in `pyproject.toml`; bump them together so `uv.lock` keeps one format.

**Branching:** trunk-based, with no `develop` branch. Feature/fix PRs go straight into `master` and are tested on Python 3.11 + 3.14; `release/X.Y.Z` PRs into `master` get the full 3.11–3.14 matrix and an automatic version bump + `CHANGES.md` build; publishing a GitHub Release ships to PyPI. Details are in `AGENTS.md` ("Branching & Release Flow").

## Architecture

The PyPI distribution is `xdiffly`; the import package and CLI command are both `xdiff` (entrypoint `xdiff.management:start_from_command_line_interface`). The CLI has three subcommands: `dirs`, `files` (comparison reports) and `plot` (visualize differences).

**Comparison flow** (`dirs` / `files`: CLI → core → service → report):

1. `management/cli.py` — Click commands. Validates the Dask options via `model.request.validate_dask_options`, builds keyword args, and calls `core.execute(...)` inside a `ProgressReporter` context. It owns the **exit code** (1 on `report.has_failures`) and turns `RuntimeError`/`ValueError` into `ClickException`.
2. `core/main.py` — thin facade. `execute()` / `build_request()` normalize loose CLI args into one immutable `CompareRequest` (`-v` values become `(reference_name, comparison_name)` pairs; `REF=CMP` maps different names; `--bbox` becomes a `BoundingBox`), then call `ComparisonService.default().run(...)`. `load_files` / `normalize_variables` / `normalize_bbox` are helpers also used by `plot` and tests.
3. `core/service.py` — `ComparisonService` runs **discovery → matching → comparison**.
   - `DIRECTORIES` mode: `FileSystemArtifactDiscovery` globs each tree into `Artifact`s, then `DefaultArtifactMatcher` pairs them by relative path or `--common-pattern`.
   - `FILES` mode: skips discovery/matching and builds one explicit `ArtifactMatch`.
   - Comparators live in a registry keyed by `ArtifactKind`. `compare_match()` enforces the contract: missing pair → `NoMatchFound`, kind mismatch / no comparator → `UnsupportedArtifactTypeError`, and any comparator exception is captured onto the `Comparison` instead of raised.
4. Returns a `ComparisonReport` (request + list of `Comparison`); `printlib/formatter.py` renders it with Rich.

**Comparators** (`comparators/`): `ArtifactComparator` ABC (`artifact_kind` + `compare()`). `NetcdfComparator` is the only implementation. The numeric work lives in module-level functions in `comparators/netcdf.py`: `compare_files` (opens both datasets, applies `crop_to_bbox`) → `compare_datasets` → `compare_variables`, producing `CompareResult`s with relative error, min/max diff and mask equality. By default it compares every `data_var` **plus** every dimension; string/object dtypes are skipped. Dimension-name and coordinate mismatches are logged at debug, not failed — only a shape (size) or dtype mismatch raises (see `validate_matching_metadata`). Integer fields are promoted to float64 before subtracting, because unsigned/integer subtraction wraps around; keep that when touching the differencing code (plotting does the same).

**Errors are data, not exceptions.** Past the comparator boundary, failures go on `Comparison.exception` / `CompareResult.description` (`compare_datasets` catches per variable), so one bad variable or file never aborts the report. Keep this when editing the service or comparators.

**Pass/fail semantics** (`model/`): a `CompareResult` passes only when `description` is still the default `"-"`, min/max diff and relative error are exactly `0.0`, and the masks are equal. So `description` carries a failure reason and setting it fails the variable; use `note` for a remark on a passing result (e.g. both fields all-NaN). A `Comparison` with an exception or with zero results is failed, and a report with zero comparisons has failures, so an empty match set exits 1.

**"All variables" is a sentinel.** `settings.DEFAULT_VARIABLES_TO_CHECK` is a bare `object()`; both it and `None` mean "every data var and dimension". Compare against it with `in (None, settings.DEFAULT_VARIABLES_TO_CHECK)` as `normalize_variables` and `get_dataset_variables` do.

**Kinds without a comparator.** `infer_artifact_kind()` already classifies `TEXT` and `NAMELIST` files, but nothing is registered for them, so in `dirs` mode a `--filter` that matches such files reports each pair as `UnsupportedArtifactTypeError`.

**Dask is opt-in by option, not by mode.** There is no execution-mode switch: passing `-w/--dask-workers`, `--dask-scheduler` or `--dask-scheduler-file` sets `CompareRequest.uses_dask`, and the service routes through `_compare_matches_with_dask`. That submits one task per parallelizable file pair (`_can_parallelize_match`) to a client from `core/dask_runtime.py` and runs the rest inline. The task is the module-level `compare_match` with the comparator registry as an argument, and local clusters use worker processes, so comparators, matches and the request must stay picklable. See `docs/dask.md`.

**Plotting** (`plot` subcommand, `plotting/`): it does not use the service. `plotting/spec.py` holds all the logic (reduce to a ≤2-D slice, float-safe differencing, lon/lat pickup, skip policy), reusing helpers from `comparators/netcdf.py`. It produces a `PlotSpec` (static) or a `PlotSource` (live). The renderers in `plotting/renderers/` are deliberately thin: `matplotlib_renderer.py` for `-o FILE`, `server.py` (Panel/HoloViews/Bokeh + datashader) for the live localhost server. `-v` means different things in the two modes: static output renders only the requested variables, while the live server always opens every variable and uses `-v` only to choose which one is shown first. Unplottable variables go on `skipped` (printed to stderr) rather than raising. The `--port` help text in `cli.py` hardcodes the default because importing `server.DEFAULT_PORT` there would load the plotting stack; keep the two in sync.

**Lazy imports are deliberate and tested.** `xarray` (via `load_xarray()` in `netcdf.py`), the Dask runtime, and the whole plotting stack are imported only inside the functions that need them, to keep CLI startup fast and because `dask` and `plot` are optional extras that are absent from a base `pip install xdiffly`. `tests/test_lazy_imports.py` fails if importing `xdiff.management.cli` or `xdiff.compare` loads xarray. Put heavy or optional imports inside functions or `TYPE_CHECKING` blocks.

## Adding a new artifact type

1. Create `comparators/<type>.py` implementing `ArtifactComparator`.
2. Add an `ArtifactKind` value in `model/artifact.py` and update `infer_artifact_kind()`.
3. Register the comparator in `service.load_default_comparators()`.

A full walkthrough is in `docs/custom-comparators.md`.

## Conventions

- Python 3.11–3.14; local env pins 3.14.6. Start new modules with `from __future__ import annotations`.
- `snake_case` modules/functions, `PascalCase` classes; prefer `pathlib.Path`; keep explicit type hints on public paths.
- Ruff (line length 120, rules `E/W/F/I/UP/B`) is enforced in CI. Match the surrounding style and avoid unrelated reformatting.
- `compare/ncdiff.py` is a legacy compatibility proxy to `comparators.netcdf`. Keep it working, but don't extend it.
