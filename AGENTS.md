# Repository Guidelines

## Project Overview
`xdiff` is a netCDF comparison tool (package name `xdiff`, CLI command `xdiff`). It discovers artifacts in two directory trees, matches them by filename or a common regex pattern, and produces a rich terminal report with per-variable numeric metrics. Python 3.10–3.13 is supported; the local environment uses 3.13 (see `.python-version`).

## Project Structure & Module Organization

```
xdiff/
├── management/      CLI argument parsing and entrypoints (Click-based)
├── core/
│   ├── main.py      Public facade: execute() and build_request() helpers
│   ├── service.py   ComparisonService: orchestrates discovery → matching → comparison
│   └── dask_runtime.py  Dask client helpers (lazy import, used only for non-serial modes)
├── comparators/
│   ├── base.py      ArtifactComparator ABC (artifact_kind + compare())
│   └── netcdf.py    NetcdfComparator: xarray-backed numeric comparison
├── compare/
│   └── ncdiff.py    Legacy compatibility proxy → comparators.netcdf (keep but do not extend)
├── discovery/
│   └── filesystem.py  FileSystemArtifactDiscovery: glob-based artifact discovery
├── matching/
│   └── default.py   DefaultArtifactMatcher: pairs artifacts by relative path or common pattern
├── exceptions/      Domain exceptions: AllNaN, LastTimestepTimeCheckException,
│                    NoMatchFound, UnsupportedArtifactTypeError
├── model/           Result objects: Artifact, ArtifactKind, ArtifactMatch, CompareMode,
│                    CompareRequest, CompareResult, Comparison, ComparisonReport, ExecutionMode
├── printlib/
│   ├── formatter.py Rich-based report renderer
│   └── progress.py  ProgressReporter hierarchy (RichProgressReporter, TextProgressReporter,
│                    NullProgressReporter) + create_progress_reporter() factory
├── utils/
│   ├── log.py       Logging configuration helper
│   ├── module_loading.py  import_string / cached_import helpers
│   └── regex.py     common_pattern_exists() and find_file_matches() for filename matching
└── conf/
    └── global_settings.py  Default settings (DEFAULT_NAME_TO_COMPARE, DEFAULT_COMMON_PATTERN, …)
```

Root files `pyproject.toml`, `uv.lock`, and `README.md` define packaging and user-facing behavior. `docs/` stores screenshot assets. `a/` and `b/` are bundled sample directories for local smoke tests. `changes.d/` holds towncrier changelog fragments.

## CLI Commands

The installed script is `xdiff` (defined in `pyproject.toml` under `[project.scripts]`).

```
xdiff ncdirs <REFERENCE_PATH> <COMPARISON_PATH> [OPTIONS]   # compare two directories
xdiff ncfiles <REFERENCE.nc> <COMPARISON.nc> [OPTIONS]      # compare two files directly
```

Key options shared by both subcommands:

| Option | Description |
|---|---|
| `-f / --filter` | Glob filter for files (default `*.nc`) |
| `--common-pattern` | Regex identifying a common substring between filenames (e.g. `\d{8}`) |
| `-v / --variables` | Variable(s) to compare; repeatable |
| `--last-time-step` | Compare only the last time step |
| `-m / --execution-mode` | `serial` (default), `files` (Dask per-file) |
| `-w / --dask-workers N` | Start a local Dask cluster with N worker processes |
| `--dask-scheduler` | Attach to an external Dask scheduler by address |
| `--dask-scheduler-file` | Attach to an external Dask scheduler via scheduler file |
| `--no-progress` | Disable live progress bar |

Exit code 0 = all comparisons passed; exit code 1 = one or more failures.

## Build, Test, and Development Commands

Use `uv` for all local setup and execution:

- `uv sync`: create the virtual environment and install runtime dependencies.
- `uv sync --group dev`: install runtime + dev dependencies (pytest, pytest-cov, towncrier).
- `uv run xdiff ncdirs a b`: run the CLI against the bundled sample folders.
- `uv run xdiff --help`: verify argument parsing and exposed options.
- `uv build`: create wheel and sdist artifacts in `dist/`.
- `uv run python -m compileall xdiff`: quick syntax check before opening a PR.

CI mirrors the same commands (see `.github/workflows/tests.yml`):

- `uv sync --group dev`: install runtime + dev dependencies.
- `uv run pytest --cov --cov-report=term-missing`: run tests with coverage.
- `uv run towncrier check --compare-with origin/<base>`: validate changelog entries on non-release PRs.

## Coding Style & Naming Conventions

Target Python 3.10+ and follow the existing style: four-space indentation, `snake_case` for modules/functions/variables, and `PascalCase` for classes (e.g. `ComparisonService`, `NetcdfComparator`). Keep imports grouped cleanly, prefer `pathlib.Path` over raw path strings, and preserve explicit type hints in public code paths. Use `from __future__ import annotations` at the top of new modules. No formatter or linter is committed in `pyproject.toml`, so match the surrounding code and avoid unrelated reformatting.

When adding a new artifact type: create a `comparators/<type>.py` implementing `ArtifactComparator`, add a new `ArtifactKind` value in `model/artifact.py`, update `infer_artifact_kind()`, and register the comparator in `service.load_default_comparators()`.

## Testing Guidelines

The repository runs `pytest` in CI on pull requests and pushes to `master`. Test files live under `tests/` and follow the naming pattern `test_<module>.py` (e.g. `test_service.py`, `test_ncdiff.py`). Keep fixtures small and deterministic. Run the full suite locally with:

```
uv run pytest --cov --cov-report=term-missing
```

Every non-release PR must include at least one towncrier changelog fragment under `changes.d/` using the format `<issue>.<type>.md`, where `<type>` is one of `feature`, `bugfix`, `doc`, or `misc`. The CI `towncrier check` will fail otherwise.

## Changelog Fragments

Fragment files live in `changes.d/` and follow the pattern `<issue>.<type>.md`:

| Type | Emoji heading |
|---|---|
| `feature` | 🚀 Features |
| `bugfix` | 🔧 Bugfixes |
| `doc` | Documentation |
| `misc` | Miscellaneous |

Example: `changes.d/17.feature.md` with content `Add support for comparing zarr stores.`

## CI Workflows

- `.github/workflows/tests.yml`: runs on every PR and push to `master`. Sets up uv with Python 3.10, installs with `uv sync --group dev`, validates changelog entries (skipped on release PRs), runs `pytest --cov`, and uploads coverage to Codecov.
- `.github/workflows/release-changelog.yml`: triggers on PRs from `release/X.Y.Z` branches into `master`. Bumps the version in `pyproject.toml` with `sed`, builds the changelog with `towncrier`, and commits the result back to the release branch.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects (e.g. `add files subcommand`, `fix relative error for time dtype`). Pull requests should state the problem, summarize the fix, list the validation command(s) you ran, include a changelog fragment in `changes.d/`, and update screenshots in `docs/` when terminal output changes. Release PRs must come from `release/X.Y.Z` branches into `master`; the release changelog workflow handles `pyproject.toml` and `CHANGES.md` automatically.
