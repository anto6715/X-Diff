# Repository Guidelines

## Project Overview
`xdiff` is a general-purpose tool for exploring differences between datasets (PyPI distribution `xdiffly`, CLI command and import package `xdiff`). netCDF is the only comparator today; the architecture is format-agnostic (see "adding a new artifact type"). It discovers artifacts in two directory trees, matches them by filename or a common regex pattern, and produces a rich terminal report with per-variable numeric metrics. Python 3.10–3.14 is supported; the local environment uses 3.13 (see `.python-version`).

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
├── exceptions/      Domain exceptions: LastTimestepTimeCheckException,
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
xdiff dirs <REFERENCE_PATH> <COMPARISON_PATH> [OPTIONS]   # compare two directories
xdiff files <REFERENCE.nc> <COMPARISON.nc> [OPTIONS]      # compare two files directly
xdiff plot <REFERENCE.nc> <COMPARISON.nc> [OPTIONS]       # plot where two files differ
```

Key options across the subcommands (`-v`, `--last-time-step`, `--bbox` apply to `plot` too; `-f`, `--common-pattern` and the Dask options are for `dirs`/`files` only):

| Option | Description |
|---|---|
| `-f / --filter` | Glob filter for files (default `*.nc`) |
| `--common-pattern` | Regex identifying a common substring between filenames (e.g. `\d{8}`) |
| `-v / --variables` | Variable(s) to compare; repeatable. `NAME` compares the same name on both sides; `REF=CMP` maps differently-named variables (e.g. `thetao=votemper`) |
| `--last-time-step` | Compare only the last time step |
| `--bbox LON_MIN LON_MAX LAT_MIN LAT_MAX` | Crop both inputs to a lon/lat box before comparing (same-grid inputs of different extent; supports 1-D and 2-D lon/lat coords) |
| `-w / --dask-workers N` | Run in parallel on a local Dask cluster with N workers (enables Dask) |
| `--dask-scheduler` | Attach to an external Dask scheduler by address |
| `--dask-scheduler-file` | Attach to an external Dask scheduler via scheduler file |
| `--no-progress` | Disable live progress bar |

Exit code 0 = all comparisons passed; exit code 1 = one or more failures.

`plot`-specific options (needs the `plot` extra):

| Option | Description |
|---|---|
| `-o / --output FILE` | Write a static image; the extension picks the format (`.png`/`.pdf`/`.svg`). Omit for the live interactive server. |
| `--port N` | Port for the live server (default `5006`; fails fast if busy, never auto-incremented). Ignored with `-o`. |
| `--no-open` | Do not auto-open a browser; print the URL (for headless / `ssh -L` sessions). |

Without `-o`, `plot` starts a Panel/Bokeh server bound to `localhost` and blocks until Ctrl-C (exit 0); with `-o` it renders one triptych per variable and exits.

## Build, Test, and Development Commands

Use `uv` for all local setup and execution:

- `uv sync`: create the virtual environment and install runtime + dev dependencies (the default `dev` group pulls in the `dask` extra, so the full toolchain is present).
- `uv sync --no-default-groups`: install only the base runtime — serial execution, no Dask. This mirrors what PyPI users get from `pip install xdiffly`.
- `uv sync --extra dask`: base runtime plus the optional Dask backend (`dask`, `distributed`, `bokeh`); this is what `pip install "xdiffly[dask]"` provides.
- `uv sync --extra plot`: base runtime plus the optional plotting backend (`matplotlib` for static images; `holoviews`, `panel`, `bokeh` for the live server); this is what `pip install "xdiffly[plot]"` provides. Required for the `plot` subcommand.
- `uv run xdiff dirs a b`: run the CLI against the bundled sample folders.
- `uv run xdiff --help`: verify argument parsing and exposed options.
- `uv build`: create wheel and sdist artifacts in `dist/`.
- `uv run python -m compileall xdiff`: quick syntax check before opening a PR.

CI mirrors the same commands (see `.github/workflows/tests.yml`):

- `uv sync --group dev`: install runtime + dev dependencies.
- `uv run ruff check xdiff tests` and `uv run ruff format --check xdiff tests`: lint and format checks (a dedicated `lint` job).
- `uv run pytest --cov --cov-report=term-missing`: run tests with coverage across a Python 3.10–3.14 matrix.
- `uv run towncrier check --compare-with origin/<base>`: validate changelog entries on non-release PRs.

## Coding Style & Naming Conventions

Target Python 3.10+ and follow the existing style: four-space indentation, `snake_case` for modules/functions/variables, and `PascalCase` for classes (e.g. `ComparisonService`, `NetcdfComparator`). Keep imports grouped cleanly, prefer `pathlib.Path` over raw path strings, and preserve explicit type hints in public code paths. Use `from __future__ import annotations` at the top of new modules. Ruff is configured in `pyproject.toml` (line length 120; `E`/`W`/`F`/`I`/`UP`/`B` rules) and enforced in CI — run `uv run ruff check` and `uv run ruff format` before opening a PR, and avoid unrelated reformatting.

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

- `.github/workflows/tests.yml`: runs on every PR and push to `master`. A `lint` job runs ruff (check + format) and validates changelog entries (skipped on release PRs). A `test` job runs `pytest --cov` on a Python matrix chosen by `setup-matrix`: the endpoints (3.10 + 3.14) on develop PRs, the full 3.10–3.14 range on master PRs/pushes. Coverage uploads to Codecov once, from the 3.14 job. uv dependency caching is enabled on all jobs.
- `.github/workflows/release-changelog.yml`: triggers on PRs from `release/X.Y.Z` branches into `master`. Bumps the version in `pyproject.toml` with `sed`, builds the changelog with `towncrier`, and commits the result back to the release branch.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects (e.g. `add files subcommand`, `fix relative error for time dtype`). Pull requests should state the problem, summarize the fix, list the validation command(s) you ran, include a changelog fragment in `changes.d/`, and update screenshots in `docs/` when terminal output changes. Release PRs must come from `release/X.Y.Z` branches into `master`; the release changelog workflow handles `pyproject.toml` and `CHANGES.md` automatically.
