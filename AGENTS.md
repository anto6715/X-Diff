# Repository Guidelines

## Project Structure & Module Organization
`xdiff/` contains the package code. Use `xdiff/management/` for CLI argument parsing and entrypoints, `xdiff/core/` for top-level execution flow, `xdiff/compare/` for dataset comparison logic, `xdiff/model/` for result objects, `xdiff/printlib/` for terminal formatting, and `xdiff/utils/` for reusable helpers. Default settings live in `xdiff/conf/global_settings.py`. Root files such as `pyproject.toml`, `poetry.lock`, and `README.md` define packaging and user-facing behavior. `docs/` stores screenshot assets, while `a/` and `b/` are useful sample directories for local smoke tests.

## Build, Test, and Development Commands
Use `uv` for local setup and command execution:

- `uv venv`: create the local virtual environment in `.venv/`.
- `uv pip install --python .venv/bin/python -e .`: install the package and runtime dependencies in editable mode.
- `.venv/bin/ncpare a b`: run the CLI against the bundled sample folders.
- `.venv/bin/ncpare --help`: verify argument parsing and exposed options.
- `poetry build`: create wheel and sdist artifacts in `dist/`.
- `.venv/bin/python -m compileall xdiff`: quick syntax check before opening a PR.

## Coding Style & Naming Conventions
Target Python 3.10+ and follow the existing style: four-space indentation, `snake_case` for modules/functions/variables, and `PascalCase` for classes such as `Comparison`. Keep imports grouped cleanly, prefer `pathlib.Path` over raw path strings, and preserve explicit type hints in public code paths. No formatter or linter configuration is committed in `pyproject.toml`, so match the surrounding code and avoid unrelated reformatting.

## Testing Guidelines
The repository now runs `pytest` in CI on pull requests and pushes to `master`. Every non-release pull request is expected to include a `newsfragments/*.md` file and passes a `towncrier check` against its base branch in CI. For new comparison logic, add tests under `tests/` using names like `test_common_pattern_matching.py`, keep fixtures small and deterministic, and use `poetry run pytest --cov --cov-report=term-missing` before release-oriented changes.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `introduce pathlib (#4)` and `set version 0.2.5`. Keep commit titles concise, lower-noise, and focused on one change. Pull requests should state the problem, summarize the fix, list the validation command(s) you ran, include a `newsfragments/*.md` file for user-facing changes, and update screenshots in `docs/` when terminal output changes. Release PRs should come from `release/X.Y.Z` branches into `master`; the release changelog workflow will update `pyproject.toml` and `CHANGELOG.md` automatically on those branches.
