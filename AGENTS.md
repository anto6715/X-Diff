# Repository Guidelines

## Project Structure & Module Organization
`nccompare/` contains the package code. Use `nccompare/management/` for CLI argument parsing and entrypoints, `nccompare/core/` for top-level execution flow, `nccompare/compare/` for dataset comparison logic, `nccompare/model/` for result objects, `nccompare/printlib/` for terminal formatting, and `nccompare/utils/` for reusable helpers. Default settings live in `nccompare/conf/global_settings.py`. Root files such as `pyproject.toml`, `poetry.lock`, and `README.md` define packaging and user-facing behavior. `docs/` stores screenshot assets, while `a/` and `b/` are useful sample directories for local smoke tests.

## Build, Test, and Development Commands
Use `uv` for local setup and command execution:

- `uv venv`: create the local virtual environment in `.venv/`.
- `uv pip install --python .venv/bin/python -e .`: install the package and runtime dependencies in editable mode.
- `.venv/bin/ncpare a b`: run the CLI against the bundled sample folders.
- `.venv/bin/ncpare --help`: verify argument parsing and exposed options.
- `poetry build`: create wheel and sdist artifacts in `dist/`.
- `.venv/bin/python -m compileall nccompare`: quick syntax check before opening a PR.

## Coding Style & Naming Conventions
Target Python 3.10+ and follow the existing style: four-space indentation, `snake_case` for modules/functions/variables, and `PascalCase` for classes such as `Comparison`. Keep imports grouped cleanly, prefer `pathlib.Path` over raw path strings, and preserve explicit type hints in public code paths. No formatter or linter configuration is committed in `pyproject.toml`, so match the surrounding code and avoid unrelated reformatting.

## Testing Guidelines
There is no committed automated test suite or coverage gate yet. Every change should include at least one reproducible validation step, usually a CLI smoke test with `a/` and `b/`. For new comparison logic, add `pytest` tests under `tests/` using names like `test_common_pattern_matching.py`, and keep fixtures small and deterministic.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `introduce pathlib (#4)` and `set version 0.2.5`. Keep commit titles concise, lower-noise, and focused on one change. Pull requests should state the problem, summarize the fix, list the validation command(s) you ran, and include updated screenshots in `docs/` when terminal output changes.
