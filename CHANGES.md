# Changelog

All notable changes to this project will be documented in this file.

This repository uses `towncrier` to collect unreleased changes under `changes.d/`.

<!-- towncrier release notes start -->

## __[xdiff-0.3.1](https://github.com/anto6715/X-Diff/tree/0.3.1) - 2026-06-22__

### 🔧 Bugfixes

[#18](https://github.com/anto6715/X-Diff/pull/18) - Comparing netCDF variables no longer fails when dimensions share the same shape but have different names; the name difference is logged at debug instead. Only a true shape (size) mismatch now raises.

## __[xdiff-0.3.0](https://github.com/anto6715/X-Diff/tree/0.3.0) - 2026-06-20__

### 🚀 Features

[#15.dask-files](https://github.com/anto6715/X-Diff/pull/15.dask-files) - Added explicit Dask-backed file-level execution, optional Dask packaging extras, and lazy netCDF imports so CLI startup stays lightweight until an actual comparison begins.

[#15](https://github.com/anto6715/X-Diff/pull/15) - When `xdiff` creates a local Dask cluster via `--dask-workers`, it now prints both the scheduler URL and the browser dashboard link automatically so users can inspect the local cluster without passing an external scheduler.

[#16](https://github.com/anto6715/X-Diff/pull/16) - Added live CLI progress reporting with discovered file counts, scheduled comparison totals, and Rich-based progress updates during long comparison runs.
  Improved the final CLI report with a run summary, an explicit failed-comparisons table, and clearer per-comparison status labels.
  Added per-comparison live result lines during progress reporting so each completed file pair immediately prints a compact PASSED or FAILED outcome before the final summary.

 - Removed the `-m/--execution-mode` option. Dask parallel execution is now enabled implicitly by passing a backend option — `--dask-workers` for a local cluster, or `--dask-scheduler`/`--dask-scheduler-file` for an external one; with no backend option, comparisons run serially. **Breaking change:** `--execution-mode` no longer exists. CLI help text is also target-neutral now ("datasets" instead of "netCDF files") to reflect the tool's general-purpose direction.

### 🔧 Bugfixes

[#15](https://github.com/anto6715/X-Diff/pull/15) - Hardened netCDF comparisons so mismatched dimensions, coordinates, and empty comparison results fail explicitly, fixed `--last-time-step` handling for non-leading time axes, and returned a non-zero CLI status when comparisons fail.

 - Allow netCDF variable comparisons to continue when coordinate metadata differs, comparing values positionally when dimensions and shapes match.

 - Restricted supported Python versions to 3.10 through 3.13, pinned the default local interpreter to Python 3.13, and updated the uv installation examples to avoid creating Python 3.14 environments.

### Miscellaneous

[#14](https://github.com/anto6715/X-Diff/pull/14) - - Rename ncpare to xdiff
  - Adoption of Cclick subcommands
  - Direct support to file comparison
  - Architecture refactoring
  - Added Towncrier-based changelog management, seeded the historical changelog, enforced changelog-entry checks in pull-request CI, and automated release-branch changelog generation for PRs into `master`.

[#15](https://github.com/anto6715/X-Diff/pull/15) - Made `dask`, `distributed`, and `bokeh` regular runtime dependencies so local Dask execution and dashboard support are installed by default.

 - Migrate project configuration, CI, and documentation from Poetry to uv.

 - Published on PyPI under the general-purpose name `xdiffly` (install with `pip install xdiffly` / `uv tool install xdiffly`). The CLI command and import package remain `xdiff`.

## 0.2.5 (2025-12-11)

### 🔧 Bugfixes

- Improved comparison handling so non-comparable variable types are skipped more safely.

## 0.2.2 (2024-08-06)

### Documentation

- Removed the README table of contents and updated image links to point to the GitHub-hosted assets.

## 0.2.1 (2024-08-06)

### 🚀 Features

- Reworked comparison result rendering around the `Comparison` model and the Rich-based terminal table output.

### 🔧 Bugfixes

- Improved variable selection, relative-error handling, and the internal comparison flow performance.

## 0.2.0 (2024-07-30)

### 🚀 Features

- Switched core path handling to `pathlib`.

## 0.1.3 (2024-07-30)

### 🚀 Features

- Introduced the compare result model used by the dataset comparison flow.

## 0.1.2 (2024-07-30)

### Miscellaneous

- Published the 0.1.2 package version update.

## 0.1.1 (2024-07-30)

### 🚀 Features

- Published the first working package release.

## 0.1.0 (2024-07-22)

### 🚀 Features

- Created the initial project skeleton.
