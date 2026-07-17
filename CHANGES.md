# Changelog

All notable changes to this project will be documented in this file.

This repository uses `towncrier` to collect unreleased changes under `changes.d/`.

<!-- towncrier release notes start -->

## __[xdiff-0.4.0](https://github.com/anto6715/X-Diff/tree/0.4.0) - 2026-07-17__

### 🚀 Features

[#20](https://github.com/anto6715/X-Diff/pull/20) - Add support for Python 3.14. (The supported range was later narrowed to 3.11–3.14 in this same release — see the Miscellaneous note for [#27].)

[#23](https://github.com/anto6715/X-Diff/pull/23) - `-v/--variables` now accepts `REF=CMP` to compare differently-named variables (e.g. `-v thetao=votemper -v lon=longitude`), on both the `dirs` and `files` commands. Plain `-v NAME` still compares the same name on both sides. Mapped pairs are labelled `ref -> cmp` in the report.

[#24](https://github.com/anto6715/X-Diff/pull/24) - Add `--bbox LON_MIN LON_MAX LAT_MIN LAT_MAX` (on both `dirs` and `files`) to crop both inputs to a longitude/latitude window before comparing, for same-grid inputs of different extent. Both 1-D rectilinear (`.sel`) and 2-D curvilinear (NEMO-style `nav_lon`/`nav_lat`, `.where`) coordinates are supported; coordinates are located via CF `standard_name`/`units` then common names. A box that locates no coordinates or selects no data fails with a clear error rather than silently comparing nothing.

[#26](https://github.com/anto6715/X-Diff/pull/26) - Add `xdiff plot REF CMP -v VAR -o out.png` to render the difference between two netCDF fields to a static PNG/PDF/SVG — one full-size, smoothly shaded map per variable on plain lon/lat axes (latitude-corrected aspect, cropped to the data domain, land drawn from the data's NaN mask), diverging colormap centered at 0 with a robust color limit.

[#27](https://github.com/anto6715/X-Diff/pull/27) - Add the live interactive mode to `xdiff plot`: run it without `-o` to serve the plots on a local Panel/Bokeh server (`--port`, `--no-open`). A sidebar drives a one-variable-at-a-time view — browse every variable in the file, step through time/depth levels, adjust the colour limit and colormap live (zoom preserved), toggle smooth/blocks rendering, and optionally overlay a web-map basemap (Carto/OSM/Esri, online only). The difference is datashaded so it re-aggregates server-side as you zoom (scroll to zoom) and scales to large grids; a min/max readout shows the true magnitude and hover reads off values. Maps are plain lon/lat with land drawn from the data's NaN mask (no cartopy/geoviews). The reference and comparison maps sit in a collapsed card at the bottom.

### 🔧 Bugfixes

[#21](https://github.com/anto6715/X-Diff/pull/21) - Fix silent integer underflow when comparing integer-typed netCDF variables. Subtracting two unsigned/narrow integer fields wrapped around (e.g. `uint8` `10 - 12` became `254`), corrupting the reported min/max difference and relative error. Integer operands are now promoted to float before the difference is computed.

[#22](https://github.com/anto6715/X-Diff/pull/22) - Two variables that are entirely NaN in the same positions are now reported as identical (a passing check with an explanatory note) instead of raising an error. When all differences are NaN but the NaN masks differ, the check still fails, reporting that the valid regions do not overlap. Two empty (zero-size) variables are likewise treated as identical rather than being mislabelled as all-NaN.

[#27](https://github.com/anto6715/X-Diff/pull/27) - Fix the `plot` extra failing to install: `uv`'s universal resolution over the supported Python range backed `datashader`'s transitive `numba` down to 0.53.1 (an sdist that hard-fails to build on any Python ≥3.10), so `uv sync` broke. Pin `numba>=0.60.0,<1.0.0` so the extra resolves to wheels that build on 3.10–3.14. Also derive `validate_runtime`'s rejection message from the supported-version constants instead of a hard-coded string, so `xdiff` no longer misreports its own supported range.

### Miscellaneous

[#19](https://github.com/anto6715/X-Diff/pull/19) - Make the release-changelog workflow idempotent: skip the changelog build when no towncrier fragments remain, so re-triggered runs on a release branch no longer fail after the release commit has already been applied.

[#20](https://github.com/anto6715/X-Diff/pull/20) - Make Dask an optional dependency. The base install is now serial-only and lighter; install the `dask` extra (`uv tool install "xdiffly[dask]"`, or `uv sync --extra dask` from a source checkout) to enable parallel execution. Supplying a Dask backend option without the extra raises a clear, actionable error. Also add ruff lint/format checks to CI and split it into dedicated `lint` and `test` jobs.

[#27](https://github.com/anto6715/X-Diff/pull/27) - Drop Python 3.10 (now 3.11–3.14) and pin the local interpreter to the exact patch `3.14.6`. CPython 3.14.0 has a numpy crash (a `bool |= float32-array == scalar` compare segfaults — exactly the idiom xarray uses to decode fill values), fixed in later 3.14 patches; a bare `3.14` pin floated to the broken 3.14.0. Fetching 3.14.6 needs uv ≥ 0.11, so the CI `setup-uv` pin is bumped too. Dropping 3.10 also lets the `xarray` cap move off the stale `<2025` (3.10 was xarray's blocker — its last 3.10 release is 2025.6.1), so `xarray` now resolves to the current 2026.x line. The other stale calendar/major caps are lifted alongside it — `pandas` (`<3` → `<4`, now 3.0.3) and the `dask`/`distributed` extra pins (`<2025` → `<2027`, now 2026.x).

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
