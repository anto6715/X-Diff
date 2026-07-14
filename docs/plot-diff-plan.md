# Plot-Diff Feature — Development Plan (`xdiff plot`)

> **Status:** approved design, not yet implemented.
> **Audience:** an implementing agent picking up one iteration in a fresh session.
> **How to use this doc:** read §1–§4 for the locked design, then implement exactly one
> iteration from §5, following the per-iteration checklist. You do **not** need the
> conversation that produced this — everything needed is here plus `AGENTS.md` and
> `CLAUDE.md`. Re-verify any `function name` against current source before relying on it
> (line numbers drift; names are stable).

---

## 1. Goal & converged decisions

Turn a comparison from *numbers* (min/max/relative error) into a *picture* of **where** two
netCDF fields differ. New subcommand **`xdiff plot REF CMP ...`**.

**Locked decisions (do not re-litigate — these were converged with the user):**

1. **Two output modes, selected by presence of `-o`:**
   - **No `-o` (default) → live interactive server.** Build the plot, start a **Panel/Bokeh
     server** on `http://localhost:PORT`, auto-open the browser, and **block** until Ctrl-C.
     Data stays in the running process; the browser talks to it over a websocket. **When
     `xdiff` exits, the server stops and the plot dies.** This is intentional (ephemeral, no
     stale artifacts). **No standalone `.html` file is ever written** — the user explicitly
     rejected embedded-HTML files.
   - **`-o FILE.{png,pdf,svg}` → static image, one-shot.** Render with matplotlib, write the
     file(s), exit. For reports/scripting.
2. **Remote/HPC is a first-class target.** The server binds `localhost` only; the user views
   it via `ssh -L PORT:localhost:PORT`. Provide `--port` (a fixed default; override for
   custom) and `--no-open` (print the URL instead of launching a browser — for headless).
3. **Interactive stack = hvplot / holoviews / Panel** (reuses Bokeh, already present via the
   `dask` extra; consistent with the existing Dask-dashboard Bokeh server). **Static = matplotlib.**
4. **One optional extra `xdiff[plot]`** bundling matplotlib + hvplot + holoviews + panel
   (+ bokeh). Lazy-imported, mirroring the `dask` extra pattern.
5. **Content per variable = triptych: `reference | comparison | difference`.** The diff panel
   uses a **diverging colormap centered at 0** with **symmetric limits** (`±max(|nanmin|,|nanmax|)`).
   Use lon/lat for axes when detectable.
6. **Reuse the comparison options** so you plot exactly what you'd compare: `-v` (including
   `REF=CMP` mapping), `--bbox`, `--last-time-step`.
7. **Scope = single file pair** (mirror `xdiff files`: `plot REF CMP`). Directory-wide
   plotting is out of scope for the MVP (deferred to §5, Iteration 3).
8. **Multi-variable output naming (static):** 1 variable → write exactly `-o` (`diff.png`);
   N variables → insert the label into the stem (`diff.png` → `diff_thetao.png`, …). Unknown/
   absent extension on `-o` → clear error listing supported extensions.

---

## 2. Architecture

Two concerns, cleanly split (mirrors the existing `ArtifactComparator` seam):

### 2a. `PlotSpec` — backend-agnostic description of what to draw
All the *logic* lives here; both renderers consume it. Suggested shape (a new package
`xdiff/plotting/spec.py`):

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class VariablePlot:
    label: str                 # "thetao" or "thetao -> votemper" (from _as_variable_pair)
    reference: np.ndarray      # reduced to <=2-D
    comparison: np.ndarray     # same shape as reference
    difference: np.ndarray     # reference - comparison, computed in float (see §2c)
    lon: np.ndarray | None     # 1-D or 2-D; axis coordinates, or None if undetectable
    lat: np.ndarray | None
    diff_limit: float          # symmetric colour limit = max(abs(nanmin), abs(nanmax)) of difference
    units: str | None          # from ref_da.attrs.get("units")
    dims: tuple[str, ...]      # reduced dims, for axis labels / 1-D vs 2-D dispatch

@dataclass(frozen=True)
class PlotSpec:
    reference_path: Path
    comparison_path: Path
    variables: list[VariablePlot]
```

**Builder** (`build_plot_spec(...) -> PlotSpec`), the heart of the feature:
```python
def build_plot_spec(
    reference_path: Path,
    comparison_path: Path,
    variables,                 # normalized (ref, cmp) pairs or None, same shape as CompareRequest.variables
    *,
    last_time_step: bool,
    bbox: BoundingBox | None,
) -> PlotSpec
```
Steps (reuse existing helpers — see §7):
1. `xr = load_xarray()`; open both datasets (`with xr.open_dataset(...) as ...`).
2. If `bbox`: `dataset = crop_to_bbox(dataset, bbox)` on **both** (reuse as-is).
3. `pairs = get_dataset_variables(reference_ds, variables)` → list of `(ref_name, cmp_name)`.
4. For each pair: `ref_da = reference_ds[ref_name]`, `cmp_da = comparison_ds[cmp_name]`.
   - **Reduce to ≤2-D** (new helper `reduce_to_plottable`, §2b).
   - Build `label` exactly like `compare_datasets`: `ref_name if ref_name==cmp_name else f"{ref_name} -> {cmp_name}"`.
   - Compute `difference` with the **integer-underflow-safe** subtraction (§2c).
   - `lon_name, lat_name = locate_horizontal_coords(reference_ds)`; take their values for axes
     (None if absent → renderer falls back to index axes).
   - `diff_limit = float(max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))))` (guard all-NaN → 0/skip).
5. Return `PlotSpec`. **Read `.values` (materialize) inside the `with` block** so arrays outlive
   the closed datasets.

### 2b. N-D → 2-D reduction (new helper)
`reduce_to_plottable(field: xr.DataArray, *, last_time_step: bool) -> xr.DataArray`:
- Horizontal dims = the dims of the lon/lat coordinate(s) (1-D: `(lon,)`,`(lat,)`; 2-D
  curvilinear: `(y, x)`). Keep those.
- For every **other** dim: if it is the time dim (`find_time_dims_name`), select last step when
  `last_time_step` else first; for any other extra dim (e.g. depth) select index 0 and **log at
  info** which index was taken.
- Result should be ≤2-D. If it ends up 1-D → renderer draws line plots (ref/cmp/diff lines).
  If 0-D or no horizontal dims → skip that variable with an info log (record nothing in the spec).

### 2c. Integer-safe difference (copy the policy from `compare_variables`)
Mirror the existing guard so plots don't show wrapped values:
```python
if np.issubdtype(ref_values.dtype, np.integer):
    difference = ref_values.astype(np.float64) - cmp_values.astype(np.float64)
else:
    difference = ref_values - cmp_values
```

### 2d. Renderer seam (two lifecycles)
`-o` presence dispatches between two renderers with **different lifecycles**:

```python
# xdiff/plotting/renderers/matplotlib_renderer.py
def render_to_files(spec: PlotSpec, output: Path) -> list[Path]:
    """Write one static image per variable; return the paths. One-shot."""

# xdiff/plotting/renderers/server.py   (Iteration 2)
def serve(spec: PlotSpec, *, port: int, open_browser: bool, address: str = "localhost") -> None:
    """Start a Panel/Bokeh server, (optionally) open the browser, and BLOCK until Ctrl-C."""
```
The command chooses: `-o` given → `render_to_files`; else → `serve`. Keep the extension→format
mapping (`.png/.pdf/.svg` → matplotlib) in a tiny helper; unknown extension → `ValueError`.

---

## 3. Packaging

- **`pyproject.toml`:** add to `[project.optional-dependencies]`:
  ```toml
  plot = [
      "matplotlib>=3.8,<4.0",
      "hvplot>=0.10,<0.12",
      "holoviews>=1.18,<2.0",
      "panel>=1.4,<2.0",
      "bokeh>=3.1,<4.0",
  ]
  ```
  (Pin ranges are a starting point — the implementing agent should resolve/lock actual
  compatible versions with `uv lock` and confirm they install on 3.10–3.14, exactly as was
  done for the `dask` extra.)
- **Dev group:** append `"xdiffly[plot]"` to `[dependency-groups] dev` (so CI/tests get the
  plotting stack), mirroring the existing `"xdiffly[dask]"` line.
- **Lazy imports, always inside functions** (never at module top): mirror `load_xarray()` and
  `dask_runtime._load_distributed()` — raise a clear `RuntimeError` telling the user to
  `uv tool install "xdiffly[plot]"` / `uv sync --extra plot` when the import fails.
- **Static rendering must be headless:** call `matplotlib.use("Agg")` **before** importing
  `pyplot`, so PNG/PDF rendering needs no display (CI, remote login nodes).

---

## 4. CLI specification (`xdiff plot`)

New Click command in `xdiff/management/cli.py`, structured like `compare_files`:

```
xdiff plot REFERENCE_PATH COMPARISON_PATH [options]

Arguments:
  REFERENCE_PATH   existing .nc file      (reuse the _validate_netcdf_file callback)
  COMPARISON_PATH  existing .nc file

Options:
  -v, --variables ...     repeatable; NAME or REF=CMP  (identical to files/dirs)
  --last-time-step        (identical)
  --bbox LON_MIN LON_MAX LAT_MIN LAT_MAX   (reuse @_bbox_option)
  -o, --output PATH       static image; extension picks format (.png/.pdf/.svg). Omit for live server.
  --port N                server port for the live mode (default: a fixed port, e.g. 5006)
  --no-open               do not auto-open the browser; just print the URL (server mode)
```

**Behaviour:**
- Build normalized inputs exactly like the other commands: `variables=variables or DEFAULT`,
  route `-v` through `normalize_variables` and `--bbox` through `normalize_bbox` (reuse!). You
  can either call a new `core.build_plot_spec(...)` facade, or reuse `build_request` and read
  `request.variables` / `request.bbox` (preferred — one normalization path).
- `-o` present → `render_to_files`, print written paths, **exit 0**.
- `-o` absent → `serve(...)`, which blocks; on Ctrl-C exit 0 cleanly (catch `KeyboardInterrupt`).
- Wrap user-facing errors (missing coords, empty bbox, unknown extension, no plottable
  variables) as `click.ClickException` via the same `(RuntimeError, ValueError)` pattern used
  in `_render_report`.
- **Port default:** pick a fixed, predictable default (e.g. `5006`) so the SSH tunnel is set up
  once; if busy, either fail with a clear message or increment — document the choice. Do **not**
  clash with the Dask dashboard (`:8787`).

---

## 5. Iteration breakdown (each = one branch, one PR, one session)

Follow the repo workflow for every iteration: branch off `develop`, add a towncrier fragment in
`changes.d/<pr>.feature.md`, run `uv run ruff check/format` + `uv run pytest`, open a PR to
`develop`, and drive the real CLI end-to-end (see `AGENTS.md`). Merges are squash to `develop`.

### Iteration 1 — scaffold + `PlotSpec` + static matplotlib renderer
**Deliverable:** `xdiff plot REF CMP -v VAR -o out.png` produces a triptych PNG and exits.
Interactive (no `-o`) is **not** built yet → if `-o` is absent, exit with a clear message
("interactive server not available yet; pass -o FILE.png for a static image"). This keeps
Iteration 1 fully testable with no server complexity.

Add:
- `pyproject.toml`: `plot` extra (matplotlib is the only part Iteration 1 needs, but you may
  declare the full extra now); add `xdiffly[plot]` to dev group; `uv lock`.
- `xdiff/plotting/__init__.py`, `xdiff/plotting/spec.py` (`PlotSpec`, `VariablePlot`,
  `build_plot_spec`, `reduce_to_plottable`).
- `xdiff/plotting/renderers/__init__.py`, `.../matplotlib_renderer.py` (`render_to_files`,
  `Agg` backend, triptych for 2-D, line plots for 1-D, NaN → blank, diverging cmap centered 0,
  multi-var naming helper, extension→format helper).
- `xdiff/management/cli.py`: the `plot` command (static path only).
Tests (`tests/test_plotting.py`): `build_plot_spec` on a synthetic 2-D pair (labels, diff
values, diff_limit, lon/lat picked up, integer-safe diff, bbox applied, `REF=CMP` mapping,
last_time_step reduction, extra-dim reduction, skip 0-D); extension/naming helper (1 var exact,
N vars suffixed, bad extension raises); matplotlib smoke (`render_to_files` writes a non-empty
file, no exception). Drive real CLI: `xdiff plot a.nc b.nc -v sst -o /tmp/x.png` → file exists.
**Acceptance:** static triptych renders for 2-D and 1-D vars; ruff clean; tests pass on 3.10–3.14.

### Iteration 2 — live interactive Panel/Bokeh server (default path)
**Deliverable:** `xdiff plot REF CMP -v VAR` (no `-o`) serves the plot at
`http://localhost:PORT`, opens the browser, blocks, dies on Ctrl-C.

Add:
- `xdiff/plotting/renderers/server.py`: `serve(spec, *, port, open_browser, address)` — build a
  holoviews/hvplot object per variable (image/quadmesh for 2-D, curve for 1-D; diverging cmap
  centered 0 for the diff), compose into a **single Panel layout** (all variables on one page),
  and `panel.serve(...)`/`.show(...)` bound to `localhost`, blocking. Handle `KeyboardInterrupt`.
- `cli.py`: wire the no-`-o` path to `serve`; add `--port`/`--no-open`; remove the Iteration-1
  "not available" stub.
- Docs: README section on the live server + the `ssh -L` remote recipe; `AGENTS.md` option table.
Tests: build the holoviews/panel object from a `PlotSpec` **without serving** (assert it
constructs, has the expected number of panels); do **not** start a real server in unit tests
(optionally an integration test behind a marker that starts and immediately stops it).
**Acceptance:** live server opens and is interactive (pan/zoom/hover); Ctrl-C exits 0; `--no-open`
prints the URL; binds localhost only.

### Iteration 3+ — future (not MVP; only if requested)
- Live widgets: a **time/depth slider** that recomputes server-side (now possible because the
  process stays alive).
- **datashader** for very large grids (dynamic re-aggregation on zoom).
- **Directory mode** (`plot` over a matched tree, like `dirs`) — many pairs → many pages/files.
- `--diff-only` slim layout; histogram-of-differences panel; cartopy/geoviews coastlines.

---

## 6. Reuse map (call these — do not re-implement)

| Need | Reuse | Module |
|---|---|---|
| Lazy xarray import + clear error | `load_xarray()` | `xdiff/comparators/netcdf.py` |
| Crop to bbox (1-D `.sel` + 2-D `.where`, guards) | `crop_to_bbox(dataset, bbox)` | netcdf.py |
| Locate lon/lat names (CF standard_name/units/common) | `locate_horizontal_coords(dataset)` | netcdf.py |
| Select comparable `(ref,cmp)` pairs (filters string dtype) | `get_dataset_variables(dataset, variables)` | netcdf.py |
| Coerce a spec to a pair | `_as_variable_pair(item)` | netcdf.py |
| Last-time-step selection | `select_last_time_step(field)` | netcdf.py |
| Find the time dim name | `find_time_dims_name(dims)` | netcdf.py |
| datetime/timedelta dtype test | `is_time_dtype(dtype)` | netcdf.py |
| Integer-safe diff policy | pattern inside `compare_variables` | netcdf.py |
| Parse `-v` incl. `REF=CMP`, normalize | `normalize_variables`, `_parse_variable_spec` | `xdiff/core/main.py` |
| Coerce `--bbox` 4-tuple → BoundingBox | `normalize_bbox` | core/main.py |
| BoundingBox value object | `BoundingBox` | `xdiff/model/bbox.py` |
| CLI: bbox option decorator | `_bbox_option` | `xdiff/management/cli.py` |
| CLI: netCDF file arg validation | `_validate_netcdf_file` callback | cli.py |
| CLI: error-wrapping pattern | `_render_report` try/except | cli.py |
| Optional-extra + lazy-import + dev-group pattern | the `dask` extra / `dask_runtime._load_distributed` | `pyproject.toml`, `xdiff/core/dask_runtime.py` |

---

## 7. Testing strategy
- **Unit-test `build_plot_spec` / `reduce_to_plottable` hard** — that's where all logic and bugs
  live (labels, diff correctness incl. integer safety, axis pickup, reduction, mapping, bbox).
- **Smoke-test renderers** — "produces a non-empty file / constructs a valid object, no
  exception." No pixel comparisons. Never start a real blocking server in unit tests.
- Build synthetic datasets in-memory with `xr.Dataset(...)` (see existing `_rectilinear_dataset`
  helper in `tests/test_ncdiff.py`) and `tmp_path` for file round-trips.
- CI already runs a 3.10–3.14 matrix; the plot extra must resolve on all five (verify like the
  `dask` extra was verified on 3.14).

## 8. Risks / open items
- **Large grids** inflate a live server's memory / websocket traffic; datashader (Iter 3) is the
  answer. For the MVP, note the caveat; optionally offer a coarsen/stride later.
- **Port conflicts** on shared login nodes — pick default, document `--port`, fail clearly if busy.
- **hvplot/holoviews/panel version churn** — pin, lock, and verify install on the full matrix.
- **1-D vs 2-D dispatch** in renderers — keep it explicit in `PlotSpec.dims`.
- Curvilinear 2-D coords: prefer hvplot `quadmesh` (honours 2-D lon/lat) over `image`.

## 9. Session handoff protocol
1. Read this file, `AGENTS.md`, `CLAUDE.md`.
2. Confirm which iteration is next (check `git log`/PRs on `develop` for merged `plot` work).
3. Implement only that iteration; re-verify reused function names against current source.
4. Add a towncrier `feature` fragment; ruff + pytest green on the matrix; drive the real CLI.
5. Open a PR to `develop` (squash-merge). Update this file if a decision changes.
