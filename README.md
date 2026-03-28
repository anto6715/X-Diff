# netCDF Diff Comparison tool - ncpare

`ncpare` is a tool for comparing netCDF files, providing a detailed diff of their contents. It is designed to help users
identify differences between datasets stored in netCDF format.

![Python](https://img.shields.io/badge/Python->3.10-blue.svg)
[![Tests](https://github.com/anto6715/ncCompare/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/anto6715/ncCompare/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/anto6715/ncCompare/graph/badge.svg?branch=master)](https://codecov.io/gh/anto6715/ncCompare)
[![Anaconda](https://img.shields.io/badge/conda->22.11.1-green.svg)](https://anaconda.org/)
[![Pip](https://img.shields.io/badge/pip->19.0.3-brown.svg)](https://pypi.org/project/pip/)
[![netcdf4](https://img.shields.io/badge/netcdf4-1.7.1.post1-brown.svg)](https://pypi.org/project/pip/)
[![xarray](https://img.shields.io/badge/xarray-2024.6.0-brown.svg)](https://pypi.org/project/pip/)
[![rich](https://img.shields.io/badge/rich-13.7.1-brown.svg)](https://github.com/Textualize/rich?tab=readme-ov-file)

![Output](https://github.com/anto6715/ncCompare/raw/master/docs/output.png)

## Installation

### Install uv

Follow the official installer:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install in a local virtual environment (recommended for development)

Create and activate a project-local environment:

```shell
uv venv
source .venv/bin/activate
```

Then install `nccompare` inside the active environment:

```shell
uv pip install --python .venv/bin/python -e .
```

Run the CLI from the active environment:

```shell
ncpare --help
```

### Install globally with uv tool

```shell
uv tool install nccompare
```

`uv tool install` installs `ncpare` in uv's global tool environment (similar to `pipx`), not inside this repository's `.venv`.

## Usage

If you installed in `.venv`, activate it first:

```shell
source .venv/bin/activate
```

```shell
ncpare [OPTIONS] COMMAND [ARGS]...

  netCDF comparison tool.

Options:
  --version   Show the version and exit.
  -h, --help  Show this message and exit.

Commands:
  dirs   Compare two directories of netCDF files.
  files  Compare two netCDF files directly, even if their filenames differ.

```

### Select Variables

It is possible to choose which parameter to compare:

```shell
ncpare dirs folder1 folder2 -v votemper -v vosaline
```

![Variables](https://github.com/anto6715/ncCompare/raw/master/docs/variables.png)


### Filter files

As default **ncpare** read iterate over all files in **folder1** and expect to find them in **folder2**. Using filters,
it is possible to select only a subset of input files. For example:

```shell
ncpare dirs folder1 folder2 -f "*_grid_T.nc"
```

### Compare files with different filenames

It is possible to compare two files with different filenames directly:

```shell
ncpare files a/my-simu_19820101_grid_T.nc b/another-exp_19820101_grid_T.nc
```

For directory comparisons, it is still possible to match files with different names if they share a common pattern.
For example, if we have:

* `a/my-simu_19820101_grid_T.nc`
* `b/another-exp_19820101_grid_T.nc`

It is still possible to compare the file with:
```shell
ncpare dirs folder1 folder2 --common-pattern ".+_19820101_grid_T.nc"
```

Notice the regex syntax `.+` to match any pattern before `_19820101`

## Testing

GitHub Actions runs the test suite on every pull request and on pushes to `master`. Coverage is uploaded from CI to Codecov, which powers the README coverage badge.

To run the same checks locally:

```shell
poetry install --with dev
poetry run pytest --cov --cov-report=term-missing --cov-report=xml
```

The Codecov badge will start showing a real percentage after the workflow runs successfully on GitHub and the repository is connected to Codecov.

## Changelog

This repository uses `towncrier` for release notes. Every pull request must include a fragment under `newsfragments/` for user-facing changes, for example:

```text
newsfragments/+cli-filter.bugfix.md
newsfragments/+tests.doc.md
```

Validate fragments locally with:

```shell
poetry run towncrier build --draft --version 0.2.6
```

To mirror the CI-style branch check after committing or staging your fragment:

```shell
git fetch origin master:refs/remotes/origin/master
poetry run towncrier check --compare-with origin/master --staged
```

Release notes are generated from `release/X.Y.Z` branches. Open a PR from `release/X.Y.Z` to `master`, and CI will:

1. set `pyproject.toml` to version `X.Y.Z`
2. run `towncrier build --yes --version X.Y.Z`
3. commit the updated `CHANGELOG.md` and consumed fragments back to the release branch

After the release PR is merged, merge `master` back into `develop` so the generated changelog and fragment deletions return to the integration branch.

## Author

- Antonio Mariani (antonio.mariani@cmcc.it)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact
For any questions or suggestions, please open an issue on the project's GitHub repository.
