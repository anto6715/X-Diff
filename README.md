# netCDF Diff Comparison tool - ncpare

`ncpare` is a tool for comparing netCDF files, providing a detailed diff of their contents. It is designed to help users
identify differences between datasets stored in netCDF format.

![Python](https://img.shields.io/badge/Python->3.10-blue.svg)
[![Anaconda](https://img.shields.io/badge/conda->22.11.1-green.svg)](https://anaconda.org/)
[![Pip](https://img.shields.io/badge/pip->19.0.3-brown.svg)](https://pypi.org/project/pip/)
[![netcdf4](https://img.shields.io/badge/netcdf4-1.7.1.post1-brown.svg)](https://pypi.org/project/pip/)
[![xarray](https://img.shields.io/badge/xarray-2024.6.0-brown.svg)](https://pypi.org/project/pip/)
[![rich](https://img.shields.io/badge/rich-13.7.1-brown.svg)](https://github.com/Textualize/rich?tab=readme-ov-file)
![Tests](https://img.shields.io/badge/coverage-0%25-red)

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
ncpare [-h] [-f--filter FILTER_NAME] [--common_pattern COMMON_PATTERN] [--variables VARIABLES [VARIABLES ...]] [--last_time_step] [-V] folder1 folder2

netCDF Comparison Tool

positional arguments:
  folder1               Path of first folder to compare
  folder2               Path of second folder to compare

options:
  -h, --help            show this help message and exit
  -f--filter FILTER_NAME
                        Filter to select files to compare. Examples: *.nc, *_grid_*
  --common_pattern COMMON_PATTERN
                        Common file pattern in two files to compareEs mfsX_date.nc and expX_date.nc -> date.nc is the common part
  -v, --variables VARIABLES [VARIABLES ...]
                        Variable to compare
  --last_time_step      If True, compare only the last time step available in each file
  -V, --version         Print version and exit

```

### Select Variables

It is possible to choose which parameter to compare:

```shell
ncpare folder1 folder2 -v "votemper" "vosaline"
```

![Variables](https://github.com/anto6715/ncCompare/raw/master/docs/variables.png)


### Filter files

As default **ncpare** read iterate over all files in **folder1** and expect to find them in **folder2**. Using filters,
it is possible to select only a subset of input files. For example:

```shell
ncpare folder1 folder2 -f "*_grid_T.nc"
```

### Compare files with different filenames

It is possible to compare two files also if the filenames are slightly different if they have a common pattern.
For example, if we have:

* `a/my-simu_19820101_grid_T.nc`
* `b/another-exp_19820101_grid_T.nc`

It is still possible to compare the file with:
```shell
ncpare folder1 folder2 --commom-pattern ".+_19820101_grid_T.nc"
```

Notice the regex syntax `.+` to match any pattern before `_19820101`

## Author

- Antonio Mariani (antonio.mariani@cmcc.it)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact
For any questions or suggestions, please open an issue on the project's GitHub repository.
