# Dask Usage

`xdiff` runs serially by default. Dask is opt-in: supply a backend option and `xdiff` parallelizes the comparison by submitting one Dask task per comparable file pair. There is no separate mode switch — the presence of a backend option is what enables Dask.

## Install

Dask dependencies are part of the default `xdiff` environment. Install the project with uv:

```shell
uv sync
```

For development, include the dev dependency group:

```shell
uv sync --group dev
```

## Enabling Dask

With no backend option, `xdiff` runs serially. Supply exactly one of the following to run in parallel through Dask:

- `--dask-workers N` to create a local cluster on the current node
- `--dask-scheduler ...` to attach to an existing scheduler by address
- `--dask-scheduler-file ...` to attach to an existing scheduler file

`--dask-workers` and an external scheduler are mutually exclusive, as are the two scheduler options.

## Local Cluster

Use a local cluster when you are running on a workstation or inside a single allocated HPC compute node and you want `xdiff` to use that node directly.

```shell
uv run xdiff dirs a b --dask-workers 32
```

```shell
uv run xdiff files reference.nc comparison.nc --dask-workers 32
```

The local-cluster path creates one Dask worker process per requested worker. This is a conservative default for file-level netCDF I/O because it avoids thread-safety surprises and maps cleanly to HPC allocations.

## External Scheduler

Use an external scheduler when you already manage the cluster outside `xdiff`, for example through `dask-scheduler`, `dask-jobqueue`, or a site-specific HPC workflow.

Attach by scheduler address:

```shell
uv run xdiff dirs a b --dask-scheduler tcp://scheduler.example:8786
```

Attach by scheduler file:

```shell
uv run xdiff dirs a b --dask-scheduler-file scheduler.json
```

## HPC Notes

- Workers must be able to open the same file paths seen by the submitting process. Shared filesystems are the simplest setup.
- `xdiff` does not provision a multi-node cluster for you. It can either create a local cluster on the current node or attach to an existing external scheduler.
- For file-level mode, start with the number of worker processes that matches your allocated cores, then reduce it if the shared filesystem becomes the bottleneck.

## Advisories

`xdiff` reports obviously poor local-worker configurations as warnings, for example:

- more requested workers than comparable file pairs
- more requested workers than visible CPUs on the current node

These are informational only. HPC users may still choose deliberate oversubscription for their own workflows.
