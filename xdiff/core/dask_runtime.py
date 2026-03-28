"""Helpers for optional Dask execution.

The application still defaults to serial execution. This module is only used
when the caller explicitly selects a Dask-backed execution mode, which keeps
the regular CLI startup path lightweight.
"""

from __future__ import annotations

import logging
import os

from contextlib import contextmanager
from importlib import import_module
from typing import TYPE_CHECKING, Iterator

from xdiff.model.request import CompareRequest

if TYPE_CHECKING:
    from distributed import Client

logger = logging.getLogger("xdiff")


def log_local_worker_advisories(request: CompareRequest) -> None:
    """Report obviously poor local-worker counts without blocking the run."""
    if request.dask_workers is None:
        return

    visible_cpus = os.cpu_count()
    if visible_cpus is not None and request.dask_workers > visible_cpus:
        logger.warning(
            "Requested %s Dask worker(s) but only %s CPU(s) are visible on this node.",
            request.dask_workers,
            visible_cpus,
        )


def log_file_mode_advisories(request: CompareRequest, comparable_file_pairs: int) -> None:
    """Report file-mode specific worker advisories without blocking the run."""
    log_local_worker_advisories(request)
    if request.dask_workers is None:
        return

    if comparable_file_pairs < request.dask_workers:
        logger.warning(
            "Requested %s Dask worker(s) for %s comparable file pair(s); some workers will remain idle.",
            request.dask_workers,
            comparable_file_pairs,
        )


@contextmanager
def client_from_request(request: CompareRequest) -> Iterator["Client"]:
    """Attach to an external scheduler or create a local cluster for this request."""
    distributed = _load_distributed()
    cluster = None

    if request.uses_external_dask_scheduler:
        client = _connect_to_scheduler(request, distributed.Client)
    else:
        # One worker process per requested slot is the safest default for Dask-backed netCDF I/O.
        cluster = distributed.LocalCluster(
            n_workers=request.dask_workers,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
        )
        client = distributed.Client(cluster)

    try:
        yield client
    finally:
        client.close()
        if cluster is not None:
            cluster.close()


def _connect_to_scheduler(request: CompareRequest, client_type):
    if request.dask_scheduler_file is not None:
        return client_type(scheduler_file=str(request.dask_scheduler_file))
    return client_type(request.dask_scheduler)


def _load_distributed():
    try:
        return import_module("distributed")
    except ImportError as exc:
        raise RuntimeError(
            "Dask support requires the optional 'distributed' dependency. "
            "Install xdiff with the 'dask' extra or add 'distributed' to your environment."
        ) from exc
