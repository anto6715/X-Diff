"""Backward-compatible exports for netCDF comparison helpers."""

from importlib import import_module

__all__ = [
    "compare",
    "compare_datasets",
    "compare_files",
    "compare_variables",
    "compute_relative_error",
    "find_time_dims_name",
    "get_dataset_variables",
    "select_last_time_step",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)

    module = import_module("nccompare.compare.ncdiff")
    return getattr(module, name)
