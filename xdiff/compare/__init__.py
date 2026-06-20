"""Public comparison exports."""

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

_LAZY_EXPORTS = {
    "compare": ("xdiff.compare.ncdiff", "compare"),
    "compare_datasets": ("xdiff.comparators.netcdf", "compare_datasets"),
    "compare_files": ("xdiff.comparators.netcdf", "compare_files"),
    "compare_variables": ("xdiff.comparators.netcdf", "compare_variables"),
    "compute_relative_error": ("xdiff.comparators.netcdf", "compute_relative_error"),
    "find_time_dims_name": ("xdiff.comparators.netcdf", "find_time_dims_name"),
    "get_dataset_variables": ("xdiff.comparators.netcdf", "get_dataset_variables"),
    "select_last_time_step": ("xdiff.comparators.netcdf", "select_last_time_step"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
