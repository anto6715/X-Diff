"""Public core entrypoints for the package."""

import nccompare

from nccompare.core import main as main_module


def execute(**kwargs):
    nccompare.setup()
    get_version = kwargs.pop("get_version", False)

    if get_version:
        exit(0)

    return main_module.execute(**kwargs)
