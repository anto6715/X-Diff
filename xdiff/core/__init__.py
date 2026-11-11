"""Public core entrypoints for the package."""

import xdiff
from xdiff.core import main as main_module


def execute(**kwargs):
    xdiff.setup()
    return main_module.execute(**kwargs)
