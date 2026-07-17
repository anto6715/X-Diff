from __future__ import annotations

import sys

from xdiff import conf as settings
from xdiff.utils.log import configure_logging

MIN_SUPPORTED_PYTHON = (3, 11)
MAX_SUPPORTED_PYTHON = (3, 14)


def validate_runtime() -> None:
    version = sys.version_info[:2]
    if MIN_SUPPORTED_PYTHON <= version <= MAX_SUPPORTED_PYTHON:
        return

    raise RuntimeError(
        f"xdiff supports Python {MIN_SUPPORTED_PYTHON[0]}.{MIN_SUPPORTED_PYTHON[1]} through "
        f"{MAX_SUPPORTED_PYTHON[0]}.{MAX_SUPPORTED_PYTHON[1]}. "
        f"The current interpreter is Python {version[0]}.{version[1]}."
    )


def setup():
    validate_runtime()
    configure_logging(settings.LOGGING_CONFIG, settings.LOGGING)
