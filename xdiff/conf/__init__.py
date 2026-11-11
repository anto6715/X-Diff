"""Public configuration exports."""

from xdiff.conf import global_settings as settings

DEFAULT_COMMON_PATTERN = settings.DEFAULT_COMMON_PATTERN
DEFAULT_NAME_TO_COMPARE = settings.DEFAULT_NAME_TO_COMPARE
DEFAULT_VARIABLES_TO_CHECK = settings.DEFAULT_VARIABLES_TO_CHECK
LOGGING = settings.LOGGING
DEBUG = settings.DEBUG

__all__ = [
    "settings",
    "DEFAULT_COMMON_PATTERN",
    "DEFAULT_NAME_TO_COMPARE",
    "DEFAULT_VARIABLES_TO_CHECK",
    "LOGGING",
    "DEBUG",
]
