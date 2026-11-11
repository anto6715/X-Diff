import logging.config

from xdiff import conf

DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "require_debug_false": {
            "()": "xdiff.utils.log.RequireDebugFalse",
        },
        "require_debug_true": {
            "()": "xdiff.utils.log.RequireDebugTrue",
        },
    },
    "formatters": {
        "my_fmt": {
            "format": "[%(asctime)s] - %(levelname)s: %(message)s",
        }
    },
    "handlers": {
        "console": {
            "level": "WARNING",
            "filters": ["require_debug_false"],
            "class": "logging.StreamHandler",
            "formatter": "my_fmt",
        },
        "console_with_debug": {
            "level": "DEBUG",
            "filters": ["require_debug_true"],
            "class": "logging.StreamHandler",
            "formatter": "my_fmt",
        },
    },
    "loggers": {
        "xdiff": {
            "handlers": ["console", "console_with_debug"],
            "level": "DEBUG",
        },
    },
}


class RequireDebugTrue(logging.Filter):
    def filter(self, record):
        return conf.DEBUG


class RequireDebugFalse(logging.Filter):
    def filter(self, record):
        return not conf.DEBUG


def configure_logging(logging_settings):
    """Apply the default logging configuration, then any custom settings on top."""
    logging.config.dictConfig(DEFAULT_LOGGING)
    if logging_settings:
        logging.config.dictConfig(logging_settings)
