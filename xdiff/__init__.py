from xdiff import conf as settings
from xdiff.utils.log import configure_logging


def setup():
    configure_logging(settings.LOGGING_CONFIG, settings.LOGGING)
