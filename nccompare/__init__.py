from nccompare import conf as settings
from nccompare.utils.log import configure_logging


def setup():
    configure_logging(settings.LOGGING_CONFIG, settings.LOGGING)
