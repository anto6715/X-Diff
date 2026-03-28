"""
Default X-Diff settings.
"""

########################
# SETTINGS
########################

DEFAULT_MAXDEPTH = 1  # negative value remove the limit on
DEFAULT_NAME_TO_COMPARE = "*.nc"
DEFAULT_VARIABLES_TO_CHECK = object()
DEFAULT_COMMON_PATTERN = None
DTYPE_NOT_CHECKED = ["S8", "S1", "O"]  # S8|S1:char, O: string
TIME_DTYPE = ["datetime64[ns]", "<M8[ns]"]

########################
# LOG
########################

# The callable to use to configure logging
LOGGING_CONFIG = "logging.config.dictConfig"

# Custom logging configuration.
LOGGING = {}

DEBUG = False
