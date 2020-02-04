

LIGHT_CONSOLE_LOG_FORMAT = '%(message)s'

from .normal import *

# Remove timestamps from console stream

for entry in logger.handlers:
	if (isinstance(entry, logging.StreamHandler)) and (entry.formatter._fmt == CONSOLE_LOG_FORMAT):
		entry.setFormatter(logging.Formatter(LIGHT_CONSOLE_LOG_FORMAT))

