import logging
import sys

FILE_LOG_FORMAT = '%(name)s.%(levelname)s: %(message)s'
CONSOLE_LOG_FORMAT = '[%(asctime)s] %(name)s.%(levelname)s: %(message)s'

DEFAULT_LOG_LEVEL = logging.WARNING

from . import logger_name

def log_to_console(level=None):

	if level is None:
		level = DEFAULT_LOG_LEVEL

	logger = logging.getLogger(logger_name)

	# avoid creation of multiple stream handlers for logging to console
	for entry in logger.handlers:
		if (isinstance(entry, logging.StreamHandler)) and (entry.formatter._fmt == CONSOLE_LOG_FORMAT):
			return logger

	console_handler = logging.StreamHandler(stream=sys.stdout)
	console_handler.setLevel(level)
	console_handler.setFormatter(logging.Formatter(CONSOLE_LOG_FORMAT))
	logger.addHandler(console_handler)

	return logger


def log_to_file(filename, level=None):

	if level is None:
		level = DEFAULT_LOG_LEVEL

	logger = logging.getLogger(logger_name)

	# avoid creation of multiple file handlers for logging to the same file
	for entry in logger.handlers:
		if (isinstance(entry, logging.FileHandler)) and (entry.baseFilename == filename):
			return logger

	file_handler = logging.FileHandler(filename)
	file_handler.setLevel(level)
	file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT))
	logger.addHandler(file_handler)

	return logger


logger = log = log_to_console()
