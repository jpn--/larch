import logging
import sys
from . import logger_name
from contextlib import contextmanager
import time
from ..util.timesize import timesize_stack

FILE_LOG_FORMAT = '%(name)s.%(levelname)s: %(message)s'
CONSOLE_LOG_FORMAT = '[%(asctime)s] %(name)s.%(levelname)s: %(message)s'

DEFAULT_LOG_LEVEL = logging.DEBUG


def log_to_console(level=None):

	if level is None:
		level = DEFAULT_LOG_LEVEL

	logger = logging.getLogger(logger_name)

	# avoid creation of multiple stream handlers for logging to console
	for entry in logger.handlers:
		if (isinstance(entry, logging.StreamHandler)) and (entry.formatter._fmt == CONSOLE_LOG_FORMAT):
			return logger

	console_handler = logging.StreamHandler(stream=sys.stderr)
	console_handler.setLevel(level)
	console_handler.setFormatter(logging.Formatter(CONSOLE_LOG_FORMAT))
	logger.addHandler(console_handler)
	if level < logger.getEffectiveLevel():
		logger.setLevel(level)

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


@contextmanager
def timing_log(label=''):
	start_time = time.time()
	log.critical(f"<TIME BEGINS> {label}")
	try:
		yield
	except:
		log.critical(f"<TIME ERROR!> {label} <{timesize_stack(time.time() - start_time)}>")
		raise
	else:
		log.critical(f"< TIME ENDS > {label} <{timesize_stack(time.time() - start_time)}>")


class TimingLog:

	def __init__(self, label='', log=None, level=50):
		global logger
		if log is None:
			log = logger
		self.label = label
		self.log = log
		self.level = level
		self.split_time = None
		self.current_task = ''

	def __enter__(self):
		self.start_time = time.time()
		self.log.log(self.level, f"<BEGIN> {self.label}")
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		now = time.time()
		if self.split_time is not None:
			self.log.log(self.level, f"<SPLIT> {self.label} / Final <{timesize_stack(now - self.split_time)}>")
		if exc_type is None:
			self.log.log(self.level, f"<-END-> {self.label} <{timesize_stack(now - self.start_time)}>")
		else:
			self.log.log(self.level, f"<ERROR> {self.label} <{timesize_stack(now - self.start_time)}>")

	def split(self, note=''):
		if self.split_time is None:
			self.split_time = self.start_time
		now = time.time()
		if note:
			note = " / " + note
		self.log.log(self.level, f"<SPLIT> {self.label}{note} <{timesize_stack(now - self.split_time)}>")
		self.split_time = now


