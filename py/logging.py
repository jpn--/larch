#
#  Copyright 2007-2016 Jeffrey Newman
#
#  This file is part of Larch.
#
#  Larch is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Larch is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Larch.  If not, see <http://www.gnu.org/licenses/>.
#

import logging
import logging.handlers
import sys
import traceback

from logging import DEBUG, INFO, ERROR, WARNING, CRITICAL

_base_log_class = logging.getLoggerClass()

_time_format = '%I:%M:%S %p'
_mess_format = '[%(asctime)11s]%(name)s: %(message)s'

def scribe_to_stream(stream=sys.stdout, scribe=None):
	if scribe is None:
		s = logging.getLogger("")
	else:
		s = getScriber(scribe)
	h = logging.StreamHandler(stream)
	f = logging.Formatter(fmt=_mess_format,datefmt=_time_format)
	h.setFormatter(f)
	s.addHandler(h)
	s.critical("Connected log to stream %s",str(stream))

class Scriber(_base_log_class):
	def __init__(self,name):
		_base_log_class.__init__(self,name)
		self.default_level = logging.INFO
	def __call__(self, x, *args, **kwargs):
		if 'level' in kwargs:
			level = kwargs['level']
			del kwargs['level']
		else:
			level = self.default_level
		self.log(level, x, *args, **kwargs)
	def write(self, x, *args):
		self.__call__(x,*args)
	def write(self, x, *args):
		try:
			self.pending += x.format(*args)
		except AttributeError:
			self.pending = x.format(*args)
		if "\n" in self.pending:		
			if self.pending=="\n":
				self.pending=""
			if self.pending[-1]=="\n":
				self.pending = self.pending[:-1]
			self.log(50, self.pending)
			del self.pending

logging.setLoggerClass(Scriber)

def getScriber(name="",*args,**kwargs):
	if name=="" or name.lower()=="larch":
		name = "larch"
	elif len(name)>6 and name[0:6]=="larch.":
		pass
	else:
		name = "larch."+name
	return logging.getLogger(name,*args,**kwargs)

getLogger = getScriber

def setLevel(x):
	Scriber.root.setLevel(x)

Scrb = getScriber()





import collections
try:
	levels = collections.OrderedDict()
except:
	levels = {}
levels['DEBUG'] = 10
levels['INFO'] = 20
levels['WARNING'] = 30
levels['ERROR'] = 40
levels['CRITICAL'] = 50

revLevel = {}
revLevel[10]='DEBUG'
revLevel[20]='INFO'
revLevel[30]='WARNING'
revLevel[40]='ERROR'
revLevel[50]='CRITICAL'


def scribe_level_number(lvl):
	if isinstance(lvl, str):
		return levels[lvl]
	return lvl

def scribe_level_name(lvl):
	if isinstance(lvl, str):
		return lvl
	return revLevel[lvl]



def scribe_level(newLevel, silently=False):
	if isinstance(newLevel, str):
		newLevel = levels[newLevel]
	Scrb.setLevel(newLevel)
	if not silently:
		if newLevel in revLevel:
			Scrb.info('Changing log level to %s',revLevel[newLevel])
		else:
			Scrb.info('Changing log level to %i',newLevel)

def check_scribe_level():
	return Scrb.getEffectiveLevel()

try:
	fileHandler
except NameError:
	fileHandler = None

def default_formatter():
	return logging.Formatter(fmt=_mess_format,datefmt=_time_format)

def scribe_to_file(filename, residual=None, overwrite=False, *, fmt=None, datefmt=None):
	global fileHandler
	if fileHandler:
		fileHandler.flush()
		Scrb.removeHandler(fileHandler)
		fileHandler.close()
		fileHandler = None
	if overwrite:
		mode = 'w'
	else:
		mode = 'a'
	if filename is None or filename=="": return
	if residual:
		f = open(filename, mode)
		f.write(residual)
		f.close()
		mode = 'a'
	fileHandler = logging.FileHandler(filename, mode)
	if fmt is None:
		fmt=_mess_format
	if datefmt is None:
		datefmt=_time_format
	fileHandler.setFormatter(logging.Formatter(fmt=fmt,datefmt=datefmt))
	Scrb.addHandler(fileHandler)
	Scrb.critical("Connected log to %s",filename)




def spew(level=10):
	scribe_to_stream()
	setLevel(level)
	return logging.getLogger("")


_easy_logger = None

def easy(level=-1, label="", *, filename=None, file_fmt='[%(name)s] %(message)s'):
	global _easy_logger
	if file_fmt is None:
		file_fmt = _mess_format
	if filename:
		scribe_to_file(filename, fmt=file_fmt)
	if isinstance(level, str):
		label_ = level
		level = label if isinstance(label, int) else -1
		label = label_
	if isinstance(label, int):
		level_ = label
		label = level if isinstance(level, str) else ""
		level = level_
	if _easy_logger is None:
		scribe_to_stream()
		_easy_logger = 1
	if level>0: setLevel(level)
	return getScriber(label).critical

def easy_debug(label=""):
	global _easy_logger
	if _easy_logger is None:
		scribe_to_stream()
		_easy_logger = 1
	setLevel(10)
	return logging.getLogger(label).debug

def easy_logging_active():
	global _easy_logger
	if _easy_logger is None:
		return False
	else:
		return True
		
