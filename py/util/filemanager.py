__all__ = ['open_stack', 'next_stack', 'fileopen', 'filenext']

import os
import os.path
from .temporaryfile import TemporaryFile
import types

def filename_split(filename):
	pathlocation, basefile = os.path.split(filename)
	basefile_list = basefile.split(".")
	if len(basefile_list)>1:
		basename = ".".join(basefile_list[:-1])
		extension = "." + basefile_list[-1]
	else:
		basename = basefile_list[0]
		extension = ""
	return (pathlocation, basename, extension)
	
def filename_fuse(pathlocation, basename, extension):
	x = os.path.join(pathlocation, basename)
	if extension != "": x+="."+extension
	return x

def rotate_file(filename, format="%(basename)s.%(number)03i%(extension)s"):
	if os.path.exists(filename):		
		pathlocation, basename, extension = filename_split(filename)
		fn = lambda n: os.path.join(pathlocation,format%{'basename':basename, 'extension':extension, 'number':n})
		n = 1
		while os.path.exists(fn(n)):
			n += 1
		while n > 1:
			os.rename(fn(n-1),fn(n))
			n -= 1
		os.rename(filename,fn(1))
	else:
		from ..core import LarchError
		raise LarchError("File %s does not exist"%filename)

def new_stack_file(filename, format="%(basename)s.%(number)03i%(extension)s", *, always_number=False):
	import os.path
	if os.path.exists(filename):		
		pathlocation, basename, extension = filename_split(filename)
		fn = lambda n: os.path.join(pathlocation,format%{'basename':basename, 'extension':extension, 'number':n})
		n = 1
		while os.path.exists(fn(n)):
			n += 1
		return fn(n)
	else:
		return filename

def top_stack_file(filename, format="%(basename)s.%(number)03i%(extension)s"):
	import os.path
	if os.path.exists(filename):		
		pathlocation, basename, extension = filename_split(filename)
		fn = lambda n: os.path.join(pathlocation,format%{'basename':basename, 'extension':extension, 'number':n})
		n = 1
		if not os.path.exists(fn(n)):
			return filename
		while os.path.exists(fn(n)):
			n += 1
		return fn(n-1)
	else:
		from ..core import LarchError
		raise LarchError("File %s does not exist"%filename)


def next_stack(filename, format="{basename:s}.{number:03d}{extension:s}", suffix=None, plus=0, allow_natural=False):
	"""Finds the next file name in this stack that does not yet exist.
	
	Parameters
	----------
	filename : str or None
		The base file name to use for this stack.  New files would have a number
		appended after the basename but before the dot extension.  For example,
		if the filename is "/tmp/boo.txt", the first file created will be named
		"/tmp/boo.001.txt".  If None, then a temporary file is created instead.
	
	
	Other Parameters
	----------------
	suffix : str, optional
		If given, use this file extension instead of any extension given in the filename
		argument.  The unsual use case for this parameter is when filename is None,
		and a temporary file of a particular kind is desired.
	format : str, optional
		If given, use this format string to generate new stack file names in a
		different format.
	plus : int, optional
		If given, increase the returned filenumber by this amount more than what
		is needed to generate a new file.  This can be useful with pytables.
	allow_natural : bool
		If true, this function will return the unedited	`filename` parameter
		if that file does not already exist. Otherwise will always have a 
		number appended to the name.
		
	"""
	if allow_natural and not os.path.exists(filename):
		return filename
	pathlocation, basename, extension = filename_split(filename)
	if suffix is not None:
		extension = "."+suffix
	fn = lambda n: os.path.join(pathlocation,format.format(basename=basename, extension=extension, number=n))
	n = 1
	while os.path.exists(fn(n)):
		n += 1
	return fn(n+plus)


default_webbrowser = 'chrome'
from .temporaryfile import _open_in_chrome_or_something

def open_stack(filename=None, *arg, format="{basename:s}.{number:03d}{extension:s}", suffix=None, **kwarg):
	"""Opens the next file in this stack for writing.
	
	Parameters
	----------
	filename : str, optional
		The base file name to use for this stack.  New files will have a number
		appended after the basename but before the dot extension.  For example,
		if the filename is "/tmp/boo.txt", the first file created will be
		named "/tmp/boo.001.txt".  If None, which is the default, then a temporary
		file is created instead.
	
	
	Other Parameters
	----------------
	suffix : str, optional
		If given, use this file extension instead of any extension given in the filename
		argument.  The unsual use case for this parameter is when filename is None,
		and a temporary file of a particular kind is desired.
	format : str, optional
		If given, use this format string to generate new stack file names in a
		different format.
		
	Notes
	-----
	Other positional and keyword arguments are passed through to the normal Python :func:`open`
	function.
	
	The returned file-like object also has an extra `view` method, which will open
	the file in the default webbrowser.

	"""
	if filename is None:
		f = TemporaryFile(suffix=suffix if suffix is not None else '', **kwarg)
	else:
		f = open( next_stack(filename, format, suffix) , *arg, **kwarg)
	if default_webbrowser.lower() == 'chrome':
		f.view = types.MethodType( lambda self: _open_in_chrome_or_something('file://'+os.path.realpath(self.name)), f )
	else:
		f.view = types.MethodType( lambda self: webbrowser.open('file://'+os.path.realpath(self.name)), f )
	return f

fileopen = open_stack
filenext = next_stack