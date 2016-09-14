
import warnings
import keyword
import re

def make_valid_identifier(x):
	x = str(x)
	if keyword.iskeyword(x):
		y = "_"+x
		warnings.warn("name {0} is a python keyword, converting to {1}".format(x,y), stacklevel=2)
	else:
		y = x
	replacer = re.compile('(\W+)')
	y = replacer.sub("_", y)
	if not y.isidentifier():
		y = "_"+y
	if y!=x:
		warnings.warn("name {0} is not a valid python identifier, converting to {1}".format(x,y), stacklevel=2)
	return y

def parenthize(x, signs_qualify=False):
	"""Wrap a string in parenthesis if needed for unambiguous clarity.
	
	Parameters
	----------
	x : str
		The string to wrap
	signs_qualify : bool
		If True, a leading + or - on a number triggers the parenthesis (defaults False)
	
	"""
	x = str(x).strip()
	replacer = re.compile('(\W+)')
	if replacer.search(x):
		if signs_qualify:
			numeric = re.compile('^(([1-9][0-9]*\.?[0-9]*)|(\.[0-9]+))([Ee][+-]?[0-9]+)?\Z')
		else:
			numeric = re.compile('^[+-]?(([1-9][0-9]*\.?[0-9]*)|(\.[0-9]+))([Ee][+-]?[0-9]+)?\Z')
		if numeric.search(x):
			return x
		return "({})".format(x)
	return x