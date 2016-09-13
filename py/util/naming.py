
import warnings
import keyword
import re

def make_valid_identifier(x):
	x = str(x)
	if keyword.iskeyword(x):
		y = "_"+x
		warnings.warn("name {0} is a python keyword, converting to {1}".format(x,y))
	else:
		y = x
	replacer = re.compile('(\W+)')
	y = replacer.sub("_", y)
	if not y.isidentifier():
		y = "_"+y
	if y!=x:
		warnings.warn("name {0} is not a valid python identifier, converting to {1}".format(x,y))
	return y