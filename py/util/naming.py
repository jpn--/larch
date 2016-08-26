
import warnings
import keyword


def make_valid_identifier(x):
	x = str(x)
	if keyword.iskeyword(x):
		y = "_"+x
		warnings.warn("name {0} is a python keyword, converting to {1}".format(x,y))
	else:
		y = x
	if '.' in y:
		y = y.replace('.','_')
	if ':' in y:
		y = y.replace(':','_')
	if ',' in y:
		y = y.replace(',','_')
	if not y.isidentifier():
		y = "_"+y
	if y!=x:
		warnings.warn("name {0} is not a valid python identifier, converting to {1}".format(x,y))
	return y