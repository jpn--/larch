
import numpy
import pandas
import ast
import re


### CATEGORICAL / ONE-HOT EXPANSION ###

def _to_categorical(y, num_classes=None, dtype='float32'):
	"""Converts a class vector (integers) to binary class matrix.

	E.g. for use with categorical_crossentropy.

	Parameters
	----------
	y: class vector to be converted into a matrix
		(integers from 0 to num_classes).
	num_classes: total number of classes.
	dtype: The data type expected by the input, as a string
		(`float32`, `float64`, `int32`...)

	Returns
	-------
	A binary matrix representation of the input. The classes axis
		is placed last.

	Notes
	-----
	This function is from keras.utils
	"""
	y = numpy.array(y, dtype='int')
	input_shape = y.shape
	if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
		input_shape = tuple(input_shape[:-1])
	y = y.ravel()
	if not num_classes:
		num_classes = numpy.max(y) + 1
	n = y.shape[0]
	categorical = numpy.zeros((n, num_classes), dtype=dtype)
	categorical[numpy.arange(n), y] = 1
	output_shape = input_shape + (num_classes,)
	categorical = numpy.reshape(categorical, output_shape)
	return categorical

def to_categorical(arr, index=None, dtype='float32'):

	if isinstance(arr, (pandas.DataFrame, pandas.Series)) and index is None:
		index = arr.index

	codes, positions = numpy.unique(arr, return_inverse=True)
	return pandas.DataFrame(
		data=_to_categorical(positions, dtype=dtype),
		columns=codes,
		index=index,
	)

def categorical_expansion(s, column=None, inplace=False, drop=False):
	"""
	Expand a pandas Series into a DataFrame containing a categorical dummy variables.

	This is sometimes called "one-hot" encoding.

	Parameters
	----------
	s : pandas.Series or pandas.DataFrame
		The input data
	column : str, optional
		If `s` is given as a DataFrame, expand this column
	inplace : bool, default False
		If true, add the result directly as new columns in `s`.
	drop : bool, default False
		If true, drop the existing column from `s`. Has no effect if
		`inplace` is not true.

	Returns
	-------
	pandas.DataFrame
		Only if inplace is false.
	"""
	if isinstance(s, pandas.DataFrame):
		input = s
		if column is not None and column not in s.columns:
			raise KeyError(f'key not found "{column}"')
		if len(s.columns) == 1:
			s = s.iloc[:, 0]
			column = s.columns[0]
		elif column is not None:
			s = s.loc[:, column]
	else:
		input = None

	onehot = to_categorical(s)
	onehot.columns = [f'{s.name}=={_}' for _ in onehot.columns]
	if inplace and input is not None:
		input[onehot.columns] = onehot
		if drop:
			input.drop([column], axis=1)
	else:
		return onehot

one_hot_expansion = categorical_expansion


### FOURIER EXPANSION ###

def fourier_expansion_names(basename, length=4):
	"""
	Get the names of items for a fourier series expansion.

	Parameters
	----------
	basename : str
		The input data name
	length : int
		Length of expansion series

	Returns
	-------
	list of str
	"""
	columns = []
	for i in range(length):
		func = 'cos' if i % 2 else 'sin'
		mult = ((i // 2) + 1) * 2
		columns.append(f'{func}({basename}*{mult}*pi)')
	return columns


def fourier_expansion(s, length=4, column=None, inplace=False):
	"""
	Expand a pandas Series into a DataFrame containing a fourier series.

	Parameters
	----------
	s : pandas.Series or pandas.DataFrame
		The input data
	length : int
		Length of expansion series
	column : str, optional
		If `s` is given as a DataFrame, use this column

	Returns
	-------
	pandas.DataFrame
	"""
	if isinstance(s, pandas.DataFrame):
		input = s
		if len(s.columns) == 1:
			s = s.iloc[:, 0]
		elif column is not None:
			s = s.loc[:, column]
	else:
		input = None
	columns = fourier_expansion_names(s.name, length=length)
	df = pandas.DataFrame(
		data=0,
		index=s.index,
		columns=columns,
	)
	for i, col in enumerate(columns):
		func = numpy.cos if i % 2 else numpy.sin
		mult = ((i // 2) + 1) * 2
		df.iloc[:, i] = func(s * mult * numpy.pi)
	if inplace and input is not None:
		input[columns] = df
	else:
		return df


### PIECEWISE LINEAR ###

def piece(x, low_bound, high_bound):
	if low_bound is None:
		if high_bound is None:
			return x
		else:
			return numpy.fmin(x, high_bound)
	else:
		if high_bound is None:
			return numpy.fmax(x-low_bound, 0)
		else:
			return numpy.fmax(numpy.fmin(x, high_bound)-low_bound, 0)


def parse_piece(s):
	if s is not None:
		float_regex = '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
		z = re.match(f"^piece\((.*),({float_regex}|None),({float_regex}|None)\)$", s)
		if z:
			# data is exactly and only a piecewise part
			base_def = z.group(1)
			lowbound = ast.literal_eval(z.group(2))
			highbound = ast.literal_eval(z.group(4))
			return base_def, lowbound, highbound, True
		z = re.match(f".*piece\((.*),({float_regex}|None),({float_regex}|None)\)", s)
		if z:
			# data is a piecewise part possibly interacted with something else
			base_def = z.group(1)
			lowbound = ast.literal_eval(z.group(2))
			highbound = ast.literal_eval(z.group(4))
			return base_def, lowbound, highbound, False
	return None, None, None, None

def piecewise_linear(x, p=None, breaks=None):
	from ..roles import P,X

	if p is None:
		p = x

	# Flip x and p if given backwards
	if isinstance(x, P) and isinstance(p, X):
		p,x = x,p

	if breaks is None:
		raise ValueError('missing required argument `breaks`')

	xs = piecewise_linear_data_names(x, breaks)
	ps = piecewise_linear_parameter_names(p, breaks)

	return sum(X(x_) * P(p_) for x_,p_ in zip(xs,ps))


def piecewise_linear_parameter_names(basename, breaks):
	from ..roles import P,X

	lex = '②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'

	# first leg
	b = breaks[0]
	z = [P(f"{basename} ① up to {b}")]
	# middle legs
	for i in range(len(breaks) - 1):
		b0 = breaks[i]
		b1 = breaks[i + 1]
		z += [P(f"{basename} {lex[i]} {b0} to {b1}")]
	# last leg
	i = len(breaks) - 1
	b0 = breaks[i]
	z += [P(f"{basename} {lex[i]} over {b0}")]
	return z


def piecewise_linear_data_names(basename, breaks):
	# first leg
	b = breaks[0]
	z = [f"piece({basename},None,{b})"]
	# middle legs
	for i in range(len(breaks) - 1):
		b0 = breaks[i]
		b1 = breaks[i + 1]
		z += [f"piece({basename},{b0},{b1})"]
	# last leg
	i = len(breaks) - 1
	b0 = breaks[i]
	z += [f"piece({basename},{b0},None)"]
	return z


def piecewise_expansion(s, breaks, column=None, inplace=False):
	"""
	Expand a pandas Series into a DataFrame containing a piecewise linear series.

	Parameters
	----------
	s : pandas.Series or pandas.DataFrame
		The input data
	breaks : list
		Piecewise linear articulation breakpoints
	column : str, optional
		If `s` is given as a DataFrame, use this column

	Returns
	-------
	pandas.DataFrame
	"""
	if isinstance(s, pandas.DataFrame):
		input = s
		if len(s.columns) == 1:
			s = s.iloc[:, 0]
		elif column is not None:
			s = s.loc[:, column]
	else:
		input = None

	columns = piecewise_linear_data_names(s.name, breaks=breaks)
	df = pandas.DataFrame(
		data=0,
		index=s.index,
		columns=columns,
	)

	# first leg
	b = breaks[0]
	df.iloc[:, 0] = piece(s, None, b)

	# middle legs
	for i in range(len(breaks) - 1):
		b0 = breaks[i]
		b1 = breaks[i + 1]
		df.iloc[:, i + 1] = piece(s, b0, b1)
	# last leg
	i = len(breaks) - 1
	b0 = breaks[i]
	df.iloc[:, i + 1] = piece(s, b0, None)

	if inplace and input is not None:
		input[columns] = df
	else:
		return df


### IDCE MANIPULATIONS ###

def _nonzero_minimum_rescale_slower(x):
	x_ = x[x!=0]
	if x_.size == 0:
		s = 1
	else:
		s = numpy.min(x[x!=0])
	return x/s

def _nonzero_minimum_rescale(x):
	s = numpy.ma.masked_equal(x, 0.0, copy=False).min()
	if numpy.isnan(s) or s==0:
		s = 1
	return x/s


def ratio_min(s, column=None, inplace=False, result_name=None):
	"""
	Modify a pandas Series or DataFrame in `idce` format to scale within cases relative to the minimum value.

	Parameters
	----------
	s : pandas.Series or pandas.DataFrame
		The input data
	column : str, optional
		If `s` is given as a DataFrame, use this column
	inplace : bool, default False
		modify `s` in-place?
	result_name : str, optional
		If `s` is given as a DataFrame, and `inplace` is True, write to this column

	Returns
	-------
	pandas.DataFrame
	"""
	index = s.index
	if isinstance(s, pandas.DataFrame):
		input = s
		if len(s.columns) == 1:
			s = s.iloc[:, 0]
		elif column is not None:
			s = s.loc[:, column]
	else:
		input = None

	divisor = s.groupby(index.labels[0]).transform(numpy.min)
	divisor[divisor==0] = 1.0
	scaled = s/divisor

	# scaled = s.groupby(index.labels[0]).transform(_nonzero_minimum_rescale)

	if result_name is None and column is not None:
		result_name = f'ratio_min({column})'

	if inplace and input is not None and result_name is not None:
		input[result_name] = scaled
	else:
		return scaled
