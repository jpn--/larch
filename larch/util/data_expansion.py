
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

def categorical_compression(df, columns=None, inplace=False, drop=False, name=None):
	"""
	Compress a one-hot encoded part of a pandas DataFrame into a categorical variable.

	Parameters
	----------
	s : pandas.DataFrame
		The input data
	columns : Sequence[str], optional
		Use these columns of `df`, or all columns if not given.
	inplace : bool, default False
		If true, add the result directly as new columns in `df`.
	drop : bool, default False
		If true, drop the existing columns from `df`. Has no effect if
		`inplace` is not true.

	Returns
	-------
	pandas.Series
		Only if inplace is false.
	"""
	input = df
	if columns is None:
		columns = df.columns
	onehot = df[columns]
	result = pandas.Series(
		numpy.full(onehot.index.shape, numpy.nan),
		dtype=pandas.CategoricalDtype(columns),
		index=onehot.index,
	)
	for i in onehot.columns:
		result.loc[onehot[i]==1] = i
	if inplace and name is not None:
		input[name] = result
		if drop:
			input.drop(columns, axis=1)
	else:
		return result

one_hot_compression = categorical_compression


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
	"""
	Clip the values in an array.

	This function differs from the usual `numpy.clip`
	in that the result is shifted by `low_bound` if it is
	given, so that the valid result range is a contiguous
	block of non-negative values starting from 0.

	Parameters
	----------
	x : array-like
		Array containing elements to clip
	low_bound, high_bound : scalar, array-like, or None
		Minimum and maximum values. If None, clipping is not
		performed on the corresponding edge. Both may be
		None, which results in a noop. Both are broadcast
		against `x`.

	Returns
	-------
	clipped_array : array-like
	"""
	if low_bound is None:
		if high_bound is None:
			return x
		else:
			return numpy.minimum(x, high_bound)
	else:
		if high_bound is None:
			return numpy.maximum(x-low_bound, 0)
		else:
			return numpy.clip(x, low_bound, high_bound)-low_bound


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
	from ..model import linear

	if p is None:
		p = x

	# Flip x and p if given backwards
	if isinstance(x, linear.ParameterRef_C) and isinstance(p, linear.DataRef_C):
		p,x = x,p

	if breaks is None:
		raise ValueError('missing required argument `breaks`')

	xs = piecewise_linear_data_names(x, breaks)
	ps = piecewise_linear_parameter_names(p, breaks)

	return sum(X(x_) * P(p_) for x_,p_ in zip(xs,ps))


def piecewise_linear_parameter_names(basename, breaks):
	from ..roles import P,X
	width = 1
	if len(breaks) > 9:
		width = 2
	# first leg
	b = breaks[0]
	z = [P(f"{basename}[{1:{width}d}]: up to {b}")]
	# middle legs
	for i in range(len(breaks) - 1):
		b0 = breaks[i]
		b1 = breaks[i + 1]
		z += [P(f"{basename}[{i+2:{width}d}]: {b0} to {b1}")]
	# last leg
	i = len(breaks) - 1
	b0 = breaks[i]
	z += [P(f"{basename}[{i+2:{width}d}]: over {b0}")]
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


def infer_breaks(fun):
	"""
	Infer break points from a piecewise_linear LinearFunction.

	Parameters
	----------
	fun : LinearFunction_C
	"""
	m = re.compile('piece\(.*,(.*),(.*)\)')
	_0 = m.match(fun[0].data)
	if _0.group(1) != 'None':
		raise NotImplementedError('Bounded Minimum')
	breaks = [ast.literal_eval(_0.group(2))]
	for i in fun[1:-1]:
		_n = m.match(i.data)
		if breaks[-1] != ast.literal_eval(_n.group(1)):
			raise NotImplementedError('Diconnected')
		breaks.append(ast.literal_eval(_n.group(2)))
	_n = m.match(fun[-1].data)
	if breaks[-1] != ast.literal_eval(_n.group(1)):
		raise NotImplementedError('Diconnected Top')
	if _n.group(2) != 'None':
		raise NotImplementedError('Bounded Maximum')
	return breaks

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

	from ..model import linear

	if isinstance(breaks, linear.LinearFunction_C):
		breaks = infer_breaks(breaks)

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
		If `s` is given as a DataFrame, and `inplace` is True, write to this column.
		When not given, the result is named "ratio_min(<column>)"

	Returns
	-------
	pandas.DataFrame

	Raises
	------
	ValueError
		The input argument `s` is not in idce format (i.e., it does not have
		a MultiIndex as the index or that MultiIndex does not have 2 levels.)
	"""
	index = s.index

	if not isinstance(s.index, pandas.MultiIndex) or len(s.index.levels) != 2:
		raise ValueError('`s` must be in idce format, with a 2-level multi-index.')

	if column is None and isinstance(s, pandas.Series):
		column = s.name

	if isinstance(s, pandas.DataFrame):
		input = s
		if len(s.columns) == 1:
			s = s.iloc[:, 0]
		elif column is not None:
			s = s.loc[:, column]
	else:
		input = None

	divisor = s.groupby(index.codes[0]).transform(numpy.min)
	divisor[divisor==0] = 1.0
	scaled = s/divisor

	# scaled = s.groupby(index.labels[0]).transform(_nonzero_minimum_rescale)

	if result_name is None and column is not None:
		result_name = f'ratio_min({column})'

	if inplace and input is not None and result_name is not None:
		input[result_name] = scaled
	else:
		return scaled
