
import pandas
import numpy
import operator

from sklearn.preprocessing import OneHotEncoder
from typing import Mapping

def one_hot_encode(vector, dtype='float32', categories=None, index=None):

	if isinstance(vector, pandas.Series):
		index = vector.index
		vector = vector.values

	encoder = OneHotEncoder(
		categories='auto' if categories is None else [categories,],
		sparse=False,
		dtype=dtype,
	).fit(vector.reshape(-1,1))

	return pandas.DataFrame(
		data = encoder.transform(vector.reshape(-1,1)),
		columns=encoder.categories_,
		index=index
	)

def periodize(
		values,
		mapping=None,
		default=None,
		right=True,
		left=True,
		**kwargs,
):
	"""
	Label sections of a continuous variable.

	This function contrasts with `pandas.cut` in that
	there can be multiple non-contiguous sections of the
	underlying continuous interval that obtain the same
	categorical value.

	Parameters
	----------
	values : array-like
		The values to label. If given as a pandas.Series,
		the returned values will also be a Series,
		with a categorical dtype.
	mapping : Collection or Mapping
		A mapping, or a collection of 2-tuples giving
		key-value pairs (not necessarily unique keys).
		The keys (or first values) will be the new values,
		and the values (or second values) are 2-tuples
		giving upper and lower bounds.
	default : any, default None
		Keys not inside any labeled interval will get
		this value instead.
	right : bool, default True
		Whether to include the upper bound[s] in the
		intervals for labeling.
	left : bool, default True
		Whether to include the lower bound[s] in the
		intervals for labeling.
	**kwargs :
		Are added to `mapping`.

	Returns
	-------
	array-like

	Example
	-------
	>>> import pandas
	>>> h = pandas.Series(range(1,24))
	>>> periodize(h, default='OP', AM=(6.5, 9), PM=(16, 19))
	0     OP
	1     OP
	2     OP
	3     OP
	4     OP
	5     OP
	6     AM
	7     AM
	8     AM
	9     OP
	10    OP
	11    OP
	12    OP
	13    OP
	14    OP
	15    PM
	16    PM
	17    PM
	18    PM
	19    OP
	20    OP
	21    OP
	22    OP
	dtype: category
	Categories (3, object): ['AM', 'OP', 'PM']
	"""
	if mapping is None:
		mapping = []

	if isinstance(mapping, Mapping):
		mapping = list(mapping.items())

	mapping.extend(kwargs.items())

	if isinstance(values, pandas.Series):
		x = pandas.Series(index=values.index, data=default)
	else:
		x = numpy.full(values.shape, default)

	if right:
		rop = operator.le
	else:
		rop = operator.lt

	if left:
		lop = operator.ge
	else:
		lop = operator.gt

	for k,(lowerbound,upperbound) in mapping:
		if lowerbound is None:
			lowerbound = -numpy.inf
		if upperbound is None:
			upperbound = numpy.inf
		x[lop(values,lowerbound) & rop(values,upperbound)] = k

	if isinstance(x, pandas.Series):
		x = x.astype('category')

	return x
