
import pandas
import numpy
import operator

from sklearn.preprocessing import OneHotEncoder

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

	Parameters
	----------
	values : array-like
		The values to label.
	mapping : Mapping
		Keys of this mapping will be the new values,
		and the values are 2-tuples giving upper and
		lower bounds.
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
	"""
	if mapping is None:
		mapping = {}

	mapping.update(kwargs)

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

	for k,(lowerbound,upperbound) in mapping.items():
		if lowerbound is None:
			lowerbound = -numpy.inf
		if upperbound is None:
			upperbound = numpy.inf
		x[lop(values,lowerbound) & rop(values,upperbound)] = k

	if isinstance(x, pandas.Series):
		x = x.astype('category')

	return x

