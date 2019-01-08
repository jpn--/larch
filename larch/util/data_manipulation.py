
import pandas
import numpy

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

