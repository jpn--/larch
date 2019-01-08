

import numpy
import pandas

def tiny_perturbation(arr):
	fr = numpy.frexp(arr)
	perturb = numpy.random.random(fr[0].shape) * 1e-12
	return numpy.ldexp(fr[0]+perturb, fr[1])


def rank_within_cases(df, values, groupdef, output_col=None, tiny_perturb=False):
	if tiny_perturb:
		x = tiny_perturbation(df[values])
	else:
		x = df[values]

	result = x.groupby(df[groupdef]).rank("first", ascending=False)

	if output_col is not None:
		df[output_col] = result

	return result


def probability_to_rank(arr):
	"""Convert a probability array to ranks.

	Parameters
	----------
	arr: array-like
		The probability array to convert.  Each row is a case observation
		and should have probability that sums to 1.

	Returns
	-------
	ranks
		An array or DataFrame (as arr) with ranks.
	"""

	temp = arr.argsort(axis=-1)
	ranks = numpy.empty_like(temp)
	seq = numpy.arange(1,arr.shape[1]+1)[::-1]
	for i in range(arr.shape[0]):
		ranks[i,temp[i]] = seq

	if isinstance(arr, pandas.DataFrame):
		return pandas.DataFrame(ranks, columns=arr.columns, index=arr.index)
	else:
		return ranks


def top_k_accuracy(ranks, choices, k=1):
	"""
	Compute top-N accuracy.

	Parameters
	----------
	ranks: array-like
		Ranked alternatives.  The best alternative in a row should be ranked 1,
		the next 2, and so on.
	choices
		A one-hot encoded or weighted choice array.
	k : int or iterable of ints
		k

	Returns
	-------
	float
		Fraction of rows where the chosen alternative is ranked at k or better.
	"""
	if isinstance(k, (int, numpy.integer)):
		ranks = numpy.asarray(ranks)
		choices = numpy.asarray(choices)
		_1 = (ranks[choices > 0] <= k)
		_2 = (choices[choices > 0])
		return (_1*_2).sum(axis=None) / (choices).sum(axis=None)

	results = {}
	for k_ in k:
		results[k_] = top_k_accuracy(ranks, choices, k=k_)

	return pandas.Series(results)

