
import pandas


def joint_parameter_names(*models):
	"""Build a set of parameter names across multiple models."""
	jnames = set()
	for model in models:
		jnames |= set(model.pf.index)
	return jnames


def joint_parameter_indexing(ordering, *models):
	paramset = joint_parameter_names(*models)
	out = []
	import re
	for category in ordering:
		category_name = category[0]
		category_params = []
		for category_pattern in category[1:]:
			category_params.extend(sorted(i for i in paramset if re.match(category_pattern, i) is not None))
			paramset -= set(category_params)
		out.append([category_name, category_params])
	if len(paramset):
		out.append(['Other', sorted(paramset)])

	tuples = []
	for c, pp in out:
		for p in pp:
			tuples.append((c, p))

	ix = pandas.MultiIndex.from_tuples(tuples, names=['Category', 'Parameter'])
	return ix


def joint_parameter_summary(models, ordering, t_stats=True, loglike=True):
	"""
	Create a joint parameter summary table.

	Parameters
	----------
	models : Collection[Model]
	ordering : tuple
	t_stats : bool, default True

	Returns
	-------
	pandas.DataFrame
	"""

	subheads = ['Param']
	if t_stats:
		subheads.append('t-Stat')

	heads = [(model.title, colname) for model in models for colname in subheads]

	summary = pandas.DataFrame(
		index=joint_parameter_indexing(ordering, *models),
		columns=pandas.MultiIndex.from_tuples(heads),
		data="",
	)

	for model in models:
		for param, x in model.parameter_summary('df').iterrows():
			if isinstance(param, tuple):
				param = param[1]
			summary.loc[(slice(None), param), (model.title, 'Param')] = x['Value']
			if t_stats:
				try:
					t = x['t Stat']
				except KeyError:
					pass
				else:
					summary.loc[(slice(None), param), (model.title, 't-Stat')] = t

	if loglike:
		for model in models:
			try:
				ll = model.most_recent_estimation_result.loglike
			except:
				pass
			else:
				summary.loc[('Log Likelihood', 'Converged'), (model.title, 'Param')] = "{:,.2f}".format(ll)

			try:
				ll_null = model.loglike_null(-1)
			except:
				pass
			else:
				summary.loc[('Log Likelihood', 'Null'), (model.title, 'Param')] = "{:,.2f}".format(ll_null)

	return summary