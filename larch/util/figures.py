

from matplotlib import pyplot as plt
import pandas, numpy
from .plotting import plot_as_svg_xhtml

def distribution_on_continuous_idca_variable(
		model,
		continuous_variable,
		continuous_variable_label=None,
		bins=25,
		range=None,
		prob_label="Modeled",
		obs_label="Observed",
		header=None,
		subselector=None,
):
	"""

	Parameters
	----------
	model : Model
	continuous_variable : str
	continuous_variable_label
	bins
	range
	prob_label : str, optional
		A label to put in the legend for the modeled probabilities
	obs_label : str, optional
		A label to put in the legend for the observed choices

	Returns
	-------
	Elem
	"""

	if model is None:
		return lambda x: distribution_on_continuous_idca_variable(
		x,
		continuous_variable,
		continuous_variable_label=continuous_variable_label,
		bins=bins,
		range=range,
		prob_label=prob_label,
		obs_label=obs_label,
		header=header,
		subselector=subselector,
		)

	sel = model.selector_eval()

	if subselector is None:
		cv = model.dataservice.array_idca(continuous_variable, selector=sel).reshape(-1)
		model_result = model.work.total_probability[:, :model.n_alts]
		model_choice = model.data.choice.values
		if model.data.weight is not None:
			model_result = model_result.copy()
			model_result *= model.data.weight.values[:,None]
			model_choice = model_choice.copy()
			model_choice *= model.data.weight.values[:,None]
	else:
		subsel = model.dataservice.array_idco(subselector, selector=sel, dtype=bool).squeeze()
		cv = model.dataservice.array_idca(continuous_variable, selector=sel)[subsel]
		cv = cv.reshape(-1)
		model_result = model.work.total_probability[subsel, :model.n_alts]
		model_choice = model.data.choice.values[subsel]
		if model.data.weight is not None:
			model_result = model_result.copy()
			model_result *= model.data.weight.values[subsel,None]
			model_choice = model_choice.copy()
			model_choice *= model.data.weight.values[subsel,None]

	y, x = numpy.histogram(
		cv,
		weights=model_result.reshape(-1),
		bins=bins,
		range=range,
	)

	y_, x_ = numpy.histogram(
		cv,
		weights=model_choice.reshape(-1),
		bins=x,
	)

	if continuous_variable_label is None:
		continuous_variable_label = continuous_variable

	x_midpoints = (x[1:] + x[:-1]) / 2

	plt.ioff()
	plt.clf()
	plt.plot(x_midpoints, y, label=prob_label)
	plt.plot(x_midpoints, y_, label=obs_label)
	plt.legend()
	plt.xlabel(continuous_variable_label)
	plt.tight_layout(pad=0.5)
	result = plot_as_svg_xhtml(plt, header=header)
	plt.clf()
	return result