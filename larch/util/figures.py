

from matplotlib import pyplot as plt
import pandas, numpy
from .plotting import plot_as_svg_xhtml

def distribution_on_continuous_idca_variable(
		model,
		continuous_variable,
		xlabel=None,
		ylabel='Relative Frequency',
		style='hist',
		bins=25,
		range=None,
		prob_label="Modeled",
		obs_label="Observed",
		subselector=None,
		probability=None,
		bw_method=None,
		**kwargs,
):
	"""
	Generate a figure of observed and modeled choices over a range of variable values.

	Parameters
	----------
	model : Model
		The discrete choice model to analyze.
	continuous_variable : str
		The name of an `idca` variable that is continuous.  If this name exactly
		matches that of an `idca` column in the model's loaded `dataframes`, then
		those values are used, otherwise the variable is loaded from the model's
		`dataservice`.
	xlabel : str, optional
		A label to use for the x-axis of the resulting figure.  If not given,
		the value of `continuous_variable` is used.  Set to `False` to omit the
		x-axis label.
	ylabel : str, default "Relative Frequency"
		A label to use for the y-axis of the resulting figure.
	style : {'hist', 'kde'}
		The style of figure to produce, either a histogram or a kernel density plot.
	bins : int, default 25
		The number of bins to use, only applicable to histogram style.
	range : 2-tuple, optional
		A range to truncate the figure.
	prob_label : str, default "Modeled"
		A label to put in the legend for the modeled probabilities
	obs_label : str, default "Observed"
		A label to put in the legend for the observed choices
	subselector : str or array-like, optional
		A filter to apply to cases. If given as a string, this is loaded from the
		model's `dataservice` as an `idco` variable.
	probability : array-like, optional
		The pre-calculated probability array for all cases in this analysis.
		If not given, the probability array is calculated at the current parameter
		values.

	Other Parameters
	----------------
	header : str, optional
		A header to attach to the figure.  The header is not generated using
		matplotlib, but instead is prepended to the xml output with a header tag before the
		rendered svg figure.


	Returns
	-------
	Elem
	"""

	if model is None:
		return lambda x: distribution_on_continuous_idca_variable(
		x,
		continuous_variable,
		xlabel=xlabel,
		bins=bins,
		range=range,
		prob_label=prob_label,
		obs_label=obs_label,
		subselector=subselector,
			**kwargs,
		)

	if model.dataframes and model.dataframes.data_ca is not None and continuous_variable in model.dataframes.data_ca:
		cv = model.dataframes.data_ca[continuous_variable].values.reshape(-1)
	else:
		cv = model.dataservice.make_dataframes({'ca': [continuous_variable]}, explicit=True).array_ca().reshape(-1)

	if probability is None:
		probability = model.probability()

	model_result = probability[:, :model.dataframes.n_alts]
	model_choice = model.dataframes.data_ch.values
	if model.dataframes.data_wt is not None:
		model_result = model_result.copy()
		model_result *= model.dataframes.data_wt.values[:,None]
		model_choice = model_choice.copy()
		model_choice *= model.dataframes.data_wt.values[:,None]

	if subselector is not None:
		if isinstance(subselector, str):
			subselector = model.dataservice.make_dataframes({'co': [subselector]}, explicit=True).array_co(dtype=bool).reshape(-1)
		cv = cv.reshape(*model_result.shape)[subselector].reshape(-1)
		model_result = model_result[subselector]
		model_choice = model_choice[subselector]

	if style == 'kde':
		import scipy.stats
		kernel_result = scipy.stats.gaussian_kde(cv, bw_method=bw_method, weights=model_result.reshape(-1))
		common_bw = kernel_result.covariance_factor()
		kernel_choice = scipy.stats.gaussian_kde(cv, bw_method=common_bw, weights=model_choice.reshape(-1))

		if range is None:
			range = (cv.min(), cv.max())

		x_midpoints = numpy.linspace(*range, 250)
		y = kernel_result(x_midpoints)
		y_ = kernel_choice(x_midpoints)


	else:
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
		x_midpoints = (x[1:] + x[:-1]) / 2

		x_doubled = numpy.zeros((x.shape[0]-1)*2)
		x_doubled[::2] = x[:-1]
		x_doubled[1::2] = x[1:]

		y_doubled = numpy.zeros((y.shape[0])*2)
		y_doubled_ = numpy.zeros((y.shape[0])*2)

		y_doubled[::2] = y
		y_doubled[1::2] = y
		y_doubled_[::2] = y_
		y_doubled_[1::2] = y_

		y, y_ = y_doubled, y_doubled_
		x_midpoints = x_doubled

	if xlabel is None:
		xlabel = continuous_variable
	if xlabel is False:
		xlabel = None

	fig, ax = plt.subplots()
	if style=='kde':
		ax.plot(x_midpoints, y, label=prob_label, lw=1.5)
		ax.fill_between(x_midpoints, y_, label=obs_label, step=None, facecolor='#ffbe4d', edgecolor='#ffa200', lw=1.5)
	else:
		ax.plot(x_midpoints, y, label=prob_label, lw=1.5)
		ax.fill_between(x_midpoints, y_, label=obs_label, step=None, facecolor='#ffbe4d', edgecolor='#ffa200', lw=1.5)
	ax.legend()
	ax.set_xlabel(xlabel)
	ax.set_yticks([])
	ax.set_ylabel(ylabel)
	fig.tight_layout(pad=0.5)
	result = plot_as_svg_xhtml(fig, **kwargs)
	fig.clf()
	plt.close(fig)
	return result


from .. import Model

Model.distribution_on_continuous_idca_variable = distribution_on_continuous_idca_variable



def distribution_on_continuous_idco_variable(
		model,
		continuous_variable,
		bins=None,
		pct_bins=20,
		figsize=(12, 4),
		**kwargs,
):

	if isinstance(continuous_variable, str):
		if model.dataframes and model.dataframes.data_co is not None and continuous_variable in model.dataframes.data_co:
			x = model.dataframes.data_co[continuous_variable].values.reshape(-1)
		else:
			x = model.dataservice.make_dataframes({'co': [continuous_variable]}, explicit=True).array_co().reshape(-1)
	else:
		x = continuous_variable
		try:
			continuous_variable = continuous_variable.name
		except AttributeError:
			continuous_variable = ''

	h_pr = {}
	h_ch = {}


	pr = model.probability(
		return_dataframe='names',
	)

	ch = model.dataframes.data_ch

	if bins is None:
		if isinstance(pct_bins, int):
			bins = numpy.percentile(x, numpy.linspace(0, 100, pct_bins + 1))
		else:
			bins = numpy.percentile(x, pct_bins)

	for i in range(pr.shape[1]):
		h_pr[i], _ = numpy.histogram(
			x,
			weights=pr.iloc[:, i],
			bins=bins,
		)
		h_ch[i], _ = numpy.histogram(
			x,
			weights=ch.iloc[:, i],
			bins=bins,
		)

	h_pr = pandas.DataFrame(h_pr)
	h_pr.index = bins[:-1]
	h_pr.columns = pr.columns
	h_pr = (h_pr / h_pr.values.sum(1).reshape(-1, 1))

	h_ch = pandas.DataFrame(h_ch)
	h_ch.index = bins[:-1]
	h_ch.columns = pr.columns
	h_ch = (h_ch / h_ch.values.sum(1).reshape(-1, 1))

	fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

	bottom0 = 0
	bottom1 = 0
	bin_widths = bins[1:] - bins[:-1]

	for i in h_pr.columns:
		ax0.bar(
			h_pr.index,
			height=h_pr[i],
			bottom=bottom0,
			width=bin_widths,
			align='edge',
			label=i,
		)
		bottom0 = h_pr[i] + bottom0
		ax1.bar(
			h_ch.index,
			height=h_ch[i],
			bottom=bottom1,
			width=bin_widths,
			align='edge',
		)
		bottom1 = h_ch[i] + bottom1

	ax0.set_ylim(0, 1)
	ax0.set_xlim(bins[0], bins[-1])
	ax0.set_title('Modeled Shares')

	ax1.set_ylim(0, 1)
	ax1.set_xlim(bins[0], bins[-1])
	ax1.set_title('Observed Shares')
	if continuous_variable:
		ax0.set_xlabel(continuous_variable)
		ax1.set_xlabel(continuous_variable)

	fig.legend(
		loc='center right',
	)

	# fig.tight_layout(pad=0.5)
	result = plot_as_svg_xhtml(fig, **kwargs)
	fig.clf()
	plt.close(fig)
	return result



Model.distribution_on_continuous_idco_variable = distribution_on_continuous_idco_variable
