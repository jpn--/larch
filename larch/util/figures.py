

from matplotlib import pyplot as plt
import pandas, numpy
from .plotting import plot_as_svg_xhtml


def pseudo_bar_data(x_bins, y, gap=0):
	"""
	Parameters
	----------
	x_bins : array-like, shape=(N+1,)
		The bin boundaries
	y : array-like, shape=(N,)
		The bar heights

	Returns
	-------
	x, y
	"""
	# midpoints = (x_bins[1:] + x_bins[:-1]) / 2
	# widths = x_bins[1:] - x_bins[:-1]
	if gap:
		x_doubled = numpy.zeros(((x_bins.shape[0] - 1) * 4), dtype=numpy.float)
		x_doubled[::4] = x_bins[:-1]
		x_doubled[1::4] = x_bins[:-1]
		x_doubled[2::4] = x_bins[1:] - gap
		x_doubled[3::4] = x_bins[1:] - gap
		y_doubled = numpy.zeros(((y.shape[0]) * 4), dtype=y.dtype)
		y_doubled[1::4] = y
		y_doubled[2::4] = y
	else:
		x_doubled = numpy.zeros((x_bins.shape[0] - 1) * 2, dtype=x_bins.dtype)
		x_doubled[::2] = x_bins[:-1]
		x_doubled[1::2] = x_bins[1:]
		y_doubled = numpy.zeros((y.shape[0]) * 2, dtype=y.dtype)
		y_doubled[::2] = y
		y_doubled[1::2] = y
	return x_doubled, y_doubled


def distribution_on_idca_variable(
		model,
		x,
		xlabel=None,
		ylabel='Relative Frequency',
		style='hist',
		bins=None,
		pct_bins=20,
		range=None,
		prob_label="Modeled",
		obs_label="Observed",
		subselector=None,
		probability=None,
		bw_method=None,
		discrete=None,
		ax=None,
		format='figure',
		**kwargs,
):
	"""
	Generate a figure of observed and modeled choices over a range of variable values.

	Parameters
	----------
	model : Model
		The discrete choice model to analyze.
	x : str or array-like
		The name of an `idca` variable, or an array giving its values.  If this name exactly
		matches that of an `idca` column in the model's loaded `dataframes`, then
		those values are used, otherwise the variable is loaded from the model's
		`dataservice`.
	xlabel : str, optional
		A label to use for the x-axis of the resulting figure.  If not given,
		the value of `x` is used if it is a string.  Set to `False` to omit the
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
	ax : matplotlib.Axes, optional
		If given, the figure will be drawn on these axes and they will be returned,
		otherwise new blank axes are used to draw the figure.
	format : {'figure','svg'}, default 'figure'
		How to return the result if it is a figure. The default is to return
		the raw matplotlib Figure instance (or the `ax` if given), set to `svg` to get a SVG
		rendering as an xmle.Elem.

	Other Parameters
	----------------
	header : str, optional
		A header to attach to the figure.  The header is not generated using
		matplotlib, but instead is prepended to the xml output with a header tag before the
		rendered svg figure.


	Returns
	-------
	Elem or Axes
		Returns `ax` if given as an argument, otherwise returns a rendering as an Elem
	"""

	if model is None:
		return lambda mdl: distribution_on_idca_variable(
			mdl,
			xlabel=xlabel,
			ylabel=ylabel,
			style=style,
			bins=bins,
			pct_bins=pct_bins,
			range=range,
			prob_label=prob_label,
			obs_label=obs_label,
			subselector=subselector,
			probability=probability,
			bw_method=bw_method,
			discrete=discrete,
			ax=ax,
			format=format,
			**kwargs,
		)

	if isinstance(x, str):
		x_label = x
		if model.dataframes and model.dataframes.data_ca_or_ce is not None and x in model.dataframes.data_ca_or_ce:
			x = model.dataframes.data_ca_or_ce[x].values.reshape(-1)
		else:
			x = model.dataservice.make_dataframes({'ca': [x]}, explicit=True).array_ca().reshape(-1)
	else:
		try:
			x_label = x.name
		except AttributeError:
			x_label = ''

	# if model.dataframes and model.dataframes.data_ca is not None and continuous_variable in model.dataframes.data_ca:
	# 	cv = model.dataframes.data_ca[continuous_variable].values.reshape(-1)
	# else:
	# 	cv = model.dataservice.make_dataframes({'ca': [continuous_variable]}, explicit=True).array_ca().reshape(-1)

	discrete_values = None
	if discrete:
		discrete_values = numpy.unique(x)
	elif discrete is None:
		from .histograms import seems_like_discrete_data
		discrete, discrete_values = seems_like_discrete_data(numpy.asarray(x).reshape(-1), return_uniques=True)

	x_discrete_labels = None if discrete_values is None else [str(i) for i in discrete_values]

	if bins is None:
		if x_discrete_labels is not None:
			bins = numpy.arange(len(x_discrete_labels)+1)
		if isinstance(x.dtype, pandas.CategoricalDtype):
			discrete_values = numpy.arange(len(x_discrete_labels))
			bins = numpy.arange(len(x_discrete_labels)+1)
			x = x.cat.codes
		elif isinstance(pct_bins, int):
			bins = numpy.percentile(x, numpy.linspace(0, 100, pct_bins + 1))
		else:
			bins = numpy.percentile(x, pct_bins)

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
		x = numpy.asarray(x).reshape(*model_result.shape)[subselector].reshape(-1)
		model_result = model_result[subselector]
		model_choice = model_choice[subselector]

	if style == 'kde':
		import scipy.stats
		kernel_result = scipy.stats.gaussian_kde(x, bw_method=bw_method, weights=model_result.reshape(-1))
		common_bw = kernel_result.covariance_factor()
		kernel_choice = scipy.stats.gaussian_kde(x, bw_method=common_bw, weights=model_choice.reshape(-1))

		if range is None:
			range = (x.min(), x.max())

		x_points = numpy.linspace(*range, 250)
		y_points_1 = kernel_result(x_points)
		y_points_2 = kernel_choice(x_points)

	else:
		if range is not None:
			range_low, range_high = range
			if range_low is None:
				range_low = x.min()
			if range_high is None:
				range_high = x.max()
			range = (range_low, range_high)

		y_points_1, x1 = numpy.histogram(
			x,
			weights=model_result.reshape(-1),
			bins=bins,
			range=range,
		)

		y_points_2, x2 = numpy.histogram(
			x,
			weights=model_choice.reshape(-1),
			bins=x1,
		)

		shift = 0.4 if discrete else 0
		gap = 0.2 if discrete else 0

		x_points, y_points_1 = pseudo_bar_data(x1 - shift, y_points_1, gap=gap)
		x_points, y_points_2 = pseudo_bar_data(x1 - shift, y_points_2, gap=gap)


	if xlabel is None:
		xlabel = x_label
	if xlabel is False:
		xlabel = None

	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = None

	ax.plot(x_points, y_points_1, label=prob_label, lw=1.5)
	ax.fill_between(x_points, y_points_2, label=obs_label, step=None, facecolor='#ffbe4d', edgecolor='#ffa200', lw=1.5)
	ax.legend()
	if not discrete:
		ax.set_xlim(x_points[0], x_points[-1])
	if x_discrete_labels is not None:
		ax.set_xticks(numpy.arange(len(x_discrete_labels)))
		ax.set_xticklabels(x_discrete_labels)
	ax.set_xlabel(xlabel)
	ax.set_yticks([])
	ax.set_ylabel(ylabel)
	if fig is None:
		return ax
	fig.tight_layout(pad=0.5)
	if format == 'svg':
		result = plot_as_svg_xhtml(fig, **kwargs)
		fig.clf()
		plt.close(fig)
	else:
		result = fig
	return result


from .. import Model

Model.distribution_on_idca_variable = distribution_on_idca_variable



def distribution_on_idco_variable(
		model,
		x,
		xlabel=None,
		bins=None,
		pct_bins=20,
		figsize=(12, 4),
		style='stacked',
		discrete=None,
		xlim=None,
		include_nests=False,
		exclude_alts=None,
		filter=None,
		format='figure',
		**kwargs,
):
	"""
	Generate a figure of variables over a range of variable values.

	Parameters
	----------
	model : Model
		The discrete choice model to analyze.
	x : str or array-like
		The name of an `idco` variable, or an array giving its values.  If this name exactly
		matches that of an `idco` column in the model's loaded `dataframes`, then
		those values are used, otherwise the variable is loaded from the model's
		`dataservice`.
	xlabel : str, optional
		A label to use for the x-axis of the resulting figure.  If not given,
		the value of `x` is used if it is a string.  Set to `False` to omit the
		x-axis label.
	bins : int, optional
		The number of equal-sized bins to use.
	pct_bins : int or array-like, default 20
		The number of equal-mass bins to use.
	style : {'stacked', 'dataframe', 'many'}
		The type of output to generate.
	discrete : bool, default False
		Whether to treat the data values explicitly as discrete (vs continuous)
		data.  This will change the styling and automatic bin generation.  If
		there are very few unique values, the data will be assumed to be
		discrete anyhow.
	xlim : 2-tuple, optional
		Explicitly set the range of values shown on the x axis of generated
		figures.  This can truncate long tails.  The actual histogram bins
		are not changed.
	include_nests : bool, default False
		Whether to include nests in the figure.
	exclude_alts : Collection, optional
		Alternatives to exclude from the figure.
	filter : str, optional
		A filter that will be used to select only a subset of cases.
	format : {'figure','svg'}, default 'figure'
		How to return the result if it is a figure. The default is to return
		the raw matplotlib Figure instance, ot set to `svg` to get a SVG
		rendering as an xmle.Elem.

	Returns
	-------
	Figure, DataFrame, or Elem
	"""

	if style not in {'stacked', 'dataframe', 'many'}:
		raise ValueError("style must be in {'stacked', 'dataframe', 'many'}")

	if include_nests and style == 'stacked' and exclude_alts is None:
		import warnings
		warnings.warn("including nests in a stacked figure is likely to give "
					  "misleading results unless constituent alternatives are omitted")

	if exclude_alts is None:
		exclude_alts = set()

	if isinstance(x, str):
		x_label = x
		if model.dataframes and model.dataframes.data_co is not None and x in model.dataframes.data_co:
			x = model.dataframes.data_co[x].values.reshape(-1)
		else:
			x = model.dataservice.make_dataframes({'co': [x]}, explicit=True).array_co().reshape(-1)
	else:
		try:
			x_label = x.name
		except AttributeError:
			x_label = ''

	if filter:
		_ds = model.dataservice if model.dataservice is not None else model.dataframes
		filter_ = _ds.make_dataframes(
			{'co': [filter]},
			explicit=True,
			float_dtype=bool,
		).array_co().reshape(-1)
		x = x[filter_]
	else:
		filter_ = slice(None)

	h_pr = {}
	h_ch = {}

	discrete_values = None
	if discrete:
		discrete_values = numpy.unique(x)
	elif discrete is None:
		from .histograms import seems_like_discrete_data
		discrete, discrete_values = seems_like_discrete_data(x, return_uniques=True)

	pr = model.probability(
		return_dataframe='names',
		include_nests=bool(include_nests),
	).loc[filter_,:]

	if include_nests:
		ch = model.dataframes.data_ch_cascade(model.graph).loc[filter_,:]
	else:
		ch = model.dataframes.data_ch.loc[filter_,:]


	if model.dataframes.data_wt is None:
		wt = 1
	else:
		wt = model.dataframes.data_wt.values.reshape(-1)[filter_]

	x_discrete_labels = None if discrete_values is None else [str(i) for i in discrete_values]

	if bins is None:
		if isinstance(x.dtype, pandas.CategoricalDtype):
			discrete_values = numpy.arange(len(x_discrete_labels))
			bins = numpy.arange(len(x_discrete_labels)+1)
			x = x.cat.codes
		elif isinstance(pct_bins, int):
			bins = numpy.percentile(x, numpy.linspace(0, 100, pct_bins + 1))
		else:
			bins = numpy.percentile(x, pct_bins)

	n_alts = model.graph.n_elementals()
	columns = {}

	for i in range(pr.shape[1]):
		columns[i] = pr.columns[i]
		if i < n_alts or include_nests is True or model.graph.standard_sort[i] in include_nests:
			if model.graph.standard_sort[i] == model.graph.root_id:
				continue
			if model.graph.standard_sort[i] in exclude_alts:
				continue
			h_pr[i], _ = numpy.histogram(
				x,
				weights=pr.iloc[:, i] * wt,
				bins=bins,
			)
			h_ch[i], _ = numpy.histogram(
				x,
				weights=ch.iloc[:, i] * wt,
				bins=bins,
			)

	h_pr = pandas.DataFrame(h_pr)
	h_pr.index = pandas.IntervalIndex.from_breaks(bins) # bins[:-1]
	h_pr.rename(columns=columns, inplace=True)
	h_pr_share = (h_pr / h_pr.values.sum(1).reshape(-1, 1))

	h_ch = pandas.DataFrame(h_ch)
	h_ch.index = h_pr.index
	h_ch.rename(columns=columns, inplace=True)
	h_ch_share = (h_ch / h_ch.values.sum(1).reshape(-1, 1))

	if discrete:
		x_placement = numpy.arange(len(bins)-1)
		x_alignment = 'center'
		bin_widths = 0.8
	else:
		x_placement = bins[:-1]
		x_alignment = 'edge'
		bin_widths = bins[1:] - bins[:-1]

	if xlabel is None:
		xlabel = x_label
	if xlabel is False:
		xlabel = None

	if xlim is None:
		xlim = (bins[0], bins[-1])

	if style == 'dataframe':

		result = pandas.concat({
			'Modeled Shares': h_pr_share,
			'Observed Shares': h_ch_share,
		}, axis=1, sort=False)
		result['Count', '*'] = h_pr.sum(1)

		if x_label:
			result.index.name = xlabel

	elif style == 'stacked':

		fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

		bottom0 = 0
		bottom1 = 0

		for i in h_pr_share.columns:
			ax0.bar(
				x_placement,
				height=h_pr_share[i],
				bottom=bottom0,
				width=bin_widths,
				align=x_alignment,
				label=i,
			)
			bottom0 = h_pr_share[i].values + bottom0
			ax1.bar(
				x_placement,
				height=h_ch_share[i],
				bottom=bottom1,
				width=bin_widths,
				align=x_alignment,
			)
			bottom1 = h_ch_share[i].values + bottom1

		ax0.set_ylim(0, 1)
		if not discrete:
			ax0.set_xlim(*xlim)
		if x_discrete_labels is not None:
			ax0.set_xticks(numpy.arange(len(x_discrete_labels)))
			ax0.set_xticklabels(x_discrete_labels)
		ax0.set_title('Modeled Shares')

		ax1.set_ylim(0, 1)
		if not discrete:
			ax1.set_xlim(*xlim)
		if x_discrete_labels is not None:
			ax1.set_xticks(numpy.arange(len(x_discrete_labels)))
			ax1.set_xticklabels(x_discrete_labels)
		ax1.set_title('Observed Shares')
		if xlabel:
			ax0.set_xlabel(xlabel)
			ax1.set_xlabel(xlabel)

		fig.legend(
			loc='center right',
		)

		# fig.tight_layout(pad=0.5)
		if format == 'svg':
			result = plot_as_svg_xhtml(fig, **kwargs)
			fig.clf()
			plt.close(fig)
		else:
			result = fig

	else:

		fig, axes = plt.subplots(len(h_pr_share.columns), 1, figsize=figsize)

		shift = 0.4 if discrete else 0

		for n,i in enumerate(h_pr_share.columns):
			x_, y_ = pseudo_bar_data(bins-shift, h_pr_share[i], gap=0.2 if discrete else 0)
			axes[n].plot(x_, y_, label='Modeled' if n==0 else None, lw=1.5)

			x_ch_, y_ch_ = pseudo_bar_data(bins-shift, h_ch_share[i], gap=0.2 if discrete else 0)
			axes[n].fill_between(
				x_ch_, y_ch_, label='Observed' if n==0 else None, step=None,
				facecolor='#ffbe4d', edgecolor='#ffa200',
				lw=1.5,
			)
			if not discrete:
				axes[n].set_xlim(*xlim)
			if x_discrete_labels is not None:
				axes[n].set_xticks(numpy.arange(len(x_discrete_labels)))
				axes[n].set_xticklabels(x_discrete_labels)
			axes[n].set_ylabel(i)

			# axes[n].legend(
			# 	# loc='center right',
			# )

		legnd = axes[0].legend(
			loc='lower center',
			ncol=2,
			borderaxespad=0,
			bbox_to_anchor=(0.5, 1.08)
		)

		if xlabel:
			axes[-1].set_xlabel(xlabel)
		#fig.tight_layout(pad=0.5)
		if format == 'svg':
			result = plot_as_svg_xhtml(fig, bbox_extra_artists=[legnd],  **kwargs)
			fig.clf()
			plt.close(fig)
		else:
			result = fig

	return result



Model.distribution_on_idco_variable = distribution_on_idco_variable
