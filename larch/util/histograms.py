
import numpy
import numpy.ma as ma
from .colors import hexcolor

try:
	from numpy.lib.function_base import _hist_bin_selectors
except ImportError:
	# moved in later version of numpy
	from numpy.lib.histograms import _hist_bin_selectors  #


def robust_min_max(s, threshold_for_dropping_zeros=0.35, cut_after_stdevs=5, percentiles_to_cut=1):
	# Get s as an array (it should be already though)
	s = numpy.asarray(s)
	# mask out infs and nans
	sx = ma.masked_array(s, mask=~numpy.isfinite(s))
	n_values = sx.count()
	# Check if data is mostly zeros
	n_zeros = (s==0).sum()
	if n_zeros/n_values > threshold_for_dropping_zeros:
		sx.mask |= (sx == 0)
		zeros_are_dropped = True
		n_values = sx.count()
	else:
		zeros_are_dropped = False
	if n_values==0:
		# no data is left, default values
		bottom, top, range_is_truncated = 0, 1, 100
	else:
		mean = sx.mean()
		stdev = sx.std()
		bottom = sx.min()
		top = sx.max()
		range_is_truncated = 0
		cut_bottom = (bottom < mean - cut_after_stdevs * stdev)
		cut_top = (top > mean + cut_after_stdevs * stdev)
		if percentiles_to_cut>0:
			if cut_bottom:
				bottom_ = numpy.percentile(sx, percentiles_to_cut)
				if not numpy.isnan(bottom_):
					bottom = bottom_
					range_is_truncated += percentiles_to_cut
			if cut_top:
				top_ = numpy.percentile(sx, 100-percentiles_to_cut)
				if not numpy.isnan(top_):
					top = top_
					range_is_truncated += percentiles_to_cut
	return bottom, top, zeros_are_dropped, range_is_truncated, sx


def fixed_min_max(s, fixed_min, fixed_max, threshold_for_dropping_zeros=0.35):
	# Get s as an array (it should be already though)
	s = numpy.asarray(s)
	# mask out infs and nans
	sx = ma.masked_array(s, mask=numpy.isnan(s))
	n_values_finite = n_values = sx.count()

	n_zeros = (s==0).sum()
	if n_zeros/n_values > threshold_for_dropping_zeros:
		sx.mask |= (sx == 0)
		zeros_are_dropped = True
		n_values = sx.count()
	else:
		zeros_are_dropped = False

	# Over max
	if fixed_max is not None:
		sx.mask |= (sx >= fixed_max)
	n_values_overmax = n_values - sx.count()
	n_values = sx.count()

	# Under Min
	if fixed_min is not None:
		sx.mask |= (sx <= fixed_min)
	n_values_undermin = n_values - sx.count()
	n_values = sx.count()

	pct_over = n_values_overmax / n_values_finite
	pct_under = n_values_undermin / n_values_finite

	return sx, pct_under, pct_over, zeros_are_dropped


def get_histogram_bins(data_to_bin, bins='sturges', percentiles_to_cut=1, maxrange=None, fixedrange=None, threshold_for_dropping_zeros=0.35):
	if fixedrange is None:
		bottom, top, zeros_are_dropped, range_is_truncated, sx = robust_min_max(data_to_bin, percentiles_to_cut=percentiles_to_cut)
		pct_under, pct_over = 0, 0
	else:
		bottom, top = fixedrange[0], fixedrange[1]
		sx, pct_under, pct_over, zeros_are_dropped = fixed_min_max(data_to_bin, bottom, top, threshold_for_dropping_zeros=threshold_for_dropping_zeros)
		range_is_truncated = (pct_under + pct_over)*100
	bottom = float(bottom) if bottom is not None else sx.min()-1e-8
	top = float(top) if top is not None else sx.max()+1e-8

	if numpy.isinf(bottom):
		bottom = sx.min()
	if numpy.isinf(top):
		top = sx.max()

	if maxrange is not None:
		if maxrange[0] > bottom:
			bottom = float(maxrange[0])
		if maxrange[1] < top:
			top = float(maxrange[1])

	if bottom > top: # unclear how this would happen, but just in case
		bottom, top = top, bottom

	# generate number of bins based on named algorithm
	if isinstance(bins, str):
		try:
			width = _hist_bin_selectors[bins](sx)
		except IndexError:
			width = 1
		if numpy.isnan(width):
			width = 1
		try:
			if width:
				bins = int(numpy.ceil((top - bottom) / width))
			else:
				bins = 1
		except OverflowError:
			bins = 10
		# The spark graphs get hard to read if the bin slices are too thin, so we will max out at 50 bins
		if bins > 50:
			bins = 50
		# The spark graphs look weird if the bin slices are too fat, so we will min at 10 bins
		if bins < 10:
			bins = 10

	# generate actual bins based on integer number of bins
	if isinstance(bins, int):
		try:
			bins = numpy.linspace(bottom, top, num=bins, endpoint=True, retstep=False, dtype=None)
		except UnboundLocalError:
			raise

	return bins, bottom, top, zeros_are_dropped, range_is_truncated, sx, pct_under, pct_over


from numpy.ma.core import nomask as _nomask
import pandas

def _compress_by_mask(arr, mask=None):
	if arr is None:
		return None
	if isinstance(arr, numpy.ma.core.MaskedArray):
		if mask is None:
			mask = arr
		arr = arr._data
	if isinstance(arr, pandas.Series):
		arr = arr.data
	if isinstance(arr, memoryview):
		arr = numpy.asarray(arr)
	data = numpy.ndarray.ravel(arr)
	if mask._mask is not _nomask:
		data = data.compress(numpy.logical_not(numpy.ndarray.ravel(mask._mask)))
	return data


def histogram_data(
		data_to_bin,
		bins='sturges',
		weights=None,
		ch_weights=None,
		density=None,
		percentiles_to_cut=1,
		maxrange=None,
		fixedrange=None,
		threshold_for_dropping_zeros=0.35
):
	bins, bottom, top, zeros_are_dropped, range_is_truncated, sx, pct_under, pct_over = get_histogram_bins(
		data_to_bin, bins=bins, percentiles_to_cut=percentiles_to_cut, maxrange=maxrange, fixedrange=fixedrange, threshold_for_dropping_zeros=threshold_for_dropping_zeros,
	)

	h, bin_edges = numpy.histogram(_compress_by_mask(sx), bins=bins, range=(bottom, top), weights=_compress_by_mask(weights,sx), density=False)
	if ch_weights is not None:
		ch_h, _ = numpy.histogram(_compress_by_mask(sx), bins=bin_edges, weights=_compress_by_mask(ch_weights,sx), density=False, range=(bottom, top))
	else:
		ch_h = None

	if density:
		if fixedrange is not None and ch_weights is not None:
			# Get adjusted density for choice heights
			db = numpy.array(numpy.diff(bin_edges), float)
			if weights is None:
				h = h / db / sx._data.size
			else:
				h = h / db / weights.sum()
			ch_h = ch_h / db / ch_weights.sum()
		else:
			db = numpy.array(numpy.diff(bin_edges), float)
			h = h / db / h.sum()
			ch_h = ch_h / db / ch_h.sum()


	return h, bin_edges, zeros_are_dropped, range_is_truncated, ch_h, pct_under, pct_over


_histogram_notes = {
	hexcolor('orange'): "Histograms are orange if the zeros are numerous and have been excluded.",
	hexcolor('red'):    "Histograms are red if the zeros are numerous and have been excluded, and the displayed range truncates some extreme outliers.",
	hexcolor('forest'): "Histograms are green if the displayed range truncates some extreme outliers.",
	hexcolor('ocean'):  None,
	hexcolor('night'):  "Histograms are purple if the data is represented as discrete values.",
}

def draw_histogram_figure(
		bin_heights, bin_edges, zeros_are_dropped, range_is_truncated,
		title=None, xlabel=None, ylabel=None,
		xticks=False, yticks=False,
		xticklabels=None, yticklabels=False,
		frame_off='all', notetaker=None, bgcolor=None,
		figwidth=0.75, figheight=0.2,
		return_format='svg',
		attach_metadata=True,
		ch_heights=None,
		discrete=False,
		left_thermo=None,
		right_thermo=None,
	):
	from matplotlib import pyplot as plt
	fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
	widths = bin_edges[1:] - bin_edges[:-1]
	if discrete:
		widths = widths * 0.8
		color = hexcolor('night')
	elif zeros_are_dropped:
		if range_is_truncated and not left_thermo and not right_thermo:
			color = hexcolor('red')
		else:
			color = hexcolor('orange')
	elif range_is_truncated and not left_thermo and not right_thermo:
		color = hexcolor('forest')
	else:
		color = hexcolor('ocean')
	rects1 = ax.bar(
		bin_edges[:-1],
		bin_heights,
		widths,
		color=color,
		align= 'center' if discrete else 'edge'
	)

	chart_top = bin_heights.max()

	if ch_heights is not None:
		rects2 = ax.bar(
			bin_edges[:-1],
			height=numpy.zeros_like(ch_heights),
			width=widths,
			bottom=ch_heights,
			edgecolor='black',
			linewidth=1.0,
			align='center' if discrete else 'edge',
			clip_on=False
		)
		chart_top = max(chart_top, ch_heights.max())

	if left_thermo is not None:
		leftmost, rightmost = bin_edges[0], bin_edges[-1]
		full_width = rightmost-leftmost
		ax.bar(
			numpy.asarray([leftmost - full_width * 0.05,]),
			height=chart_top * left_thermo,
			width=numpy.asarray([full_width * 0.035, ]),
			color='red',
			align='edge'
		)
		ax.bar(
			numpy.asarray([leftmost - full_width * 0.05,]),
			height=chart_top * (1-left_thermo),
			width=numpy.asarray([full_width * 0.035, ]),
			bottom=chart_top * left_thermo,
			color='#e9edd3',
			align='edge'
		)
		# Thermo-border
		ax.bar(
			numpy.asarray([leftmost - full_width * 0.05,]),
			height=chart_top,
			width=numpy.asarray([full_width * 0.035, ]),
			color='#FFFFFF00',
			align='edge',
			edgecolor = 'black',
			linewidth = 0.5,
			clip_on=False
		)

	if right_thermo is not None:
		leftmost, rightmost = bin_edges[0], bin_edges[-1]
		full_width = rightmost-leftmost
		ax.bar(
			numpy.asarray([rightmost + full_width * 0.015,]),
			height=chart_top * right_thermo,
			width=numpy.asarray([full_width * 0.035, ]),
			color='red',
			align='edge'
		)
		ax.bar(
			numpy.asarray([rightmost + full_width * 0.015,]),
			height=chart_top * (1-right_thermo),
			width=numpy.asarray([full_width * 0.035, ]),
			bottom=chart_top * right_thermo,
			color='#e9edd3',
			align='edge'
		)
		# Thermo-border
		ax.bar(
			numpy.asarray([rightmost + full_width * 0.015,]),
			height=chart_top,
			width=numpy.asarray([full_width * 0.035, ]),
			color='#FFFFFF00',
			align='edge',
			edgecolor = 'black',
			linewidth = 0.5,
			clip_on=False
		)


	if figheight:
		fig.set_figheight(figheight)
	if figwidth:
		fig.set_figwidth(figwidth)
	fig.set_dpi(300)

	if frame_off == 'all' or frame_off is True or frame_off==1:
		frame_off = {'left', 'right', 'top', 'bottom'}
	elif frame_off is None:
		frame_off = set()
	elif not isinstance(frame_off, set):
		print(type(frame_off))
		print(frame_off)
		frame_off = set(frame_off)

	if xlabel: ax.set_xlabel(xlabel)
	if ylabel: ax.set_ylabel(ylabel)
	if title: ax.set_title(title)
	if not isinstance(xticks, (list,tuple,numpy.ndarray)) and not xticks:
		ax.get_xaxis().set_ticks([])
	else:
		frame_off.discard('bottom')
		if xticklabels is not None:
			ax.get_xaxis().set_ticklabels(xticklabels)
			ax.get_xaxis().set_ticks(xticks)
	if not isinstance(yticks, (list,tuple,numpy.ndarray)) and not yticks:
		ax.get_yaxis().set_ticks([])
	else:
		frame_off.discard('left')
		if yticklabels is not None:
			ax.get_yaxis().set_ticklabels(yticklabels)
			ax.get_yaxis().set_ticks(yticks)

	if frame_off is not None:
		if isinstance(frame_off, str):
			frame_off = [frame_off]
		from .plotting import adjust_spines
		sides = {'left', 'right', 'top', 'bottom'}
		for side in frame_off:
			sides.remove(side)
		adjust_spines(ax, sides)

	if bgcolor is not None:
		ax.set_axis_bgcolor(bgcolor)

	if notetaker is not None and color != hexcolor('ocean'):
		notetaker.add(_histogram_notes[color])

	if return_format.lower() == 'svg':
		from .plotting import plot_as_svg_xhtml
		try:
			if 'left' in frame_off and 'bottom' in frame_off:
				plt.tight_layout(pad=0)
			elif figheight<=1.0:
				plt.tight_layout(pad=figheight/2)
			else:
				plt.tight_layout(pad=0.5)
		except ValueError:
			pass # tight layout sometimes fails when the figsize is too small
		tooltip = _histogram_notes[color] or ""
		if ch_heights is not None:
			if len(tooltip):
				tooltip += "\n"
			tooltip += "Black marks indicate frequencies considering only chosen alternatives."
		if left_thermo or right_thermo:
			if len(tooltip):
				tooltip += "\n"
			tooltip += "Side thermometers indicate the fraction of all observations at or beyond the limits, and are scaled independently."

		ret = plot_as_svg_xhtml(plt, tooltip=tooltip)
		plt.clf()
		plt.close()  # do not showing empty windows?
		if attach_metadata:
			ret.metadata = dict(
				bin_heights=bin_heights,
				bin_edges=bin_edges,
				zeros_are_dropped=zeros_are_dropped,
				range_is_truncated=range_is_truncated)
			ret.attrib['zeros_are_dropped'] = str(zeros_are_dropped)
			ret.attrib['range_is_truncated'] = str(range_is_truncated)
		ret.attrib['style'] = f"min-width:{figwidth}in"
		return ret

def histogram_figure(
		data_to_bin,
		bins='sturges',
		weights=None,
		ch_weights=None,
		return_all=False,
		percentiles_to_cut=1,
		maxrange=None,
		fixedrange=None,
		piecerange=None,
		mask=None,
		**kwargs
):
	density = None if ch_weights is None else True
	discrete = (bins=='discrete')

	if mask is not None:
		data_to_bin = data_to_bin[mask]
		if weights is not None:
			weights = weights[mask]
		if ch_weights is not None:
			ch_weights = ch_weights[mask]

	threshold_for_dropping_zeros = 0.35

	if piecerange is not None and piecerange != (None,None):
		if piecerange[0] is None:
			fixedrange = (None, piecerange[1])
		elif piecerange[1] is None:
			fixedrange = (0, data_to_bin.max()+1e-8)
			threshold_for_dropping_zeros = 100
		else:
			fixedrange = (0, piecerange[1]-piecerange[0])
			threshold_for_dropping_zeros = 100
		percentiles_to_cut = 0
		discrete = False

	if discrete:
		bin_labels, bin_inv = numpy.unique(data_to_bin, return_inverse = True)
		bin_heights = numpy.bincount(bin_inv, weights)
		bin_edges = numpy.arange(len(bin_heights)+1)
		zeros_are_dropped, range_is_truncated = False, False
		if ch_weights is not None:
			ch_heights = numpy.bincount(bin_inv, ch_weights)
		else:
			ch_heights = None
		if density:
			bin_heights = bin_heights/(bin_heights.sum())
			if ch_heights is not None:
				ch_heights = ch_heights/(ch_heights.sum())
		if (('xticks' not in kwargs) and (kwargs.get('figwidth',1)>=2)) or (kwargs.get('xticks') is True):
			kwargs['xticklabels'] = bin_labels
			kwargs['xticks'] = bin_edges[:-1]
		pct_under, pct_over = 0,0
	else:
		bin_heights, bin_edges, zeros_are_dropped, range_is_truncated, ch_heights, pct_under, pct_over = histogram_data(
			data_to_bin, bins, weights, ch_weights, density=density, percentiles_to_cut=percentiles_to_cut, maxrange=maxrange, fixedrange=fixedrange, threshold_for_dropping_zeros=threshold_for_dropping_zeros
		)
		bin_labels = None

	if piecerange is not None and piecerange[0] is not None:
		bin_edges += piecerange[0]

	try:
		f = draw_histogram_figure(
			bin_heights, bin_edges, zeros_are_dropped, range_is_truncated,
			ch_heights = ch_heights,
			discrete=discrete,
			left_thermo = pct_under or None,
			right_thermo=pct_over or None,

			**kwargs
		)
	except:
		print("bin_heights\n",bin_heights)
		print("bin_edges\n", bin_edges)
		print("zeros_are_dropped\n", zeros_are_dropped)
		print("range_is_truncated\n", range_is_truncated)
		raise
	if return_all:
		return f, bin_heights, bin_edges, zeros_are_dropped, range_is_truncated
	else:
		return f


def sizable_histogram_figure(*args, sizer=1, discrete=None, **kwargs):
	kwargs['figwidth'] = sizer
	kwargs['figheight'] = sizer / 3.5
	if sizer >= 2:
		kwargs['frame_off'] = {'top', 'right'}
		kwargs['xticks'] = True
		#kwargs['yticks'] = True
	if sizer < 4:
		kwargs.pop('title', None)
	if sizer < 3:
		kwargs.pop('xlabel', None)
		kwargs.pop('ylabel', None)
	if discrete is not None:
		if discrete:
			kwargs['bins'] = 'discrete'
	return histogram_figure(*args, **kwargs)

def seems_like_discrete_data(arr, dictionary=None):
	if numpy.issubdtype(arr.dtype, numpy.bool_):
		#print('seems_like_discrete_data? YES bool')
		return True
	else:
		pass
		#print('seems_like_discrete_data? not bool but',arr.dtype)
	if dictionary is None:
		if len(numpy.unique(arr[:100]))<6:
			if len(numpy.unique(arr[:1000])) < 6:
				if len(numpy.unique(arr)) < 6:
					#print('seems_like_discrete_data? YES uniques < 6')
					return True
		#print('seems_like_discrete_data? too many and no dictionary')
	else:
		uniq = numpy.unique(arr)
		not_in_dict = 0
		for i in uniq:
			if i not in dictionary:
				not_in_dict += 1
		if not_in_dict > 2:
			#print(f'seems_like_discrete_data? dictionary but {not_in_dict} missing keys')
			return False
		else:
			#print(f'seems_like_discrete_data? dictionary with {not_in_dict} missing keys')
			return True
	return False