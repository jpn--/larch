

# Attempt to import the jupyter manager first.
# This will initialize the matplotlib module correctly before importing it.
try:
	from .. import jupyter
except:
	pass


from io import BytesIO
from ..util.xhtml import XHTML, XML_Builder
from ..util.arraytools import is_all_integer
import os

import matplotlib.pyplot as plt
import numpy

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

def plot_as_svg_xhtml(pyplot, classname='figure', headerlevel=2, header=None, anchor=1, **format):
	existing_format_keys = list(format.keys())
	for key in existing_format_keys:
		if key.upper()!=key: format[key.upper()] = format[key]
	if 'GRAPHWIDTH' not in format and 'GRAPHHEIGHT' in format: format['GRAPHWIDTH'] = format['GRAPHHEIGHT']
	if 'GRAPHWIDTH' in format and 'GRAPHHEIGHT' not in format: format['GRAPHHEIGHT'] = format['GRAPHWIDTH']*.67
	import xml.etree.ElementTree as ET
	ET.register_namespace("","http://www.w3.org/2000/svg")
	ET.register_namespace("xlink","http://www.w3.org/1999/xlink")
	imgbuffer = BytesIO()
#	if 'GRAPHWIDTH' in format and 'GRAPHHEIGHT' in format:
#		pyplot.figure(figsize=(format['GRAPHWIDTH'],format['GRAPHHEIGHT']))
	pyplot.savefig(imgbuffer, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='svg',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
	x = XML_Builder("div", {'class':classname})
	if header:
		x.hn(headerlevel, header, anchor=anchor)
	xx = x.close()
	xx << ET.fromstring(imgbuffer.getvalue().decode())
	return xx

def plt_as_svg_xhtml(classname='figure', headerlevel=2, header=None, anchor=1, clf_after=True):
	import xml.etree.ElementTree as ET
	ET.register_namespace("","http://www.w3.org/2000/svg")
	ET.register_namespace("xlink","http://www.w3.org/1999/xlink")
	imgbuffer = BytesIO()
	plt.savefig(imgbuffer, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='svg',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
	x = XML_Builder("div", {'class':classname})
	if header:
		x.hn(headerlevel, header, anchor=anchor)
	xx = x.close()
	_cache = ET.fromstring(imgbuffer.getvalue().decode())
	xx << _cache
	if clf_after:
		plt.clf()
	return xx


class default_mplstyle():

	def __enter__(self):
		from matplotlib import pyplot
		sty = os.path.join( os.path.dirname(__file__), 'larch.mplstyle' )
		self._contxt = pyplot.style.context((sty))
		self._contxt.__enter__()

	def __exit__(self, exc_type, exc_value, traceback):
		self._contxt.__exit__(exc_type, exc_value, traceback)


_color_rgb256 = {}
_color_rgb256['sky'] = (35,192,241)
_color_rgb256['ocean'] = (29,139,204)
_color_rgb256['night'] = (100,120,186)
_color_rgb256['forest'] = (39,182,123)
_color_rgb256['lime'] = (128,189,1)
_color_rgb256['orange'] = (246,147,0)
_color_rgb256['red'] = (246,1,0)

def hexcolor(color):
	c = _color_rgb256[color.casefold()]
	return "#{}{}{}".format(*(hex(c[i])[-2:] if c[i]>15 else "0"+hex(c[i])[-1:] for i in range(3)))

def strcolor_rgb256(color):
	c = _color_rgb256[color.casefold()]
	return "{},{},{}".format(*c)

_threshold_for_dropping_zeros_in_histograms = 0.35



_spark_histogram_notes = {
	hexcolor('orange'): "Histograms are orange if the zeros are numerous and have been excluded.",
	hexcolor('red'): "Histograms are red if the zeros are numerous and have been excluded, and the displayed range truncates some extreme outliers.",
	hexcolor('forest'): "Histograms are green if the displayed range truncates some extreme outliers.",
	hexcolor('ocean'): None,
}


def spark_histogram_rangefinder(data, bins):
	data = numpy.asarray(data)
	
	all_integer = issubclass(data.dtype.type, numpy.integer)
	if not all_integer:
		all_integer = is_all_integer_or_nan(data)

	# Check if data is mostly zeros
	n_zeros = (data==0).sum()
	if n_zeros > (data.size) * _threshold_for_dropping_zeros_in_histograms:
		use_data = data[data!=0]
		use_color = hexcolor('orange')
		zeros_have_been_dropped = True
	else:
		use_data = data
		use_color = hexcolor('ocean')
		zeros_have_been_dropped = False

	use_data = use_data[~numpy.isnan(use_data)]

	if use_data.size > 0:
		data_stdev = use_data.std()
		data_mean = use_data.mean()
		data_min = use_data.min()
		data_max = use_data.max()
		data_has_been_truncated = False
		bottom = data_min
		top = data_max
		if (data_min < (data_mean - 5*data_stdev)):
			bottom = numpy.nanpercentile(use_data,0.5)
			if data_min==0 and bottom>0 and not zeros_have_been_dropped:
				bottom = data_min
			else:
				data_has_been_truncated = True
		if (data_max > (data_mean + 5*data_stdev)):
			top = numpy.nanpercentile(use_data,99.5)
			if data_max==0 and top<0 and not zeros_have_been_dropped:
				top = data_max
			else:
				data_has_been_truncated = True
		if data_has_been_truncated:
			use_data = use_data[ (use_data>bottom) & (use_data<top) ]
			if zeros_have_been_dropped:
				use_color = hexcolor('red')
			else:
				use_color = hexcolor('forest')

	if isinstance(bins, str):
		if use_data.size == 0:
			# handle empty arrays. Can't determine range, so use 0-1.
			mn, mx = 0.0, 1.0
		else:
			mn, mx = use_data.min() + 0.0, use_data.max() + 0.0
		width = numpy.lib.function_base._hist_bin_selectors[bins](use_data)
		try:
			if width:
				bins = int(numpy.ceil((mx - mn) / width))
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

	# whatever else, if the data is integer then no more bins than values
	if all_integer and bins > (top-bottom+1):
		bins = (top-bottom+1)
	
	return bottom, top, use_color, bins

def spark_histogram_maker(data, bins=20, title=None, xlabel=None, ylabel=None, xticks=False, yticks=False,
						  frame=False, notetaker=None, prerange=None, duo_filter=None, data_for_bins=None,
						  figwidth=0.75, figheight=0.2, subplots_adjuster=0,
						  **silently_ignore_other_kwargs):

	data = numpy.asarray(data)
	if data_for_bins is None:
		use_data_for_bins = data
	else:
		use_data_for_bins = numpy.asarray(data_for_bins)
	if duo_filter is not None: duo_filter = numpy.asarray(duo_filter)

	use_data_for_bins = use_data_for_bins[~numpy.isnan(use_data_for_bins)]

	# Check if data is mostly zeros
	n_zeros = (use_data_for_bins==0).sum()
	if n_zeros > (use_data_for_bins.size) * _threshold_for_dropping_zeros_in_histograms:
		if duo_filter is not None: use_duo_filter = duo_filter[data!=0]
		use_data = data[data!=0]
		use_data_for_bins = use_data_for_bins[use_data_for_bins!=0]
		use_color = hexcolor('orange')
	else:
		if duo_filter is not None: use_duo_filter = duo_filter
		use_data = data
		use_color = hexcolor('ocean')
	bgcolor = None

	if duo_filter is not None: use_duo_filter = use_duo_filter[~numpy.isnan(use_data)]
	use_data = use_data[~numpy.isnan(use_data)]

	if use_data_for_bins.size > 0:
		data_stdev = use_data_for_bins.std()
		data_mean = use_data_for_bins.mean()
		data_min = use_data_for_bins.min()
		data_max = use_data_for_bins.max()
		if (data_min < data_mean - 5*data_stdev) or data_max > data_mean + 5*data_stdev:
			if (data_min < data_mean - 5*data_stdev):
				bottom = numpy.nanpercentile(use_data_for_bins,0.5)
			else:
				bottom = data_min
			if data_max > data_mean + 5*data_stdev:
				top = numpy.nanpercentile(use_data_for_bins,99.5)
			else:
				top = data_max
			if duo_filter is not None:
				use_duo_filter = use_duo_filter[(use_data>=bottom) & (use_data<=top)]
			use_data = use_data[ (use_data>=bottom) & (use_data<=top) ]
			use_data_for_bins = use_data_for_bins[ (use_data_for_bins>=bottom) & (use_data_for_bins<=top) ]
			if use_color == hexcolor('orange'):
				use_color = hexcolor('red')
			else:
				use_color = hexcolor('forest')
		else:
			bottom = data_min
			top = data_max

#	if use_data.size > 0:
#		data_stdev = use_data.std()
#		data_mean = use_data.mean()
#		data_min = use_data.min()
#		data_max = use_data.max()
#		if (data_min < data_mean - 5*data_stdev) or data_max > data_mean + 5*data_stdev:
#			bottom = numpy.nanpercentile(use_data,0.5)
#			top = numpy.nanpercentile(use_data,99.5)
#			if duo_filter is not None:
#				use_duo_filter = use_duo_filter[(use_data>bottom) & (use_data<top)]
#			use_data = use_data[ (use_data>bottom) & (use_data<top) ]
#			if use_color == hexcolor('orange'):
#				use_color = hexcolor('red')
#			else:
#				use_color = hexcolor('forest')

	if isinstance(bins, str):
		if use_data_for_bins.size == 0:
			# handle empty arrays. Can't determine range, so use 0-1.
			mn, mx = 0.0, 1.0
		else:
			mn, mx = numpy.nanmin(use_data_for_bins) + 0.0, numpy.nanmax(use_data_for_bins) + 0.0
		try:
			width = numpy.lib.function_base._hist_bin_selectors[bins](use_data_for_bins)
		except IndexError:
			width = 1
		if numpy.isnan(width):
			width = 1
		try:
			if width:
				bins = int(numpy.ceil((mx - mn) / width))
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

	try:
		bins = numpy.linspace(bottom, top, num=bins, endpoint=True, retstep=False, dtype=None)
	except UnboundLocalError:
		pass

#	if prerange is None:
#		bottom, top, use_color, bins = spark_histogram_rangefinder(use_data, bins)
#	else:
#		bottom, top, use_color, bins = prerange

	if duo_filter is None:
		plt.clf()
		try:
			n, bins, patches = plt.hist(use_data, bins, normed=1, facecolor=use_color, linewidth=0, alpha=1.0)
		except:
			print("<data>\n",data,"</data>")
			print("<bins>\n",bins,"</bins>")
			raise

		fig = plt.gcf()
		fig.set_figheight(figheight)
		fig.set_figwidth(figwidth)
		fig.set_dpi(300)
		if xlabel: plt.xlabel(xlabel)
		if ylabel: plt.ylabel(ylabel)
		if title: plt.title(title)
		if not xticks: fig.axes[0].get_xaxis().set_ticks([])
		if not yticks: fig.axes[0].get_yaxis().set_ticks([])
		if not frame: fig.axes[0].axis('off')

		if bgcolor is not None:
			ax = plt.gca()
			ax.set_axis_bgcolor(bgcolor)

		plt.subplots_adjust(left=0+subplots_adjuster, bottom=0+subplots_adjuster, right=1-subplots_adjuster, top=1-subplots_adjuster, wspace=0, hspace=0)
		ret = plot_as_svg_xhtml(fig)
		plt.clf()
	else:
		ret = [None,None]
		for duo in (1,0):

			plt.clf()
			try:
				use_filt_data = use_data[use_duo_filter if duo else ~use_duo_filter]
				n, bins, patches = plt.hist(use_filt_data, bins, normed=1, facecolor=use_color, linewidth=0, alpha=1.0)
			except:
				print("<data>\n",data,"</data>")
				print("<bins>\n",bins,"</bins>")
				raise
			fig = plt.gcf()
			fig.set_figheight(0.2)
			fig.set_figwidth(0.75)
			fig.set_dpi(300)
			if xlabel: plt.xlabel(xlabel)
			if ylabel: plt.ylabel(ylabel)
			if title: plt.title(title)
			if not xticks: fig.axes[0].get_xaxis().set_ticks([])
			if not yticks: fig.axes[0].get_yaxis().set_ticks([])
			if not frame: fig.axes[0].axis('off')

			if bgcolor is not None:
				ax = plt.gca()
				ax.set_axis_bgcolor(bgcolor)

			plt.subplots_adjust(left=0+subplots_adjuster, bottom=0+subplots_adjuster, right=1-subplots_adjuster, top=1-subplots_adjuster, wspace=0, hspace=0)
			ret[duo] = plot_as_svg_xhtml(fig)
			plt.clf()

	if notetaker is not None and use_color!=hexcolor('ocean'):
		notetaker.add( _spark_histogram_notes[use_color] )

	return ret





def spark_pie_maker(data, notetaker=None, figheight=0.2, figwidth=0.75, labels=None, show_labels=False, shadow=False,
					subplots_adjuster=0, explode=None, wedge_linewidth=0, tight=False, frame=False,
					**kwargs):
	plt.clf()
	fig = plt.gcf()
	fig.set_figheight(figheight)
	fig.set_figwidth(figwidth)
	fig.set_dpi(300)
	C_sky = (35,192,241)
	C_night = (100,120,186)
	C_forest = (39,182,123)
	C_ocean = (29,139,204)
	C_lime = (128,189,1)
	lab = None
	if show_labels:
		lab = [str(i) for i in labels]
	# The slices will be ordered and plotted counter-clockwise.
	plt.pie(data, explode=None, labels=lab,
			colors=[hexcolor('sky'),hexcolor('night'),hexcolor('forest'),hexcolor('ocean')],
			#autopct='%1.1f%%',
			shadow=shadow, startangle=90,
			wedgeprops={'linewidth':wedge_linewidth, 'clip_on':False, 'joinstyle':'round'},
			frame=False,
			)
	plt.subplots_adjust(left=0+subplots_adjuster, bottom=0+subplots_adjuster, right=1-subplots_adjuster, top=1-subplots_adjuster, wspace=0, hspace=0)
	# Set aspect ratio to be equal so that pie is drawn as a circle.
	plt.axis('equal')
	#if tight:
	#	plt.tight_layout()
	ret = plot_as_svg_xhtml(fig)
	plt.clf()
	return ret


def spark_category_bar_maker(data, notetaker=None, figheight=0.2, figwidth=0.75, labels=None, show_labels=False, shadow=False,
					subplots_adjuster=0, explode=None, linewidth=0, tight=False, frame=False, gap=0.2, tilt_thresh=35,
					**kwargs):
	plt.clf()
	fig = plt.gcf()
	fig.set_figheight(figheight)
	fig.set_figwidth(figwidth)
	fig.set_dpi(300)
	C_sky = (35,192,241)
	C_night = (100,120,186)
	C_forest = (39,182,123)
	C_ocean = (29,139,204)
	C_lime = (128,189,1)
	lab = None
	rotation = 0
	if show_labels and labels is not None and len(labels)>0:
		lab = [str(i) for i in labels]
		lab_len = max([len(i) for i in lab])
		if (lab_len*len(lab))>tilt_thresh:
			lab = ["{0: ^{1}s}".format(i,lab_len) for i in lab]
			rotation=23
	# The slices will be ordered and plotted counter-clockwise.
	ind = numpy.arange(len(data))
	plt.bar(ind+(gap/2.), data, width=1.0-gap, color=hexcolor('night'), linewidth=linewidth)
	if lab is not None:
		plt.xticks(ind + 1/2., lab, rotation=rotation)
	ret = plot_as_svg_xhtml(fig)
	plt.clf()
	return ret


def spark_histogram(data, *arg, pie_chart_cutoff=4, notetaker=None, prerange=None, duo_filter=None,
					data_for_bins=None, pie_chart_type='pie', dictionary=None, **kwarg):
	if prerange is not None:
		return spark_histogram_maker(data, *arg, notetaker=notetaker, prerange=prerange, duo_filter=duo_filter, data_for_bins=data_for_bins, **kwarg)
	try:
		flat_data = data.flatten()
	except:
		flat_data = data
	
	flat_data_nonnan = flat_data[~numpy.isnan(flat_data)]

	spark_pie_or_bar_maker = spark_pie_maker
	if pie_chart_type == 'bar':
		spark_pie_or_bar_maker = spark_category_bar_maker

	uniq = numpy.unique(flat_data_nonnan[:100])
	uniq_counts = None
	pie_chart_cutoff = int(pie_chart_cutoff)
	if len(uniq)<=pie_chart_cutoff:
		if duo_filter is not None:
			uniq0, uniq_counts0 = numpy.unique(flat_data_nonnan[~duo_filter], return_counts=True)
			uniq1, uniq_counts1 = numpy.unique(flat_data_nonnan[duo_filter], return_counts=True)
			if dictionary:
				uniq0 = [(dictionary[i] if i in dictionary else i) for i in uniq0]
				uniq1 = [(dictionary[i] if i in dictionary else i) for i in uniq1]
		else:
			uniq, uniq_counts = numpy.unique(flat_data_nonnan, return_counts=True)
			if dictionary:
				uniq = [(dictionary[i] if i in dictionary else i) for i in uniq]

	if uniq_counts is not None and len(uniq_counts)<=pie_chart_cutoff:
		if notetaker is not None:
			if pie_chart_type=='pie':
				notetaker.add( "Graphs are represented as pie charts if the data element has {} or fewer distinct values.".format(pie_chart_cutoff) )
			else:
				notetaker.add( "Graphs are represented as categorical bar charts if the data element has {} or fewer distinct values.".format(pie_chart_cutoff) )
		if duo_filter is not None:
			return spark_pie_or_bar_maker(uniq_counts0, labels=uniq0, **kwarg), spark_pie_or_bar_maker(uniq_counts1, labels=uniq1, **kwarg)
		else:
			return spark_pie_or_bar_maker(uniq_counts, labels=uniq, **kwarg)
	return spark_histogram_maker(data, *arg, notetaker=notetaker, duo_filter=duo_filter, data_for_bins=data_for_bins, **kwarg)



### ORIGINAL CODE
#
#def computed_factor_figure(m, y_funcs, y_labels=None,
#						   max_x=1, min_x=0, header=None,
#                           xaxis_label=None, yaxis_label=None,
#						   logscale_x=False, logscale_f=False, figsize=(6.5,3)):
#	from matplotlib import pyplot as plt
#	import numpy as np
#	with default_mplstyle():
#		x = np.linspace(min_x, max_x)
#		y = []
#		for yf in y_funcs:
#			y.append( yj(x,m) )
##		y1 = x*P(var+'1', default_value=0).value(m)+x**2*P(var+'2', default_value=0).value(m)+x**3*P(var+'3', default_value=0).value(m)
##		if 'LowIncome_{}'.format(var) in m:
##			y2 = x*P('LowIncome_'+var).value(m)+x**2*P('LowIncome_'+var+'^2', default_value=0).value(m)+x**3*P('LowIncome_'+var+'^3', default_value=0).value(m)
##		else:
##			y2 = None
##		if 'HighIncome_{}'.format(var) in m:
##			y3 = x*P('HighIncome_'+var).value(m)+x**2*P('HighIncome_'+var+'^2', default_value=0).value(m)+x**3*P('HighIncome_'+var+'^3', default_value=0).value(m)
##		else:
##			y3 = None
#		fig = plt.figure(figsize=figsize)
#		ax = plt.subplot(111)
#		ax.set_xlim(min_x,max_x)
#		if logscale_x:
#			ax.set_xscale('log', nonposx='clip', nonposy='clip')
#		if x_label is not None:
#			ax.set_xlabel(xaxis_label)
#		if y_label is not None:
#			ax.set_ylabel(yaxis_label)
#		lgnd_hands = []
#		
#		for n, iy in enumerate(y):
#			if y_labels and len(y_labels)>n:
#				iy_label = y_labels[n]
#			else:
#				iy_label = None
#			l1=plt.plot(x, iy, linewidth=2, label=iy_label)
#			lgnd_hands += l1
#		
#
##		if y2 is not None:
##			l2=plt.plot(x, y1+y2, linewidth=2, label='Low Income', color='r')
##			lgnd_hands += l2
##		if y3 is not None:
##			l3=plt.plot(x, y1+y3, linewidth=2, label='High Income', color='g')
##			lgnd_hands += l3
##		if y2 is None and y3 is None:
##			l1=plt.plot(x, y1, linewidth=2, label='All Travelers', color='b')
##			lgnd_hands += l1
##		else:
##			l1=plt.plot(x, y1, linewidth=2, label='Other Incomes', color='b')
##			lgnd_hands += l1
#		box = ax.get_position()
#		ax.set_position([box.x0, box.y0+0.1, box.width * 0.5, box.height-0.1])
#		ax.minorticks_on()
#		#ax.grid(b=True, which='major', color='0.2', linestyle='-', linewidth=0.1)
#
#		hist_vals = None
#		hist_wgts = None
#		# col_n = [n for n,c in enumerate(m.needs()['UtilityCA'].get_variables()) if 'log(attr_' in c][0]
#		# hist_wgts = np.exp(m.Data("UtilityCA")[:,:,col_n].flatten())
#		# col_n = [n for n,c in enumerate(m.needs()['UtilityCA'].get_variables()) if c=='SOV_Distance_AM'][0]
#		# hist_vals = m.Data("UtilityCA")[:,:,col_n].flatten()
#		# ax2 = plt.twinx(ax)
#		# if logscale_f:
#		# 	ax2.set_yscale('log', nonposx='clip', nonposy='clip')
#		# ax2.grid(b=False)
#		# ax2.set_ylabel('Frequency')
#		# ax2.set_position([box.x0, box.y0+0.1, box.width * 0.5, box.height-0.1])
#		# ax2.set_xlim(0,max_x)
#		# binspace = 50
#		# n,bins,patches = plt.hist(hist_vals, bins=binspace, histtype='stepfilled', normed=False, weights=hist_wgts, color='yellow', facecolor='yellow', label='Frequency among Attractions', linewidth=0.1)
#		# lgnd_hands += [patches[0],]
#		# ax.set_zorder(ax2.get_zorder()+1)
#
#		pushover = 1.05 if hist_vals is None or hist_wgts is None else 1.25
#		ax.legend(loc='center left', bbox_to_anchor=(pushover,0.5), handles=lgnd_hands, fontsize=9)
#		ax.patch.set_visible(False)
#		ax.set_xlim(min_x,max_x)
#	return plotting.plot_as_svg_xhtml(plt, header=header, headerlevel=2)
#



def computed_factor_figure(m, y_funcs, y_labels=None,
						   max_x=1, min_x=0, header=None,
                           xaxis_label=None, yaxis_label=None,
						   logscale_x=False, logscale_f=False, figsize=(6.5,3)):
	from matplotlib import pyplot as plt
	import numpy as np
	with default_mplstyle():
		x = np.linspace(min_x, max_x)
		y = []
		for yf in y_funcs:
			y.append( yj(x,m) )
		fig = plt.figure(figsize=figsize)
		ax = plt.subplot(111)
		ax.set_xlim(min_x,max_x)
		if logscale_x:
			ax.set_xscale('log', nonposx='clip', nonposy='clip')
		if x_label is not None:
			ax.set_xlabel(xaxis_label)
		if y_label is not None:
			ax.set_ylabel(yaxis_label)
		lgnd_hands = []
		
		for n, iy in enumerate(y):
			if y_labels and len(y_labels)>n:
				iy_label = y_labels[n]
			else:
				iy_label = None
			l1=plt.plot(x, iy, linewidth=2, label=iy_label)
			lgnd_hands += l1
		
		box = ax.get_position()
		ax.set_position([box.x0, box.y0+0.1, box.width * 0.5, box.height-0.1])
		ax.minorticks_on()

		hist_vals = None
		hist_wgts = None

		pushover = 1.05 if hist_vals is None or hist_wgts is None else 1.25
		ax.legend(loc='center left', bbox_to_anchor=(pushover,0.5), handles=lgnd_hands, fontsize=9)
		ax.patch.set_visible(False)
		ax.set_xlim(min_x,max_x)
	return plotting.plot_as_svg_xhtml(plt, header=header, headerlevel=2)




class ComputedFactor(tuple):
	def __new__(cls, label, func):
		return super().__new__(cls, (label, func))
	@property
	def label(self):
		return self[0]
	@property
	def func(self):
		return self[1]



def computed_factor_figure_v2(m, y_funcs, y_labels=None,
							   max_x=1, min_x=0, header=None,
							   xaxis_label=None, yaxis_label=None,
							   logscale_x=False, logscale_f=False, figsize=(6.5,3)):
	"""Plots a computed factor.
	
	Parameters
	----------
	m : Model
		The model underlying the computed factor.
	y_funcs : ComputedFactor or sequence of ComputedFactors
		A list or other sequence of ComputedFactor.  Each ComputedFactor.func should accept two positional
		parameters given as x,m where x is the array of data values for the relevant attribute, 
		and m is the model passed to this function.
	y_labels : sequence of str
		A set of labels to apply to the y_funcs.  If given, this should match the length of `y_funcs`.
	xaxis_label : str
		As expected.
	yaxis_label : str
		As expected.
	"""
	
	if isinstance(y_funcs, ComputedFactor):
		y_funcs = [y_funcs,]
	
	def maker(ref_to_m):
		from matplotlib import pyplot as plt
		import numpy as np
		with default_mplstyle():
			x = np.linspace(min_x, max_x)
			y = []
			y_labels = []
			for yf in y_funcs:
				y.append( yf.func(x,ref_to_m) )
				y_labels.append( yf.label )
			fig = plt.figure(figsize=figsize)
			ax = plt.subplot(111)
			ax.set_xlim(min_x,max_x)
			if logscale_x:
				ax.set_xscale('log', nonposx='clip', nonposy='clip')
			if xaxis_label is not None:
				ax.set_xlabel(xaxis_label)
			if yaxis_label is not None:
				ax.set_ylabel(yaxis_label)
			lgnd_hands = []
			
			for n, iy in enumerate(y):
				if y_labels and len(y_labels)>n:
					iy_label = y_labels[n]
				else:
					iy_label = None
				l1=plt.plot(x, iy, linewidth=2, label=iy_label)
				lgnd_hands += l1
			
			box = ax.get_position()
			ax.set_position([box.x0, box.y0+0.1, box.width * 0.5, box.height-0.1])
			ax.minorticks_on()

			hist_vals = None
			hist_wgts = None

			pushover = 1.05 if hist_vals is None or hist_wgts is None else 1.25
			ax.legend(loc='center left', bbox_to_anchor=(pushover,0.5), handles=lgnd_hands, fontsize=9)
			ax.patch.set_visible(False)
			ax.set_xlim(min_x,max_x)
		return plot_as_svg_xhtml(plt, header=header, headerlevel=2)
	m.add_to_report(maker)




def svg_computed_factor_figure_with_derivative(m, y_funcs,
							   max_x=1, min_x=0, header=None,
							   xaxis_label=None, yaxis_label=None,
							   logscale_x=False, logscale_f=False, figsize=(11,3),
							   epsilon=0.001, supress_legend=True,
							   short_header=None, headerlevel=2
							   ):
	"""Plots a computed factor and it's derivative.
	
	Parameters
	----------
	m : Model
		The model underlying the computed factor.
	y_funcs : LinearFunction or ComputedFactor or sequence of ComputedFactors
		A list or other sequence of ComputedFactor.  Each ComputedFactor.func should accept two positional
		parameters given as x,m where x is the array of data values for the relevant attribute, 
		and m is the model passed to this function.
	xaxis_label, yaxis_label : str
		As you might expect.
	"""
	
	from ..core import LinearFunction
	
	if isinstance(y_funcs, LinearFunction):
		y_funcs = y_funcs.evaluator1d()

	if isinstance(y_funcs, ComputedFactor):
		y_funcs = [y_funcs,]

	y_funcs = [(i.evaluator1d() if isinstance(i,LinearFunction) else i) for i in y_funcs]

	# We use the forward derivative here, not backward, because of the ubiquity of zero-bounded piecewise terms in utility functions.
	dy_funcs = [ComputedFactor(label="∂"+cf.label, func=(lambda x,m: (cf.func(x+epsilon,m) - cf.func(x,m))*(1/epsilon)), ) for cf in y_funcs]

	from matplotlib import pyplot as plt
	import numpy as np
	with default_mplstyle():
		x = np.linspace(min_x, max_x, 200) # Use 200 for extra resolution over default of 50
		y = []
		y_labels = []
		for yf in y_funcs:
			y.append( yf.func(x,m) )
			y_labels.append( yf.label )
		fig = plt.figure(figsize=figsize)
		ax = plt.subplot(121)
		ax.set_xlim(min_x,max_x)
		if logscale_x:
			ax.set_xscale('log', nonposx='clip', nonposy='clip')
		if xaxis_label is not None:
			ax.set_xlabel(xaxis_label)
		if yaxis_label is not None:
			ax.set_ylabel(yaxis_label)
		lgnd_hands = []
		
		for n, iy in enumerate(y):
			if y_labels and len(y_labels)>n:
				iy_label = y_labels[n]
			else:
				iy_label = None
			l1=plt.plot(x, iy, linewidth=2, label=iy_label)
			lgnd_hands += l1
		
		box = ax.get_position()
		if not supress_legend:
			ax.set_position([box.x0, box.y0+0.1, box.width * 0.5, box.height-0.1])
		else:
			ax.set_position([box.x0, box.y0+0.1, box.width-0.1, box.height-0.1])
		ax.minorticks_on()

		hist_vals = None
		hist_wgts = None

		pushover = 1.05 if hist_vals is None or hist_wgts is None else 1.25
		if not supress_legend:
			ax.legend(loc='center left', bbox_to_anchor=(pushover,0.5), handles=lgnd_hands, fontsize=9)
		ax.patch.set_visible(False)
		ax.set_xlim(min_x,max_x)

		# Marginals....
		dy = []
		dy_labels = []
		for yf in dy_funcs:
			dy.append( yf.func(x,m) )
			dy_labels.append( yf.label )
		ax = plt.subplot(122)
		ax.set_xlim(min_x,max_x)
		if logscale_x:
			ax.set_xscale('log', nonposx='clip', nonposy='clip')
		if xaxis_label is not None:
			ax.set_xlabel(xaxis_label)
		if yaxis_label is not None:
			ax.set_ylabel("Marginal "+yaxis_label)
		lgnd_hands = []
		
		for n, iy in enumerate(dy):
			if dy_labels and len(dy_labels)>n:
				iy_label = dy_labels[n]
			else:
				iy_label = None
			l1=plt.plot(x, iy, linewidth=2, label=iy_label)
			lgnd_hands += l1
		
		box = ax.get_position()
		if not supress_legend:
			ax.set_position([box.x0, box.y0+0.1, box.width * 0.5, box.height-0.1])
		else:
			ax.set_position([box.x0, box.y0+0.1, box.width-0.1, box.height-0.1])
		ax.minorticks_on()

		hist_vals = None
		hist_wgts = None

		pushover = 1.05 if hist_vals is None or hist_wgts is None else 1.25
		if not supress_legend:
			ax.legend(loc='center left', bbox_to_anchor=(pushover,0.5), handles=lgnd_hands, fontsize=9)
		ax.patch.set_visible(False)
		ax.set_xlim(min_x,max_x)

		return plt_as_svg_xhtml(header=header, headerlevel=headerlevel, anchor=short_header or header)

def new_xhtml_computed_factor_figure_with_derivative(self, figurename, y_funcs, header=None, short_header=None, headerlevel=2, autoregister=True, **kwargs):
	caller = lambda *arg, **kw: self.svg_computed_factor_figure_with_derivative(y_funcs, header=header, short_header=short_header, headerlevel=headerlevel, **kwargs)
	self.new_xhtml_section(caller, figurename, register=autoregister)



def svg_validation_distribution(m, factorarray, range, bins, headerlevel, header, short_header=None, to_report=True, immediate=False, log_scale=False, figsize=(6.5,3)):
	"""A figure showing the distribution of a factor across real and modeled observations.

	Parameters
	----------
	m : larch.Model (self)
	factorarray : ndarray [nCases, nAlts], or str
		The array of factors to validate on.  Or give the name of an idca variable.
	range : tuple
	bins : int or str
		These get passed to plt.hist.
	headerlevel : int
	header : str
	short_header : str or None
	"""
	if isinstance(factorarray,str):
		factorarray = m.df.array_idca(factorarray).squeeze()
	
	from matplotlib import pyplot as plt
	mod = m
	if mod.data.weight is None:
		pr = mod.work.probability[:, :mod.nAlts()]
		ch = mod.data.choice.squeeze()
	else:
		pr = mod.work.probability[:, :mod.nAlts()] * mod.data.weight
		ch = mod.data.choice.squeeze() * mod.data.weight
	plt.clf()
	fig = plt.figure(figsize=figsize)
	try:
		h1 = plt.hist(factorarray.flatten(), weights=pr.flatten(), histtype="stepfilled", bins=bins, alpha=0.7, normed=True, range=range, label='Modeled', log=log_scale)
	except (UnboundLocalError, ValueError):
		## matplotlib error sometimes here.  a bugfix is likely coming soon, but not yet in 1.5.3
		pass
	try:
		h2 = plt.hist(factorarray.flatten(), weights=ch.flatten(), histtype="stepfilled", bins=bins, alpha=0.7, normed=True, range=range, label='Observed', log=log_scale)
	except (UnboundLocalError, ValueError):
		## matplotlib error sometimes here.  a bugfix is likely coming soon, but not yet in 1.5.3
		pass
	plt.legend()
	return plt_as_svg_xhtml(header=header, headerlevel=headerlevel, anchor=short_header or header)



def new_xhtml_validation_distribution(self, figurename, factorarray,
								 header=None, headerlevel=2, short_header=None,
								 autoregister=True,
								 **kwargs):
	"""
	A figure showing the distribution of a factor across real and modeled observations.
	
	Parameters
	----------
	figurename : str
		A name for the mapset
	factorarray : ndarray [nCases, nAlts], or str
		The array of factors to validate on.  Or give the name of an idca variable.
	header : str, optional
		A header to prepend to the mapset, as a <hN> tag.
	headerlevel : int, optional
		The level N in the header tag.
	short_header : str, optional
		A shortened version of the header, for the table of contents.
		
	Note
	----
	Other keyword arguments will be passed through to `svg_validation_distribution`
	when that function is called at report generation time.
	"""
	caller = lambda *arg, **kw: self.svg_validation_distribution(factorarray, *arg, **kw, **kwargs, headerlevel=headerlevel, header=header, short_header=short_header)
	self.new_xhtml_section(caller, figurename, register=autoregister)



def svg_observations_latlong(mod, lat, lon, extent=None, figsize=(6.0,3.0),
							gridsize=60, headfont='Roboto Slab', textfont='Roboto',
							colormap='rainbow', tight_layout=True,
							headerlevel=2, header=None, short_header=None):
	"""
	A validation mapset for destination choice and similar models.
	
	Parameters
	----------
	lat, lon : ndarray
		Latitude and Longitude of the zonal centroids. Should be vectors
		with length equal to number of alternatives.
	extent : None or 4-tuple
		The limits of the map, give as (west_lon,east_lon,south_lat,north_lat).  If 
		None the limits are found automatically.
	figsize : tuple
		The (width,height) for the figure.
	show_diffs : {'linear', 'log', False}
		Whether to include a differences map, with linear or log scale.
	scaled_diffs : bool
		Include a scaled version of the diffs, scaled by log(obs). This de-emphasizes
		larger diffs when the total trips to the zone is very large.
	header : str, optional
		A header to prepend to the mapset, as a <hN> tag.
	headerlevel : int, optional
		The level N in the header tag.
	short_header : str, optional
		A shortened version of the header, for the table of contents.
	"""
	from matplotlib import pyplot as plt
	import matplotlib.colors as colors
	from matplotlib.ticker import LogLocator
	import matplotlib.cm as cm
	n_subplots = 1
	plot_n = 1
	if mod.data.weight is None:
		pr = mod.work.probability[:, :mod.nAlts()]
		ch = mod.data.choice.squeeze()
		wlabel = ""
	else:
		pr = mod.work.probability[:, :mod.nAlts()] * mod.data.weight
		ch = mod.data.choice.squeeze() * mod.data.weight
		wlabel = "Weighted "
	pr_0 = pr.sum(0).flatten()
	ch_0 = ch.sum(0).flatten()
	plt.clf()
	fig = plt.figure(figsize=figsize, tight_layout=tight_layout)

	def next_subplot(title=None, axisbg=None, ticks_off=True):
		ax = plt.subplot(n_subplots,1,next_subplot.plot_n, axisbg=axisbg)
		next_subplot.plot_n+=1
		if title:
			ax.set_title(title, fontname=headfont)
		if ticks_off:
			ax.get_xaxis().set_ticks([])
			ax.get_yaxis().set_ticks([])
		return ax

	next_subplot.plot_n = 1

	ax1 = next_subplot('Observed')
	hb1 = ax1.hexbin(lon, lat, C=ch_0, gridsize=gridsize, xscale='linear',
					 yscale='linear', bins=None, extent=extent, cmap=colormap, 
					 norm=None, alpha=None, linewidths=0.1, edgecolors='none', 
					 reduce_C_function=numpy.sum, mincnt=None, marginals=False, data=None, )

	# Renorm hb1 and hb2 to same scale
	renorm = colors.LogNorm()
	vst = numpy.vstack([hb1.get_array(),])
	vst = vst[vst!=0]
	renorm.autoscale_None( vst )
	hb1.set_norm( renorm )

	# Colorbars
	cb = plt.colorbar(hb1, ax=ax1, ticks=LogLocator(subs=range(10)))
	cb.set_label(wlabel+'Counts', fontname=textfont)
	for l in cb.ax.yaxis.get_ticklabels():
		l.set_family(textfont)

	return plt_as_svg_xhtml(headerlevel=headerlevel, header=header, anchor=short_header or header, clf_after=False)



def svg_validation_latlong(mod, lat, lon, extent=None, figsize=(6.0,10), show_diffs='linear', scaled_diffs=True,
							gridsize=60, headfont='Roboto Slab', textfont='Roboto',
							colormap='rainbow', tight_layout=True,
							headerlevel=2, header=None, short_header=None,
							colormin=None, colormax=None):
	"""
	A validation mapset for destination choice and similar models.
	
	Parameters
	----------
	lat, lon : ndarray
		Latitude and Longitude of the zonal centroids. Should be vectors
		with length equal to number of alternatives.
	extent : None or 4-tuple
		The limits of the map, give as (west_lon,east_lon,south_lat,north_lat).  If 
		None the limits are found automatically.
	figsize : tuple
		The (width,height) for the figure.
	show_diffs : {'linear', 'log', False}
		Whether to include a differences map, with linear or log scale.
	scaled_diffs : bool
		Include a scaled version of the diffs, scaled by log(obs). This de-emphasizes
		larger diffs when the total trips to the zone is very large.
	header : str, optional
		A header to prepend to the mapset, as a <hN> tag.
	headerlevel : int, optional
		The level N in the header tag.
	short_header : str, optional
		A shortened version of the header, for the table of contents.
	colormin, colormax : numeric
		If given use these values as the lower (upper) bound of the colormap normalization.
	"""
	from matplotlib import pyplot as plt
	import matplotlib.colors as colors
	from matplotlib.ticker import LogLocator
	import matplotlib.cm as cm
	n_subplots = 2
	if show_diffs:
		n_subplots += 1
	if scaled_diffs:
		n_subplots += 1
	plot_n = 1
	if mod.data.weight is None:
		pr = mod.work.probability[:, :mod.nAlts()]
		ch = mod.data.choice.squeeze()
		wlabel = ""
	else:
		pr = mod.work.probability[:, :mod.nAlts()] * mod.data.weight
		ch = mod.data.choice.squeeze() * mod.data.weight
		wlabel = "Weighted "
	pr_0 = pr.sum(0).flatten()
	ch_0 = ch.sum(0).flatten()
	plt.clf()
	fig = plt.figure(figsize=figsize, tight_layout=tight_layout)

	def next_subplot(title=None, axisbg=None, ticks_off=True):
		ax = plt.subplot(n_subplots,1,next_subplot.plot_n, axisbg=axisbg)
		next_subplot.plot_n+=1
		if title:
			ax.set_title(title, fontname=headfont)
		if ticks_off:
			ax.get_xaxis().set_ticks([])
			ax.get_yaxis().set_ticks([])
		return ax

	next_subplot.plot_n = 1

	ax1 = next_subplot('Observed')
	hb1 = ax1.hexbin(lon, lat, C=ch_0, gridsize=gridsize, xscale='linear',
					 yscale='linear', bins=None, extent=extent, cmap=colormap, 
					 norm=None, alpha=None, linewidths=0.1, edgecolors='none', 
					 reduce_C_function=numpy.sum, mincnt=None, marginals=False, data=None, )

	ax2 = next_subplot('Modeled')
	hb2 = ax2.hexbin(lon, lat, C=pr_0, gridsize=gridsize, xscale='linear',
					 yscale='linear', bins=None, extent=extent, cmap=colormap, 
					 norm=None, alpha=None, linewidths=0.1, edgecolors='none', 
					 reduce_C_function=numpy.sum, mincnt=None, marginals=False, data=None, )

	# Renorm hb1 and hb2 to same scale
	renorm = colors.LogNorm()
	vst = numpy.vstack([hb1.get_array(),hb2.get_array()])
	vst = vst[vst!=0]
	renorm.vmin=colormin
	renorm.vmax=colormax
	renorm.autoscale_None( vst )
	hb1.set_norm( renorm )
	hb2.set_norm( renorm )

	# Colorbars
	cb = plt.colorbar(hb1, ax=ax1, ticks=LogLocator(subs=range(10)))
	cb.set_label(wlabel+'Counts', fontname=textfont)
	for l in cb.ax.yaxis.get_ticklabels():
		l.set_family(textfont)

	cb = plt.colorbar(hb2, ax=ax2, ticks=LogLocator(subs=range(10)))
	cb.set_label(wlabel+'Total Prob', fontname=textfont)
	for l in cb.ax.yaxis.get_ticklabels():
		l.set_family(textfont)

	# Diffs plots
	pr_ch_diff = pr_0-ch_0
	if show_diffs:
		ax = next_subplot('Raw Over/Under-Prediction')
		hb = ax.hexbin(lon, lat, C=pr_ch_diff, gridsize=gridsize, xscale='linear',
					   yscale='linear', bins=None, extent=extent, 
					   cmap='bwr_r', norm=None, alpha=None, linewidths=0.1, 
					   edgecolors='none', reduce_C_function=numpy.sum, mincnt=None, 
					   marginals=False, data=None, )

		# Re-norm
		mid,top = numpy.percentile(numpy.abs(hb.get_array()),[66,99])
		if show_diffs=='log':
			norm = colors.SymLogNorm(linthresh=mid, linscale=0.025, vmin=-top, vmax=top)
		else:
			norm = colors.Normalize(vmin=-top, vmax=top)
		hb.set_norm( norm )
		cb = plt.colorbar(hb, ax=ax, extend='both')
		cb.set_label(wlabel+'Over / Under', fontname=textfont)
		for l in cb.ax.yaxis.get_ticklabels():
			l.set_family(textfont)
	else:
		mid,top = None,None


	if scaled_diffs:
		pr_ch_diff_ = hb2.get_array()-hb1.get_array()
		pr_ch_scale = numpy.log1p((hb1.get_array()))+1
		# Get bin centners
		verts = hb2.get_offsets()
		binx,biny = numpy.zeros_like(pr_ch_scale), numpy.zeros_like(pr_ch_scale)
		for offc in range(verts.shape[0]):
			binx[offc],biny[offc] = verts[offc][0],verts[offc][1]

		ax = next_subplot('Adjusted Over/Under-Prediction')
		hb = ax.hexbin(binx, biny, C=pr_ch_diff_/pr_ch_scale, gridsize=gridsize,
					   xscale='linear', yscale='linear', bins=None, extent=extent, 
					   cmap='bwr_r', norm=None, alpha=None, linewidths=0.1, 
					   edgecolors='none', reduce_C_function=numpy.mean, mincnt=None, 
					   marginals=False, data=None, )
		# Re-norm
		if mid is None or top is None:
			mid,top = numpy.percentile(numpy.abs(hb.get_array()),[66,99])
		if show_diffs=='log':
			norm = colors.SymLogNorm(linthresh=mid, linscale=0.025, vmin=-top, vmax=top)
		else:
			norm = colors.Normalize(vmin=-top, vmax=top)
		hb.set_norm( norm )
		plt.annotate('Raw Difference / (1+log(Observed Count+1))', xycoords='axes fraction', xy=(0.5,0),
						textcoords='offset points', xytext=(0,-2),
						horizontalalignment='center', verticalalignment='top',
						fontsize='x-small', fontstyle='italic')
		cb = plt.colorbar(hb, ax=ax, extend='both')
		cb.set_label(wlabel+'Over / Under', fontname=textfont)
		for l in cb.ax.yaxis.get_ticklabels():
			l.set_family(textfont)

	return plt_as_svg_xhtml(headerlevel=headerlevel, header=header, anchor=short_header or header)




def new_xhtml_validation_latlong(self, figurename, lat, lon,
								 header=None, headerlevel=2, short_header=None,
								 autoregister=True,
								 **kwargs):
	"""
	A validation mapset section for destination choice and similar models.
	
	Parameters
	----------
	figurename : str
		A name for the mapset
	lat, lon : ndarray
		Latitude and Longitude of the zonal centroids. Should be vectors
		with length equal to number of alternatives.
	header : str, optional
		A header to prepend to the mapset, as a <hN> tag.
	headerlevel : int, optional
		The level N in the header tag.
	short_header : str, optional
		A shortened version of the header, for the table of contents.
		
	Note
	----
	Other keyword arguments will be passed through to `svg_validation_latlong`
	when that function is called at report generation time.
	"""
	caller = lambda *arg, **kw: self.svg_validation_latlong(lat, lon, *arg, **kw, **kwargs, headerlevel=headerlevel, header=header, short_header=short_header)
	self.new_xhtml_section(caller, figurename, register=autoregister)




#def validation_latlong_figure(m, lat, lon, headerlevel, header, short_header=None, 
#                              figsize=(11,3), immediate=False, to_report=True, 
#                              showdiffs='linear', gridsize=60, headfont=('Roboto Slab',), 
#                              textfont=('Roboto',), scaled_diffs=True,
#                              colormap='rainbow'):
#	from matplotlib import pyplot as plt
#	import matplotlib.colors as colors
#	from matplotlib.ticker import LogLocator
#	import matplotlib.cm as cm
#	if short_header is None:
#		short_header = header
#	def map_validation_maker(mod):
#		n_subplots = 2
#		if showdiffs:
#			n_subplots += 1
#		if scaled_diffs:
#			n_subplots += 1
#		plot_n = 1
#		if mod.data.weight is None:
#			pr = mod.work.probability[:, :mod.nAlts()]
#			ch = mod.data.choice.squeeze()
#			wlabel = ""
#		else:
#			pr = mod.work.probability[:, :mod.nAlts()] * mod.data.weight
#			ch = mod.data.choice.squeeze() * mod.data.weight
#			wlabel = "Weighted "
#		pr_0 = pr.sum(0).flatten()
#		ch_0 = ch.sum(0).flatten()
#		half_vir_blank =  tuple((1-((1-i)*0.2)) for i in cm.viridis(1.0))
#		plt.clf()
#		fig = plt.figure(figsize=figsize)
#		ax1 = plt.subplot(n_subplots,1,plot_n, axisbg=None)
#		plot_n+=1
#		ax1.set_title('Observed', fontname=headfont)
#		hb1 = ax1.hexbin(lon, lat, C=ch_0, gridsize=gridsize, xscale='linear', 
#						 yscale='linear', bins=None, extent=None, cmap=colormap, 
#						 norm=None, alpha=None, linewidths=0.1, edgecolors='none', 
#						 reduce_C_function=numpy.sum, mincnt=None, marginals=False, data=None, )
#		ax1.get_xaxis().set_ticks([])
#		ax1.get_yaxis().set_ticks([])
#		ax2 = plt.subplot(n_subplots,1,plot_n, axisbg=None)
#		plot_n+=1
#		hb2 = ax2.hexbin(lon, lat, C=pr_0, gridsize=gridsize, xscale='linear', 
#						 yscale='linear', bins=None, extent=None, cmap=colormap, 
#						 norm=None, alpha=None, linewidths=0.1, edgecolors='none', 
#						 reduce_C_function=numpy.sum, mincnt=None, marginals=False, data=None, )
#		ax2.get_xaxis().set_ticks([])
#		ax2.get_yaxis().set_ticks([])
#		ax2.set_title('Modeled', fontname=headfont)
#		# Renorm hb1 and hb2 to same scale
#		renorm = colors.LogNorm()
#		vst = numpy.vstack([hb1.get_array(),hb2.get_array()])
#		vst = vst[vst!=0]
#		renorm.autoscale_None( vst )
#		hb1.set_norm( renorm )
#		hb2.set_norm( renorm )
#		# Colorbars
#		cb = plt.colorbar(hb1, ax=ax1, ticks=LogLocator(subs=range(10)))
#		cb.set_label(wlabel+'Counts', fontname=textfont)
#		for l in cb.ax.yaxis.get_ticklabels():
#			l.set_family(textfont)
#		cb = plt.colorbar(hb2, ax=ax2, ticks=LogLocator(subs=range(10)))
#		cb.set_label(wlabel+'Total Prob', fontname=textfont)
#		for l in cb.ax.yaxis.get_ticklabels():
#			l.set_family(textfont)
#		pr_ch_diff = pr_0-ch_0
#		# Diffs plot
#		if showdiffs:
#			ax = plt.subplot(n_subplots,1,plot_n)
#			plot_n+=1
#			hb = ax.hexbin(lon, lat, C=pr_ch_diff, gridsize=gridsize, xscale='linear', 
#						   yscale='linear', bins=None, extent=None, 
#						   cmap='bwr_r', norm=None, alpha=None, linewidths=0.1, 
#						   edgecolors='none', reduce_C_function=numpy.sum, mincnt=None, 
#						   marginals=False, data=None, )
#			ax.get_xaxis().set_ticks([])
#			ax.get_yaxis().set_ticks([])
#			# Re-norm
#			mid,top = numpy.percentile(numpy.abs(hb.get_array()),[66,97])
#			if showdiffs=='log':
#				norm = colors.SymLogNorm(linthresh=mid, linscale=0.025, vmin=-top, vmax=top)
#			else:
#				norm = colors.Normalize(vmin=-top, vmax=top)
#			hb.set_norm( norm )
#			ax.set_title('Raw Over/Under-Prediction', fontname=headfont)
#			cb = plt.colorbar(hb, ax=ax, extend='both')
#			cb.set_label(wlabel+'Over / Under', fontname=textfont)
#			for l in cb.ax.yaxis.get_ticklabels():
#				l.set_family(textfont)
#		else:
#			mid,top = None,None
#
#		
#		if scaled_diffs:
#			pr_ch_diff_ = hb2.get_array()-hb1.get_array()
#			pr_ch_scale = numpy.log1p((hb1.get_array()))+1
#			# Get bin centners
#			verts = hb2.get_offsets()
#			binx,biny = numpy.zeros_like(pr_ch_scale), numpy.zeros_like(pr_ch_scale)
#			for offc in range(verts.shape[0]):
#				binx[offc],biny[offc] = verts[offc][0],verts[offc][1]
#			ax = plt.subplot(n_subplots,1,plot_n)
#			plot_n+=1
#			hb = ax.hexbin(binx, biny, C=pr_ch_diff_/pr_ch_scale, gridsize=gridsize, 
#						   xscale='linear', yscale='linear', bins=None, extent=None, 
#						   cmap='bwr_r', norm=None, alpha=None, linewidths=0.1, 
#						   edgecolors='none', reduce_C_function=numpy.mean, mincnt=None, 
#						   marginals=False, data=None, )
#			ax.get_xaxis().set_ticks([])
#			ax.get_yaxis().set_ticks([])
#			# Re-norm
#			if mid is None or top is None:
#				mid,top = numpy.percentile(numpy.abs(hb.get_array()),[66,97])
#			if showdiffs=='log':
#				norm = colors.SymLogNorm(linthresh=mid, linscale=0.025, vmin=-top, vmax=top)
#			else:
#				norm = colors.Normalize(vmin=-top, vmax=top)
#			hb.set_norm( norm )
#			ax.set_title('Adjusted Over/Under-Prediction', fontname=headfont)
#			plt.annotate('Raw Difference / (1+log(Observed Count+1))', xycoords='axes fraction', xy=(0.5,0),
#							textcoords='offset points', xytext=(0,-5),
#							horizontalalignment='center', verticalalignment='top',
#							fontsize='small', fontstyle='italic')
#			cb = plt.colorbar(hb, ax=ax, extend='both')
#			cb.set_label(wlabel+'Over / Under', fontname=textfont)
#			for l in cb.ax.yaxis.get_ticklabels():
#				l.set_family(textfont)
#		return None #plot_as_svg_xhtml(plt, header=header, headerlevel=headerlevel, anchor=short_header)
#	if to_report:
#		m.add_to_report(map_validation_maker)
#	if immediate:
#		map_validation_maker(m)
#		plt.show()
#
