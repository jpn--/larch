
from io import BytesIO
from ..util.xhtml import XHTML, XML_Builder
import os

import matplotlib.pyplot as plt
import numpy

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

	# Check if data is mostly zeros
	n_zeros = (data==0).sum()
	if n_zeros > (data.size) * _threshold_for_dropping_zeros_in_histograms:
		use_data = data[data!=0]
		use_color = hexcolor('orange')
	else:
		use_data = data
		use_color = hexcolor('ocean')

	use_data = use_data[~numpy.isnan(use_data)]

	if use_data.size > 0:
		data_stdev = use_data.std()
		data_mean = use_data.mean()
		data_min = use_data.min()
		data_max = use_data.max()
		if (data_min < data_mean - 5*data_stdev) or (data_max > data_mean + 5*data_stdev):
			if (data_min < data_mean - 5*data_stdev):
				bottom = numpy.nanpercentile(use_data,0.5)
			else:
				bottom = data_min
			if data_max > data_mean + 5*data_stdev:
				top = numpy.nanpercentile(use_data,99.5)
			else:
				top = data_max
			use_data = use_data[ (use_data>bottom) & (use_data<top) ]
			if use_color == hexcolor('orange'):
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

	return bottom, top, use_color, bins


def spark_histogram_maker(data, bins=20, title=None, xlabel=None, ylabel=None, xticks=False, yticks=False, frame=False, notetaker=None, prerange=None, duo_filter=None, data_for_bins=None):

	data = numpy.asarray(data)
	if data_for_bins is None:
		use_data_for_bins = data
	else:
		use_data_for_bins = numpy.asarray(data_for_bins)
	if duo_filter is not None: duo_filter = numpy.asarray(duo_filter)

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
				use_duo_filter = use_duo_filter[(use_data>bottom) & (use_data<top)]
			use_data = use_data[ (use_data>bottom) & (use_data<top) ]
			use_data_for_bins = use_data_for_bins[ (use_data_for_bins>bottom) & (use_data_for_bins<top) ]
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
			mn, mx = use_data_for_bins.min() + 0.0, use_data_for_bins.max() + 0.0
		width = numpy.lib.function_base._hist_bin_selectors[bins](use_data_for_bins)
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

		plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
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

			plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
			ret[duo] = plot_as_svg_xhtml(fig)
			plt.clf()

	if notetaker is not None and use_color!=hexcolor('ocean'):
		notetaker.add( _spark_histogram_notes[use_color] )

	return ret





def spark_pie_maker(data, notetaker=None):
	plt.clf()
	fig = plt.gcf()
	fig.set_figheight(0.2)
	fig.set_figwidth(0.75)
	fig.set_dpi(300)
	C_sky = (35,192,241)
	C_night = (100,120,186)
	C_forest = (39,182,123)
	C_ocean = (29,139,204)
	C_lime = (128,189,1)
	# The slices will be ordered and plotted counter-clockwise.
	plt.pie(data, explode=None, labels=None, colors=[hexcolor('sky'),hexcolor('night'),hexcolor('forest'),hexcolor('ocean')],
			#autopct='%1.1f%%',
			shadow=False, startangle=90,
			wedgeprops={'linewidth':0, 'clip_on':False},
			frame=False)
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
	# Set aspect ratio to be equal so that pie is drawn as a circle.
	plt.axis('equal')
	ret = plot_as_svg_xhtml(fig)
	plt.clf()
	return ret



def spark_histogram(data, *arg, pie_chart_cutoff=4, notetaker=None, prerange=None, duo_filter=None, data_for_bins=None, **kwarg):
	if prerange is not None:
		return spark_histogram_maker(data, *arg, notetaker=notetaker, prerange=prerange, duo_filter=duo_filter, data_for_bins=data_for_bins, **kwarg)
	try:
		flat_data = data.flatten()
	except:
		flat_data = data
	uniq = numpy.unique(flat_data[:100])
	uniq_counts = None
	pie_chart_cutoff = int(pie_chart_cutoff)
	if len(uniq)<=pie_chart_cutoff:
		if duo_filter is not None:
			uniq0, uniq_counts0 = numpy.unique(flat_data[~duo_filter], return_counts=True)
			uniq1, uniq_counts1 = numpy.unique(flat_data[duo_filter], return_counts=True)
		else:
			uniq, uniq_counts = numpy.unique(flat_data, return_counts=True)
	if uniq_counts is not None and len(uniq_counts)<=pie_chart_cutoff:
		if notetaker is not None:
			notetaker.add( "Graphs are represented as pie charts if the data element has {} or fewer distinct values.".format(pie_chart_cutoff) )
		if duo_filter is not None:
			return spark_pie_maker(uniq_counts0), spark_pie_maker(uniq_counts1)
		else:
			return spark_pie_maker(uniq_counts)
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




def computed_factor_figure_with_derivative(m, y_funcs, y_labels=None,
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

	dy_funcs = [ComputedFactor(label="∂"+cf.label, func=(lambda x,m: (cf.func(x,m) - cf.func(x-epsilon,m))*(1/epsilon)), ) for cf in y_funcs]

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
				dy.append( yf.func(x,ref_to_m) )
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

		if short_header is None:
			return plot_as_svg_xhtml(plt, header=header, headerlevel=headerlevel)
		else:
			return plot_as_svg_xhtml(plt, header=header, headerlevel=headerlevel, anchor=short_header)

	m.add_to_report(maker)



def validation_distribution_figure(m, factorarray, range, bins, headerlevel, header, short_header=None, to_report=True, immediate=False):
	"""A figure showing the distribution of a factor across real and modeled observations.

	This is an experimental function, use at your own risk

	Parameters
	----------
	m : larch.Model (self)
	factorarray : ndarray [nCases, nAlts], or str
		The array of factors to validate on.  Or give the name of an idca variable.
	range : tuple
	bins : int or str
	headerlevel : int
	header : str
	short_header : str or None
	"""
	if isinstance(factorarray,str):
		factorarray = m.df.array_idca(factorarray).squeeze()
	
	from matplotlib import pyplot as plt
	if short_header is None:
		short_header = header

	def distance_validation_maker(mod):
		if mod.data.weight is None:
			pr = mod.work.probability[:, :mod.nAlts()]
			ch = mod.data.choice.squeeze()
		else:
			pr = mod.work.probability[:, :mod.nAlts()] * mod.data.weight
			ch = mod.data.choice.squeeze() * mod.data.weight
		plt.clf()
		h1 = plt.hist(factorarray.flatten(), weights=pr.flatten(), histtype="stepfilled", bins=bins, alpha=0.7, normed=True, range=range, label='Modeled')
		h2 = plt.hist(factorarray.flatten(), weights=ch.flatten(), histtype="stepfilled", bins=bins, alpha=0.7, normed=True, range=range, label='Observed')
		#plt.legend(handles=[h1[1],h2[-1]])
		plt.legend()
		return plot_as_svg_xhtml(plt, header=header, headerlevel=headerlevel, anchor=short_header)

	if to_report:
		m.add_to_report(distance_validation_maker)
	if immediate:
		distance_validation_maker(m)
		plt.show()



