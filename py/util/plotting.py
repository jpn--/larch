
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


def spark_histogram_maker(data, bins=20, title=None, xlabel=None, ylabel=None, xticks=False, yticks=False, frame=False, notetaker=None):

	data = numpy.asarray(data)

	# Check if data is mostly zeros
	n_zeros = (data==0).sum()
	if n_zeros > (data.size) * _threshold_for_dropping_zeros_in_histograms:
		use_data = data[data!=0]
		use_color = hexcolor('orange')
	else:
		use_data = data
		use_color = hexcolor('ocean')
	bgcolor = None

	use_data = use_data[~numpy.isnan(use_data)]

	if use_data.size > 0:
		data_stdev = use_data.std()
		data_mean = use_data.mean()
		data_min = use_data.min()
		data_max = use_data.max()
		if (data_min < data_mean - 5*data_stdev) or data_max > data_mean + 5*data_stdev:
			bottom = numpy.nanpercentile(use_data,0.5)
			top = numpy.nanpercentile(use_data,99.5)
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
		if width:
			bins = int(numpy.ceil((mx - mn) / width))
		else:
			bins = 1
		# The spark graphs get hard to read if the bin slices are too thin, so we will max out at 50 bins
		if bins > 50:
			bins = 50
		# The spark graphs look weird if the bin slices are too fat, so we will min at 10 bins
		if bins < 10:
			bins = 10

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



def spark_histogram(data, *arg, pie_chart_cutoff=4, notetaker=None, **kwarg):
	try:
		flat_data = data.flatten()
	except:
		flat_data = data
	uniq = numpy.unique(flat_data[:100])
	uniq_counts = None
	pie_chart_cutoff = int(pie_chart_cutoff)
	if len(uniq)<=pie_chart_cutoff:
		uniq, uniq_counts = numpy.unique(flat_data, return_counts=True)
	if uniq_counts is not None and len(uniq_counts)<=pie_chart_cutoff:
		if notetaker is not None:
			notetaker.add( "Graphs are represented as pie charts if the data element has {} or fewer distinct values.".format(pie_chart_cutoff) )
		return spark_pie_maker(uniq_counts)
	return spark_histogram_maker(data, *arg, notetaker=notetaker, **kwarg)







def computed_factor_figure(m, y_funcs, y_labels=None,
						   max_x=1, min_x=0, header=None,
                           xaxis_label=None, yaxis_label=None,
						   logscale_x=False, logscale_f=False, figsize=(6.5,3)):
	from matplotlib import pyplot as plt
	import numpy as np
	with plt.style.context(('/Users/jpn/Dropbox/Larch/py/util/larch.mplstyle')):
		x = np.linspace(min_x, max_x)
		y = []
		for yf in y_funcs:
			y.append( yj(x,m) )
#		y1 = x*P(var+'1', default_value=0).value(m)+x**2*P(var+'2', default_value=0).value(m)+x**3*P(var+'3', default_value=0).value(m)
#		if 'LowIncome_{}'.format(var) in m:
#			y2 = x*P('LowIncome_'+var).value(m)+x**2*P('LowIncome_'+var+'^2', default_value=0).value(m)+x**3*P('LowIncome_'+var+'^3', default_value=0).value(m)
#		else:
#			y2 = None
#		if 'HighIncome_{}'.format(var) in m:
#			y3 = x*P('HighIncome_'+var).value(m)+x**2*P('HighIncome_'+var+'^2', default_value=0).value(m)+x**3*P('HighIncome_'+var+'^3', default_value=0).value(m)
#		else:
#			y3 = None
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
		

#		if y2 is not None:
#			l2=plt.plot(x, y1+y2, linewidth=2, label='Low Income', color='r')
#			lgnd_hands += l2
#		if y3 is not None:
#			l3=plt.plot(x, y1+y3, linewidth=2, label='High Income', color='g')
#			lgnd_hands += l3
#		if y2 is None and y3 is None:
#			l1=plt.plot(x, y1, linewidth=2, label='All Travelers', color='b')
#			lgnd_hands += l1
#		else:
#			l1=plt.plot(x, y1, linewidth=2, label='Other Incomes', color='b')
#			lgnd_hands += l1
		box = ax.get_position()
		ax.set_position([box.x0, box.y0+0.1, box.width * 0.5, box.height-0.1])
		ax.minorticks_on()
		#ax.grid(b=True, which='major', color='0.2', linestyle='-', linewidth=0.1)

		hist_vals = None
		hist_wgts = None
		# col_n = [n for n,c in enumerate(m.needs()['UtilityCA'].get_variables()) if 'log(attr_' in c][0]
		# hist_wgts = np.exp(m.Data("UtilityCA")[:,:,col_n].flatten())
		# col_n = [n for n,c in enumerate(m.needs()['UtilityCA'].get_variables()) if c=='SOV_Distance_AM'][0]
		# hist_vals = m.Data("UtilityCA")[:,:,col_n].flatten()
		# ax2 = plt.twinx(ax)
		# if logscale_f:
		# 	ax2.set_yscale('log', nonposx='clip', nonposy='clip')
		# ax2.grid(b=False)
		# ax2.set_ylabel('Frequency')
		# ax2.set_position([box.x0, box.y0+0.1, box.width * 0.5, box.height-0.1])
		# ax2.set_xlim(0,max_x)
		# binspace = 50
		# n,bins,patches = plt.hist(hist_vals, bins=binspace, histtype='stepfilled', normed=False, weights=hist_wgts, color='yellow', facecolor='yellow', label='Frequency among Attractions', linewidth=0.1)
		# lgnd_hands += [patches[0],]
		# ax.set_zorder(ax2.get_zorder()+1)

		pushover = 1.05 if hist_vals is None or hist_wgts is None else 1.25
		ax.legend(loc='center left', bbox_to_anchor=(pushover,0.5), handles=lgnd_hands, fontsize=9)
		ax.patch.set_visible(False)
		ax.set_xlim(min_x,max_x)
	return plotting.plot_as_svg_xhtml(plt, header=header, headerlevel=2)

