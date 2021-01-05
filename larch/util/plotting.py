
from io import BytesIO
from xmle import Elem
import pandas
import numpy
from typing import Collection, Callable, Mapping

def adjust_spines(ax, spines):
	for loc, spine in ax.spines.items():
		if loc in spines:
			#spine.set_position(('outward', 10))  # outward by 10 points
			#spine.set_smart_bounds(True)
			pass
		else:
			spine.set_color('none')  # don't draw spine
	# turn off ticks where there is no spine
	if 'left' in spines:
		ax.yaxis.set_ticks_position('left')
	else:
		# no yaxis ticks
		ax.yaxis.set_ticks([])
	if 'bottom' in spines:
		ax.xaxis.set_ticks_position('bottom')
	else:
		# no xaxis ticks
		ax.xaxis.set_ticks([])



def plot_as_svg_xhtml(
		fig,
		classname='figure',
		headerlevel=2,
		header=None,
		anchor=1,
		transparent=True,
		tooltip=None,
		bbox_extra_artists=None,
		pad_inches=0.1,
		**format,
):
	existing_format_keys = list(format.keys())
	for key in existing_format_keys:
		if key.upper()!=key: format[key.upper()] = format[key]
	if 'GRAPHWIDTH' not in format and 'GRAPHHEIGHT' in format: format['GRAPHWIDTH'] = format['GRAPHHEIGHT']
	if 'GRAPHWIDTH' in format and 'GRAPHHEIGHT' not in format: format['GRAPHHEIGHT'] = format['GRAPHWIDTH']*.67
	import xml.etree.ElementTree as ET
	ET.register_namespace("","http://www.w3.org/2000/svg")
	ET.register_namespace("xlink","http://www.w3.org/1999/xlink")
	imgbuffer = BytesIO()
	fig.savefig(
		imgbuffer, dpi=None, facecolor="none", edgecolor='w',
		orientation='portrait', format='svg',
		transparent=transparent, bbox_inches="tight", pad_inches=pad_inches,
		bbox_extra_artists=bbox_extra_artists,
	)
	x = Elem("div", {'class':classname})
	if header:
		x.hn(headerlevel, header, anchor=anchor)
	x << ET.fromstring(imgbuffer.getvalue().decode())
	if tooltip is not None:
		x[0][1].insert(0, Elem("title", text=tooltip))
	return x



def line_graph(
		x,
		y,
		x_title=None,
		y_title=None,
		show_legend=None,
		**kwargs,
):
	"""
	Generate a line graph.

	Parameters
	----------
	x : array-like
	y : array-like or pandas.Series or pandas.DataFrame
	x_title : str
	y_title : str
	show_legend : bool, optional
	**kwargs
	"""
	useful_legend = False

	if x_title is None:
		x_title = getattr(x, 'name', None)

	from matplotlib import pyplot as plt
	fig, ax = plt.subplots()

	if isinstance(y, pandas.DataFrame):
		for col in y.columns:
			ax.plot(x, y[col], label=col)
			useful_legend = True
	elif isinstance(y, pandas.Series):
		plt.plot(x, y.values, label=y.name)
		if y.name is not None:
			useful_legend = True
	elif isinstance(y, Collection) and all(isinstance(k, Callable) for k in y):
		for k in y:
			this_y = k(x)
			label_y = getattr(this_y, 'name', None)
			ax.plot(x, this_y, label=label_y)
			useful_legend = useful_legend or (label_y is not None)
	elif isinstance(y, Mapping):
		for col, val in y.items():
			ax.plot(x, val, label=col)
		useful_legend = True
	else:
		ax.plot(x, y)

	if x_title is not None:
		ax.set_xlabel(x_title)

	if y_title is not None:
		ax.set_ylabel(y_title)

	if show_legend is None and useful_legend:
		ax.legend()
	elif show_legend:
		ax.legend()

	fig.tight_layout(pad=0.5)
	result = plot_as_svg_xhtml(fig, **kwargs)
	plt.close(fig)

	return result


