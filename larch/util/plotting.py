
from io import BytesIO
from xmle import Elem


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



def plot_as_svg_xhtml(pyplot, classname='figure', headerlevel=2, header=None, anchor=1, transparent=True, tooltip=None, **format):
	existing_format_keys = list(format.keys())
	for key in existing_format_keys:
		if key.upper()!=key: format[key.upper()] = format[key]
	if 'GRAPHWIDTH' not in format and 'GRAPHHEIGHT' in format: format['GRAPHWIDTH'] = format['GRAPHHEIGHT']
	if 'GRAPHWIDTH' in format and 'GRAPHHEIGHT' not in format: format['GRAPHHEIGHT'] = format['GRAPHWIDTH']*.67
	import xml.etree.ElementTree as ET
	ET.register_namespace("","http://www.w3.org/2000/svg")
	ET.register_namespace("xlink","http://www.w3.org/1999/xlink")
	imgbuffer = BytesIO()
	pyplot.savefig(imgbuffer, dpi=None, facecolor='w', edgecolor='w',
					orientation='portrait', papertype=None, format='svg',
					transparent=transparent, bbox_inches=None, pad_inches=0.1,
					frameon=None)
	x = Elem("div", {'class':classname})
	if header:
		x.hn(headerlevel, header, anchor=anchor)
	x << ET.fromstring(imgbuffer.getvalue().decode())
	if tooltip is not None:
		x[0][1].insert(0, Elem("title", text=tooltip))
	return x
