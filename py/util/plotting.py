
from io import BytesIO
from ..util.xhtml import XHTML, XML_Builder
import os

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

