


class JupyterManager:
	"""Manages jupyter display for a :class:`Model`.	"""

	def __init__(self, model):
		self._model = model
		from IPython.display import display, HTML
		self._show_xml = lambda x, *arg, **kwarg: display(HTML(x(*arg, **kwarg).tostring().decode()))
		self._show_art = lambda x, *arg, **kwarg: display(x(*arg, **kwarg))

	def __getitem__(self, key):
		if isinstance(key,str):
			try:
				art_obj = getattr(self._model, "art_{}".format(key.casefold()))
				self._show_art(art_obj)
			except AttributeError:
				xml_obj = getattr(self._model, "xhtml_{}".format(key.casefold()))
				self._show_xml(xml_obj)
		else:
			raise TypeError("invalid jupyter item")

	def __repr__(self):
		return '<JupyterManager>'

	def __str__(self):
		return repr(self)

	def __getattr__(self, key):
		if key=='_model':
			return self.__dict__['_model']
		return self.__getitem__(key)

	def __call__(self, *arg):
		for key in arg:
			try:
				art_obj = getattr(self._model, "art_{}".format(key.casefold()))
				self._show_art(art_obj)
			except AttributeError:
				xml_obj = getattr(self._model, "xhtml_{}".format(key.casefold()))
				self._show_xml(xml_obj)

from . import styles



_default_css_jupyter = """

@import url(https://fonts.googleapis.com/css?family=Roboto:400,700,500italic,100italic|Roboto+Mono:300,400,700|Roboto+Slab:200,900);

.error_report {color:red; font-family:monospace;}

div.output_wrapper {""" + styles.body_font + """}

div.output_wrapper table {border-collapse:collapse;}

div.output_wrapper table, div.output_wrapper th, div.output_wrapper td {
	border: 1px solid #999999;
	font-family:"Roboto Mono", monospace;
	font-size:90%;
	font-weight:400;
	}
	
div.output_wrapper th, div.output_wrapper td { padding:2px; }

div.output_wrapper td.parameter_category {
	font-family:"Roboto", monospace;
	font-weight:500;
	background-color: #f4f4f4; 
	font-style: italic;
	}

div.output_wrapper th {
	font-family:"Roboto", monospace;
	font-weight:700;
	}
	
.larch_signature {""" + styles.signature_font + """ }
.larch_name_signature {""" + styles.signature_name_font + """}

.larch_head_tag {font-size:150%; font-weight:900; font-family:"Roboto Slab", Verdana;}
.larch_head_tag_ver {font-size:80%; font-weight:200; font-family:"Roboto Slab", Verdana;}

div.output_wrapper a.parameter_reference {font-style: italic; text-decoration: none}

div.output_wrapper .strut2 {min-width:1in}

div.output_wrapper .histogram_cell { padding-top:1; padding-bottom:1; vertical-align:center; }

div.output_wrapper .raw_log pre {
	font-family:"Roboto Mono", monospace;
	font-weight:300;
	font-size:70%;
	}

div.output_wrapper caption {
    caption-side: bottom;
	text-align: left;
	font-family: Roboto;
	font-style: italic;
	font-weight: 100;
	font-size: 80%;
}
"""


def stylesheet():
	from IPython.display import display, HTML
	display(HTML("<style>{}</style>".format(_default_css_jupyter)))


def larch_tag():
	from .util.xhtml import XML_Builder
	xsign = XML_Builder("div", {'class':'larch_head_tag'})
	from .built import longversion as version
	from .util.img import favicon
	xsign.start('p', {'style':'float:left;margin-top:6px'})
	xsign.start('img', {'width':"32", 'height':"32", 'src':"data:image/png;base64,{}".format(favicon), 'style':'float:left;position:relative;top:-3px;padding-right:0.2em;' })
	xsign.end('img')
	xsign.data(" Larch ".format())
	xsign.start('span', {'class':'larch_head_tag_ver'})
	xsign.data(version)
	xsign.end('span')
	xsign.end('p')
	xsign.start('img', { 'height':"48", 'src':"https://www.camsys.com/sites/default/files/camsys_logo.png", 'style':'float:right;height:40px;margin-top:0' })
	xsign.end('img')
	xsign.close()
	from IPython.display import display, HTML
	s= xsign.dumps()
	display(HTML(s))





def ipython_status(magic_matplotlib=True):
	message_set = set()
	try:
		cfg = get_ipython().config
	except:
		message_set.add('Not IPython')
	else:
		import IPython
#		try:
#			get_ipython().magic("matplotlib inline")
#		except IPython.core.error.UsageError:
#			message_set.add('IPython inline plotting not available')
#		else:
		message_set.add('IPython')

		# Caution: cfg is an IPython.config.loader.Config
		if cfg['IPKernelApp']:
			message_set.add('IPython QtConsole')

			try:
				if cfg['IPKernelApp']['pylab'] == 'inline':
					message_set.add('pylab inline')
				else:
					message_set.add('pylab loaded but not inline')
			except:
				message_set.add('pylab not loaded')
		elif cfg['TerminalIPythonApp']:
			try:
				if cfg['TerminalIPythonApp']['pylab'] == 'inline':
					message_set.add('pylab inline')
				else:
					message_set.add('pylab loaded but not inline')
			except:
				message_set.add('pylab not loaded')
	return message_set



if 'IPython' in ipython_status():
	try:
		stylesheet()
		larch_tag()
	except:
		pass


