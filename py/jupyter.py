


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
