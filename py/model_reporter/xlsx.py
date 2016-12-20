




class XlsxModelReporter():



	class XlsxManager:
		"""Manages xlsx reporting for a :class:`Model`.	"""

		def __init__(self, model, return_xhtml=False):
			self._model = model

		def _get_item(self, key):
			candidate = None
			if isinstance(key,str):
				try:
					art_obj = getattr(self._model, "art_{}".format(key.casefold()))
					candidate = lambda *arg,**kwarg: art_obj().to_xlsx(*arg,**kwarg)
				except AttributeError:
					pass
				try:
					art_obj = self._model._user_defined_art[key.casefold()]
					candidate = lambda *arg,**kwarg: art_obj().to_xlsx(*arg,**kwarg)
				except (AttributeError, KeyError):
					pass
				if candidate is None:
					import warnings
					warnings.warn("xlsx builder for '{}' not found".format(key.casefold()))
					# TODO build a warning art here instead of raising an exception.
					raise TypeError("invalid item")
			else:
				raise TypeError("invalid item")
			return candidate

		def __getitem__(self, key):
			return self._get_item(key, True)

		def __repr__(self):
			return '<XlsxManager>'

		def __str__(self):
			return repr(self)

		def __getattr__(self, key):
			if key in ('_model',):
				return self.__dict__[key]
			return self.__getitem__(key)
	
		def __dir__(self):
			candidates = set()
			for j in dir(self._model):
				if len(j)>4 and j[:4]=='art_':
					candidates.add(j[4:])
			try:
				self._model._user_defined_art
			except AttributeError:
				pass
			else:
				candidates.update(i.casefold() for i in self._user_defined_art.keys())
			return candidates
	
		def __setattr__(self, key, val):
			if key[0]=='_':
				super().__setattr__(key, val)
			else:
				raise NotImplementedError('no new xlsx sections allowed (yet)')
				#self._model.new_xhtml_section(val, key)

		def __setitem__(self, key, val):
			if key[0]=='_':
				super().__setitem__(key, val)
			else:
				raise NotImplementedError('no new xlsx sections allowed (yet)')
				#self._model.new_xhtml_section(val, key)
	
		def __call__(self, *args, filename=None, book=None, final=True, overwrite=False, **kwarg):
			if filename is None and book is None:
				raise TypeError('both filename and book cannot be None for xlsx')
			import xlsxwriter
			if book is None:
				from ..util.filemanager import next_stack
				if not overwrite:
					filename = next_stack(filename)
				book = xlsxwriter.workbook.Workbook(filename, {'strings_to_numbers': True, 'strings_to_formulas': False})

			for arg in self._model.iter_cats(args):
				if isinstance(arg, str):
					self._get_item(arg)(book)
				elif isinstance(arg, list):
					self( *(self._model._inflate_cats(arg)) )
			if final:
				book.close()
			else:
				return book






	@property
	def xlsx(self):
		"""A :class:`XlsxManager` interface for the model.
		
		This method creates an excel report on the model. Call it with
		any number of string arguments to include those named report sections.
		
		All other parameters must be passed as keywords.
		
		Other Parameters
		----------------
		filename : None or str
			If None (the default) a xlsxwriter Workbook must be provided as `book` (see below).
			Otherwise, this should name a file into which the xlsx
			report will be written.  If that file already exists it will by default not 
			be overwritten, instead a new filename will be spooled off the given name.
		book : xlsxwriter.workbook.Workbook
			Provide an already opened workbook object.
		final : bool
			If True (the default) the workbook will be written and closed at the end of this
			function call.  Otherwise, the function will return a still-open workbook object,
			which can still receive more input, but the caller will need to save/close it when 
			done.
		overwrite : bool
			If True, filename will be overwritten if it exists.  If a `book` argument is given instead
			of a `filename`, this is ignored.
			
		Returns
		-------
		xlsxwriter.workbook.Workbook or None
		"""
		return XlsxModelReporter.XlsxManager(self)
