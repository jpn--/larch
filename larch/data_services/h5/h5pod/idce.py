
from .generic import *
import warnings

class H5PodCE(H5Pod):
	"""
	A HDF5 group node containing :ref:`idce` format data.

	The CE pod is conceptually similar to the CA pod, although alternatives are included
	element-wise, so that each array of data is stored as a vector, and casewise boundaries
	are permitted to be ragged.

	The H5PodCE has two special vectors, indicating the case-index and alt-index position of
	each item within the array, respectively.

	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._caseindex = None
		self._altindex = None

		# to translate ids to indexes...
	    # uniq_labels, uniq_indexes = numpy.unique(labels, return_inverse=True)

	@property
	def podtype(self):
		return 'idce'

	@property
	def n_cases(self):
		if self._caseindex is None:
			raise ValueError("no caseindexes set")
		return self._groupnode._v_children[self._caseindex][:].max()+1

	@property
	def n_alts(self):
		if self._altindex is None:
			raise ValueError("no altindexes set")
		return self._groupnode._v_children[self._altindex][:].max()+1

	@property
	def shape(self):
		"""The shape of the represented idca data (i.e., cases by alts)."""
		try:
			return (self.n_cases, self.n_alts, )
		except ValueError:
			raise NoKnownShape()

	@property
	def metashape(self):
		"""The shape of the underlying raw data (i.e., number of listed casealts)."""
		return super().shape


	def create_indexes_from_labels(self, indexes_name, labels_name=None, labels_data=None, make=None, **kwargs):
		"""

		Parameters
		----------
		indexes_name : str
			Name for the new array to be created.
		labels_name : str, optional
			Name of the existing array that gives the labels to be uniquely indexed.
		labels_data : ndarray, optional
			Array of labels.  Must have same shape as self.
		make : {'cases','alts',None}
			The newly created indexes will become this dimension for the outputs.
		kwargs
			passed to add_array

		Returns
		-------
		uniq_labels : ndarray
			An ordered array of labels corresponding to the indexes created.
		"""

		if labels_name is None and labels_data is None:
			raise ValueError("one of labels_name or labels_data must be given")

		if make not in ('cases','alts',None):
			raise ValueError("make must be in {'cases','alts',None}")

		if labels_data is None:
			labels_data = self._groupnode._v_children[labels_name][:]

		try:
			if labels_data.shape != self.metashape:
				raise IncompatibleShape(f'labels_data.shape of {labels_data.shape} does not match self.metashape {self.metashape}')
		except NoKnownShape:
			pass

		uniq_labels, uniq_indexes = numpy.unique(labels_data, return_inverse=True)

		self.add_array(indexes_name, uniq_indexes, **kwargs)

		if make == 'cases':
			self._caseindex = indexes_name
		elif make == 'alts':
			self._altindex = indexes_name

		return uniq_labels


	def set_casealt_indexes(self, case_index, alts_index):
		self._caseindex = case_index
		self._altindex = alts_index


	def merge_external_data(
			self, other,
			self_on, other_on=None, other_on_index=False,
			dupe_suffix="_copy", original_source=None, names=None, overwrite=False,
			other_dedupe=False, log=lambda *x: None, **kwargs
	):
		"""
		Merge data into this H5PodCE.

		Every casealt in the current (receiving) H5PodCO should match one or zero casealts in the
		imported data.

		Parameters
		----------
		other : H5PodCO or pandas.DataFrame or str
			The other data table to be merged.  Can be another DT, or a DataFrame, or the
			path to a file of type {csv, xlsx, dbf}.
		self_on : label or list, or array-like
			Field names to join on in this H5PodCE. Can be a vector or list of vectors
			of correct length to use a particular vector as the join key(s) instead of columns
		other_on : label or list, or array-like
			Field names to join on in other DataFrame, or vector/list of vectors per self_on
		dupe_suffix : str
			A suffix to add to variables that duplicate variables names already in this H5PodCE
		original_source : str, optional
			Give the original source of this data.  If not given and the filename can be
			inferred from other, that name will be used.
		names : iterable or dict, optional
			If given as a list, only these names will be merged from the other data.
			If given as a dict, the keys are the names that will be merged and the
			values are the new names in this DT.
		"""
		if other_on is None:
			if not other_on_index:
				other_on = self_on
			else:
				other_on = []

		if names is not None and not isinstance(names, dict):
			if isinstance(names, str):
				names = {names: names}
			else:
				names = {n: n for n in names}

		def _warn_nothing_done_():
			if names is not None:
				names_warn = '\n  from names:\n    ' + '\n    '.join(names.keys())
			else:
				names_warn = ''
			warnings.warn('nothing was imported into {}{}'.format(self.filename, names_warn))

		if isinstance(self_on, str):
			self_on = (self_on,)
		if isinstance(other_on, str):
			other_on = (other_on,)

		if self_on is None:
			baseframe = None
		elif isinstance(self_on, str):
			baseframe = self.dataframe[self_on]
		else:
			baseframe = self.dataframe[tuple(self_on)]
		anything_imported = False

		# get the other data (to be merged) as a pandas DataFrame
		if isinstance(other, H5PodCE):
			# we have a H5PodCO
			og = other._groupnode
			if other_on_index:
				raise TypeError('other_on_index cannot be True for other as H5PodCO')
			if original_source is None:
				original_source = other._h5f.filename
			if names is None:
				# we are potentially importing everything, so load everything into memory
				other_df = other.dataframe[other.names()]
			else:
				# we are importing only some things, so load only what will be needed
				use_names = set(names.keys()) | (set(other_on))
				for col in other.names():
					if col in other_on:
						continue
					if col not in names or (names[col] in self._groupnode and not overwrite):
						use_names.discard(col)
				if use_names == set(other_on):
					# there is nothing left to load but the matching keys,
					# so nothing will be loaded.  Abort now and save a bunch of work.
					_warn_nothing_done_()
					return
				other_df = other.dataframe[use_names]
		elif isinstance(other, pandas.DataFrame):
			# it is already a dataframe, so convenient!
			other_df = other
			og = None
		else:
			raise TypeError("bad type for other data")

		if other_dedupe:
			try:
				other_df.drop_duplicates(
					subset=None if other_on_index else other_on,
					keep='first',
					inplace=True
				)
			except TypeError as err:
				raise ValueError("Drop Duplicates Error") from err

		if baseframe is not None:
			new_df = pandas.merge(
				baseframe,
				other_df,
				left_on=self_on,
				right_on=None if other_on_index else other_on,
				right_index=other_on_index,
				how='left',
				suffixes=('', dupe_suffix),
			)
		else:
			new_df = other_df

		log(new_df)
		for col in new_df.columns:
			if names is None and col not in self._groupnode:
				log('importing "{}" into {}'.format(col, self.filename))
				try:
					self.add_array(col, arr=new_df[col].values)
				except UnicodeEncodeError:
					# The datatype has some problems converting to H5, warn and skip it.
					warnings.warn(f"unable to import {col}, the datatype is weird")
					continue
				anything_imported = True
				if original_source is not None:
					self._groupnode._v_children[col]._v_attrs.ORIGINAL_SOURCE = original_source
				if og is not None and col in og and 'DICTIONARY' in og._v_children[col]._v_attrs:
					self._groupnode._v_children[col]._v_attrs.DICTIONARY = og._v_children[col]._v_attrs.DICTIONARY
			elif names is not None and col in names and (names[col] not in self._groupnode or overwrite):
				log('importing "{}" as "{}" into {}'.format(col, names[col], self.filename))
				try:
					self.add_array(names[col], arr=new_df[col].values)
				except UnicodeEncodeError:
					# The datatype has some problems converting to H5, warn and skip it.
					warnings.warn(f"unable to import {col}, the datatype is weird")
					continue
				anything_imported = True
				if original_source is not None:
					self._groupnode._v_children[names[col]]._v_attrs.ORIGINAL_SOURCE = original_source
				if og is not None and col in og and 'DICTIONARY' in og._v_children[col]._v_attrs:
					self._groupnode._v_children[names[col]]._v_attrs.DICTIONARY = og._v_children[col]._v_attrs.DICTIONARY
		if not anything_imported:
			_warn_nothing_done_()


	@classmethod
	def from_dataframe(
			cls,
			df,
			h5filename=None,
			h5mode='a',
			h5groupnode='/',
			inmemory=False,
			temp=False,
			complevel=1,
			complib='zlib',
			ident=None,
			rename_columns=None,
			force_natural_names=True,
			caseindex=None,
			caselabels=None,
			altindex=None,
			altlabels=None,
	):
		self = cls(h5filename, h5mode, h5groupnode, inmemory=inmemory, temp=temp, complevel=complevel, complib=complib, ident=ident)
		if rename_columns:
			df.columns = [(rename_columns[c] if c in rename_columns else c) for c in df.columns]
		df.columns = [c.replace("/",'_') for c in df.columns]
		if force_natural_names:
			from ....util.naming import make_valid_identifier
			df.columns = [make_valid_identifier(c, True) for c in df.columns]
		self.merge_external_data(df, self_on=None, other_on=None)

		# if caseindex is None and caselabels is None:
		# 	raise ValueError("must give caseindex or caselabels")
		# if altindex is None and altlabels is None:
		# 	raise ValueError("must give caseindex or caselabels")

		if caseindex is not None and caseindex in self:
			self._caseindex = caseindex
			if caselabels is not None:
				import warnings
				warnings.warn("caselabels is ignored when caseindex is an existing column")
		else:
			if caseindex is None:
				import uuid
				caseindex = '_cIdx_'+str(uuid.uuid1()).replace('-','')
			if caselabels is not None:
				from ....warning import ignore_warnings
				with ignore_warnings():
					self.create_indexes_from_labels(caseindex, labels_name=caselabels)
				self._caseindex = caseindex

		if altindex is not None and altindex in self:
			self._altindex = altindex
			if altlabels is not None:
				import warnings
				warnings.warn("altlabels is ignored when altindex is an existing column")
		else:
			if altindex is None:
				import uuid
				altindex = '_aIdx_'+str(uuid.uuid1()).replace('-','')
			if altlabels is not None:
				from ....warning import ignore_warnings
				with ignore_warnings():
					self.create_indexes_from_labels(altindex, labels_name=altlabels)
				self._altindex = altindex

		return self

	@classmethod
	def from_csv(
			cls,
			*args,
			h5filename=None,
			h5mode='a',
			h5groupnode='/',
			inmemory=False,
			temp=False,
			complevel=1,
			complib='zlib',
			ident=None,
			rename_columns=None,
			force_natural_names=True,
			caseindex=None,
			caselabels=None,
			altindex=None,
			altlabels=None,
			**kwargs,
	):
		# self = cls(h5filename, h5mode, h5groupnode, inmemory=inmemory, temp=temp, complevel=complevel, complib=complib, ident=ident)
		df = pandas.read_csv(*args, **kwargs)
		return cls.from_dataframe(
			df,
			h5filename=h5filename,
			h5mode=h5mode,
			h5groupnode=h5groupnode,
			inmemory=inmemory,
			temp=temp,
			complevel=complevel,
			complib=complib,
			ident=ident,
			rename_columns=rename_columns,
			force_natural_names=force_natural_names,
			caseindex=caseindex,
			caselabels=caselabels,
			altindex=altindex,
			altlabels=altlabels,
		)

	@classmethod
	def from_excel(cls, *args, h5filename=None, h5mode='a', h5groupnode='/',
				   inmemory=False, temp=False, complevel=1, complib='zlib', ident=None,
				   rename_columns=None, force_natural_names=True,
				   **kwargs):
		self = cls(h5filename, h5mode, h5groupnode, inmemory=inmemory, temp=temp, complevel=complevel, complib=complib, ident=ident)
		if 'engine' not in kwargs:
			kwargs['engine'] = 'openpyxl'
		df = pandas.read_excel(*args, **kwargs)
		if rename_columns:
			df.columns = [(rename_columns[c] if c in rename_columns else c) for c in df.columns]
		df.columns = [c.replace("/",'_') for c in df.columns]
		if force_natural_names:
			from ....util.naming import make_valid_identifier
			df.columns = [make_valid_identifier(c, True) for c in df.columns]
		self.merge_external_data(df, self_on=None, other_on=None)
		return self


	def as_array_idca(self, *names, caseindex=None, altindex=None, dtype=numpy.float64, present=None, **kwargs):

		if caseindex is None:
			caseindex = self._caseindex
		if altindex is None:
			altindex = self._altindex

		cx = self._groupnode._v_children[caseindex][:]
		ax = self._groupnode._v_children[altindex][:]

		shape = (cx.max()+1, ax.max()+1, len(names))
		result = numpy.zeros(shape, dtype=dtype)

		for nx,n in enumerate(names):
			if present == n:
				result[cx,ax,nx] = 1
			else:
				result[cx,ax,nx] = self._groupnode._v_children[n][:]

		return result


	def load_data_item(self, name, result, selector=None):
		"""Load a slice of the pod arrays into an array in memory"""
		return super().load_meta_data_item(name, result, selector=selector)

	def _shape_for_get_data_item(self, selector=None):
		if selector is not None:
			import warnings
			warnings.warn('selector not compatible for idce get_data_item')
		return self.metashape

	def new_alternative_codes(
			self,
			groupby,
			caseindex=None,
			name=None,
			create_index=None,
			padding_levels=4,
			groupby_prefixes=None,
			overwrite=False,
	):
		"""

		Parameters
		----------
		groupby
		caseindex
		name
		create_index
		padding_levels : int, optional
			Space for this number of "extra" levels is reserved in each mask, beyond the number of
			detected categories.  This is critical if these alternative codes will be used for OGEV models
			that require extra nodes at levels that are cross-nested.
		overwrite : bool, default False
			Should existing variables with `name` be overwritten. It False and they already exist, a
			NodeError is raised.

		Returns
		-------
		pandas.DataFrame, SystematicAlternatives

		"""

		if caseindex is None:
			caseindex = self._caseindex

		if isinstance(groupby, str):
			groupby = (groupby,)

		if groupby_prefixes is None:
			groupby_prefixes = ["" for g in groupby]

		df = self.dataframe.__getitem__( (caseindex, *groupby) )

		s = df.groupby([caseindex, *groupby]).cumcount()+1

		masks = numpy.zeros(len(groupby)+1, dtype=numpy.int64)

		label_offset = int(numpy.ceil(numpy.log2(s.max())))

		uniqs = [numpy.unique(df.iloc[:,i+1], return_inverse=True) for i,g in enumerate(groupby)]

		if len(groupby) > 0:
			n_uniqs = [len(uniqs[i][0])+1+padding_levels for i,g in enumerate(groupby)]
		else:
			n_uniqs = []

		bitmask_sizes = [int(numpy.ceil(numpy.log2(g))) for g in n_uniqs]

		#x = numpy.unique(df[groupby[0]], return_inverse=True)[1]+1
		if len(bitmask_sizes):
			x = uniqs[0][1]+1
			masks[0] = 2**bitmask_sizes[0]-1
		else:
			x = numpy.zeros(len(df), dtype=int)

		for i in range(1, len(bitmask_sizes)):
			x <<= bitmask_sizes[i]
			masks <<= bitmask_sizes[i]
			#x += numpy.unique(df[groupby[i]], return_inverse=True)[1]+1
			x += uniqs[i][1]+1
			masks[i] = 2**bitmask_sizes[i]-1

		x <<= label_offset
		masks <<= label_offset
		x += s
		masks[-1] = 2**label_offset-1

		if name is not None:
			self.add_array(name, x.values, overwrite=overwrite)

		if create_index is not None:
			altcodes = self.create_indexes_from_labels(
				indexes_name=create_index,
				labels_data=x,
				make='alts',
				overwrite=overwrite,
			)
		else:
			altcodes = numpy.unique(x)

		from ...general import SystematicAlternatives
		sa = SystematicAlternatives(
			masks=masks,
			groupby=groupby,
			categoricals=[
				u[0] if groupby_prefixes[n] is None else [groupby_prefixes[n]+str(uu) for uu in u[0]]
				for n,u in enumerate(uniqs)
			],
			altcodes=altcodes,
			padding_levels=padding_levels,
		)

		return x, sa

