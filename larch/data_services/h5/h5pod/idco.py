
from .generic import *
from ...general import _sqz_same, _sqz_same_trailing_neg_ok
import warnings

class H5PodCO(H5Pod):
	"""
	A HDF5 group node containing :ref:`idco` format data.

	The CO pod is the most basic H5Pod. It includes a selection of data columns, where
	each column is a vector of values attributable to casewise observations.  The ordering
	of the data is important, and all data columns in an H5PodCO must have the same length
	and be ordered the same way, so that any group of columns can be simply concantenated to
	create a 2-d :ref:`idco` array of data.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	@property
	def podtype(self):
		return 'idco'


	@classmethod
	def from_dataframe(cls, df, h5filename=None, h5mode='a', h5groupnode='/',
					   inmemory=False, temp=False, complevel=1, complib='zlib', ident=None):
		self = cls(h5filename, h5mode, h5groupnode, inmemory=inmemory, temp=temp, complevel=complevel, complib=complib, ident=ident)
		self.merge_external_data(df, self_on=None, other_on=None)
		return self

	@classmethod
	def from_csv(cls, *args, h5filename=None, h5mode='a', h5groupnode='/',
				 inmemory=False, temp=False, complevel=1, complib='zlib', ident=None,
				 rename_columns=None, force_natural_names=True,
				 **kwargs):
		self = cls(h5filename, h5mode, h5groupnode, inmemory=inmemory, temp=temp, complevel=complevel, complib=complib, ident=ident)
		df = pandas.read_csv(*args, **kwargs)
		if rename_columns:
			df.columns = [(rename_columns[c] if c in rename_columns else c) for c in df.columns]
		df.columns = [c.replace("/",'_') for c in df.columns]
		if force_natural_names:
			from ....util.naming import make_valid_identifier
			df.columns = [make_valid_identifier(c, True) for c in df.columns]
		self.merge_external_data(df, self_on=None, other_on=None)
		return self

	@classmethod
	def from_excel(cls, *args, h5filename=None, h5mode='a', h5groupnode='/',
				   inmemory=False, temp=False, complevel=1, complib='zlib', ident=None,
				   rename_columns=None, force_natural_names=True,
				   **kwargs):
		self = cls(h5filename, h5mode, h5groupnode, inmemory=inmemory, temp=temp, complevel=complevel, complib=complib, ident=ident)
		df = pandas.read_excel(*args, **kwargs)
		if rename_columns:
			df.columns = [(rename_columns[c] if c in rename_columns else c) for c in df.columns]
		df.columns = [c.replace("/",'_') for c in df.columns]
		if force_natural_names:
			from ....util.naming import make_valid_identifier
			df.columns = [make_valid_identifier(c, True) for c in df.columns]
		self.merge_external_data(df, self_on=None, other_on=None)
		return self


	def copy_external_data(self, other, names=None, original_source=None,
						log=lambda *x: None, **kwargs):
		"""
		Copy data into this H5PodCO.

		Every case in the current (receiving) H5PodCO should match exactly one case in the
		imported data, in the same order, so that no merge is undertaken; the data is simplied copied.

		Parameters
		----------
		other : H5PodCO
			The other data table to be merged.  Can be another H5PodCO, or a DataFrame, or the
			path to a file of type {csv, xlsx, dbf}.
		names : iterable or dict, optional
			If given as a list, only these names will be merged from the other data.
			If given as a dict, the keys are the names that will be merged and the
			values are the new names in this DT.
		original_source : str, optional
			Give the original source of this data.  If not given and the filename can be
			inferred from other, that name will be used.
		"""
		if names is not None and not isinstance(names, dict):
			if isinstance(names, str):
				names = {names: names}
			else:
				names = {n: n for n in names}
		if not isinstance(other, H5PodCO):
			raise TypeError("currently can merge copy H5PodCO")
		if original_source is None:
			original_source = other._h5f.filename
		anything_imported = False
		for col in other.names():
			if names is None and col not in self._groupnode:
				log('importing "{}" into {}'.format(col, self.filename))
				self.add_array(col, arr=other._groupnode._v_children[col][:])
				anything_imported = True
				if original_source is not None:
					self._groupnode._v_children[col]._v_attrs.ORIGINAL_SOURCE = original_source
				if col in other._groupnode and 'DICTIONARY' in other._groupnode._v_children[col]._v_attrs:
					self._groupnode._v_children[col]._v_attrs.DICTIONARY = other._groupnode._v_children[col]._v_attrs.DICTIONARY
			elif names is not None and col in names and names[col] not in self._groupnode:
				log('importing "{}" as "{}" into {}'.format(col, names[col], self.filename))
				self.add_array(names[col], arr=other._groupnode._v_children[col][:])
				anything_imported = True
				if original_source is not None:
					self._groupnode._v_children[names[col]]._v_attrs.ORIGINAL_SOURCE = original_source
				if col in other._groupnode and 'DICTIONARY' in other._groupnode._v_children[col]._v_attrs:
					self._groupnode._v_children[names[col]]._v_attrs.DICTIONARY = other._groupnode._v_children[col]._v_attrs.DICTIONARY
		if not anything_imported:
			if names is not None:
				names_warn = '\n  from names:\n    ' + '\n    '.join(names.keys())
			else:
				names_warn = ''
			warnings.warn('nothing was imported into {}{}'.format(self.filename, names_warn))


	def merge_external_data(
			self, other,
			self_on, other_on=None, other_on_index=False,
			dupe_suffix="_copy", original_source=None, names=None, overwrite=False,
			other_dedupe=False, log=lambda *x: None, **kwargs
	):
		"""
		Merge data into this H5PodCO.

		Every case in the current (receiving) H5PodCO should match one or zero cases in the
		imported data.

		Parameters
		----------
		other : H5PodCO or pandas.DataFrame or str
			The other data table to be merged.  Can be another DT, or a DataFrame, or the
			path to a file of type {csv, xlsx, dbf}.
		self_on : label or list, or array-like
			Field names to join on in this H5PodCO. Can be a vector or list of vectors
			of correct length to use a particular vector as the join key instead of columns
		other_on : label or list, or array-like
			Field names to join on in other DataFrame, or vector/list of vectors per self_on
		dupe_suffix : str
			A suffix to add to variables that duplicate variables names already in this H5PodCO
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

		# if isinstance(other, pandas.DataFrame):
		# 	return self.merge_into_idco_from_dataframe(other, self_on, other_on, dupe_suffix=dupe_suffix,
		# 											   original_source=original_source, names=names)
		# if isinstance(other, str) and os.path.exists(other):
		# 	return self.merge_into_idco_from_csv(other, self_on, other_on, dupe_suffix=dupe_suffix,
		# 										 original_source=original_source, names=names, **kwargs)

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
		if isinstance(other, H5PodCO):
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

	def pluck_from_omx(self, other_omx, rowindexes, colindexes, names=None, overwrite=False):
		"""
		Pluck values from an OMX file into new :ref:`idco` variables.

		This method takes O and D index numbers and plucks the individual matrix values at
		those coordinates. New idco variables will be created in the DT file that contain
		the plucked values, so that the new variables represent actual arrays and not links to
		the original matrix.  The OMX filename is marked as the original source of the data.

		Parameters
		----------
		other_omx : OMX or str
			Either an OMX or a filename to an OMX file.
		rowindexes, colindexes : array
			Zero-based index array for the row (origins) and columns (destinations) that will
			be plucked into the new variable.
		names : str or list or dict
			If a str, only that single named matrix table in the OMX will be plucked.
			If a list, all of the named matrix tables in the OMX will be plucked.
			If a dict, the keys give the matrix tables to pluck from and the values
			give the new variable names to create in this DT.


		See Also
		--------
		GroupNode.add_external_omx

		"""
		from ....omx import OMX
		if isinstance(other_omx, str):
			other_omx = OMX(other_omx)
		if names is None:
			names = {n: n for n in other_omx.data._v_children}

		if names is not None and not isinstance(names, dict):
			if isinstance(names, str):
				names = {names: names}
			else:
				names = {n: n for n in names}

		if isinstance(rowindexes, str):
			rowarr = self._groupnode._v_children[rowindexes][:]
		elif isinstance(rowindexes, numpy.ndarray):
			rowarr = rowindexes
		elif isinstance(rowindexes, int) and isinstance(colindexes, (str, numpy.ndarray)):
			rowarr = None  # fix in a moment
		else:
			raise TypeError('rowindexes must be str, int, or ndarray')

		if isinstance(colindexes, str):
			colarr = self._groupnode._v_children[colindexes][:]
		elif isinstance(colindexes, numpy.ndarray):
			colarr = colindexes
		elif isinstance(colindexes, int) and rowarr is not None:
			colarr = numpy.full_like(rowarr, colindexes, dtype=int)
		else:
			raise TypeError('colindexes must be str, int, or ndarray')

		if rowarr is None:
			#rowarr = numpy.full_like(clarr, rowindexes, dtype=int)
			raise NotImplementedError

		for matrix_page in names:
			self.add_array(names[matrix_page],
									 other_omx[matrix_page][rowarr, colarr],
									 original_source=other_omx.filename,
									 overwrite=overwrite)

	def as_idca(self):
		ret = H5PodCOasCA(self, ident=self.ident+"_as_idca")
		return ret

	def fatten(self, fat_filename=None, fat_groupnode=None, fat_dim=None, skips=(), keeps=(), overwrite=False):
		"""Generate a new H5PodCA with data broadcast across a new dimension."""

		if fat_filename is None:
			fat_filename = "{}_fat{}".format(*os.path.splitext(self.filename))

		if fat_filename == self.filename:
			raise ValueError("cannot overwrite same file (maybe one day)")

		if fat_groupnode is None:
			fat_groupnode = self._groupnode._v_pathname

		if fat_dim is None:
			raise TypeError('must give fat_dim')

		from .idca import H5PodCA
		import re
		result = H5PodCA(fat_filename, 'a' if not overwrite else 'w', fat_groupnode)
		for n in self.names():
			keep_n = True
			for s in skips:
				if re.match(s,n):
					keep_n = False
					break
			if len(keeps):
				k_n = [re.match(k, n) for k in keeps]
				if not numpy.any(k_n):
					keep_n = False
			if not keep_n:
				continue
			arr = self.__getattr__(n)[:]
			fat = numpy.empty(arr.shape + (fat_dim,), dtype=arr.dtype)
			fat[..., :] = arr[..., None]
			try:
				di = self.__getattr__(n)._v_attrs.DICTIONARY
			except AttributeError:
				di = None
			try:
				ti = self.__getattr__(n)._v_attrs.TITLE
			except AttributeError:
				ti = None
			result.add_array(n, fat, title=ti, dictionary=di)
		return result

	def reshape_to(self, shape, filename=None, groupnode=None, skips=(), keeps=(), overwrite=False, **kwargs):
		"""Generate a new H5PodCA with data reshaped to fill a new dimension."""

		assert isinstance(shape, (list,tuple))

		if filename is None:
			filename = "{}_ca{}".format(*os.path.splitext(self.filename))

		if filename == self.filename:
			raise ValueError("cannot overwrite same file (maybe one day)")

		if groupnode is None:
			groupnode = self._groupnode._v_pathname

		from .idca import H5PodCA
		import re
		result = H5PodCA(filename, 'a' if not overwrite else 'w', groupnode, **kwargs)
		for n in self.names():
			keep_n = True
			for s in skips:
				if re.match(s,n):
					keep_n = False
					break
			if len(keeps):
				k_n = [re.match(k, n) for k in keeps]
				if not numpy.any(k_n):
					keep_n = False
			if not keep_n:
				continue
			arr = self.__getattr__(n)[:].reshape(shape)
			try:
				di = self.__getattr__(n)._v_attrs.DICTIONARY
			except AttributeError:
				di = None
			try:
				ti = self.__getattr__(n)._v_attrs.TITLE
			except AttributeError:
				ti = None
			result.add_array(n, arr, title=ti, dictionary=di)
		return result

class H5PodCOasCA(H5PodCO):

	@property
	def podtype(self):
		return 'idca'

	def load_data_item(self, name, result, selector=None):
		"""Load a slice of the pod arrays into an array in memory"""
		# convert a single name string to a one item list
		_sqz_same_trailing_neg_ok(result.shape, [selector_len_for(selector, self.shape[0]), *self.shape[1:]])
		result[:,:] = self._evaluate_single_item(name, selector)[:,None]
		return result

	@property
	def shape(self):
		"""The shape of the pod.

		"""
		return super().shape + (self.trailing_dim,)
		# if 'SHAPE' in self._groupnode._v_attrs:
		# 	return tuple(self._groupnode._v_attrs['SHAPE'][:]) + (self.trailing_dim,)
		# if len(self.names()):
		# 	for v in self._groupnode._v_children.values():
		# 		try:
		# 			found_shape = v.shape
		# 		except:
		# 			pass
		# 		else:
		# 			try:
		# 				self.shape = found_shape
		# 			except:
		# 				pass
		# 			return tuple(found_shape) + (self.trailing_dim,)
		# raise NoKnownShape()

	@property
	def trailing_dim(self):
		try:
			return self._trailing_dim
		except AttributeError:
			return -1

	@trailing_dim.setter
	def trailing_dim(self, value):
		self._trailing_dim = int(value)

