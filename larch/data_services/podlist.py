import numpy
from collections.abc import MutableSequence
from .pod import Pod
from .general import selector_len_for
from ..util import Dict

import logging
from ..log import logger_name
logger = logging.getLogger(logger_name+".data")

class EmptyPodsError(ValueError):
	pass

class Pods(MutableSequence):

	def __init__(self, initial=()):
		super().__init__()
		self._datalist = []
		for i in initial:
			if isinstance(i, Pod):
				self.__check_shape(i, i.ident)
				self._datalist.append(i)
			else:
				raise TypeError('Pods can only contain Pod members')

	def __delitem__(self, item):
		del self._datalist[item]

	def __getitem__(self, item):
		try:
			return self._datalist[item]
		except:
			raise

	def __len__(self):
		return len(self._datalist)

	def __setitem__(self, item, value):
		if not isinstance(value, Pod):
			raise TypeError('Pods can only contain Pod members')
		self.__check_shape(value, value.ident)
		self._datalist[item] = value

	def insert(self, position, item):
		if not isinstance(item, Pod):
			raise TypeError('Pods can only contain Pod members')
		self.__check_shape(item, item.ident)
		self._datalist.insert(position, item)

	@property
	def shape(self):
		if len(self._datalist):
			for i in range(len(self._datalist)):
				if self._datalist[i].shape[-1] != -1:
					return self._datalist[i].shape
			return self._datalist[0].shape
		raise EmptyPodsError('no pods')

	@property
	def metashape(self):
		"""The metashape of the first pod in the list that has a metashape."""
		if len(self._datalist):
			for i in range(len(self._datalist)):
				try:
					return self._datalist[i].metashape
				except:
					pass
			raise ValueError('no pods with metashape')
		raise EmptyPodsError('no pods')

	def __check_shape(self, prospect, ident=None):
		if len(self._datalist)==0:
			return
		if self.shape != prospect.shape:
			# permit if only last dim fails and last dim is -1 for prospect or local
			if self.shape[:-1] == prospect.shape[:-1]:
				if self.shape[-1] == -1 or prospect.shape[-1] == -1:
					return
			if ident:
				raise ValueError(f'trying to add {prospect.shape} ({ident}) to {self._datalist[0].shape}')
			raise ValueError(f'trying to add {prospect.shape} to {self._datalist[0].shape}')

	def names(self):
		nameset = set()
		for dat in self._datalist:
			nameset |= dat.nameset()
		return nameset

	nameset = names

	def _load_natural_data_item(self, name, result, selector=None):
		"""


		Parameters
		----------
		name : str
			The identifier for the data that will be loaded.
		result : ndarray
			This is the array (typically a slice of an array)
			into which the data will be loaded. It must be given and it
			must be the correct shape.
		selector : slice or array-like, optional
			This will slice the first dimension (cases) of the result.
			There may be a performace hit for using slices.

		Raises
		------
		KeyError
			This DataList cannot resolve the given name as a natural name.
		"""
		for dat in self._datalist:
			if name in dat:
				return dat.load_data_item(name, result, selector=selector)
		raise KeyError(f"{name} not found")

	def _get_natural_data_ref(self, name):
		"""

		Parameters
		----------
		name : str
			The identifier for the data that will be accessed.

		Raises
		------
		KeyError
			This DataList cannot resolve the given name as a natural name.
		"""
		for dat in self._datalist:
			if name in dat:
				return dat.__getattr__(name)
		raise KeyError(f"{name} not found")

	def _get_natural_data_item(self, name, selector=None):
		"""

		Parameters
		----------
		name : str
			The identifier for the data that will be loaded.
		selector : slice or array-like, optional
			This will slice the first dimension (cases) of the result.
			There may be a performace hit for using slices.

		Raises
		------
		KeyError
			This DataList cannot resolve the given name as a natural name.
		"""
		for dat in self._datalist:
			if name in dat:
				return dat.get_data_item(name, selector=selector)
		raise KeyError(f"{name} not found")

	def _get_natural_data_dictionary(self, name):
		"""

		Parameters
		----------
		name : str
			The identifier for the data dictionary that will be loaded.

		Raises
		------
		KeyError
			This DataList cannot resolve the given name as a natural name.
		"""
		for dat in self._datalist:
			if name in dat:
				return dat.__getattr__(name)._v_attrs.DICTIONARY
		raise KeyError(f"{name} not found")

	def _get_natural_data_description(self, name):
		"""

		Parameters
		----------
		name : str
			The identifier for the data description that will be loaded.

		Raises
		------
		KeyError
			This DataList cannot resolve the given name as a natural name.
		"""
		for dat in self._datalist:
			if name in dat:
				return dat.__getattr__(name)._v_attrs.TITLE
		raise KeyError(f"{name} not found")

	def dtype_of(self, name):
		"""

		Parameters
		----------
		name : str
			The identifier for the data dictionary that will be loaded.

		Raises
		------
		KeyError
			This DataList cannot resolve the given name as a natural name.
		"""
		for dat in self._datalist:
			if name in dat:
				return dat.__getattr__(name).dtype
		raise KeyError(f"{name} not found")

	def get_data_mask(self, name):
		from io import BytesIO
		for dat in self._datalist:
			if name in dat:
				return dat.durable_mask
		# No natural name match, so build composite mask
		from tokenize import tokenize, untokenize, NAME, OP, STRING, NUMBER
		mask = 0xFFFFFFFF
		try:
			name_encode = name.encode('utf-8')
		except AttributeError:
			name_encode = str(name).encode('utf-8')
		g = tokenize(BytesIO(name_encode).readline)
		for toknum, tokval, _, _, _ in g:
			if toknum == NAME:
				for i,dat in enumerate(self._datalist):
					if tokval in dat:
						mask &= dat.durable_mask
						break
				else:
					# no dat contains this natural name
					return 0
		return mask


	def load_data_item(self, name, result, selector=None):
		try:
			self._load_natural_data_item(name, result, selector)
		except KeyError:
			from tokenize import tokenize, untokenize, NAME, OP, STRING, NUMBER
			from ..util.aster import asterize
			DOT = (OP, '.')
			COLON = (OP, ':')
			COMMA = (OP, ',')
			OBRAC = (OP, '[')
			CBRAC = (OP, ']')
			OPAR = (OP, '(')
			CPAR = (OP, ')')
			EQUAL = (OP, '=')
			from io import BytesIO
			recommand = [(NAME, 'result'), OBRAC, COLON, CBRAC, EQUAL]
			try:
				name_encode = name.encode('utf-8')
			except AttributeError:
				name_encode = str(name).encode('utf-8')
			# dims = len(self.shape)
			g = tokenize(BytesIO(name_encode).readline)
			if selector is None:
				screen_tokens = [(NAME, 'None'), ]
			else:
				screen_tokens = [(NAME, 'selector'), ]
				# try:
				# 	slicer_encode = selector.encode('utf-8')
				# except AttributeError:
				# 	slicer_encode = str(selector).encode('utf-8')
				# screen_tokens = [(toknum, tokval) for toknum, tokval, _, _, _ in
				# 				 tokenize(BytesIO(slicer_encode).readline)]
			for toknum, tokval, _, _, _ in g:
				if toknum == NAME:
					for i,dat in enumerate(self._datalist):
						if tokval in dat:
							# replace NAME tokens
							partial = [
								(NAME, 'self'),
								OBRAC, (NUMBER, str(i)), CBRAC,
								DOT, (NAME, 'get_data_item'),
								OPAR, (STRING, f"'{tokval}'"), COMMA, *screen_tokens, CPAR,
							]
							recommand.extend(partial)
							break
					else:
						# no dat contains this natural name
						# put the name back in raw, maybe it works cause it's a global, more likely
						# the exception manager below will catch it.
						recommand.append((toknum, tokval))
				else:
					recommand.append((toknum, tokval))
			try:
				ret = untokenize(recommand).decode('utf-8')
			except:
				print("<recommand>")
				print(recommand)
				print("</recommand>")
				raise
			# print("<ret>")
			# print(ret)
			# print("</ret>")
			j = asterize(ret, mode="exec")
			from ..util.aster import inXd
			from numpy import log, exp, log1p, absolute, fabs, sqrt, isnan, isfinite, logaddexp, fmin, fmax, nan_to_num, sin, cos, pi
			from ..util.common_functions import piece, normalize
			try:
				exec(j)
			except Exception as exc:
				args = exc.args
				if not args:
					arg0 = ''
				else:
					arg0 = args[0]
				arg0 = arg0 + '\nwithin parsed command: "{!s}"'.format(name)
				if "max" in name:
					arg0 = arg0 + '\n(note to get the maximum of arrays use "fmax" not "max")'.format(name)
				if "min" in name:
					arg0 = arg0 + '\n(note to get the minimum of arrays use "fmin" not "min")'.format(name)
				if isinstance(exc, NameError):
					badname = str(exc).split("'")[1]
					goodnames = {
						'log', 'exp', 'log1p', 'absolute', 'fabs', 'sqrt', 'isnan',
						'isfinite', 'logaddexp', 'fmin', 'fmax', 'nan_to_num', 'piece', 'normalize',
					}
					for dat in self._datalist:
						goodnames |= dat.nameset()
					from ..util.text_manip import case_insensitive_close_matches
					did_you_mean_list = case_insensitive_close_matches(badname, goodnames, n=3, cutoff=0.1, excpt=None)
					if len(did_you_mean_list) > 0:
						arg0 = arg0 + '\n' + "did you mean {}?".format(
							" or ".join("'{}'".format(s) for s in did_you_mean_list))
				exc.args = (arg0,) + args[1:]
				raise

	def get_data_items(self, names, *arg, selector=None, dtype=None):
		"""

		Parameters
		----------
		names : list of str
			The identifier for the data that will be loaded.
		selector : slice or array-like, optional
			This will slice the first dimension (cases) of the result.
			There may be a performace hit for using slices.
		dtype : dtype, optional
			The dtype for the array to return. If the dtype is not given,
			float64 will be used.

		Returns
		-------
		result : ndarray
			This is the array (typically a slice of an array)
			into which the data will be loaded.

		"""
		logger.debug(f'Loading data from HDF5 into a new array...')
		if isinstance(names, str):
			names = (names, )
		names = names + arg
		if dtype is None:
			if len(names) != 1:
				dtype = numpy.float64
			else:
				try:
					dtype = self.dtype_of(names[0])
				except Exception as err:
					dtype = numpy.float64
		result_shape = self.shape_result(selector, names)
		try:
			result = numpy.zeros(result_shape, dtype=dtype)
		except ValueError as err:
			err.args = (err.args[0]+ f', result_shape={result_shape}',) + err.args[1:]
			raise
		for i,name in enumerate(names):
			logger.info(f' - loading {name} ...')
			self.load_data_item(name, result[...,i], selector=selector)
		logger.debug(f'Completed loading data from HDF5.')
		return result

	def get_data_masks(self, names):
		return [self.get_data_mask(i) for i in names]

	def shape_result(self, selector=None, names=None, use_metashape=False):
		if use_metashape:
			if names is None:
				return self.metashape
			else:
				return ( *(self.metashape), len(names), )
		else:
			if names is None:
				return (
					selector_len_for(selector, self.shape[0]),
					*(self.shape[1:]),
				)
			else:
				return (
					selector_len_for(selector, self.shape[0]),
					*(self.shape[1:]),
					len(names),
				)

	def load_data_items(self, names, *arg, result=None, selector=None, dtype=numpy.float64, mask_pattern=0, mask_names=None, log=None, use_metashape=False):
		"""

		Parameters
		----------
		names : list of str
			The identifier for the data that will be loaded.
		selector : slice or array-like, optional
			This will slice the first dimension (cases) of the result.
			There may be a performace hit for using slices.
		dtype : dtype, optional
			The dtype for the array to return. If the dtype is not given,
			float64 will be used.
		mask_pattern : int
		mask_names : array-like, optional
			An array of int32 of the same shape as `names`.  For each item, if the
			mask_names value bitwise-and the mask_pattern evaluates
			as True, then the data will be skipped. Note the default mask_pattern is 0 so
			even if mask_names is given, everything will be loaded unless the the mask_pattern
			is set to some other value. This is an optimization tool
			for reloading data that may not have changed (e.g. for logsum generation).

		Returns
		-------
		result : ndarray
			This is the array (typically a slice of an array)
			into which the data will be loaded.

		"""
		logger.debug(f'Loading data from HDF5 ...')
		if isinstance(names, str):
			names = (names, )
		names = names + arg
		if result is None:
			logger.warning('Initializing a new array: {}'.format(
				self.shape_result(selector, names, use_metashape=use_metashape)
			))
			result = numpy.zeros(self.shape_result(selector, names, use_metashape=use_metashape), dtype=dtype)
		else:
			from .general import _sqz_same, NotSameShapeError
			output_shape = self.shape_result(selector, names, use_metashape=use_metashape)
			try:
				_sqz_same(result.shape, output_shape)
			except NotSameShapeError:
				if (result.shape[0] != output_shape[0]) or (result.shape[1] < output_shape[1]):
					raise
		if mask_names is None:
			for i,name in enumerate(names):
				logger.info(f' - loading {name}')
				self.load_data_item(name, result[...,i], selector=selector)
		else:
			for i,name in enumerate(names):
				try:
					if not (mask_names[i] & mask_pattern):
						logger.info(f' - loading {name}')
						self.load_data_item(name, result[...,i], selector=selector)
					else:
						logger.info(f' - not loading {name} (masked)')
				except:
					raise
		logger.debug(f'Completed loading data from HDF5.')
		return result

	def load_casealt_indexes(self, caseindexes=None, altindexes=None, selector=None):
		if selector is not None:
			raise NotImplementedError()
		if caseindexes is None:
			#print("!! inits",self.shape_result(selector, ['_caseindexes_']))
			caseindexes = numpy.zeros(self.metashape, dtype=numpy.int64).reshape(-1)
		if altindexes is None:
			#print("!! inits",self.shape_result(selector, ['_altindexes_']))
			altindexes = numpy.zeros(self.metashape, dtype=numpy.int64).reshape(-1)
		from .h5 import H5PodCE
		for dat in self._datalist:
			if isinstance(dat, H5PodCE):
				dat.load_meta_data_item(dat._caseindex, caseindexes, selector=selector)
				dat.load_meta_data_item(dat._altindex , altindexes , selector=selector)
		return caseindexes, altindexes

	def __repr__(self):
		s = "<larch.data_services.Pods>"
		for dat in self._datalist:
			s += "\n | "
			s += repr(dat).replace("\n","\n | ")
		return s

	def _pull_arrays(self, var1, var2=None, *, selector=None):
		b = None
		if isinstance(selector, str):
			selector_ = self.get_data_items(selector, dtype=bool)
			a = self.get_data_items(var1)[selector_]
			if var2 is not None:
				b = self.get_data_items(var2)[selector_]
		else:
			a = self.get_data_items(var1, selector=selector)
			if var2 is not None:
				b = self.get_data_items(var2, selector=selector)
		return a, b

	def statistics_for(self, var, histogram=True, selector=None, ch_var=None, flatten=False, **kwargs):
		# ch = None
		# if isinstance(selector, str):
		# 	selector_ = self.get_data_items(selector, dtype=bool)
		# 	a = self.get_data_items(var)[selector_]
		# 	if ch_var is not None:
		# 		ch = self.get_data_items(ch_var)[selector_]
		# else:
		# 	a = self.get_data_items(var, selector=selector)
		# 	if ch_var is not None:
		# 		ch = self.get_data_items(ch_var, selector=selector)
		a, ch = self._pull_arrays(var, ch_var, selector=selector)

		if flatten:
			a = a.reshape(-1)
			if ch is not None:
				ch = ch.reshape(-1)

		from ..util.statistics import statistics_for_array
		result = statistics_for_array(a, histogram=histogram, varname=var, ch_weights=ch, **kwargs)
		try:
			descrip = self._get_natural_data_description(var)
		except:
			pass
		else:
			if descrip is not None and descrip!="":
				result.description=descrip
		try:
			dictionary = self._get_natural_data_dictionary(var)
		except:
			pass
		else:
			result.dictionary=Dict(dictionary)
		return result

	def get_row(self, rownum, lookup=True):
		result = Dict()
		for i in self.names():
			result[i] = self._groupnode._v_children[i][rownum]
			if lookup:
				try:
					d = self._groupnode._v_children[i]._v_attrs.DICTIONARY
				except (KeyError, AttributeError):
					pass
				else:
					if result[i] in d:
						result[i] = f"{result[i]} ({d[result[i]]})"
		return result

	def crosstab(self, rowvar, colvar, selector=None):
		a, b = self._pull_arrays(rowvar, colvar, selector=selector)
		import pandas
		return pandas.crosstab(
			a.squeeze(),
			b.squeeze(),
			rownames=[rowvar],
			colnames=[colvar],
		)


class PodsCA(Pods):

	def __init__(self, *args, n_alts=-1, **kwargs):
		self.n_alts = n_alts
		super().__init__(*args, **kwargs)

	@property
	def shape(self):
		s = super().shape
		if len(s)>1 and s[1] == -1:
			s = (s[0], self.n_alts, *s[2:])
		return s

