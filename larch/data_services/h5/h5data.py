from .h5pod import *
from .h5podgroup import H5PodGroup
from .. import _reserved_names_
from ..general import _sqz_same, selector_len_for

class H5Data():

	def __init__(self, altids=None, altnames=None, pods=None):

		self._master_n_cases = None
		self._master_altids = altids or []
		self._master_altnames = altnames or []

		self._pods_idco = H5PodGroup()
		self._pods_idca = H5PodGroup()
		self._pods_idce = H5PodGroup()
		self._pods_idga = H5PodGroup()

		self._vars_idco = Dict()
		self._vars_idca = Dict()
		self._vars_idga = Dict()

		if pods is not None:
			for pod in pods:
				self.add_pod(pod)

	def add_pod(self, pod):
		if isinstance(pod, H5PodCO):
			if self._master_n_cases is None:
				self._master_n_cases = pod.n_cases
			elif self._master_n_cases != pod.n_cases:
				raise ValueError(
					f"incompatible n_cases, have {self._master_n_cases} adding {pod.n_cases} in pod {pod!r}")
			self._pods_idco.append(pod)
			for k, v in pod._groupnode._v_children.items():
				if k not in _reserved_names_:
					self._vars_idco[k] = v
		elif isinstance(pod, H5PodCA):
			if self._master_n_cases is None:
				self._master_n_cases = pod.n_cases
			elif self._master_n_cases != pod.n_cases:
				raise ValueError(
					f"incompatible n_cases, have {self._master_n_cases} adding {pod.n_cases} in pod {pod!r}")
			self._pods_idca.append(pod)
			for k, v in pod._groupnode._v_children.items():
				if k not in _reserved_names_:
					self._vars_idca[k] = v
		elif isinstance(pod, H5PodCE):
			self._pods_idce.append(pod)
		elif isinstance(pod, H5PodGA):
			self._pods_idga.append(pod)
			for k, v in pod._groupnode._v_children.items():
				if k not in _reserved_names_:
					self._vars_idga[k] = v, pod._rowindex

	def set_alternatives(self, altids=None, altnames=None):
		if altids is not None:
			self._master_altids = altids
		if altnames is not None:
			self._master_altnames = altnames

	def alternative_codes(self):
		return self._master_altids

	def vars_co(self):
		return self._vars_idco

	def _get_dims_of_command(self, cmd_as_utf8):
		from tokenize import tokenize, untokenize, NAME, OP, STRING
		from io import BytesIO
		g = tokenize(BytesIO(cmd_as_utf8).readline)
		# identify needed dims
		dims = 1
		for toknum, tokval, _, _, _ in g:
			if toknum == NAME and ((tokval in self._vars_idca) or (tokval in self._vars_idga)):
				dims = 2
				break
		return dims

	def _remake_command(self, cmd, selector=None):
		from tokenize import tokenize, untokenize, NAME, OP, STRING
		DOT = (OP, '.')
		COLON = (OP, ':')
		COMMA = (OP, ',')
		OBRAC = (OP, '[')
		CBRAC = (OP, ']')
		OPAR = (OP, '(')
		CPAR = (OP, ')')
		from io import BytesIO
		recommand = []
		try:
			cmd_encode = cmd.encode('utf-8')
		except AttributeError:
			cmd_encode = str(cmd).encode('utf-8')
		dims = self._get_dims_of_command(cmd_encode)
		g = tokenize(BytesIO(cmd_encode).readline)
		if selector is None:
			screen_tokens = [COLON,]
		else:
			try:
				slicer_encode = selector.encode('utf-8')
			except AttributeError:
				slicer_encode = str(selector).encode('utf-8')
			screen_tokens = [(toknum, tokval) for toknum, tokval, _, _, _ in tokenize(BytesIO(slicer_encode).readline)]
		for toknum, tokval, _, _, _ in g:
			if toknum == NAME and tokval in self._vars_idca:
				if dims == 1:
					raise IncompatibleShape("cannot use idca.{} in an idco expression".format(tokval))
				# replace NAME tokens
				partial = [(NAME, 'self'), DOT, (NAME, '_vars_idca'), DOT, (NAME, tokval), OBRAC, ]
				partial += screen_tokens
				if dims > 1:
					partial += [COMMA, COLON, ]
				partial += [CBRAC, ]
				recommand.extend(partial)
			elif toknum == NAME and tokval in self._vars_idga:
				if dims == 1:
					raise IncompatibleShape("cannot use idca.{} in an idco expression".format(tokval))
				# replace NAME tokens
				if selector is None:
					partial = [(NAME, 'self'), DOT, (NAME, '_idga'), OPAR, (STRING, "'{}'".format(tokval)), CPAR, OBRAC, COLON]
				else:
					partial = [(NAME, 'self'), DOT, (NAME, '_idga'), OPAR, (STRING, "'{}'".format(tokval)), COMMA, ]
					partial += screen_tokens
					partial += [CPAR, OBRAC, COLON]
				if dims > 1:
					partial += [COMMA, COLON, ]
				partial += [CBRAC, ]
				recommand.extend(partial)
			elif toknum == NAME and tokval in self._vars_idco:
				# replace NAME tokens
				partial = [(NAME, 'self'), DOT, (NAME, '_vars_idco'), DOT, (NAME, tokval), OBRAC, ]
				partial += screen_tokens
				partial += [CBRAC,]
				if dims > 1:
					partial += [OBRAC, COLON, COMMA, (NAME, 'None'), CBRAC, ]
				recommand.extend(partial)
			else:
				recommand.append((toknum, tokval))
		# print("<recommand>")
		# print(recommand)
		# print("</recommand>")
		ret = untokenize(recommand).decode('utf-8')
		return asterize(ret)



	def _idga(self, varname, screen=None):
		if screen is None:
			screen=slice(None)
		valuenode, indexnode = self._vars_idga[varname]

		transpose_values = ('transpose_values' in valuenode._v_attrs)

		selectionlist = indexnode[screen]
		unique_selectionlist, rebuilder = numpy.unique(selectionlist, return_inverse=True)
		if transpose_values:
			if len(unique_selectionlist)==1 and unique_selectionlist[0]==0:
				return numpy.broadcast_to(valuenode[:], (len(rebuilder), valuenode.shape[0]))
			else:
				raise TypeError("transpose_values must be a vector")
		shapelen = len(valuenode.shape)
		if shapelen==1:
			values = valuenode[unique_selectionlist]
			return values[rebuilder]
		elif shapelen==2:
			values = valuenode[unique_selectionlist,:]
			return values[rebuilder,:]
		elif shapelen==3:
			values = valuenode[unique_selectionlist,:,:]
			return values[rebuilder,:,:]
		else:
			raise TypeError("shapelen cannot exceed 3")


	def _evaluate_single_cmd(self, cmd, selector=None):
		j = self._remake_command(cmd, selector=selector)
		try:
			return eval(j)
		except Exception as exc:
			args = exc.args
			if not args:
				arg0 = ''
			else:
				arg0 = args[0]
			arg0 = arg0 + '\nwithin parsed command: "{!s}"'.format(cmd)
			if "max" in cmd:
				arg0 = arg0 + '\n(note to get the maximum of arrays use "fmax" not "max")'.format(cmd)
			if "min" in cmd:
				arg0 = arg0 + '\n(note to get the maximum of arrays use "fmin" not "min")'.format(cmd)
			if isinstance(exc, NameError):
				badname = str(exc).split("'")[1]
				goodnames = dir()
				from ...util.text_manip import case_insensitive_close_matches
				did_you_mean_list = case_insensitive_close_matches(badname, goodnames, n=3, cutoff=0.1, excpt=None)
				if len(did_you_mean_list) > 0:
					arg0 = arg0 + '\n' + "did you mean {}?".format(
						" or ".join("'{}'".format(s) for s in did_you_mean_list))
			exc.args = (arg0,) + args[1:]
			raise

	def _load_into(self, item, result):
		if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[-1], slice):
			names, slice_ = item[:-1], item[-1]
		else:
			names = item
			slice_ = None

		# convert a single name string to a one item list
		if isinstance(names, str):
			names = [names, ]

		dims = 1
		for cmd in names:
			try:
				cmd_encode = cmd.encode('utf-8')
			except AttributeError:
				cmd_encode = str(cmd).encode('utf-8')
			i = self._get_dims_of_command(cmd_encode)
			if i > dims:
				dims = i

		if dims == 2:
			_sqz_same(result.shape, [selector_len_for(slice_, self._master_n_cases), len(self._master_altids), len(names)])
		elif dims == 1:
			_sqz_same(result.shape, [selector_len_for(slice_, self._master_n_cases), len(names)])
		else:
			raise TypeError("invalid dims")

		# important globals
		from ...util.aster import inXd
		from numpy import log, exp, log1p, absolute, fabs, sqrt, isnan, isfinite, logaddexp, fmin, fmax, nan_to_num, sin, cos, pi
		from ...util.common_functions import piece, normalize

		for i, cmd in enumerate(names):
			result[..., i] = self._evaluate_single_cmd(cmd, selector=slice_)

		return result

	def __getitem__(self, item):
		if isinstance(item, tuple) and len(item)>=2 and isinstance(item[-1], slice):
			names, slice_ = item[:-1], item[-1]
		else:
			names = item
			slice_ = None

		# convert a single name string to a one item list
		if isinstance(names, str):
			names = [names,]

		dims = 1
		for cmd in names:
			try:
				cmd_encode = cmd.encode('utf-8')
			except AttributeError:
				cmd_encode = str(cmd).encode('utf-8')
			i = self._get_dims_of_command(cmd_encode)
			if i > dims:
				dims = i

		dtype = numpy.float64

		if dims == 2:
			result = numpy.zeros( [selector_len_for(slice_, self._master_n_cases), len(self._master_altids), len(names)], dtype=dtype)
		elif dims == 1:
			result = numpy.zeros([selector_len_for(slice_, self._master_n_cases), len(names)], dtype=dtype)
		else:
			raise TypeError("invalid dims")

		# important globals
		from ...util.aster import inXd
		from numpy import log, exp, log1p, absolute, fabs, sqrt, isnan, isfinite, logaddexp, fmin, fmax, nan_to_num, sin, cos, pi
		from ...util.common_functions import piece, normalize

		for i, cmd in enumerate(names):
			result[...,i] = self._evaluate_single_cmd(cmd, selector=slice_)

		return result

	def caseindexes(self, screen=None):
		if screen is None:
			return numpy.arange(self._master_n_cases)
		else:
			return numpy.arange(self._master_n_cases)[screen]

	def array_idco(self, *vars, dtype=numpy.float64, screen=None, strip_nan=True):
		"""Extract a set of idco values.

		Parameters
		----------
		vars : tuple of str
			A tuple (or other iterable) giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idco` format variables.
		screen : None or slice
			If given, use this to slice the caseids used to build
			the array.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.

		Other Parameters
		----------------
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably
			numpy.int64, numpy.float64, or numpy.bool_.

		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,len(vars)).
		"""
		result = self._pods_idco._load_into( vars, screen, result=None, dtype=dtype)
		if strip_nan:
			result = numpy.nan_to_num(result)
		return result


	def array_idca(self, *vars, dtype=numpy.float64, screen=None, strip_nan=True):
		"""Extract a set of idca values.

		Parameters
		----------
		vars : tuple of str
			A tuple (or other iterable) giving the expressions (often column names, but any valid
			SQLite expression works) to extract as :ref:`idco` format variables.
		screen : None or slice
			If given, use this to slice the caseids used to build
			the array.
		strip_nan : bool
			If True (the default) then NAN values are converted to 0, and INF
			values are converted to the largest magnitude real number representable
			in the selected dtype.

		Other Parameters
		----------------
		dtype : str or dtype
			Describe the data type you would like the output array to adopt, probably
			numpy.int64, numpy.float64, or numpy.bool_.

		Returns
		-------
		data : ndarray
			An array with specified dtype, of shape (n_cases,len(vars)).
		"""
		result = self._pods_idca._load_into( vars, screen, result=None, dtype=dtype)
		if strip_nan:
			result = numpy.nan_to_num(result)
		return result

	def __repr__(self):
		s = f"<larch.{self.__class__.__name__}>"
		s += f"\n  cases: {self._master_n_cases}"
		s += f"\n  alternatives:"
		for i,n in zip(self._master_altids,self._master_altnames):
			s += f"\n    {i}: {n}"
		return s
