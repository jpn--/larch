import numpy
from ...util.aster import asterize
from ..general import _sqz_same, selector_len_for

class H5PodGroup():
	"""
	A DataPodGroup is a list of DataPods that all provide identically shaped data arrays.
	"""

	def __init__(self, members=None):
		self._pods = []
		if members is not None:
			for m in members:
				self.append(m)

	def append(self, x):
		from . import H5Pod
		if not isinstance(x, H5Pod):
			raise TypeError
		return self._pods.append(x)

	@property
	def shape(self):
		if len(self._pods):
			return self._pods[0].shape
		raise ValueError('no pods')

	def _getgroup(self, i):
		return self._pods[i]._groupnode

	def __getattr__(self, item):
		if item[:10]=='_groupnode' and len(item)>10:
			i = int(item[10:])
			return self._pods[i]._groupnode
		if item[:5]=='_pods' and len(item)>5:
			i = int(item[5:])
			return self._pods[i]
		raise AttributeError(item)

	def __len__(self):
		return len(self._pods)

	def _remake_command(self, cmd, selector=None, receiver=None):
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

		if receiver:
			recommand += [(NAME, receiver), OBRAC, COLON, CBRAC, (OP, '='), ]

		try:
			cmd_encode = cmd.encode('utf-8')
		except AttributeError:
			cmd_encode = str(cmd).encode('utf-8')
		dims = len(self.shape)
		g = tokenize(BytesIO(cmd_encode).readline)
		if selector is None:
			screen_tokens = [COLON,]
		else:
			# try:
			# 	slicer_encode = selector.encode('utf-8')
			# except AttributeError:
			# 	slicer_encode = str(selector).encode('utf-8')
			# screen_tokens = [(toknum, tokval) for toknum, tokval, _, _, _ in tokenize(BytesIO(slicer_encode).readline)]
			screen_tokens = [(NAME, 'selector'), ]
		for toknum, tokval, _, _, _ in g:
			tok_accept = False
			for j in range(len(self)):
				if toknum == NAME and tokval in self._pods[j]:
					# replace NAME tokens
					partial = [(NAME, 'self'), DOT, (NAME, f'_pods{j}'), OBRAC, (STRING, f"'{tokval}'"), COMMA, ]
					partial += screen_tokens
					partial += [CBRAC, DOT, (NAME, f'squeeze'), OPAR, CPAR ]
					recommand.extend(partial)
					tok_accept = True
					break
			if not tok_accept:
				recommand.append((toknum, tokval))
		ret = untokenize(recommand).decode('utf-8')
		return asterize(ret, mode="exec" if receiver is not None else "eval"), ret


	def _evaluate_single_item(self, cmd, selector=None, receiver=None):
		j, j_raw = self._remake_command(cmd, selector=selector, receiver='receiver' if receiver is not None else None)
		# important globals
		from ...util.aster import inXd
		from numpy import log, exp, log1p, absolute, fabs, sqrt, isnan, isfinite, logaddexp, fmin, fmax, nan_to_num, sin, cos, pi
		from ...util.common_functions import piece, normalize, boolean
		try:
			if receiver is not None:
				exec(j)
			else:
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
				arg0 = arg0 + '\n(note to get the minimum of arrays use "fmin" not "min")'.format(cmd)
			if isinstance(exc, NameError):
				badname = str(exc).split("'")[1]
				goodnames = set(dir())
				from ...util.text_manip import case_insensitive_close_matches
				did_you_mean_list = case_insensitive_close_matches(badname, goodnames, n=3, cutoff=0.1, excpt=None)
				if len(did_you_mean_list) > 0:
					arg0 = arg0 + '\n' + "did you mean {}?".format(
						" or ".join("'{}'".format(s) for s in did_you_mean_list))
			exc.args = (arg0,) + args[1:]
			raise

	def __getitem__(self, item):
		if isinstance(item, tuple) and len(item)>=2 and isinstance(item[-1], slice):
			names, slice_ = item[:-1], item[-1]
		else:
			names = item
			slice_ = None

		# convert a single name string to a one item list
		if isinstance(names, str):
			names = [names,]

		dtype = numpy.float64

		result = numpy.zeros( [selector_len_for(slice_, self.shape[0]), *self.shape[1:], len(names)], dtype=dtype)

		for i, cmd in enumerate(names):
			result[...,i] = self._evaluate_single_item(cmd, slice_)

		return result

	def _load_into(self, names, slc, result=None, dtype=numpy.float64):
		# convert a single name string to a one item list
		if isinstance(names, str):
			names = [names, ]
		if result is None:
			# initialize a new array of the indicated dtype
			result = numpy.zeros([selector_len_for(slc, self.shape[0]), *self.shape[1:], len(names)], dtype=dtype)
		else:
			_sqz_same (result.shape,[selector_len_for(slc, self.shape[0]), *self.shape[1:], len(names)])
		for i, cmd in enumerate(names):
			result[..., i] = self._evaluate_single_item(cmd, slc)
		return result

	def load_into(self, names, selector=None, result_array=None, dtype=numpy.float64, mask=None):
		"""
		Load some data.

		Parameters
		----------
		names : iterable of str
			The variable names to load.
		selector : slice, optional
			A slice or bool array used to identify which rows to load.
		result_array : ndarray, optional
			The array to load the data into.  Must be the correct size. If not
			given, an array will be created.
		dtype : dtype, optional
			The dtype for the result_array to be created.
		mask : list, optional
			A list of the same shape as `names`.  For each item, if the mask evaluates
			as False, then the data will be skipped.  This is an optimization tool
			for reloading data that may not have changed (e.g. for logsum generation).

		Returns
		-------
		ndarray
		"""
		# convert a single name string to a one item list
		if isinstance(names, str):
			names = [names, ]
		if result_array is None:
			# initialize a new array of the indicated dtype
			result_array = numpy.zeros([selector_len_for(selector, self.shape[0]), *self.shape[1:], len(names)], dtype=dtype)
		else:
			_sqz_same(result_array.shape,[selector_len_for(selector, self.shape[0]), *self.shape[1:], len(names)])
		if mask is None:
			for i, cmd in enumerate(names):
				self._evaluate_single_item(cmd, selector, receiver=result_array[..., i])
		else:
			for i, cmd in enumerate(names):
				if mask[i]:
					self._evaluate_single_item(cmd, selector, receiver=result_array[..., i])
		return result_array

	def __repr__(self):
		s = f"<larch.{self.__class__.__name__}>"
		return s
