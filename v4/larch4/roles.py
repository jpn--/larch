import keyword as _keyword
import re as _re
# from .core import LinearComponent, LinearFunction
# from .exceptions import LarchError

from .util.naming import parenthize, valid_identifier_or_parenthized_string
from .util.touch_notifier import TouchNotify

_chi = u"\u03C7"
_beta = u"\u03B2"
_hahook = u"\u04FC"

_ParameterRef_repr_txt = 'P'
_DataRef_repr_txt = 'X'


class Role(type):
	def __getattr__(cls, key):
		try:
			return super().getattr(key)
		except AttributeError:
			return cls(key)


_data_description_catalog = {}


class DataRef(str, metaclass=Role):
	def __new__(cls, name, **kwarg):
		if hasattr(str, name):
			raise NameError("cannot create DataRef with the name of a str method ({})".format(name))
		if name in ('_descrip', 'descrip', '_role'):
			raise NameError("cannot create DataRef with the name ({})".format(name))
		try:
			raise TypeError("debugging")  # The getattr here is not really needed?
			return getattr(cls, name)
		except:
			self = super().__new__(cls, name)
			self._role = 'data'
			return self

	def __init__(self, name, descrip=None):
		if descrip is not None:
			_data_description_catalog[self] = descrip
		super().__init__()

	@property
	def descrip(self):
		if self in _data_description_catalog:
			return _data_description_catalog[self]
		return None

	@descrip.setter
	def descrip(self, value):
		if value is not None:
			_data_description_catalog[self] = value

	def __repr__(self):
		s = super().__str__()
		if _re.match("[_A-Za-z][_a-zA-Z0-9]*$", s) and not _keyword.iskeyword(s):
			return "{}.{}".format(_DataRef_repr_txt, s)
		else:
			return "{}('{}')".format(_DataRef_repr_txt, s)

	def __add__(self, other):
		if isinstance(other, (ParameterRef, LinearComponent, LinearFunction)):
			return LinearComponent(data=str(self), param="CONSTANT") + other
		return DataRef("{}+{}".format(parenthize(self), parenthize(other, True)))

	def __radd__(self, other):
		if isinstance(other, (ParameterRef, LinearComponent, LinearFunction)):
			return other + LinearComponent(data=str(self), param="CONSTANT")
		return DataRef("{}+{}".format(parenthize(other), parenthize(self, True)))

	def __sub__(self, other):
		if isinstance(other, (ParameterRef, LinearComponent, LinearFunction)):
			return LinearComponent(data=str(self), param="CONSTANT") - other
		return DataRef("{}-{}".format(parenthize(self), parenthize(other, True)))

	def __rsub__(self, other):
		if isinstance(other, (ParameterRef, LinearComponent, LinearFunction)):
			return other - LinearComponent(data=str(self), param="CONSTANT")
		return DataRef("{}-{}".format(parenthize(other), parenthize(self, True)))

	def __mul__(self, other):
		if isinstance(other, LinearFunction):
			trial = LinearFunction()
			for component in other:
				trial += self * component
			return trial
		if isinstance(other, LinearComponent):
			return LinearComponent(data=str(self * other.data), param=str(other.param))
		if type(other) is ParameterRef:
			return LinearComponent(data=str(self), param=str(other))
		if type(other) is _param_negate:
			return LinearComponent(data=str(-self), param=str(other._orig))
		if other == 0:
			return 0
		return DataRef("{}*{}".format(parenthize(self), parenthize(other, True)))

	def __rmul__(self, other):
		if isinstance(other, LinearFunction):
			raise NotImplementedError
		if isinstance(other, LinearComponent):
			return LinearComponent(data=str(other.data * self), param=str(other.param))
		if type(other) is ParameterRef:
			return LinearComponent(data=str(self), param=str(other))
		if type(other) is _param_negate:
			return LinearComponent(data=str(-self), param=str(other._orig))
		if other == 0:
			return 0
		return DataRef("{}*{}".format(parenthize(other), parenthize(self, True)))

	def __truediv__(self, other):
		if isinstance(other, LinearFunction):
			raise NotImplementedError
		if isinstance(other, LinearComponent):
			raise NotImplementedError
		if isinstance(other, ParameterRef):
			raise NotImplementedError
		return DataRef("{}/{}".format(parenthize(self), parenthize(other, True)))

	def __rtruediv__(self, other):
		if isinstance(other, LinearFunction):
			raise NotImplementedError
		if isinstance(other, LinearComponent):
			raise NotImplementedError
		if isinstance(other, ParameterRef):
			raise NotImplementedError
		return DataRef("{}/{}".format(parenthize(other), parenthize(self, True)))

	def __neg__(self):
		return DataRef("-({})".format(self, True))

	def __pos__(self):
		return self

	def __eq__(self, other):
		if isinstance(other, DataRef):
			if repr(other) == repr(self):
				return True
		if isinstance(other, ParameterRef):
			return False
		if isinstance(other, str):
			if other == super().__str__():
				return True
		return False

	def __hash__(self):
		return hash(super().__str__())

	def eval(self, namespace=None, *, globals=None, **more_namespace):
		import numpy
		use_namespace = {'exp': numpy.exp, 'log': numpy.log, 'log1p': numpy.log1p, 'fabs': numpy.fabs,
		                 'sqrt': numpy.sqrt,
		                 'absolute': numpy.absolute, 'isnan': numpy.isnan, 'isfinite': numpy.isfinite,
		                 'logaddexp': numpy.logaddexp, 'fmin': numpy.fmin, 'fmax': numpy.fmax,
		                 'nan_to_num': numpy.nan_to_num}
		if namespace is not None:
			use_namespace.update(namespace)
		use_namespace.update(more_namespace)
		return eval(self, globals, use_namespace)


class ParameterRef(str, metaclass=Role):
	"""
	An abstract reference to a parameter, which may or may not be included in any given model.

	Parameters
	----------
	name : str
		The name of the parameter to reference.
	default : numeric or None
		When a targeted model does not include the referenced parameter, use
		this value for :meth:`value` and :meth:`str`. If given as None, those
		methods will raise an exception.
	fmt : str or None
		The format string to use for :meth:`str`.
	"""

	def __new__(cls, name, default=None, fmt=None, default_value=None, format=None):
		if fmt is None and format is not None:
			fmt = format
		if fmt is None:
			fmt = "{}"
		if default is None and default_value is not None:
			default = default_value
		self = super().__new__(cls, name)
		self._fmt = fmt
		self._name = name
		self._role = 'parameter'
		self._default_value = default
		return self

	def __repr__(self):
		s = super().__str__()
		if _re.match("[_A-Za-z][_a-zA-Z0-9]*$", s) and not _keyword.iskeyword(s):
			return "{}.{}".format(_ParameterRef_repr_txt, s)
		else:
			return "{}('{}')".format(_ParameterRef_repr_txt, s)

	def __str__(self):
		return super().__str__()

	def value(self, m):
		"""
		The value of the parameter in a given model.

		Parameters
		----------
		m : Model
			The model from which to extract a parameter value.

		Raises
		------
		LarchError
			When the model does not contain a parameter with the same
			name as this ParameterRef, and the default_value for this
			ParameterRef is None.
		"""
		try:
			v = m.metaparameter(str(self)).value
		except (LarchError, KeyError):
			if self._default_value is not None:
				v = self._default_value
			else:
				raise
		return v

	def name(self, n):
		"""
		Set or re-set the name of the parameter referenced.

		Parameters
		----------
		n : str
			The new name of the parameter.

		Returns
		-------
		ParameterRef
			This method returns the self object, to facilitate method chaining.
		"""
		self._name = n
		return self

	def fmt(self, format):
		"""
		Set or re-set the format string for the parameter referenced.

		Parameters
		----------
		format : str
			The new format string. This will be used to format the parameter value when calling :meth:`str`.

		Returns
		-------
		ParameterRef
			This method returns the self object, to facilitate method chaining.
		"""
		if format is None or format == "":
			self._fmt = "{}"
		else:
			self._fmt = format
		try:
			self._fmt.format(0)
		except:
			raise LarchError("invalid formating string")
		return self

	def default_value(self, val):
		"""
		Set or re-set the default value.

		Parameters
		----------
		val : numeric or None
			The new default value. This will be used by :meth:`value` when
			the parameter is not included in the targeted model. Set to None
			to raise an exception instead.

		Returns
		-------
		ParameterRef
			This method returns the self object, to facilitate method chaining.
		"""
		self._default_value = val
		return self

	def strf(self, m, fmt=None):
		"""
		Gives the :meth:`value` of the parameter in a given model as a string.

		The string is formated using the :meth:`fmt` string if given. If not
		given, Python's default string formatting is used.

		Parameters
		----------
		m : Model
			The model from which to extract a parameter value.
		fmt : str or None
			If not None, a format string which will be used, overriding any
			previous setting by a :meth:`fmt` command.  The override is
			valid for this method call only, and does not change the
			format for future calls.

		Raises
		------
		KeyError
			When the model does not contain a parameter with the same
			name as this ParameterRef, and the default_value for this
			ParameterRef is None.
		"""
		try:
			if fmt is not None:
				return fmt.format(self.value(m))
			else:
				return self._fmt.format(self.value(m))
		except LarchError as err:
			if "not in model" in str(err):
				return "NA"
			else:
				raise
		except KeyError:
			return "NA"

	def getname(self):
		return self._name

	def valid(self, m):
		"""
		Check if this ParameterRef would give a value for a given model.

		Parameters
		----------
		m : Model
			The model from which to extract a parameter value.

		Returns
		-------
		bool
			False if the value method would raise an exception, and True otherwise.
		"""
		if str(self) in m:
			return True
		if self._default_value is not None:
			return True
		return False

	def __add__(self, other):
		if other == 0:
			return self
		if isinstance(other, (DataRef, LinearComponent, LinearFunction)):
			return LinearComponent(param=str(self), data="1") + other
		return _param_add(self, other)

	def __radd__(self, other):
		if other == 0:
			return self
		if isinstance(other, (DataRef, LinearComponent, LinearFunction)):
			return other + LinearComponent(param=str(self), data="1")
		return _param_add(other, self)

	def __sub__(self, other):
		if other == 0:
			return self
		if isinstance(other, (DataRef, LinearComponent, LinearFunction)):
			return LinearComponent(param=str(self), data="1") - other
		return _param_subtract(self, other)

	def __rsub__(self, other):
		if other == 0:
			return self
		if isinstance(other, (DataRef, LinearComponent, LinearFunction)):
			return other - LinearComponent(param=str(self), data="1")
		return _param_subtract(other, self)

	def __mul__(self, other):
		if isinstance(other, (LinearComponent, LinearFunction)):
			raise TypeError("cannot do ParameterRef * LinearComponent")
		if type(other) is DataRef:
			return LinearComponent(data=str(other), param=str(self))
		return _param_multiply(self, other)

	def __rmul__(self, other):
		if isinstance(other, (LinearComponent, LinearFunction)):
			raise TypeError("cannot do LinearComponent * ParameterRef")
		if type(other) is DataRef:
			return LinearComponent(data=str(other), param=str(self))
		return _param_multiply(other, self)

	def __truediv__(self, other):
		if isinstance(other, DataRef):
			return self * (1 / other)
		if isinstance(other, (LinearComponent, LinearFunction)):
			raise TypeError("cannot do ParameterRef / LinearComponent")
		return _param_divide(self, other)

	def __rtruediv__(self, other):
		if isinstance(other, DataRef):
			raise TypeError("cannot do DataRef / ParameterRef")
		if isinstance(other, (LinearComponent, LinearFunction)):
			raise TypeError("cannot do LinearComponent / ParameterRef")
		return _param_divide(other, self)

	def __neg__(self):
		return _param_negate(self)

	def __pos__(self):
		return self

	def __eq__(self, other):
		if isinstance(other, ParameterRef):
			if repr(other) == repr(self):
				return True
		if isinstance(other, DataRef):
			return False
		if isinstance(other, str):
			if other == super().__str__():
				return True
		return False

	def __hash__(self):
		return hash(repr(self))

	def _to_Linear(self):
		return LinearComponent(param=str(self), data="1")


class _param_math_binaryop(ParameterRef):
	def __new__(cls, left, right):
		self = super().__new__(cls, "")
		self._left = left
		self._right = right
		if isinstance(self._left, ParameterRef) and isinstance(self._right, ParameterRef):
			if left._fmt == "{}" and right._fmt != "{}":
				self._fmt = right._fmt
			else:
				self._fmt = left._fmt
		elif isinstance(self._left, ParameterRef):
			self._fmt = left._fmt
		elif isinstance(self._right, ParameterRef):
			self._fmt = right._fmt
		else:
			self._fmt = "{}"
		self._name = ""
		self._role = 'parameter_math'
		return self


class _param_add(_param_math_binaryop):
	def value(self, m):
		if isinstance(self._left, ParameterRef):
			x = self._left.value(m)
		else:
			x = self._left
		if isinstance(self._right, ParameterRef):
			x += self._right.value(m)
		else:
			x += self._right
		return x

	def valid(self, m):
		if isinstance(self._left, ParameterRef):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, ParameterRef):
			x &= self._right.valid(m)
		return x

	def __repr__(self):
		return "({} + {})".format(repr(self._left), repr(self._right))

	def _to_Linear(self):
		return self._left._to_Linear() + self._right._to_Linear()

	def __add__(self, other):
		if isinstance(other, (DataRef, LinearComponent, LinearFunction)):
			return self._to_Linear() + other
		return super().__add__(other)


class _param_subtract(_param_math_binaryop):
	def value(self, m):
		if isinstance(self._left, ParameterRef):
			x = self._left.value(m)
		else:
			x = self._left
		if isinstance(self._right, ParameterRef):
			x -= self._right.value(m)
		else:
			x -= self._right
		return x

	def valid(self, m):
		if isinstance(self._left, ParameterRef):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, ParameterRef):
			x &= self._right.valid(m)
		return x

	def __repr__(self):
		return "({} - {})".format(repr(self._left), repr(self._right))


class _param_multiply(_param_math_binaryop):
	def value(self, m):
		if isinstance(self._left, ParameterRef):
			x = self._left.value(m)
		else:
			x = self._left
		if isinstance(self._right, ParameterRef):
			x *= self._right.value(m)
		else:
			x *= self._right
		return x

	def valid(self, m):
		if isinstance(self._left, ParameterRef):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, ParameterRef):
			x &= self._right.valid(m)
		return x

	def __repr__(self):
		return "({} * {})".format(repr(self._left), repr(self._right))


class _param_divide(_param_math_binaryop):
	def value(self, m):
		if isinstance(self._left, ParameterRef):
			x = self._left.value(m)
		else:
			x = self._left
		try:
			if isinstance(self._right, ParameterRef):
				x /= self._right.value(m)
			else:
				x /= self._right
		except ZeroDivisionError:
			return float('NaN')
		return x

	def valid(self, m):
		if isinstance(self._left, ParameterRef):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, ParameterRef):
			x &= self._right.valid(m)
		return x

	def __repr__(self):
		return "({} / {})".format(repr(self._left), repr(self._right))

	def __str__(self):
		return "({} / {})".format(repr(self._left), repr(self._right))


class _param_negate(ParameterRef):
	def __new__(cls, orig):
		self = super().__new__(cls, "")
		self._orig = orig
		self._fmt = orig._fmt
		self._name = "-({})".format(orig.getname())
		self._role = 'parameter_math'
		return self

	def value(self, m):
		if isinstance(self._orig, ParameterRef):
			return -self._orig.value(m)
		else:
			return -self._orig

	def valid(self, m):
		if isinstance(self._orig, ParameterRef):
			return self._orig.valid(m)
		else:
			return True

	def __repr__(self):
		return "-({})".format(repr(self._orig))


try:
	from numpy import log as _log, exp as _exp
except ImportError:
	from math import log as _log, exp as _exp


def log(x):
	if isinstance(x, DataRef):
		return DataRef("log({})".format(x))
	else:
		return _log(x)


def exp(x):
	if isinstance(x, DataRef):
		return DataRef("exp({})".format(x))
	else:
		return _exp(x)


class CombinedRef(metaclass=Role):
	def __new__(cls, s):
		return LinearComponent(data=DataRef(s), param=ParameterRef(s))


# ===========================================================================

class LinearComponent(TouchNotify):
	def __init__(self, *args, param=None, data=None, **kwargs):
		if len(args):
			if isinstance(args[0], LinearComponent):
				super().__init__(**kwargs)
				self._param = param or args[0]._param
				self._data = data or args[0]._data
				self.set_touch_callback(args[0].get_touch_callback())
			else:
				raise TypeError('init LinearComponent with LinearComponent or keyword arguments')
		super().__init__(**kwargs)
		self._param = param or '1'
		self._data = data or '1'

	@property
	def data(self):
		return DataRef(self._data)

	@data.setter
	def data(self, value):
		self._data = value

	@property
	def param(self):
		return ParameterRef(self._param)

	@param.setter
	def param(self, value):
		self._param = value

	def __pos__(self):
		return self

	def __add__(self, other):
		if other == ():
			return self
		elif isinstance(other, (LinearFunction, LinearComponent)):
			return (LinearFunction(touch_callback=self.get_touch_callback()) + self) + other
		elif other == 0:
			return self
		else:
			raise NotImplementedError()

	def __radd__(self, other):
		return other.__add__(self)

	def __mul__(self, other):
		if isinstance(other, (int, float, DataRef)):
			return LinearComponent(data=self.data * other, param=self.param, touch_callback=self.get_touch_callback())
		else:
			raise TypeError('unsupported operand type(s) for LinearComponent: {}'.format(type(other)))

	def __rmul__(self, other):
		if isinstance(other, (int, float, DataRef)):
			return LinearComponent(data=self.data * other, param=self.param, touch_callback=self.get_touch_callback())
		else:
			raise TypeError('unsupported operand type(s) for LinearComponent: {}'.format(type(other)))

	def __imul__(self, other):
		if isinstance(other, (int, float, DataRef)):
			self.data = self.data * other
		else:
			raise TypeError('unsupported operand type(s) for LinearComponent: {}'.format(type(other)))

	def __iter__(self):
		return iter(LinearFunction(touch_callback=self.get_touch_callback()) + self)

	def __eq__(self, other):
		if isinstance(other, LinearFunction) and len(other) == 1:
			other = other[0]
		if not isinstance(other, LinearComponent):
			return False
		if self.param != other.param:
			return False
		if self.data != other.data:
			return False
		return True

	def __repr__(self):
		return f"({self.param!r} * {self.data!r})"

# ===========================================================================

class LinearFunction(TouchNotify, list):
	def __init__(self, *args, touch_callback=None, **kwargs):
		if len(args) == 1 and args[0] is None:
			args = ()
		super().__init__(*args, **kwargs)
		self.set_touch_callback(touch_callback)

	def __add__(self, other):
		if other == ():
			return self
		if isinstance(other, LinearFunction):
			return super().__add__(other)
		if isinstance(other, ParameterRef):
			other = LinearComponent(param=other, touch_callback=self.get_touch_callback())
		if isinstance(other, LinearComponent):
			result = type(self)(self, touch_callback=self.get_touch_callback())
			result.append(other)
			return result
		raise TypeError("cannot add type {} to LinearFunction".format(type(other)))

	def __iadd__(self, other):
		if isinstance(other, ParameterRef):
			other = LinearComponent(param=other, touch_callback=self.get_touch_callback())
		if other == ():
			return self
		elif isinstance(other, LinearFunction):
			super().__iadd__(other)
		elif isinstance(other, LinearComponent):
			super().append(other)
		else:
			raise TypeError("cannot add type {} to LinearFunction".format(type(other)))
		return self

	def __radd__(self, other):
		return LinearFunction(touch_callback=self.get_touch_callback()) + other + self

	def __pos__(self):
		return self

	def __mul__(self, other):
		trial = LinearFunction(touch_callback=self.get_touch_callback())
		for component in self:
			trial += component * other
		return trial

	def __rmul__(self, other):
		trial = LinearFunction(touch_callback=self.get_touch_callback())
		for component in self:
			trial += other * component
		return trial

	# def evaluate(self, dataspace, model):
	# 	if len(self) > 0:
	# 		i = self[0]
	# 		y = i.data.eval(**dataspace) * i.param.default_value(0).value(model)
	# 	for i in self[1:]:
	# 		y += i.data.eval(**dataspace) * i.param.default_value(0).value(model)
	# 	return y
	#
	# def evaluator1d(self, factorlabel='', x_ident=None):
	# 	if x_ident is None:
	# 		try:
	# 			x_ident = self._x_ident
	# 		except AttributeError:
	# 			raise TypeError('a x_ident must be given')
	# 	if x_ident is None:
	# 		raise TypeError('a x_ident must be given')
	# 	from .util.plotting import ComputedFactor
	# 	return ComputedFactor(label=factorlabel, func=lambda x, m: self.evaluate({x_ident: x}, m))

	def __contains__(self, val):
		if isinstance(val, ParameterRef):
			for i in self:
				if i.param == val:
					return True
			return False
		if isinstance(val, DataRef):
			for i in self:
				if i.data == val:
					return True
			return False
		raise TypeError("the searched for content must be of type ParameterRef or DataRef")

	def _index_of(self, val):
		if isinstance(val, ParameterRef):
			for n, i in enumerate(self):
				if i.param == val:
					return n
			raise KeyError('ParameterRef not found')
		if isinstance(val, DataRef):
			for n, i in enumerate(self):
				if i.data == val:
					return n
			raise KeyError('DataRef not found')
		raise TypeError("the searched for content must be of type ParameterRef or DataRef")

	def reformat_param(self, container=None, pattern=None, repl=None, **kwargs):
		"""
		Transform all the parameters in the LinearFunction.

		Parameters
		----------
		container : str
			A format string, into which the previous parameters are formatted.
			Use this to append things to the parameter names.
		pattern : str
		repl : str
			Passed to `re.sub` with each existing parameter as the base string
			to be searched.
		"""
		import re
		r = LinearFunction(touch_callback=self.get_touch_callback())
		for i in self:
			if pattern is None:
				param = i.param
			else:
				if repl is None:
					raise TypeError('must give repl with pattern')
				param = re.sub(pattern, repl, i.param, **kwargs)
			if container is None:
				container = '{}'
			r += LinearComponent(data=i.data, param=container.format(param), touch_callback=self.get_touch_callback())
		try:
			r._x_ident = self._x_ident
		except AttributeError:
			pass
		return r

	def reformat_data(self, container=None, pattern=None, repl=None, **kwargs):
		"""
		Transform all the data in the LinearFunction.

		Parameters
		----------
		container : str
			A format string, into which the previous data strings are formatted.
			Use this to apply common global transforms to the data.
		pattern : str
		repl : str
			Passed to `re.sub` with each existing data string as the base string
			to be searched.
		"""
		import re
		r = LinearFunction(touch_callback=self.get_touch_callback())
		for i in self:
			if pattern is None:
				data = i.data
			else:
				if repl is None:
					raise TypeError('must give repl with pattern')
				data = re.sub(pattern, repl, i.data, **kwargs)
			if container is None:
				container = '{}'
			r += LinearComponent(data=container.format(data), param=i.param, touch_callback=self.get_touch_callback())
		try:
			r._x_ident = self._x_ident
		except AttributeError:
			pass
		return r

	def __eq__(self, other):
		if not isinstance(other, LinearFunction):
			return False
		if len(self) != len(other):
			return False
		for i, j in zip(self, other):
			if i != j: return False
		return True

	def __getstate__(self):
		state = {}
		state['code'] = self.__code__()
		try:
			state['_x_ident'] = self._x_ident
		except AttributeError:
			pass
		return state

	def __setstate__(self, state):
		self.__init__()
		self.__iadd__(eval(state['code']))
		if '_x_ident' in state:
			self._x_ident = state['_x_ident']

	def __repr__(self):
		result = " + ".join(repr(i) for i in self)
		if len(result)<80:
			return result
		else:
			return "  "+result.replace(" + ","\n+ ")

	def set_touch_callback(self, callback):
		super().set_touch_callback(callback)
		for i in self:
			i.set_touch_callback(callback)

# ====================================================================================================

from itertools import chain
_RaiseKeyError = object() # singleton for no-default behavior

class DictOfLinearFunction(TouchNotify, dict):
	@staticmethod # because this doesn't make sense as a global function.
	def _process_args(mapping=(), **kwargs):
		if hasattr(mapping, 'items'):
			mapping = getattr(mapping, 'items')()
		return ((k, v) for k, v in chain(mapping, getattr(kwargs, 'items')()))
	def __init__(self, mapping=(), touch_callback=None, alts_validator=None, **kwargs):
		if mapping is None:
			mapping = ()
		super().__init__(self._process_args(mapping, **kwargs))
		self.set_touch_callback(touch_callback if callable(touch_callback) else lambda: None)
		self._alts_validator = alts_validator
	def set_alts_validator(self, av):
		self._alts_validator = av
	def __getitem__(self, k):
		try:
			return super().__getitem__(k)
		except KeyError:
			if self._alts_validator is None:
				raise
			if self._alts_validator(k):
				super().__setitem__(k, LinearFunction(touch_callback=self.get_touch_callback()))
				return super().__getitem__(k)
			else:
				raise
	def __setitem__(self, k, v):
		if isinstance(v, LinearComponent):
			v = LinearFunction(touch_callback=self.get_touch_callback()) + v
		elif isinstance(v, LinearFunction):
			v.set_touch_callback(self.get_touch_callback())
		else:
			raise TypeError("only accepts LinearFunction values")
		return super().__setitem__(k, v)
	def setdefault(self, k, default=None):
		if default is None:
			default = LinearFunction(touch_callback=self.get_touch_callback())
		if isinstance(default, LinearComponent):
			default = LinearFunction(touch_callback=self.get_touch_callback()) + default
		elif isinstance(default, LinearFunction):
			default.set_touch_callback(self.get_touch_callback())
		else:
			raise TypeError("only accepts LinearFunction values")
		return super().setdefault(k, default)
	def copy(self): # don't delegate w/ super - dict.copy() -> dict :(
		return type(self)(self)
	def __repr__(self):
		return '{0}({1})'.format(type(self).__name__, super().__repr__())
	def set_touch_callback(self, callback):
		super().set_touch_callback(callback)
		for i in self.values():
			i.set_touch_callback(callback)


#
#
#

globals()[_ParameterRef_repr_txt] = ParameterRef
globals()[_DataRef_repr_txt] = DataRef
globals()[_ParameterRef_repr_txt + _DataRef_repr_txt] = CombinedRef


P = ParameterRef
X = DataRef
PX = CombinedRef

__all__ = [_ParameterRef_repr_txt, _DataRef_repr_txt, _ParameterRef_repr_txt + _DataRef_repr_txt]
