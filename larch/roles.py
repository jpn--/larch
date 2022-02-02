import keyword as _keyword
import re as _re
from numbers import Number
import numpy
import copy

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
			if cls is ParameterRef or cls is DataRef or cls is CombinedRef:
				return cls(key)
			else:
				raise


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

	def __getnewargs__(self):
		return (str(self), )

	def __getstate__(self):
		return self.__dict__

	def __setstate__(self, d):
		self.__dict__.update(d)

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
		if type(other) is _param_multiply:
			if isinstance(other._left, ParameterRef):
				return LinearComponent(data=str(self), param=str(other._left)) * other._right
			else:
				return LinearComponent(data=str(self), param=str(other._right)) * other._left
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

	def __and__(self, other):
		if isinstance(other, (ParameterRef, LinearComponent, LinearFunction)):
			raise NotImplementedError
		return DataRef("{}&{}".format(parenthize(self), parenthize(other, True)))

	def __rand__(self, other):
		if isinstance(other, (ParameterRef, LinearComponent, LinearFunction)):
			raise NotImplementedError
		return DataRef("{}&{}".format(parenthize(other), parenthize(self, True)))

	def __or__(self, other):
		if isinstance(other, (ParameterRef, LinearComponent, LinearFunction)):
			raise NotImplementedError
		return DataRef("{}|{}".format(parenthize(self), parenthize(other, True)))

	def __ror__(self, other):
		if isinstance(other, (ParameterRef, LinearComponent, LinearFunction)):
			raise NotImplementedError
		return DataRef("{}|{}".format(parenthize(other), parenthize(self, True)))

	def __xor__(self, other):
		if isinstance(other, (ParameterRef, LinearComponent, LinearFunction)):
			raise NotImplementedError
		return DataRef("{}^{}".format(parenthize(self), parenthize(other, True)))

	def __rxor__(self, other):
		if isinstance(other, (ParameterRef, LinearComponent, LinearFunction)):
			raise NotImplementedError
		return DataRef("{}^{}".format(parenthize(other), parenthize(self, True)))

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
		from .util.common_functions import piece, hard_sigmoid
		use_namespace = {'exp': numpy.exp, 'log': numpy.log, 'log1p': numpy.log1p, 'fabs': numpy.fabs,
		                 'sqrt': numpy.sqrt,
		                 'absolute': numpy.absolute, 'isnan': numpy.isnan, 'isfinite': numpy.isfinite,
		                 'logaddexp': numpy.logaddexp, 'fmin': numpy.fmin, 'fmax': numpy.fmax,
		                 'nan_to_num': numpy.nan_to_num,
						 'piece': piece, 'hard_sigmoid':hard_sigmoid,}
		if namespace is not None:
			use_namespace.update(namespace)
		use_namespace.update(more_namespace)
		return eval(self, globals, use_namespace)

	def __getattr__(self, item):
		return DataRef(".".join([self,item]))

	def __copy__(self):
		# str are immutable
		return self

	def __deepcopy__(self, memodict):
		return copy.deepcopy(str(self), memodict)

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

	def __new__(cls, name, default=None, fmt=None, default_value=None, format=None, scale=None):
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

	def __xml__(self, resolve_parameters=None):
		from xmle import Elem
		x = Elem('div')
		a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self))
		if resolve_parameters is not None:
			plabel = "{0:.3g}".format(resolve_parameters.pvalue(self, default_value="This is a Parameter"))
		else:
			plabel = "This is a Parameter"
		a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
		return x

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
			v = m.pvalue(str(self))
		except KeyError:
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
		except Exception as err:
			raise ValueError("invalid formating string") from err
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
			if isinstance(self, _param_multiply):
				if isinstance(self._left, ParameterRef):
					return LinearComponent(data=str(other), param=str(self._left)) * self._right
				else:
					return LinearComponent(data=str(other), param=str(self._right)) * self._left
			return LinearComponent(data=str(other), param=str(self))
		return _param_multiply(self, other)

	def __rmul__(self, other):
		if isinstance(other, (LinearComponent, LinearFunction)):
			raise TypeError("cannot do LinearComponent * ParameterRef")
		if type(other) is DataRef:
			if isinstance(self, _param_multiply):
				if isinstance(self._left, ParameterRef):
					return LinearComponent(data=str(other), param=str(self._left)) * self._right
				else:
					return LinearComponent(data=str(other), param=str(self._right)) * self._left
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

	def _is_scaled_parameter(self):
		if type(self) is ParameterRef:
			return (self, 1.0)
		return None

	@property
	def unscaled_parameter(self):
		y = self._is_scaled_parameter()
		if y is None:
			return None
		return y[0]

	@property
	def scaling_of_parameter(self):
		y = self._is_scaled_parameter()
		if y is None:
			return 1
		return y[1]

	def __le__(self, other):
		from .model.guidance import ParameterScaledOrdering
		try:
			oth = other._is_scaled_parameter()
		except AttributeError:
			raise TypeError("other is not scaled parameter")
		if oth:
			return ParameterScaledOrdering(None, *self._is_scaled_parameter(), *oth)
		else:
			raise TypeError("other is not scaled parameter")

	def __ge__(self, other):
		from .model.guidance import ParameterScaledOrdering
		try:
			oth = other._is_scaled_parameter()
		except AttributeError:
			raise TypeError("other is not scaled parameter")
		if oth:
			return ParameterScaledOrdering(None, *oth, *self._is_scaled_parameter(), snap_out = 'greater?')
		else:
			raise TypeError("other is not scaled parameter")

	def lessthan(self, other):
		s = self._is_scaled_parameter()
		if s is None:
			raise TypeError("self is not scaled parameter")
		from .model.guidance import ParameterScaledOrdering, ParameterBound
		try:
			oth = other._is_scaled_parameter()
		except AttributeError:
			if isinstance(other, Number):
				return ParameterBound(None, *self._is_scaled_parameter(), None, other)
			raise TypeError("other is not scaled parameter")
		if oth:
			return ParameterScaledOrdering(None, *self._is_scaled_parameter(), *oth)
		else:
			raise TypeError("other is not scaled parameter")

	def __lshift__(self, other):
		return self.lessthan(other)

	def greaterthan(self, other):
		s = self._is_scaled_parameter()
		if s is None:
			raise TypeError("self is not scaled parameter")
		from .model.guidance import ParameterScaledOrdering, ParameterBound
		try:
			oth = other._is_scaled_parameter()
		except AttributeError:
			if isinstance(other, Number):
				return ParameterBound(None, None, other, *self._is_scaled_parameter())
			raise TypeError("other is not scaled parameter")
		if oth:
			return ParameterScaledOrdering(None, *oth, *self._is_scaled_parameter(), snap_out = 'greater')
		else:
			raise TypeError("other is not scaled parameter")

	def __rshift__(self, other):
		return self.greaterthan(other)

	def equalto(self, other):
		s = self._is_scaled_parameter()
		if s is None:
			raise TypeError("self is not scaled parameter")
		from .model.guidance import ParameterScaledEquality
		try:
			oth = other._is_scaled_parameter()
		except AttributeError:
			raise TypeError("other is not scaled parameter")
		if oth:
			return ParameterScaledEquality(None, *self._is_scaled_parameter(), *oth)
		else:
			raise TypeError("other is not scaled parameter")

	def __matmul__(self, other):
		return self.equalto(other)

	def __rmatmul__(self, other):
		try:
			return other.equalto(self)
		except AttributeError:
			raise NotImplementedError

	def between(self, lower, upper):
		return self.greaterthan(lower), self.lessthan(upper)

	def __copy__(self):
		# str are immutable
		return self

	def __deepcopy__(self, memodict):
		return copy.deepcopy(str(self), memodict)

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

	def __getnewargs__(self):
		return (self._left, self._right)

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

	def __str__(self):
		return "{} + {}".format(repr(self._left), repr(self._right))

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

	def __str__(self):
		return "{} - {}".format(repr(self._left), repr(self._right))


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

	def __str__(self):
		return "{} * {}".format(repr(self._left), repr(self._right))

	def _is_scaled_parameter(self):
		if type(self._left) is ParameterRef and isinstance(self._right, Number):
			return (self._left, self._right)
		elif type(self._right) is ParameterRef and isinstance(self._left, Number):
			return (self._right, self._left)
		else:
			return None

	def __mul__(self, other):
		x = self._is_scaled_parameter()
		if x and isinstance(other, Number):
			return x[0] * (x[1] * other)
		return super().__mul__(other)

	def __rmul__(self, other):
		x = self._is_scaled_parameter()
		if x and isinstance(other, Number):
			return x[0] * (x[1] * other)
		return super().__mul__(other)

	def __eq__(self, other):
		if isinstance(other, _param_multiply):
			if (self._left == other._left) and (self._right == other._right):
				return True
			elif (self._left == other._right) and (self._right == other._left):
				return True
			else:
				return False
		elif isinstance(other, ParameterRef):
			if (self._left == other) and (self._right == 1):
				return True
			elif (self._left == 1) and (self._right == other):
				return True
			else:
				return False
		else:
			raise NotImplementedError(f"type of other is {type(other)}")

	def __ne__(self, other):
		return not (self==other)

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
		return "{} / {}".format(repr(self._left), repr(self._right))


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

	def __str__(self):
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
		return LinearComponent(data=str(s), param=str(s))

# +++

class LinearComponent2(tuple):
	__slots__ = []
	def __new__(cls, param, data='1', scale=1):
		return tuple.__new__(cls, (str(param), str(data), scale))

	@property
	def param(self):
		return ParameterRef(self[0])

	@property
	def data(self):
		return DataRef(self[1])

	@property
	def scale(self):
		return self[2]

	def __pos__(self):
		return self

	def __repr__(self):
		try:
			if self.scale == 1.0:
				try:
					data_is_1 = (float(self.data) == 1)
				except:
					data_is_1 = False
				if data_is_1:
					return f"({self.param!r})"
				else:
					return f"({self.param!r} * {self.data!r})"
			return f"({self.param!r} * {self.scale} * {self.data!r})"
		except AttributeError:
			return f"<{self.__class__.__name__} {id(self)} with error>"

	def __add__(self, other):
		if other == ():
			return self
		elif isinstance(other, LinearComponent2):
			return LinearFunction2([self, other])
		elif isinstance(other, LinearFunction2):
			return LinearFunction2([self, *other])
		elif other == 0:
			return self
		elif isinstance(other, ParameterRef):
			return LinearFunction2([self, LinearComponent2(param=other)])
		else:
			raise NotImplementedError()

	def __radd__(self, other):
		if other == 0:
			return self
		return other.__add__(self)

	def __mul__(self, other):
		if isinstance(other, (int, float, )):
			return self.__class__(
				param=self.param,
				data=self.data,
				scale=self.scale * other,
			)
		elif isinstance(other, (DataRef, )):
			return self.__class__(
				param=self.param,
				data=self.data * other,
				scale=self.scale,
			)
		else:
			raise TypeError(f'unsupported operand type(s) for {self.__class__.__name__}: {type(other)}')

	def __rmul__(self, other):
		if isinstance(other, (int, float, )):
			return self.__class__(
				param=self.param,
				data=self.data,
				scale=self.scale * other,
			)
		elif isinstance(other, (DataRef, )):
			return self.__class__(
				param=self.param,
				data=self.data * other,
				scale=self.scale,
			)
		else:
			raise TypeError(f'unsupported operand type(s) for {self.__class__.__name__}: {type(other)}')

	def __iter__(self):
		return iter(LinearFunction2([self]))

	def __eq__(self, other):
		if isinstance(other, LinearFunction2) and len(other) == 1:
			other = other[0]
		if not isinstance(other, LinearComponent2):
			return False
		if self.param != other.param:
			return False
		if self.data != other.data:
			return False
		if self.scale != other.scale:
			return False
		return True

	def __xml__(self, exponentiate_parameter=False, resolve_parameters=None):
		from xmle import Elem
		x = Elem('div')
		# x << tooltipped_style()

		if resolve_parameters is not None:
			if exponentiate_parameter:
				plabel = resolve_parameters.pvalue(self.param, default_value="This is a Parameter")
				if not isinstance(plabel, str):
					plabel = "exp({0:.3g})={1:.3g}".format(plabel, numpy.exp(plabel))
			else:
				plabel = "{0:.3g}".format(resolve_parameters.pvalue(self.param, default_value="This is a Parameter"))
		else:
			plabel = "This is a Parameter"


		data_tail = " * "
		try:
			if float(self.data)==1:
				data_tail = ""
		except:
			pass

		if self.scale == 1.0:
			if exponentiate_parameter:
				x.elem('span', tail="exp(")
				a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.param), tail=")"+data_tail)
				a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
			else:
				a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.param), tail=data_tail)
				a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
		else:
			if exponentiate_parameter:
				x.elem('span', tail="exp(")
				a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.param), tail=f" * {self.scale}){data_tail}")
				a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
			else:
				a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.param), tail=f" * {self.scale}{data_tail}")
				a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
		if data_tail == " * ":
			a_x = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.data))
			a_x.elem('span', attrib={'class':'tooltiptext'}, text="This is Data")
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()

	def _evaluate(self, p_getter, x_namespace=None, **kwargs):
		return self.scale * p_getter(str(self.param)) * self.data.eval(namespace=x_namespace, **kwargs)

	def __copy__(self):
		return self.__class__(
			param=self.param,
			data=self.data,
			scale=self.scale,
		)



from collections.abc import MutableSequence


def _try_mangle(instance):
	try:
		instance.mangle()
	except AttributeError as err:
		pass # print(f"No Mangle R: {err}")


class LinearFunction2(MutableSequence):

	def __init__(self, init=None):
		self.__func = list()
		self.__instance = None
		if init is not None:
			for i in init:
				if isinstance(i, LinearComponent2):
					self.__func.append(i)
				else:
					raise TypeError(f'members of {self.__class__.__name__} must be LinearComponent2')

	def set_instance(self, instance):
		self.__instance = instance

	def __get__(self, instance, owner):
		#print(f"get from descriptor object\n  instance={instance}\n  owner={owner}")
		self.__instance = instance
		return self

	def __set__(self, instance, values):
		#print(f"set in descriptor object\n  instance={instance}\n  values={values}")
		self.__init__(values)
		self.__instance = instance
		_try_mangle(self.__instance)

	def __delete__(self, instance):
		#print("deleted in descriptor object", instance)
		self.__instance = instance
		self.__func = list()
		_try_mangle(self.__instance)

	def __set_name__(self, owner, name):
		#print(f"setname in descriptor object\n  owner={owner}\n  name={name}\n  self={self}")
		self.name = name

	def __getitem__(self, item):
		return self.__func[item]

	def __setitem__(self, key, value):
		if isinstance(value, LinearComponent2):
			self.__func[key] = value
			_try_mangle(self.__instance)
		else:
			raise TypeError(f'members of {self.__class__.__name__} must be LinearComponent2')

	def __delitem__(self, key):
		del self.__func[key]
		_try_mangle(self.__instance)

	def __len__(self):
		return len(self.__func)

	def insert(self, index, value):
		if isinstance(value, LinearComponent2):
			self.__func.insert(index, value)
			_try_mangle(self.__instance)
		else:
			raise TypeError(f'members of {self.__class__.__name__} must be LinearComponent2')

	def __add__(self, other):
		if other == ():
			return self
		if isinstance(other, LinearFunction2):
			return type(self)(*self.__func, *other.__func)
		if isinstance(other, ParameterRef):
			other = LinearComponent2(param=other)
		if isinstance(other, LinearComponent2):
			result = type(self)(self)
			result.append(other)
			return result
		raise TypeError("cannot add type {} to LinearFunction".format(type(other)))

	def __iadd__(self, other):
		if isinstance(other, ParameterRef):
			other = LinearComponent2(param=other)
		if other == ():
			return self
		elif isinstance(other, LinearFunction2):
			self.__func.extend(other)
		elif isinstance(other, LinearComponent2):
			self.append(other)
		else:
			raise TypeError("cannot add type {} to LinearFunction".format(type(other)))
		return self

	def __radd__(self, other):
		return LinearFunction2([*other, *self])

	def __pos__(self):
		return self

	def __mul__(self, other):
		trial = LinearFunction2()
		for component in self:
			trial.append(component * other)
		return trial

	def __rmul__(self, other):
		trial = LinearFunction2()
		for component in self:
			trial += other * component
		return trial

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

	#######
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

		Examples
		--------
		>>> from larch.roles import P,X
		>>> f = P.InVehTime * X.IVTT + P.OutOfVehTime * X.OVTT
		>>> f1 = f.reformat_param('{}_Suffix')
		>>> str(f1)
		'P.InVehTime_Suffix * X.IVTT + P.OutOfVehTime_Suffix * X.OVTT'
		>>> f2 = f.reformat_param(pattern='(Veh)', repl='Vehicle')
		>>> str(f2)
		'P.InVehicleTime * X.IVTT + P.OutOfVehicleTime * X.OVTT'

		"""
		import re
		r = self.__class__()
		for i in self:
			if pattern is None:
				param = i.param
			else:
				if repl is None:
					raise TypeError('must give repl with pattern')
				param = re.sub(pattern, repl, i.param, **kwargs)
			if container is None:
				container = '{}'
			r += LinearComponent2(data=i.data, param=container.format(param), scale=i.scale)
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
		r = self.__class__()
		for i in self:
			if pattern is None:
				data = i.data
			else:
				if repl is None:
					raise TypeError('must give repl with pattern')
				data = re.sub(pattern, repl, i.data, **kwargs)
			if container is None:
				container = '{}'
			r += LinearComponent2(data=container.format(data), param=i.param, scale=i.scale)
		try:
			r._x_ident = self._x_ident
		except AttributeError:
			pass
		return r

	def __code__(self):
		return " + ".join(f"({repr(i)})" for i in self)

	def __eq__(self, other):
		if not isinstance(other, LinearFunction2):
			return False
		if len(self) != len(other):
			return False
		for i, j in zip(self, other):
			if i != j: return False
		return True

	# def __getstate__(self):
	# 	state = {}
	# 	state['code'] = self.__code__()
	# 	try:
	# 		state['_x_ident'] = self._x_ident
	# 	except AttributeError:
	# 		pass
	# 	return state
	#
	# def __setstate__(self, state):
	# 	self.__init__()
	# 	if state['code']:
	# 		self.__iadd__(eval(state['code']))
	# 	if '_x_ident' in state:
	# 		self._x_ident = state['_x_ident']

	def __repr__(self):
		if len(self):
			result = " + ".join(repr(i) for i in self)
			if len(result)<80:
				return result
			else:
				return "  "+result.replace(" + ","\n+ ")
		return f"<Empty {self.__class__.__name__}>"

	def __xml__(self, linebreaks=False, lineprefix="", exponentiate_parameters=False, resolve_parameters=None):
		from xmle import Elem
		x = Elem('div', attrib={'class':'LinearFunc'})
		for n,i in enumerate(self):
			ix_ = list(i.__xml__(exponentiate_parameter=exponentiate_parameters, resolve_parameters=resolve_parameters))
			if linebreaks:
				if n>0 or lineprefix:
					ix_.insert(0,Elem('br', tail = lineprefix+" + "))
			else:
				if n < len(self)-1:
					if ix_[-1].tail is None:
						ix_[-1].tail = " + "
					else:
						ix_[-1].tail += " + "
			for ii in ix_:
				x << ii
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()

	def data(self, cls=None):
		if cls is None:
			return [_.data for _ in self]
		else:
			return [cls(_.data) for _ in self]

	def _evaluate(self, p_getter, x_namespace=None, **more_x_namespace):
		if hasattr(p_getter,'pvalue') and callable(p_getter.pvalue):
			p_getter = p_getter.pvalue
		return sum(j._evaluate(p_getter, x_namespace=x_namespace, **more_x_namespace) for j in self)

	def copy(self):
		result = self.__class__(self)
		return result

	def __deepcopy__(self, memodict):
		result = self.__class__()
		for i in self:
			result.append(copy.deepcopy(i, memodict))
		return result

	def _linear_plot_2d_data(self, p_getter, x_name, x_min, x_max, n_points=100, **other_namespace):
		import numpy
		if hasattr(self, 'plotting_namespace') and len(other_namespace)==0:
			other_namespace = self.plotting_namespace
		x = numpy.linspace(x_min, x_max, n_points)
		y = self._evaluate(p_getter, {x_name:x}, **other_namespace)
		return x,y

	def linear_plot_2d(self, p_getter, x_name, x_min, x_max, n_points=100, *, xlabel=None, svg=True, header=None, **other_namespace):

		# Delayed evaluation mode...
		if p_getter is None:
			return lambda x: self.linear_plot_2d(x, x_name, x_min, x_max, n_points=n_points, xlabel=xlabel, svg=svg, header=header, **other_namespace)

		# Active evaluation mode...
		from matplotlib import pyplot as plt
		plt.clf()
		x,y = self._linear_plot_2d_data(p_getter, x_name, x_min, x_max, n_points, **other_namespace)
		if hasattr(self, 'plotting_label'):
			plt.plot(x, y, label=self.plotting_label)
		else:
			plt.plot(x, y)
		if xlabel is None:
			plt.xlabel(x_name)
		else:
			plt.xlabel(xlabel)
		plt.tight_layout(pad=0.5)
		from .util.plotting import plot_as_svg_xhtml
		if svg is True:
			svg = {}
		if svg or svg == {}:
			if header is not None:
				svg['header'] = header
			return plot_as_svg_xhtml(plt, **svg)
		else:
			plt.show()

	def _inplot_linear_plot_2d(self, plt, p_getter, x_name, x_min, x_max, n_points=100, *, xlabel=None, svg=True, header=None, **other_namespace):

		# Delayed evaluation mode...
		if p_getter is None:
			return lambda x: self._inplot_linear_plot_2d(plt, x, x_name, x_min, x_max, n_points=n_points, xlabel=xlabel, svg=svg, header=header, **other_namespace)

		# Active evaluation mode...
		x,y = self._linear_plot_2d_data(p_getter, x_name, x_min, x_max, n_points, **other_namespace)
		if hasattr(self, 'plotting_label'):
			plt.plot(x, y, label=self.plotting_label)
		else:
			plt.plot(x,y)

	def total_ordering_increasing(self):
		from toolz.itertoolz import sliding_window
		snaps = []
		for windows_size in range(2, len(self)-1):
			for sub_p in sliding_window(windows_size, self):
				snaps.append(sub_p[0].param.lessthan(sub_p[-1].param))

		return snaps


from collections.abc import MutableMapping

class DictOfLinearFunction2(MutableMapping):

	def __init__(self, mapping=None, alts_validator=None, **kwargs):
		self.__map = {}
		if mapping is None:
			mapping = {}
		for k,v in mapping.items():
			self.__map[k] = LinearFunction2(v)
		for k,v in kwargs.items():
			try:
				self.__map[k] = LinearFunction2(v)
			except:
				print(v)
				print(type(v))
				raise

		self._alts_validator = alts_validator
		self.__instance = None

	def __get__(self, instance, owner):
		#print(f"get from descriptor object\n  instance={instance}\n  owner={owner}")
		self.__instance = instance
		return self

	def __set__(self, instance, values):
		#print(f"set in descriptor object\n  instance={instance}\n  values={values}")
		self.__init__(values)
		self.__instance = instance
		_try_mangle(self.__instance)

	def __delete__(self, instance):
		#print("deleted in descriptor object", instance)
		self.__instance = instance
		self.__func = list()
		_try_mangle(self.__instance)

	def __set_name__(self, owner, name):
		#print(f"setname in descriptor object\n  owner={owner}\n  name={name}\n  self={self}")
		self.name = name


	def set_alts_validator(self, av):
		self._alts_validator = av

	def __getitem__(self, k):
		try:
			v = self.__map[k]
			v.set_instance(self.__instance)
			return v
		except KeyError:
			if self._alts_validator is None or self._alts_validator(k):
				v = self.__map[k] = LinearFunction2()
				v.set_instance(self.__instance)
				return v
			else:
				raise

	def __setitem__(self, k, v):
		if isinstance(v, ParameterRef):
			v = v * DataRef('1')
		if isinstance(v, int) and v==0:
			v = LinearFunction2()
		elif isinstance(v, LinearComponent2):
			v = LinearFunction2([v])
		elif isinstance(v, list):
			v = LinearFunction2(v)
		elif not isinstance(v, LinearFunction2):
			raise TypeError(f"only accepts LinearFunction2 values, not {type(v)}")
		if "with error" in repr(v):
			raise ValueError("found error here")
		v.set_instance(self.__instance)
		self.__map[k] = v
		_try_mangle(self.__instance)

	def __delitem__(self, key):
		del self.__map[key]
		_try_mangle(self.__instance)

	def __iter__(self):
		return iter(self.__map.keys())

	def __len__(self):
		return len(self.__map)

	def copy(self):
		return type(self)(self)

	def __repr__(self):
		return '{0}({1})'.format(type(self).__name__, repr(self.__map))

	def __xml__(self):
		from pprint import pformat
		from xmle import Elem
		x = Elem('div')
		t = x.elem('table', style="margin-top:1px;")
		t.elem('caption', text=f"<larch.{self.__class__.__name__}>", style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;font-style:normal;font-size:100%;padding:0px;")
		if len(self):
			tr = t.elem('tr')
			tr.elem('th', text="alt")
			tr.elem('th', text='formula')
			for k,v in self.items():
				tr = t.elem('tr')
				tr.elem('td', text=str(k))
				try:
					v_ = v.__xml__()
				except AttributeError:
					tr.elem('td', text=str(v))
				else:
					tr.elem('td') << v_
		else:
			tr = t.elem('tr')
			tr.elem('td', text="<empty>")
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()

# ===========================================================================

class LinearComponent0(TouchNotify):
	def __init__(self, *args, param=None, data=None, scale=None, **kwargs):
		if len(args):
			if isinstance(args[0], LinearComponent0):
				super().__init__(**kwargs)
				self._param = param or args[0]._param
				self._data = data or args[0]._data
				self._scale = scale or args[0]._scale
			else:
				raise TypeError('init LinearComponent0 with LinearComponent0 or keyword arguments')
		super().__init__(**kwargs)
		self._param = param or '1'
		self._data = data or '1'
		self._scale = scale or 1.0

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

	@property
	def scale(self):
		return self._scale

	@scale.setter
	def scale(self, value):
		self._scale = value

	def __pos__(self):
		return self

	def __add__(self, other):
		if other == ():
			return self
		elif isinstance(other, (LinearFunction, LinearComponent0)):
			return (LinearFunction() + self) + other
		elif other == 0:
			return self
		else:
			raise NotImplementedError()

	def __radd__(self, other):
		if other == 0:
			return self
		return other.__add__(self)

	def __mul__(self, other):
		if isinstance(other, (int, float, )):
			return LinearComponent0(data=self.data, param=self.param, scale=self.scale * other)
		elif isinstance(other, (DataRef, )):
			return LinearComponent0(data=self.data * other, param=self.param, scale=self.scale)
		else:
			raise TypeError('unsupported operand type(s) for LinearComponent0: {}'.format(type(other)))

	def __rmul__(self, other):
		if isinstance(other, (int, float, )):
			return LinearComponent0(data=self.data, param=self.param, scale=self.scale * other)
		elif isinstance(other, (DataRef, )):
			return LinearComponent0(data=self.data * other, param=self.param, scale=self.scale)
		else:
			raise TypeError('unsupported operand type(s) for LinearComponent0: {}'.format(type(other)))

	def __imul__(self, other):
		if isinstance(other, (int, float, )):
			self.scale = self.scale * other
		elif isinstance(other, (DataRef, )):
			self.data = self.data * other
		else:
			raise TypeError('unsupported operand type(s) for LinearComponent0: {}'.format(type(other)))

	def __iter__(self):
		return iter(LinearFunction() + self)

	def __eq__(self, other):
		if isinstance(other, LinearFunction) and len(other) == 1:
			other = other[0]
		if not isinstance(other, LinearComponent0):
			return False
		if self.param != other.param:
			return False
		if self.data != other.data:
			return False
		if self.scale != other.scale:
			return False
		return True

	def __repr__(self):
		try:
			if self.scale == 1.0:
				return f"({self.param!r} * {self.data!r})"
			return f"({self.param!r} * {self.scale} * {self.data!r})"
		except AttributeError:
			return f"<LinearComponent0 {id(self)} with error>"

	def __xml__(self, exponentiate_parameter=False, resolve_parameters=None):
		from xmle import Elem
		x = Elem('div')
		# x << tooltipped_style()

		if resolve_parameters is not None:
			if exponentiate_parameter:
				plabel = resolve_parameters.pvalue(self.param, default_value="This is a Parameter")
				if not isinstance(plabel, str):
					plabel = "exp({0:.3g})={1:.3g}".format(plabel, numpy.exp(plabel))
			else:
				plabel = "{0:.3g}".format(resolve_parameters.pvalue(self.param, default_value="This is a Parameter"))
		else:
			plabel = "This is a Parameter"

		if self.scale == 1.0:
			if exponentiate_parameter:
				x.elem('span', tail="exp(")
				a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.param), tail=") * ")
				a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
			else:
				a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.param), tail=" * ")
				a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
		else:
			if exponentiate_parameter:
				x.elem('span', tail="exp(")
				a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.param), tail=f" * {self.scale}) * ")
				a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
			else:
				a_p = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.param), tail=f" * {self.scale} * ")
				a_p.elem('span', attrib={'class':'tooltiptext'}, text=plabel)
		a_x = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.data))
		a_x.elem('span', attrib={'class':'tooltiptext'}, text="This is Data")
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()

	def _evaluate(self, p_getter, x_namespace=None, **kwargs):
		return self._scale * p_getter(str(self.param)) * self.data.eval(namespace=x_namespace, **kwargs)

	def __copy__(self):
		return LinearComponent0(
			param=self._param,
			data=self._data,
			scale=self._scale,
		)

	def __deepcopy__(self, memodict):
		return LinearComponent0(
			param=copy.deepcopy(self._param if isinstance(self._param, str) else str(self._param), memodict),
			data=copy.deepcopy(self._data if isinstance(self._data, str) else str(self._data), memodict),
			scale=copy.deepcopy(self._scale, memodict),
		)

# ===========================================================================

class LinearFunction0(TouchNotify, list):
	def __init__(self, *args, **kwargs):
		if len(args) == 1 and args[0] is None:
			args = ()
		super().__init__(*args, **kwargs)

	def __add__(self, other):
		if other == ():
			return self
		if isinstance(other, LinearFunction0):
			return type(self)(super().__add__(other))
		if isinstance(other, ParameterRef):
			other = LinearComponent0(param=other)
		if isinstance(other, LinearComponent0):
			result = type(self)(self)
			result.append(other)
			return result
		raise TypeError("cannot add type {} to LinearFunction0".format(type(other)))

	def __iadd__(self, other):
		if isinstance(other, ParameterRef):
			other = LinearComponent0(param=other)
		if other == ():
			return self
		elif isinstance(other, LinearFunction0):
			super().__iadd__(other)
		elif isinstance(other, LinearComponent0):
			super().append(other)
		else:
			raise TypeError("cannot add type {} to LinearFunction0".format(type(other)))
		return self

	def __radd__(self, other):
		return LinearFunction0() + other + self

	def __pos__(self):
		return self

	def __mul__(self, other):
		trial = LinearFunction0()
		for component in self:
			trial += component * other
		return trial

	def __rmul__(self, other):
		trial = LinearFunction0()
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
		Transform all the parameters in the LinearFunction0.

		Parameters
		----------
		container : str
			A format string, into which the previous parameters are formatted.
			Use this to append things to the parameter names.
		pattern : str
		repl : str
			Passed to `re.sub` with each existing parameter as the base string
			to be searched.

		Examples
		--------
		>>> from larch.roles import P,X
		>>> f = P.InVehTime * X.IVTT + P.OutOfVehTime * X.OVTT
		>>> f1 = f.reformat_param('{}_Suffix')
		>>> str(f1)
		'P.InVehTime_Suffix * X.IVTT + P.OutOfVehTime_Suffix * X.OVTT'
		>>> f2 = f.reformat_param(pattern='(Veh)', repl='Vehicle')
		>>> str(f2)
		'P.InVehicleTime * X.IVTT + P.OutOfVehicleTime * X.OVTT'

		"""
		import re
		r = LinearFunction0()
		for i in self:
			if pattern is None:
				param = i.param
			else:
				if repl is None:
					raise TypeError('must give repl with pattern')
				param = re.sub(pattern, repl, i.param, **kwargs)
			if container is None:
				container = '{}'
			r += LinearComponent0(data=i.data, param=container.format(param))
		try:
			r._x_ident = self._x_ident
		except AttributeError:
			pass
		return r

	def reformat_data(self, container=None, pattern=None, repl=None, **kwargs):
		"""
		Transform all the data in the LinearFunction0.

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
		r = LinearFunction0()
		for i in self:
			if pattern is None:
				data = i.data
			else:
				if repl is None:
					raise TypeError('must give repl with pattern')
				data = re.sub(pattern, repl, i.data, **kwargs)
			if container is None:
				container = '{}'
			r += LinearComponent0(data=container.format(data), param=i.param)
		try:
			r._x_ident = self._x_ident
		except AttributeError:
			pass
		return r

	def __code__(self):
		return " + ".join(f"({repr(i)})" for i in self)

	def __eq__(self, other):
		if not isinstance(other, LinearFunction0):
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
		if state['code']:
			self.__iadd__(eval(state['code']))
		if '_x_ident' in state:
			self._x_ident = state['_x_ident']

	def __repr__(self):
		result = " + ".join(repr(i) for i in self)
		if len(result)<80:
			return result
		else:
			return "  "+result.replace(" + ","\n+ ")

	def __xml__(self, linebreaks=False, lineprefix="", exponentiate_parameters=False, resolve_parameters=None):
		from xmle import Elem
		x = Elem('div', attrib={'class':'LinearFunc'})
		for n,i in enumerate(self):
			ix_ = list(i.__xml__(exponentiate_parameter=exponentiate_parameters, resolve_parameters=resolve_parameters))
			if linebreaks:
				if n>0 or lineprefix:
					ix_.insert(0,Elem('br', tail = lineprefix+" + "))
			else:
				if n < len(self)-1:
					if ix_[-1].tail is None:
						ix_[-1].tail = " + "
					else:
						ix_[-1].tail += " + "
			for ii in ix_:
				x << ii
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()

	def data(self, cls=None):
		if cls is None:
			return [_.data for _ in self]
		else:
			return [cls(_.data) for _ in self]

	def set_touch_callback(self, callback):
		super().set_touch_callback(callback)
		for i in self:
			i.set_touch_callback(callback)

	def _evaluate(self, p_getter, x_namespace=None, **more_x_namespace):
		if hasattr(p_getter,'pvalue') and callable(p_getter.pvalue):
			p_getter = p_getter.pvalue
		return sum(j._evaluate(p_getter, x_namespace=x_namespace, **more_x_namespace) for j in self)

	def copy(self):
		result = LinearFunction0() + self
		return result

	def __deepcopy__(self, memodict):
		result = LinearFunction0()
		for i in self:
			result.append(copy.deepcopy(i, memodict))
		return result

	def _linear_plot_2d_data(self, p_getter, x_name, x_min, x_max, n_points=100, **other_namespace):
		import numpy
		if hasattr(self, 'plotting_namespace') and len(other_namespace)==0:
			other_namespace = self.plotting_namespace
		x = numpy.linspace(x_min, x_max, n_points)
		y = self._evaluate(p_getter, {x_name:x}, **other_namespace)
		return x,y

	def linear_plot_2d(self, p_getter, x_name, x_min, x_max, n_points=100, *, xlabel=None, svg=True, header=None, **other_namespace):

		# Delayed evaluation mode...
		if p_getter is None:
			return lambda x: self.linear_plot_2d(x, x_name, x_min, x_max, n_points=n_points, xlabel=xlabel, svg=svg, header=header, **other_namespace)

		# Active evaluation mode...
		from matplotlib import pyplot as plt
		plt.clf()
		x,y = self._linear_plot_2d_data(p_getter, x_name, x_min, x_max, n_points, **other_namespace)
		if hasattr(self, 'plotting_label'):
			plt.plot(x, y, label=self.plotting_label)
		else:
			plt.plot(x, y)
		if xlabel is None:
			plt.xlabel(x_name)
		else:
			plt.xlabel(xlabel)
		plt.tight_layout(pad=0.5)
		from .util.plotting import plot_as_svg_xhtml
		if svg is True:
			svg = {}
		if svg or svg == {}:
			if header is not None:
				svg['header'] = header
			return plot_as_svg_xhtml(plt, **svg)
		else:
			plt.show()

	def _inplot_linear_plot_2d(self, plt, p_getter, x_name, x_min, x_max, n_points=100, *, xlabel=None, svg=True, header=None, **other_namespace):

		# Delayed evaluation mode...
		if p_getter is None:
			return lambda x: self._inplot_linear_plot_2d(plt, x, x_name, x_min, x_max, n_points=n_points, xlabel=xlabel, svg=svg, header=header, **other_namespace)

		# Active evaluation mode...
		x,y = self._linear_plot_2d_data(p_getter, x_name, x_min, x_max, n_points, **other_namespace)
		if hasattr(self, 'plotting_label'):
			plt.plot(x, y, label=self.plotting_label)
		else:
			plt.plot(x,y)

	def total_ordering_increasing(self):
		from toolz.itertoolz import sliding_window
		snaps = []
		for windows_size in range(2, len(self)-1):
			for sub_p in sliding_window(windows_size, self):
				snaps.append(sub_p[0].param.lessthan(sub_p[-1].param))

		return snaps

# ====================================================================================================
def multiple_linear_plot_2d(linear_funcs, p_getter, x_name, x_min, x_max, n_points=100, *, xlabel=None, svg=True, header=None, other_namespace=None):

	# Delayed evaluation mode...
	if p_getter is None:
		return lambda x: multiple_linear_plot_2d(linear_funcs, x, x_name, x_min, x_max, n_points=n_points, xlabel=xlabel,
													 svg=svg, header=header, other_namespace=other_namespace)

	# Active evaluation mode...
	from matplotlib import pyplot as plt
	plt.clf()

	if other_namespace is None:
		for lf in linear_funcs:
			lf._inplot_linear_plot_2d(plt, p_getter, x_name, x_min, x_max, n_points, xlabel=xlabel, svg=svg, header=header)
	else:
		for lf, oname in zip(linear_funcs, other_namespace):
			lf._inplot_linear_plot_2d(plt, p_getter, x_name, x_min, x_max, n_points, xlabel=xlabel, svg=svg, header=header, **oname)

	if xlabel is not None:
		plt.xlabel(xlabel)

	plt.legend()

	plt.tight_layout(pad=0.5)
	from .util.plotting import plot_as_svg_xhtml
	if svg is True:
		svg = {}
	if svg or svg == {}:
		if header is not None:
			svg['header'] = header
		return plot_as_svg_xhtml(plt, **svg)
	else:
		plt.show()


# ====================================================================================================

from itertools import chain
_RaiseKeyError = object() # singleton for no-default behavior

class DictOfLinearFunction0(TouchNotify, dict):
	@staticmethod # because this doesn't make sense as a global function.
	def _process_args(mapping=(), **kwargs):
		if hasattr(mapping, 'items'):
			mapping = getattr(mapping, 'items')()
		return ((k, v) for k, v in chain(mapping, getattr(kwargs, 'items')()))
	def __init__(self, mapping=(), alts_validator=None, **kwargs):
		if mapping is None:
			mapping = ()
		super().__init__(self._process_args(mapping, **kwargs))
		self._alts_validator = alts_validator
	def set_alts_validator(self, av):
		self._alts_validator = av
	def __getitem__(self, k):
		try:
			return super().__getitem__(k)
		except KeyError:
			if self._alts_validator is None or self._alts_validator(k):
				super().__setitem__(k, LinearFunction0())
				return super().__getitem__(k)
			else:
				raise
	def __setitem__(self, k, v):
		if isinstance(v, ParameterRef):
			v = v * DataRef('1')
		if isinstance(v, int) and v==0:
			v = LinearFunction0()
		elif isinstance(v, LinearComponent0):
			v = LinearFunction0() + v
		elif isinstance(v, LinearFunction0):
			pass
		elif isinstance(v, list):
			v_ = LinearFunction0()
			for vv in v:
				v_ = v_ + vv
			v = v_
		else:
			raise TypeError("only accepts LinearFunction0 values")
		if "with error" in repr(v):
			raise ValueError("found error here")
		return super().__setitem__(k, v)
	def setdefault(self, k, default=None):
		if default is None:
			default = LinearFunction0()
		if isinstance(default, LinearComponent0):
			default = LinearFunction0() + default
		elif isinstance(default, LinearFunction0):
			pass
		else:
			raise TypeError("only accepts LinearFunction0 values")
		return super().setdefault(k, default)
	def copy(self): # don't delegate w/ super - dict.copy() -> dict :(
		return type(self)(self)
	def copy_without_touch_callback(self, deep=True):
		result = type(self)()
		if deep:
			for k in self.keys():
				result[k] = list(self[k])
		else:
			for k in self.keys():
				result[k] = self[k]
		return result
	def __repr__(self):
		return '{0}({1})'.format(type(self).__name__, super().__repr__())
	def set_touch_callback(self, callback):
		super().set_touch_callback(callback)
		for i in self.values():
			i.set_touch_callback(callback)
	def __xml__(self):
		from pprint import pformat
		from xmle import Elem
		x = Elem('div')
		t = x.elem('table', style="margin-top:1px;")
		t.elem('caption', text=f"<larch.{self.__class__.__name__}>", style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;font-style:normal;font-size:100%;padding:0px;")
		if len(self):
			tr = t.elem('tr')
			tr.elem('th', text="alt")
			tr.elem('th', text='formula')
			for k,v in self.items():
				tr = t.elem('tr')
				tr.elem('td', text=str(k))
				try:
					v_ = v.__xml__()
				except AttributeError:
					tr.elem('td', text=str(v))
				else:
					tr.elem('td') << v_
		else:
			tr = t.elem('tr')
			tr.elem('td', text="<empty>")
		return x
	def _repr_html_(self):
		return self.__xml__().tostring()
#
#
#

class DictOfStrings(TouchNotify, dict):
	@staticmethod # because this doesn't make sense as a global function.
	def _process_args(mapping=(), **kwargs):
		if hasattr(mapping, 'items'):
			mapping = getattr(mapping, 'items')()
		return ((k, v) for k, v in chain(mapping, getattr(kwargs, 'items')()))
	def __init__(self, mapping=(), **kwargs):
		if mapping is None:
			mapping = ()
		super().__init__(self._process_args(mapping, **kwargs))
	def __contains__(self, k):
		return super().__contains__(str(k))
	def __getitem__(self, k):
		return super().__getitem__(str(k))
	def __setitem__(self, k, v):
		return super().__setitem__(str(k), str(v))
	def setdefault(self, k, default=""):
		return super().setdefault(str(k), str(default))
	def copy(self): # don't delegate w/ super - dict.copy() -> dict :(
		return type(self)(self)
	def __repr__(self):
		return '{0}({1})'.format(type(self).__name__, super().__repr__())
	def __xml__(self):
		from pprint import pformat
		from xmle import Elem
		x = Elem('div')
		t = x.elem('table', style="margin-top:1px;")
		t.elem('caption', text=f"<larch.{self.__class__.__name__}>", style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;font-style:normal;font-size:100%;padding:0px;")
		if len(self):
			tr = t.elem('tr')
			tr.elem('th', text="key")
			tr.elem('th', text='value')
			for k,v in self.items():
				tr = t.elem('tr')
				tr.elem('td', text=str(k))
				tr.elem('td', text=str(v))
		else:
			tr = t.elem('tr')
			tr.elem('td', text="<empty>")
		return x
	def _repr_html_(self):
		return self.__xml__().tostring()

class DictOfStringKeys(TouchNotify, dict):
	@staticmethod # because this doesn't make sense as a global function.
	def _process_args(mapping=(), **kwargs):
		if hasattr(mapping, 'items'):
			mapping = getattr(mapping, 'items')()
		return ((k, v) for k, v in chain(mapping, getattr(kwargs, 'items')()))
	def __init__(self, mapping=(), **kwargs):
		if mapping is None:
			mapping = ()
		super().__init__(self._process_args(mapping, **kwargs))
	def __contains__(self, k):
		return super().__contains__(str(k))
	def __getitem__(self, k):
		return super().__getitem__(str(k))
	def __setitem__(self, k, v):
		return super().__setitem__(str(k), v)
	def setdefault(self, k, default=""):
		return super().setdefault(str(k), default)
	def copy(self): # don't delegate w/ super - dict.copy() -> dict :(
		return type(self)(self)
	def __repr__(self):
		return '{0}({1})'.format(type(self).__name__, super().__repr__())
	def __xml__(self):
		from pprint import pformat
		from xmle import Elem
		x = Elem('div')
		t = x.elem('table', style="margin-top:1px;")
		t.elem('caption', text=f"<larch.{self.__class__.__name__}>", style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;font-style:normal;font-size:100%;padding:0px;")
		if len(self):
			tr = t.elem('tr')
			tr.elem('th', text="key")
			tr.elem('th', text='value')
			for k,v in self.items():
				tr = t.elem('tr')
				tr.elem('td', text=str(k))
				tr.elem('td', text=str(v))
		else:
			tr = t.elem('tr')
			tr.elem('td', text="<empty>")
		return x
	def _repr_html_(self):
		return self.__xml__().tostring()



from .model.linear import ParameterRef_C, DataRef_C, LinearFunction_C, \
	LinearComponent_C, DictOfLinearFunction_C, Ref_Gen

LinearFunction = LinearFunction_C
LinearComponent = LinearComponent_C
DictOfLinearFunction = DictOfLinearFunction_C

P = Ref_Gen(ParameterRef_C)
X = Ref_Gen(DataRef_C)

def PX(z):
	return P(z) * X(z)

globals()[_ParameterRef_repr_txt] = P
globals()[_DataRef_repr_txt] = X
globals()[_ParameterRef_repr_txt + _DataRef_repr_txt] = PX

__all__ = [_ParameterRef_repr_txt, _DataRef_repr_txt, _ParameterRef_repr_txt + _DataRef_repr_txt]
