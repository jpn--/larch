# cython: language_level=3, embedsignature=True

import re as _re
import keyword as _keyword
import numpy as _numpy
import sys
import re
from numbers import Number as _Number

from ..util.naming import parenthize, valid_identifier_or_parenthized_string


_ParameterRef_C_repr_txt = "P"
_DataRef_repr_txt = 'X'

_null_ = '_'

_boolmatch = re.compile("^boolean\((.+)==(.+)\)$")


def _what_is(thing):
	if isinstance(thing, (ParameterRef_C, DataRef_C)):
		return repr(thing)
	if isinstance(thing, (str, int, float)):
		return f"{thing.__class__.__name__}({thing})"
	if isinstance(thing, LinearComponent_C):
		return f"{thing.__class__.__name__}({thing!r})"
	return f"<{thing.__class__.__name__}>"

def _unsupported_operands(op,a,b):
	return f"unsupported operands for {op}: '{_what_is(a)}' and '{_what_is(b)}'"

cdef class UnicodeRef_C(unicode):
	"""
	A common base class for all larch named reference types.

	This class itself has no features and should not be instantiated.
	Instead create :class:`ParameterRef_C` or :class:`DataRef_C` objects as needed.
	"""

cdef class Ref_Gen:

	def __init__(self, kind):
		self._kind = kind

	def __getattr__(self, key):
		return self._kind(key)

	def __call__(self, arg):
		return self._kind(str(arg))

	def __getitem__(self, arg):
		return self._kind(str(arg))


P = Ref_Gen(ParameterRef_C)
X = Ref_Gen(DataRef_C)

cdef class ParameterRef_C(UnicodeRef_C):

	_precedence = 99

	def __init__(self, *args):
		self._formatting = None

	def set_fmt(self, str formatting):
		self._formatting = formatting
		return self

	def __repr__(self):
		if _re.match("[_A-Za-z][_a-zA-Z0-9]*$", self) and not _keyword.iskeyword(self):
			return "{}.{}".format(_ParameterRef_C_repr_txt, self)
		else:
			return "{}('{}')".format(_ParameterRef_C_repr_txt, self)

	def __eq__(self, other):
		if isinstance(other, str) and not isinstance(other, DataRef_C):
			if str(self) == str(other):
				return True
		return False

	def __ne__(self, other):
		return not self.__eq__(other)

	def __hash__(self):
		return super().__hash__()

	def __pos__(self):
		return self

	def __add__(self, other):
		if isinstance(self, ParameterRef_C):
			if other == 0:
				return self
			if isinstance(other, (ParameterRef_C, LinearComponent_C, LinearFunction_C)):
				return LinearComponent_C(param=str(self), data="1") + other
			if isinstance(other, _Number):
				from .linear_math import ParameterAdd
				return ParameterAdd(self, other)
		elif isinstance(other, ParameterRef_C):
			if self == 0:
				return other
			if isinstance(self, (LinearComponent_C, LinearFunction_C)):
				return self + LinearComponent_C(param=str(other), data="1")
			if isinstance(self, _Number):
				from .linear_math import ParameterAdd
				return ParameterAdd(self, other)
		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} + {_what_is(other)}")

	def __sub__(self, other):
		if isinstance(self, ParameterRef_C):
			if other == 0:
				return self
			if isinstance(other, (ParameterRef_C, LinearComponent_C, LinearFunction_C)):
				return LinearComponent_C(param=str(self), data="1") - other
			if isinstance(other, _Number):
				from .linear_math import ParameterSubtract
				return ParameterSubtract(self, other)
		elif isinstance(other, ParameterRef_C):
			if self == 0:
				return other
			if isinstance(self, (LinearComponent_C, LinearFunction_C)):
				return self - LinearComponent_C(param=str(other), data="1")
			if isinstance(self, _Number):
				from .linear_math import ParameterSubtract
				return ParameterSubtract(self, other)
		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} - {_what_is(other)}")

	def __mul__(self, other):
		if isinstance(self, ParameterRef_C):
			if isinstance(other, DataRef_C):
				return LinearComponent_C(param=str(self), data=str(other))
			if isinstance(other, _Number):
				#  return LinearComponent_C(param=str(self), data=str(other))
				return LinearComponent_C(param=str(self), data=str("1"), scale=other)
			if isinstance(other, ParameterRef_C):
				if self == _null_:
					return other
				if other == _null_:
					return self
				from .linear_math import ParameterMultiply
				return ParameterMultiply(self, other)
			if isinstance(other, LinearComponent_C):
				if self == _null_:
					return other
				if other.param == _null_:
					return LinearComponent_C(param=str(self), data=str(other.data), scale=other.scale)
				from .linear_math import ParameterMultiply
				return ParameterMultiply(self, other)
		elif isinstance(other, ParameterRef_C):
			if isinstance(self, DataRef_C):
				return LinearComponent_C(param=str(other), data=str(self))
			if isinstance(self, _Number):
				# return LinearComponent_C(param=str(other), data=str(self))
				return LinearComponent_C(param=str(other), data=str("1"), scale=self)
		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} * {_what_is(other)}")

	def __truediv__(self, other):
		if isinstance(self, ParameterRef_C):
			if isinstance(other, ParameterRef_C):
				from .linear_math import ParameterDivide
				return ParameterDivide(self, other)
			elif isinstance(other, _Number):
				return LinearComponent_C(param=str(self), data=str("1"), scale=1/other)
			elif isinstance(other, DataRef_C):
				return LinearComponent_C(param=str(self), data=str(1/other), scale=1)
		elif isinstance(other, ParameterRef_C):
			return NotImplemented
		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} / {_what_is(other)}")

	def value(self, *args):
		"""
		The value of the parameter in a given model.

		Parameters
		----------
		m : Model
			The model from which to extract a parameter value.

		Returns
		-------
		float
		"""
		s = str(self)
		for m in args:
			if isinstance(m, dict) and s in m:
				return m[s]
			else:
				try:
					return m.get_value(s)
				except KeyError:
					pass
		raise KeyError(s)

	def string(self, m):
		"""
		The value of the parameter in a given model, as a formatted string.

		Parameters
		----------
		m : Model
			The model from which to extract a parameter value.

		Returns
		-------
		str
		"""
		if self._formatting is None:
			return "{:.3g}".format(self.value(m))
		else:
			return self._formatting.format(self.value(m))

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
		return False

	def as_pmath(self):
		from .linear_math import ParameterNoop
		return ParameterNoop(self)

	def __xml__(self, resolve_parameters=None, value_in_tooltips=True):

		if resolve_parameters is not None:
			if value_in_tooltips:
				p_display = repr(self)
				p_tooltip = self.string(resolve_parameters)
			else:
				p_display = self.string(resolve_parameters)
				p_tooltip = repr(self)
		else:
			p_display = repr(self)
			p_tooltip = "This is a Parameter"

		from xmle import Elem
		x = Elem('div')
		if use_tooltips:
			a_p = x.elem('div', attrib={'class':'tooltipped'}, text=p_display)
			a_p.elem('span', attrib={'class':'tooltiptext'}, text=p_tooltip)
		else:
			a_p = x.elem('span', attrib={'class':'Larch_Parameter'}, text=p_display)
		return x

cdef class DataRef_C(UnicodeRef_C):

	def __repr__(self):
		if _re.match("[_A-Za-z][_a-zA-Z0-9]*$", self) and not _keyword.iskeyword(self):
			return "{}.{}".format(_DataRef_repr_txt, self)
		else:
			return "{}('{}')".format(_DataRef_repr_txt, self)

	def __eq__(self, other):
		if isinstance(other, str) and not isinstance(other, ParameterRef_C):
			if str(self) == str(other):
				return True
		return False

	def __ne__(self, other):
		return not self.__eq__(other)

	def __hash__(self):
		return super().__hash__()

	def __pos__(self):
		return self

	def __add__(self, other):
		if isinstance(self, (DataRef_C, _Number)) and isinstance(other, (DataRef_C, _Number)):
			if self == "0" or self == "0.0" or self == 0:
				return other
			if other == "0" or other == "0.0" or other == 0:
				return self
			# Double zero is trapped here as it is used to flag duplicate terms in a utility function.
			if self == "00":
				return DataRef_C("0+{}".format(parenthize(self), parenthize(other, True)))
			if other == "00":
				return DataRef_C("{}+0".format(parenthize(self), parenthize(other, True)))
			return DataRef_C("{}+{}".format(parenthize(self), parenthize(other, True)))
		if isinstance(self, DataRef_C) and isinstance(other, ParameterRef_C):
			return P(_null_)*self + other

		# Don't return NotImplemented just raise TypeError when adding a DataRef_C and a plain string.
		# This will disallow the __radd__ method on the plain string.
		if isinstance(self, (DataRef_C, _Number)) and isinstance(other, str) and not isinstance(other, DataRef_C):
			raise TypeError(_unsupported_operands('-', self, other))
		if isinstance(other, (DataRef_C, _Number)) and isinstance(self, str) and not isinstance(self, DataRef_C):
			raise TypeError(_unsupported_operands('-', self, other))

		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} + {_what_is(other)}")

	def __sub__(self, other):
		if isinstance(self, (DataRef_C, _Number)) and isinstance(other, (DataRef_C, _Number)):
			return DataRef_C("{}-{}".format(parenthize(self), parenthize(other, True)))
		if isinstance(self, DataRef_C) and isinstance(other, ParameterRef_C):
			return P(_null_)*self - other
		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} - {_what_is(other)}")

	def __mul__(self, other):
		if isinstance(self, DataRef_C):
			if isinstance(other, (DataRef_C, _Number)):
				if self == "1" or self == "1.0" or self == 1:
					return other
				if other == "1" or other == "1.0" or other == 1:
					return self
				if isinstance(other, _Number):
					return P(_null_) * other * self
				if self == other and self[:8] == 'boolean(' and self[-1:]==')':
					# Squaring a boolean does not change it
					return self
				if self[:8] == 'boolean(' and self[-1:]==')' and other[:8] == 'boolean(' and other[-1:]==')':
					# Check for two mutually exclusive conditions
					boolmatch1 = _boolmatch.match(self)
					if boolmatch1:
						boolmatch2 = _boolmatch.match(other)
						if boolmatch2:
							if boolmatch1.group(1) == boolmatch2.group(1):
								if boolmatch1.group(2) != boolmatch2.group(2):
									return DataRef_C('0')
				return DataRef_C("{}*{}".format(parenthize(self), parenthize(other, True)))
			if isinstance(other, ParameterRef_C):
				return LinearComponent_C(param=str(other), data=str(self))
			if isinstance(other, LinearComponent_C):
				return LinearComponent_C(param=str(other.param), data=str(self * other.data), scale=other.scale )
			if isinstance(other, LinearFunction_C):
				return LinearFunction_C([self * c for c in other] )
		elif isinstance(other, DataRef_C):
			if isinstance(self, (DataRef_C, _Number)):
				if self == "1" or self == "1.0" or self == 1:
					return other
				if other == "1" or other == "1.0" or other == 1:
					return self
				if isinstance(self, _Number):
					return P(_null_) * self * other
				return DataRef_C("{}*{}".format(parenthize(self), parenthize(other, True)))
			if isinstance(self, ParameterRef_C):
				return LinearComponent_C(param=str(self), data=str(other))
			if isinstance(self, LinearComponent_C):
				return LinearComponent_C(param=str(self.param), data=str(self.data * other) )
			if isinstance(self, LinearFunction_C):
				return LinearFunction_C([c*other for c in self] )
		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} * {_what_is(other)}")

	def __truediv__(self, other):
		if isinstance(self, (DataRef_C, _Number)) and isinstance(other, (DataRef_C, _Number)):
			return DataRef_C("{}/{}".format(parenthize(self), parenthize(other, True)))
		if isinstance(self, ParameterRef_C) and isinstance(other, (DataRef_C, _Number)):
			return self * DataRef_C("1/{}".format(parenthize(other, True)))
		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} / {_what_is(other)}")

	def __and__(self, other):
		if isinstance(self, (DataRef_C, _Number)) and isinstance(other, (DataRef_C, _Number)):
			return DataRef_C("{}&{}".format(parenthize(self), parenthize(other, True)))
		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} & {_what_is(other)}")

	def __or__(self, other):
		if isinstance(self, (DataRef_C, _Number)) and isinstance(other, (DataRef_C, _Number)):
			return DataRef_C("{}|{}".format(parenthize(self), parenthize(other, True)))
		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} | {_what_is(other)}")

	def __xor__(self, other):
		if isinstance(self, (DataRef_C, _Number)) and isinstance(other, (DataRef_C, _Number)):
			return DataRef_C("{}^{}".format(parenthize(self), parenthize(other, True)))
		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} ^ {_what_is(other)}")

	def __floordiv__(self, other):
		if isinstance(self, (DataRef_C, _Number)) and isinstance(other, (DataRef_C, _Number)):
			return DataRef_C("{}//{}".format(parenthize(self),parenthize(other, True)))
		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} // {_what_is(other)}")

	def __pow__(self, other, modulo):
		if modulo is not None:
			raise NotImplementedError(f"no pow with modulo on {self.__class__.__name__}")
		if isinstance(self, (DataRef_C, _Number)) and isinstance(other, (DataRef_C, _Number)):
			return DataRef_C("{}**{}".format(parenthize(self),parenthize(other, True)))
		return NotImplemented # raise NotImplementedError(f"{_what_is(self)} ** {_what_is(other)}")

	def __invert__(self):
		return DataRef_C("~{}".format(parenthize(self, True)))

	def __neg__(self):
		return DataRef_C("-{}".format(parenthize(self, True)))

	def __pos__(self):
		return self

	def eval(self, namespace=None, *, globals=None, **more_namespace):
		import numpy
		from ..util.common_functions import piece
		use_namespace = {'exp': numpy.exp, 'log': numpy.log, 'log1p': numpy.log1p, 'fabs': numpy.fabs,
		                 'sqrt': numpy.sqrt,
		                 'absolute': numpy.absolute, 'isnan': numpy.isnan, 'isfinite': numpy.isfinite,
		                 'logaddexp': numpy.logaddexp, 'fmin': numpy.fmin, 'fmax': numpy.fmax,
		                 'nan_to_num': numpy.nan_to_num,
						 'piece': piece,}
		if namespace is not None:
			use_namespace.update(namespace)
		use_namespace.update(more_namespace)
		return eval(self, globals, use_namespace)


use_tooltips = False

cdef class LinearComponent_C:

	def __init__(self, unicode param, unicode data='1', l4_float_t scale=1):
		self._param = param
		self._data = data
		self._scale = scale

	@property
	def param(self):
		return ParameterRef_C(self._param)

	@property
	def data(self):
		return DataRef_C(self._data)

	@property
	def scale(self):
		return self._scale

	def __pos__(self):
		return self

	def __neg__(self):
		return self.__class__(param=self._param, data=self._data, scale=-self._scale)

	def __repr__(self):
		try:
			if self.scale == 1.0:
				try:
					data_is_1 = (float(self.data) == 1)
				except:
					data_is_1 = False
				if data_is_1:
					return f"{self.param!r}"
				else:
					return f"{self.param!r} * {self.data!r}"
			return f"{self.param!r} * {self.scale} * {self.data!r}"
		except AttributeError:
			return f"<{self.__class__.__name__} {id(self)} with error>"

	def _str_exponentiate(self):
		try:
			if self.scale == 1.0:
				try:
					data_is_1 = (float(self.data) == 1)
				except:
					data_is_1 = False
				if data_is_1:
					return f"exp({self.param!r})"
				else:
					return f"exp({self.param!r}) * {self.data!r}"
			return f"exp({self.param!r}) * {self.scale} * {self.data!r}"
		except AttributeError:
			return f"<{self.__class__.__name__} {id(self)} with error>"

	def __add__(self, other):
		if isinstance(self, LinearComponent_C):
			if other == () or other == 0:
				return self
			elif isinstance(other, LinearComponent_C):
				if other.data == '0' or other.data == '0.0' or other.scale == 0:
					return self
				if self.data == '0' or self.data == '0.0' or self.scale == 0:
					return other
				return LinearFunction_C([self, other])
			elif isinstance(other, LinearFunction_C):
				return LinearFunction_C([self, *other])
			elif isinstance(other, ParameterRef_C):
				return self + LinearComponent_C(param=str(other))
			elif isinstance(other, DataRef_C):
				return self + LinearComponent_C(param=_null_, data=str(other))
			else:
				try:
					return self.as_pmath() + other
				except NotImplementedError:
					pass
		elif isinstance(other, LinearComponent_C):
			if self == () or self == 0:
				return other
			elif isinstance(self, ParameterRef_C):
				#return LinearFunction_C([LinearComponent_C(param=str(self)), other])
				return LinearComponent_C(param=str(self)) + other
			elif isinstance(self, DataRef_C):
				return LinearComponent_C(param=_null_, data=str(self)) + other
			elif isinstance(self, LinearFunction_C):
				return LinearFunction_C([*self, ]) + other
			else:
				try:
					return other + self.as_pmath()
				except NotImplementedError:
					pass
		raise NotImplementedError(f"{_what_is(self)} + {_what_is(other)}")

	def __sub__(self, other):
		if isinstance(self, LinearComponent_C):
			if other == () or other == 0:
				return self
			elif isinstance(other, LinearComponent_C):
				return LinearFunction_C([self, -other])
			elif isinstance(other, LinearFunction_C):
				return LinearFunction_C([self, *(-other)])
			elif isinstance(other, ParameterRef_C):
				return LinearFunction_C([self, -LinearComponent_C(param=str(other))])
			elif isinstance(other, ParameterRef_C):
				return LinearFunction_C([self, -LinearComponent_C(param=_null_, data=str(other))])
			else:
				try:
					return self.as_pmath() - other
				except NotImplementedError:
					pass
		raise NotImplementedError(f"{_what_is(self)} + {_what_is(other)}")

	def __mul__(self, other):
		if isinstance(self, LinearComponent_C):
			if isinstance(other, (int, float, )):
				return self.__class__(
					param=str(self.param),
					data=str(self.data),
					scale=self.scale * other,
				)
			if isinstance(other, (DataRef_C, )):
				return self.__class__(
					param=str(self.param),
					data=str(self.data * other),
					scale=self.scale,
				)
			if isinstance(other, (LinearComponent_C, )):
				if self.param == _null_:
					return self.__class__(
						param=str(other.param),
						data=str(self.data * other.data),
						scale=self.scale * other.scale,
					)
				if other.param == _null_:
					return self.__class__(
						param=str(self.param),
						data=str(self.data * other.data),
						scale=self.scale * other.scale,
					)
				from .linear_math import ParameterMultiply
				return ParameterMultiply(
					self.as_pmath(),
					other.as_pmath(),
				)
			if isinstance(other, (ParameterRef_C, )):
				if other == _null_:
					return self
				elif self.param == _null_:
					return self.__class__(
						param=str(other),
						data=str(self.data),
						scale=self.scale,
					)
			try:
				return self.as_pmath() * other
			except NotImplementedError:
				pass
		if isinstance(other, LinearComponent_C) and isinstance(self, _Number):
			return other.__class__(
				param=str(other.param),
				data=str(other.data),
				scale=other.scale * self,
			)
		raise NotImplementedError(f"{_what_is(self)} * {_what_is(other)}")

	def __truediv__(self, other):
		if isinstance(self, LinearComponent_C):
			if isinstance(other, (int, float, )):
				return self.__class__(
					param=str(self.param),
					data=str(self.data),
					scale=self.scale / other,
				)
			elif isinstance(other, (DataRef_C, )):
				return self.__class__(
					param=str(self.param),
					data=str(self.data / other),
					scale=self.scale,
				)
			elif isinstance(other, (LinearComponent_C, )):
				from .linear_math import ParameterDivide
				return ParameterDivide(
					self.as_pmath(),
					other.as_pmath(),
				)
			else:
				try:
					return self.as_pmath() / other
				except NotImplementedError:
					pass
		raise NotImplementedError(f"{_what_is(self)} / {_what_is(other)}")

	def __iter__(self):
		return iter(LinearFunction_C([self]))

	def __eq__(self, other):
		if isinstance(other, LinearFunction_C) and len(other) == 1:
			other = other[0]
		if not isinstance(other, LinearComponent_C):
			return False
		if self.param != other.param:
			return False
		if self.data != other.data:
			return False
		if self.scale != other.scale:
			return False
		return True

	def __xml__(self, exponentiate_parameter=False, resolve_parameters=None, value_in_tooltips=True):
		from xmle import Elem

		if use_tooltips:
			x = Elem('div')
			# x << tooltipped_style()
			if resolve_parameters is not None:
				if exponentiate_parameter:
					if value_in_tooltips:
						p_tooltip = f"exp({self.param.string(resolve_parameters)}) = {self.param.value(resolve_parameters):.4g}"
						p_display = f"{repr(self.param)}"
					else:
						p_display = f"{self.param.string(resolve_parameters)}"
						p_tooltip = f"exp({repr(self.param)})"
				else:
					if value_in_tooltips:
						p_tooltip = self.param.string(resolve_parameters)
						p_display = repr(self.param)
					else:
						p_display = self.param.string(resolve_parameters)
						p_tooltip = repr(self.param)
			else:
				p_display = repr(self.param)
				p_tooltip = "This is a Parameter"



			data_tail = " * "
			try:
				if float(self.data)==1:
					data_tail = ""
			except:
				pass

			if self.scale == 1.0:
				if exponentiate_parameter:
					x.elem('span', tail="exp(")
					a_p = x.elem('div', attrib={'class':'tooltipped'}, text=p_display, tail=")"+data_tail)
					a_p.elem('span', attrib={'class':'tooltiptext'}, text=p_tooltip)
				else:
					a_p = x.elem('div', attrib={'class':'tooltipped'}, text=p_display, tail=data_tail)
					a_p.elem('span', attrib={'class':'tooltiptext'}, text=p_tooltip)
			else:
				if exponentiate_parameter:
					x.elem('span', tail="exp(")
					a_p = x.elem('div', attrib={'class':'tooltipped'}, text=p_display, tail=f" * {self.scale}){data_tail}")
					a_p.elem('span', attrib={'class':'tooltiptext'}, text=p_tooltip)
				else:
					a_p = x.elem('div', attrib={'class':'tooltipped'}, text=p_display, tail=f" * {self.scale}{data_tail}")
					a_p.elem('span', attrib={'class':'tooltiptext'}, text=p_tooltip)
			if data_tail == " * ":
				a_x = x.elem('div', attrib={'class':'tooltipped'}, text=repr(self.data))
				a_x.elem('span', attrib={'class':'tooltiptext'}, text="This is Data")

		else:
			x = Elem('pre')
			# x << tooltipped_style()
			if resolve_parameters is not None:
				if exponentiate_parameter:
					if value_in_tooltips:
						p_display = f"{repr(self.param)}"
					else:
						p_display = f"{self.param.string(resolve_parameters)}"
				else:
					if value_in_tooltips:
						p_display = repr(self.param)
					else:
						p_display = self.param.string(resolve_parameters)
			else:
				p_display = repr(self.param)

			data_tail = " * "
			try:
				if float(self.data)==1:
					data_tail = ""
			except:
				pass

			if self.scale == 1.0:
				if exponentiate_parameter:
					x.elem('span', tail="exp(")
					a_p = x.elem('span', attrib={'class':'LinearFunc_Param'}, text=p_display, tail=")"+data_tail)
				else:
					a_p = x.elem('span', attrib={'class':'LinearFunc_Param'}, text=p_display, tail=data_tail)
			else:
				if exponentiate_parameter:
					x.elem('span', tail="exp(")
					a_p = x.elem('span', attrib={'class':'LinearFunc_Param'}, text=p_display, tail=f" * {self.scale}){data_tail}")
				else:
					a_p = x.elem('span', attrib={'class':'LinearFunc_Param'}, text=p_display, tail=f" * {self.scale}{data_tail}")
			if data_tail == " * ":
				a_x = x.elem('span', attrib={'class':'LinearFunc_Data'}, text=repr(self.data))
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()

	def evaluate(self, p_getter, x_namespace=None, **kwargs):
		return self.scale * p_getter(str(self.param)) * self.data.eval(namespace=x_namespace, **kwargs)

	def __copy__(self):
		return self.__class__(
			param=str(self.param),
			data=str(self._data),
			scale=self.scale,
		)

	def as_pmath(self):
		from .linear_math import ParameterNoop, ParameterMultiply
		if self.data != "1":
			try:
				scale = float(self.data) * self.scale
			except:
				pass
			else:
				if scale == 1:
					return ParameterNoop(self.param)
				else:
					return ParameterMultiply(self.param, scale)
			raise NotImplementedError('data is not 1')
		if self.scale == 1:
			return ParameterNoop(self.param)
		else:
			return ParameterMultiply(self.param, self.scale)

def _try_mangle(instance):
	try:
		instance.mangle()
	except AttributeError as err:
		pass # print(f"No Mangle L: {err}")

def _try_mangle_h(instance_holder):
	try:
		instance_holder._instance.mangle()
	except AttributeError as err:
		pass # print(f"No Mangle L2: {err}")


cdef class LinearFunction_C:

	def __init__(self, init=None):
		self._func = list()
		#self._instance = None
		if init is not None:
			for i in init:
				if isinstance(i, LinearComponent_C):
					self._func.append(i)
				else:
					raise TypeError(f'members of {self.__class__.__name__} must be LinearComponent_C')

	def set_instance(self, instance):
		self._instance = instance

	def __fresh(self, instance):
		newself = LinearFunction_C()
		newself._instance = instance
		setattr(instance, self.private_name, newself)
		return newself

	def __get__(self, instance, owner):

		cdef LinearFunction_C newself

		if instance is None:
			return self
		try:
			newself = getattr(instance, self.private_name)
		except AttributeError:
			newself = self.__fresh(instance)
		if newself is None:
			newself = self.__fresh(instance)
		return newself

	def __set__(self, instance, values):

		cdef LinearFunction_C newself

		try:
			newself = getattr(instance, self.private_name)
		except AttributeError:
			newself = self.__fresh(instance)
		if newself is None:
			newself = self.__fresh(instance)
		newself.__init__(values)
		#_try_mangle_h(newself)
		try:
			newself._instance.mangle()
		except AttributeError as err:
			pass # print(f"No Mangle L2: {err}")
		else:
			pass # print(f"Yes Mangle L2: {newself._instance}")

	def __delete__(self, instance):

		cdef LinearFunction_C newself

		try:
			newself = getattr(instance, self.private_name)
		except AttributeError:
			newself = self.__fresh(instance)
		newself.__init__()
		newself._instance = instance
		#_try_mangle_h(newself)
		try:
			newself._instance.mangle()
		except AttributeError as err:
			pass # print(f"No Mangle L2: {err}")
		else:
			pass # print(f"Yes Mangle L2: {newself._instance}")


	def __set_name__(self, owner, name):
		self.name = name
		self.private_name = "_"+name

	def __getitem__(self, item):
		return self._func[item]

	def __setitem__(self, key, LinearComponent_C value not None):
		self._func[key] = value
		_try_mangle(self._instance)

	def __delitem__(self, key):
		del self._func[key]
		_try_mangle(self._instance)

	def remove_data(self, data):
		i = len(self._func)
		while i > 0:
			i -= 1
			if self._func[i].data == data:
				del self._func[i]

	def remove_param(self, param):
		i = len(self._func)
		while i > 0:
			i -= 1
			if self._func[i].param == param:
				del self._func[i]

	def __len__(self):
		return len(self._func)

	def insert(self, index, LinearComponent_C value not None):
		self._func.insert(index, value)
		_try_mangle(self._instance)

	def append(self, LinearComponent_C value not None):
		self._func.append(value)
		_try_mangle(self._instance)

	def extend(self, values):
		for v in values:
			if not isinstance(v, LinearComponent_C):
				raise TypeError("cannot add type {} to LinearFunction".format(type(v)))
		self._func.extend(values)
		_try_mangle(self._instance)

	def __add__(self, other):
		if isinstance(self, LinearFunction_C):
			if other == ():
				return self
			if isinstance(other, LinearFunction_C):
				return self.__class__([*list(self), *list(other)])
			if isinstance(other, ParameterRef_C):
				other = LinearComponent_C(param=str(other))
			if isinstance(other, LinearComponent_C):
				result = self.__class__(self)
				if not(other.data == '0' or other.data == '0.0' or other.scale==0):
					result.append(other)
				return result
			from .linear_math import _ParameterOp, ParameterAdd
			if isinstance(other, (_ParameterOp, _Number)):
				return ParameterAdd(self.as_pmath(), other)
		if isinstance(other, LinearFunction_C):
			from .linear_math import _ParameterOp, ParameterAdd
			if isinstance(self, (_ParameterOp, _Number)):
				return ParameterAdd(self, other.as_pmath())
		raise NotImplementedError(f"{_what_is(self)} + {_what_is(other)}")

	def __iadd__(self, other):
		if isinstance(other, ParameterRef_C):
			other = LinearComponent_C(param=str(other))
		if other == ():
			return self
		elif isinstance(other, LinearFunction_C):
			self._func.extend(other)
			_try_mangle(self._instance)
		elif isinstance(other, LinearComponent_C):
			self.append(other)
		else:
			raise TypeError("cannot add type {} to LinearFunction".format(type(other)))
		return self

	def __pos__(self):
		return self

	def __neg__(self):
		return self.__class__(-i for i in self)

	def __sub__(self, other):
		if isinstance(self, LinearFunction_C):
			if other == ():
				return self
			if isinstance(other, LinearFunction_C):
				return self.__class__([*list(self), *list(-other)])
			if isinstance(other, ParameterRef_C):
				other = LinearComponent_C(param=str(other))
			if isinstance(other, LinearComponent_C):
				result = self.__class__(self)
				result.append(-other)
				return result
			from .linear_math import _ParameterOp, ParameterSubtract
			if isinstance(other, (_ParameterOp, _Number)):
				return ParameterSubtract(self.as_pmath(), other)
		if isinstance(other, LinearFunction_C):
			from .linear_math import _ParameterOp, ParameterSubtract
			if isinstance(self, (_ParameterOp, _Number)):
				return ParameterSubtract(self, other.as_pmath())
		raise NotImplementedError(f"{_what_is(self)} + {_what_is(other)}")

	def __mul__(self, other):
		from .linear_math import _ParameterOp, ParameterMultiply
		if isinstance(self, LinearFunction_C) and isinstance(other, _Number):
			trial = LinearFunction_C()
			for component in self:
				trial.append(component * other)
			return trial
		if isinstance(self, LinearFunction_C) and isinstance(other, (ParameterRef_C, _ParameterOp)):
			if isinstance(other, ParameterRef_C):
				if other == _null_:
					return self
				trial = LinearFunction_C()
				for component in self:
					if component.param == _null_:
						trial.append(component * other)
					else:
						trial = None
						break
				if trial is not None:
					return trial
			return ParameterMultiply(self.as_pmath(), other)
		if isinstance(self, LinearFunction_C) and isinstance(other, LinearFunction_C):
			return sum(
				i*j
				for i in self
				for j in other
			)
		if isinstance(other, LinearFunction_C) and isinstance(self, _Number):
			trial = LinearFunction_C()
			for component in other:
				trial.append(self * component)
			return trial
		if isinstance(other, LinearFunction_C) and isinstance(self, (ParameterRef_C, _ParameterOp)):
			if isinstance(self, ParameterRef_C):
				if self == _null_:
					return other
				trial = LinearFunction_C()
				for component in other:
					if component.param == _null_:
						trial.append(self * component)
					else:
						trial = None
						break
				if trial is not None:
					return trial
			return ParameterMultiply(self, other.as_pmath())
		try:
			trial = LinearFunction_C()
			for component in self:
				trial.append(component * other)
			return trial
		except NotImplementedError:
			return NotImplemented

	def __truediv__(self, other):
		from .linear_math import _ParameterOp, ParameterDivide
		if isinstance(self, LinearFunction_C) and isinstance(other, (ParameterRef_C, _ParameterOp, _Number)):
			return ParameterDivide(self.as_pmath(), other)
		if isinstance(other, LinearFunction_C) and isinstance(self, (ParameterRef_C, _ParameterOp, _Number)):
			return ParameterDivide(self, other.as_pmath())
		return NotImplemented

	def __contains__(self, val):
		if isinstance(val, ParameterRef_C):
			for i in self:
				if i.param == val:
					return True
			return False
		if isinstance(val, DataRef_C):
			for i in self:
				if i.data == val:
					return True
			return False
		raise TypeError("the searched for content must be of type ParameterRef_C or DataRef")

	def _index_of(self, val):
		if isinstance(val, ParameterRef_C):
			for n, i in enumerate(self):
				if i.param == val:
					return n
			raise KeyError('ParameterRef_C not found')
		if isinstance(val, DataRef_C):
			for n, i in enumerate(self):
				if i.data == val:
					return n
			raise KeyError('DataRef not found')
		raise TypeError("the searched for content must be of type ParameterRef_C or DataRef")

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
		'(P.InVehTime_Suffix * X.IVTT) + (P.OutOfVehTime_Suffix * X.OVTT)'
		>>> f2 = f.reformat_param(pattern='(Veh)', repl='Vehicle')
		>>> str(f2)
		'(P.InVehicleTime * X.IVTT) + (P.OutOfVehicleTime * X.OVTT)'

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
			r += LinearComponent_C(data=str(i.data), param=container.format(param), scale=i.scale)
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
			r += LinearComponent_C(data=container.format(data), param=str(i.param), scale=i.scale)
		return r

	def __code__(self):
		return " + ".join(f"({repr(i)})" for i in self)

	def __eq__(self, other):
		if not isinstance(other, LinearFunction_C):
			return False
		if len(self) != len(other):
			return False
		for i, j in zip(self, other):
			if i != j: return False
		return True

	def __repr__(self):
		if len(self):
			result = " + ".join(repr(i) for i in self)
			if len(result)<80:
				return result
			else:
				return "  "+result.replace(" + ","\n+ ")
		return f"<Empty {self.__class__.__name__}>"

	def __xml__(
			self,
			linebreaks=False,
			lineprefix="",
			exponentiate_parameters=False,
			resolve_parameters=None,
			value_in_tooltips=True,
	):
		from xmle import Elem
		x = Elem('div' if use_tooltips else 'pre', attrib={'class':'LinearFunc'})
		for n,i in enumerate(self):
			ix_ = i.__xml__(
				exponentiate_parameter=exponentiate_parameters,
				resolve_parameters=resolve_parameters,
				value_in_tooltips=value_in_tooltips,
			).getchildren()
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
		if len(self) == 0:
			x << Elem('span', text=repr(self))
		return x

	def _repr_html_(self):
		return self.__xml__().tostring()

	def data(self, cls=None):
		if cls is None:
			return [_.data for _ in self]
		else:
			return [cls(_.data) for _ in self]

	def evaluate(self, param_source, x_namespace=None, **more_x_namespace):
		"""
		Evaluate the linear function in the context of some parameters and data.

		Typically all of the data given will be scalar values (to compute a
		scalar result) or a single data item will be a vector of possible
		values (to get a vector result).

		Parameters
		----------
		param_source : Model-like
			The source of the current parameter values.
		x_namespace : dict, optional
			A namespace of data values.
		**more_x_namespace : any
			More data values

		Returns
		-------
		numeric or array-like
		"""
		if hasattr(param_source, 'pvalue') and callable(param_source.pvalue):
			param_source = param_source.pvalue
		return sum(j.evaluate(param_source, x_namespace=x_namespace, **more_x_namespace) for j in self)

	def value(self, *args):
		return self.as_pmath().value(*args)

	def copy(self):
		result = self.__class__(self)
		return result

	def __deepcopy__(self, memodict):
		result = self.__class__()
		import copy
		for i in self:
			result.append(copy.deepcopy(i, memodict))
		return result

	def _linear_plot_2d_data(self, p_getter, x_name, x_min, x_max, n_points=100, **other_namespace):
		import numpy
		if hasattr(self, 'plotting_namespace') and len(other_namespace)==0:
			other_namespace = self.plotting_namespace
		x = numpy.linspace(x_min, x_max, n_points)
		y = self.evaluate(p_getter, {x_name:x}, **other_namespace)
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
		from ..util.plotting import plot_as_svg_xhtml
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

	def as_pmath(self):
		for i in self:
			if i.data != "1":
				raise NotImplementedError(f"{type(self)} has non-unit data")
		from .linear_math import ParameterMultiply, ParameterAdd
		if len(self) == 0:
			return 0
		elif len(self) == 1:
			return self[0].as_pmath()
		else:
			x = ParameterAdd(self[0].as_pmath(), self[1].as_pmath())
			for i in self[2:]:
				x = ParameterAdd(x, i.as_pmath())
			return x

cdef class DictOfLinearFunction_C:

	def __init__(self, mapping=None, alts_validator=None, **kwargs):
		self._map = {}
		if mapping is None:
			mapping = {}
		for k,v in mapping.items():
			self._map[k] = LinearFunction_C(v)
		for k,v in kwargs.items():
			try:
				self._map[k] = LinearFunction_C(v)
			except:
				print(v)
				print(type(v))
				raise

		self._alts_validator = alts_validator
		#self._instance = None

	def __fresh(self, instance):
		cdef DictOfLinearFunction_C newself
		newself = DictOfLinearFunction_C()
		newself._instance = instance
		setattr(instance, self.private_name, newself)
		return newself

	def __get__(self, instance, owner):

		cdef DictOfLinearFunction_C newself

		if instance is None:
			return self
		try:
			newself = getattr(instance, self.private_name)
		except AttributeError:
			newself = self.__fresh(instance)
		if newself is None:
			newself = self.__fresh(instance)
		return newself

	def __set__(self, instance, values):

		cdef DictOfLinearFunction_C newself

		try:
			newself = getattr(instance, self.private_name)
		except AttributeError:
			newself = self.__fresh(instance)
		if newself is None:
			newself = self.__fresh(instance)
		newself.__init__(values)
		#_try_mangle_h(newself)
		try:
			newself._instance.mangle()
		except AttributeError as err:
			pass # print(f"No Mangle L2: {err}")
		else:
			pass # print(f"Yes Mangle L2: {newself._instance}")

	def __delete__(self, instance):
		self.__set__(instance, None)

	def __set_name__(self, owner, name):
		self.name = name
		self.private_name = "_"+name

	def set_alts_validator(self, av):
		self._alts_validator = av

	def __getitem__(self, k):
		try:
			v = self._map[k]
			v.set_instance(self._instance)
			return v
		except KeyError:
			if self._alts_validator is None or self._alts_validator(k):
				v = self._map[k] = LinearFunction_C()
				v.set_instance(self._instance)
				return v
			else:
				raise

	def __setitem__(self, k, v):
		if isinstance(v, ParameterRef_C):
			v = v * DataRef_C('1')
		if isinstance(v, int) and v==0:
			v = LinearFunction_C()
		elif isinstance(v, LinearComponent_C):
			v = LinearFunction_C([v])
		elif isinstance(v, list):
			v = LinearFunction_C(v)
		elif not isinstance(v, LinearFunction_C):
			raise TypeError(f"values in {self.__class__} can only have type LinearFunction_C, not {type(v)}")
		if "with error" in repr(v):
			raise ValueError("found error here")
		v.set_instance(self._instance)
		self._map[k] = v
		_try_mangle(self._instance)

	def __delitem__(self, key):
		del self._map[key]
		_try_mangle(self._instance)

	def __iter__(self):
		return iter(self._map.keys())

	def __len__(self):
		return len(self._map)

	def keys(self):
		return self._map.keys()

	def items(self):
		return self._map.items()

	def values(self):
		return self._map.values()

	def copy(self):
		return type(self)(self)

	def __repr__(self):
		return '{0}({1})'.format(type(self).__name__, repr(self._map))

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
			for k,v in self._map.items():
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


cdef class GenericContainerCy:

	def __init__(self, arg):
		self.ident = arg

	def mangle(self):
		print(":MANGLE:",self.ident)

	def stats(self):
		print("LF:",self.lf)
		print("DLF:",self.dlf)
		print("_LF:",self._lf)
		print("_DLF:",self._dlf)


class GenericContainerPy(GenericContainerCy):

	lf = LinearFunction_C()
	dlf = DictOfLinearFunction_C()

	def __init__(self, arg):
		super().__init__(arg)

	def stats(self):
		print("LF:",self.lf)
		print("DLF:",self.dlf)
		super().stats()

