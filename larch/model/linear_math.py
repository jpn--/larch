
from .linear import ParameterRef_C, _what_is

_precedence = {
	'lambda': 0,
	'if': 1,
	'or': 2,
	'and': 3,
	'not': 4,
	'compare': 5,
	'|': 6,
	'^': 7,
	'&': 8,
	'>>': 9,
	'+': 10,
	'*': 11,
	'~': 12,
	'**': 13,
}

def _parenthize_repr(obj, local_precedence):
	obj_precedence = getattr(obj, '_precedence', 0)
	if obj_precedence < local_precedence:
		return f"({repr(obj)})"
	else:
		return repr(obj)


class _ParameterOp():

	def set_fmt(self, formatting):
		self._formatting = formatting
		return self

	def value(self, *args):
		raise NotImplementedError('abstract base class')

	def string(self, m):
		"""
		The value of the parameter math from a given model, as a formatted string.

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

	def __add__(self, other):
		if isinstance(other, (ParameterRef_C, _ParameterOp)):
			return ParameterAdd(self, other)
		raise NotImplementedError(f"{_what_is(self)} + {_what_is(other)}")

	def __radd__(self, other):
		if isinstance(other, (ParameterRef_C, _ParameterOp)):
			return ParameterAdd(other, self)
		raise NotImplementedError(f"{_what_is(other)} + {_what_is(self)}")

	def __sub__(self, other):
		if isinstance(other, (ParameterRef_C, _ParameterOp)):
			return ParameterSubtract(self, other)
		raise NotImplementedError(f"{_what_is(self)} - {_what_is(other)}")

	def __rsub__(self, other):
		if isinstance(other, (ParameterRef_C, _ParameterOp)):
			return ParameterSubtract(other, self)
		raise NotImplementedError(f"{_what_is(other)} - {_what_is(self)}")

	def __mul__(self, other):
		if isinstance(other, (ParameterRef_C, _ParameterOp)):
			return ParameterMultiply(self, other)
		raise NotImplementedError(f"{_what_is(self)} * {_what_is(other)}")

	def __rmul__(self, other):
		if isinstance(other, (ParameterRef_C, _ParameterOp)):
			return ParameterMultiply(other, self)
		raise NotImplementedError(f"{_what_is(other)} * {_what_is(self)}")

	def __truediv__(self, other):
		if isinstance(other, (ParameterRef_C, _ParameterOp)):
			return ParameterDivide(self, other)
		raise NotImplementedError(f"{_what_is(self)} / {_what_is(other)}")

	def __rtruediv__(self, other):
		if isinstance(other, (ParameterRef_C, _ParameterOp)):
			return ParameterDivide(other, self)
		raise NotImplementedError(f"{_what_is(other)} / {_what_is(self)}")


ParameterOp = _ParameterOp

class _ParameterUnaryOp(_ParameterOp):
	_op = '??'
	_precedence = 0

	def __init__(self, operand):
		self._operand = operand
		self._formatting = getattr(operand, '_formatting', None)

	def valid(self, m):
		return self._operand.valid(m)

	def __repr__(self):
		return "{}{}".format(
			self._op,
			_parenthize_repr(self._operand, self._precedence),
		)


class _ParameterBinaryOp(_ParameterOp):
	_op = '??'
	_precedence = 0

	def __init__(self, left, right):
		self._left = left
		self._right = right
		self._formatting = getattr(left, '_formatting', None) or getattr(right, '_formatting', None)

	def valid(self, m):
		if isinstance(self._left, (ParameterRef_C, _ParameterOp)):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, (ParameterRef_C, _ParameterOp)):
			x &= self._right.valid(m)
		return x

	def __repr__(self):
		return "{} {} {}".format(
			_parenthize_repr(self._left, self._precedence),
			self._op,
			_parenthize_repr(self._right, self._precedence),
		)

	def __str__(self):
		return "{} {} {}".format(
			_parenthize_repr(self._left, self._precedence),
			self._op,
			_parenthize_repr(self._right, self._precedence),
		)


class ParameterAdd(_ParameterBinaryOp):
	_op = '+'
	_precedence = 10

	def value(self, *args):
		if isinstance(self._left, (ParameterRef_C, _ParameterOp)):
			x = self._left.value(*args)
		else:
			x = self._left
		if isinstance(self._right, (ParameterRef_C, _ParameterOp)):
			x += self._right.value(*args)
		else:
			x += self._right
		return x


class ParameterMultiply(_ParameterBinaryOp):
	_op = '*'
	_precedence = 11

	def value(self, *args):
		if isinstance(self._left, (ParameterRef_C, _ParameterOp)):
			x = self._left.value(*args)
		else:
			x = self._left
		if isinstance(self._right, (ParameterRef_C, _ParameterOp)):
			x *= self._right.value(*args)
		else:
			x *= self._right
		return x


class ParameterSubtract(_ParameterBinaryOp):
	_op = '-'
	_precedence = 10

	def value(self, *args):
		if isinstance(self._left, (ParameterRef_C, _ParameterOp)):
			x = self._left.value(*args)
		else:
			x = self._left
		if isinstance(self._right, (ParameterRef_C, _ParameterOp)):
			x -= self._right.value(*args)
		else:
			x -= self._right
		return x


class ParameterDivide(_ParameterBinaryOp):
	_op = '/'
	_precedence = 11

	def value(self, *args):
		if isinstance(self._left, (ParameterRef_C, _ParameterOp)):
			x = self._left.value(*args)
		else:
			x = self._left
		if isinstance(self._right, (ParameterRef_C, _ParameterOp)):
			x /= self._right.value(*args)
		else:
			x /= self._right
		return x


class ParameterNoop(_ParameterUnaryOp):
	_op = ''
	_precedence = 98

	def value(self, *args):
		return self._operand.value(*args)


class ParameterNegate(_ParameterUnaryOp):
	_op = '-'
	_precedence = 12

	def value(self, *args):
		return -self._operand.value(*args)

