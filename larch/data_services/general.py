import numpy

class NotSameShapeError(ValueError):
	pass


def _sqz(shape):
	if not isinstance(shape, tuple):
		shape = tuple(shape)
	while len(shape) and shape[-1] == 1:
		shape = shape[:-1]
	return shape

def _sqz_same(shape1, shape2, silent=False, comment=None):
	if _sqz(shape1) != _sqz(shape2):
		if silent:
			return False
		if comment is not None:
			raise NotSameShapeError(f"shapes not same: {shape1} vs {shape2}   # {comment}")
		raise NotSameShapeError(f"shapes not same: {shape1} vs {shape2}")
	return True


def _sqz_same_trailing_neg_ok(shape1, shape2, silent=False, comment=None):
	s1, s2 = _sqz(shape1), _sqz(shape2)
	if s1 != s2:
		# check for match on all but trailing dim
		if s1[:-1] == s2[:-1]:
			# check for a negative in the trailing dim
			if s1[-1] < 0 or s2[-1] < 0:
				return True
		if silent:
			return False
		if comment is not None:
			raise NotSameShapeError(f"shapes not same: {shape1} vs {shape2}   # {comment}")
		raise NotSameShapeError(f"shapes not same: {shape1} vs {shape2}")
	return True



def selector_len_for(selector, seqlen):
	if selector is None:
		return seqlen
	if isinstance(selector, slice):
		start, stop, step = selector.indices(seqlen)
		return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
	if isinstance(selector, numpy.ndarray):
		if selector.dtype == numpy.bool:
			if selector.shape[0] != seqlen:
				raise ValueError('bool array selector must be same size as unselected first dimension')
			return selector.sum()
		if numpy.issubdtype(selector.dtype, numpy.integer):
			return len(selector)
	raise TypeError('selector must be None or slice or ndarray')



def bitmask_shift_value(b):
	s = 1
	r = bin(b)
	while r[-s] == '0':
		s += 1
	return s-1


class SystematicAlternatives:

	__slots__ = ['masks', 'groupby', 'categoricals', 'altcodes', 'padding_levels']

	def __init__(self, masks, groupby, categoricals, altcodes, padding_levels):
		self.masks = masks
		self.groupby = groupby
		self.categoricals = categoricals
		self.altcodes = altcodes
		self.padding_levels = padding_levels

	def __repr__(self):
		s = "<larch.SystematicAlternatives>"
		for g,c in zip(self.groupby, self.categoricals):
			s += f"\n | {g}:"
			s += f"\n |   {str(c)}"
		return s

	def alternative_name_from_code(self, code):
		"""

		Parameters
		----------
		code : int
			The code integer for an alternative.

		Returns
		-------
		str
			A descriptive name for the alternative, made by concatenating the various
			categorical names with the within-category integer.

		"""
		name = ""
		for i,mask in enumerate(self.masks[:-1]):
			j = (mask & code) >> bitmask_shift_value(mask)
			j -= 1
			if j<0:
				name += "§-"
			else:
				try:
					name += f"{self.categoricals[i][j]}-"
				except IndexError:
					name += f"+{j+1-len(self.categoricals[i])}-"
		mask = self.masks[-1]
		j = (mask & code) >> bitmask_shift_value(mask)
		if j==0:
			name += f"§"
		else:
			name += f"{j}"
		return name

	@property
	def altnames(self):
		return [self.alternative_name_from_code(code) for code in self.altcodes]