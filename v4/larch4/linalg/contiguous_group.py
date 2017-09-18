import numpy


class Blocker:
	def __init__(self, outer_shape, inner_shapes, dtype=numpy.float64):
		self.inner_sizes = [numpy.prod(s) for s in inner_shapes]
		last_dim = numpy.sum(self.inner_sizes)
		self.outer = numpy.zeros(tuple(outer_shape) + (last_dim,), dtype=dtype)
		i = 0
		self.inners = []
		for j, inner_shape in enumerate(inner_shapes):
			inner_j = self.outer[..., i:i + self.inner_sizes[j]]
			self.inners.append(inner_j.reshape(*outer_shape, *inner_shape))
			i += self.inner_sizes[j]

	def __getitem__(self, item):
		return self.outer.__getitem__(item)

	def __setitem__(self, item, value):
		return self.outer.__setitem__(item, value)

	@property
	def shape_prefix(self):
		return self.outer.shape[:-1]

	@property
	def dtype(self):
		return self.outer.dtype