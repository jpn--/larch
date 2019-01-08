
import apsw
import numpy
import pandas

from .sqlite_arrays import _sqlite_array_2d_float64

class Connection(apsw.Connection):

	def read_data(self, query, dtype=numpy.float64, index_col=None):
		if dtype==numpy.float64:
			return _sqlite_array_2d_float64(self.sqlite3pointer(), query, index_col=index_col)
		raise TypeError("cannot read in {}".format(dtype))


