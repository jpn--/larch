from ._core import LarchError

from .core import SQLiteError, FacetError, LarchCacheError, ProvisioningError, MatrixInverseError

class NoResultsError(LarchError):
	pass

class TooManyResultsError(LarchError):
	pass