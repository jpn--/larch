from ._core import LarchError

class NoResultsError(LarchError):
	pass

class TooManyResultsError(LarchError):
	pass