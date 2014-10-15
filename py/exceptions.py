from ._core import ELM_Error

class NoResultsError(ELM_Error):
	pass

class TooManyResultsError(ELM_Error):
	pass