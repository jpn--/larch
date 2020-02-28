
# Errors

class MissingDataError(ValueError):
	pass

class DuplicateColumnNames(ValueError):
	pass

class BHHHSimpleStepFailure(RuntimeError):
	pass

# Warnings

class ParameterNotInModelWarning(UserWarning):
	pass

