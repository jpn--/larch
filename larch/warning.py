
import warnings
import contextlib

@contextlib.contextmanager
def ignore_warnings():
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		yield


class NoDefaultValueWarning(Warning):
	pass

def no_default_value(message):
	#warnings.warn(message, NoDefaultValueWarning, stacklevel=2)
	pass

