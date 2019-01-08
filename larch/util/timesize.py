import time
from datetime import datetime

def timesize_single(t):
	if t<60:
		return f"{t:.2f}s"
	elif t<3600:
		return f"{t/60:.2f}m"
	elif t<86400:
		return f"{t/3600:.2f}h"
	else:
		return f"{t/86400:.2f}d"


def timesize_stack(t):
	if t<60:
		return f"{t:.2f}s"
	elif t<3600:
		return f"{t//60:.0f}m {timesize_stack(t%60)}"
	elif t<86400:
		return f"{t//3600:.0f}h {timesize_stack(t%3600)}"
	else:
		return f"{t//86400:.0f}d {timesize_stack(t%86400)}"



class Seconds(float):

	def __repr__(self):
		t = self
		if t < 60:
			return f"{t:.2f}s"
		elif t < 3600:
			return f"{t//60:.0f}m {timesize_stack(t%60)}"
		elif t < 86400:
			return f"{t//3600:.0f}h {timesize_stack(t%3600)}"
		else:
			return f"{t//86400:.0f}d {timesize_stack(t%86400)}"


class Timer:

	def __init__(self):
		self.restart()

	def restart(self):
		self._starttime = datetime.now()
		self._stoptime = None
		return self._starttime

	def start(self):
		return self._starttime

	def stop(self):
		self._stoptime = datetime.now()
		return self._stoptime

	def elapsed(self):
		if self._stoptime:
			return (self._stoptime - self._starttime)
		else:
			return (datetime.now() - self._starttime)

	def __enter__(self):
		self.restart()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.stop()
