

from ..jupyter import jupyter_active


def _dummy_progressbar(n, **kwarg):
	for i in range(n):
		yield i

progressbar = _dummy_progressbar

try:
	from tqdm import tnrange
except ImportError:
	# tqdm is unavailable, use a fallback dummy class
	tnrange = _dummy_progressbar


try:
	from tqdm import trange
except ImportError:
	# tqdm is unavailable, use a fallback dummy class
	trange = _dummy_progressbar



def activate():
	global progressbar
	if jupyter_active:
#		progressbar = tnrange
		progressbar = trange
	else:
		progressbar = trange


def deactivate():
	global progressbar
	progressbar = _dummy_progressbar


# default on
activate()
