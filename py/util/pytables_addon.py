
import numpy


def select_with_repeated(valuenode, indexnode, screen=None):
	if screen is None:
		screen=slice(None)
	selectionlist = indexnode[screen]
	unique_selectionlist, rebuilder = numpy.unique(selectionlist, return_inverse=True)
	shapelen = len(valuenode.shape)
	if shapelen==1:
		values = valuenode[unique_selectionlist]
		return values[rebuilder]
	elif shapelen==2:
		values = valuenode[unique_selectionlist,:]
		return values[rebuilder,:]
	elif shapelen==3:
		values = valuenode[unique_selectionlist,:,:]
		return values[rebuilder,:,:]
	else:
		raise TypeError("shapelen cannot exceed 3")
