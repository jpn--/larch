
import numpy



def select_with_repeated1(groupnode, screen=None):
	if screen is None:
		screen=slice(None)
	transpose_values = ('transpose_values' in groupnode._v_attrs)
	valuenode = groupnode._values_
	indexnode = groupnode._index_
	selectionlist = indexnode[screen]
	unique_selectionlist, rebuilder = numpy.unique(selectionlist, return_inverse=True)
	if transpose_values:
		if len(unique_selectionlist)==1 and unique_selectionlist[0]==0:
			return numpy.broadcast_to(valuenode[:], (len(rebuilder), valuenode.shape[0]))
		else:
			raise TypeError("transpose_values must be a vector")
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


def validate_with_repeated1(groupnode, screen=None):
	if screen is None:
		screen=0
	return select_with_repeated1(groupnode, screen)













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


def validate_with_repeated(valuenode, indexnode, screen=None):
	if screen is None:
		screen=0
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
