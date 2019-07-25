
import numpy as np
import pandas as pd
import pandas.core.algorithms as algos

def remove_unused_level(multiindex, level=0):
	"""
	Remove unused labels from a level.

	Create a new MultiIndex from the current that removes
	unused levels from a given level[s], meaning that they are
	not expressed in that level's labels.
	The resulting MultiIndex will have the same outward
	appearance, meaning the same .values and ordering. It will also
	be .equals() to the original.

	Parameters
	----------
	multiindex : pandas.MultiIndex
	level : int or List[int]

	Returns
	-------
	pandas.MultiIndex

	"""

	new_levels = []
	new_codes = []
	if isinstance(level, int):
		level = (level, )

	changed = False
	for n, (lev, level_codes) in enumerate(zip(multiindex.levels, multiindex.codes)):

		if n in level:
			# Since few levels are typically unused, bincount() is more
			# efficient than unique() - however it only accepts positive values
			# (and drops order):
			uniques = np.where(np.bincount(level_codes + 1) > 0)[0] - 1
			has_na = int(len(uniques) and (uniques[0] == -1))

			if len(uniques) != len(lev) + has_na:
				# We have unused levels
				changed = True

				# Recalculate uniques, now preserving order.
				# Can easily be cythonized by exploiting the already existing
				# "uniques" and stop parsing "level_codes" when all items
				# are found:
				uniques = algos.unique(level_codes)
				if has_na:
					na_idx = np.where(uniques == -1)[0]
					# Just ensure that -1 is in first position:
					uniques[[0, na_idx[0]]] = uniques[[na_idx[0], 0]]

				# codes get mapped from uniques to 0:len(uniques)
				# -1 (if present) is mapped to last position
				code_mapping = np.zeros(len(lev) + has_na)
				# ... and reassigned value -1:
				code_mapping[uniques] = np.arange(len(uniques)) - has_na

				level_codes = code_mapping[level_codes]

				# new levels are simple
				lev = lev.take(uniques[has_na:])

		new_levels.append(lev)
		new_codes.append(level_codes)

	result = multiindex.view()

	if changed:
		result._reset_identity()
		result._set_levels(new_levels, validate=False)
		result._set_codes(new_codes, validate=False)

	return result


def replace_levels(multiindex, level, new_label_array):
	"""
	Replace the labels in a level with a new order.

	Parameters
	----------
	multiindex : pandas.MultiIndex
	level : int
	new_label_array : array-like

	Returns
	-------
	pandas.MultiIndex
	"""

	levels = [i for i in multiindex.levels]
	codes = [i for i in multiindex.codes]

	new_label_array = np.asarray(new_label_array)
	old_label_array = multiindex.levels[level]
	where_in_new_label_array = {}
	for n, j in enumerate(old_label_array):
		try:
			where_in_new_label_array[n] = int(np.where(j == new_label_array)[0])
		except TypeError:
			raise ValueError(f"missing {j} in new_label_array, all the old labels must appear")

	old_codes = list(multiindex.codes[level])
	codes[level] = list(map(where_in_new_label_array.get, old_codes))
	levels[level] = new_label_array

	return pd.MultiIndex(levels=levels, codes=codes, names=multiindex.names)