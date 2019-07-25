
import numpy as np
import pandas as pd
import pandas.core.algorithms as algos

def remove_unused_level(multiindex, level=0):
	"""
	Create a new MultiIndex from the current that removes
	unused levels, meaning that they are not expressed in the labels.
	The resulting MultiIndex will have the same outward
	appearance, meaning the same .values and ordering. It will also
	be .equals() to the original.
	.. versionadded:: 0.20.0
	Returns
	-------
	MultiIndex
	Examples
	--------
	>>> mi = pd.MultiIndex.from_product([range(2), list('ab')])
	>>> mi
	MultiIndex([(0, 'a'),
				(0, 'b'),
				(1, 'a'),
				(1, 'b')],
			   )
	>>> mi[2:]
	MultiIndex([(1, 'a'),
				(1, 'b')],
			   )
	The 0 from the first level is not represented
	and can be removed
	>>> mi2 = mi[2:].remove_unused_levels()
	>>> mi2.levels
	FrozenList([[1], ['a', 'b']])
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

