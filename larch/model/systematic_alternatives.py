
import pandas
import numpy

from . import Model


def bitmask_shift_value(b):
	s = 1
	r = bin(b)
	while r[-s] == '0':
		s += 1
	return s-1


class SystematicAlternatives:

	__slots__ = [
		'masks', 'groupby', 'categoricals', '_categoricals_source', 'altcodes', 'padding_levels',
		'_suggested_caseindex', '_suggested_alts_name',
	]

	def __init__(self, masks, groupby, categoricals, categoricals_source, altcodes, padding_levels,
				 suggested_caseindex=None, suggested_alts_name=None):
		self.masks = masks
		self.groupby = groupby
		self.categoricals = categoricals
		self._categoricals_source = categoricals_source
		self.altcodes = altcodes
		self.padding_levels = padding_levels
		self._suggested_caseindex = suggested_caseindex
		self._suggested_alts_name = suggested_alts_name

	def __repr__(self):
		s = "<larch.data_services.SystematicAlternatives>"
		for g,c in zip(self.groupby, self.categoricals):
			s += f"\n | {g}:"
			s += f"\n |   {str(c)}"
		return s

	def alternative_name_from_code(self, code):
		"""

		Parameters
		----------
		code : int
			The code integer for an alternative.

		Returns
		-------
		str
			A descriptive name for the alternative, made by concatenating the various
			categorical names with the within-category integer.

		"""
		name = ""
		for i,mask in enumerate(self.masks[:-1]):
			j = (mask & code) >> bitmask_shift_value(mask)
			j -= 1
			if j<0:
				name += "§-"
			else:
				try:
					name += f"{self.categoricals[i][j]}-"
				except IndexError:
					name += f"+{j+1-len(self.categoricals[i])}-"
		mask = self.masks[-1]
		j = (mask & code) >> bitmask_shift_value(mask)
		if j==0:
			name += f"§"
		else:
			name += f"{j}"
		return name

	@property
	def altnames(self):
		return [self.alternative_name_from_code(code) for code in self.altcodes]

	def compute_alternative_codes(
			self,
			df,
			caseindex=None,
			name=None,
			overwrite=False,
	):
		"""
		Compute alternative codes consistent with this SystematicAlternatives.

		One example where this method is useful is to recode a testing or validation DataFrame in the same way
		that a training DataFrame was coded.

		Parameters
		----------
		df : pandas.DataFrame
			An idca or idce dataframe, where each row contains attributes of one particular alternative.
		caseindex : str
			If the case identifier is a column in `df` or the index of `df` is not named, this gives
			the name to use.
		name : str
			Name of the new alternative codes column.
		overwrite : bool, default False
			Should existing variable with `name` be overwritten. It False and it already exists, a
			'_v#' suffix is added.

		Returns
		-------
		pandas.DataFrame
		"""
		result, _ = new_alternative_codes(
			df,
			groupby=self.groupby,
			caseindex=caseindex if caseindex is not None else self._suggested_caseindex,
			name=name if name is not None else self._suggested_alts_name,
			padding_levels=self.padding_levels,
			groupby_prefixes=None,
			overwrite=overwrite,
			complete_features_list=self._categoricals_source,
		)
		return result

def _remap_position(i, found_alts):
	x = numpy.where(found_alts == i)[0]
	if x.size > 0:
		return x[0]
	else:
		return -1


def _multiple_new_alternative_codes(
		df_list,
		groupby,
		caseindex=None,
		name='alternative_code',
		padding_levels=4,
		groupby_prefixes=None,
		overwrite=False,
		complete_features_list=None,
):
	"""

	Parameters
	----------
	df_list : list of pandas.DataFrame
		A list of idca or idce dataframes, where each row contains attributes of one particular alternative.
	groupby : str or list
		The column or columns that will define the new alternative codes. Every unique combination
		of values in these columns will be grouped together in such a way as to allow a nested logit
		tree to be build with the unique values defining the nest memberships.  As such, the order
		of columns given in this list is meaningful.
	caseindex : str
		If the case identifier is a column in `df` or the index of `df` is not named, this gives
		the name to use.
	name : str
		Name of the new alternative codes column.
	padding_levels : int, default 4
		Space for this number of "extra" levels is reserved in each mask, beyond the number of
		detected categories.  This is critical if these alternative codes will be used for OGEV models
		that require extra nodes at levels that are cross-nested.
	overwrite : bool, default False
		Should existing variable with `name` be overwritten. It False and it already exists, a
		'_v#' suffix is added.
	complete_features_list : dict, optional
		Give a complete, ordered enumeration of all possible feature values for each `groupby` level.
		If any level is not specifically identified, the unique values observed in this dataset are used.
		This argument can be important for OGEV models (e.g. when grouping by time slot but some time slots
		have no alternatives present) or when train and test data may not include a completely overlapping
		set of feature values.

	Returns
	-------
	pandas.DataFrame, SystematicAlternatives

	"""

	df1_list = []
	for df in df_list:

		if df.index.nlevels > 1:
			if caseindex is None:
				caseindex = df.index.names[0]
			if caseindex is None:
				raise ValueError('cannot identify caseindex, existing multi-index top level is un-named')
			df1 = df.reset_index()
		elif df.index.name is not None:
			caseindex = df.index.name
			df1 = df.reset_index()
		elif caseindex is not None:
			df1 = df
		else:
			raise ValueError('cannot identify caseindex, existing index is un-named')

		df1_list.append(df1)

	if isinstance(groupby, str):
		groupby = (groupby,)

	if groupby_prefixes is None:
		groupby_prefixes = ["" for g in groupby]

	_df_list = [df1.__getitem__([caseindex, *groupby]) for df1 in df1_list]

	s_list = [(_df.groupby([caseindex, *groupby]).cumcount() + 1) for _df in _df_list]

	s_max_list = [s.max() for s in s_list]
	s_max_overall = numpy.max(s_max_list)

	masks = numpy.zeros(len(groupby) + 1, dtype=numpy.int64)

	label_offset = int(numpy.ceil(numpy.log2(s_max_overall)))

	if complete_features_list is None:
		complete_features_list = {}

	all_uniqs = {}
	for i, g in enumerate(groupby):
		if g in complete_features_list:
			all_uniqs[g] = complete_features_list[g]
		else:
			all_uniqs[g] = set()
			for _df in _df_list:
				all_uniqs[g] |= set(numpy.unique(_df.iloc[:, i + 1]))
			all_uniqs[g] = sorted(all_uniqs[g])

	uniqs_list = [
		[numpy.unique(_df.iloc[:, i + 1], return_inverse=True) for i, g in enumerate(groupby)]
		for _df in _df_list
	]

	for uniqs in uniqs_list:
		for i, g in enumerate(groupby):
			found_alts, found_map = uniqs[i]
			use_alts = all_uniqs[g]
			_remap = lambda _y: _remap_position(_y, found_alts)
			use_map = {_remap(_nn): _n for _n, _nn in enumerate(use_alts) if _remap(_nn) >= 0}
			uniqs[i] = (
				all_uniqs[g],
				numpy.asarray(list(map(use_map.__getitem__, found_map)))
			)

	if len(groupby) > 0:
		n_uniqs = [len(all_uniqs[g]) + 1 + padding_levels for i, g in enumerate(groupby)]
	else:
		n_uniqs = []

	bitmask_sizes = [int(numpy.ceil(numpy.log2(g))) for g in n_uniqs]

	x_list = []
	for uniqs, s, _df, df1 in zip(uniqs_list, s_list, _df_list, df1_list):
		if len(bitmask_sizes):
			x = uniqs[0][1] + 1
			masks[0] = 2 ** bitmask_sizes[0] - 1
		else:
			x = numpy.zeros(len(_df), dtype=int)

		for i in range(1, len(bitmask_sizes)):
			x <<= bitmask_sizes[i]
			masks <<= bitmask_sizes[i]
			x += uniqs[i][1] + 1
			masks[i] = 2 ** bitmask_sizes[i] - 1

		x <<= label_offset
		masks <<= label_offset
		x += s
		masks[-1] = 2 ** label_offset - 1

		altcodes = numpy.unique(x)

		if name in df1.columns and not overwrite:
			_number = 2
			while f'{name}_v{_number}' in df1.columns:
				_number += 1
			name = f'{name}_v{_number}'

		x_list.append(x)

	sa = SystematicAlternatives(
		masks=masks,
		groupby=groupby,
		categoricals=[
			all_uniqs[g] if groupby_prefixes[n] is None else [groupby_prefixes[n] + str(uu) for uu in all_uniqs[g]]
			for n, g in enumerate(groupby)
		],
		categoricals_source={
			g:all_uniqs[g]
			for g in groupby
		},
		altcodes=altcodes,
		padding_levels=padding_levels,
		suggested_caseindex=caseindex,
		suggested_alts_name=name,
	)

	for n, df1 in enumerate(df1_list):
		df1[name] = x_list[n]
		df1_list[n] = df1.set_index([caseindex, name])

	return df1_list, sa





def new_alternative_codes(
		df,
		groupby,
		caseindex=None,
		name='alternative_code',
		padding_levels=4,
		groupby_prefixes=None,
		overwrite=False,
		complete_features_list=None,
):
	"""Create new systematic alternatives.

	Parameters
	----------
	df : pandas.DataFrame
		An idca or idce dataframe, where each row contains attributes of one particular alternative.
	groupby : str or list
		The column or columns that will define the new alternative codes. Every unique combination
		of values in these columns will be grouped together in such a way as to allow a nested logit
		tree to be build with the unique values defining the nest memberships.  As such, the order
		of columns given in this list is meaningful.
	caseindex : str
		If the case identifier is a column in `df` or the index of `df` is not named, this gives
		the name to use.
	name : str
		Name of the new alternative codes column.
	padding_levels : int, default 4
		Space for this number of "extra" levels is reserved in each mask, beyond the number of
		detected categories.  This is critical if these alternative codes will be used for OGEV models
		that require extra nodes at levels that are cross-nested.
	overwrite : bool, default False
		Should existing variable with `name` be overwritten. If False and it already exists, a
		'_v#' suffix is added.
	complete_features_list : dict, optional
		Give a complete, ordered enumeration of all possible feature values for each `groupby` level.
		If any level is not specifically identified, the unique values observed in this dataset are used.
		This argument can be important for OGEV models (e.g. when grouping by time slot but some time slots
		have no alternatives present) or when train and test data may not include a completely overlapping
		set of feature values.

	Returns
	-------
	pandas.DataFrame, SystematicAlternatives

	"""

	if isinstance(df, (list, tuple)):
		return _multiple_new_alternative_codes(
			df,
			groupby,
			caseindex=caseindex,
			name=name,
			padding_levels=padding_levels,
			groupby_prefixes=groupby_prefixes,
			overwrite=overwrite,
			complete_features_list=complete_features_list,
		)

	if df.index.nlevels > 1:
		if caseindex is None:
			caseindex = df.index.names[0]
		if caseindex is None:
			raise ValueError('cannot identify caseindex, existing multi-index top level is un-named')
		df1 = df.reset_index()
	elif df.index.name is not None:
		caseindex = df.index.name
		df1 = df.reset_index()
	elif caseindex is not None:
		df1 = df
	else:
		raise ValueError('cannot identify caseindex, existing index is un-named')

	if isinstance(groupby, str):
		groupby = (groupby,)

	if groupby_prefixes is None:
		groupby_prefixes = ["" for g in groupby]

	_df = df1.__getitem__([caseindex, *groupby])

	s = _df.groupby([caseindex, *groupby]).cumcount() + 1

	masks = numpy.zeros(len(groupby) + 1, dtype=numpy.int64)

	label_offset = int(numpy.ceil(numpy.log2(s.max())))

	uniqs = [numpy.unique(_df.iloc[:, i + 1], return_inverse=True) for i, g in enumerate(groupby)]

	if complete_features_list is None:
		complete_features_list = {}

	for i, g in enumerate(groupby):
		if g in complete_features_list:
			found_alts, found_map = uniqs[i]
			use_alts = complete_features_list[g]
			_remap = lambda _y: _remap_position(_y, found_alts)
			use_map = {_remap(_nn): _n for _n, _nn in enumerate(use_alts) if _remap(_nn) >= 0}
			uniqs[i] = (
				complete_features_list[g],
				numpy.asarray(list(map(use_map.__getitem__, found_map)))
			)

	if len(groupby) > 0:
		n_uniqs = [len(uniqs[i][0]) + 1 + padding_levels for i, g in enumerate(groupby)]
	else:
		n_uniqs = []

	bitmask_sizes = [int(numpy.ceil(numpy.log2(g))) for g in n_uniqs]

	if len(bitmask_sizes):
		x = uniqs[0][1] + 1
		masks[0] = 2 ** bitmask_sizes[0] - 1
	else:
		x = numpy.zeros(len(_df), dtype=int)

	for i in range(1, len(bitmask_sizes)):
		x <<= bitmask_sizes[i]
		masks <<= bitmask_sizes[i]
		x += uniqs[i][1] + 1
		masks[i] = 2 ** bitmask_sizes[i] - 1

	x <<= label_offset
	masks <<= label_offset
	x += s
	masks[-1] = 2 ** label_offset - 1

	altcodes = numpy.unique(x)

	if name in df1.columns and not overwrite:
		_number = 2
		while f'{name}_v{_number}' in df1.columns:
			_number += 1
		name = f'{name}_v{_number}'

	sa = SystematicAlternatives(
		masks=masks,
		groupby=groupby,
		categoricals=[
			u[0] if groupby_prefixes[n] is None else [groupby_prefixes[n] + str(uu) for uu in u[0]]
			for n, u in enumerate(uniqs)
		],
		categoricals_source={
			g:u[0]
			for g, u in zip(groupby, uniqs)
		},
		altcodes=altcodes,
		padding_levels=padding_levels,
		suggested_caseindex=caseindex,
		suggested_alts_name=name,
	)

	df1[name] = x
	df1 = df1.set_index([caseindex, name])

	return df1, sa



def magic_nesting(model, sys_alts=None, mu_parameters=None, mu_prefix='MU_'):
	"""
	Automatically build an NL model based on categories used for idce.

	Parameters
	----------
	model : larch.Model
	sys_alts : SystematicAlternatives, optional
		As returned by `new_alternative_codes`.  If not given, this method will
		attempt to extract sys_alts from the current `dataframes` or `dataservice`.
	mu_parameters : iterable, optional
		A list of mu parameter names, to be applied to each level defined by groupby. If not given,
		the list is created with `mu_prefix` and `sys_alts.groupby`.
	mu_prefix : str, optional
		A prefix to append to each item in `sys_alts.groupby` if `mu_parameters` is omitted.

	"""
	if sys_alts is None:
		try:
			sys_alts = model.dataframes.sys_alts
		except AttributeError:
			pass

	if sys_alts is None:
		try:
			sys_alts = model.dataservice.sys_alts
		except AttributeError:
			pass

	if sys_alts is None:
		raise ValueError('cannot find sys_alts')

	return magic_ogev_nesting(
		model,
		ogev_coverage=[1 for _ in sys_alts.groupby],
		sys_alts=sys_alts,
		mu_parameters=mu_parameters,
		mu_prefix=mu_prefix,
	)



def magic_ogev_nesting(model, ogev_coverage, sys_alts=None, mu_parameters=None, mu_prefix='MU_', ):
	"""
	Automatically build an OGEV model based on categories used for idce.

	Parameters
	----------
	model : larch.Model
	ogev_coverage : iterable
		For each grouby layer in `sys_alts`, how many ordered cross-nests should be created.  Length
		of this iterable must match the length of the groupby in `sys_alts`.  Give all 1's to create
		a normal nested logit model, or set some values greater than 1 to create an OGEV model. Note
		that multiple layers can have values greater than 1, but the resulting model is quite likely
		to be unwieldy.
	sys_alts : SystematicAlternatives
		As returned by `new_alternative_codes`.  If not given, this method will
		attempt to extract sys_alts from the current `dataframes`.
	mu_parameters : iterable, optional
		A list of mu parameter names, to be applied to each level defined by groupby. If not given,
		the list is created with `mu_prefix` and `sys_alts.groupby`.
	mu_prefix : str, optional
		A prefix to append to each item in `sys_alts.groupby` if `mu_parameters` is omitted.

	"""
	if sys_alts is None:
		try:
			sys_alts = model.dataframes.sys_alts
		except AttributeError:
			pass

	if sys_alts is None:
		try:
			sys_alts = model.dataservice.sys_alts
		except AttributeError:
			pass

	if sys_alts is None:
		raise ValueError('cannot find sys_alts')

	groupby = sys_alts.groupby
	masks = sys_alts.masks
	unique_alt_codes = sys_alts.altcodes
	prev_level_mask = 0
	prev_mask_1 = 0
	prev_t = 0

	if mu_parameters is None:
		mu_parameters = [f'{mu_prefix}{i}' for i in groupby]

	for level in range(len(groupby)):
		level_mask = int(numpy.sum(masks[:level+1]))
		mask_1 = 1 << bitmask_shift_value(level_mask)

		t = ogev_coverage[level]-1

		if t > sys_alts.padding_levels:
			raise ValueError('the level of OGEV coverage exceeds the padding level for sys_alts')

		unique_level_nest_codes = set(numpy.unique(unique_alt_codes & level_mask))
		if t > 0:
			for each_t in range(t):
				unique_level_nest_codes |= set(numpy.unique((unique_alt_codes+mask_1*(each_t+1)) & level_mask))

		for nestcode in unique_level_nest_codes:
			model.graph.add_node(nestcode, parameter=mu_parameters[level], name=sys_alts.alternative_name_from_code(nestcode))

		for nestcode in unique_level_nest_codes:
			if level == 0:
				model.graph.add_edge(model.graph.root_id, nestcode)
			else:
				model.graph.add_edge(nestcode & prev_level_mask, nestcode)
				if prev_t > 0:
					for each_t in range(prev_t):
						model.graph.add_edge((nestcode+prev_mask_1*(each_t+1)) & prev_level_mask, nestcode)

		prev_level_mask = level_mask
		prev_mask_1 = mask_1
		prev_t = t

	# Elemental Alternatives
	level_mask = numpy.cumsum(masks)

	for altcode in unique_alt_codes:
		model.graph.add_edge(altcode & prev_level_mask, altcode)

		if prev_t > 0:
			for each_t in range(prev_t):
				model.graph.add_edge((altcode + prev_mask_1 * (each_t+1)) & prev_level_mask, altcode)

	# Strip nesting nodes with no successors
	# Not actually needed; only required nodes are added in the first place...
	# elemental_mask = masks[-1]
	# for code, out_degree in self.graph.out_degree:
	# 	if not out_degree:
	# 		if not (code & elemental_mask):
	# 			self.graph.remove_node(code)

	model.mangle()


from .model import Model
setattr(Model, 'magic_nesting', magic_nesting)
setattr(Model, 'magic_ogev_nesting', magic_ogev_nesting)