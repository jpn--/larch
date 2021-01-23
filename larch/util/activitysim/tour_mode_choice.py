import os
import numpy as np
import pandas as pd
import yaml
from typing import Collection
from .. import Dict

from .general import (
	remove_apostrophes,
	construct_nesting_tree,
	linear_utility_from_spec,
	explicit_value_parameters,
	apply_coefficients,
)
from ... import Model, DataFrames, P, X

def clean_values(
		values,
		alt_names,
		choice_col='override_choice',
		alt_names_to_codes=None,
		choice_code='override_choice_code',
):
	"""

	Parameters
	----------
	values : pd.DataFrame
	alt_names : Collection
	override_choice : str
		The columns of `values` containing the observed choices.
	alt_names_to_codes : Mapping, optional
		If the `override_choice` column contains alternative names,
		use this mapping to convert the names into alternative code
		numbers.
	choice_code : str, default 'override_choice_code'
		The name of the observed choice code number column that will
		be added to `values`.

	Returns
	-------
	pd.DataFrame
	"""
	values = remove_apostrophes(values)
	values.fillna(0, inplace=True)
	# Remove choosers with invalid observed choice
	values = values[values.override_choice.isin(alt_names)]
	# Convert choices to code numbers
	if alt_names_to_codes is not None:
		values[choice_code] = values[choice_col].map(alt_names_to_codes)
	else:
		values[choice_code] = values[choice_col]
	return values




def tour_mode_choice_model(
		edb_directory="output/estimation_data_bundle/tour_mode_choice/",
		coefficients_file="tour_mode_choice_coefficients.csv",
		coefficients_template="tour_mode_choice_coefficients_template.csv",
		spec_file="tour_mode_choice_SPEC.csv",
		settings_file="tour_mode_choice_model_settings.yaml",
		values_file="tour_mode_choice_values_combined.csv",
		return_data=False,
):

	def _read_csv(filename, **kwargs):
		return pd.read_csv(os.path.join(edb_directory, filename), **kwargs)

	coefficients = _read_csv(
		coefficients_file,
		index_col='coefficient_name',
	)

	coef_template = _read_csv(
		coefficients_template,
		index_col='coefficient_name',
	)

	spec = _read_csv(
		spec_file,
	)
	spec = remove_apostrophes(spec, ['Label'])
	alt_names = list(spec.columns[3:])
	alt_codes = np.arange(1, len(alt_names) + 1)
	alt_names_to_codes = dict(zip(alt_names, alt_codes))
	alt_codes_to_names = dict(zip(alt_codes, alt_names))

	values = _read_csv(
		values_file,
		index_col='tour_id',
	)
	values = clean_values(
		values,
		alt_names,
		alt_names_to_codes=alt_names_to_codes,
		choice_code='override_choice_code',
	)

	with open(os.path.join(edb_directory, settings_file), "r") as yf:
		settings = yaml.load(
			yf,
			Loader=yaml.SafeLoader,
		)

	tree = construct_nesting_tree(alt_names, settings['NESTS'])

	purposes = list(coef_template.columns)

	# Setup purpose specific models
	m = {purpose: Model(graph=tree) for purpose in purposes}
	for alt_code, alt_name in tree.elemental_names().items():
		# Read in base utility function for this alt_name
		u = linear_utility_from_spec(
			spec, x_col='Label', p_col=alt_name,
			ignore_x=('#',),
		)
		for purpose in purposes:
			# Modify utility function based on template for purpose
			u_purp = sum(
				(
						P(coef_template[purpose].get(i.param, i.param))
						* i.data * i.scale
				)
				for i in u
			)
			m[purpose].utility_co[alt_code] = u_purp

	for model in m.values():
		explicit_value_parameters(model)
	apply_coefficients(coefficients, m)

	avail = {}
	for acode, aname in alt_codes_to_names.items():
		unavail_cols = list(
			(values[i.data] if i.data in values else values.eval(i.data))
			for i in m[purposes[0]].utility_co[acode]
			if i.param == "-999"
		)
		if len(unavail_cols):
			avail[acode] = sum(unavail_cols) == 0
		else:
			avail[acode] = 1
	avail = pd.DataFrame(avail).astype(np.int8)
	avail.index = values.index

	d = DataFrames(
		co=values,
		av=avail,
		alt_codes=alt_codes,
		alt_names=alt_names,
	)

	for purpose, model in m.items():
		model.dataservice = d.selector_co(f"tour_type=='{purpose}'")
		model.choice_co_code = 'override_choice_code'

	from larch.model.model_group import ModelGroup
	mg = ModelGroup(m.values())

	if return_data:
		return mg, Dict(
			values=values,
			avail=avail,
			coefficients=coefficients,
			coef_template=coef_template,
			spec=spec,
		)

	return mg