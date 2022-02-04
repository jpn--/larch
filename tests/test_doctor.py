import larch, pytest
from larch import P,X,PX
from pytest import approx, raises
import numpy
import pandas

def test_nan_weight():
	m = larch.Model.Example(1, legacy=True)
	m.weight_co_var = 'hhowndum'
	m.load_data()
	m.dataframes.data_wt = m.dataframes.data_wt.div(m.dataframes.data_wt)

	m, diagnosis = m.doctor()
	assert type(m) is larch.Model
	assert 'nan_weight' in diagnosis
	assert diagnosis.nan_weight['n'].iloc[0] == 1882

	m, diagnosis = m.doctor(repair_nan_wt=True)
	assert type(m) is larch.Model
	assert 'nan_weight' in diagnosis
	assert diagnosis.nan_weight['n'].iloc[0] == 1882

	m, diagnosis = m.doctor()
	assert type(m) is larch.Model
	assert 'nan_weight' not in diagnosis
	assert len(diagnosis) == 0

def test_doctor_before_load_data():

	with raises(ValueError):
		larch.Model().doctor()
