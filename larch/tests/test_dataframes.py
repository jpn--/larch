import pandas
import os
import gzip
import io
import numpy
from pytest import approx

from ..data_warehouse import example_file

from .. import DataFrames

def test_dfs_info():

	from ..data_warehouse import example_file
	df = pandas.read_csv(example_file("MTCwork.csv.gz"))
	df.set_index(['casenum', 'altnum'], inplace=True)

	ds = DataFrames(df)

	s = io.StringIO()

	ds.info(out=s)

	assert s.getvalue() == (
		'larch.DataFrames:  (not computation-ready)\n'
		'  n_cases: 5029\n'
		'  n_alts: 6\n'
		'  data_ce: 36 variables, 22033 rows\n'
		'  data_co: <not populated>\n'
		'  data_av: <populated>\n')

	s = io.StringIO()
	ds.info(out=s, verbose=True)

	assert s.getvalue() == (
		'larch.DataFrames:  (not computation-ready)\n  n_cases: 5029\n  n_alts: 6\n  data_ce: 22033 rows\n'
		'    - chose\n    - ivtt\n    - ovtt\n    - tottime\n    - totcost\n    - hhid\n    - perid\n'
		'    - numalts\n    - dist\n    - wkzone\n    - hmzone\n    - rspopden\n    - rsempden\n'
		'    - wkpopden\n    - wkempden\n    - vehavdum\n    - femdum\n    - age\n    - drlicdum\n'
		'    - noncadum\n    - numveh\n    - hhsize\n    - hhinc\n    - famtype\n    - hhowndum\n'
		'    - numemphh\n    - numadlt\n    - nmlt5\n    - nm5to11\n    - nm12to16\n    - wkccbd\n'
		'    - wknccbd\n    - corredis\n    - vehbywrk\n    - vocc\n    - wgt\n'
		'  data_co: <not populated>\n  data_av: <populated>\n')

	assert not ds.computational
	assert not ds.is_computational_ready()
	ds.computational = True
	assert ds.is_computational_ready()
	assert ds.computational
	s = io.StringIO()

	ds.info(out=s)

	assert s.getvalue() == (
		'larch.DataFrames:\n'
		'  n_cases: 5029\n'
		'  n_alts: 6\n'
		'  data_ce: 36 variables, 22033 rows\n'
		'  data_co: <not populated>\n'
		'  data_av: <populated>\n')

def test_service_idco():

	df = pandas.read_csv(example_file("MTCwork.csv.gz"))
	df.set_index(['casenum', 'altnum'], inplace=True)

	dfs = DataFrames(df, crack=True)
	check1 = dfs.make_idco('1')
	assert (check1 == 1).shape == (5029, 1)
	assert numpy.all(check1 == 1)

	check2 = dfs.make_idco('age')
	assert check2.shape == (5029, 1)
	assert numpy.all(check2.head(5) == [35, 40, 28, 34, 43])
	assert numpy.all(check2.tail(5) == [58, 33, 34, 35, 37])

	check3 = dfs.make_idco('age', '1')
	assert check3.shape == (5029, 2)
