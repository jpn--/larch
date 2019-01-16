import pandas
import os
import gzip
import io

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

