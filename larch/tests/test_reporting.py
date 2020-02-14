#
import numpy
import pandas
import larch
from larch import Model, P, X
import larch.exampville
import os
from pytest import approx


def test_ch_av_summary_output():

	skims = larch.OMX(larch.exampville.files.skims, mode='r')
	hh = pandas.read_csv(larch.exampville.files.hh)
	pp = pandas.read_csv(larch.exampville.files.person)
	tour = pandas.read_csv(larch.exampville.files.tour)

	pp_col = ['PERSONID', 'HHID', 'HHIDX', 'AGE', 'WORKS',
       'N_WORK_TOURS', 'N_OTHER_TOURS', 'N_TOURS', 'N_TRIPS', 'N_TRIPS_HBW',
       'N_TRIPS_HBO', 'N_TRIPS_NHB']

	raw = tour.merge(hh, on='HHID').merge(pp[pp_col], on=('HHID', 'PERSONID'))
	raw["HOMETAZi"] = raw["HOMETAZ"] - 1
	raw["DTAZi"] = raw["DTAZ"] - 1

	raw = raw[raw.TOURPURP == 1]

	f_tour = raw.join(
		skims.get_rc_dataframe(
			raw.HOMETAZi, raw.DTAZi,
		)
	)

	DA = 1
	SR = 2
	Walk = 3
	Bike = 4
	Transit = 5

	dfs = larch.DataFrames(
		co=f_tour,
		alt_codes=[DA, SR, Walk, Bike, Transit],
		alt_names=['DA', 'SR', 'Walk', 'Bike', 'Transit'],
	)

	m = larch.Model(dataservice=dfs)
	m.title = "Exampville Work Tour Mode Choice v1"

	m.utility_co[DA] = (
			+ P.InVehTime * X.AUTO_TIME
			+ P.Cost * X.AUTO_COST  # dollars per mile
	)

	m.utility_co[SR] = (
			+ P.ASC_SR
			+ P.InVehTime * X.AUTO_TIME
			+ P.Cost * (X.AUTO_COST * 0.5)  # dollars per mile, half share
			+ P("HighInc:SR") * X("INCOME>75000")
	)

	m.utility_co[Walk] = (
			+ P.ASC_Walk
			+ P.NonMotorTime * X.WALK_TIME
			+ P("HighInc:Walk") * X("INCOME>75000")
	)

	m.utility_co[Bike] = (
			+ P.ASC_Bike
			+ P.NonMotorTime * X.BIKE_TIME
			+ P("HighInc:Bike") * X("INCOME>75000")
	)

	m.utility_co[Transit] = (
			+ P.ASC_Transit
			+ P.InVehTime * X.TRANSIT_IVTT
			+ P.OutVehTime * X.TRANSIT_OVTT
			+ P.Cost * X.TRANSIT_FARE
			+ P("HighInc:Transit") * X("INCOME>75000")
	)

	# No choice or avail data set
	m.load_data()
	q = m.dataframes.choice_avail_summary()
	assert numpy.array_equal(q.columns, ['name', 'chosen', 'available'])
	assert q.index.identical(pandas.Index([1, 2, 3, 4, 5, '< Total All Alternatives >'], dtype='object'))
	assert numpy.array_equal(q.values, [
		['DA', None, None],
		['SR', None, None],
		['Walk', None, None],
		['Bike', None, None],
		['Transit', None, None],
		['', 0, ''],
	])

	# Reasonable choice and avail data set
	m.choice_co_code = 'TOURMODE'
	m.availability_co_vars = {
		DA: 'AGE >= 16',
		SR: '1',
		Walk: 'WALK_TIME < 60',
		Bike: 'BIKE_TIME < 60',
		Transit: 'TRANSIT_FARE>0',
	}
	m.load_data()
	q = m.dataframes.choice_avail_summary()
	assert numpy.array_equal(q.columns, ['name', 'chosen', 'available'])
	assert q.index.identical(pandas.Index([1, 2, 3, 4, 5, '< Total All Alternatives >'], dtype='object'))
	assert numpy.array_equal(q['name'].values, ['DA', 'SR', 'Walk', 'Bike', 'Transit', ''])
	assert numpy.array_equal(q['chosen'].values, [6052.,  810.,  196.,   72.,  434., 7564.])
	assert numpy.array_equal(q['available'].values,
							 numpy.array([7564.0, 7564.0, 4179.0, 7564.0, 4199.0, ''], dtype=object))

	# Unreasonable choice and avail data set
	m.choice_co_code = 'TOURMODE'
	m.availability_co_vars = {
		DA: 'AGE >= 26',
		SR: '1',
		Walk: 'WALK_TIME < 60',
		Bike: 'BIKE_TIME < 60',
		Transit: 'TRANSIT_FARE>0',
	}
	m.load_data()
	q = m.dataframes.choice_avail_summary()
	assert numpy.array_equal(q.columns, ['name', 'chosen', 'available', 'chosen but not available'])
	assert q.index.identical(pandas.Index([1, 2, 3, 4, 5, '< Total All Alternatives >'], dtype='object'))
	assert numpy.array_equal(q['name'].values, ['DA', 'SR', 'Walk', 'Bike', 'Transit', ''])
	assert numpy.array_equal(q['chosen'].values, [6052.,  810.,  196.,   72.,  434., 7564.])
	assert numpy.array_equal(q['available'].values,
							 numpy.array([6376.0, 7564.0, 4179.0, 7564.0, 4199.0, ''], dtype=object))
	assert numpy.array_equal(q['chosen but not available'].values, [942.0, 0.0, 0.0, 0.0, 0.0, 942.0])



def test_excel_metadata():

	import larch.util.excel
	if larch.util.excel.xlsxwriter is not None:
		from larch.util.excel import _make_excel_writer

		from larch.util.temporaryfile import TemporaryDirectory

		tempdir = TemporaryDirectory()
		os.path.join(tempdir, 'larchtest.xlsx')

		m = larch.example(1)
		m.load_data()
		m.loglike_null()
		m.maximize_loglike()

		xl = _make_excel_writer(m, os.path.join(tempdir, 'larchtest.xlsx'), save_now=False)
		xl.add_metadata('self', m)
		xl.add_metadata('short', 123)
		xl.add_metadata('plain', 'text')
		xl.save()

		md = larch.util.excel.load_metadata(os.path.join(tempdir, 'larchtest.xlsx'))
		assert len(md) == 3
		assert isinstance(md['self'], larch.Model)
		assert md['short'] == 123
		assert md['plain'] == 'text'

		md_m = larch.util.excel.load_metadata(os.path.join(tempdir, 'larchtest.xlsx'), 'self')
		assert isinstance(md_m, larch.Model)

		assert larch.util.excel.load_metadata(os.path.join(tempdir, 'larchtest.xlsx'), 'short') == 123

		from numpy import inf
		assert md_m._get_cached_loglike_values() == approx(
			{'nil': 0.0, 'null': -7309.600971749679, 'constants_only': 0.0, 'best': -inf})

		md_m.estimation_statistics()


def test_parameter_summary():

	m = larch.example(1)
	m.load_data()
	m.loglike_null()
	m.set_values(**{
		'ASC_BIKE': -2.3763275319243244,
		'ASC_SR2': -2.1780143286612037,
		'ASC_SR3P': -3.725078388760564,
		'ASC_TRAN': -0.6708609582690096,
		'ASC_WALK': -0.20677521181801753,
		'hhinc#2': -0.0021699381002406883,
		'hhinc#3': 0.0003577067151217295,
		'hhinc#4': -0.00528632366072714,
		'hhinc#5': -0.012807975284603574,
		'hhinc#6': -0.009686302933787567,
		'totcost': -0.00492023540098787,
		'tottime': -0.05134209452571549,
	})
	m.loglike()
	assert m.parameter_summary().tostring() == (
		'<div><table><thead><tr><th style="text-align: left;">Category</th><th style="text-align: left;">Parameter</th>'
		'<th>Value</th><th>Null Value</th></tr></thead><tbody><tr><th rowspan="2" style="vertical-align: top; text-alig'
		'n: left;">LOS</th><th style="vertical-align: top; text-align: left;">totcost</th><td>-0.00492</td><td>0.0</td>'
		'</tr><tr><th style="vertical-align: top; text-align: left;">tottime</th><td>-0.05134</td><td>0.0</td></tr><tr>'
		'<th rowspan="5" style="vertical-align: top; text-align: left;">ASCs</th><th style="vertical-align: top; text-a'
		'lign: left;">ASC_BIKE</th><td>-2.376</td><td>0.0</td></tr><tr><th style="vertical-align: top; text-align: left'
		';">ASC_SR2</th><td>-2.178</td><td>0.0</td></tr><tr><th style="vertical-align: top; text-align: left;">ASC_SR3P'
		'</th><td>-3.725</td><td>0.0</td></tr><tr><th style="vertical-align: top; text-align: left;">ASC_TRAN</th><td>-'
		'0.6709</td><td>0.0</td></tr><tr><th style="vertical-align: top; text-align: left;">ASC_WALK</th><td>-0.2068</t'
		'd><td>0.0</td></tr><tr><th rowspan="5" style="vertical-align: top; text-align: left;">Income</th><th style="ve'
		'rtical-align: top; text-align: left;">hhinc#2</th><td>-0.00217</td><td>0.0</td></tr><tr><th style="vertical-al'
		'ign: top; text-align: left;">hhinc#3</th><td>0.0003577</td><td>0.0</td></tr><tr><th style="vertical-align: top'
		'; text-align: left;">hhinc#4</th><td>-0.005286</td><td>0.0</td></tr><tr><th style="vertical-align: top; text-a'
		'lign: left;">hhinc#5</th><td>-0.01281</td><td>0.0</td></tr><tr><th style="vertical-align: top; text-align: lef'
		't;">hhinc#6</th><td>-0.009686</td><td>0.0</td></tr></tbody></table></div>')
	m.ordering = ()
	assert m.parameter_summary().tostring() == (
		'<div><table><thead><tr><th style="text-align: left;">Parameter</th><th>Value</th><th>Null Value</th></tr></the'
		'ad><tbody><tr><th style="vertical-align: top; text-align: left;">ASC_BIKE</th><td>-2.376</td><td>0.0</td></tr>'
		'<tr><th style="vertical-align: top; text-align: left;">ASC_SR2</th><td>-2.178</td><td>0.0</td></tr><tr><th sty'
		'le="vertical-align: top; text-align: left;">ASC_SR3P</th><td>-3.725</td><td>0.0</td></tr><tr><th style="vertic'
		'al-align: top; text-align: left;">ASC_TRAN</th><td>-0.6709</td><td>0.0</td></tr><tr><th style="vertical-align:'
		' top; text-align: left;">ASC_WALK</th><td>-0.2068</td><td>0.0</td></tr><tr><th style="vertical-align: top; tex'
		't-align: left;">hhinc#2</th><td>-0.00217</td><td>0.0</td></tr><tr><th style="vertical-align: top; text-align: '
		'left;">hhinc#3</th><td>0.0003577</td><td>0.0</td></tr><tr><th style="vertical-align: top; text-align: left;">h'
		'hinc#4</th><td>-0.005286</td><td>0.0</td></tr><tr><th style="vertical-align: top; text-align: left;">hhinc#5</'
		'th><td>-0.01281</td><td>0.0</td></tr><tr><th style="vertical-align: top; text-align: left;">hhinc#6</th><td>-0'
		'.009686</td><td>0.0</td></tr><tr><th style="vertical-align: top; text-align: left;">totcost</th><td>-0.00492</'
		'td><td>0.0</td></tr><tr><th style="vertical-align: top; text-align: left;">tottime</th><td>-0.05134</td><td>0.'
		'0</td></tr></tbody></table></div>')
