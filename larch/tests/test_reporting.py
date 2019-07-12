#
import numpy
import pandas
import larch
from larch import Model, P, X
import larch.exampville

def test_ch_av_summary_output():

	skims = larch.OMX(larch.exampville.files.skims, mode='r')
	hh = pandas.read_csv(larch.exampville.files.hh)
	pp = pandas.read_csv(larch.exampville.files.person)
	tour = pandas.read_csv(larch.exampville.files.tour)

	raw = tour.merge(hh, on='HHID').merge(pp, on=('HHID', 'PERSONID'))
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
	assert numpy.array_equal(q['chosen'].values, [3984., 570., 114., 47., 237., 4952.])
	assert numpy.array_equal(q['available'].values,
							 numpy.array([4952.0, 4952.0, 2789.0, 4952.0, 2651.0, ''], dtype=object))

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
	assert numpy.array_equal(q['chosen'].values, [3984., 570., 114., 47., 237., 4952.])
	assert numpy.array_equal(q['available'].values,
							 numpy.array([4077.0, 4952.0, 2789.0, 4952.0, 2651.0, ''], dtype=object))
	assert numpy.array_equal(q['chosen but not available'].values, [693.0, 0.0, 0.0, 0.0, 0.0, 693.0])

