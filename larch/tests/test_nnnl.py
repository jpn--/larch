from ..model import Model
import numpy
from pytest import approx
import pandas
from ..data_services.examples import ITINERARY_RAW
from ..data_services import H5PodCE
from .. import DataService, Model
from ..util.common_functions import fourier_series, piecewise_linear

import pytest

@pytest.mark.skip(reason="NNNL is not functioning correctly presently")
def test_2_tier():
	df = ITINERARY_RAW()

	df['outDepTimeHours'] = numpy.round(df['outDepTime'] / 3600).astype(int)

	pod_ce = H5PodCE.from_dataframe(
		df,
		caselabels='individual',
		altlabels='alternative',
	)

	_, sa = pod_ce.new_alternative_codes(
		groupby=[
			# 'nFlights',
			'outDepTimeHours',
		],
		name='AltsByFlightsDepHours',
		create_index='AltsByFlightsDepHours_idx',
		groupby_prefixes=[
			# "F",
			"H",
		],
		overwrite=True,
	)

	dx = DataService.from_CE_and_SA(pod_ce, sa)


	m = Model(dataservice=dx)

	from .. import P,X,PX

	m.utility_ca = (
			+ sum(PX(f'normalize({i})') for i in [
				'totalTripDurationMinutes',
				'log(totalPrice)',
				'nFlights',
			])
			+ fourier_series("outDepTime/86400", length=4)
			+ fourier_series("outArrTime/86400", length=4)
	)
	m.availability_var = 'is_avail'
	m.choice_ca_var = 'choice'
	m.magic_nesting(sa)
	m.unmangle()
	m.estimate()

	assert m.loglike() == approx(-1552.7201587407562)

	from larch.model.nnnl import NNNL
	n = NNNL(m)
	n.finalize()
	n.load_data()

	n.set_values(m.pf.value / m.pf.value['MU_outDepTimeHours'])
	n.set_value('MU_outDepTimeHours', m.pf.value['MU_outDepTimeHours'])

	assert n.loglike(n.pf.value) == approx( m.loglike(m.pf.value) )

	numpy.testing.assert_array_almost_equal(
		n.d_logsums(),
		m.d_logsums()
	)




@pytest.mark.skip(reason="NNNL is not functioning correctly presently")
def test_3_tier():
	df = ITINERARY_RAW()

	df['outDepTimeHours'] = numpy.round(df['outDepTime'] / 3600).astype(int)

	pod_ce = H5PodCE.from_dataframe(
		df,
		caselabels='individual',
		altlabels='alternative',
	)

	_, sa = pod_ce.new_alternative_codes(
		groupby=[
			'nFlights',
			'outDepTimeHours',
		],
		name='AltsByFlightsDepHours',
		create_index='AltsByFlightsDepHours_idx',
		groupby_prefixes=[
			"F",
			"H",
		],
		overwrite=True,
	)

	dx = DataService.from_CE_and_SA(pod_ce, sa)


	m = Model(dataservice=dx)

	from .. import P,X,PX

	m.utility_ca = (
			+ sum(PX(f'normalize({i})') for i in [
				'totalTripDurationMinutes',
				'log(totalPrice)',
				'nFlights',
			])
			+ fourier_series("outDepTime/86400", length=4)
			+ fourier_series("outArrTime/86400", length=4)
	)
	m.availability_var = 'is_avail'
	m.choice_ca_var = 'choice'
	m.magic_nesting(sa)
	m.unmangle()

	m.set_values({
		'MU_nFlights': 0.75,
		'MU_outDepTimeHours': 0.5,
		'cos_2πoutArrTime/86400': -0.24352756182332694,
		'cos_2πoutDepTime/86400': -0.5941495103991815,
		'cos_4πoutArrTime/86400': 0.07133782207978602,
		'cos_4πoutDepTime/86400': -0.45117462119982826,
		'normalize(log(totalPrice))': -1.225219579598701,
		'normalize(nFlights)': -2.3288120787774154,
		'normalize(totalTripDurationMinutes)': -0.44072158058922295,
		'sin_2πoutArrTime/86400': -1.0494189248239782,
		'sin_2πoutDepTime/86400': 0.06320548375577714,
		'sin_4πoutArrTime/86400': -0.4007737199632005,
		'sin_4πoutDepTime/86400': -0.15198982460097238,
	})

	m.load_data()

	assert approx(-1555.1201442643503) ==  m.loglike()

	from larch.model.nnnl import NNNL
	n = NNNL(m)
	n.finalize()
	n.load_data()

	n.set_values(m.pf.value / m.pf.value['MU_outDepTimeHours'])
	n.set_value('MU_outDepTimeHours', m.pf.value['MU_outDepTimeHours'] / m.pf.value['MU_nFlights'])
	n.set_value('MU_nFlights', m.pf.value['MU_nFlights'])

	assert n.loglike(n.pf.value) == approx( m.loglike(m.pf.value) )

	numpy.testing.assert_array_almost_equal(
		n.d_logsums(),
		m.d_logsums()
	)

