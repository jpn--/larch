from larch.model import Model
import numpy
from pytest import approx
import pandas
import pytest

def qmnl_straw_man_model_1():

	from larch.roles import P,X

	altcodes = (1, 2, 3, 4, 5, 6)
	from larch.data_services.examples import MTC
	dt = MTC()

	p = Model(parameters=[], alts=altcodes, dataservice=dt, graph=None)


	from larch.roles import P,X

	p.utility_ca = (
		+ P('tottime') * X('tottime')
		+ P('totcost') * X('totcost')
	)

	p.utility_co = {
		2: (P('ASC#2') * X('1') + P('hhinc#2') * X('hhinc')),
		3: (P('ASC#3') * X('1') + P('hhinc#3') * X('hhinc')),
		4: (P('ASC#4') * X('1') + P('hhinc#4') * X('hhinc')),
		5: (P('ASC#5') * X('1') + P('hhinc#5') * X('hhinc')),
		6: (P('ASC#6') * X('1') + P('hhinc#6') * X('hhinc')),
	}

	p.quantity_ca = (
		+ P("FakeSizeAlt") * X('altnum+1')
		+ P("FakeSizeIvtt") * X('ivtt+1')
	)

	p.availability_var = '_avail_'
	p.choice_ca_var = '_choice_'

	p.load_data()
	return p


def test_qmnl():
	pq = qmnl_straw_man_model_1()
	assert( pq.loglike() == approx(-8486.55377320886))

	dll = pq.d_loglike()

	names = ['ASC#2', 'ASC#5', 'hhinc#3', 'ASC#4', 'ASC#6', 'hhinc#6', 'ASC#3',
       'hhinc#4', 'tottime', 'hhinc#2', 'hhinc#5', 'totcost', 'FakeSizeIvtt',
       'FakeSizeAlt']

	correct = pandas.Series(
		[-6.76598075e+02, -4.43123432e+02, -6.73124640e+04,
		 -4.91818432e+02, 3.97095316e+00, -1.38966274e+03,
		 -1.16626503e+03, -3.06932152e+04, -4.87328610e+04,
		 -4.02490548e+04, -2.72367637e+04, 1.45788603e+05,
		 8.69414603e+01, -8.69414603e+01],
			index = names)

	for n in names:
		assert (correct.loc[n] == approx(dll.loc[n]))

@pytest.mark.skip(reason="test is for manual use to check multithreading")
def test_saturate_cpu(howlong=15):
	pq = qmnl_straw_man_model_1()
	pq.initialize_graph(alternative_codes=[1,2,3,4,5,6])
	pq.graph.add_node(10, children=(5, 6), parameter='MU_nonmotor')
	pq.graph.add_node(11, children=(1, 2, 3), parameter='MU_car')
	assert( pq.loglike() == approx(-8486.55377320886))

	import time
	starttime = time.time()
	while time.time() < starttime + howlong:
		dll = pq.d_loglike()
