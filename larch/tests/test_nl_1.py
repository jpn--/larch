import pytest
from pytest import approx, raises
import larch
from .test_regressions import *

@pytest.fixture(scope="module")
def simple_nl():
	m = larch.example(1)
	mot = m.graph.new_node(parameter="MuSR", children=[2, 3])
	m.set_values(**{
		'ASC_BIKE': -2.369583417234141,
		'ASC_SR2': -2.1003772176440494,
		'ASC_SR3P': -3.1652310001168176,
		'ASC_TRAN': -0.6716514097035154,
		'ASC_WALK': -0.20573835594484602,
		'MuSR': 0.656152401524297,
		'hhinc#2': -0.0018493368570253318,
		'hhinc#3': -0.0005878635612891558,
		'hhinc#4': -0.0051673845801924615,
		'hhinc#5': -0.012777495307881156,
		'hhinc#6': -0.00967649032713631,
		'totcost': -0.004808452908429623,
		'tottime': -0.05107162036470516,
	})
	m.load_data()
	return m


def test_loglike(simple_nl):
	assert simple_nl.loglike() == approx(-3623.8414801535305)

def test_compute_utility_array(simple_nl, dataframe_regression):
	arr = simple_nl.utility()
	dataframe_regression.check(
		np.concatenate([
			arr[:3],
			arr[-3:],
		])
	)

def test_compute_probability_array(simple_nl, dataframe_regression):
	arr = simple_nl.probability()
	dataframe_regression.check(
		np.concatenate([
			arr[:3],
			arr[-3:],
		])
	)

def test_compute_exputility_array(simple_nl):
	with raises(NotImplementedError):
		simple_nl.exputility()

def test_compute_covariance(simple_nl, dataframe_regression):
	simple_nl.calculate_parameter_covariance()
	dataframe_regression.check(simple_nl.pf)
