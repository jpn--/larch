import larch
import numpy as np
import pandas as pd

from larch import P, X
from larch.model.constraints import RatioBound, OrderingBound
from pytest import approx

import larch.torch

def test_max_ratio_1():

	#m0 = larch.Model.Example(1)
	m1 = larch.Model.Example(1)

	#m0.utility_ca = P.tottime * X.tottime + P.tottime * X.totcost
	#m0.remove_unused_parameters()

	#m0.set_value(P.tottime, maximum=0)
	m1.set_value(P.totcost, maximum=0)

	c1 = RatioBound(m1, P.tottime, P.totcost, min_ratio=0.75, max_ratio=1.0, scale=1)

	#m0.load_data()
	m1.load_data()

	m1.constraints = [c1,]

	# r0 = m0.maximize_loglike(
	# 	method='slsqp',
	# )
	r1 = m1.maximize_loglike(
		method='slsqp',
	)

	assert r1['loglike'] == approx(-3764.1459333329217)
	assert r1.message == 'Optimization terminated successfully.'
	assert m1['tottime'].value / m1['totcost'].value == approx(1.0, rel=1e-3)
	assert c1.fun(m1.pf.value) == approx(0, abs=1e-5)

	# m0.calculate_parameter_covariance()
	m1.calculate_parameter_covariance()

	# assert m0.pf.loc['tottime', 'std err'] == approx(0.000247, rel=5e-3)
	assert m1.pf.loc['tottime', 'constrained std err'] == approx(0.000247, rel=5e-3)
	assert m1.pf.loc['totcost', 'constrained std err'] == approx(0.000247, rel=5e-3)

def test_max_ratio_2():

	# m0 = larch.Model.Example(1)
	m1 = larch.Model.Example(1)

	# m0.utility_ca = P.tottime * (X.tottime * 2) + P.tottime * X.totcost
	# m0.remove_unused_parameters()

	# m0.set_value(P.tottime, maximum=0)
	m1.set_value(P.totcost, maximum=0)

	c1 = RatioBound(m1, P.tottime, P.totcost, min_ratio=0.75, max_ratio=2.0, scale=1)

	# m0.load_data()
	m1.load_data()

	m1.constraints = [c1,]

	# r0 = m0.maximize_loglike(
	# 	method='slsqp',
	# )
	r1 = m1.maximize_loglike(
		method='slsqp',
	)

	assert m1['tottime'].value / m1['totcost'].value == approx(2.0, rel=1e-3)

	assert r1['loglike'] == approx(-3730.8942681224476)

	assert r1.message == 'Optimization terminated successfully.'
	assert c1.fun(m1.pf.value) == approx(0, abs=1e-5)

	# m0.calculate_parameter_covariance()
	m1.calculate_parameter_covariance()

	# assert m0.pf.loc['tottime', 'std err'] == approx(0.000248, rel=5e-3)
	assert m1.pf.loc['tottime', 'constrained std err'] == approx(0.000248*2, rel=5e-3)
	assert m1.pf.loc['totcost', 'constrained std err'] == approx(0.000248, rel=5e-3)


def test_constraint_ordering_1():
	#m0 = larch.Model.Example(1)
	m1 = larch.Model.Example(1)

	# m0.utility_co[2] = P.ASC_SR2 + P('hhinc#23') * X.hhinc
	# m0.utility_co[3] = P.ASC_SR3P + P('hhinc#23') * X.hhinc
	# m0.remove_unused_parameters()

	c3 = OrderingBound(m1, "hhinc#3", "hhinc#2")

	# m0.load_data()
	m1.load_data()

	m1.constraints = [c3,]

	# r0 = m0.maximize_loglike(
	# 	method='slsqp',
	# )

	r1 = m1.maximize_loglike(
		method='slsqp',
	)

	assert m1['hhinc#2'].value / m1['hhinc#3'].value == approx(1.0, rel=1e-3)
	assert r1.message == 'Optimization terminated successfully.'
	assert c3.fun(m1.pf.value) == approx(0, abs=1e-5)

	# m0.calculate_parameter_covariance()
	m1.calculate_parameter_covariance()

	# assert m0.pf.loc['hhinc#23', 'std err'] == approx(0.001397, rel=5e-3)
	assert m1.pf.loc['hhinc#2', 'constrained std err'] == approx(0.001397, rel=5e-3)
	assert m1.pf.loc['hhinc#3', 'constrained std err'] == approx(0.001397, rel=5e-3)

