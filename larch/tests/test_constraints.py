import larch
import numpy as np
import pandas as pd

from larch import P, X
from larch.model.constraints import RatioBound, OrderingBound
from pytest import approx

def test_max_ratio_1():

	#m0 = larch.Model.Example(1)
	m1 = larch.Model.Example(1)

	#m0.utility_ca = P.tottime * X.tottime + P.tottime * X.totcost
	#m0.remove_unused_parameters()

	#m0.set_value(P.tottime, maximum=0)
	m1.set_value(P.totcost, maximum=0)

	c1 = RatioBound(P.tottime, P.totcost, min_ratio=0.75, max_ratio=1.0, scale=1)

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
	assert m1.pf.loc['tottime', 'std err'] == approx(0.000247, rel=5e-3)
	assert m1.pf.loc['totcost', 'std err'] == approx(0.000247, rel=5e-3)

def test_max_ratio_2():

	# m0 = larch.Model.Example(1)
	m1 = larch.Model.Example(1)

	# m0.utility_ca = P.tottime * (X.tottime * 2) + P.tottime * X.totcost
	# m0.remove_unused_parameters()

	# m0.set_value(P.tottime, maximum=0)
	m1.set_value(P.totcost, maximum=0)

	c1 = RatioBound(P.tottime, P.totcost, min_ratio=0.75, max_ratio=2.0, scale=1)

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
	assert m1.pf.loc['tottime', 'std err'] == approx(0.000248*2, rel=5e-3)
	assert m1.pf.loc['totcost', 'std err'] == approx(0.000248, rel=5e-3)


def test_constraint_ordering_1():
	#m0 = larch.Model.Example(1)
	m1 = larch.Model.Example(1)

	# m0.utility_co[2] = P.ASC_SR2 + P('hhinc#23') * X.hhinc
	# m0.utility_co[3] = P.ASC_SR3P + P('hhinc#23') * X.hhinc
	# m0.remove_unused_parameters()

	c3 = OrderingBound("hhinc#3", "hhinc#2")

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
	assert m1.pf.loc['hhinc#2', 'std err'] == approx(0.001397, rel=5e-3)
	assert m1.pf.loc['hhinc#3', 'std err'] == approx(0.001397, rel=5e-3)


def test_lower_bound():

	#m0 = larch.Model.Example(1)
	m1 = larch.Model.Example(1)

	#m0.lock_value('ASC_TRAN', 0)
	m1.set_value('ASC_TRAN', minimum=0)

	#m0.load_data()
	m1.load_data()

	# r0=m0.torch.maximize_loglike(
	# 	method='slsqp',
	# 	options={'ftol': 1e-09},
	# )

	r1=m1.maximize_loglike(
		method='slsqp',
		options={'ftol': 1e-09},
		quiet=True,
	)

	assert r1.message == 'Optimization terminated successfully.'

	# m0.calculate_parameter_covariance()
	m1.calculate_parameter_covariance()

	assert r1.loglike == approx(-3639.0010819576023)

	assert dict(m1.pf['value']) == approx({
		'ASC_BIKE': -2.136234299750425,
		'ASC_SR2': -2.041878696832068,
		'ASC_SR3P': -3.568790119252386,
		'ASC_TRAN': 0.0,
		'ASC_WALK': 0.20999688402874506,
		'hhinc#2': -0.0034420525424271203,
		'hhinc#3': -0.001058543967530453,
		'hhinc#4': -0.012338043439238819,
		'hhinc#5': -0.014560247040617383,
		'hhinc#6': -0.012146632951225103,
		'totcost': -0.0048938088864863595,
		'tottime': -0.059272225511784224,
	})

	assert dict(m1.pf['std err']) == approx({
		'ASC_BIKE': 0.30102780156769227,
		'ASC_SR2': 0.10147191931629226,
		'ASC_SR3P': 0.17587511190420965,
		'ASC_TRAN': 0.0,
		'ASC_WALK': 0.176809227982435,
		'hhinc#2': 0.0015471476930952455,
		'hhinc#3': 0.002545194004687296,
		'hhinc#4': 0.0012854684301876374,
		'hhinc#5': 0.005332349790044528,
		'hhinc#6': 0.0030443866859315343,
		'totcost': 0.00023967125878600402,
		'tottime': 0.0027703331869058825,
	})

	assert dict(m1.pf['unconstrained std err']) == approx({
		'ASC_BIKE': 0.3048352068287236,
		'ASC_SR2': 0.10486177423525046,
		'ASC_SR3P': 0.17846252159720813,
		'ASC_TRAN': 0.1330469233545672,
		'ASC_WALK': 0.19586113875694,
		'hhinc#2': 0.0015673042154757647,
		'hhinc#3': 0.002560982061812878,
		'hhinc#4': 0.001926809603173801,
		'hhinc#5': 0.005344633810524164,
		'hhinc#6': 0.0030865953487895412,
		'totcost': 0.00023967145587986189,
		'tottime': 0.0032194798668345615,
	})


def test_upper_bound():
	#m0 = larch.Model.Example(1)
	m1 = larch.Model.Example(1)

	#m0.lock_value('tottime', -0.1)
	m1.set_value('tottime', maximum=-0.1)

	#m0.load_data()
	m1.load_data()

	# r0 = m0.torch.maximize_loglike(
	# 	method='slsqp',
	# 	options={'ftol': 1e-09},
	# )

	r1 = m1.maximize_loglike(
		method='slsqp',
		options={'ftol': 1e-09},
		quiet=True,
	)

	assert r1.message == 'Optimization terminated successfully.'

	#m0.calculate_parameter_covariance()
	m1.calculate_parameter_covariance()

	assert r1.loglike == approx(-3725.072026618359)

	assert dict(m1.pf['value']) == approx({
		'ASC_BIKE': -1.8321554435700236,
		'ASC_SR2': -1.975716749548033,
		'ASC_SR3P': -3.4820713271093124,
		'ASC_TRAN': 0.23339260484283975,
		'ASC_WALK': 1.1107847788929406,
		'hhinc#2': -0.0022113178715119385,
		'hhinc#3': 0.0002181973410492235,
		'hhinc#4': -0.005496049691058452,
		'hhinc#5': -0.012532676196820506,
		'hhinc#6': -0.010789794806056898,
		'totcost': -0.005088177160006318,
		'tottime': -0.1
	})

	assert dict(m1.pf['std err']) == approx({
		'ASC_BIKE': 0.30013608145167614,
		'ASC_SR2': 0.10555671986388032,
		'ASC_SR3P': 0.17980894548116735,
		'ASC_TRAN': 0.12312298703458116,
		'ASC_WALK': 0.17807354839871625,
		'hhinc#2': 0.0015763567265900031,
		'hhinc#3': 0.0025749894562580172,
		'hhinc#4': 0.001920571529389678,
		'hhinc#5': 0.005249394394490925,
		'hhinc#6': 0.003146693697448889,
		'totcost': 0.00024654869488240555,
		'tottime': 0
	}, abs=1e-10)

	assert dict(m1.pf['unconstrained std err']) == approx({
		'ASC_BIKE': 0.30323766594813406,
		'ASC_SR2': 0.1069872808913486,
		'ASC_SR3P': 0.18097129736247558,
		'ASC_TRAN': 0.1439393314870008,
		'ASC_WALK': 0.20788197245193588,
		'hhinc#2': 0.0015763569302413402,
		'hhinc#3': 0.0025750088819569447,
		'hhinc#4': 0.001921038696332169,
		'hhinc#5': 0.005249394449971851,
		'hhinc#6': 0.0031492030054873824,
		'totcost': 0.0002472663647304817,
		'tottime': 0.004377145909550393,
	})

def test_multi_constraints():
	#m0 = larch.Model.Example(1)
	m1 = larch.Model.Example(1)

	#m0.lock_value('tottime', -0.1)
	m1.set_value('tottime', maximum=-0.1)

	#m0.utility_co[2] = P.ASC_SR2 + P('hhinc#23') * X.hhinc
	#m0.utility_co[3] = P.ASC_SR3P + P('hhinc#23') * X.hhinc
	#m0.remove_unused_parameters()

	m1.constraints.append(OrderingBound("hhinc#3 <= hhinc#2"))

	#m0.load_data()
	m1.load_data()

	# r0 = m0.torch.maximize_loglike(
	# 	method='slsqp',
	# 	options={'ftol': 1e-09},
	# )

	r1 = m1.maximize_loglike(
		method='slsqp',
		options={'ftol': 1e-09},
		quiet=True,
	)
	assert r1.message == 'Positive directional derivative for linesearch'
	m1.constraints.rescale(0.1)

	r1 = m1.maximize_loglike(
		method='slsqp',
		options={'ftol': 1e-09},
		quiet=True,
	)
	assert r1.message == 'Positive directional derivative for linesearch'

	m1.constraints.rescale(0.01)
	r1 = m1.maximize_loglike(
		method='slsqp',
		options={'ftol': 1e-09},
		quiet=True,
	)

	assert r1.loglike == approx(-3725.439832484451)

	assert r1.message == 'Optimization terminated successfully.'

	#m0.calculate_parameter_covariance()

	m1.calculate_parameter_covariance()

	assert dict(m1.pf['value']) == approx({
		'ASC_BIKE': -1.8336226490217296,
		'ASC_SR2': -2.008575006161787,
		'ASC_SR3P': -3.372494233637122,
		'ASC_TRAN': 0.23543257975862672,
		'ASC_WALK': 1.1106100812079671,
		'hhinc#2': -0.0016453210270738865,
		'hhinc#3': -0.001645187376205547,
		'hhinc#4': -0.0055451755658633,
		'hhinc#5': -0.012519484434344206,
		'hhinc#6': -0.01079425276590098,
		'totcost': -0.005093387068047046,
		'tottime': -0.1
	})

	assert dict(m1.pf['std err']) == approx({
		'ASC_BIKE': 0.30013699696981455,
		'ASC_SR2': 0.09856612504524623,
		'ASC_SR3P': 0.1250682262709841,
		'ASC_TRAN': 0.12311806485021345,
		'ASC_WALK': 0.17807446535212992,
		'hhinc#2': 0.0014236497001240312,
		'hhinc#3': 0.001423649700124031,
		'hhinc#4': 0.0019212691279916797,
		'hhinc#5': 0.005248506439174827,
		'hhinc#6': 0.003146916477051659,
		'totcost': 0.0002466030726673273,
		'tottime': 0
	}, abs=1e-10)

	assert dict(m1.pf['unconstrained std err']) == approx({
		'ASC_BIKE': 0.3032386370524189,
		'ASC_SR2': 0.10695557451038115,
		'ASC_SR3P': 0.18033743391454088,
		'ASC_TRAN': 0.1439964957540911,
		'ASC_WALK': 0.20789603304640025,
		'hhinc#2': 0.0015638823624232918,
		'hhinc#3': 0.002640071485204856,
		'hhinc#4': 0.00192260391409179,
		'hhinc#5': 0.0052485070222418155,
		'hhinc#6': 0.0031494462988719317,
		'totcost': 0.00024742269603570923,
		'tottime': 0.004377029904009681
	})

