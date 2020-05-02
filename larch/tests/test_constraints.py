import larch
import numpy as np
import pandas as pd

from larch import P, X
from larch.model.constraints import RatioBound, OrderingBound, FixedBound
from larch.model.constraints import interpret_contraint
from pytest import approx
import pytest
import sys

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

	assert r1['loglike'] == approx(-3764.1459333329217, rel=1e-4)
	assert r1.message == 'Optimization terminated successfully.'
	assert m1['tottime'].value / m1['totcost'].value == approx(1.0, rel=1e-3)
	assert c1.fun(m1.pf.value) == approx(0, abs=1e-5)

	# m0.calculate_parameter_covariance()
	m1.calculate_parameter_covariance()

	# assert m0.pf.loc['tottime', 'std_err'] == approx(0.000247, rel=5e-3)
	assert m1.pf.loc['tottime', 'std_err'] == approx(0.000247, rel=5e-3)
	assert m1.pf.loc['totcost', 'std_err'] == approx(0.000247, rel=5e-3)

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

	# assert m0.pf.loc['tottime', 'std_err'] == approx(0.000248, rel=5e-3)
	assert m1.pf.loc['tottime', 'std_err'] == approx(0.000248*2, rel=5e-3)
	assert m1.pf.loc['totcost', 'std_err'] == approx(0.000248, rel=5e-3)


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

	# assert m0.pf.loc['hhinc#23', 'std_err'] == approx(0.001397, rel=5e-3)
	assert m1.pf.loc['hhinc#2', 'std_err'] == approx(0.001397, rel=5e-3)
	assert m1.pf.loc['hhinc#3', 'std_err'] == approx(0.001397, rel=5e-3)


def test_lower_bound():

	m1 = larch.Model.Example(1)
	m1.set_value('ASC_TRAN', minimum=0)
	m1.load_data()

	# constraint tests are unstable across platforms
	# r1=m1.maximize_loglike(
	# 	method='slsqp',
	# 	options={'ftol': 1e-09},
	# 	quiet=True,
	# )
	# assert r1.message == 'Optimization terminated successfully.'
	m1.set_values({'ASC_BIKE': -2.1362348000501323,
				   'ASC_SR2': -2.041879053530548,
				   'ASC_SR3P': -3.568789444368573,
				   'ASC_TRAN': 3.7800630376890545e-12,
				   'ASC_WALK': 0.2099970488220958,
				   'hhinc#2': -0.003442043141464587,
				   'hhinc#3': -0.0010585520966694998,
				   'hhinc#4': -0.012338038801568696,
				   'hhinc#5': -0.014560216659416916,
				   'hhinc#6': -0.012146634316726645,
				   'totcost': -0.00489380761583881,
				   'tottime': -0.059272247433483374})

	m1.calculate_parameter_covariance()

	assert m1.loglike() == approx(-3639.0010819576023)

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
	}, abs=1e-10, rel=1e-2)

	se_ = {
		'ASC_BIKE': 0.30102780156769227,
		'ASC_SR2': 0.10147191931629226,
		'ASC_SR3P': 0.17587511190420965,
		'ASC_TRAN': np.nan,
		'ASC_WALK': 0.176809227982435,
		'hhinc#2': 0.0015471476930952455,
		'hhinc#3': 0.002545194004687296,
		'hhinc#4': 0.0012854684301876374,
		'hhinc#5': 0.005332349790044528,
		'hhinc#6': 0.0030443866859315343,
		'totcost': 0.00023967125878600402,
		'tottime': 0.0027703331869058825,
	}
	assert dict(m1.pf['std_err']) == approx(se_, abs=1e-10, rel=1e-2, nan_ok=True)

	assert dict(m1.pf['unconstrained_std_err']) == approx({
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
	}, rel=1e-2)


@pytest.mark.skip(reason="constraint tests are unstable across platforms")
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
		options={'ftol': 1e-12},
		quiet=True,
	)

	assert r1.message == 'Optimization terminated successfully.'

	#m0.calculate_parameter_covariance()
	m1.calculate_parameter_covariance()

	assert r1.loglike == approx(-3725.072026618359)

	# assert dict(m1.pf['value']) == approx({
	# 	'ASC_BIKE': -1.832639738018494,
	# 	'ASC_SR2': -1.975601900695666,
	# 	'ASC_SR3P': -3.4822938489427426,
	# 	'ASC_TRAN': 0.23349082549768843,
	# 	'ASC_WALK': 1.1111842511724719,
	# 	'hhinc#2': -0.002212546400939385,
	# 	'hhinc#3': 0.00022218881480540761,
	# 	'hhinc#4': -0.0054968611164901825,
	# 	'hhinc#5': -0.012523465955589126,
	# 	'hhinc#6': -0.010794428135692613,
	# 	'totcost': -0.005088026048518891,
	# 	'tottime': -0.1
	# }, rel=1e-2)
	assert m1.pf.loc['tottime', 'value'] == approx(-0.1, rel=1e-3)

	assert dict(m1.pf['std_err']) == approx({
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
	}, abs=1e-10, rel=1e-2)

	assert dict(m1.pf['unconstrained_std_err']) == approx({
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
	}, rel=1e-2)

@pytest.mark.skip(reason="constraint tests are unstable across platforms")
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
	# assert r1.message == 'Positive directional derivative for linesearch'
	m1.constraints.rescale(0.1)

	r1 = m1.maximize_loglike(
		method='slsqp',
		options={'ftol': 1e-09},
		quiet=True,
	)
	# assert r1.message == 'Positive directional derivative for linesearch'

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
	}, rel=1e-2)

	# Problem: Travis is failing on this test, giving
	#    tottime as NaN.  Disabling until we find a
	#    similar problem we can test and diagnose
	assert dict(m1.pf['std_err']) == approx({
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
	}, abs=1e-10, rel=5e-2)

	assert dict(m1.pf['unconstrained_std_err']) == approx({
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
	}, rel=5e-2)

def test_overspec():

	m0 = larch.Model.Example(1)
	m0.utility_ca = m0.utility_ca + P.failpar * X('1')
	m0.utility_co[1] = P.ASC_DA
	m0.lock_value('tottime', -0.1)
	m0.utility_co[2] = P.ASC_SR2 + P('hhinc#23') * X.hhinc
	m0.utility_co[3] = P.ASC_SR3P + P('hhinc#23') * X.hhinc
	m0.remove_unused_parameters()
	m0.load_data()
	# constraint tests are unstable across platforms
	# r0 = m0.maximize_loglike(
	# 	quiet=True,
	# )
	m0.set_values({
		'ASC_BIKE': -0.8550063261138748,
		'ASC_DA': 0.9780172816142935,
		'ASC_SR2': -1.0303087193826583,
		'ASC_SR3P': -2.394702207497934,
		'ASC_TRAN': 1.2134607482035888,
		'ASC_WALK': 2.0885392231767055,
		'failpar': -1.2138930556454395e-14,
		'hhinc#23': -0.001647452425832848,
		'hhinc#4': -0.005545798283823439,
		'hhinc#5': -0.012530050562373019,
		'hhinc#6': -0.010792561322141715,
		'totcost': -0.005093524162084949,
		'tottime': -0.1,
	})
	m0.calculate_parameter_covariance()
	possover = m0.possible_overspecification
	assert possover.data.shape == (7, 2)
	assert all(
		possover.data.index.sort_values() == [
			'ASC_BIKE', 'ASC_DA', 'ASC_SR2', 'ASC_SR3P',
			'ASC_TRAN', 'ASC_WALK', 'failpar',
		])

def test_parameter_summary():
	import larch
	from larch.model.constraints import RatioBound, OrderingBound, FixedBound

	m0 = larch.Model.Example(1)
	m0.lock_value('tottime', -0.05)
	m0.set_value('totcost', maximum=-0.005, minimum=-0.02)
	m0.constraints.append(OrderingBound("hhinc#3 <= hhinc#2"))
	m0.load_data()

	# constraint tests are unstable across platforms
	# r0 = m0.maximize_loglike(
	# 	method='slsqp',
	# 	options={'ftol': 1e-09},
	# 	quiet=True,
	# )
	m0.set_values({
		'ASC_BIKE': -2.401574701017008,
		'ASC_SR2': -2.2234574452767037,
		'ASC_SR3P': -3.630001719635557,
		'ASC_TRAN': -0.7000998451196682,
		'ASC_WALK': -0.25495834390295924,
		'hhinc#2': -0.0015948074747861667,
		'hhinc#3': -0.0015948074747729564,
		'hhinc#4': -0.005385664916039662,
		'hhinc#5': -0.01284021327300144,
		'hhinc#6': -0.009658848920781143,
		'totcost': -0.005000000000003634,
		'tottime': -0.05,
	})

	m0.calculate_parameter_covariance()

	ps = m0.parameter_summary('df').data
	stable_df(ps, 'parameter_summary_test')

	try:
		assert ps.loc[('LOS', 'totcost'), 'Value'] == "-0.00500"
		assert ps.loc[('LOS', 'tottime'), 'Value'] == "-0.0500"
		assert ps.loc[('Income', 'hhinc#2'), 'Value'] == "-0.00159"
		assert ps.loc[('Income', 'hhinc#3'), 'Value'] == "-0.00159"

		assert ps.loc[('LOS', 'totcost'), 'Std Err'] == " NA"
		assert ps.loc[('LOS', 'tottime'), 'Std Err'] == " NA"
		assert ps.loc[('Income', 'hhinc#2'), 'Std Err'] == " 0.00140"
		assert ps.loc[('Income', 'hhinc#3'), 'Std Err'] == " 0.00140"

		assert ps.loc[('LOS', 'totcost'), 't Stat'] == " NA"
		assert ps.loc[('LOS', 'tottime'), 't Stat'] == " NA"
		assert ps.loc[('Income', 'hhinc#2'), 't Stat'] == "-1.14"
		assert ps.loc[('Income', 'hhinc#3'), 't Stat'] == "-1.14"

		assert ps.loc[('LOS', 'totcost'), 'Signif'] == "[***]"
		assert ps.loc[('LOS', 'tottime'), 'Signif'] == ""
		assert ps.loc[('Income', 'hhinc#2'), 'Signif'] == ""
		assert ps.loc[('Income', 'hhinc#3'), 'Signif'] == ""

		assert ps.loc[('LOS', 'totcost'), 'Constrained'] == "totcost ≤ -0.005"
		assert ps.loc[('LOS', 'tottime'), 'Constrained'] == "fixed value"
		assert ps.loc[('Income', 'hhinc#2'), 'Constrained'] == "hhinc#3 ≤ hhinc#2"
		assert ps.loc[('Income', 'hhinc#3'), 'Constrained'] == "hhinc#3 ≤ hhinc#2"
	except:
		print(ps.iloc[:,:3])
		print(ps.iloc[:,3:])
		raise

def test_contraint_interpretation():

	assert interpret_contraint("aaaa > bbbb") == OrderingBound('bbbb', 'aaaa')
	assert interpret_contraint("aaaa < bbbb") == OrderingBound('aaaa', 'bbbb')
	assert interpret_contraint("aaaa/bbbb > 3") == RatioBound('aaaa', 'bbbb', min_ratio=3)
	assert interpret_contraint("aaaa/bbbb < 3") == RatioBound('aaaa', 'bbbb', max_ratio=3)
	assert interpret_contraint("aaaa / bbbb > 3") == RatioBound('aaaa', 'bbbb', min_ratio=3)
	assert interpret_contraint("aaaa / bbbb < 3") == RatioBound('aaaa', 'bbbb', max_ratio=3)
	assert interpret_contraint("aaaa / bbbb >= 3") == RatioBound('aaaa', 'bbbb', min_ratio=3)
	assert interpret_contraint("aaaa / bbbb <= 3") == RatioBound('aaaa', 'bbbb', max_ratio=3)
	assert interpret_contraint("3.1 < aaaa / bbbb <= 3.2") == RatioBound('aaaa', 'bbbb', min_ratio=3.1, max_ratio=3.2)
	assert interpret_contraint("aaaa > 3") == FixedBound('aaaa', minimum=3)
	assert interpret_contraint("aaaa < 3") == FixedBound('aaaa', maximum=3)
	assert interpret_contraint("3.1 < aaaa < 3.5") == FixedBound('aaaa', minimum=3.1, maximum=3.5)

	assert interpret_contraint("aaaa/bbbb > 1/2") == RatioBound('aaaa', 'bbbb', min_ratio=1/2)
	assert interpret_contraint("aaaa/bbbb < 1/2") == RatioBound('aaaa', 'bbbb', max_ratio=1/2)
	assert interpret_contraint("aaaa / bbbb > 1/2") == RatioBound('aaaa', 'bbbb', min_ratio=1/2)
	assert interpret_contraint("aaaa / bbbb < 1/2") == RatioBound('aaaa', 'bbbb', max_ratio=1/2)
	assert interpret_contraint("aaaa / bbbb >= 1/2") == RatioBound('aaaa', 'bbbb', min_ratio=1/2)
	assert interpret_contraint("aaaa / bbbb <= 1/2") == RatioBound('aaaa', 'bbbb', max_ratio=1/2)
	assert interpret_contraint("1/4 < aaaa / bbbb <= 3/4") == RatioBound('aaaa', 'bbbb', min_ratio=1/4, max_ratio=3/4)
	assert interpret_contraint("aaaa > 3/4") == FixedBound('aaaa', minimum=3/4)
	assert interpret_contraint("aaaa < 3/4") == FixedBound('aaaa', maximum=3/4)
	assert interpret_contraint("1/4 < aaaa < 3/4") == FixedBound('aaaa', minimum=1/4, maximum=3/4)

from . import stable_df

def test_constraint_parameter_summary_ratios():

	m = larch.Model.Example(1)
	m.set_value('totcost', minimum=-0.001)
	m.set_value('hhinc#3', minimum=0.0025)
	m.set_value('hhinc#5', minimum=-0.008)
	m.lock_value('hhinc#6', value=-0.009)
	m.load_data()
	m.set_values({
		'ASC_BIKE': -2.203264336154524,
		'ASC_SR2': -1.8913348121897215,
		'ASC_SR3P': -3.2288258012739846,
		'ASC_TRAN': -0.520325914514216,
		'ASC_WALK': 0.12488474492871472,
		'hhinc#2': -0.0010437239183118503,
		'hhinc#3': 0.002500000762078847,
		'hhinc#4': -0.0017845440559583583,
		'hhinc#5': -0.007999992934338814,
		'hhinc#6': -0.009,
		'totcost': -0.0009999993257563983,
		'tottime': -0.052413890729296815,
	})
	m.calculate_parameter_covariance()
	stable_df(m.parameter_summary().data, 'like_ratio_test')
