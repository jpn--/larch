from ..examples import example
from .. import Model
from ..roles import PX
from pytest import approx

def test_301():

	d = example(300, 'd')

	m = Model(dataservice=d)

	v = [
		"timeperiod==2",
		"timeperiod==3",
		"timeperiod==4",
		"timeperiod==5",
		"timeperiod==6",
		"timeperiod==7",
		"timeperiod==8",
		"timeperiod==9",
		"carrier==2",
		"carrier==3",
		"carrier==4",
		"carrier==5",
		"equipment==2",
		"fare_hy",
		"fare_ly",
		"elapsed_time",
		"nb_cnxs",
	]

	m.utility_ca = sum(PX(i) for i in v)
	m.choice_ca_var = 'choice'
	m.load_data()

	result = m.maximize_loglike()

	assert result.loglike == approx(-777770.0688722526)
	assert result.x['carrier==2'] == approx(0.11720047917232307)
	assert result.logloss == approx(3.306873650593341)
