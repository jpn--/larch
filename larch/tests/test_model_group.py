import larch
import pandas as pd
from larch import P,X,PX
from pytest import approx
from larch.data_warehouse import example_file

def test_simple_model_group():

	df = pd.read_csv(example_file("MTCwork.csv.gz"))
	df.set_index(['casenum','altnum'], inplace=True)
	d = larch.DataFrames.from_idce(df, choice='chose', crack=True)
	d.set_alternative_names({
		1: 'DA',
		2: 'SR2',
		3: 'SR3+',
		4: 'Transit',
		5: 'Bike',
		6: 'Walk',
	})

	m0 = larch.Model(dataservice=d)
	m0.utility_co[2] = P("ASC_SR2")  + P("hhinc#2") * X("hhinc")
	m0.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m0.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m0.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m0.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m0.utility_ca = (
		(P("tottime_m")*X("tottime") + P("totcost_m")*X("totcost"))*X("femdum == 0")
		+
		(P("tottime_f")*X("tottime") + P("totcost_f")*X("totcost"))*X("femdum == 1")
	)

	m1 = larch.Model(dataservice=d.selector_co("femdum == 0"))
	m1.utility_co[2] = P("ASC_SR2")  + P("hhinc#2") * X("hhinc")
	m1.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m1.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m1.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m1.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m1.utility_ca = P("tottime_m")*X("tottime") + P("totcost_m")*X("totcost")

	m2 = larch.Model(dataservice=d.selector_co("femdum == 1"))
	m2.utility_co[2] = P("ASC_SR2")  + P("hhinc#2") * X("hhinc")
	m2.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m2.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m2.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m2.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m2.utility_ca = P("tottime_f")*X("tottime") + P("totcost_f")*X("totcost")

	m0.load_data()
	assert m0.loglike2().ll == approx(-7309.600971749625)

	m1.load_data()
	assert m1.loglike2().ll == approx(-4068.8091617468717)

	m2.load_data()
	assert m2.loglike2().ll == approx(-3240.7918100027578)

	from larch.model.model_group import ModelGroup

	mg = ModelGroup([m1,m2])

	assert mg.loglike2().ll == approx(-7309.600971749625)
	assert mg.loglike() == approx(-7309.600971749625)

	pd.testing.assert_series_equal(
		mg.loglike2().dll.sort_index(), m0.loglike2().dll.sort_index()
	)

	m0.simple_step_bhhh()
	mg.set_values(**m0.pf.value)

	pd.testing.assert_series_equal(
		mg.loglike2().dll.sort_index(), m0.loglike2().dll.sort_index()
	)

	assert mg.loglike2().ll == approx(-4926.4822036792275)
	assert mg.check_d_loglike().data.similarity.min() > 4

	result = mg.maximize_loglike(method='slsqp')
	assert result.loglike == approx(-3620.697668335103)

	mg2 = ModelGroup([])
	mg2.append(m1)
	mg2.append(m2)
	assert mg2.loglike() == approx(-3620.697667552756)
