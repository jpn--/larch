
from ..examples import MTC, SWISSMETRO
from pytest import approx
import numpy
import pandas

def test_dataframes_mnl5():
	d = MTC()
	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', )
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	from larch.dataframes import DataFrames
	from larch.model import Model

	j = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_ch,
	)

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")

	m5.dataframes = j

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.01862990704919887,
	}

	ll2 = m5.loglike2(beta_in1, return_series=True)

	q1_dll = {
		'ASC_BIKE': -139.43832,
		'ASC_SR2': -788.00574,
		'ASC_SR3P': -126.84879,
		'ASC_TRAN': -357.75186,
		'ASC_WALK': -116.137886,
		'hhinc#2': -46416.28,
		'hhinc#3': -8353.63,
		'hhinc#4': -21409.012,
		'hhinc#5': -8299.654,
		'hhinc#6': -7395.375,
		'totcost': 39520.043,
		'tottime': -26556.303,
	}

	assert -4930.3212890625 == approx(ll2.ll)

	for k in q1_dll:
		assert q1_dll[k] == approx(dict(ll2.dll)[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict(ll2.dll)[k]}"

	# Test calculate_parameter_covariance doesn't choke if all holdfasts are on:
	m5.lock_values(*beta_in1.keys())
	m5.calculate_parameter_covariance()

	assert numpy.all(m5.pf['std err'] == 0)
	assert numpy.all(m5.pf['robust std err'] == 0)


def test_dataframes_mnl5_ca():
	d = MTC()
	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', )
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	from larch.dataframes import DataFrames
	from larch import Model

	j = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_ch,
	)

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_ca = PX("tottime") + PX("totcost")

	m5.dataframes = j

	beta_in1 = {
		'totcost': -0.001336661560553717,
		'tottime': -0.01862990704919887,
	}

	ll2 = m5.loglike2(beta_in1, return_series=True)

	q1_dll = {
		'totcost': 173090.625000,
		'tottime': -24771.804688,
	}

	assert -6904.966796875 == approx(ll2.ll, rel=1e-5)

	for k in q1_dll:
		assert q1_dll[k] == approx(dict(ll2.dll)[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict(ll2.dll)[k]}"


def test_dataframes_mnl5_co():
	d = MTC()
	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', )
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	from larch.dataframes import DataFrames
	from larch import Model

	j = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_ch,
	)

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.dataframes = j

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
	}

	ll2 = m5.loglike2(beta_in1, return_series=True)

	q1_dll = {
		'ASC_BIKE': -139.2947,
		'ASC_SR2': -598.531,
		'ASC_SR3P': -77.68647,
		'ASC_TRAN': -715.4206,
		'ASC_WALK': -235.8408,
		'hhinc#2': -35611.855,
		'hhinc#3': -5276.0254,
		'hhinc#4': -42263.88,
		'hhinc#5': -8355.174,
		'hhinc#6': -13866.567,
	}

	assert -5594.70654296875 == approx(ll2.ll, rel=1e-5)

	for k in q1_dll:
		assert q1_dll[k] == approx(dict(ll2.dll)[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict(ll2.dll)[k]}"


def test_dataframes_mnl5q():
	d = MTC()
	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', 'altnum+1', 'ivtt+1')
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	from larch.dataframes import DataFrames
	from larch import Model

	j = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_ch,
	)

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")

	m5.quantity_ca = (
			+ P("FakeSizeAlt") * X('altnum+1')
			+ P("FakeSizeIvtt") * X('ivtt+1')
	)

	m5.dataframes = j

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.01862990704919887,
		'FakeSizeAlt': 0.123,
	}

	ll2 = m5.loglike2(beta_in1, return_series=True)

	q1_dll = {
		'ASC_BIKE': -272.10342,
		'ASC_SR2': -884.91547,
		'ASC_SR3P': -181.50142,
		'ASC_TRAN': -519.74567,
		'ASC_WALK': -37.595825,
		'FakeSizeAlt': -104.97095044599027,
		'FakeSizeIvtt': 104.971085,
		'hhinc#2': -51884.465,
		'hhinc#3': -11712.436,
		'hhinc#4': -30848.334,
		'hhinc#5': -15970.957,
		'hhinc#6': -3269.796,
		'totcost': 59049.66,
		'tottime': -34646.656,
	}

	assert -5598.75244140625 == approx(ll2.ll, rel=1e-5), f"ll2.ll={ll2.ll}"

	for k in q1_dll:
		assert q1_dll[k] == approx(dict(ll2.dll)[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict(ll2.dll)[k]}"

	correct_null_dloglike = {
		'ASC_SR2': -676.598075,
		'ASC_SR3P': -1166.26503,
		'ASC_TRAN': -491.818432,
		'ASC_BIKE': -443.123432,
		'ASC_WALK': 3.9709531565833474,
		'FakeSizeAlt': -86.9414603,
		'FakeSizeIvtt': 86.94146025657632,
		'hhinc#2': -40249.0548,
		'hhinc#3': -67312.464,
		'hhinc#4': -30693.2152,
		'hhinc#5': -27236.7637,
		'hhinc#6': -1389.66274,
		'totcost': 145788.60324123362,
		'tottime': -48732.861026938794,
	}

	ll0 = m5.loglike2('null', return_series=True)
	assert (ll0.ll == approx(-8486.55377320886))
	for k in dict(ll0.dll):
		assert dict(ll0.dll)[k] == approx(
			correct_null_dloglike[k]), f'{k}  {dict(ll0.dll)[k]} == {(dict(correct_null_dloglike)[k])}'


def test_dataframes_mnl5qt():
	d = MTC()
	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', 'altnum+1', 'ivtt+1')
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	from larch.dataframes import DataFrames
	from larch import Model

	j = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_ch,
	)

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")

	m5.quantity_ca = (
			+ P("FakeSizeAlt") * X('altnum+1')
			+ P("FakeSizeIvtt") * X('ivtt+1')
	)

	m5.quantity_scale = P.Theta

	m5.dataframes = j

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.01862990704919887,
		'FakeSizeAlt': 0.123,
		'Theta': 1.0,
	}

	ll2 = m5.loglike2(beta_in1, return_series=True)

	q1_dll = {
		'ASC_BIKE': -272.10342,
		'ASC_SR2': -884.91547,
		'ASC_SR3P': -181.50142,
		'ASC_TRAN': -519.74567,
		'ASC_WALK': -37.595825,
		'FakeSizeAlt': -104.971085,
		'FakeSizeIvtt': 104.971085,
		'hhinc#2': -51884.465,
		'hhinc#3': -11712.436,
		'hhinc#4': -30848.334,
		'hhinc#5': -15970.957,
		'hhinc#6': -3269.796,
		'totcost': 59049.66,
		'tottime': -34646.656,
		'Theta': -838.5296020507812,
	}

	assert -5598.75244140625 == approx(ll2.ll, rel=1e-5), f"ll2.ll={ll2.ll}"

	for k in q1_dll:
		assert q1_dll[k] == approx(dict(ll2.dll)[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict(ll2.dll)[k]}"

	correct_null_dloglike = {
		'ASC_SR2': -676.598075,
		'ASC_SR3P': -1166.26503,
		'ASC_TRAN': -491.818432,
		'ASC_BIKE': -443.123432,
		'ASC_WALK': 3.970966339111328,
		'FakeSizeAlt': -86.9414603,
		'FakeSizeIvtt': 86.94156646728516,
		'hhinc#2': -40249.0548,
		'hhinc#3': -67312.464,
		'hhinc#4': -30693.2152,
		'hhinc#5': -27236.7637,
		'hhinc#6': -1389.66274,
		'totcost': 145788.421875,
		'tottime': -48732.99609375,
		'Theta': -1362.409129,
	}

	ll0 = m5.loglike2('null', return_series=True)
	assert (ll0.ll == approx(-8486.55377320886))
	dict_ll0_dll = dict(ll0.dll)
	for k in dict_ll0_dll:
		assert dict_ll0_dll[k] == approx(correct_null_dloglike[k], rel=1e-5), f'{k}  {dict_ll0_dll[k]} == {(correct_null_dloglike[k])}'


def test_dataframes_nl5():
	d = MTC()
	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', )
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	from larch.dataframes import DataFrames
	from larch import Model

	j = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_ch,
	)

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")

	m5.dataframes = j

	m5.graph.add_node(9, children=(5, 6), parameter='MU_NonMotorized')

	m5.mangle()
	m5.unmangle()
	m5._refresh_derived_arrays()
	m5.pf_sort()
	m5.mangle()
	m5.unmangle()
	m5._refresh_derived_arrays()

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.01862990704919887,
	}

	ll2 = m5.loglike2(beta_in1, return_series=True)

	q1_dll = {
		'ASC_BIKE': -139.43832,
		'ASC_SR2': -788.00574,
		'ASC_SR3P': -126.84879,
		'ASC_TRAN': -357.75186,
		'ASC_WALK': -116.137886,
		'hhinc#2': -46416.28,
		'hhinc#3': -8353.63,
		'hhinc#4': -21409.012,
		'hhinc#5': -8299.654,
		'hhinc#6': -7395.375,
		'totcost': 39520.043,
		'tottime': -26556.303,
	}

	assert approx(ll2.ll) == -4930.3212890625
	dict_ll2_dll = dict(ll2.dll)

	for k in q1_dll:
		assert q1_dll[k] == approx(dict_ll2_dll[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict_ll2_dll[k]}"

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.01862990704919887,
		'MU_NonMotorized': 0.5,
	}

	ll2b = m5.loglike2(beta_in1, return_series=True)

	q1_dllb = {
		'ASC_BIKE': -94.071343,
		'ASC_SR2': -800.092341,
		'ASC_SR3P': -129.354567,
		'ASC_TRAN': -369.808551,
		'ASC_WALK': -114.786728,
		'MU_NonMotorized': -34.816070,
		'hhinc#2': -47089.611079,
		'hhinc#3': -8505.116916,
		'hhinc#4': -22071.859018,
		'hhinc#5': -5844.969336,
		'hhinc#6': -7168.859044,
		'totcost': 37322.528282,
		'tottime': -26479.290942,
	}

	assert -4897.764630665653 == approx(ll2b.ll)

	dict_ll2b_dll = dict(ll2b.dll)

	for k in q1_dllb:
		assert q1_dllb[k] == approx(dict_ll2b_dll[k], rel=1e-5), f"{k} {q1_dllb[k]} != {dict_ll2b_dll[k]}"

	chk = m5.check_d_loglike()

	assert chk.data['similarity'].min() > 4


def test_dataframes_cnl():

	d = SWISSMETRO()
	df_co = d.dataframe_idco('TRAIN_TT', 'SM_TT', 'CAR_TT', "CAR_CO", "SM_CO*(GA==0)", "TRAIN_CO*(GA==0)", "1",
							 "PURPOSE in (1,3)")
	df_av = d.dataframe_idca('avail', dtype=numpy.int8).unstack()
	df_ch = d.dataframe_idca('choice').unstack()
	df_ch.columns = [1, 2, 3]

	from larch.dataframes import DataFrames
	from larch import Model

	j = DataFrames(
		co=df_co[df_co["PURPOSE in (1,3)"].astype(bool)],
		av=df_av[df_co["PURPOSE in (1,3)"].astype(bool)],
		ch=df_ch[df_co["PURPOSE in (1,3)"].astype(bool)],
	)

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[1] = (P.ASC_TRAIN
						+ P.B_TIME * X.TRAIN_TT
						+ P.B_COST * X("TRAIN_CO*(GA==0)"))
	m5.utility_co[2] = (P.B_TIME * X.SM_TT
						+ P.B_COST * X("SM_CO*(GA==0)"))
	m5.utility_co[3] = (P.ASC_CAR
						+ P.B_TIME * X.CAR_TT
						+ P.B_COST * X("CAR_CO"))

	m5.dataframes = j

	m5.graph.add_node(10, children=(1, 3), parameter='MU_existing')
	m5.graph.add_node(11, children=(1, 2), parameter='MU_public')

	m5.mangle()
	m5.unmangle()
	m5._refresh_derived_arrays()
	m5.pf_sort()
	m5.mangle()
	m5.unmangle()
	m5._refresh_derived_arrays()

	m5.set_value("MU_existing", 0.9)

	chk = m5.check_d_loglike()
	assert chk.data.similarity.min() > 3

	assert -6883.805502501951 == approx(m5.loglike())

	b = {
		'ASC_CAR': -0.23820609209707852,
		'ASC_TRAIN': 0.09239345059935042,
		'B_COST': -0.008211788384438165,
		'B_TIME': -0.007794096882716668,
		'MU_existing': 0.39852464867437226,
		'MU_public': 0.24579911999197182,
	}

	assert -5214.063369873166 == approx(m5.loglike(b))

	correct_d_loglike = {
		'ASC_CAR': 0.00578043679257878,
		'ASC_TRAIN': -0.0034066095954343734,
		'B_COST': -0.4321800266172815,
		'B_TIME': 0.24953050221341755,
		'MU_existing': 2.6733742032547525e-05,
		'MU_public': -0.002787061801606372,
	}

	compute_d_loglike = dict(m5.d_loglike(b))

	for k in correct_d_loglike:
		assert compute_d_loglike[k] == approx(correct_d_loglike[k], rel=1e-5), f"{k}: {compute_d_loglike[k]} != {approx(correct_d_loglike[k])}"


def test_weighted_bhhh():

	d = MTC()

	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', )
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	df_co2 = pandas.concat([df_co, df_co]).reset_index(drop=True)
	df_ca2 = pandas.concat([df_ca.unstack(), df_ca.unstack()]).reset_index(drop=True).stack()
	df_av2 = pandas.concat([df_av.unstack(), df_av.unstack()]).reset_index(drop=True).stack()

	df_chX = pandas.DataFrame(
		numpy.zeros_like(df_ch.values),
		index=df_ch.index,
		columns=df_ch.columns,
	)
	df_chX = df_chX.unstack()
	df_chX.iloc[:, 1] = 2.0
	df_chX = df_chX.stack()

	df_ch2 = pandas.concat([df_ch.unstack(), df_chX.unstack()]).reset_index(drop=True).stack()

	from larch import DataFrames, Model

	j1 = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_chX + df_ch,
	)

	j2 = DataFrames(
		co=df_co2,
		ca=df_ca2,
		av=df_av2,
		ch=df_ch2,
	)

	j1.autoscale_weights()
	j2.autoscale_weights()

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.01862990704919887,
	}

	m5.pf_sort()

	m5.dataframes = j1
	ll1 = m5.loglike2_bhhh(beta_in1, return_series=True)

	m5.dataframes = j2
	ll2 = m5.loglike2_bhhh(beta_in1, return_series=True)

	q1_dll = {
		'ASC_BIKE': -518.3145850719714,
		'ASC_SR2': 6659.9870966633935,
		'ASC_SR3P': -702.5461471592637,
		'ASC_TRAN': -2069.2556854096474,
		'ASC_WALK': -680.4136747673049,
		'hhinc#2': 390300.04704708763,
		'hhinc#3': -44451.89987844542,
		'hhinc#4': -117769.88300441334,
		'hhinc#5': -29774.93396444093,
		'hhinc#6': -36754.12651709895,
		'totcost': -280658.27799924824,
		'tottime': -66172.15328009706,
	}

	assert (j1.weight_normalization, j2.weight_normalization) == (3.0, 1.5)

	assert (ll1.ll, ll2.ll) == approx((-18829.858031378415, -18829.858031378433))
	dict_ll1_dll = dict(ll1.dll)
	dict_ll2_dll = dict(ll2.dll)

	for k in q1_dll:
		assert q1_dll[k] == approx(dict_ll1_dll[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict_ll1_dll[k]}"
		assert q1_dll[k] == approx(dict_ll2_dll[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict_ll2_dll[k]}"

	bhhh_correct = {
		('ASC_BIKE', 'ASC_BIKE'): 102.45820678523617,
		('ASC_BIKE', 'ASC_SR2'): -285.118579955482,
		('ASC_BIKE', 'ASC_SR3P'): 19.760852749005203,
		('ASC_BIKE', 'ASC_TRAN'): 76.57398723818505,
		('ASC_BIKE', 'ASC_WALK'): 32.34845311016433,
		('ASC_BIKE', 'hhinc#2'): -16063.872652571783,
		('ASC_BIKE', 'hhinc#3'): 1231.2989480753972,
		('ASC_BIKE', 'hhinc#4'): 4436.148875125188,
		('ASC_BIKE', 'hhinc#5'): 5322.601183863066,
		('ASC_BIKE', 'hhinc#6'): 1870.2419763938155,
		('ASC_BIKE', 'totcost'): 651.7010091466211,
		('ASC_BIKE', 'tottime'): 3410.0934222585456,
		('ASC_SR2', 'ASC_BIKE'): -285.118579955482,
		('ASC_SR2', 'ASC_SR2'): 6156.860848428149,
		('ASC_SR2', 'ASC_SR3P'): -412.00155512537174,
		('ASC_SR2', 'ASC_TRAN'): -1312.7312255613908,
		('ASC_SR2', 'ASC_WALK'): -461.36148965146094,
		('ASC_SR2', 'hhinc#2'): 360960.5064929568,
		('ASC_SR2', 'hhinc#3'): -25836.407171358598,
		('ASC_SR2', 'hhinc#4'): -73800.95798232155,
		('ASC_SR2', 'hhinc#5'): -16063.872652571783,
		('ASC_SR2', 'hhinc#6'): -23821.20824758219,
		('ASC_SR2', 'totcost'): -255314.72401330443,
		('ASC_SR2', 'tottime'): -26043.72175056138,
		('ASC_SR3P', 'ASC_BIKE'): 19.760852749005203,
		('ASC_SR3P', 'ASC_SR2'): -412.00155512537174,
		('ASC_SR3P', 'ASC_SR3P'): 194.01812799791645,
		('ASC_SR3P', 'ASC_TRAN'): 75.72048947864153,
		('ASC_SR3P', 'ASC_WALK'): 21.268891825845834,
		('ASC_SR3P', 'hhinc#2'): -25836.407171358598,
		('ASC_SR3P', 'hhinc#3'): 11951.689201021733,
		('ASC_SR3P', 'hhinc#4'): 4654.093182472213,
		('ASC_SR3P', 'hhinc#5'): 1231.2989480753972,
		('ASC_SR3P', 'hhinc#6'): 1291.5421535124271,
		('ASC_SR3P', 'totcost'): -873.8405986324893,
		('ASC_SR3P', 'tottime'): 2847.414853230761,
		('ASC_TRAN', 'ASC_BIKE'): 76.57398723818505,
		('ASC_TRAN', 'ASC_SR2'): -1312.7312255613908,
		('ASC_TRAN', 'ASC_SR3P'): 75.72048947864153,
		('ASC_TRAN', 'ASC_TRAN'): 795.8802149505715,
		('ASC_TRAN', 'ASC_WALK'): 106.98212315599287,
		('ASC_TRAN', 'hhinc#2'): -73800.95798232155,
		('ASC_TRAN', 'hhinc#3'): 4654.093182472214,
		('ASC_TRAN', 'hhinc#4'): 43744.97724907039,
		('ASC_TRAN', 'hhinc#5'): 4436.148875125188,
		('ASC_TRAN', 'hhinc#6'): 5766.003794801968,
		('ASC_TRAN', 'totcost'): -289.2663620264443,
		('ASC_TRAN', 'tottime'): 17843.995404571946,
		('ASC_WALK', 'ASC_BIKE'): 32.34845311016433,
		('ASC_WALK', 'ASC_SR2'): -461.36148965146094,
		('ASC_WALK', 'ASC_SR3P'): 21.268891825845834,
		('ASC_WALK', 'ASC_TRAN'): 106.98212315599287,
		('ASC_WALK', 'ASC_WALK'): 257.42682167907014,
		('ASC_WALK', 'hhinc#2'): -23821.20824758219,
		('ASC_WALK', 'hhinc#3'): 1291.5421535124271,
		('ASC_WALK', 'hhinc#4'): 5766.003794801967,
		('ASC_WALK', 'hhinc#5'): 1870.2419763938155,
		('ASC_WALK', 'hhinc#6'): 12476.241347103956,
		('ASC_WALK', 'totcost'): -4472.049603002317,
		('ASC_WALK', 'tottime'): 7147.124392574634,
		('hhinc#2', 'ASC_BIKE'): -16063.872652571783,
		('hhinc#2', 'ASC_SR2'): 360960.5064929568,
		('hhinc#2', 'ASC_SR3P'): -25836.407171358598,
		('hhinc#2', 'ASC_TRAN'): -73800.95798232155,
		('hhinc#2', 'ASC_WALK'): -23821.20824758219,
		('hhinc#2', 'hhinc#2'): 27739015.863625936,
		('hhinc#2', 'hhinc#3'): -2107887.1245679897,
		('hhinc#2', 'hhinc#4'): -5551797.986970257,
		('hhinc#2', 'hhinc#5'): -1180261.954185782,
		('hhinc#2', 'hhinc#6'): -1731206.5786703676,
		('hhinc#2', 'totcost'): -15915701.570008647,
		('hhinc#2', 'tottime'): -1404099.9397647786,
		('hhinc#3', 'ASC_BIKE'): 1231.2989480753972,
		('hhinc#3', 'ASC_SR2'): -25836.407171358598,
		('hhinc#3', 'ASC_SR3P'): 11951.689201021733,
		('hhinc#3', 'ASC_TRAN'): 4654.093182472214,
		('hhinc#3', 'ASC_WALK'): 1291.5421535124271,
		('hhinc#3', 'hhinc#2'): -2107887.1245679897,
		('hhinc#3', 'hhinc#3'): 985365.3310746389,
		('hhinc#3', 'hhinc#4'): 365082.18492341647,
		('hhinc#3', 'hhinc#5'): 96863.5554530797,
		('hhinc#3', 'hhinc#6'): 105646.64587551638,
		('hhinc#3', 'totcost'): -18197.967760858926,
		('hhinc#3', 'tottime'): 173393.88742844874,
		('hhinc#4', 'ASC_BIKE'): 4436.148875125188,
		('hhinc#4', 'ASC_SR2'): -73800.95798232155,
		('hhinc#4', 'ASC_SR3P'): 4654.093182472213,
		('hhinc#4', 'ASC_TRAN'): 43744.97724907039,
		('hhinc#4', 'ASC_WALK'): 5766.003794801967,
		('hhinc#4', 'hhinc#2'): -5551797.986970257,
		('hhinc#4', 'hhinc#3'): 365082.18492341647,
		('hhinc#4', 'hhinc#4'): 3238425.3752979557,
		('hhinc#4', 'hhinc#5'): 328415.2849030458,
		('hhinc#4', 'hhinc#6'): 431510.210405761,
		('hhinc#4', 'totcost'): -203942.90089356547,
		('hhinc#4', 'tottime'): 996131.6491729214,
		('hhinc#5', 'ASC_BIKE'): 5322.601183863066,
		('hhinc#5', 'ASC_SR2'): -16063.872652571783,
		('hhinc#5', 'ASC_SR3P'): 1231.2989480753972,
		('hhinc#5', 'ASC_TRAN'): 4436.148875125188,
		('hhinc#5', 'ASC_WALK'): 1870.2419763938155,
		('hhinc#5', 'hhinc#2'): -1180261.954185782,
		('hhinc#5', 'hhinc#3'): 96863.5554530797,
		('hhinc#5', 'hhinc#4'): 328415.2849030458,
		('hhinc#5', 'hhinc#5'): 376816.6222240454,
		('hhinc#5', 'hhinc#6'): 139543.81527069444,
		('hhinc#5', 'totcost'): 64871.45953365492,
		('hhinc#5', 'tottime'): 191184.20323081187,
		('hhinc#6', 'ASC_BIKE'): 1870.2419763938155,
		('hhinc#6', 'ASC_SR2'): -23821.20824758219,
		('hhinc#6', 'ASC_SR3P'): 1291.5421535124271,
		('hhinc#6', 'ASC_TRAN'): 5766.003794801968,
		('hhinc#6', 'ASC_WALK'): 12476.241347103956,
		('hhinc#6', 'hhinc#2'): -1731206.5786703676,
		('hhinc#6', 'hhinc#3'): 105646.64587551638,
		('hhinc#6', 'hhinc#4'): 431510.210405761,
		('hhinc#6', 'hhinc#5'): 139543.81527069444,
		('hhinc#6', 'hhinc#6'): 872604.6192807176,
		('hhinc#6', 'totcost'): -95081.80795513398,
		('hhinc#6', 'tottime'): 364196.75026433205,
		('totcost', 'ASC_BIKE'): 651.7010091466211,
		('totcost', 'ASC_SR2'): -255314.72401330443,
		('totcost', 'ASC_SR3P'): -873.8405986324893,
		('totcost', 'ASC_TRAN'): -289.2663620264443,
		('totcost', 'ASC_WALK'): -4472.049603002317,
		('totcost', 'hhinc#2'): -15915701.570008647,
		('totcost', 'hhinc#3'): -18197.967760858926,
		('totcost', 'hhinc#4'): -203942.90089356547,
		('totcost', 'hhinc#5'): 64871.45953365492,
		('totcost', 'hhinc#6'): -95081.80795513398,
		('totcost', 'totcost'): 67516567.00440338,
		('totcost', 'tottime'): -775245.3645765022,
		('tottime', 'ASC_BIKE'): 3410.0934222585456,
		('tottime', 'ASC_SR2'): -26043.72175056138,
		('tottime', 'ASC_SR3P'): 2847.414853230761,
		('tottime', 'ASC_TRAN'): 17843.995404571946,
		('tottime', 'ASC_WALK'): 7147.124392574634,
		('tottime', 'hhinc#2'): -1404099.9397647786,
		('tottime', 'hhinc#3'): 173393.88742844874,
		('tottime', 'hhinc#4'): 996131.6491729214,
		('tottime', 'hhinc#5'): 191184.20323081187,
		('tottime', 'hhinc#6'): 364196.75026433205,
		('tottime', 'totcost'): -775245.3645765022,
		('tottime', 'tottime'): 910724.9012464394,
	}

	assert dict(ll1.bhhh.unstack()) == approx(bhhh_correct)
	assert dict(ll2.bhhh.unstack()) == approx(bhhh_correct)

	assert m5.check_d_loglike().data.similarity.min() > 4


def test_weighted_nl_bhhh():

	d = MTC()

	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', )
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	df_co2 = pandas.concat([df_co, df_co]).reset_index(drop=True)
	df_ca2 = pandas.concat([df_ca.unstack(), df_ca.unstack()]).reset_index(drop=True).stack()
	df_av2 = pandas.concat([df_av.unstack(), df_av.unstack()]).reset_index(drop=True).stack()

	df_chX = pandas.DataFrame(
		numpy.zeros_like(df_ch.values),
		index=df_ch.index,
		columns=df_ch.columns,
	)
	df_chX = df_chX.unstack()
	df_chX.iloc[:, 1] = 2.0
	df_chX = df_chX.stack()

	df_ch2 = pandas.concat([df_ch.unstack(), df_chX.unstack()]).reset_index(drop=True).stack()

	from larch import DataFrames, Model

	j1 = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_chX + df_ch,
	)

	j2 = DataFrames(
		co=df_co2,
		ca=df_ca2,
		av=df_av2,
		ch=df_ch2,
	)


	j1.autoscale_weights()
	j2.autoscale_weights()

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")

	m5.initialize_graph(alternative_codes=[1,2,3,4,5,6])
	m5.graph.add_node(10, children=(5, 6), parameter='MU_nonmotor')
	m5.graph.add_node(11, children=(1, 2, 3), parameter='MU_car')

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.01862990704919887,
	}

	m5.pf_sort()

	m5.dataframes = j1
	ll1 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)

	m5.dataframes = j2
	ll2 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)

	q1_dll = {
		'ASC_BIKE': -518.3145850719714,
		'ASC_SR2': 6659.9870966633935,
		'ASC_SR3P': -702.5461471592637,
		'ASC_TRAN': -2069.2556854096474,
		'ASC_WALK': -680.4136747673049,
		'hhinc#2': 390300.04704708763,
		'hhinc#3': -44451.89987844542,
		'hhinc#4': -117769.88300441334,
		'hhinc#5': -29774.93396444093,
		'hhinc#6': -36754.12651709895,
		'totcost': -280658.27799924824,
		'tottime': -66172.15328009706,
	}

	assert (j1.weight_normalization, j2.weight_normalization) == (3.0, 1.5)
	assert (ll1.ll, ll2.ll) == approx((-18829.858031378415, -18829.858031378433))
	dict_ll1_dll = dict(ll1.dll)
	dict_ll2_dll = dict(ll2.dll)
	for k in q1_dll:
		assert q1_dll[k] == approx(dict_ll1_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll1_dll[k]}"
		assert q1_dll[k] == approx(dict_ll2_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll2_dll[k]}"

	bhhh_correct = {
		('ASC_BIKE', 'ASC_BIKE'): 102.45820678523614,
		('ASC_BIKE', 'ASC_SR2'): -285.118579955482,
		('ASC_BIKE', 'ASC_SR3P'): 19.760852749005203,
		('ASC_BIKE', 'ASC_TRAN'): 76.573987238185,
		('ASC_BIKE', 'ASC_WALK'): 32.348453110164314,
		('ASC_BIKE', 'MU_car'): -219.27982884201612,
		('ASC_BIKE', 'MU_nonmotor'): 69.38016083689357,
		('ASC_BIKE', 'hhinc#2'): -16063.872652571805,
		('ASC_BIKE', 'hhinc#3'): 1231.2989480753972,
		('ASC_BIKE', 'hhinc#4'): 4436.148875125189,
		('ASC_BIKE', 'hhinc#5'): 5322.601183863066,
		('ASC_BIKE', 'hhinc#6'): 1870.2419763938155,
		('ASC_BIKE', 'totcost'): 651.7010091466225,
		('ASC_BIKE', 'tottime'): 3410.093422258546,
		('ASC_SR2', 'ASC_BIKE'): -285.118579955482,
		('ASC_SR2', 'ASC_SR2'): 6156.860848428152,
		('ASC_SR2', 'ASC_SR3P'): -412.00155512537197,
		('ASC_SR2', 'ASC_TRAN'): -1312.7312255613883,
		('ASC_SR2', 'ASC_WALK'): -461.36148965146094,
		('ASC_SR2', 'MU_car'): 3491.370489213462,
		('ASC_SR2', 'MU_nonmotor'): -244.85789490150424,
		('ASC_SR2', 'hhinc#2'): 360960.5064929566,
		('ASC_SR2', 'hhinc#3'): -25836.407171358624,
		('ASC_SR2', 'hhinc#4'): -73800.95798232155,
		('ASC_SR2', 'hhinc#5'): -16063.872652571805,
		('ASC_SR2', 'hhinc#6'): -23821.208247582188,
		('ASC_SR2', 'totcost'): -255314.7240133045,
		('ASC_SR2', 'tottime'): -26043.721750561366,
		('ASC_SR3P', 'ASC_BIKE'): 19.760852749005203,
		('ASC_SR3P', 'ASC_SR2'): -412.00155512537197,
		('ASC_SR3P', 'ASC_SR3P'): 194.01812799791654,
		('ASC_SR3P', 'ASC_TRAN'): 75.72048947864153,
		('ASC_SR3P', 'ASC_WALK'): 21.26889182584585,
		('ASC_SR3P', 'MU_car'): 108.36680636408329,
		('ASC_SR3P', 'MU_nonmotor'): 11.214695666269147,
		('ASC_SR3P', 'hhinc#2'): -25836.407171358624,
		('ASC_SR3P', 'hhinc#3'): 11951.689201021742,
		('ASC_SR3P', 'hhinc#4'): 4654.093182472214,
		('ASC_SR3P', 'hhinc#5'): 1231.298948075397,
		('ASC_SR3P', 'hhinc#6'): 1291.5421535124285,
		('ASC_SR3P', 'totcost'): -873.8405986324798,
		('ASC_SR3P', 'tottime'): 2847.414853230762,
		('ASC_TRAN', 'ASC_BIKE'): 76.573987238185,
		('ASC_TRAN', 'ASC_SR2'): -1312.7312255613883,
		('ASC_TRAN', 'ASC_SR3P'): 75.72048947864153,
		('ASC_TRAN', 'ASC_TRAN'): 795.8802149505714,
		('ASC_TRAN', 'ASC_WALK'): 106.98212315599287,
		('ASC_TRAN', 'MU_car'): -930.0194408568389,
		('ASC_TRAN', 'MU_nonmotor'): 58.620406514681704,
		('ASC_TRAN', 'hhinc#2'): -73800.95798232155,
		('ASC_TRAN', 'hhinc#3'): 4654.093182472214,
		('ASC_TRAN', 'hhinc#4'): 43744.97724907036,
		('ASC_TRAN', 'hhinc#5'): 4436.148875125189,
		('ASC_TRAN', 'hhinc#6'): 5766.003794801971,
		('ASC_TRAN', 'totcost'): -289.26636202646205,
		('ASC_TRAN', 'tottime'): 17843.995404571957,
		('ASC_WALK', 'ASC_BIKE'): 32.348453110164314,
		('ASC_WALK', 'ASC_SR2'): -461.36148965146094,
		('ASC_WALK', 'ASC_SR3P'): 21.26889182584585,
		('ASC_WALK', 'ASC_TRAN'): 106.98212315599287,
		('ASC_WALK', 'ASC_WALK'): 257.4268216790701,
		('ASC_WALK', 'MU_car'): -372.1302408319178,
		('ASC_WALK', 'MU_nonmotor'): 84.61754029339109,
		('ASC_WALK', 'hhinc#2'): -23821.208247582188,
		('ASC_WALK', 'hhinc#3'): 1291.5421535124283,
		('ASC_WALK', 'hhinc#4'): 5766.003794801973,
		('ASC_WALK', 'hhinc#5'): 1870.2419763938155,
		('ASC_WALK', 'hhinc#6'): 12476.241347103962,
		('ASC_WALK', 'totcost'): -4472.049603002317,
		('ASC_WALK', 'tottime'): 7147.124392574635,
		('MU_car', 'ASC_BIKE'): -219.27982884201612,
		('MU_car', 'ASC_SR2'): 3491.370489213462,
		('MU_car', 'ASC_SR3P'): 108.36680636408329,
		('MU_car', 'ASC_TRAN'): -930.0194408568389,
		('MU_car', 'ASC_WALK'): -372.1302408319178,
		('MU_car', 'MU_car'): 3048.745424759559,
		('MU_car', 'MU_nonmotor'): -215.09078801898488,
		('MU_car', 'hhinc#2'): 209343.58472860648,
		('MU_car', 'hhinc#3'): 4904.0435782757395,
		('MU_car', 'hhinc#4'): -54645.9131974324,
		('MU_car', 'hhinc#5'): -12553.188578813708,
		('MU_car', 'hhinc#6'): -20397.013767114706,
		('MU_car', 'totcost'): -130926.84443581512,
		('MU_car', 'tottime'): -19903.25282015786,
		('MU_nonmotor', 'ASC_BIKE'): 69.38016083689357,
		('MU_nonmotor', 'ASC_SR2'): -244.85789490150424,
		('MU_nonmotor', 'ASC_SR3P'): 11.214695666269147,
		('MU_nonmotor', 'ASC_TRAN'): 58.620406514681704,
		('MU_nonmotor', 'ASC_WALK'): 84.61754029339109,
		('MU_nonmotor', 'MU_car'): -215.09078801898488,
		('MU_nonmotor', 'MU_nonmotor'): 106.11047953795583,
		('MU_nonmotor', 'hhinc#2'): -13788.1097937347,
		('MU_nonmotor', 'hhinc#3'): 711.5550773867994,
		('MU_nonmotor', 'hhinc#4'): 3435.2281478513137,
		('MU_nonmotor', 'hhinc#5'): 3589.230542100178,
		('MU_nonmotor', 'hhinc#6'): 4773.341396211997,
		('MU_nonmotor', 'totcost'): -571.9687206358277,
		('MU_nonmotor', 'tottime'): 3082.561197668304,
		('hhinc#2', 'ASC_BIKE'): -16063.872652571805,
		('hhinc#2', 'ASC_SR2'): 360960.5064929566,
		('hhinc#2', 'ASC_SR3P'): -25836.407171358624,
		('hhinc#2', 'ASC_TRAN'): -73800.95798232155,
		('hhinc#2', 'ASC_WALK'): -23821.208247582188,
		('hhinc#2', 'MU_car'): 209343.58472860648,
		('hhinc#2', 'MU_nonmotor'): -13788.1097937347,
		('hhinc#2', 'hhinc#2'): 27739015.863625906,
		('hhinc#2', 'hhinc#3'): -2107887.124567991,
		('hhinc#2', 'hhinc#4'): -5551797.986970259,
		('hhinc#2', 'hhinc#5'): -1180261.9541857818,
		('hhinc#2', 'hhinc#6'): -1731206.5786703678,
		('hhinc#2', 'totcost'): -15915701.570008647,
		('hhinc#2', 'tottime'): -1404099.9397647784,
		('hhinc#3', 'ASC_BIKE'): 1231.2989480753972,
		('hhinc#3', 'ASC_SR2'): -25836.407171358624,
		('hhinc#3', 'ASC_SR3P'): 11951.689201021742,
		('hhinc#3', 'ASC_TRAN'): 4654.093182472214,
		('hhinc#3', 'ASC_WALK'): 1291.5421535124283,
		('hhinc#3', 'MU_car'): 4904.0435782757395,
		('hhinc#3', 'MU_nonmotor'): 711.5550773867994,
		('hhinc#3', 'hhinc#2'): -2107887.124567991,
		('hhinc#3', 'hhinc#3'): 985365.3310746389,
		('hhinc#3', 'hhinc#4'): 365082.18492341647,
		('hhinc#3', 'hhinc#5'): 96863.55545307972,
		('hhinc#3', 'hhinc#6'): 105646.64587551646,
		('hhinc#3', 'totcost'): -18197.96776085889,
		('hhinc#3', 'tottime'): 173393.88742844868,
		('hhinc#4', 'ASC_BIKE'): 4436.148875125189,
		('hhinc#4', 'ASC_SR2'): -73800.95798232155,
		('hhinc#4', 'ASC_SR3P'): 4654.093182472214,
		('hhinc#4', 'ASC_TRAN'): 43744.97724907036,
		('hhinc#4', 'ASC_WALK'): 5766.003794801973,
		('hhinc#4', 'MU_car'): -54645.9131974324,
		('hhinc#4', 'MU_nonmotor'): 3435.2281478513137,
		('hhinc#4', 'hhinc#2'): -5551797.986970259,
		('hhinc#4', 'hhinc#3'): 365082.18492341647,
		('hhinc#4', 'hhinc#4'): 3238425.3752979534,
		('hhinc#4', 'hhinc#5'): 328415.284903046,
		('hhinc#4', 'hhinc#6'): 431510.21040576114,
		('hhinc#4', 'totcost'): -203942.9008935652,
		('hhinc#4', 'tottime'): 996131.649172921,
		('hhinc#5', 'ASC_BIKE'): 5322.601183863066,
		('hhinc#5', 'ASC_SR2'): -16063.872652571805,
		('hhinc#5', 'ASC_SR3P'): 1231.298948075397,
		('hhinc#5', 'ASC_TRAN'): 4436.148875125189,
		('hhinc#5', 'ASC_WALK'): 1870.2419763938155,
		('hhinc#5', 'MU_car'): -12553.188578813708,
		('hhinc#5', 'MU_nonmotor'): 3589.230542100178,
		('hhinc#5', 'hhinc#2'): -1180261.9541857818,
		('hhinc#5', 'hhinc#3'): 96863.55545307972,
		('hhinc#5', 'hhinc#4'): 328415.284903046,
		('hhinc#5', 'hhinc#5'): 376816.62222404545,
		('hhinc#5', 'hhinc#6'): 139543.8152706944,
		('hhinc#5', 'totcost'): 64871.45953365474,
		('hhinc#5', 'tottime'): 191184.20323081187,
		('hhinc#6', 'ASC_BIKE'): 1870.2419763938155,
		('hhinc#6', 'ASC_SR2'): -23821.208247582188,
		('hhinc#6', 'ASC_SR3P'): 1291.5421535124285,
		('hhinc#6', 'ASC_TRAN'): 5766.003794801971,
		('hhinc#6', 'ASC_WALK'): 12476.241347103962,
		('hhinc#6', 'MU_car'): -20397.013767114706,
		('hhinc#6', 'MU_nonmotor'): 4773.341396211997,
		('hhinc#6', 'hhinc#2'): -1731206.5786703678,
		('hhinc#6', 'hhinc#3'): 105646.64587551646,
		('hhinc#6', 'hhinc#4'): 431510.21040576114,
		('hhinc#6', 'hhinc#5'): 139543.8152706944,
		('hhinc#6', 'hhinc#6'): 872604.6192807175,
		('hhinc#6', 'totcost'): -95081.8079551341,
		('hhinc#6', 'tottime'): 364196.75026433234,
		('totcost', 'ASC_BIKE'): 651.7010091466225,
		('totcost', 'ASC_SR2'): -255314.7240133045,
		('totcost', 'ASC_SR3P'): -873.8405986324798,
		('totcost', 'ASC_TRAN'): -289.26636202646205,
		('totcost', 'ASC_WALK'): -4472.049603002317,
		('totcost', 'MU_car'): -130926.84443581512,
		('totcost', 'MU_nonmotor'): -571.9687206358277,
		('totcost', 'hhinc#2'): -15915701.570008647,
		('totcost', 'hhinc#3'): -18197.96776085889,
		('totcost', 'hhinc#4'): -203942.9008935652,
		('totcost', 'hhinc#5'): 64871.45953365474,
		('totcost', 'hhinc#6'): -95081.8079551341,
		('totcost', 'totcost'): 67516567.00440338,
		('totcost', 'tottime'): -775245.3645765011,
		('tottime', 'ASC_BIKE'): 3410.093422258546,
		('tottime', 'ASC_SR2'): -26043.721750561366,
		('tottime', 'ASC_SR3P'): 2847.414853230762,
		('tottime', 'ASC_TRAN'): 17843.995404571957,
		('tottime', 'ASC_WALK'): 7147.124392574635,
		('tottime', 'MU_car'): -19903.25282015786,
		('tottime', 'MU_nonmotor'): 3082.561197668304,
		('tottime', 'hhinc#2'): -1404099.9397647784,
		('tottime', 'hhinc#3'): 173393.88742844868,
		('tottime', 'hhinc#4'): 996131.649172921,
		('tottime', 'hhinc#5'): 191184.20323081187,
		('tottime', 'hhinc#6'): 364196.75026433234,
		('tottime', 'totcost'): -775245.3645765011,
		('tottime', 'tottime'): 910724.9012464393,
	}
	assert dict(ll1.bhhh.unstack()) == approx(bhhh_correct)
	assert dict(ll2.bhhh.unstack()) == approx(bhhh_correct)

	dll_casewise_A = ll2.dll_casewise / j2.weight_normalization
	dll_casewise_B = dll_casewise_A / j2.data_wt.values

	corrected_bhhh = pandas.DataFrame(
		numpy.dot(dll_casewise_A.T, dll_casewise_B),
		index=ll2.dll.index,
		columns=ll2.dll.index,
	) *  j2.weight_normalization

	assert dict(ll1.bhhh.unstack()) == approx(dict(corrected_bhhh.unstack()))



def test_weighted_nl2_bhhh():
	d = MTC()

	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', )
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	df_co2 = pandas.concat([df_co, df_co]).reset_index(drop=True)
	df_ca2 = pandas.concat([df_ca.unstack(), df_ca.unstack()]).reset_index(drop=True).stack()
	df_av2 = pandas.concat([df_av.unstack(), df_av.unstack()]).reset_index(drop=True).stack()

	df_chX = pandas.DataFrame(
		numpy.zeros_like(df_ch.values),
		index=df_ch.index,
		columns=df_ch.columns,
	)
	df_chX = df_chX.unstack()
	df_chX.iloc[:, 1] = 2.0
	df_chX = df_chX.stack()

	df_ch2 = pandas.concat([df_ch.unstack(), df_chX.unstack()]).reset_index(drop=True).stack()

	from larch import DataFrames, Model

	j1 = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_chX + df_ch,
	)

	j2 = DataFrames(
		co=df_co2,
		ca=df_ca2,
		av=df_av2,
		ch=df_ch2,
	)

	j1.autoscale_weights()
	j2.autoscale_weights()

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")

	m5.initialize_graph(alternative_codes=[1, 2, 3, 4, 5, 6])
	m5.graph.add_node(10, children=(5, 6), parameter='MU_nonmotor')
	m5.graph.add_node(11, children=(1, 2, 3), parameter='MU_car')
	m5.graph.add_node(12, children=(4, 10), parameter='MU_motor')

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.01862990704919887,
		'MU_motor': 0.8,
		'MU_nonmotor': 0.6,
		'MU_car': 0.4,
	}

	m5.pf_sort()

	m5.dataframes = j1
	ll1 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)
	z = m5.check_d_loglike().data.similarity.min()
	assert z > 4, z

	m5.dataframes = j2
	ll2 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)
	z = m5.check_d_loglike().data.similarity.min()
	assert z > 4, z

	q1_dll = {'ASC_BIKE': -504.39658000027384,
			  'ASC_SR2': 19024.71862867697,
			  'ASC_SR3P': 219.58877070822,
			  'ASC_TRAN': -2777.1804805641996,
			  'ASC_WALK': -765.8457046863416,
			  'MU_car': 28719.01003828339,
			  'MU_motor': -1765.1358882225932,
			  'MU_nonmotor': -334.63321113336883,
			  'hhinc#2': 1132970.5412826273,
			  'hhinc#3': 11739.936711817763,
			  'hhinc#4': -159874.9987026767,
			  'hhinc#5': -28910.092921036034,
			  'hhinc#6': -41603.81344709484,
			  'totcost': -1288926.528416055,
			  'tottime': -26541.38424594015}

	assert (j1.weight_normalization, j2.weight_normalization) == (3.0, 1.5)
	assert (ll1.ll, ll2.ll) == approx((-24481.35323844337, -24481.35323844337))
	assert (ll1.ll == approx(ll2.ll))

	dict_ll1_dll = dict(ll1.dll)
	dict_ll2_dll = dict(ll2.dll)

	for k in q1_dll:
		assert q1_dll[k] == approx(dict_ll1_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll1_dll[k]}"
		assert q1_dll[k] == approx(dict_ll2_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll2_dll[k]}"

	bhhh_correct = {('ASC_BIKE', 'ASC_BIKE'): 154.48714238211153,
					('ASC_BIKE', 'ASC_SR2'): -738.7854564814061,
					('ASC_BIKE', 'ASC_SR3P'): -4.301978918250828,
					('ASC_BIKE', 'ASC_TRAN'): 77.11480400852297,
					('ASC_BIKE', 'ASC_WALK'): -6.368448179206997,
					('ASC_BIKE', 'MU_car'): -1179.9029288396655,
					('ASC_BIKE', 'MU_motor'): 150.95072473216015,
					('ASC_BIKE', 'MU_nonmotor'): 102.1858743697233,
					('ASC_BIKE', 'hhinc#2'): -42235.1104152968,
					('ASC_BIKE', 'hhinc#3'): -160.86189070481015,
					('ASC_BIKE', 'hhinc#4'): 4614.891034013686,
					('ASC_BIKE', 'hhinc#5'): 7782.554401095187,
					('ASC_BIKE', 'hhinc#6'): -51.49598181960236,
					('ASC_BIKE', 'totcost'): 19640.395092668485,
					('ASC_BIKE', 'tottime'): 724.5514955075083,
					('ASC_SR2', 'ASC_BIKE'): -738.7854564814061,
					('ASC_SR2', 'ASC_SR2'): 43329.01034566799,
					('ASC_SR2', 'ASC_SR3P'): -475.14156376513563,
					('ASC_SR2', 'ASC_TRAN'): -4109.113773332244,
					('ASC_SR2', 'ASC_WALK'): -1221.6942591321904,
					('ASC_SR2', 'MU_car'): 61121.65962952458,
					('ASC_SR2', 'MU_motor'): -2575.2754574632345,
					('ASC_SR2', 'MU_nonmotor'): -543.1673827849247,
					('ASC_SR2', 'hhinc#2'): 2593234.4044680865,
					('ASC_SR2', 'hhinc#3'): -30252.293897665764,
					('ASC_SR2', 'hhinc#4'): -240130.15318531267,
					('ASC_SR2', 'hhinc#5'): -42235.110415296804,
					('ASC_SR2', 'hhinc#6'): -66484.96361729817,
					('ASC_SR2', 'totcost'): -2722755.8518074444,
					('ASC_SR2', 'tottime'): 36264.96777985287,
					('ASC_SR3P', 'ASC_BIKE'): -4.301978918250828,
					('ASC_SR3P', 'ASC_SR2'): -475.14156376513563,
					('ASC_SR3P', 'ASC_SR3P'): 993.7416647991923,
					('ASC_SR3P', 'ASC_TRAN'): -56.457413527921354,
					('ASC_SR3P', 'ASC_WALK'): -8.290715205591201,
					('ASC_SR3P', 'MU_car'): 4256.707637736697,
					('ASC_SR3P', 'MU_motor'): -16.784439691493535,
					('ASC_SR3P', 'MU_nonmotor'): -4.505259115618738,
					('ASC_SR3P', 'hhinc#2'): -30252.293897665764,
					('ASC_SR3P', 'hhinc#3'): 59699.49555432012,
					('ASC_SR3P', 'hhinc#4'): -3217.2039607787374,
					('ASC_SR3P', 'hhinc#5'): -160.86189070481015,
					('ASC_SR3P', 'hhinc#6'): -292.11245664731786,
					('ASC_SR3P', 'totcost'): -136639.36518683203,
					('ASC_SR3P', 'tottime'): 1221.713199489567,
					('ASC_TRAN', 'ASC_BIKE'): 77.11480400852297,
					('ASC_TRAN', 'ASC_SR2'): -4109.113773332244,
					('ASC_TRAN', 'ASC_SR3P'): -56.457413527921354,
					('ASC_TRAN', 'ASC_TRAN'): 1133.572930284945,
					('ASC_TRAN', 'ASC_WALK'): 114.42900066432574,
					('ASC_TRAN', 'MU_car'): -6013.187739864936,
					('ASC_TRAN', 'MU_motor'): 449.67463821610204,
					('ASC_TRAN', 'MU_nonmotor'): 50.01524128881465,
					('ASC_TRAN', 'hhinc#2'): -240130.15318531267,
					('ASC_TRAN', 'hhinc#3'): -3217.2039607787374,
					('ASC_TRAN', 'hhinc#4'): 63218.91895668433,
					('ASC_TRAN', 'hhinc#5'): 4614.891034013686,
					('ASC_TRAN', 'hhinc#6'): 6552.868813258239,
					('ASC_TRAN', 'totcost'): 227426.05127724586,
					('ASC_TRAN', 'tottime'): 13351.526935637983,
					('ASC_WALK', 'ASC_BIKE'): -6.368448179206997,
					('ASC_WALK', 'ASC_SR2'): -1221.6942591321904,
					('ASC_WALK', 'ASC_SR3P'): -8.290715205591201,
					('ASC_WALK', 'ASC_TRAN'): 114.42900066432574,
					('ASC_WALK', 'ASC_WALK'): 357.6626631418752,
					('ASC_WALK', 'MU_car'): -1965.649559092812,
					('ASC_WALK', 'MU_motor'): 294.9144861941048,
					('ASC_WALK', 'MU_nonmotor'): 51.25557776912339,
					('ASC_WALK', 'hhinc#2'): -66484.96361729817,
					('ASC_WALK', 'hhinc#3'): -292.11245664731774,
					('ASC_WALK', 'hhinc#4'): 6552.868813258239,
					('ASC_WALK', 'hhinc#5'): -51.49598181960249,
					('ASC_WALK', 'hhinc#6'): 17497.41883840556,
					('ASC_WALK', 'totcost'): 11858.192554059988,
					('ASC_WALK', 'tottime'): 6014.405097068796,
					('MU_car', 'ASC_BIKE'): -1179.9029288396655,
					('MU_car', 'ASC_SR2'): 61121.65962952458,
					('MU_car', 'ASC_SR3P'): 4256.707637736697,
					('MU_car', 'ASC_TRAN'): -6013.187739864936,
					('MU_car', 'ASC_WALK'): -1965.649559092812,
					('MU_car', 'MU_car'): 115453.55743870192,
					('MU_car', 'MU_motor'): -4071.8833233537725,
					('MU_car', 'MU_nonmotor'): -906.8466572896994,
					('MU_car', 'hhinc#2'): 3745024.8406667328,
					('MU_car', 'hhinc#3'): 239987.26964595896,
					('MU_car', 'hhinc#4'): -360556.20656797994,
					('MU_car', 'hhinc#5'): -68738.12096301645,
					('MU_car', 'hhinc#6'): -111500.35695638353,
					('MU_car', 'totcost'): -3897167.907753843,
					('MU_car', 'tottime'): 59847.4819379508,
					('MU_motor', 'ASC_BIKE'): 150.95072473216015,
					('MU_motor', 'ASC_SR2'): -2575.2754574632345,
					('MU_motor', 'ASC_SR3P'): -16.784439691493535,
					('MU_motor', 'ASC_TRAN'): 449.67463821610204,
					('MU_motor', 'ASC_WALK'): 294.9144861941048,
					('MU_motor', 'MU_car'): -4071.8833233537725,
					('MU_motor', 'MU_motor'): 609.2888134736735,
					('MU_motor', 'MU_nonmotor'): 126.75068880263409,
					('MU_motor', 'hhinc#2'): -145841.4840899703,
					('MU_motor', 'hhinc#3'): -590.0541549982431,
					('MU_motor', 'hhinc#4'): 24356.07988399558,
					('MU_motor', 'hhinc#5'): 8305.257699609288,
					('MU_motor', 'hhinc#6'): 15250.463807158652,
					('MU_motor', 'totcost'): 59065.476908360346,
					('MU_motor', 'tottime'): 8369.8679574357,
					('MU_nonmotor', 'ASC_BIKE'): 102.1858743697233,
					('MU_nonmotor', 'ASC_SR2'): -543.1673827849247,
					('MU_nonmotor', 'ASC_SR3P'): -4.505259115618738,
					('MU_nonmotor', 'ASC_TRAN'): 50.01524128881465,
					('MU_nonmotor', 'ASC_WALK'): 51.25557776912339,
					('MU_nonmotor', 'MU_car'): -906.8466572896994,
					('MU_nonmotor', 'MU_motor'): 126.75068880263409,
					('MU_nonmotor', 'MU_nonmotor'): 142.25948715857803,
					('MU_nonmotor', 'hhinc#2'): -31327.395238162186,
					('MU_nonmotor', 'hhinc#3'): -233.88414840871866,
					('MU_nonmotor', 'hhinc#4'): 3058.6025660453615,
					('MU_nonmotor', 'hhinc#5'): 5050.630152620362,
					('MU_nonmotor', 'hhinc#6'): 3139.2561715645115,
					('MU_nonmotor', 'totcost'): 5987.507387526554,
					('MU_nonmotor', 'tottime'): 724.237849253407,
					('hhinc#2', 'ASC_BIKE'): -42235.1104152968,
					('hhinc#2', 'ASC_SR2'): 2593234.4044680865,
					('hhinc#2', 'ASC_SR3P'): -30252.293897665764,
					('hhinc#2', 'ASC_TRAN'): -240130.15318531267,
					('hhinc#2', 'ASC_WALK'): -66484.96361729817,
					('hhinc#2', 'MU_car'): 3745024.8406667328,
					('hhinc#2', 'MU_motor'): -145841.4840899703,
					('hhinc#2', 'MU_nonmotor'): -31327.395238162186,
					('hhinc#2', 'hhinc#2'): 201662258.0531394,
					('hhinc#2', 'hhinc#3'): -2544549.265258612,
					('hhinc#2', 'hhinc#4'): -18465978.129297145,
					('hhinc#2', 'hhinc#5'): -3102351.476272743,
					('hhinc#2', 'hhinc#6'): -4991093.676906733,
					('hhinc#2', 'totcost'): -170854966.34845263,
					('hhinc#2', 'tottime'): 2443690.5500738095,
					('hhinc#3', 'ASC_BIKE'): -160.86189070481015,
					('hhinc#3', 'ASC_SR2'): -30252.293897665764,
					('hhinc#3', 'ASC_SR3P'): 59699.49555432012,
					('hhinc#3', 'ASC_TRAN'): -3217.2039607787374,
					('hhinc#3', 'ASC_WALK'): -292.11245664731774,
					('hhinc#3', 'MU_car'): 239987.26964595896,
					('hhinc#3', 'MU_motor'): -590.0541549982431,
					('hhinc#3', 'MU_nonmotor'): -233.88414840871866,
					('hhinc#3', 'hhinc#2'): -2544549.265258612,
					('hhinc#3', 'hhinc#3'): 4866050.873976185,
					('hhinc#3', 'hhinc#4'): -274144.7119888196,
					('hhinc#3', 'hhinc#5'): -7617.198223617872,
					('hhinc#3', 'hhinc#6'): -13520.596113151361,
					('hhinc#3', 'totcost'): -8569432.533358011,
					('hhinc#3', 'tottime'): 87563.19336918782,
					('hhinc#4', 'ASC_BIKE'): 4614.891034013686,
					('hhinc#4', 'ASC_SR2'): -240130.15318531267,
					('hhinc#4', 'ASC_SR3P'): -3217.2039607787374,
					('hhinc#4', 'ASC_TRAN'): 63218.91895668433,
					('hhinc#4', 'ASC_WALK'): 6552.868813258239,
					('hhinc#4', 'MU_car'): -360556.20656797994,
					('hhinc#4', 'MU_motor'): 24356.07988399558,
					('hhinc#4', 'MU_nonmotor'): 3058.6025660453615,
					('hhinc#4', 'hhinc#2'): -18465978.129297145,
					('hhinc#4', 'hhinc#3'): -274144.7119888196,
					('hhinc#4', 'hhinc#4'): 4723291.333450135,
					('hhinc#4', 'hhinc#5'): 341698.4642162235,
					('hhinc#4', 'hhinc#6'): 508856.09679124475,
					('hhinc#4', 'totcost'): 14102964.453828128,
					('hhinc#4', 'tottime'): 723773.6837364545,
					('hhinc#5', 'ASC_BIKE'): 7782.554401095187,
					('hhinc#5', 'ASC_SR2'): -42235.110415296804,
					('hhinc#5', 'ASC_SR3P'): -160.86189070481015,
					('hhinc#5', 'ASC_TRAN'): 4614.891034013686,
					('hhinc#5', 'ASC_WALK'): -51.49598181960249,
					('hhinc#5', 'MU_car'): -68738.12096301645,
					('hhinc#5', 'MU_motor'): 8305.257699609288,
					('hhinc#5', 'MU_nonmotor'): 5050.630152620362,
					('hhinc#5', 'hhinc#2'): -3102351.476272743,
					('hhinc#5', 'hhinc#3'): -7617.198223617872,
					('hhinc#5', 'hhinc#4'): 341698.4642162235,
					('hhinc#5', 'hhinc#5'): 549476.2640342958,
					('hhinc#5', 'hhinc#6'): -1879.4932678683165,
					('hhinc#5', 'totcost'): 1205769.6903003368,
					('hhinc#5', 'tottime'): 40907.81660310792,
					('hhinc#6', 'ASC_BIKE'): -51.49598181960236,
					('hhinc#6', 'ASC_SR2'): -66484.96361729817,
					('hhinc#6', 'ASC_SR3P'): -292.11245664731786,
					('hhinc#6', 'ASC_TRAN'): 6552.868813258239,
					('hhinc#6', 'ASC_WALK'): 17497.41883840556,
					('hhinc#6', 'MU_car'): -111500.35695638353,
					('hhinc#6', 'MU_motor'): 15250.463807158652,
					('hhinc#6', 'MU_nonmotor'): 3139.2561715645115,
					('hhinc#6', 'hhinc#2'): -4991093.676906733,
					('hhinc#6', 'hhinc#3'): -13520.596113151361,
					('hhinc#6', 'hhinc#4'): 508856.09679124475,
					('hhinc#6', 'hhinc#5'): -1879.4932678683165,
					('hhinc#6', 'hhinc#6'): 1235525.4271056913,
					('hhinc#6', 'totcost'): 890038.1676551632,
					('hhinc#6', 'tottime'): 295733.65975676605,
					('totcost', 'ASC_BIKE'): 19640.395092668485,
					('totcost', 'ASC_SR2'): -2722755.8518074444,
					('totcost', 'ASC_SR3P'): -136639.36518683203,
					('totcost', 'ASC_TRAN'): 227426.05127724586,
					('totcost', 'ASC_WALK'): 11858.192554059988,
					('totcost', 'MU_car'): -3897167.907753843,
					('totcost', 'MU_motor'): 59065.476908360346,
					('totcost', 'MU_nonmotor'): 5987.507387526554,
					('totcost', 'hhinc#2'): -170854966.34845263,
					('totcost', 'hhinc#3'): -8569432.533358011,
					('totcost', 'hhinc#4'): 14102964.453828128,
					('totcost', 'hhinc#5'): 1205769.6903003368,
					('totcost', 'hhinc#6'): 890038.1676551632,
					('totcost', 'totcost'): 438715104.2612953,
					('totcost', 'tottime'): -5816899.428380899,
					('tottime', 'ASC_BIKE'): 724.5514955075083,
					('tottime', 'ASC_SR2'): 36264.96777985287,
					('tottime', 'ASC_SR3P'): 1221.713199489567,
					('tottime', 'ASC_TRAN'): 13351.526935637983,
					('tottime', 'ASC_WALK'): 6014.405097068796,
					('tottime', 'MU_car'): 59847.4819379508,
					('tottime', 'MU_motor'): 8369.8679574357,
					('tottime', 'MU_nonmotor'): 724.237849253407,
					('tottime', 'hhinc#2'): 2443690.5500738095,
					('tottime', 'hhinc#3'): 87563.19336918782,
					('tottime', 'hhinc#4'): 723773.6837364545,
					('tottime', 'hhinc#5'): 40907.81660310792,
					('tottime', 'hhinc#6'): 295733.65975676605,
					('tottime', 'totcost'): -5816899.428380899,
					('tottime', 'tottime'): 1134062.8413517221}

	assert dict(ll1.bhhh.unstack()) == approx(bhhh_correct)
	assert dict(ll2.bhhh.unstack()) == approx(bhhh_correct)
	assert dict(ll1.bhhh.unstack()) == approx(dict(ll2.bhhh.unstack()))

	dll_casewise_A = ll2.dll_casewise / j2.weight_normalization
	if j2.data_wt is not None:
		dll_casewise_B = dll_casewise_A / j2.data_wt.values
	else:
		dll_casewise_B = dll_casewise_A

	corrected_bhhh = pandas.DataFrame(
		numpy.dot(dll_casewise_A.T, dll_casewise_B),
		index=ll2.dll.index,
		columns=ll2.dll.index,
	) * j2.weight_normalization

	assert dict(ll1.bhhh.unstack()) == approx(dict(corrected_bhhh.unstack()))

	assert ll2.dll.values == approx(ll2.dll_casewise.sum(0))


def test_weighted_qnl_bhhh():

	d = MTC()

	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', 'tottime+100', 'totcost+50')
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	df_co2 = pandas.concat([df_co, df_co]).reset_index(drop=True)
	df_ca2 = pandas.concat([df_ca.unstack(), df_ca.unstack()]).reset_index(drop=True).stack()
	df_av2 = pandas.concat([df_av.unstack(), df_av.unstack()]).reset_index(drop=True).stack()

	df_chX = pandas.DataFrame(
		numpy.zeros_like(df_ch.values),
		index=df_ch.index,
		columns=df_ch.columns,
	)
	df_chX = df_chX.unstack()
	df_chX.iloc[:, 1] = 2.0
	df_chX = df_chX.stack()

	df_ch2 = pandas.concat([df_ch.unstack(), df_chX.unstack()]).reset_index(drop=True).stack()

	from larch import DataFrames, Model

	j1 = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_chX + df_ch,
	)

	j2 = DataFrames(
		co=df_co2,
		ca=df_ca2,
		av=df_av2,
		ch=df_ch2,
	)

	j1.autoscale_weights()

	j2.autoscale_weights()

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")

	m5.initialize_graph(alternative_codes=[1, 2, 3, 4, 5, 6])
	m5.graph.add_node(10, children=(5, 6), parameter='MU_nonmotor')
	m5.graph.add_node(11, children=(1, 2, 3), parameter='MU_car')
	m5.graph.add_node(12, children=(4, 10), parameter='MU_motor')

	m5.quantity_ca = PX("tottime+100") + PX("totcost+50")

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.01862990704919887,
		'MU_motor': 0.8,
		'MU_nonmotor': 0.6,
		'MU_car': 0.4,
		'totcost+50': 2,
	}

	m5.pf_sort()

	m5.dataframes = j1
	ll1 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)
	assert m5.check_d_loglike().data.similarity.min() > 4

	m5.dataframes = j2
	ll2 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)
	assert m5.check_d_loglike().data.similarity.min() > 4

	q1_dll = {'ASC_BIKE': -211.29923846797828,
			  'ASC_SR2': 22245.040250878912,
			  'ASC_SR3P': 347.15185128497643,
			  'ASC_TRAN': -3426.9381225801453,
			  'ASC_WALK': -397.47915188625046,
			  'MU_car': 52254.062732240636,
			  'MU_motor': -1306.1578665159548,
			  'MU_nonmotor': -171.48445034651812,
			  'hhinc#2': 1324520.0722549502,
			  'hhinc#3': 20676.66103956428,
			  'hhinc#4': -192937.30280015932,
			  'hhinc#5': -12277.224491430377,
			  'hhinc#6': -22592.39037661451,
			  'totcost': -2043038.5470828116,
			  'totcost+50': -856.8768428455897,
			  'tottime': -12413.533350134443,
			  'tottime+100': 856.876842845588}

	assert (j1.weight_normalization, j2.weight_normalization) == (3.0, 1.5)

	assert (ll1.ll, ll2.ll) == approx((-31802.098963182885, -31802.098963182885))

	assert (ll1.ll == approx(ll2.ll))

	dict_ll1_dll = dict(ll1.dll)
	dict_ll2_dll = dict(ll2.dll)

	for k in q1_dll:
		assert q1_dll[k] == approx(dict_ll1_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll1_dll[k]}"
		assert q1_dll[k] == approx(dict_ll2_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll2_dll[k]}"

	bhhh_correct = {('ASC_BIKE', 'ASC_BIKE'): 114.4251362713502,
					('ASC_BIKE', 'ASC_SR2'): -401.36691075428723,
					('ASC_BIKE', 'ASC_SR3P'): -3.199477838419059,
					('ASC_BIKE', 'ASC_TRAN'): 35.732518979094245,
					('ASC_BIKE', 'ASC_WALK'): -21.998305369885234,
					('ASC_BIKE', 'MU_car'): -880.7574488222571,
					('ASC_BIKE', 'MU_motor'): 134.29998966042848,
					('ASC_BIKE', 'MU_nonmotor'): 95.51131868701871,
					('ASC_BIKE', 'hhinc#2'): -22637.568615942328,
					('ASC_BIKE', 'hhinc#3'): -156.36775653099275,
					('ASC_BIKE', 'hhinc#4'): 2266.1904081469224,
					('ASC_BIKE', 'hhinc#5'): 5591.231146752164,
					('ASC_BIKE', 'hhinc#6'): -939.3767918137345,
					('ASC_BIKE', 'totcost'): 9355.705173389544,
					('ASC_BIKE', 'totcost+50'): 7.264603289573004,
					('ASC_BIKE', 'tottime'): -232.8517672039079,
					('ASC_BIKE', 'tottime+100'): -7.264603289573047,
					('ASC_SR2', 'ASC_BIKE'): -401.36691075428723,
					('ASC_SR2', 'ASC_SR2'): 52630.84443972891,
					('ASC_SR2', 'ASC_SR3P'): -208.2225319202076,
					('ASC_SR2', 'ASC_TRAN'): -5570.045171516203,
					('ASC_SR2', 'ASC_WALK'): -828.5141882017737,
					('ASC_SR2', 'MU_car'): 117781.04325445561,
					('ASC_SR2', 'MU_motor'): -2319.169031704163,
					('ASC_SR2', 'MU_nonmotor'): -358.0102155603283,
					('ASC_SR2', 'hhinc#2'): 3148325.5527452324,
					('ASC_SR2', 'hhinc#3'): -12404.55015529356,
					('ASC_SR2', 'hhinc#4'): -321358.88345842547,
					('ASC_SR2', 'hhinc#5'): -22637.568615942328,
					('ASC_SR2', 'hhinc#6'): -44867.80739151402,
					('ASC_SR2', 'totcost'): -4295397.168565195,
					('ASC_SR2', 'totcost+50'): -1929.997244326729,
					('ASC_SR2', 'tottime'): 54945.16106709013,
					('ASC_SR2', 'tottime+100'): 1929.997244326727,
					('ASC_SR3P', 'ASC_BIKE'): -3.199477838419059,
					('ASC_SR3P', 'ASC_SR2'): -208.2225319202076,
					('ASC_SR3P', 'ASC_SR3P'): 1003.1573375892003,
					('ASC_SR3P', 'ASC_TRAN'): -82.489901409276,
					('ASC_SR3P', 'ASC_WALK'): -7.522359220390586,
					('ASC_SR3P', 'MU_car'): 6148.731317219248,
					('ASC_SR3P', 'MU_motor'): -19.73788773813555,
					('ASC_SR3P', 'MU_nonmotor'): -3.5868894139251393,
					('ASC_SR3P', 'hhinc#2'): -12404.55015529356,
					('ASC_SR3P', 'hhinc#3'): 60384.48715890228,
					('ASC_SR3P', 'hhinc#4'): -4773.734475201185,
					('ASC_SR3P', 'hhinc#5'): -156.36775653099272,
					('ASC_SR3P', 'hhinc#6'): -321.51566832624457,
					('ASC_SR3P', 'totcost'): -195005.11873966618,
					('ASC_SR3P', 'totcost+50'): -57.07300966192541,
					('ASC_SR3P', 'tottime'): 1753.5356217530475,
					('ASC_SR3P', 'tottime+100'): 57.07300966192533,
					('ASC_TRAN', 'ASC_BIKE'): 35.732518979094245,
					('ASC_TRAN', 'ASC_SR2'): -5570.045171516203,
					('ASC_TRAN', 'ASC_SR3P'): -82.489901409276,
					('ASC_TRAN', 'ASC_TRAN'): 1561.1588051811975,
					('ASC_TRAN', 'ASC_WALK'): 76.53442532284116,
					('ASC_TRAN', 'MU_car'): -12492.849054924072,
					('ASC_TRAN', 'MU_motor'): 407.73045331286824,
					('ASC_TRAN', 'MU_nonmotor'): 33.712764700334745,
					('ASC_TRAN', 'hhinc#2'): -321358.88345842547,
					('ASC_TRAN', 'hhinc#3'): -4773.734475201185,
					('ASC_TRAN', 'hhinc#4'): 85170.41064773081,
					('ASC_TRAN', 'hhinc#5'): 2266.1904081469224,
					('ASC_TRAN', 'hhinc#6'): 5004.480689180803,
					('ASC_TRAN', 'totcost'): 334381.6742163283,
					('ASC_TRAN', 'totcost+50'): 215.1438275624966,
					('ASC_TRAN', 'tottime'): 15555.576449608358,
					('ASC_TRAN', 'tottime+100'): -215.14382756249677,
					('ASC_WALK', 'ASC_BIKE'): -21.998305369885234,
					('ASC_WALK', 'ASC_SR2'): -828.5141882017737,
					('ASC_WALK', 'ASC_SR3P'): -7.522359220390586,
					('ASC_WALK', 'ASC_TRAN'): 76.53442532284116,
					('ASC_WALK', 'ASC_WALK'): 281.20060072141774,
					('ASC_WALK', 'MU_car'): -1630.101792680316,
					('ASC_WALK', 'MU_motor'): 315.811655256776,
					('ASC_WALK', 'MU_nonmotor'): 15.382810995522687,
					('ASC_WALK', 'hhinc#2'): -44867.80739151402,
					('ASC_WALK', 'hhinc#3'): -321.51566832624457,
					('ASC_WALK', 'hhinc#4'): 5004.480689180802,
					('ASC_WALK', 'hhinc#5'): -939.3767918137345,
					('ASC_WALK', 'hhinc#6'): 13250.69044755796,
					('ASC_WALK', 'totcost'): 8375.85450595358,
					('ASC_WALK', 'totcost+50'): 1.3445414940886116,
					('ASC_WALK', 'tottime'): 4904.039044545349,
					('ASC_WALK', 'tottime+100'): -1.3445414940888838,
					('MU_car', 'ASC_BIKE'): -880.7574488222571,
					('MU_car', 'ASC_SR2'): 117781.04325445561,
					('MU_car', 'ASC_SR3P'): 6148.731317219248,
					('MU_car', 'ASC_TRAN'): -12492.849054924072,
					('MU_car', 'ASC_WALK'): -1630.101792680316,
					('MU_car', 'MU_car'): 312342.7134659766,
					('MU_car', 'MU_motor'): -4775.497563246944,
					('MU_car', 'MU_nonmotor'): -730.5577865002751,
					('MU_car', 'hhinc#2'): 7205172.648634898,
					('MU_car', 'hhinc#3'): 360003.5818421476,
					('MU_car', 'hhinc#4'): -743754.3177440444,
					('MU_car', 'hhinc#5'): -50670.5965232693,
					('MU_car', 'hhinc#6'): -91816.7683607486,
					('MU_car', 'totcost'): -10836234.999983579,
					('MU_car', 'totcost+50'): -4829.6923345401465,
					('MU_car', 'tottime'): 151537.51100935883,
					('MU_car', 'tottime+100'): 4829.692334540142,
					('MU_motor', 'ASC_BIKE'): 134.29998966042848,
					('MU_motor', 'ASC_SR2'): -2319.169031704163,
					('MU_motor', 'ASC_SR3P'): -19.73788773813555,
					('MU_motor', 'ASC_TRAN'): 407.73045331286824,
					('MU_motor', 'ASC_WALK'): 315.811655256776,
					('MU_motor', 'MU_car'): -4775.497563246944,
					('MU_motor', 'MU_motor'): 724.4261315633672,
					('MU_motor', 'MU_nonmotor'): 137.34936804948558,
					('MU_motor', 'hhinc#2'): -129752.75505412505,
					('MU_motor', 'hhinc#3'): -903.7481484499674,
					('MU_motor', 'hhinc#4'): 23198.692153116197,
					('MU_motor', 'hhinc#5'): 7039.2191386735185,
					('MU_motor', 'hhinc#6'): 15326.954859509155,
					('MU_motor', 'totcost'): 54730.860824187155,
					('MU_motor', 'totcost+50'): 44.26890319114984,
					('MU_motor', 'tottime'): 8526.328590702655,
					('MU_motor', 'tottime+100'): -44.268903191150415,
					('MU_nonmotor', 'ASC_BIKE'): 95.51131868701871,
					('MU_nonmotor', 'ASC_SR2'): -358.0102155603283,
					('MU_nonmotor', 'ASC_SR3P'): -3.5868894139251393,
					('MU_nonmotor', 'ASC_TRAN'): 33.712764700334745,
					('MU_nonmotor', 'ASC_WALK'): 15.382810995522687,
					('MU_nonmotor', 'MU_car'): -730.5577865002751,
					('MU_nonmotor', 'MU_motor'): 137.34936804948558,
					('MU_nonmotor', 'MU_nonmotor'): 125.4610317634725,
					('MU_nonmotor', 'hhinc#2'): -20498.891883189535,
					('MU_nonmotor', 'hhinc#3'): -199.09190169318276,
					('MU_nonmotor', 'hhinc#4'): 2227.8351942101954,
					('MU_nonmotor', 'hhinc#5'): 4627.509599608355,
					('MU_nonmotor', 'hhinc#6'): 1076.7067806851492,
					('MU_nonmotor', 'totcost'): 2863.6842252387814,
					('MU_nonmotor', 'totcost+50'): 3.3721148121600755,
					('MU_nonmotor', 'tottime'): 145.49198209455562,
					('MU_nonmotor', 'tottime+100'): -3.372114812160218,
					('hhinc#2', 'ASC_BIKE'): -22637.568615942328,
					('hhinc#2', 'ASC_SR2'): 3148325.5527452324,
					('hhinc#2', 'ASC_SR3P'): -12404.55015529356,
					('hhinc#2', 'ASC_TRAN'): -321358.88345842547,
					('hhinc#2', 'ASC_WALK'): -44867.80739151402,
					('hhinc#2', 'MU_car'): 7205172.648634898,
					('hhinc#2', 'MU_motor'): -129752.75505412505,
					('hhinc#2', 'MU_nonmotor'): -20498.891883189535,
					('hhinc#2', 'hhinc#2'): 243996387.336173,
					('hhinc#2', 'hhinc#3'): -1013597.9862056787,
					('hhinc#2', 'hhinc#4'): -24514725.011886124,
					('hhinc#2', 'hhinc#5'): -1644932.6104088873,
					('hhinc#2', 'hhinc#6'): -3350183.704181544,
					('hhinc#2', 'totcost'): -269715687.05097497,
					('hhinc#2', 'totcost+50'): -115285.24163010289,
					('hhinc#2', 'tottime'): 3647886.050161818,
					('hhinc#2', 'tottime+100'): 115285.2416301027,
					('hhinc#3', 'ASC_BIKE'): -156.36775653099275,
					('hhinc#3', 'ASC_SR2'): -12404.55015529356,
					('hhinc#3', 'ASC_SR3P'): 60384.48715890228,
					('hhinc#3', 'ASC_TRAN'): -4773.734475201185,
					('hhinc#3', 'ASC_WALK'): -321.51566832624457,
					('hhinc#3', 'MU_car'): 360003.5818421476,
					('hhinc#3', 'MU_motor'): -903.7481484499674,
					('hhinc#3', 'MU_nonmotor'): -199.09190169318276,
					('hhinc#3', 'hhinc#2'): -1013597.9862056787,
					('hhinc#3', 'hhinc#3'): 4930334.737718876,
					('hhinc#3', 'hhinc#4'): -400182.3211494456,
					('hhinc#3', 'hhinc#5'): -11135.30468889834,
					('hhinc#3', 'hhinc#6'): -20821.937205379018,
					('hhinc#3', 'totcost'): -12658608.701901834,
					('hhinc#3', 'totcost+50'): -3426.924658741228,
					('hhinc#3', 'tottime'): 122585.36115672352,
					('hhinc#3', 'tottime+100'): 3426.9246587412163,
					('hhinc#4', 'ASC_BIKE'): 2266.1904081469224,
					('hhinc#4', 'ASC_SR2'): -321358.88345842547,
					('hhinc#4', 'ASC_SR3P'): -4773.734475201185,
					('hhinc#4', 'ASC_TRAN'): 85170.41064773081,
					('hhinc#4', 'ASC_WALK'): 5004.480689180802,
					('hhinc#4', 'MU_car'): -743754.3177440444,
					('hhinc#4', 'MU_motor'): 23198.692153116197,
					('hhinc#4', 'MU_nonmotor'): 2227.8351942101954,
					('hhinc#4', 'hhinc#2'): -24514725.011886124,
					('hhinc#4', 'hhinc#3'): -400182.3211494456,
					('hhinc#4', 'hhinc#4'): 6289654.822118052,
					('hhinc#4', 'hhinc#5'): 163798.83366643416,
					('hhinc#4', 'hhinc#6'): 414858.3194557155,
					('hhinc#4', 'totcost'): 20263542.996370286,
					('hhinc#4', 'totcost+50'): 12117.951296728346,
					('hhinc#4', 'tottime'): 823184.8075015809,
					('hhinc#4', 'tottime+100'): -12117.95129672835,
					('hhinc#5', 'ASC_BIKE'): 5591.231146752164,
					('hhinc#5', 'ASC_SR2'): -22637.568615942328,
					('hhinc#5', 'ASC_SR3P'): -156.36775653099272,
					('hhinc#5', 'ASC_TRAN'): 2266.1904081469224,
					('hhinc#5', 'ASC_WALK'): -939.3767918137345,
					('hhinc#5', 'MU_car'): -50670.5965232693,
					('hhinc#5', 'MU_motor'): 7039.2191386735185,
					('hhinc#5', 'MU_nonmotor'): 4627.509599608355,
					('hhinc#5', 'hhinc#2'): -1644932.6104088873,
					('hhinc#5', 'hhinc#3'): -11135.30468889834,
					('hhinc#5', 'hhinc#4'): 163798.83366643416,
					('hhinc#5', 'hhinc#5'): 400275.07141990436,
					('hhinc#5', 'hhinc#6'): -65905.33615257757,
					('hhinc#5', 'totcost'): 571656.1922534222,
					('hhinc#5', 'totcost+50'): 469.6990911445654,
					('hhinc#5', 'tottime'): -9734.760755067302,
					('hhinc#5', 'tottime+100'): -469.69909114456794,
					('hhinc#6', 'ASC_BIKE'): -939.3767918137345,
					('hhinc#6', 'ASC_SR2'): -44867.80739151402,
					('hhinc#6', 'ASC_SR3P'): -321.51566832624457,
					('hhinc#6', 'ASC_TRAN'): 5004.480689180803,
					('hhinc#6', 'ASC_WALK'): 13250.69044755796,
					('hhinc#6', 'MU_car'): -91816.7683607486,
					('hhinc#6', 'MU_motor'): 15326.954859509155,
					('hhinc#6', 'MU_nonmotor'): 1076.7067806851492,
					('hhinc#6', 'hhinc#2'): -3350183.704181544,
					('hhinc#6', 'hhinc#3'): -20821.937205379018,
					('hhinc#6', 'hhinc#4'): 414858.3194557155,
					('hhinc#6', 'hhinc#5'): -65905.33615257757,
					('hhinc#6', 'hhinc#6'): 910529.077566162,
					('hhinc#6', 'totcost'): 683167.9617302283,
					('hhinc#6', 'totcost+50'): 331.32394741804126,
					('hhinc#6', 'tottime'): 239751.86229111557,
					('hhinc#6', 'tottime+100'): -331.32394741805484,
					('totcost', 'ASC_BIKE'): 9355.705173389544,
					('totcost', 'ASC_SR2'): -4295397.168565195,
					('totcost', 'ASC_SR3P'): -195005.11873966618,
					('totcost', 'ASC_TRAN'): 334381.6742163283,
					('totcost', 'ASC_WALK'): 8375.85450595358,
					('totcost', 'MU_car'): -10836234.999983579,
					('totcost', 'MU_motor'): 54730.860824187155,
					('totcost', 'MU_nonmotor'): 2863.6842252387814,
					('totcost', 'hhinc#2'): -269715687.05097497,
					('totcost', 'hhinc#3'): -12658608.701901834,
					('totcost', 'hhinc#4'): 20263542.996370286,
					('totcost', 'hhinc#5'): 571656.1922534222,
					('totcost', 'hhinc#6'): 683167.9617302283,
					('totcost', 'totcost'): 896926742.2307765,
					('totcost', 'totcost+50'): 157738.87419232324,
					('totcost', 'tottime'): -10553059.764125746,
					('totcost', 'tottime+100'): -157738.87419232284,
					('totcost+50', 'ASC_BIKE'): 7.264603289573004,
					('totcost+50', 'ASC_SR2'): -1929.997244326729,
					('totcost+50', 'ASC_SR3P'): -57.07300966192541,
					('totcost+50', 'ASC_TRAN'): 215.1438275624966,
					('totcost+50', 'ASC_WALK'): 1.3445414940886116,
					('totcost+50', 'MU_car'): -4829.6923345401465,
					('totcost+50', 'MU_motor'): 44.26890319114984,
					('totcost+50', 'MU_nonmotor'): 3.3721148121600755,
					('totcost+50', 'hhinc#2'): -115285.24163010289,
					('totcost+50', 'hhinc#3'): -3426.924658741228,
					('totcost+50', 'hhinc#4'): 12117.951296728346,
					('totcost+50', 'hhinc#5'): 469.6990911445654,
					('totcost+50', 'hhinc#6'): 331.32394741804126,
					('totcost+50', 'totcost'): 157738.87419232324,
					('totcost+50', 'totcost+50'): 86.06217838421836,
					('totcost+50', 'tottime'): -2852.252457662469,
					('totcost+50', 'tottime+100'): -86.06217838421821,
					('tottime', 'ASC_BIKE'): -232.8517672039079,
					('tottime', 'ASC_SR2'): 54945.16106709013,
					('tottime', 'ASC_SR3P'): 1753.5356217530475,
					('tottime', 'ASC_TRAN'): 15555.576449608358,
					('tottime', 'ASC_WALK'): 4904.039044545349,
					('tottime', 'MU_car'): 151537.51100935883,
					('tottime', 'MU_motor'): 8526.328590702655,
					('tottime', 'MU_nonmotor'): 145.49198209455562,
					('tottime', 'hhinc#2'): 3647886.050161818,
					('tottime', 'hhinc#3'): 122585.36115672352,
					('tottime', 'hhinc#4'): 823184.8075015809,
					('tottime', 'hhinc#5'): -9734.760755067302,
					('tottime', 'hhinc#6'): 239751.86229111557,
					('tottime', 'totcost'): -10553059.764125746,
					('tottime', 'totcost+50'): -2852.252457662469,
					('tottime', 'tottime'): 1270353.9912685396,
					('tottime', 'tottime+100'): 2852.2524576624587,
					('tottime+100', 'ASC_BIKE'): -7.264603289573047,
					('tottime+100', 'ASC_SR2'): 1929.997244326727,
					('tottime+100', 'ASC_SR3P'): 57.07300966192533,
					('tottime+100', 'ASC_TRAN'): -215.14382756249677,
					('tottime+100', 'ASC_WALK'): -1.3445414940888838,
					('tottime+100', 'MU_car'): 4829.692334540142,
					('tottime+100', 'MU_motor'): -44.268903191150415,
					('tottime+100', 'MU_nonmotor'): -3.372114812160218,
					('tottime+100', 'hhinc#2'): 115285.2416301027,
					('tottime+100', 'hhinc#3'): 3426.9246587412163,
					('tottime+100', 'hhinc#4'): -12117.95129672835,
					('tottime+100', 'hhinc#5'): -469.69909114456794,
					('tottime+100', 'hhinc#6'): -331.32394741805484,
					('tottime+100', 'totcost'): -157738.87419232284,
					('tottime+100', 'totcost+50'): -86.06217838421821,
					('tottime+100', 'tottime'): 2852.2524576624587,
					('tottime+100', 'tottime+100'): 86.06217838421819}

	assert dict(ll1.bhhh.unstack()) == approx(bhhh_correct)

	assert dict(ll2.bhhh.unstack()) == approx(bhhh_correct)

	assert dict(ll1.bhhh.unstack()) == approx(dict(ll2.bhhh.unstack()))
	dll_casewise_A = ll2.dll_casewise / j2.weight_normalization
	if j2.data_wt is not None:
		dll_casewise_B = numpy.asarray(dll_casewise_A) / j2.data_wt.values
	else:
		dll_casewise_B = numpy.asarray(dll_casewise_A)
	corrected_bhhh = pandas.DataFrame(
		numpy.dot(dll_casewise_A.T, dll_casewise_B),
		index=ll2.dll.index,
		columns=ll2.dll.index,
	) * j2.weight_normalization
	assert dict(ll1.bhhh.unstack()) == approx(dict(corrected_bhhh.unstack()))

	assert ll2.dll.values == approx(ll2.dll_casewise.sum(0))



def test_weighted_qmnl_bhhh():

	d = MTC()

	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', 'tottime+100', 'totcost+50')
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	df_co2 = pandas.concat([df_co, df_co]).reset_index(drop=True)
	df_ca2 = pandas.concat([df_ca.unstack(), df_ca.unstack()]).reset_index(drop=True).stack()
	df_av2 = pandas.concat([df_av.unstack(), df_av.unstack()]).reset_index(drop=True).stack()

	df_chX = pandas.DataFrame(
		numpy.zeros_like(df_ch.values),
		index=df_ch.index,
		columns=df_ch.columns,
	)
	df_chX = df_chX.unstack()
	df_chX.iloc[:, 1] = 2.0
	df_chX = df_chX.stack()

	df_ch2 = pandas.concat([df_ch.unstack(), df_chX.unstack()]).reset_index(drop=True).stack()

	from larch import DataFrames, Model

	j1 = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_chX + df_ch,
	)

	j2 = DataFrames(
		co=df_co2,
		ca=df_ca2,
		av=df_av2,
		ch=df_ch2,
	)

	j1.autoscale_weights()
	j2.autoscale_weights()

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")

	m5.quantity_ca = PX("tottime+100") + PX("totcost+50")

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.01862990704919887,
		'totcost+50': 2,
	}
	m5.pf_sort()

	m5.dataframes = j1
	ll1 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)
	assert m5.check_d_loglike().data.similarity.min() > 4

	m5.dataframes = j2
	ll2 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)
	assert m5.check_d_loglike().data.similarity.min() > 4

	q1_dll = {'ASC_BIKE': -292.0559358088135,
			  'ASC_SR2': 7378.46798009938,
			  'ASC_SR3P': -397.5109055024128,
			  'ASC_TRAN': -2653.878092601135,
			  'ASC_WALK': -433.5897153403382,
			  'hhinc#2': 433077.76185866666,
			  'hhinc#3': -24972.656543299872,
			  'hhinc#4': -148369.2076480757,
			  'hhinc#5': -16880.561260504124,
			  'hhinc#6': -24178.351295438788,
			  'totcost': -610004.4777740919,
			  'totcost+50': -208.02502194759336,
			  'tottime': -67014.4098307907,
			  'tottime+100': 208.0250219475932}

	assert (j1.weight_normalization, j2.weight_normalization) == (3.0, 1.5)

	assert (ll1.ll, ll2.ll) == approx((-20857.82673588217, -20857.82673588217))
	assert (ll1.ll == approx(ll2.ll))

	dict_ll1_dll = dict(ll1.dll)
	dict_ll2_dll = dict(ll2.dll)

	for k in q1_dll:
		assert q1_dll[k] == approx(dict_ll1_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll1_dll[k]}"
		assert q1_dll[k] == approx(dict_ll2_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll2_dll[k]}"

	bhhh_correct = {('ASC_BIKE', 'ASC_BIKE'): 66.91075132687081,
					('ASC_BIKE', 'ASC_SR2'): -185.15587784484097,
					('ASC_BIKE', 'ASC_SR3P'): 7.6987347944623234,
					('ASC_BIKE', 'ASC_TRAN'): 63.154380211164266,
					('ASC_BIKE', 'ASC_WALK'): 14.809804862366414,
					('ASC_BIKE', 'hhinc#2'): -10373.309023994574,
					('ASC_BIKE', 'hhinc#3'): 482.8744118883239,
					('ASC_BIKE', 'hhinc#4'): 3669.7005738878643,
					('ASC_BIKE', 'hhinc#5'): 3389.5845242269074,
					('ASC_BIKE', 'hhinc#6'): 894.3140912350091,
					('ASC_BIKE', 'totcost'): 3542.96431647177,
					('ASC_BIKE', 'totcost+50'): -0.8747499258668625,
					('ASC_BIKE', 'tottime'): 2194.272052073627,
					('ASC_BIKE', 'tottime+100'): 0.8747499258668603,
					('ASC_SR2', 'ASC_BIKE'): -185.15587784484097,
					('ASC_SR2', 'ASC_SR2'): 6810.314490385717,
					('ASC_SR2', 'ASC_SR3P'): -303.6901755050966,
					('ASC_SR2', 'ASC_TRAN'): -1707.0014929313056,
					('ASC_SR2', 'ASC_WALK'): -348.55369969799017,
					('ASC_SR2', 'hhinc#2'): 399901.6858005815,
					('ASC_SR2', 'hhinc#3'): -18897.04722837613,
					('ASC_SR2', 'hhinc#4'): -95043.73259866357,
					('ASC_SR2', 'hhinc#5'): -10373.309023994574,
					('ASC_SR2', 'hhinc#6'): -18057.76204507422,
					('ASC_SR2', 'totcost'): -433546.6306182172,
					('ASC_SR2', 'totcost+50'): -189.66773767206283,
					('ASC_SR2', 'tottime'): -29371.60836960929,
					('ASC_SR2', 'tottime+100'): 189.6677376720628,
					('ASC_SR3P', 'ASC_BIKE'): 7.6987347944623234,
					('ASC_SR3P', 'ASC_SR2'): -303.6901755050966,
					('ASC_SR3P', 'ASC_SR3P'): 170.24420513413526,
					('ASC_SR3P', 'ASC_TRAN'): 57.93297644913393,
					('ASC_SR3P', 'ASC_WALK'): 9.495048606638454,
					('ASC_SR3P', 'hhinc#2'): -18897.047228376134,
					('ASC_SR3P', 'hhinc#3'): 10311.170089182635,
					('ASC_SR3P', 'hhinc#4'): 3490.8262820366513,
					('ASC_SR3P', 'hhinc#5'): 482.8744118883239,
					('ASC_SR3P', 'hhinc#6'): 619.677029595399,
					('ASC_SR3P', 'totcost'): -2145.3240873303675,
					('ASC_SR3P', 'totcost+50'): -0.8124860122548567,
					('ASC_SR3P', 'tottime'): 1938.9430718655804,
					('ASC_SR3P', 'tottime+100'): 0.8124860122548574,
					('ASC_TRAN', 'ASC_BIKE'): 63.154380211164266,
					('ASC_TRAN', 'ASC_SR2'): -1707.0014929313056,
					('ASC_TRAN', 'ASC_SR3P'): 57.93297644913393,
					('ASC_TRAN', 'ASC_TRAN'): 1121.712708287138,
					('ASC_TRAN', 'ASC_WALK'): 104.67304222158444,
					('ASC_TRAN', 'hhinc#2'): -95043.73259866357,
					('ASC_TRAN', 'hhinc#3'): 3490.8262820366517,
					('ASC_TRAN', 'hhinc#4'): 60885.532585972,
					('ASC_TRAN', 'hhinc#5'): 3669.7005738878643,
					('ASC_TRAN', 'hhinc#6'): 5941.0155512443525,
					('ASC_TRAN', 'totcost'): 42303.495419779254,
					('ASC_TRAN', 'totcost+50'): 43.11029487377358,
					('ASC_TRAN', 'tottime'): 24166.959138438546,
					('ASC_TRAN', 'tottime+100'): -43.110294873773576,
					('ASC_WALK', 'ASC_BIKE'): 14.809804862366414,
					('ASC_WALK', 'ASC_SR2'): -348.55369969799017,
					('ASC_WALK', 'ASC_SR3P'): 9.495048606638454,
					('ASC_WALK', 'ASC_TRAN'): 104.67304222158444,
					('ASC_WALK', 'ASC_WALK'): 199.42006623502635,
					('ASC_WALK', 'hhinc#2'): -18057.76204507422,
					('ASC_WALK', 'hhinc#3'): 619.6770295953988,
					('ASC_WALK', 'hhinc#4'): 5941.015551244353,
					('ASC_WALK', 'hhinc#5'): 894.3140912350091,
					('ASC_WALK', 'hhinc#6'): 9467.82678028681,
					('ASC_WALK', 'totcost'): 1651.3092167772688,
					('ASC_WALK', 'totcost+50'): -7.67733274866181,
					('ASC_WALK', 'tottime'): 5782.86787395335,
					('ASC_WALK', 'tottime+100'): 7.6773327486618115,
					('hhinc#2', 'ASC_BIKE'): -10373.309023994574,
					('hhinc#2', 'ASC_SR2'): 399901.6858005815,
					('hhinc#2', 'ASC_SR3P'): -18897.047228376134,
					('hhinc#2', 'ASC_TRAN'): -95043.73259866357,
					('hhinc#2', 'ASC_WALK'): -18057.76204507422,
					('hhinc#2', 'hhinc#2'): 30733950.484026298,
					('hhinc#2', 'hhinc#3'): -1535702.1027311594,
					('hhinc#2', 'hhinc#4'): -7117026.512948193,
					('hhinc#2', 'hhinc#5'): -760393.3992416217,
					('hhinc#2', 'hhinc#6'): -1314494.2781731542,
					('hhinc#2', 'totcost'): -26868413.738335926,
					('hhinc#2', 'totcost+50'): -11242.201853442557,
					('hhinc#2', 'tottime'): -1578291.4214597486,
					('hhinc#2', 'tottime+100'): 11242.201853442557,
					('hhinc#3', 'ASC_BIKE'): 482.8744118883239,
					('hhinc#3', 'ASC_SR2'): -18897.04722837613,
					('hhinc#3', 'ASC_SR3P'): 10311.170089182635,
					('hhinc#3', 'ASC_TRAN'): 3490.8262820366517,
					('hhinc#3', 'ASC_WALK'): 619.6770295953988,
					('hhinc#3', 'hhinc#2'): -1535702.1027311594,
					('hhinc#3', 'hhinc#3'): 843271.532858375,
					('hhinc#3', 'hhinc#4'): 270709.677436316,
					('hhinc#3', 'hhinc#5'): 37607.738269849986,
					('hhinc#3', 'hhinc#6'): 52276.853686747025,
					('hhinc#3', 'totcost'): -94350.12448577741,
					('hhinc#3', 'totcost+50'): -28.422043443548695,
					('hhinc#3', 'tottime'): 117277.26825272018,
					('hhinc#3', 'tottime+100'): 28.42204344354878,
					('hhinc#4', 'ASC_BIKE'): 3669.7005738878643,
					('hhinc#4', 'ASC_SR2'): -95043.73259866357,
					('hhinc#4', 'ASC_SR3P'): 3490.8262820366513,
					('hhinc#4', 'ASC_TRAN'): 60885.532585972,
					('hhinc#4', 'ASC_WALK'): 5941.015551244353,
					('hhinc#4', 'hhinc#2'): -7117026.512948193,
					('hhinc#4', 'hhinc#3'): 270709.677436316,
					('hhinc#4', 'hhinc#4'): 4478989.131115964,
					('hhinc#4', 'hhinc#5'): 270429.912372026,
					('hhinc#4', 'hhinc#6'): 458265.5895676509,
					('hhinc#4', 'totcost'): 2140975.7584440005,
					('hhinc#4', 'totcost+50'): 2225.492327040068,
					('hhinc#4', 'tottime'): 1338412.7179203192,
					('hhinc#4', 'tottime+100'): -2225.4923270400677,
					('hhinc#5', 'ASC_BIKE'): 3389.5845242269074,
					('hhinc#5', 'ASC_SR2'): -10373.309023994574,
					('hhinc#5', 'ASC_SR3P'): 482.8744118883239,
					('hhinc#5', 'ASC_TRAN'): 3669.7005738878643,
					('hhinc#5', 'ASC_WALK'): 894.3140912350091,
					('hhinc#5', 'hhinc#2'): -760393.3992416217,
					('hhinc#5', 'hhinc#3'): 37607.738269849986,
					('hhinc#5', 'hhinc#4'): 270429.912372026,
					('hhinc#5', 'hhinc#5'): 242283.98531011157,
					('hhinc#5', 'hhinc#6'): 67803.3191437671,
					('hhinc#5', 'totcost'): 223315.19581432926,
					('hhinc#5', 'totcost+50'): -11.891020264044968,
					('hhinc#5', 'tottime'): 123754.02394542051,
					('hhinc#5', 'tottime+100'): 11.89102026404496,
					('hhinc#6', 'ASC_BIKE'): 894.3140912350091,
					('hhinc#6', 'ASC_SR2'): -18057.76204507422,
					('hhinc#6', 'ASC_SR3P'): 619.677029595399,
					('hhinc#6', 'ASC_TRAN'): 5941.0155512443525,
					('hhinc#6', 'ASC_WALK'): 9467.82678028681,
					('hhinc#6', 'hhinc#2'): -1314494.2781731542,
					('hhinc#6', 'hhinc#3'): 52276.853686747025,
					('hhinc#6', 'hhinc#4'): 458265.5895676509,
					('hhinc#6', 'hhinc#5'): 67803.3191437671,
					('hhinc#6', 'hhinc#6'): 652774.8948963538,
					('hhinc#6', 'totcost'): 225622.6878936836,
					('hhinc#6', 'totcost+50'): -260.8841609726519,
					('hhinc#6', 'tottime'): 295018.1931756011,
					('hhinc#6', 'tottime+100'): 260.88416097265196,
					('totcost', 'ASC_BIKE'): 3542.96431647177,
					('totcost', 'ASC_SR2'): -433546.6306182172,
					('totcost', 'ASC_SR3P'): -2145.3240873303675,
					('totcost', 'ASC_TRAN'): 42303.495419779254,
					('totcost', 'ASC_WALK'): 1651.3092167772688,
					('totcost', 'hhinc#2'): -26868413.738335926,
					('totcost', 'hhinc#3'): -94350.12448577741,
					('totcost', 'hhinc#4'): 2140975.7584440005,
					('totcost', 'hhinc#5'): 223315.19581432926,
					('totcost', 'hhinc#6'): 225622.6878936836,
					('totcost', 'totcost'): 106373969.23982364,
					('totcost', 'totcost+50'): 20563.918169569974,
					('totcost', 'tottime'): -103140.63602656369,
					('totcost', 'tottime+100'): -20563.918169569974,
					('totcost+50', 'ASC_BIKE'): -0.8747499258668625,
					('totcost+50', 'ASC_SR2'): -189.66773767206283,
					('totcost+50', 'ASC_SR3P'): -0.8124860122548567,
					('totcost+50', 'ASC_TRAN'): 43.11029487377358,
					('totcost+50', 'ASC_WALK'): -7.67733274866181,
					('totcost+50', 'hhinc#2'): -11242.201853442557,
					('totcost+50', 'hhinc#3'): -28.422043443548695,
					('totcost+50', 'hhinc#4'): 2225.492327040068,
					('totcost+50', 'hhinc#5'): -11.891020264044968,
					('totcost+50', 'hhinc#6'): -260.8841609726519,
					('totcost+50', 'totcost'): 20563.918169569974,
					('totcost+50', 'totcost+50'): 12.154094937459242,
					('totcost+50', 'tottime'): 1.0226330657446017,
					('totcost+50', 'tottime+100'): -12.154094937459242,
					('tottime', 'ASC_BIKE'): 2194.272052073627,
					('tottime', 'ASC_SR2'): -29371.60836960929,
					('tottime', 'ASC_SR3P'): 1938.9430718655804,
					('tottime', 'ASC_TRAN'): 24166.959138438546,
					('tottime', 'ASC_WALK'): 5782.86787395335,
					('tottime', 'hhinc#2'): -1578291.4214597486,
					('tottime', 'hhinc#3'): 117277.26825272018,
					('tottime', 'hhinc#4'): 1338412.7179203192,
					('tottime', 'hhinc#5'): 123754.02394542051,
					('tottime', 'hhinc#6'): 295018.1931756011,
					('tottime', 'totcost'): -103140.63602656369,
					('tottime', 'totcost+50'): 1.0226330657446017,
					('tottime', 'tottime'): 1044910.2979153896,
					('tottime', 'tottime+100'): -1.0226330657444684,
					('tottime+100', 'ASC_BIKE'): 0.8747499258668603,
					('tottime+100', 'ASC_SR2'): 189.6677376720628,
					('tottime+100', 'ASC_SR3P'): 0.8124860122548574,
					('tottime+100', 'ASC_TRAN'): -43.110294873773576,
					('tottime+100', 'ASC_WALK'): 7.6773327486618115,
					('tottime+100', 'hhinc#2'): 11242.201853442557,
					('tottime+100', 'hhinc#3'): 28.42204344354878,
					('tottime+100', 'hhinc#4'): -2225.4923270400677,
					('tottime+100', 'hhinc#5'): 11.89102026404496,
					('tottime+100', 'hhinc#6'): 260.88416097265196,
					('tottime+100', 'totcost'): -20563.918169569974,
					('tottime+100', 'totcost+50'): -12.154094937459242,
					('tottime+100', 'tottime'): -1.0226330657444684,
					('tottime+100', 'tottime+100'): 12.154094937459242}

	assert dict(ll1.bhhh.unstack()) == approx(bhhh_correct)
	assert dict(ll2.bhhh.unstack()) == approx(bhhh_correct)
	assert dict(ll1.bhhh.unstack()) == approx(dict(ll2.bhhh.unstack()))
	assert ll2.dll.sort_index().values == approx(ll2.dll_casewise.sum(0).sort_index().values)

	dll_casewise_A = ll2.dll_casewise / j2.weight_normalization
	if j2.data_wt is not None:
		dll_casewise_B = numpy.asarray(dll_casewise_A) / j2.data_wt.values
	else:
		dll_casewise_B = numpy.asarray(dll_casewise_A)

	corrected_bhhh = pandas.DataFrame(
		numpy.dot(dll_casewise_A.T, dll_casewise_B),
		index=ll2.dll.index,
		columns=ll2.dll.index,
	) * j2.weight_normalization

	assert dict(ll1.bhhh.unstack()) == approx(dict(corrected_bhhh.unstack()))

def test_dataframes_mnl_with_zero_weighteds():
	d = MTC()

	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', )
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	df_co2 = pandas.concat([df_co, df_co]).reset_index(drop=True)
	df_ca2 = pandas.concat([df_ca.unstack(), df_ca.unstack()]).reset_index(drop=True).stack()
	df_av2 = pandas.concat([df_av.unstack(), df_av.unstack()]).reset_index(drop=True).stack()

	df_chX = pandas.DataFrame(
		numpy.zeros_like(df_ch.values),
		index=df_ch.index,
		columns=df_ch.columns,
	)
	df_chX = df_chX.unstack()
	df_chX.iloc[::2, 1] = 3.0

	df_chX = df_chX.stack()

	df_ch2 = pandas.concat([df_ch.unstack(), df_chX.unstack()]).reset_index(drop=True).stack()

	from larch import DataFrames, Model

	j1 = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_chX + df_ch,
	)

	j2 = DataFrames(
		co=df_co2,
		ca=df_ca2,
		av=df_av2,
		ch=df_ch2,
	)

	j1.autoscale_weights()
	j2.autoscale_weights()

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.01862990704919887,
	}

	m5.pf_sort()

	m5.dataframes = j1
	ll1 = m5.loglike2_bhhh(beta_in1, return_series=True)

	m5.dataframes = j2
	ll2 = m5.loglike2_bhhh(beta_in1, return_series=True)

	q1_dll = {
		'ASC_BIKE': -428.39753611030653,
		'ASC_SR2': 4810.2986856048,
		'ASC_SR3P': -556.7833361373218,
		'ASC_TRAN': -1655.85403447997,
		'ASC_WALK': -533.7822967238249,
		'hhinc#2': 281968.0302440858,
		'hhinc#3': -35336.670487581134,
		'hhinc#4': -95004.22107716063,
		'hhinc#5': -24668.281182428742,
		'hhinc#6': -28861.99492516596,
		'totcost': -199887.06056428683,
		'tottime': -56631.935335538095,
	}

	assert (j1.weight_normalization, j2.weight_normalization) == (2.500298270033804, 1.250149135016902)
	assert (ll1.ll, ll2.ll) == approx((-15400.565204261877, -15400.565204261879))

	dict_ll1_dll = dict(ll1.dll)
	dict_ll2_dll = dict(ll2.dll)

	for k in q1_dll:
		assert q1_dll[k] == approx(dict_ll1_dll[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict_ll1_dll[k]}"
		assert q1_dll[k] == approx(dict_ll2_dll[k], rel=1e-5), f"{k} {q1_dll[k]} != {dict_ll2_dll[k]}"

	bhhh_correct = {('ASC_BIKE', 'ASC_BIKE'): 92.14674685121092,
					('ASC_BIKE', 'ASC_SR2'): -214.77177905341645,
					('ASC_BIKE', 'ASC_SR3P'): 15.6572869432314,
					('ASC_BIKE', 'ASC_TRAN'): 60.69855333216971,
					('ASC_BIKE', 'ASC_WALK'): 24.880666629112287,
					('ASC_BIKE', 'hhinc#2'): -12042.623211131993,
					('ASC_BIKE', 'hhinc#3'): 985.3263248637774,
					('ASC_BIKE', 'hhinc#4'): 3548.37701471491,
					('ASC_BIKE', 'hhinc#5'): 4775.738707168541,
					('ASC_BIKE', 'hhinc#6'): 1438.62133398825,
					('ASC_BIKE', 'totcost'): -316.061958396488,
					('ASC_BIKE', 'tottime'): 2908.015222444746,
					('ASC_SR2', 'ASC_BIKE'): -214.77177905341645,
					('ASC_SR2', 'ASC_SR2'): 4784.169674201138,
					('ASC_SR2', 'ASC_SR3P'): -307.21395215556345,
					('ASC_SR2', 'ASC_TRAN'): -1001.2336976830427,
					('ASC_SR2', 'ASC_WALK'): -343.9454313966797,
					('ASC_SR2', 'hhinc#2'): 280239.6455367927,
					('ASC_SR2', 'hhinc#3'): -19231.5591262376,
					('ASC_SR2', 'hhinc#4'): -56480.539084738346,
					('ASC_SR2', 'hhinc#5'): -12042.623211131993,
					('ASC_SR2', 'hhinc#6'): -17460.094996712465,
					('ASC_SR2', 'totcost'): -199826.47315304854,
					('ASC_SR2', 'tottime'): -18784.75554420598,
					('ASC_SR3P', 'ASC_BIKE'): 15.6572869432314,
					('ASC_SR3P', 'ASC_SR2'): -307.21395215556345,
					('ASC_SR3P', 'ASC_SR3P'): 184.71800513554535,
					('ASC_SR3P', 'ASC_TRAN'): 53.375377775237865,
					('ASC_SR3P', 'ASC_WALK'): 15.232846933314944,
					('ASC_SR3P', 'hhinc#2'): -19231.559126237604,
					('ASC_SR3P', 'hhinc#3'): 11330.643263827798,
					('ASC_SR3P', 'hhinc#4'): 3336.3522607049913,
					('ASC_SR3P', 'hhinc#5'): 985.3263248637774,
					('ASC_SR3P', 'hhinc#6'): 941.6421093048237,
					('ASC_SR3P', 'totcost'): -6569.840872617441,
					('ASC_SR3P', 'tottime'): 2366.8827123972787,
					('ASC_TRAN', 'ASC_BIKE'): 60.69855333216971,
					('ASC_TRAN', 'ASC_SR2'): -1001.2336976830427,
					('ASC_TRAN', 'ASC_SR3P'): 53.375377775237865,
					('ASC_TRAN', 'ASC_TRAN'): 690.2173239963141,
					('ASC_TRAN', 'ASC_WALK'): 75.11356593453685,
					('ASC_TRAN', 'hhinc#2'): -56480.539084738346,
					('ASC_TRAN', 'hhinc#3'): 3336.3522607049913,
					('ASC_TRAN', 'hhinc#4'): 38237.073388094395,
					('ASC_TRAN', 'hhinc#5'): 3548.377014714909,
					('ASC_TRAN', 'hhinc#6'): 4157.749919273613,
					('ASC_TRAN', 'totcost'): -6168.605964427145,
					('ASC_TRAN', 'tottime'): 15623.261962127483,
					('ASC_WALK', 'ASC_BIKE'): 24.880666629112287,
					('ASC_WALK', 'ASC_SR2'): -343.9454313966797,
					('ASC_WALK', 'ASC_SR3P'): 15.232846933314944,
					('ASC_WALK', 'ASC_TRAN'): 75.11356593453685,
					('ASC_WALK', 'ASC_WALK'): 225.71543900823264,
					('ASC_WALK', 'hhinc#2'): -17460.094996712465,
					('ASC_WALK', 'hhinc#3'): 941.6421093048237,
					('ASC_WALK', 'hhinc#4'): 4157.749919273613,
					('ASC_WALK', 'hhinc#5'): 1438.6213339882497,
					('ASC_WALK', 'hhinc#6'): 10844.633089355342,
					('ASC_WALK', 'totcost'): -4080.0845919275107,
					('ASC_WALK', 'tottime'): 6231.749962852353,
					('hhinc#2', 'ASC_BIKE'): -12042.623211131993,
					('hhinc#2', 'ASC_SR2'): 280239.6455367927,
					('hhinc#2', 'ASC_SR3P'): -19231.559126237604,
					('hhinc#2', 'ASC_TRAN'): -56480.539084738346,
					('hhinc#2', 'ASC_WALK'): -17460.094996712465,
					('hhinc#2', 'hhinc#2'): 21426222.45878453,
					('hhinc#2', 'hhinc#3'): -1559656.9258410155,
					('hhinc#2', 'hhinc#4'): -4234166.185057233,
					('hhinc#2', 'hhinc#5'): -881184.5330974476,
					('hhinc#2', 'hhinc#6'): -1248129.3832445068,
					('hhinc#2', 'totcost'): -12593729.784063647,
					('hhinc#2', 'tottime'): -991882.2741298374,
					('hhinc#3', 'ASC_BIKE'): 985.3263248637774,
					('hhinc#3', 'ASC_SR2'): -19231.5591262376,
					('hhinc#3', 'ASC_SR3P'): 11330.643263827798,
					('hhinc#3', 'ASC_TRAN'): 3336.3522607049913,
					('hhinc#3', 'ASC_WALK'): 941.6421093048237,
					('hhinc#3', 'hhinc#2'): -1559656.9258410155,
					('hhinc#3', 'hhinc#3'): 931288.4286490775,
					('hhinc#3', 'hhinc#4'): 258309.52813089255,
					('hhinc#3', 'hhinc#5'): 77655.35926709934,
					('hhinc#3', 'hhinc#6'): 76888.19770643165,
					('hhinc#3', 'totcost'): -383519.0058637618,
					('hhinc#3', 'tottime'): 144418.90354394988,
					('hhinc#4', 'ASC_BIKE'): 3548.37701471491,
					('hhinc#4', 'ASC_SR2'): -56480.539084738346,
					('hhinc#4', 'ASC_SR3P'): 3336.3522607049913,
					('hhinc#4', 'ASC_TRAN'): 38237.073388094395,
					('hhinc#4', 'ASC_WALK'): 4157.749919273613,
					('hhinc#4', 'hhinc#2'): -4234166.185057233,
					('hhinc#4', 'hhinc#3'): 258309.52813089255,
					('hhinc#4', 'hhinc#4'): 2838037.4107418098,
					('hhinc#4', 'hhinc#5'): 263422.19380452204,
					('hhinc#4', 'hhinc#6'): 315389.62732528825,
					('hhinc#4', 'totcost'): -515243.3726198388,
					('hhinc#4', 'tottime'): 879572.8534793495,
					('hhinc#5', 'ASC_BIKE'): 4775.738707168541,
					('hhinc#5', 'ASC_SR2'): -12042.623211131993,
					('hhinc#5', 'ASC_SR3P'): 985.3263248637774,
					('hhinc#5', 'ASC_TRAN'): 3548.377014714909,
					('hhinc#5', 'ASC_WALK'): 1438.6213339882497,
					('hhinc#5', 'hhinc#2'): -881184.5330974476,
					('hhinc#5', 'hhinc#3'): 77655.35926709934,
					('hhinc#5', 'hhinc#4'): 263422.19380452204,
					('hhinc#5', 'hhinc#5'): 338956.0014128014,
					('hhinc#5', 'hhinc#6'): 105989.9077508439,
					('hhinc#5', 'totcost'): 24087.54528209317,
					('hhinc#5', 'tottime'): 163613.54174492488,
					('hhinc#6', 'ASC_BIKE'): 1438.62133398825,
					('hhinc#6', 'ASC_SR2'): -17460.094996712465,
					('hhinc#6', 'ASC_SR3P'): 941.6421093048237,
					('hhinc#6', 'ASC_TRAN'): 4157.749919273613,
					('hhinc#6', 'ASC_WALK'): 10844.633089355342,
					('hhinc#6', 'hhinc#2'): -1248129.3832445068,
					('hhinc#6', 'hhinc#3'): 76888.19770643165,
					('hhinc#6', 'hhinc#4'): 315389.62732528825,
					('hhinc#6', 'hhinc#5'): 105989.9077508439,
					('hhinc#6', 'hhinc#6'): 754014.6065718601,
					('hhinc#6', 'totcost'): -50932.53314973853,
					('hhinc#6', 'tottime'): 316800.1064162473,
					('totcost', 'ASC_BIKE'): -316.061958396488,
					('totcost', 'ASC_SR2'): -199826.47315304854,
					('totcost', 'ASC_SR3P'): -6569.840872617441,
					('totcost', 'ASC_TRAN'): -6168.605964427145,
					('totcost', 'ASC_WALK'): -4080.0845919275107,
					('totcost', 'hhinc#2'): -12593729.784063647,
					('totcost', 'hhinc#3'): -383519.0058637618,
					('totcost', 'hhinc#4'): -515243.3726198388,
					('totcost', 'hhinc#5'): 24087.54528209317,
					('totcost', 'hhinc#6'): -50932.53314973853,
					('totcost', 'totcost'): 58991615.55748509,
					('totcost', 'tottime'): -869052.0745660642,
					('tottime', 'ASC_BIKE'): 2908.015222444746,
					('tottime', 'ASC_SR2'): -18784.75554420598,
					('tottime', 'ASC_SR3P'): 2366.8827123972787,
					('tottime', 'ASC_TRAN'): 15623.261962127483,
					('tottime', 'ASC_WALK'): 6231.749962852353,
					('tottime', 'hhinc#2'): -991882.2741298374,
					('tottime', 'hhinc#3'): 144418.90354394988,
					('tottime', 'hhinc#4'): 879572.8534793495,
					('tottime', 'hhinc#5'): 163613.54174492488,
					('tottime', 'hhinc#6'): 316800.1064162473,
					('tottime', 'totcost'): -869052.0745660642,
					('tottime', 'tottime'): 821704.3790813853}

	assert dict(ll1.bhhh.unstack()) == approx(bhhh_correct)
	assert dict(ll2.bhhh.unstack()) == approx(bhhh_correct)

	assert m5.check_d_loglike().data.similarity.min() > 4

def test_dataframes_holdfast_1():

	d = MTC()

	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', 'tottime+100', 'totcost+50')
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	df_co2 = pandas.concat([df_co, df_co]).reset_index(drop=True)
	df_ca2 = pandas.concat([df_ca.unstack(), df_ca.unstack()]).reset_index(drop=True).stack()
	df_av2 = pandas.concat([df_av.unstack(), df_av.unstack()]).reset_index(drop=True).stack()

	df_chX = pandas.DataFrame(
		numpy.zeros_like(df_ch.values),
		index=df_ch.index,
		columns=df_ch.columns,
	)
	df_chX = df_chX.unstack()
	df_chX.iloc[:, 1] = 2.0
	df_chX = df_chX.stack()

	df_ch2 = pandas.concat([df_ch.unstack(), df_chX.unstack()]).reset_index(drop=True).stack()

	from larch import DataFrames, Model

	j1 = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_chX + df_ch,
	)

	j2 = DataFrames(
		co=df_co2,
		ca=df_ca2,
		av=df_av2,
		ch=df_ch2,
	)

	j1.autoscale_weights()
	j2.autoscale_weights()

	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")
	m5.quantity_ca = PX("tottime+100") + PX("totcost+50")

	m5.quantity_scale = P('THETA')

	m5.lock_value('tottime', -0.01862990704919887)
	m5.lock_value('tottime+100', 0)

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.123,
		'totcost+50': 2,
	}

	m5.pf_sort()

	assert j1.computational
	m5.dataframes = j1
	ll1 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)

	assert m5.check_d_loglike().data.similarity.min() > 4

	assert j2.computational
	m5.dataframes = j2
	ll2 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)
	similarity = m5.check_d_loglike().data.similarity.min()
	assert similarity > 4

	q1_dll = {'ASC_BIKE': -292.0559358088135,
			  'ASC_SR2': 7378.46798009938,
			  'ASC_SR3P': -397.5109055024128,
			  'ASC_TRAN': -2653.878092601135,
			  'ASC_WALK': -433.5897153403382,
			  'hhinc#2': 433077.76185866666,
			  'hhinc#3': -24972.656543299872,
			  'hhinc#4': -148369.2076480757,
			  'hhinc#5': -16880.561260504124,
			  'hhinc#6': -24178.351295438788,
			  'totcost': -610004.4777740919,
			  'totcost+50': -208.02502194759336,
			  'THETA': -2717.0267875558216,
			  'tottime': 0,
			  'tottime+100': 0}


	assert (j1.weight_normalization, j2.weight_normalization) == (3.0, 1.5)
	assert (ll1.ll, ll2.ll) == approx((-20857.82673588217, -20857.82673588217))

	assert (ll1.ll == approx(ll2.ll))

	dict_ll1_dll = dict(ll1.dll)
	dict_ll2_dll = dict(ll2.dll)


	for k in q1_dll:
		assert q1_dll[k] == approx(dict_ll1_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll1_dll[k]}"
		assert q1_dll[k] == approx(dict_ll2_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll2_dll[k]}"
	bhhh_correct = {
		('ASC_BIKE', 'ASC_BIKE'): 66.91075132687075,
		('ASC_BIKE', 'ASC_SR2'): -185.155877844841,
		('ASC_BIKE', 'ASC_SR3P'): 7.698734794462322,
		('ASC_BIKE', 'ASC_TRAN'): 63.154380211164245,
		('ASC_BIKE', 'ASC_WALK'): 14.809804862366414,
		('ASC_BIKE', 'THETA'): 22.76125438365778,
		('ASC_BIKE', 'hhinc#2'): -10373.30902399457,
		('ASC_BIKE', 'hhinc#3'): 482.8744118883242,
		('ASC_BIKE', 'hhinc#4'): 3669.7005738878634,
		('ASC_BIKE', 'hhinc#5'): 3389.584524226908,
		('ASC_BIKE', 'hhinc#6'): 894.3140912350088,
		('ASC_BIKE', 'totcost'): 3542.9643164717727,
		('ASC_BIKE', 'totcost+50'): -0.8747499258668594,
		('ASC_BIKE', 'tottime'): 0.0,
		('ASC_BIKE', 'tottime+100'): 0.0,
		('ASC_SR2', 'ASC_BIKE'): -185.155877844841,
		('ASC_SR2', 'ASC_SR2'): 6810.314490385714,
		('ASC_SR2', 'ASC_SR3P'): -303.69017550509665,
		('ASC_SR2', 'ASC_TRAN'): -1707.0014929313043,
		('ASC_SR2', 'ASC_WALK'): -348.5536996979901,
		('ASC_SR2', 'THETA'): -2066.233888885031,
		('ASC_SR2', 'hhinc#2'): 399901.6858005817,
		('ASC_SR2', 'hhinc#3'): -18897.04722837615,
		('ASC_SR2', 'hhinc#4'): -95043.73259866364,
		('ASC_SR2', 'hhinc#5'): -10373.30902399457,
		('ASC_SR2', 'hhinc#6'): -18057.762045074225,
		('ASC_SR2', 'totcost'): -433546.6306182174,
		('ASC_SR2', 'totcost+50'): -189.66773767206286,
		('ASC_SR2', 'tottime'): 0.0,
		('ASC_SR2', 'tottime+100'): 0.0,
		('ASC_SR3P', 'ASC_BIKE'): 7.698734794462322,
		('ASC_SR3P', 'ASC_SR2'): -303.69017550509665,
		('ASC_SR3P', 'ASC_SR3P'): 170.2442051341352,
		('ASC_SR3P', 'ASC_TRAN'): 57.93297644913396,
		('ASC_SR3P', 'ASC_WALK'): 9.495048606638454,
		('ASC_SR3P', 'THETA'): 7.028166871649721,
		('ASC_SR3P', 'hhinc#2'): -18897.04722837615,
		('ASC_SR3P', 'hhinc#3'): 10311.170089182642,
		('ASC_SR3P', 'hhinc#4'): 3490.8262820366563,
		('ASC_SR3P', 'hhinc#5'): 482.8744118883242,
		('ASC_SR3P', 'hhinc#6'): 619.677029595399,
		('ASC_SR3P', 'totcost'): -2145.3240873303475,
		('ASC_SR3P', 'totcost+50'): -0.8124860122548639,
		('ASC_SR3P', 'tottime'): 0.0,
		('ASC_SR3P', 'tottime+100'): 0.0,
		('ASC_TRAN', 'ASC_BIKE'): 63.154380211164245,
		('ASC_TRAN', 'ASC_SR2'): -1707.0014929313043,
		('ASC_TRAN', 'ASC_SR3P'): 57.93297644913396,
		('ASC_TRAN', 'ASC_TRAN'): 1121.712708287138,
		('ASC_TRAN', 'ASC_WALK'): 104.67304222158444,
		('ASC_TRAN', 'THETA'): 466.2971533223996,
		('ASC_TRAN', 'hhinc#2'): -95043.73259866364,
		('ASC_TRAN', 'hhinc#3'): 3490.8262820366563,
		('ASC_TRAN', 'hhinc#4'): 60885.53258597199,
		('ASC_TRAN', 'hhinc#5'): 3669.7005738878634,
		('ASC_TRAN', 'hhinc#6'): 5941.015551244351,
		('ASC_TRAN', 'totcost'): 42303.49541977925,
		('ASC_TRAN', 'totcost+50'): 43.11029487377359,
		('ASC_TRAN', 'tottime'): 0.0,
		('ASC_TRAN', 'tottime+100'): 0.0,
		('ASC_WALK', 'ASC_BIKE'): 14.809804862366414,
		('ASC_WALK', 'ASC_SR2'): -348.5536996979901,
		('ASC_WALK', 'ASC_SR3P'): 9.495048606638454,
		('ASC_WALK', 'ASC_TRAN'): 104.67304222158444,
		('ASC_WALK', 'ASC_WALK'): 199.42006623502647,
		('ASC_WALK', 'THETA'): 19.706660381788637,
		('ASC_WALK', 'hhinc#2'): -18057.76204507422,
		('ASC_WALK', 'hhinc#3'): 619.677029595399,
		('ASC_WALK', 'hhinc#4'): 5941.015551244351,
		('ASC_WALK', 'hhinc#5'): 894.3140912350088,
		('ASC_WALK', 'hhinc#6'): 9467.826780286812,
		('ASC_WALK', 'totcost'): 1651.3092167772722,
		('ASC_WALK', 'totcost+50'): -7.677332748661808,
		('ASC_WALK', 'tottime'): 0.0,
		('ASC_WALK', 'tottime+100'): 0.0,
		('THETA', 'ASC_BIKE'): 22.76125438365778,
		('THETA', 'ASC_SR2'): -2066.233888885031,
		('THETA', 'ASC_SR3P'): 7.028166871649721,
		('THETA', 'ASC_TRAN'): 466.2971533223996,
		('THETA', 'ASC_WALK'): 19.706660381788637,
		('THETA', 'THETA'): 1139.8401930921243,
		('THETA', 'hhinc#2'): -123154.34044207048,
		('THETA', 'hhinc#3'): 640.569641314502,
		('THETA', 'hhinc#4'): 24655.90610351544,
		('THETA', 'hhinc#5'): 1510.0584523826587,
		('THETA', 'hhinc#6'): 1928.2217854957846,
		('THETA', 'totcost'): 282258.3984123302,
		('THETA', 'totcost+50'): 104.89731164684805,
		('THETA', 'tottime'): 0.0,
		('THETA', 'tottime+100'): 0.0,
		('hhinc#2', 'ASC_BIKE'): -10373.30902399457,
		('hhinc#2', 'ASC_SR2'): 399901.6858005817,
		('hhinc#2', 'ASC_SR3P'): -18897.04722837615,
		('hhinc#2', 'ASC_TRAN'): -95043.73259866364,
		('hhinc#2', 'ASC_WALK'): -18057.76204507422,
		('hhinc#2', 'THETA'): -123154.34044207048,
		('hhinc#2', 'hhinc#2'): 30733950.484026313,
		('hhinc#2', 'hhinc#3'): -1535702.1027311594,
		('hhinc#2', 'hhinc#4'): -7117026.512948193,
		('hhinc#2', 'hhinc#5'): -760393.3992416215,
		('hhinc#2', 'hhinc#6'): -1314494.278173153,
		('hhinc#2', 'totcost'): -26868413.73833593,
		('hhinc#2', 'totcost+50'): -11242.201853442559,
		('hhinc#2', 'tottime'): 0.0,
		('hhinc#2', 'tottime+100'): 0.0,
		('hhinc#3', 'ASC_BIKE'): 482.8744118883242,
		('hhinc#3', 'ASC_SR2'): -18897.04722837615,
		('hhinc#3', 'ASC_SR3P'): 10311.170089182642,
		('hhinc#3', 'ASC_TRAN'): 3490.8262820366563,
		('hhinc#3', 'ASC_WALK'): 619.677029595399,
		('hhinc#3', 'THETA'): 640.569641314502,
		('hhinc#3', 'hhinc#2'): -1535702.1027311594,
		('hhinc#3', 'hhinc#3'): 843271.532858375,
		('hhinc#3', 'hhinc#4'): 270709.67743631615,
		('hhinc#3', 'hhinc#5'): 37607.738269849986,
		('hhinc#3', 'hhinc#6'): 52276.85368674705,
		('hhinc#3', 'totcost'): -94350.12448577807,
		('hhinc#3', 'totcost+50'): -28.4220434435491,
		('hhinc#3', 'tottime'): 0.0,
		('hhinc#3', 'tottime+100'): 0.0,
		('hhinc#4', 'ASC_BIKE'): 3669.7005738878634,
		('hhinc#4', 'ASC_SR2'): -95043.73259866364,
		('hhinc#4', 'ASC_SR3P'): 3490.8262820366563,
		('hhinc#4', 'ASC_TRAN'): 60885.53258597199,
		('hhinc#4', 'ASC_WALK'): 5941.015551244351,
		('hhinc#4', 'THETA'): 24655.90610351544,
		('hhinc#4', 'hhinc#2'): -7117026.512948193,
		('hhinc#4', 'hhinc#3'): 270709.67743631615,
		('hhinc#4', 'hhinc#4'): 4478989.131115964,
		('hhinc#4', 'hhinc#5'): 270429.912372026,
		('hhinc#4', 'hhinc#6'): 458265.58956765104,
		('hhinc#4', 'totcost'): 2140975.758443999,
		('hhinc#4', 'totcost+50'): 2225.4923270400686,
		('hhinc#4', 'tottime'): 0.0,
		('hhinc#4', 'tottime+100'): 0.0,
		('hhinc#5', 'ASC_BIKE'): 3389.584524226908,
		('hhinc#5', 'ASC_SR2'): -10373.30902399457,
		('hhinc#5', 'ASC_SR3P'): 482.8744118883242,
		('hhinc#5', 'ASC_TRAN'): 3669.7005738878634,
		('hhinc#5', 'ASC_WALK'): 894.3140912350088,
		('hhinc#5', 'THETA'): 1510.0584523826587,
		('hhinc#5', 'hhinc#2'): -760393.3992416215,
		('hhinc#5', 'hhinc#3'): 37607.738269849986,
		('hhinc#5', 'hhinc#4'): 270429.912372026,
		('hhinc#5', 'hhinc#5'): 242283.98531011154,
		('hhinc#5', 'hhinc#6'): 67803.3191437671,
		('hhinc#5', 'totcost'): 223315.19581432932,
		('hhinc#5', 'totcost+50'): -11.891020264044975,
		('hhinc#5', 'tottime'): 0.0,
		('hhinc#5', 'tottime+100'): 0.0,
		('hhinc#6', 'ASC_BIKE'): 894.3140912350088,
		('hhinc#6', 'ASC_SR2'): -18057.762045074225,
		('hhinc#6', 'ASC_SR3P'): 619.677029595399,
		('hhinc#6', 'ASC_TRAN'): 5941.015551244351,
		('hhinc#6', 'ASC_WALK'): 9467.826780286812,
		('hhinc#6', 'THETA'): 1928.2217854957846,
		('hhinc#6', 'hhinc#2'): -1314494.278173153,
		('hhinc#6', 'hhinc#3'): 52276.85368674705,
		('hhinc#6', 'hhinc#4'): 458265.58956765104,
		('hhinc#6', 'hhinc#5'): 67803.3191437671,
		('hhinc#6', 'hhinc#6'): 652774.8948963537,
		('hhinc#6', 'totcost'): 225622.6878936836,
		('hhinc#6', 'totcost+50'): -260.8841609726519,
		('hhinc#6', 'tottime'): 0.0,
		('hhinc#6', 'tottime+100'): 0.0,
		('totcost', 'ASC_BIKE'): 3542.9643164717727,
		('totcost', 'ASC_SR2'): -433546.6306182174,
		('totcost', 'ASC_SR3P'): -2145.3240873303475,
		('totcost', 'ASC_TRAN'): 42303.49541977925,
		('totcost', 'ASC_WALK'): 1651.3092167772722,
		('totcost', 'THETA'): 282258.3984123302,
		('totcost', 'hhinc#2'): -26868413.73833593,
		('totcost', 'hhinc#3'): -94350.12448577807,
		('totcost', 'hhinc#4'): 2140975.758443999,
		('totcost', 'hhinc#5'): 223315.19581432932,
		('totcost', 'hhinc#6'): 225622.6878936836,
		('totcost', 'totcost'): 106373969.23982364,
		('totcost', 'totcost+50'): 20563.918169569984,
		('totcost', 'tottime'): 0.0,
		('totcost', 'tottime+100'): 0.0,
		('totcost+50', 'ASC_BIKE'): -0.8747499258668594,
		('totcost+50', 'ASC_SR2'): -189.66773767206286,
		('totcost+50', 'ASC_SR3P'): -0.8124860122548639,
		('totcost+50', 'ASC_TRAN'): 43.11029487377359,
		('totcost+50', 'ASC_WALK'): -7.677332748661808,
		('totcost+50', 'THETA'): 104.89731164684805,
		('totcost+50', 'hhinc#2'): -11242.201853442559,
		('totcost+50', 'hhinc#3'): -28.4220434435491,
		('totcost+50', 'hhinc#4'): 2225.4923270400686,
		('totcost+50', 'hhinc#5'): -11.891020264044975,
		('totcost+50', 'hhinc#6'): -260.8841609726519,
		('totcost+50', 'totcost'): 20563.918169569984,
		('totcost+50', 'totcost+50'): 12.15409493745926,
		('totcost+50', 'tottime'): 0.0,
		('totcost+50', 'tottime+100'): 0.0,
		('tottime', 'ASC_BIKE'): 0.0,
		('tottime', 'ASC_SR2'): 0.0,
		('tottime', 'ASC_SR3P'): 0.0,
		('tottime', 'ASC_TRAN'): 0.0,
		('tottime', 'ASC_WALK'): 0.0,
		('tottime', 'THETA'): 0.0,
		('tottime', 'hhinc#2'): 0.0,
		('tottime', 'hhinc#3'): 0.0,
		('tottime', 'hhinc#4'): 0.0,
		('tottime', 'hhinc#5'): 0.0,
		('tottime', 'hhinc#6'): 0.0,
		('tottime', 'totcost'): 0.0,
		('tottime', 'totcost+50'): 0.0,
		('tottime', 'tottime'): 0.0,
		('tottime', 'tottime+100'): 0.0,
		('tottime+100', 'ASC_BIKE'): 0.0,
		('tottime+100', 'ASC_SR2'): 0.0,
		('tottime+100', 'ASC_SR3P'): 0.0,
		('tottime+100', 'ASC_TRAN'): 0.0,
		('tottime+100', 'ASC_WALK'): 0.0,
		('tottime+100', 'THETA'): 0.0,
		('tottime+100', 'hhinc#2'): 0.0,
		('tottime+100', 'hhinc#3'): 0.0,
		('tottime+100', 'hhinc#4'): 0.0,
		('tottime+100', 'hhinc#5'): 0.0,
		('tottime+100', 'hhinc#6'): 0.0,
		('tottime+100', 'totcost'): 0.0,
		('tottime+100', 'totcost+50'): 0.0,
		('tottime+100', 'tottime'): 0.0,
		('tottime+100', 'tottime+100'): 0.0
	}

	dict_ll1_bhhh = dict(ll1.bhhh.unstack())
	dict_ll2_bhhh = dict(ll2.bhhh.unstack())

	for k in bhhh_correct:
		assert dict_ll1_bhhh[k] == approx(bhhh_correct[k]), f"{k}: {dict_ll1_bhhh[k]} != {approx(bhhh_correct[k])}"

	assert dict_ll2_bhhh == approx(bhhh_correct)
	assert dict_ll1_bhhh == approx(dict_ll2_bhhh)

	assert ll2.dll.sort_index().values == approx(ll2.dll_casewise.sum(0).sort_index().values)

	dll_casewise_A = ll2.dll_casewise / j2.weight_normalization
	if j2.data_wt is not None:
		dll_casewise_B = dll_casewise_A.values / j2.data_wt.values
	else:
		dll_casewise_B = dll_casewise_A.values

	corrected_bhhh = pandas.DataFrame(
		numpy.dot(dll_casewise_A.T, dll_casewise_B),
		index=ll2.dll.index,
		columns=ll2.dll.index,
	) * j2.weight_normalization

	assert dict(ll1.bhhh.unstack()) == approx(dict(corrected_bhhh.unstack()))


def test_dataframes_holdfast_2():
	d = MTC()

	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', 'tottime+100', 'totcost+50')
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	df_co2 = pandas.concat([df_co, df_co]).reset_index(drop=True)
	df_ca2 = pandas.concat([df_ca.unstack(), df_ca.unstack()]).reset_index(drop=True).stack()
	df_av2 = pandas.concat([df_av.unstack(), df_av.unstack()]).reset_index(drop=True).stack()

	df_chX = pandas.DataFrame(
		numpy.zeros_like(df_ch.values),
		index=df_ch.index,
		columns=df_ch.columns,
	)
	df_chX = df_chX.unstack()
	df_chX.iloc[:, 1] = 2.0
	df_chX = df_chX.stack()

	df_ch2 = pandas.concat([df_ch.unstack(), df_chX.unstack()]).reset_index(drop=True).stack()

	from larch import DataFrames, Model

	j1 = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_chX + df_ch,
	)

	j2 = DataFrames(
		co=df_co2,
		ca=df_ca2,
		av=df_av2,
		ch=df_ch2,
	)
	j1.autoscale_weights()
	j2.autoscale_weights()


	m5 = Model()

	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")
	m5.quantity_ca = PX("tottime+100") + PX("totcost+50")
	m5.quantity_scale = P('THETA')
	m5.lock_value('tottime', -0.01862990704919887)
	m5.lock_value('THETA', 0.5)
	m5.lock_value('tottime+100', 0)

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.123,
		'totcost+50': 2,
		'THETA': 0.7,
	}

	m5.pf_sort()

	assert j1.computational
	m5.dataframes = j1
	ll1 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)
	assert m5.check_d_loglike().data.similarity.min() > 4

	assert j2.computational
	m5.dataframes = j2
	ll2 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)

	assert m5.check_d_loglike().data.similarity.min() > 4

	q1_dll = {'ASC_BIKE': -392.6027832998042,
			  'ASC_SR2': 7006.672643267781,
			  'ASC_SR3P': -538.6369146702652,
			  'ASC_TRAN': -2343.685625976806,
			  'ASC_WALK': -551.8744209896063,
			  'THETA': 0.0,
			  'hhinc#2': 410903.774738222,
			  'hhinc#3': -33968.57901328977,
			  'hhinc#4': -131985.71066696098,
			  'hhinc#5': -22588.272419334447,
			  'hhinc#6': -30197.288562004323,
			  'totcost': -447478.9000174226,
			  'totcost+50': -65.23882127875741,
			  'tottime': 0.0,
			  'tottime+100': 0.0}

	assert (j1.weight_normalization, j2.weight_normalization) == (3.0, 1.5)
	assert (ll1.ll, ll2.ll) == approx((-19668.715077149176, -19668.715077149176))
	assert (ll1.ll == approx(ll2.ll))

	dict_ll1_dll = dict(ll1.dll)
	dict_ll2_dll = dict(ll2.dll)

	for k in q1_dll:
		assert q1_dll[k] == approx(dict_ll1_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll1_dll[k]}"
		assert q1_dll[k] == approx(dict_ll2_dll[k], rel=1e-5), f"{k} {q1_dll[k]} = {dict_ll2_dll[k]}"
	bhhh_correct = {('ASC_BIKE', 'ASC_BIKE'): 79.83266770086034,
					('ASC_BIKE', 'ASC_SR2'): -229.79660078838938,
					('ASC_BIKE', 'ASC_SR3P'): 12.643129086860142,
					('ASC_BIKE', 'ASC_TRAN'): 70.07585828985945,
					('ASC_BIKE', 'ASC_WALK'): 22.246128313070653,
					('ASC_BIKE', 'THETA'): 0.0,
					('ASC_BIKE', 'hhinc#2'): -12904.457436127537,
					('ASC_BIKE', 'hhinc#3'): 787.8893469747022,
					('ASC_BIKE', 'hhinc#4'): 4048.6498072440104,
					('ASC_BIKE', 'hhinc#5'): 4089.2236743603944,
					('ASC_BIKE', 'hhinc#6'): 1308.8861318927086,
					('ASC_BIKE', 'totcost'): 3469.0006596303424,
					('ASC_BIKE', 'totcost+50'): -1.2447114245377238,
					('ASC_BIKE', 'tottime'): 0.0,
					('ASC_BIKE', 'tottime+100'): 0.0,
					('ASC_SR2', 'ASC_BIKE'): -229.79660078838938,
					('ASC_SR2', 'ASC_SR2'): 6456.286332248468,
					('ASC_SR2', 'ASC_SR3P'): -357.6439114513429,
					('ASC_SR2', 'ASC_TRAN'): -1489.2382432974223,
					('ASC_SR2', 'ASC_WALK'): -401.95827919487425,
					('ASC_SR2', 'THETA'): 0.0,
					('ASC_SR2', 'hhinc#2'): 378841.0679037492,
					('ASC_SR2', 'hhinc#3'): -22332.632816827598,
					('ASC_SR2', 'hhinc#4'): -83258.48625281108,
					('ASC_SR2', 'hhinc#5'): -12904.457436127537,
					('ASC_SR2', 'hhinc#6'): -20788.743910965713,
					('ASC_SR2', 'totcost'): -342743.29266519094,
					('ASC_SR2', 'totcost+50'): -73.02523549806536,
					('ASC_SR2', 'tottime'): 0.0,
					('ASC_SR2', 'tottime+100'): 0.0,
					('ASC_SR3P', 'ASC_BIKE'): 12.643129086860142,
					('ASC_SR3P', 'ASC_SR2'): -357.6439114513429,
					('ASC_SR3P', 'ASC_SR3P'): 179.1083984544908,
					('ASC_SR3P', 'ASC_TRAN'): 67.0280610716846,
					('ASC_SR3P', 'ASC_WALK'): 14.903033365763907,
					('ASC_SR3P', 'THETA'): 0.0,
					('ASC_SR3P', 'hhinc#2'): -22332.632816827598,
					('ASC_SR3P', 'hhinc#3'): 10922.35802283543,
					('ASC_SR3P', 'hhinc#4'): 4061.7724223432156,
					('ASC_SR3P', 'hhinc#5'): 787.8893469747024,
					('ASC_SR3P', 'hhinc#6'): 927.5240572798773,
					('ASC_SR3P', 'totcost'): -76.28414580089634,
					('ASC_SR3P', 'totcost+50'): -0.7231602193714833,
					('ASC_SR3P', 'tottime'): 0.0,
					('ASC_SR3P', 'tottime+100'): 0.0,
					('ASC_TRAN', 'ASC_BIKE'): 70.07585828985945,
					('ASC_TRAN', 'ASC_SR2'): -1489.2382432974223,
					('ASC_TRAN', 'ASC_SR3P'): 67.0280610716846,
					('ASC_TRAN', 'ASC_TRAN'): 921.4408309292568,
					('ASC_TRAN', 'ASC_WALK'): 108.17989313781854,
					('ASC_TRAN', 'THETA'): 0.0,
					('ASC_TRAN', 'hhinc#2'): -83258.48625281108,
					('ASC_TRAN', 'hhinc#3'): 4061.772422343217,
					('ASC_TRAN', 'hhinc#4'): 50199.13670059733,
					('ASC_TRAN', 'hhinc#5'): 4048.6498072440104,
					('ASC_TRAN', 'hhinc#6'): 5954.851732768629,
					('ASC_TRAN', 'totcost'): 24561.809476945586,
					('ASC_TRAN', 'totcost+50'): 11.048237547050611,
					('ASC_TRAN', 'tottime'): 0.0,
					('ASC_TRAN', 'tottime+100'): 0.0,
					('ASC_WALK', 'ASC_BIKE'): 22.246128313070653,
					('ASC_WALK', 'ASC_SR2'): -401.95827919487425,
					('ASC_WALK', 'ASC_SR3P'): 14.903033365763907,
					('ASC_WALK', 'ASC_TRAN'): 108.17989313781854,
					('ASC_WALK', 'ASC_WALK'): 222.69592089676067,
					('ASC_WALK', 'THETA'): 0.0,
					('ASC_WALK', 'hhinc#2'): -20788.743910965713,
					('ASC_WALK', 'hhinc#3'): 927.5240572798773,
					('ASC_WALK', 'hhinc#4'): 5954.8517327686295,
					('ASC_WALK', 'hhinc#5'): 1308.8861318927088,
					('ASC_WALK', 'hhinc#6'): 10696.424378665355,
					('ASC_WALK', 'totcost'): 348.6130744568044,
					('ASC_WALK', 'totcost+50'): -5.13912194260823,
					('ASC_WALK', 'tottime'): 0.0,
					('ASC_WALK', 'tottime+100'): 0.0,
					('THETA', 'ASC_BIKE'): 0.0,
					('THETA', 'ASC_SR2'): 0.0,
					('THETA', 'ASC_SR3P'): 0.0,
					('THETA', 'ASC_TRAN'): 0.0,
					('THETA', 'ASC_WALK'): 0.0,
					('THETA', 'THETA'): 0.0,
					('THETA', 'hhinc#2'): 0.0,
					('THETA', 'hhinc#3'): 0.0,
					('THETA', 'hhinc#4'): 0.0,
					('THETA', 'hhinc#5'): 0.0,
					('THETA', 'hhinc#6'): 0.0,
					('THETA', 'totcost'): 0.0,
					('THETA', 'totcost+50'): 0.0,
					('THETA', 'tottime'): 0.0,
					('THETA', 'tottime+100'): 0.0,
					('hhinc#2', 'ASC_BIKE'): -12904.457436127537,
					('hhinc#2', 'ASC_SR2'): 378841.0679037492,
					('hhinc#2', 'ASC_SR3P'): -22332.632816827598,
					('hhinc#2', 'ASC_TRAN'): -83258.48625281108,
					('hhinc#2', 'ASC_WALK'): -20788.743910965713,
					('hhinc#2', 'THETA'): 0.0,
					('hhinc#2', 'hhinc#2'): 29115425.137375057,
					('hhinc#2', 'hhinc#3'): -1817969.9754464463,
					('hhinc#2', 'hhinc#4'): -6246195.537930491,
					('hhinc#2', 'hhinc#5'): -946561.5001469142,
					('hhinc#2', 'hhinc#6'): -1511923.4014813653,
					('hhinc#2', 'totcost'): -21288929.588875625,
					('hhinc#2', 'totcost+50'): -4382.184968417505,
					('hhinc#2', 'tottime'): 0.0,
					('hhinc#2', 'tottime+100'): 0.0,
					('hhinc#3', 'ASC_BIKE'): 787.8893469747022,
					('hhinc#3', 'ASC_SR2'): -22332.632816827598,
					('hhinc#3', 'ASC_SR3P'): 10922.35802283543,
					('hhinc#3', 'ASC_TRAN'): 4061.772422343217,
					('hhinc#3', 'ASC_WALK'): 927.5240572798773,
					('hhinc#3', 'THETA'): 0.0,
					('hhinc#3', 'hhinc#2'): -1817969.9754464463,
					('hhinc#3', 'hhinc#3'): 896146.7930736947,
					('hhinc#3', 'hhinc#4'): 316298.235534072,
					('hhinc#3', 'hhinc#5'): 61651.00518312761,
					('hhinc#3', 'hhinc#6'): 76757.60313325317,
					('hhinc#3', 'totcost'): 43955.06045506014,
					('hhinc#3', 'totcost+50'): -33.19436403851995,
					('hhinc#3', 'tottime'): 0.0,
					('hhinc#3', 'tottime+100'): 0.0,
					('hhinc#4', 'ASC_BIKE'): 4048.6498072440104,
					('hhinc#4', 'ASC_SR2'): -83258.48625281108,
					('hhinc#4', 'ASC_SR3P'): 4061.7724223432156,
					('hhinc#4', 'ASC_TRAN'): 50199.13670059733,
					('hhinc#4', 'ASC_WALK'): 5954.8517327686295,
					('hhinc#4', 'THETA'): 0.0,
					('hhinc#4', 'hhinc#2'): -6246195.537930491,
					('hhinc#4', 'hhinc#3'): 316298.235534072,
					('hhinc#4', 'hhinc#4'): 3698094.5622448297,
					('hhinc#4', 'hhinc#5'): 298260.5654030687,
					('hhinc#4', 'hhinc#6'): 451248.05045202695,
					('hhinc#4', 'totcost'): 1245613.1181130465,
					('hhinc#4', 'totcost+50'): 551.4751653867196,
					('hhinc#4', 'tottime'): 0.0,
					('hhinc#4', 'tottime+100'): 0.0,
					('hhinc#5', 'ASC_BIKE'): 4089.2236743603944,
					('hhinc#5', 'ASC_SR2'): -12904.457436127537,
					('hhinc#5', 'ASC_SR3P'): 787.8893469747024,
					('hhinc#5', 'ASC_TRAN'): 4048.6498072440104,
					('hhinc#5', 'ASC_WALK'): 1308.8861318927088,
					('hhinc#5', 'THETA'): 0.0,
					('hhinc#5', 'hhinc#2'): -946561.5001469142,
					('hhinc#5', 'hhinc#3'): 61651.00518312761,
					('hhinc#5', 'hhinc#4'): 298260.5654030687,
					('hhinc#5', 'hhinc#5'): 290714.65894724376,
					('hhinc#5', 'hhinc#6'): 98214.15443238258,
					('hhinc#5', 'totcost'): 228090.9138188297,
					('hhinc#5', 'totcost+50'): -49.48498062956637,
					('hhinc#5', 'tottime'): 0.0,
					('hhinc#5', 'tottime+100'): 0.0,
					('hhinc#6', 'ASC_BIKE'): 1308.8861318927086,
					('hhinc#6', 'ASC_SR2'): -20788.743910965713,
					('hhinc#6', 'ASC_SR3P'): 927.5240572798773,
					('hhinc#6', 'ASC_TRAN'): 5954.851732768629,
					('hhinc#6', 'ASC_WALK'): 10696.424378665355,
					('hhinc#6', 'THETA'): 0.0,
					('hhinc#6', 'hhinc#2'): -1511923.4014813653,
					('hhinc#6', 'hhinc#3'): 76757.60313325317,
					('hhinc#6', 'hhinc#4'): 451248.05045202695,
					('hhinc#6', 'hhinc#5'): 98214.15443238258,
					('hhinc#6', 'hhinc#6'): 743425.6836787707,
					('hhinc#6', 'totcost'): 161919.03904244775,
					('hhinc#6', 'totcost+50'): -201.95962066103948,
					('hhinc#6', 'tottime'): 0.0,
					('hhinc#6', 'tottime+100'): 0.0,
					('totcost', 'ASC_BIKE'): 3469.0006596303424,
					('totcost', 'ASC_SR2'): -342743.29266519094,
					('totcost', 'ASC_SR3P'): -76.28414580089634,
					('totcost', 'ASC_TRAN'): 24561.809476945586,
					('totcost', 'ASC_WALK'): 348.6130744568044,
					('totcost', 'THETA'): 0.0,
					('totcost', 'hhinc#2'): -21288929.588875625,
					('totcost', 'hhinc#3'): 43955.06045506014,
					('totcost', 'hhinc#4'): 1245613.1181130465,
					('totcost', 'hhinc#5'): 228090.9138188297,
					('totcost', 'hhinc#6'): 161919.03904244775,
					('totcost', 'totcost'): 78653561.73931359,
					('totcost', 'totcost+50'): 8440.344463532721,
					('totcost', 'tottime'): 0.0,
					('totcost', 'tottime+100'): 0.0,
					('totcost+50', 'ASC_BIKE'): -1.2447114245377238,
					('totcost+50', 'ASC_SR2'): -73.02523549806536,
					('totcost+50', 'ASC_SR3P'): -0.7231602193714833,
					('totcost+50', 'ASC_TRAN'): 11.048237547050611,
					('totcost+50', 'ASC_WALK'): -5.13912194260823,
					('totcost+50', 'THETA'): 0.0,
					('totcost+50', 'hhinc#2'): -4382.184968417505,
					('totcost+50', 'hhinc#3'): -33.19436403851995,
					('totcost+50', 'hhinc#4'): 551.4751653867196,
					('totcost+50', 'hhinc#5'): -49.48498062956637,
					('totcost+50', 'hhinc#6'): -201.95962066103948,
					('totcost+50', 'totcost'): 8440.344463532721,
					('totcost+50', 'totcost+50'): 2.670424956450309,
					('totcost+50', 'tottime'): 0.0,
					('totcost+50', 'tottime+100'): 0.0,
					('tottime', 'ASC_BIKE'): 0.0,
					('tottime', 'ASC_SR2'): 0.0,
					('tottime', 'ASC_SR3P'): 0.0,
					('tottime', 'ASC_TRAN'): 0.0,
					('tottime', 'ASC_WALK'): 0.0,
					('tottime', 'THETA'): 0.0,
					('tottime', 'hhinc#2'): 0.0,
					('tottime', 'hhinc#3'): 0.0,
					('tottime', 'hhinc#4'): 0.0,
					('tottime', 'hhinc#5'): 0.0,
					('tottime', 'hhinc#6'): 0.0,
					('tottime', 'totcost'): 0.0,
					('tottime', 'totcost+50'): 0.0,
					('tottime', 'tottime'): 0.0,
					('tottime', 'tottime+100'): 0.0,
					('tottime+100', 'ASC_BIKE'): 0.0,
					('tottime+100', 'ASC_SR2'): 0.0,
					('tottime+100', 'ASC_SR3P'): 0.0,
					('tottime+100', 'ASC_TRAN'): 0.0,
					('tottime+100', 'ASC_WALK'): 0.0,
					('tottime+100', 'THETA'): 0.0,
					('tottime+100', 'hhinc#2'): 0.0,
					('tottime+100', 'hhinc#3'): 0.0,
					('tottime+100', 'hhinc#4'): 0.0,
					('tottime+100', 'hhinc#5'): 0.0,
					('tottime+100', 'hhinc#6'): 0.0,
					('tottime+100', 'totcost'): 0.0,
					('tottime+100', 'totcost+50'): 0.0,
					('tottime+100', 'tottime'): 0.0,
					('tottime+100', 'tottime+100'): 0.0}

	d_ll1_bhhh = dict(ll1.bhhh.unstack())
	d_ll2_bhhh = dict(ll2.bhhh.unstack())

	for k in bhhh_correct:
		assert d_ll1_bhhh[k] == approx(bhhh_correct[k]), f"{k}: {d_ll1_bhhh[k]} != {approx(bhhh_correct[k])}"
	assert d_ll2_bhhh == approx(bhhh_correct)
	assert d_ll1_bhhh == approx( d_ll2_bhhh )
	assert ll2.dll.sort_index().values == approx(ll2.dll_casewise.sum(0).sort_index().values)

	dll_casewise_A = ll2.dll_casewise / j2.weight_normalization
	if j2.data_wt is not None:
		dll_casewise_B = dll_casewise_A.values / j2.data_wt.values
	else:
		dll_casewise_B = dll_casewise_A.values
	corrected_bhhh = pandas.DataFrame(
		numpy.dot(dll_casewise_A.T, dll_casewise_B),
		index=ll2.dll.index,
		columns=ll2.dll.index,
	) * j2.weight_normalization

	assert dict(ll1.bhhh.unstack()) == approx(dict(corrected_bhhh.unstack()))




def test_dataframes_nl_holdfasts():
	d = MTC()

	df_ca = d.dataframe_idca('ivtt', 'ovtt', 'totcost', '_choice_', 'tottime', 'tottime+100', 'totcost+50')
	df_co = d.dataframe_idco('age', 'hhinc', 'hhsize', 'numveh==0')
	df_av = d.dataframe_idca('_avail_', dtype=bool)
	df_ch = d.dataframe_idca('_choice_')

	df_co2 = pandas.concat([df_co, df_co]).reset_index(drop=True)
	df_ca2 = pandas.concat([df_ca.unstack(), df_ca.unstack()]).reset_index(drop=True).stack()
	df_av2 = pandas.concat([df_av.unstack(), df_av.unstack()]).reset_index(drop=True).stack()

	df_chX = pandas.DataFrame(
		numpy.zeros_like(df_ch.values),
		index=df_ch.index,
		columns=df_ch.columns,
	)
	df_chX = df_chX.unstack()
	df_chX.iloc[:, 1] = 2.0
	df_chX = df_chX.stack()

	df_ch2 = pandas.concat([df_ch.unstack(), df_chX.unstack()]).reset_index(drop=True).stack()

	from larch import DataFrames, Model

	j1 = DataFrames(
		co=df_co,
		ca=df_ca,
		av=df_av,
		ch=df_chX + df_ch,
	)

	j2 = DataFrames(
		co=df_co2,
		ca=df_ca2,
		av=df_av2,
		ch=df_ch2,
	)

	j1.autoscale_weights()
	j2.autoscale_weights()

	m5 = Model()
	from larch.roles import P, X, PX
	m5.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
	m5.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m5.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m5.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m5.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m5.utility_ca = PX("tottime") + PX("totcost")

	m5.quantity_ca = PX("tottime+100") + PX("totcost+50")

	m5.quantity_scale = P('THETA')
	m5.lock_value('tottime', -0.01862990704919887)
	m5.lock_value('THETA', 0.5)
	m5.lock_value('tottime+100', 0)

	m5.initialize_graph(alternative_codes=[1, 2, 3, 4, 5, 6])
	m5.graph.add_node(10, children=(5, 6), parameter='MU_nonmotor')
	m5.graph.add_node(11, children=(1, 2, 3), parameter='MU_car')
	m5.graph.add_node(12, children=(4, 10), parameter='MU_motor')

	beta_in1 = {
		'ASC_BIKE': -0.8523646111088327,
		'ASC_SR2': -0.5233769323949348,
		'ASC_SR3P': -2.3202089848081027,
		'ASC_TRAN': -0.05615933557609158,
		'ASC_WALK': 0.050082767550586924,
		'hhinc#2': -0.001040241396513087,
		'hhinc#3': 0.0031822969445656542,
		'hhinc#4': -0.0017162484345735326,
		'hhinc#5': -0.004071521055900851,
		'hhinc#6': -0.0021316332241034445,
		'totcost': -0.001336661560553717,
		'tottime': -0.123,
		'totcost+50': 2,
		'THETA': 0.7,
	}

	m5.pf_sort()

	assert j1.computational
	m5.dataframes = j1
	ll1 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)
	checker = m5.check_d_loglike()

	assert checker.data.similarity.min() > 4
	assert m5.check_d_loglike().data.similarity.min() > 4

	assert j2.computational
	m5.dataframes = j2
	ll2 = m5.loglike2_bhhh(beta_in1, return_series=True, persist=True)

	similarity = m5.check_d_loglike().data.similarity.min()
	assert similarity > 4

	q1_dll = {'ASC_BIKE': -392.6027832998042,
			  'ASC_SR2': 7006.672643267781,
			  'ASC_SR3P': -538.6369146702652,
			  'ASC_TRAN': -2343.685625976806,
			  'ASC_WALK': -551.8744209896063,
			  'THETA': 0.0,
			  'hhinc#2': 410903.774738222,
			  'hhinc#3': -33968.57901328977,
			  'hhinc#4': -131985.71066696098,
			  'hhinc#5': -22588.272419334447,
			  'hhinc#6': -30197.288562004323,
			  'totcost': -447478.9000174226,
			  'totcost+50': -65.23882127875741,
			  'tottime': 0.0,
			  'tottime+100': 0.0}


	assert (j1.weight_normalization, j2.weight_normalization) == (3.0, 1.5)
	assert (ll1.ll, ll2.ll) == approx((-19668.715077149176, -19668.715077149176))
	assert (ll1.ll == approx(ll2.ll))

	d_11 = dict(ll1.dll)
	d_12 = dict(ll2.dll)

	for k in q1_dll:
		assert q1_dll[k] == approx(d_11[k], rel=1e-5), f"{k} {q1_dll[k]} = {d_11[k]}"
		assert q1_dll[k] == approx(d_12[k], rel=1e-5), f"{k} {q1_dll[k]} = {d_12[k]}"

	bhhh_correct = {('ASC_BIKE', 'ASC_BIKE'): 79.83266770086034,
					('ASC_BIKE', 'ASC_SR2'): -229.79660078838938,
					('ASC_BIKE', 'ASC_SR3P'): 12.643129086860142,
					('ASC_BIKE', 'ASC_TRAN'): 70.07585828985947,
					('ASC_BIKE', 'ASC_WALK'): 22.246128313070653,
					('ASC_BIKE', 'THETA'): 0.0,
					('ASC_BIKE', 'hhinc#2'): -12904.457436127537,
					('ASC_BIKE', 'hhinc#3'): 787.8893469747022,
					('ASC_BIKE', 'hhinc#4'): 4048.6498072440104,
					('ASC_BIKE', 'hhinc#5'): 4089.2236743603953,
					('ASC_BIKE', 'hhinc#6'): 1308.8861318927086,
					('ASC_BIKE', 'totcost'): 3469.0006596303424,
					('ASC_BIKE', 'totcost+50'): -1.2447114245377242,
					('ASC_BIKE', 'tottime'): 0.0,
					('ASC_BIKE', 'tottime+100'): 0.0,
					('ASC_BIKE', 'MU_motor'): 118.98477628392027,
					('ASC_BIKE', 'MU_car'): -201.10897471704317,
					('ASC_BIKE', 'MU_nonmotor'): 58.51018194474745,
					('ASC_SR2', 'ASC_BIKE'): -229.79660078838938,
					('ASC_SR2', 'ASC_SR2'): 6456.28633224847,
					('ASC_SR2', 'ASC_SR3P'): -357.64391145134294,
					('ASC_SR2', 'ASC_TRAN'): -1489.2382432974225,
					('ASC_SR2', 'ASC_WALK'): -401.95827919487425,
					('ASC_SR2', 'THETA'): 0.0,
					('ASC_SR2', 'hhinc#2'): 378841.0679037491,
					('ASC_SR2', 'hhinc#3'): -22332.6328168276,
					('ASC_SR2', 'hhinc#4'): -83258.48625281108,
					('ASC_SR2', 'hhinc#5'): -12904.457436127537,
					('ASC_SR2', 'hhinc#6'): -20788.74391096571,
					('ASC_SR2', 'totcost'): -342743.29266519094,
					('ASC_SR2', 'totcost+50'): -73.02523549806538,
					('ASC_SR2', 'tottime'): 0.0,
					('ASC_SR2', 'tottime+100'): 0.0,
					('ASC_SR2', 'MU_motor'): -931.6070360938243,
					('ASC_SR2', 'MU_car'): 4585.112032066163,
					('ASC_SR2', 'MU_nonmotor'): -213.28789314102121,
					('ASC_SR3P', 'ASC_BIKE'): 12.643129086860142,
					('ASC_SR3P', 'ASC_SR2'): -357.64391145134294,
					('ASC_SR3P', 'ASC_SR3P'): 179.10839845449087,
					('ASC_SR3P', 'ASC_TRAN'): 67.02806107168459,
					('ASC_SR3P', 'ASC_WALK'): 14.90303336576391,
					('ASC_SR3P', 'THETA'): 0.0,
					('ASC_SR3P', 'hhinc#2'): -22332.6328168276,
					('ASC_SR3P', 'hhinc#3'): 10922.35802283543,
					('ASC_SR3P', 'hhinc#4'): 4061.772422343216,
					('ASC_SR3P', 'hhinc#5'): 787.8893469747022,
					('ASC_SR3P', 'hhinc#6'): 927.5240572798773,
					('ASC_SR3P', 'totcost'): -76.28414580089293,
					('ASC_SR3P', 'totcost+50'): -0.7231602193714823,
					('ASC_SR3P', 'tottime'): 0.0,
					('ASC_SR3P', 'tottime+100'): 0.0,
					('ASC_SR3P', 'MU_motor'): 43.615472656134386,
					('ASC_SR3P', 'MU_car'): 121.24193120214112,
					('ASC_SR3P', 'MU_nonmotor'): 8.155713300538075,
					('ASC_TRAN', 'ASC_BIKE'): 70.07585828985947,
					('ASC_TRAN', 'ASC_SR2'): -1489.2382432974225,
					('ASC_TRAN', 'ASC_SR3P'): 67.02806107168459,
					('ASC_TRAN', 'ASC_TRAN'): 921.4408309292568,
					('ASC_TRAN', 'ASC_WALK'): 108.17989313781854,
					('ASC_TRAN', 'THETA'): 0.0,
					('ASC_TRAN', 'hhinc#2'): -83258.48625281108,
					('ASC_TRAN', 'hhinc#3'): 4061.772422343216,
					('ASC_TRAN', 'hhinc#4'): 50199.13670059733,
					('ASC_TRAN', 'hhinc#5'): 4048.6498072440104,
					('ASC_TRAN', 'hhinc#6'): 5954.851732768628,
					('ASC_TRAN', 'totcost'): 24561.809476945586,
					('ASC_TRAN', 'totcost+50'): 11.048237547050608,
					('ASC_TRAN', 'tottime'): 0.0,
					('ASC_TRAN', 'tottime+100'): 0.0,
					('ASC_TRAN', 'MU_motor'): 389.8929747533961,
					('ASC_TRAN', 'MU_car'): -1235.7210225167632,
					('ASC_TRAN', 'MU_nonmotor'): 61.922695872758545,
					('ASC_WALK', 'ASC_BIKE'): 22.246128313070653,
					('ASC_WALK', 'ASC_SR2'): -401.95827919487425,
					('ASC_WALK', 'ASC_SR3P'): 14.90303336576391,
					('ASC_WALK', 'ASC_TRAN'): 108.17989313781854,
					('ASC_WALK', 'ASC_WALK'): 222.6959208967607,
					('ASC_WALK', 'THETA'): 0.0,
					('ASC_WALK', 'hhinc#2'): -20788.743910965706,
					('ASC_WALK', 'hhinc#3'): 927.5240572798773,
					('ASC_WALK', 'hhinc#4'): 5954.851732768629,
					('ASC_WALK', 'hhinc#5'): 1308.8861318927088,
					('ASC_WALK', 'hhinc#6'): 10696.424378665353,
					('ASC_WALK', 'totcost'): 348.6130744568066,
					('ASC_WALK', 'totcost+50'): -5.139121942608227,
					('ASC_WALK', 'tottime'): 0.0,
					('ASC_WALK', 'tottime+100'): 0.0,
					('ASC_WALK', 'MU_motor'): 245.85292794792684,
					('ASC_WALK', 'MU_car'): -353.124266829837,
					('ASC_WALK', 'MU_nonmotor'): 67.27208550687067,
					('THETA', 'ASC_BIKE'): 0.0,
					('THETA', 'ASC_SR2'): 0.0,
					('THETA', 'ASC_SR3P'): 0.0,
					('THETA', 'ASC_TRAN'): 0.0,
					('THETA', 'ASC_WALK'): 0.0,
					('THETA', 'THETA'): 0.0,
					('THETA', 'hhinc#2'): 0.0,
					('THETA', 'hhinc#3'): 0.0,
					('THETA', 'hhinc#4'): 0.0,
					('THETA', 'hhinc#5'): 0.0,
					('THETA', 'hhinc#6'): 0.0,
					('THETA', 'totcost'): 0.0,
					('THETA', 'totcost+50'): 0.0,
					('THETA', 'tottime'): 0.0,
					('THETA', 'tottime+100'): 0.0,
					('THETA', 'MU_motor'): 0.0,
					('THETA', 'MU_car'): 0.0,
					('THETA', 'MU_nonmotor'): 0.0,
					('hhinc#2', 'ASC_BIKE'): -12904.457436127537,
					('hhinc#2', 'ASC_SR2'): 378841.0679037491,
					('hhinc#2', 'ASC_SR3P'): -22332.6328168276,
					('hhinc#2', 'ASC_TRAN'): -83258.48625281108,
					('hhinc#2', 'ASC_WALK'): -20788.743910965706,
					('hhinc#2', 'THETA'): 0.0,
					('hhinc#2', 'hhinc#2'): 29115425.137375057,
					('hhinc#2', 'hhinc#3'): -1817969.9754464468,
					('hhinc#2', 'hhinc#4'): -6246195.537930491,
					('hhinc#2', 'hhinc#5'): -946561.5001469143,
					('hhinc#2', 'hhinc#6'): -1511923.4014813656,
					('hhinc#2', 'totcost'): -21288929.588875625,
					('hhinc#2', 'totcost+50'): -4382.184968417505,
					('hhinc#2', 'tottime'): 0.0,
					('hhinc#2', 'tottime+100'): 0.0,
					('hhinc#2', 'MU_motor'): -50276.89958197008,
					('hhinc#2', 'MU_car'): 275693.56573483296,
					('hhinc#2', 'MU_nonmotor'): -12006.543000481077,
					('hhinc#3', 'ASC_BIKE'): 787.8893469747022,
					('hhinc#3', 'ASC_SR2'): -22332.6328168276,
					('hhinc#3', 'ASC_SR3P'): 10922.35802283543,
					('hhinc#3', 'ASC_TRAN'): 4061.772422343216,
					('hhinc#3', 'ASC_WALK'): 927.5240572798773,
					('hhinc#3', 'THETA'): 0.0,
					('hhinc#3', 'hhinc#2'): -1817969.9754464468,
					('hhinc#3', 'hhinc#3'): 896146.7930736949,
					('hhinc#3', 'hhinc#4'): 316298.23553407204,
					('hhinc#3', 'hhinc#5'): 61651.00518312762,
					('hhinc#3', 'hhinc#6'): 76757.60313325319,
					('hhinc#3', 'totcost'): 43955.06045505953,
					('hhinc#3', 'totcost+50'): -33.194364038520185,
					('hhinc#3', 'tottime'): 0.0,
					('hhinc#3', 'tottime+100'): 0.0,
					('hhinc#3', 'MU_motor'): 2711.9899543690526,
					('hhinc#3', 'MU_car'): 5625.554430258295,
					('hhinc#3', 'MU_nonmotor'): 520.5677384255127,
					('hhinc#4', 'ASC_BIKE'): 4048.6498072440104,
					('hhinc#4', 'ASC_SR2'): -83258.48625281108,
					('hhinc#4', 'ASC_SR3P'): 4061.772422343216,
					('hhinc#4', 'ASC_TRAN'): 50199.13670059733,
					('hhinc#4', 'ASC_WALK'): 5954.851732768629,
					('hhinc#4', 'THETA'): 0.0,
					('hhinc#4', 'hhinc#2'): -6246195.537930491,
					('hhinc#4', 'hhinc#3'): 316298.23553407204,
					('hhinc#4', 'hhinc#4'): 3698094.5622448297,
					('hhinc#4', 'hhinc#5'): 298260.5654030687,
					('hhinc#4', 'hhinc#6'): 451248.05045202695,
					('hhinc#4', 'totcost'): 1245613.1181130465,
					('hhinc#4', 'totcost+50'): 551.4751653867193,
					('hhinc#4', 'tottime'): 0.0,
					('hhinc#4', 'tottime+100'): 0.0,
					('hhinc#4', 'MU_motor'): 20906.57838881135,
					('hhinc#4', 'MU_car'): -72451.9363068151,
					('hhinc#4', 'MU_nonmotor'): 3641.4351273861844,
					('hhinc#5', 'ASC_BIKE'): 4089.2236743603953,
					('hhinc#5', 'ASC_SR2'): -12904.457436127537,
					('hhinc#5', 'ASC_SR3P'): 787.8893469747022,
					('hhinc#5', 'ASC_TRAN'): 4048.6498072440104,
					('hhinc#5', 'ASC_WALK'): 1308.8861318927088,
					('hhinc#5', 'THETA'): 0.0,
					('hhinc#5', 'hhinc#2'): -946561.5001469143,
					('hhinc#5', 'hhinc#3'): 61651.00518312762,
					('hhinc#5', 'hhinc#4'): 298260.5654030687,
					('hhinc#5', 'hhinc#5'): 290714.65894724376,
					('hhinc#5', 'hhinc#6'): 98214.15443238258,
					('hhinc#5', 'totcost'): 228090.91381882975,
					('hhinc#5', 'totcost+50'): -49.484980629566365,
					('hhinc#5', 'tottime'): 0.0,
					('hhinc#5', 'tottime+100'): 0.0,
					('hhinc#5', 'MU_motor'): 6524.108939512265,
					('hhinc#5', 'MU_car'): -11508.402068362084,
					('hhinc#5', 'MU_nonmotor'): 2991.3731022239717,
					('hhinc#6', 'ASC_BIKE'): 1308.8861318927086,
					('hhinc#6', 'ASC_SR2'): -20788.74391096571,
					('hhinc#6', 'ASC_SR3P'): 927.5240572798773,
					('hhinc#6', 'ASC_TRAN'): 5954.851732768628,
					('hhinc#6', 'ASC_WALK'): 10696.424378665353,
					('hhinc#6', 'THETA'): 0.0,
					('hhinc#6', 'hhinc#2'): -1511923.4014813656,
					('hhinc#6', 'hhinc#3'): 76757.60313325319,
					('hhinc#6', 'hhinc#4'): 451248.05045202695,
					('hhinc#6', 'hhinc#5'): 98214.15443238258,
					('hhinc#6', 'hhinc#6'): 743425.6836787708,
					('hhinc#6', 'totcost'): 161919.03904244784,
					('hhinc#6', 'totcost+50'): -201.9596206610393,
					('hhinc#6', 'tottime'): 0.0,
					('hhinc#6', 'tottime+100'): 0.0,
					('hhinc#6', 'MU_motor'): 12339.48151025402,
					('hhinc#6', 'MU_car'): -19354.50567586405,
					('hhinc#6', 'MU_nonmotor'): 3795.579207439534,
					('totcost', 'ASC_BIKE'): 3469.0006596303424,
					('totcost', 'ASC_SR2'): -342743.29266519094,
					('totcost', 'ASC_SR3P'): -76.28414580089293,
					('totcost', 'ASC_TRAN'): 24561.809476945586,
					('totcost', 'ASC_WALK'): 348.6130744568066,
					('totcost', 'THETA'): 0.0,
					('totcost', 'hhinc#2'): -21288929.588875625,
					('totcost', 'hhinc#3'): 43955.06045505953,
					('totcost', 'hhinc#4'): 1245613.1181130465,
					('totcost', 'hhinc#5'): 228090.91381882975,
					('totcost', 'hhinc#6'): 161919.03904244784,
					('totcost', 'totcost'): 78653561.73931359,
					('totcost', 'totcost+50'): 8440.344463532725,
					('totcost', 'tottime'): 0.0,
					('totcost', 'tottime+100'): 0.0,
					('totcost', 'MU_motor'): 11800.821569256559,
					('totcost', 'MU_car'): -253811.78939656145,
					('totcost', 'MU_nonmotor'): 1490.7283654907437,
					('totcost+50', 'ASC_BIKE'): -1.2447114245377242,
					('totcost+50', 'ASC_SR2'): -73.02523549806538,
					('totcost+50', 'ASC_SR3P'): -0.7231602193714823,
					('totcost+50', 'ASC_TRAN'): 11.048237547050608,
					('totcost+50', 'ASC_WALK'): -5.139121942608227,
					('totcost+50', 'THETA'): 0.0,
					('totcost+50', 'hhinc#2'): -4382.184968417505,
					('totcost+50', 'hhinc#3'): -33.194364038520185,
					('totcost+50', 'hhinc#4'): 551.4751653867193,
					('totcost+50', 'hhinc#5'): -49.484980629566365,
					('totcost+50', 'hhinc#6'): -201.9596206610393,
					('totcost+50', 'totcost'): 8440.344463532725,
					('totcost+50', 'totcost+50'): 2.670424956450306,
					('totcost+50', 'tottime'): 0.0,
					('totcost+50', 'tottime+100'): 0.0,
					('totcost+50', 'MU_motor'): -1.7259725260631722,
					('totcost+50', 'MU_car'): -61.16541250876779,
					('totcost+50', 'MU_nonmotor'): -1.6515919941961887,
					('tottime', 'ASC_BIKE'): 0.0,
					('tottime', 'ASC_SR2'): 0.0,
					('tottime', 'ASC_SR3P'): 0.0,
					('tottime', 'ASC_TRAN'): 0.0,
					('tottime', 'ASC_WALK'): 0.0,
					('tottime', 'THETA'): 0.0,
					('tottime', 'hhinc#2'): 0.0,
					('tottime', 'hhinc#3'): 0.0,
					('tottime', 'hhinc#4'): 0.0,
					('tottime', 'hhinc#5'): 0.0,
					('tottime', 'hhinc#6'): 0.0,
					('tottime', 'totcost'): 0.0,
					('tottime', 'totcost+50'): 0.0,
					('tottime', 'tottime'): 0.0,
					('tottime', 'tottime+100'): 0.0,
					('tottime', 'MU_motor'): 0.0,
					('tottime', 'MU_car'): 0.0,
					('tottime', 'MU_nonmotor'): 0.0,
					('tottime+100', 'ASC_BIKE'): 0.0,
					('tottime+100', 'ASC_SR2'): 0.0,
					('tottime+100', 'ASC_SR3P'): 0.0,
					('tottime+100', 'ASC_TRAN'): 0.0,
					('tottime+100', 'ASC_WALK'): 0.0,
					('tottime+100', 'THETA'): 0.0,
					('tottime+100', 'hhinc#2'): 0.0,
					('tottime+100', 'hhinc#3'): 0.0,
					('tottime+100', 'hhinc#4'): 0.0,
					('tottime+100', 'hhinc#5'): 0.0,
					('tottime+100', 'hhinc#6'): 0.0,
					('tottime+100', 'totcost'): 0.0,
					('tottime+100', 'totcost+50'): 0.0,
					('tottime+100', 'tottime'): 0.0,
					('tottime+100', 'tottime+100'): 0.0,
					('tottime+100', 'MU_motor'): 0.0,
					('tottime+100', 'MU_car'): 0.0,
					('tottime+100', 'MU_nonmotor'): 0.0,
					('MU_motor', 'ASC_BIKE'): 118.98477628392027,
					('MU_motor', 'ASC_SR2'): -931.6070360938243,
					('MU_motor', 'ASC_SR3P'): 43.615472656134386,
					('MU_motor', 'ASC_TRAN'): 389.8929747533961,
					('MU_motor', 'ASC_WALK'): 245.85292794792684,
					('MU_motor', 'THETA'): 0.0,
					('MU_motor', 'hhinc#2'): -50276.89958197008,
					('MU_motor', 'hhinc#3'): 2711.9899543690526,
					('MU_motor', 'hhinc#4'): 20906.57838881135,
					('MU_motor', 'hhinc#5'): 6524.108939512265,
					('MU_motor', 'hhinc#6'): 12339.48151025402,
					('MU_motor', 'totcost'): 11800.821569256559,
					('MU_motor', 'totcost+50'): -1.7259725260631722,
					('MU_motor', 'tottime'): 0.0,
					('MU_motor', 'tottime+100'): 0.0,
					('MU_motor', 'MU_motor'): 517.83114060502,
					('MU_motor', 'MU_car'): -794.5153697232545,
					('MU_motor', 'MU_nonmotor'): 123.81880828513776,
					('MU_car', 'ASC_BIKE'): -201.10897471704317,
					('MU_car', 'ASC_SR2'): 4585.112032066163,
					('MU_car', 'ASC_SR3P'): 121.24193120214112,
					('MU_car', 'ASC_TRAN'): -1235.7210225167632,
					('MU_car', 'ASC_WALK'): -353.124266829837,
					('MU_car', 'THETA'): 0.0,
					('MU_car', 'hhinc#2'): 275693.56573483296,
					('MU_car', 'hhinc#3'): 5625.554430258295,
					('MU_car', 'hhinc#4'): -72451.9363068151,
					('MU_car', 'hhinc#5'): -11508.402068362084,
					('MU_car', 'hhinc#6'): -19354.50567586405,
					('MU_car', 'totcost'): -253811.78939656145,
					('MU_car', 'totcost+50'): -61.16541250876779,
					('MU_car', 'tottime'): 0.0,
					('MU_car', 'tottime+100'): 0.0,
					('MU_car', 'MU_motor'): -794.5153697232545,
					('MU_car', 'MU_car'): 4467.7500738051085,
					('MU_car', 'MU_nonmotor'): -201.32243917105563,
					('MU_nonmotor', 'ASC_BIKE'): 58.51018194474745,
					('MU_nonmotor', 'ASC_SR2'): -213.28789314102121,
					('MU_nonmotor', 'ASC_SR3P'): 8.155713300538075,
					('MU_nonmotor', 'ASC_TRAN'): 61.922695872758545,
					('MU_nonmotor', 'ASC_WALK'): 67.27208550687067,
					('MU_nonmotor', 'THETA'): 0.0,
					('MU_nonmotor', 'hhinc#2'): -12006.543000481077,
					('MU_nonmotor', 'hhinc#3'): 520.5677384255127,
					('MU_nonmotor', 'hhinc#4'): 3641.4351273861844,
					('MU_nonmotor', 'hhinc#5'): 2991.3731022239717,
					('MU_nonmotor', 'hhinc#6'): 3795.579207439534,
					('MU_nonmotor', 'totcost'): 1490.7283654907437,
					('MU_nonmotor', 'totcost+50'): -1.6515919941961887,
					('MU_nonmotor', 'tottime'): 0.0,
					('MU_nonmotor', 'tottime+100'): 0.0,
					('MU_nonmotor', 'MU_motor'): 123.81880828513776,
					('MU_nonmotor', 'MU_car'): -201.32243917105563,
					('MU_nonmotor', 'MU_nonmotor'): 87.74620253294411}

	d_ll1_bhhh = dict(ll1.bhhh.unstack())
	d_ll2_bhhh = dict(ll2.bhhh.unstack())

	for k in bhhh_correct:
		assert d_ll1_bhhh[k] == approx(bhhh_correct[k]), f"{k}: {d_ll1_bhhh[k]} != {approx(bhhh_correct[k])}"

	assert d_ll2_bhhh == approx(bhhh_correct)
	assert d_ll1_bhhh == approx( d_ll2_bhhh )
	assert ll2.dll.values == approx(ll2.dll_casewise.sum(0))

	dll_casewise_A = ll2.dll_casewise / j2.weight_normalization
	if j2.data_wt is not None:
		dll_casewise_B = dll_casewise_A / j2.data_wt.values
	else:
		dll_casewise_B = dll_casewise_A

	corrected_bhhh = pandas.DataFrame(
		numpy.dot(dll_casewise_A.T, dll_casewise_B),
		index=ll2.dll.index,
		columns=ll2.dll.index,
	) * j2.weight_normalization
	assert d_ll1_bhhh == approx(dict(corrected_bhhh.unstack()))

	m5.lock_value('MU_nonmotor', 0.2)

	m5.set_value('MU_motor', 0.6)
	m5.set_value('MU_car', 0.4)

	assert m5.check_d_loglike().data.similarity.min() > 4

	ll3 = m5.loglike2_bhhh(return_series=True, persist=True)

	bhhh_correct_3 = {
		('ASC_BIKE', 'ASC_BIKE'): 787.8635539247659,
		('ASC_BIKE', 'ASC_SR2'): -311.2487927752451,
		('ASC_BIKE', 'ASC_SR3P'): -2.1229673532645164,
		('ASC_BIKE', 'ASC_TRAN'): -98.3125959460721,
		('ASC_BIKE', 'ASC_WALK'): -547.559040083254,
		('ASC_BIKE', 'THETA'): 0.0,
		('ASC_BIKE', 'hhinc#2'): -17045.49803537083,
		('ASC_BIKE', 'hhinc#3'): -93.72654979849429,
		('ASC_BIKE', 'hhinc#4'): -4395.465420289052,
		('ASC_BIKE', 'hhinc#5'): 37228.76217788725,
		('ASC_BIKE', 'hhinc#6'): -25691.729775837564,
		('ASC_BIKE', 'totcost'): -6537.571042684306,
		('ASC_BIKE', 'totcost+50'): 3.0114198943312425,
		('ASC_BIKE', 'tottime'): 0.0,
		('ASC_BIKE', 'tottime+100'): 0.0,
		('ASC_BIKE', 'MU_car'): -590.7923523533282,
		('ASC_BIKE', 'MU_motor'): 277.2360009938833,
		('ASC_BIKE', 'MU_nonmotor'): 0.0,
		('ASC_SR2', 'ASC_BIKE'): -311.2487927752451,
		('ASC_SR2', 'ASC_SR2'): 48380.055700255085,
		('ASC_SR2', 'ASC_SR3P'): -313.95834915938667,
		('ASC_SR2', 'ASC_TRAN'): -4906.616912228648,
		('ASC_SR2', 'ASC_WALK'): -896.5486924831899,
		('ASC_SR2', 'THETA'): 0.0,
		('ASC_SR2', 'hhinc#2'): 2896268.0710480455,
		('ASC_SR2', 'hhinc#3'): -19380.342246058775,
		('ASC_SR2', 'hhinc#4'): -285196.59781410196,
		('ASC_SR2', 'hhinc#5'): -17045.49803537083,
		('ASC_SR2', 'hhinc#6'): -48627.10667001831,
		('ASC_SR2', 'totcost'): -3558605.805177293,
		('ASC_SR2', 'totcost+50'): -863.3720606965653,
		('ASC_SR2', 'tottime'): 0.0,
		('ASC_SR2', 'tottime+100'): 0.0,
		('ASC_SR2', 'MU_car'): 87744.65239732794,
		('ASC_SR2', 'MU_motor'): -2088.907408540881,
		('ASC_SR2', 'MU_nonmotor'): 0.0,
		('ASC_SR3P', 'ASC_BIKE'): -2.1229673532645164,
		('ASC_SR3P', 'ASC_SR2'): -313.95834915938667,
		('ASC_SR3P', 'ASC_SR3P'): 999.8677479956997,
		('ASC_SR3P', 'ASC_TRAN'): -74.96982082058354,
		('ASC_SR3P', 'ASC_WALK'): -7.136187160419057,
		('ASC_SR3P', 'THETA'): 0.0,
		('ASC_SR3P', 'hhinc#2'): -19380.342246058775,
		('ASC_SR3P', 'hhinc#3'): 60141.10393408174,
		('ASC_SR3P', 'hhinc#4'): -4391.757165389324,
		('ASC_SR3P', 'hhinc#5'): -93.72654979849429,
		('ASC_SR3P', 'hhinc#6'): -286.3919282824528,
		('ASC_SR3P', 'totcost'): -168991.86904369068,
		('ASC_SR3P', 'totcost+50'): -26.5613869190311,
		('ASC_SR3P', 'tottime'): 0.0,
		('ASC_SR3P', 'tottime+100'): 0.0,
		('ASC_SR3P', 'MU_car'): 5173.053702286675,
		('ASC_SR3P', 'MU_motor'): -16.076903406718607,
		('ASC_SR3P', 'MU_nonmotor'): 0.0,
		('ASC_TRAN', 'ASC_BIKE'): -98.3125959460721,
		('ASC_TRAN', 'ASC_SR2'): -4906.616912228648,
		('ASC_TRAN', 'ASC_SR3P'): -74.96982082058354,
		('ASC_TRAN', 'ASC_TRAN'): 1427.8599052850898,
		('ASC_TRAN', 'ASC_WALK'): 63.07672617049582,
		('ASC_TRAN', 'THETA'): 0.0,
		('ASC_TRAN', 'hhinc#2'): -285196.59781410196,
		('ASC_TRAN', 'hhinc#3'): -4391.757165389324,
		('ASC_TRAN', 'hhinc#4'): 77469.25265893053,
		('ASC_TRAN', 'hhinc#5'): -4395.465420289052,
		('ASC_TRAN', 'hhinc#6'): 4398.140665113446,
		('ASC_TRAN', 'totcost'): 320507.0795016588,
		('ASC_TRAN', 'totcost+50'): 94.85651235985807,
		('ASC_TRAN', 'tottime'): 0.0,
		('ASC_TRAN', 'tottime+100'): 0.0,
		('ASC_TRAN', 'MU_car'): -9179.664380173408,
		('ASC_TRAN', 'MU_motor'): 212.93729585244216,
		('ASC_TRAN', 'MU_nonmotor'): 0.0,
		('ASC_WALK', 'ASC_BIKE'): -547.559040083254,
		('ASC_WALK', 'ASC_SR2'): -896.5486924831899,
		('ASC_WALK', 'ASC_SR3P'): -7.136187160419057,
		('ASC_WALK', 'ASC_TRAN'): 63.07672617049582,
		('ASC_WALK', 'ASC_WALK'): 793.3792957893288,
		('ASC_WALK', 'THETA'): 0.0,
		('ASC_WALK', 'hhinc#2'): -48627.1066700183,
		('ASC_WALK', 'hhinc#3'): -286.39192828245274,
		('ASC_WALK', 'hhinc#4'): 4398.140665113447,
		('ASC_WALK', 'hhinc#5'): -25691.729775837564,
		('ASC_WALK', 'hhinc#6'): 37257.63178846156,
		('ASC_WALK', 'totcost'): 12537.742885537444,
		('ASC_WALK', 'totcost+50'): -7.565798809079978,
		('ASC_WALK', 'tottime'): 0.0,
		('ASC_WALK', 'tottime+100'): 0.0,
		('ASC_WALK', 'MU_car'): -1607.387834417706,
		('ASC_WALK', 'MU_motor'): 243.24025944342617,
		('ASC_WALK', 'MU_nonmotor'): 0.0,
		('THETA', 'ASC_BIKE'): 0.0,
		('THETA', 'ASC_SR2'): 0.0,
		('THETA', 'ASC_SR3P'): 0.0,
		('THETA', 'ASC_TRAN'): 0.0,
		('THETA', 'ASC_WALK'): 0.0,
		('THETA', 'THETA'): 0.0,
		('THETA', 'hhinc#2'): 0.0,
		('THETA', 'hhinc#3'): 0.0,
		('THETA', 'hhinc#4'): 0.0,
		('THETA', 'hhinc#5'): 0.0,
		('THETA', 'hhinc#6'): 0.0,
		('THETA', 'totcost'): 0.0,
		('THETA', 'totcost+50'): 0.0,
		('THETA', 'tottime'): 0.0,
		('THETA', 'tottime+100'): 0.0,
		('THETA', 'MU_car'): 0.0,
		('THETA', 'MU_motor'): 0.0,
		('THETA', 'MU_nonmotor'): 0.0,
		('hhinc#2', 'ASC_BIKE'): -17045.49803537083,
		('hhinc#2', 'ASC_SR2'): 2896268.0710480455,
		('hhinc#2', 'ASC_SR3P'): -19380.342246058775,
		('hhinc#2', 'ASC_TRAN'): -285196.59781410196,
		('hhinc#2', 'ASC_WALK'): -48627.1066700183,
		('hhinc#2', 'THETA'): 0.0,
		('hhinc#2', 'hhinc#2'): 224876400.1805011,
		('hhinc#2', 'hhinc#3'): -1605673.9323227894,
		('hhinc#2', 'hhinc#4'): -21843285.06530702,
		('hhinc#2', 'hhinc#5'): -1203439.8172631417,
		('hhinc#2', 'hhinc#6'): -3629992.60419955,
		('hhinc#2', 'totcost'): -223345290.56917173,
		('hhinc#2', 'totcost+50'): -51696.75686826736,
		('hhinc#2', 'tottime'): 0.0,
		('hhinc#2', 'tottime+100'): 0.0,
		('hhinc#2', 'MU_car'): 5374291.264655409,
		('hhinc#2', 'MU_motor'): -116413.84132832923,
		('hhinc#2', 'MU_nonmotor'): 0.0,
		('hhinc#3', 'ASC_BIKE'): -93.72654979849429,
		('hhinc#3', 'ASC_SR2'): -19380.342246058775,
		('hhinc#3', 'ASC_SR3P'): 60141.10393408174,
		('hhinc#3', 'ASC_TRAN'): -4391.757165389324,
		('hhinc#3', 'ASC_WALK'): -286.39192828245274,
		('hhinc#3', 'THETA'): 0.0,
		('hhinc#3', 'hhinc#2'): -1605673.9323227894,
		('hhinc#3', 'hhinc#3'): 4907338.499483666,
		('hhinc#3', 'hhinc#4'): -372755.8403539474,
		('hhinc#3', 'hhinc#5'): -5873.122790047896,
		('hhinc#3', 'hhinc#6'): -16920.04471839094,
		('hhinc#3', 'totcost'): -10822340.254550114,
		('hhinc#3', 'totcost+50'): -1585.8102759015205,
		('hhinc#3', 'tottime'): 0.0,
		('hhinc#3', 'tottime+100'): 0.0,
		('hhinc#3', 'MU_car'): 298134.8625434516,
		('hhinc#3', 'MU_motor'): -684.118595413454,
		('hhinc#3', 'MU_nonmotor'): 0.0,
		('hhinc#4', 'ASC_BIKE'): -4395.465420289052,
		('hhinc#4', 'ASC_SR2'): -285196.59781410196,
		('hhinc#4', 'ASC_SR3P'): -4391.757165389324,
		('hhinc#4', 'ASC_TRAN'): 77469.25265893053,
		('hhinc#4', 'ASC_WALK'): 4398.140665113447,
		('hhinc#4', 'THETA'): 0.0,
		('hhinc#4', 'hhinc#2'): -21843285.06530702,
		('hhinc#4', 'hhinc#3'): -372755.8403539474,
		('hhinc#4', 'hhinc#4'): 5704991.57090463,
		('hhinc#4', 'hhinc#5'): -323099.49236557784,
		('hhinc#4', 'hhinc#6'): 386396.45049227,
		('hhinc#4', 'totcost'): 19608306.923608694,
		('hhinc#4', 'totcost+50'): 5295.647065431894,
		('hhinc#4', 'tottime'): 0.0,
		('hhinc#4', 'tottime+100'): 0.0,
		('hhinc#4', 'MU_car'): -549172.7788912444,
		('hhinc#4', 'MU_motor'): 12672.694091156853,
		('hhinc#4', 'MU_nonmotor'): 0.0,
		('hhinc#5', 'ASC_BIKE'): 37228.76217788725,
		('hhinc#5', 'ASC_SR2'): -17045.49803537083,
		('hhinc#5', 'ASC_SR3P'): -93.72654979849429,
		('hhinc#5', 'ASC_TRAN'): -4395.465420289052,
		('hhinc#5', 'ASC_WALK'): -25691.729775837564,
		('hhinc#5', 'THETA'): 0.0,
		('hhinc#5', 'hhinc#2'): -1203439.8172631417,
		('hhinc#5', 'hhinc#3'): -5873.122790047896,
		('hhinc#5', 'hhinc#4'): -323099.49236557784,
		('hhinc#5', 'hhinc#5'): 2720411.367571267,
		('hhinc#5', 'hhinc#6'): -1891965.5461027632,
		('hhinc#5', 'totcost'): -265679.9329520056,
		('hhinc#5', 'totcost+50'): 201.28098006017416,
		('hhinc#5', 'tottime'): 0.0,
		('hhinc#5', 'tottime+100'): 0.0,
		('hhinc#5', 'MU_car'): -32924.90761877353,
		('hhinc#5', 'MU_motor'): 13893.49946345282,
		('hhinc#5', 'MU_nonmotor'): 0.0,
		('hhinc#6', 'ASC_BIKE'): -25691.729775837564,
		('hhinc#6', 'ASC_SR2'): -48627.10667001831,
		('hhinc#6', 'ASC_SR3P'): -286.3919282824528,
		('hhinc#6', 'ASC_TRAN'): 4398.140665113446,
		('hhinc#6', 'ASC_WALK'): 37257.63178846156,
		('hhinc#6', 'THETA'): 0.0,
		('hhinc#6', 'hhinc#2'): -3629992.60419955,
		('hhinc#6', 'hhinc#3'): -16920.04471839094,
		('hhinc#6', 'hhinc#4'): 386396.45049227,
		('hhinc#6', 'hhinc#5'): -1891965.5461027632,
		('hhinc#6', 'hhinc#6'): 2660975.9041010938,
		('hhinc#6', 'totcost'): 980872.5081467782,
		('hhinc#6', 'totcost+50'): -228.80589980163177,
		('hhinc#6', 'tottime'): 0.0,
		('hhinc#6', 'tottime+100'): 0.0,
		('hhinc#6', 'MU_car'): -90948.16201334767,
		('hhinc#6', 'MU_motor'): 11905.777157998651,
		('hhinc#6', 'MU_nonmotor'): 0.0,
		('totcost', 'ASC_BIKE'): -6537.571042684306,
		('totcost', 'ASC_SR2'): -3558605.805177293,
		('totcost', 'ASC_SR3P'): -168991.86904369068,
		('totcost', 'ASC_TRAN'): 320507.0795016588,
		('totcost', 'ASC_WALK'): 12537.742885537444,
		('totcost', 'THETA'): 0.0,
		('totcost', 'hhinc#2'): -223345290.56917173,
		('totcost', 'hhinc#3'): -10822340.254550114,
		('totcost', 'hhinc#4'): 19608306.923608694,
		('totcost', 'hhinc#5'): -265679.9329520056,
		('totcost', 'hhinc#6'): 980872.5081467782,
		('totcost', 'totcost'): 641064420.0512064,
		('totcost', 'totcost+50'): 66798.94296261865,
		('totcost', 'tottime'): 0.0,
		('totcost', 'tottime+100'): 0.0,
		('totcost', 'MU_car'): -6979462.221026908,
		('totcost', 'MU_motor'): 39815.64014713181,
		('totcost', 'MU_nonmotor'): 0.0,
		('totcost+50', 'ASC_BIKE'): 3.0114198943312425,
		('totcost+50', 'ASC_SR2'): -863.3720606965653,
		('totcost+50', 'ASC_SR3P'): -26.5613869190311,
		('totcost+50', 'ASC_TRAN'): 94.85651235985807,
		('totcost+50', 'ASC_WALK'): -7.565798809079978,
		('totcost+50', 'THETA'): 0.0,
		('totcost+50', 'hhinc#2'): -51696.75686826736,
		('totcost+50', 'hhinc#3'): -1585.8102759015205,
		('totcost+50', 'hhinc#4'): 5295.647065431894,
		('totcost+50', 'hhinc#5'): 201.28098006017416,
		('totcost+50', 'hhinc#6'): -228.80589980163177,
		('totcost+50', 'totcost'): 66798.94296261865,
		('totcost+50', 'totcost+50'): 19.874946057564287,
		('totcost+50', 'tottime'): 0.0,
		('totcost+50', 'tottime+100'): 0.0,
		('totcost+50', 'MU_car'): -1794.0684631646286,
		('totcost+50', 'MU_motor'): 11.272344371840354,
		('totcost+50', 'MU_nonmotor'): 0.0,
		('tottime', 'ASC_BIKE'): 0.0,
		('tottime', 'ASC_SR2'): 0.0,
		('tottime', 'ASC_SR3P'): 0.0,
		('tottime', 'ASC_TRAN'): 0.0,
		('tottime', 'ASC_WALK'): 0.0,
		('tottime', 'THETA'): 0.0,
		('tottime', 'hhinc#2'): 0.0,
		('tottime', 'hhinc#3'): 0.0,
		('tottime', 'hhinc#4'): 0.0,
		('tottime', 'hhinc#5'): 0.0,
		('tottime', 'hhinc#6'): 0.0,
		('tottime', 'totcost'): 0.0,
		('tottime', 'totcost+50'): 0.0,
		('tottime', 'tottime'): 0.0,
		('tottime', 'tottime+100'): 0.0,
		('tottime', 'MU_car'): 0.0,
		('tottime', 'MU_motor'): 0.0,
		('tottime', 'MU_nonmotor'): 0.0,
		('tottime+100', 'ASC_BIKE'): 0.0,
		('tottime+100', 'ASC_SR2'): 0.0,
		('tottime+100', 'ASC_SR3P'): 0.0,
		('tottime+100', 'ASC_TRAN'): 0.0,
		('tottime+100', 'ASC_WALK'): 0.0,
		('tottime+100', 'THETA'): 0.0,
		('tottime+100', 'hhinc#2'): 0.0,
		('tottime+100', 'hhinc#3'): 0.0,
		('tottime+100', 'hhinc#4'): 0.0,
		('tottime+100', 'hhinc#5'): 0.0,
		('tottime+100', 'hhinc#6'): 0.0,
		('tottime+100', 'totcost'): 0.0,
		('tottime+100', 'totcost+50'): 0.0,
		('tottime+100', 'tottime'): 0.0,
		('tottime+100', 'tottime+100'): 0.0,
		('tottime+100', 'MU_car'): 0.0,
		('tottime+100', 'MU_motor'): 0.0,
		('tottime+100', 'MU_nonmotor'): 0.0,
		('MU_car', 'ASC_BIKE'): -590.7923523533282,
		('MU_car', 'ASC_SR2'): 87744.65239732794,
		('MU_car', 'ASC_SR3P'): 5173.053702286675,
		('MU_car', 'ASC_TRAN'): -9179.664380173408,
		('MU_car', 'ASC_WALK'): -1607.387834417706,
		('MU_car', 'THETA'): 0.0,
		('MU_car', 'hhinc#2'): 5374291.264655409,
		('MU_car', 'hhinc#3'): 298134.8625434516,
		('MU_car', 'hhinc#4'): -549172.7788912444,
		('MU_car', 'hhinc#5'): -32924.90761877353,
		('MU_car', 'hhinc#6'): -90948.16201334767,
		('MU_car', 'totcost'): -6979462.221026908,
		('MU_car', 'totcost+50'): -1794.0684631646286,
		('MU_car', 'tottime'): 0.0,
		('MU_car', 'tottime+100'): 0.0,
		('MU_car', 'MU_car'): 195600.91879872247,
		('MU_car', 'MU_motor'): -3815.108916903301,
		('MU_car', 'MU_nonmotor'): 0.0,
		('MU_motor', 'ASC_BIKE'): 277.2360009938833,
		('MU_motor', 'ASC_SR2'): -2088.907408540881,
		('MU_motor', 'ASC_SR3P'): -16.076903406718607,
		('MU_motor', 'ASC_TRAN'): 212.93729585244216,
		('MU_motor', 'ASC_WALK'): 243.24025944342617,
		('MU_motor', 'THETA'): 0.0,
		('MU_motor', 'hhinc#2'): -116413.84132832923,
		('MU_motor', 'hhinc#3'): -684.118595413454,
		('MU_motor', 'hhinc#4'): 12672.694091156853,
		('MU_motor', 'hhinc#5'): 13893.49946345282,
		('MU_motor', 'hhinc#6'): 11905.777157998651,
		('MU_motor', 'totcost'): 39815.64014713181,
		('MU_motor', 'totcost+50'): 11.272344371840354,
		('MU_motor', 'tottime'): 0.0,
		('MU_motor', 'tottime+100'): 0.0,
		('MU_motor', 'MU_car'): -3815.108916903301,
		('MU_motor', 'MU_motor'): 740.1947470350533,
		('MU_motor', 'MU_nonmotor'): 0.0,
		('MU_nonmotor', 'ASC_BIKE'): 0.0,
		('MU_nonmotor', 'ASC_SR2'): 0.0,
		('MU_nonmotor', 'ASC_SR3P'): 0.0,
		('MU_nonmotor', 'ASC_TRAN'): 0.0,
		('MU_nonmotor', 'ASC_WALK'): 0.0,
		('MU_nonmotor', 'THETA'): 0.0,
		('MU_nonmotor', 'hhinc#2'): 0.0,
		('MU_nonmotor', 'hhinc#3'): 0.0,
		('MU_nonmotor', 'hhinc#4'): 0.0,
		('MU_nonmotor', 'hhinc#5'): 0.0,
		('MU_nonmotor', 'hhinc#6'): 0.0,
		('MU_nonmotor', 'totcost'): 0.0,
		('MU_nonmotor', 'totcost+50'): 0.0,
		('MU_nonmotor', 'tottime'): 0.0,
		('MU_nonmotor', 'tottime+100'): 0.0,
		('MU_nonmotor', 'MU_car'): 0.0,
		('MU_nonmotor', 'MU_motor'): 0.0,
		('MU_nonmotor', 'MU_nonmotor'): 0.0}

	bhhh_compute_3 = dict(ll3.bhhh.unstack())

	for k in bhhh_correct:
		assert bhhh_compute_3[k] == approx(bhhh_correct_3[k]), f"{k}: {bhhh_compute_3[k]} != {approx(bhhh_correct_3[k])}"

