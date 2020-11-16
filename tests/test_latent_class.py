from pytest import approx, fixture, raises

import larch
import pandas
from larch import data_warehouse
from larch.roles import P,X

@fixture(scope="module")
def swissmetro_raw_df():
	raw_df = pandas.read_csv(data_warehouse.example_file('swissmetro.csv.gz'))
	raw_df['SM_COST'] = raw_df['SM_CO'] * (raw_df["GA"]==0)
	raw_df['TRAIN_COST'] = raw_df.eval("TRAIN_CO * (GA == 0)")
	raw_df['CAR_AV_SP'] = raw_df.eval("CAR_AV * (SP!=0)")
	raw_df['TRAIN_AV_SP'] = raw_df.eval("TRAIN_AV * (SP!=0)")
	keep = raw_df.eval("PURPOSE in (1,3) and CHOICE != 0")
	return raw_df[keep]


def test_latent_class(swissmetro_raw_df):

	dfs = larch.DataFrames(swissmetro_raw_df, alt_codes=[1,2,3])

	m1 = larch.Model(dataservice=dfs)
	m1.availability_co_vars = {
		1: "TRAIN_AV_SP",
		2: "SM_AV",
		3: "CAR_AV_SP",
	}
	m1.choice_co_code = 'CHOICE'
	m1.utility_co[1] = P("ASC_TRAIN") + X("TRAIN_CO*(GA==0)") * P("B_COST")
	m1.utility_co[2] = X("SM_CO*(GA==0)") * P("B_COST")
	m1.utility_co[3] = P("ASC_CAR") + X("CAR_CO") * P("B_COST")


	m2 = larch.Model(dataservice=dfs)
	m2.availability_co_vars = {
		1: "TRAIN_AV_SP",
		2: "SM_AV",
		3: "CAR_AV_SP",
	}
	m2.choice_co_code = 'CHOICE'
	m2.utility_co[1] = P("ASC_TRAIN") + X("TRAIN_TT") * P("B_TIME") + X("TRAIN_CO*(GA==0)") * P("B_COST")
	m2.utility_co[2] = X("SM_TT") * P("B_TIME") + X("SM_CO*(GA==0)") * P("B_COST")
	m2.utility_co[3] = P("ASC_CAR") + X("CAR_TT") * P("B_TIME") + X("CAR_CO") * P("B_COST")

	km = larch.Model()
	km.utility_co[2] = P.W_OTHER

	from larch.model.latentclass import LatentClassModel
	m = LatentClassModel(km, {1:m1, 2:m2})

	m.load_data()

	m.set_value(P.ASC_CAR, 0.125/2)
	m.set_value(P.ASC_TRAIN, -0.398/2)
	m.set_value(P.B_COST, -.0126/2)
	m.set_value(P.B_TIME, -0.028/2)
	m.set_value(P.W_OTHER, 1.095/2)

	check1 = m.check_d_loglike()

	assert dict(check1.data.analytic) == approx({
		'ASC_CAR': -81.69736186616234,
		'ASC_TRAIN': -613.131371089499,
		'B_COST': -6697.31706964777,
		'B_TIME': -40104.940072046316,
		'W_OTHER': 245.43145056623683,
	})

	assert check1.data.similarity.min() > 4

	m.set_value(P.ASC_CAR, 0.125)
	m.set_value(P.ASC_TRAIN, -0.398)
	m.set_value(P.B_COST, -.0126)
	m.set_value(P.B_TIME, -0.028)
	m.set_value(P.W_OTHER, 1.095)

	assert m.loglike() == approx(-5208.502259337974)

	check2 = m.check_d_loglike()

	assert dict(check2.data.analytic) == approx({
		'ASC_CAR': 0.6243716033364302,
		'ASC_TRAIN': 0.9297965389102578,
		'B_COST': -154.03997923797007,
		'B_TIME': 76.19297915128493,
		'W_OTHER': -0.7936963902343083,
	})

	assert check2.data.similarity.min() > 2 # similarity is a bit lower very close to the optimum


def test_latent_class_mixed_data(swissmetro_raw_df):

	dfs = larch.DataFrames(swissmetro_raw_df, alt_codes=[1,2,3])

	m1 = larch.Model(dataservice=dfs)
	m1.availability_co_vars = {
		1: "TRAIN_AV_SP",
		2: "SM_AV",
		3: "CAR_AV_SP",
	}
	m1.choice_co_code = 'CHOICE'
	m1.utility_co[1] = P("ASC_TRAIN") + X("TRAIN_CO*(GA==0)") * P("B_COST")
	m1.utility_co[2] = X("SM_CO*(GA==0)") * P("B_COST")
	m1.utility_co[3] = P("ASC_CAR") + X("CAR_CO") * P("B_COST")


	m2 = larch.Model(dataservice=dfs)
	m2.availability_co_vars = {
		1: "TRAIN_AV_SP",
		2: "SM_AV",
		3: "CAR_AV_SP",
	}
	m2.choice_co_code = 'CHOICE'
	m2.utility_co[1] = P("ASC_TRAIN") + X("TRAIN_TT") * P("B_TIME") + X("TRAIN_CO*(GA==0)") * P("B_COST")
	m2.utility_co[2] = X("SM_TT") * P("B_TIME") + X("SM_CO*(GA==0)") * P("B_COST")
	m2.utility_co[3] = P("ASC_CAR") + X("CAR_TT") * P("B_TIME") + X("CAR_CO") * P("B_COST")

	dfs2 = larch.DataFrames(swissmetro_raw_df, alt_codes=[1,2])
	km = larch.Model(dataservice=dfs2)
	km.utility_co[2] = P.W_OTHER

	from larch.model.latentclass import LatentClassModel
	with raises(ValueError):
		m = LatentClassModel(km, {1:m1, 2:m2})

def test_latent_class_full_data(swissmetro_raw_df):

	dfs = larch.DataFrames(swissmetro_raw_df, alt_codes=[1,2,3])

	m1 = larch.Model(dataservice=dfs)
	m1.availability_co_vars = {
		1: "TRAIN_AV_SP",
		2: "SM_AV",
		3: "CAR_AV_SP",
	}
	m1.choice_co_code = 'CHOICE'
	m1.utility_co[1] = P("ASC_TRAIN") + X("TRAIN_CO*(GA==0)") * P("B_COST")
	m1.utility_co[2] = X("SM_CO*(GA==0)") * P("B_COST")
	m1.utility_co[3] = P("ASC_CAR") + X("CAR_CO") * P("B_COST")


	m2 = larch.Model(dataservice=dfs)
	m2.availability_co_vars = {
		1: "TRAIN_AV_SP",
		2: "SM_AV",
		3: "CAR_AV_SP",
	}
	m2.choice_co_code = 'CHOICE'
	m2.utility_co[1] = P("ASC_TRAIN") + X("TRAIN_TT") * P("B_TIME") + X("TRAIN_CO*(GA==0)") * P("B_COST")
	m2.utility_co[2] = X("SM_TT") * P("B_TIME") + X("SM_CO*(GA==0)") * P("B_COST")
	m2.utility_co[3] = P("ASC_CAR") + X("CAR_TT") * P("B_TIME") + X("CAR_CO") * P("B_COST")

	km = larch.Model(dataservice=dfs, alts=[1,2])
	km.utility_co[2] = P.W_OTHER

	from larch.model.latentclass import LatentClassModel
	m = LatentClassModel(km, {1:m1, 2:m2})

	m.load_data()

	m.set_value(P.ASC_CAR, 0.125/2)
	m.set_value(P.ASC_TRAIN, -0.398/2)
	m.set_value(P.B_COST, -.0126/2)
	m.set_value(P.B_TIME, -0.028/2)
	m.set_value(P.W_OTHER, 1.095/2)

	check1 = m.check_d_loglike()

	assert dict(check1.data.analytic) == approx({
		'ASC_CAR': -81.69736186616234,
		'ASC_TRAIN': -613.131371089499,
		'B_COST': -6697.31706964777,
		'B_TIME': -40104.940072046316,
		'W_OTHER': 245.43145056623683,
	})
