#
# from ..data_services.examples import example_file
# from ..data_services import H5PodCE, DataService
# from ..model import Model
# import numpy
# from pytest import approx
# import pandas
# from ..roles import P, X, PX
# from collections import OrderedDict
#
#
# def test_idce_mnl():
# 	f = example_file('MTCwork.csv.gz')
#
# 	ce = H5PodCE.from_csv(f)
#
# 	ce.create_indexes_from_labels('case_ix', 'casenum')
# 	ce.create_indexes_from_labels('alt_ix', 'altnum')
#
# 	ce.set_casealt_indexes('case_ix', 'alt_ix')
#
# 	dx = DataService(
# 		ce,
# 		altids=[1, 2, 3, 4, 5, 6],
# 		altnames=['DA', 'S2', 'S3', 'TR', 'BK', 'WK'],
# 	)
#
# 	av = ce.as_array_idca('1', present='1', dtype=numpy.bool)
#
# 	assert( av.shape == (5029, 6, 1) )
#
# 	ref = Model.Example()
#
# 	ref.set_values(
# 		totcost=-0.01,
# 		tottime=-0.01,
# 		ASC_BIKE=-1,
# 		ASC_SR2=-1,
# 	)
#
# 	ref.load_data()
#
# 	ref_ll = ref.loglike()
#
# 	assert( approx(-8739.682333540684) == ref_ll )
#
# 	ref_dll = ref.d_loglike()
#
# 	dx.add_pod(ref.dataservice.idco[0], broadcastable=False)
#
# 	m = Model(dataservice=dx)
#
# 	m.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
# 	m.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
# 	m.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
# 	m.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
# 	m.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
#
# 	m.utility_ca = PX("tottime") + PX("totcost")
#
# 	m.availability_var = '_avail_'
# 	m.choice_ca_var = 'chose'
#
# 	m.title = "MTC Example 1 (Simple MNL)"
#
# 	m.set_values(
# 		totcost=-0.01,
# 		tottime=-0.01,
# 		ASC_BIKE=-1,
# 		ASC_SR2=-1,
# 	)
#
# 	m.load_data()
#
# 	m_ll = m.loglike(cache_clear=True)
#
# 	assert( approx(-8739.682333540684) == m_ll )
#
# 	m_dll = m.d_loglike(return_series=True)
#
# 	q = pandas.DataFrame.from_dict(OrderedDict([
# 		('ref', ref_dll),
# 		('ce', m_dll),
# 	]))
# 	q['diff'] = q.ref - q.ce
#
# 	for row in q.itertuples():
# 		assert( approx(row.ref) == row.ce )
#
#
# def test_idce_qmnl():
# 	f = example_file('MTCwork.csv.gz')
#
# 	ce = H5PodCE.from_csv(f)
#
# 	ce.create_indexes_from_labels('case_ix', 'casenum')
# 	ce.create_indexes_from_labels('alt_ix', 'altnum')
#
# 	ce.set_casealt_indexes('case_ix', 'alt_ix')
#
# 	dx = DataService(
# 		ce,
# 		altids=[1, 2, 3, 4, 5, 6],
# 		altnames=['DA', 'S2', 'S3', 'TR', 'BK', 'WK'],
# 	)
#
# 	av = ce.as_array_idca('1', present='1', dtype=numpy.bool)
#
# 	assert( av.shape == (5029, 6, 1) )
#
# 	ref = Model.Example()
# 	ref.quantity_ca = P.Q1 + P.Qaltnum * X.altnum
# 	ref.set_values(
# 		totcost=-0.01,
# 		tottime=-0.01,
# 		ASC_BIKE=-1,
# 		ASC_SR2=-1,
# 		Q1=-1,
# 		Qaltnum=0.1,
# 	)
#
# 	ref.load_data()
#
# 	ref_ll = ref.loglike()
#
# 	assert( approx(-11877.84511534832) == ref_ll )
#
# 	ref_dll = ref.d_loglike()
#
# 	dx.add_pod(ref.dataservice.idco[0], broadcastable=False)
#
# 	m = Model(dataservice=dx)
#
# 	m.utility_co[2] = P("ASC_SR2") + P("hhinc#2") * X("hhinc")
# 	m.utility_co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
# 	m.utility_co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
# 	m.utility_co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
# 	m.utility_co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
#
# 	m.utility_ca = PX("tottime") + PX("totcost")
# 	m.quantity_ca = P.Q1 + P.Qaltnum * X.altnum
# 	m.availability_var = '_avail_'
# 	m.choice_ca_var = 'chose'
#
# 	m.title = "MTC Example 1 (Simple MNL)"
#
# 	m.set_values(
# 		totcost=-0.01,
# 		tottime=-0.01,
# 		ASC_BIKE=-1,
# 		ASC_SR2=-1,
# 		Q1=-1,
# 		Qaltnum=0.1,
# 	)
#
# 	m.load_data()
#
# 	m_ll = m.loglike(cache_clear=True)
#
# 	assert( approx(-11877.84511534832) == m_ll )
#
# 	m_dll = m.d_loglike(return_series=True)
#
# 	q = pandas.DataFrame.from_dict(OrderedDict([
# 		('ref', ref_dll),
# 		('ce', m_dll),
# 	]))
# 	q['diff'] = q.ref - q.ce
#
# 	correct = {
# 		'hhinc#6':     -24910.315308,
# 		'hhinc#2':       3398.416194,
# 		'hhinc#5':     -19911.378054,
# 		'hhinc#3':    -134912.951431,
# 		'ASC_WALK':      -431.000440,
# 		'ASC_BIKE':      -320.626546,
# 		'ASC_TRAN':      -293.301888,
# 		'hhinc#4':     -19475.950414,
# 		'totcost':     344656.334682,
# 		'ASC_SR3P':     -2275.752299,
# 		'ASC_SR2':         75.169003,
# 		'tottime':     -52936.139683,
# 		'Q1':             528.768035,
# 		'Qaltnum':       -528.768035,
# 	}
#
# 	for row in q.itertuples():
# 		assert( approx(row.ref) == row.ce )
# 		assert( approx(row.ref) == correct[row.Index] )
#
#
#
