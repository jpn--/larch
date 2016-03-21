.. currentmodule:: larch

================================
80: Network GEV Itinerary Choice
================================

.. testsetup:: *

	import larch
	import math
	from larch.util.flux import flux

	def assertAlmostEqual(x,y,delta=0.00001):
		if math.isinf(x) ^ math.isinf(y):
			print ("%20s, %20s [%20s]"%(str(x),str(y),str(flux(x,y))))
			return -2
		if flux(x,y) > 5e-3:
			print ("%20s, %20s [%20s]"%(str(x),str(y),str(flux(x,y))))
			return -1
		else:
			print ("%20s, %20s [%20s] OK"%(str(x),str(y),str(flux(x,y))))
			return 0

	def assertNearlyEqual(x, y, sigfigs=3):
		magnitude = (abs(x) + abs(y)) / 2
		if math.isinf(magnitude): magnitude = 1
		assertAlmostEqual(x, y, delta=magnitude*(10**(-sigfigs)))




This example is an itinerary choice meta-model built using the example
itinerary choice dataset included with Larch.

.. testcode::

	import larch
	import itertools
	import numpy
	import os
	numpy.set_printoptions(linewidth=200)
	numpy.set_printoptions(threshold=1000000)

	from larch.metamodel import MetaModel

	db = larch.DB.Example('ITINERARY', shared=True)

	casefilter = ""
	db.queries.idco_query = "SELECT distinct(casenum) AS caseid, 1 as weight FROM data_ca "+casefilter
	db.queries.idca_query = "SELECT casenum AS caseid, itinerarycode AS altid, * FROM data_ca "+casefilter


	dow_set = {1,0}
	type_set = {'OW','OB','IB'}

	common_vars = [
		"carrier=2",
		"carrier=3",
		"carrier=4",
		"carrier>=5",
		"aver_fare_hy",
		"aver_fare_ly",
		"itin_num_cnxs",
		"itin_num_directs",
	]

	segmented_vars = [
		"sin2pi",
		"sin4pi",
		"sin6pi",
		"cos2pi",
		"cos4pi",
		"cos6pi",
	]



	mm = larch.Model()

	for var in common_vars:
		mm.parameter(var)
		mm.utility.ca(var)



	meta = MetaModel()

	for segment_desciptor in itertools.product(dow_set, type_set):
		dx = larch.DB.NewConnection(db)
		casefilter = " WHERE dow=={0} AND direction=='{1}'".format(*segment_desciptor)
		dx.queries = qry = larch.core.QuerySetTwoTable(dx)
		qry.idco_query = "SELECT distinct casenum AS caseid, dow, direction FROM data_ca "+casefilter
		qry.idca_query = "SELECT casenum AS caseid, itinerarycode AS altid, * FROM data_ca "+casefilter
		qry.alts_query = "SELECT * FROM itinerarycodes "
		qry.choice = 'pax_count'
		qry.avail = '1'
		qry.weight = '1'

		m = meta.sub_model[segment_desciptor] = larch.Model(dx)
		m.option.calc_std_errors = False
		m.option.calc_null_likelihood = False

		m.db.refresh_queries()
		
		if m.db.nCases()==0:
			meta.sub_ncases[segment_desciptor] = 0
			meta.sub_weight[segment_desciptor] = 0
			continue
		
		for var in common_vars:
			m.parameter(var)
			m.utility.ca(var)
			meta.parameter(var)

		for var in segmented_vars:
			built_par = var+"_{}_{}".format(*segment_desciptor)
			m.parameter(built_par)
			mm.parameter(built_par)
			m.utility.ca(var, built_par)
			meta.parameter(built_par)
			mm.utility(var+"*(dow={0})*(direction=='{1}')".format(*segment_desciptor), built_par)

		m.option.idca_avail_ratio_floor = 0
		m.setUp()
		m.weight_choice_rebalance()
		meta.sub_weight[segment_desciptor] = partwgt = m.Data("Weight").sum()
		meta.total_weight += partwgt
		meta.sub_ncases[segment_desciptor] = partncase = m.nCases()
		meta.total_ncases += partncase

	print("meta")
	print(meta)

	mm.db = db
	mm.option.idca_avail_ratio_floor = 0
	mm.setUp()
	mm.provision(idca_avail_ratio_floor=0)
	print("mm.loglike()",mm.loglike())

	rr = None

	assertNearlyEqual(mm.loglike_null(), meta.loglike_null())


	meta.option.weight_choice_rebalance = False
	meta.option.calc_std_errors = False

	llnull = meta.loglike_null()
	print("llnull",llnull)

	meta.parameter_array[:] = 0.0
	mm.parameter_array[:] = 0.0
	d1 = meta.d_loglike()
	d2 = mm.d_loglike()


	for dd1,dd2 in zip(d1,d2):
		assertNearlyEqual(dd1,dd2)

	meta.parameter_array[:] = 0.01
	mm.parameter_array[:] = 0.01

	assertNearlyEqual(mm.loglike(), meta.loglike())

	d1 = meta.d_loglike()
	d2 = mm.d_loglike()
	for dd1,dd2 in zip(d1,d2):
		assertNearlyEqual(dd1,dd2)


	r = meta.maximize_loglike("SLSQP")
	print(r)
	print(meta)



	rr = mm.maximize_loglike("SLSQP")
	print(rr)
	print(mm)










.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(201)

