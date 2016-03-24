.. currentmodule:: larch

===========================================================
220: Partially Segmented Itinerary Choice Using a MetaModel
===========================================================

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


MetaModels are a way to estimate a group of models simultaneously.

This example is an itinerary choice meta-model built using the example
itinerary choice dataset included with Larch.

.. testcode::

	import larch
	import itertools
	import numpy
	import os

	from larch.metamodel import MetaModel

	db = larch.DB.Example('ITINERARY', shared=True)

	casefilter = ""
	db.queries.idco_query = "SELECT distinct(casenum) AS caseid, 1 as weight FROM data_ca "+casefilter
	db.queries.idca_query = "SELECT casenum AS caseid, itinerarycode AS altid, * FROM data_ca "+casefilter


	dow_set = [0,1]
	type_set = ['OW','OB','IB']

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


We can construct a MetaModel in two general ways: manually (explicitly giving every submodel), or by using a
submodel factory function.  


.. testcode::

	def submodel_factory(segment_desciptor, db, common_vars, segmented_vars):

		# Create a new larch.DB based on the shared DB
		dx = larch.DB.NewConnection(db)
		casefilter = " WHERE dow=={0} AND direction=='{1}'".format(*segment_desciptor)
		dx.queries = qry = larch.core.QuerySetTwoTable(dx)
		qry.idco_query = "SELECT distinct casenum AS caseid, dow, direction FROM data_ca "+casefilter
		qry.idca_query = "SELECT casenum AS caseid, itinerarycode AS altid, * FROM data_ca "+casefilter
		qry.alts_query = "SELECT * FROM itinerarycodes "
		qry.choice = 'pax_count'
		qry.avail = '1'
		qry.weight = '1'

		# Create a new submodel using the filtered data DB
		submodel = larch.Model(dx)
		submodel.db.refresh_queries()

		# If the submodel has no cases, skip the rest of setting it up
		if submodel.db.nCases()==0:
			return submodel

		# Populate the submodel with the common parameters
		for var in common_vars:
			submodel.parameter(var)
			submodel.utility.ca(var)

		# Populate the submodel with the segmented parameters
		for var in segmented_vars:
			built_par = var+"_{}_{}".format(*segment_desciptor)
			submodel.parameter(built_par)
			submodel.utility.ca(var, built_par)

		# Set up the submodel for use
		submodel.option.idca_avail_ratio_floor = 0
		submodel.setUp()
		submodel.weight_choice_rebalance()

		return submodel


	m = MetaModel( itertools.product(dow_set, type_set), submodel_factory, args=(db, common_vars, segmented_vars) )



Let's confirm that we got a model that has the parameters we want.

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> print(m)
	====================================================================================================
	Model Parameter Estimates
	----------------------------------------------------------------------------------------------------
	Parameter       	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue   
	carrier=2       	 0          	 0          	 nan        	 nan        	 0          
	carrier=3       	 0          	 0          	 nan        	 nan        	 0          
	carrier=4       	 0          	 0          	 nan        	 nan        	 0          
	carrier>=5      	 0          	 0          	 nan        	 nan        	 0          
	aver_fare_hy    	 0          	 0          	 nan        	 nan        	 0          
	aver_fare_ly    	 0          	 0          	 nan        	 nan        	 0          
	itin_num_cnxs   	 0          	 0          	 nan        	 nan        	 0          
	itin_num_directs	 0          	 0          	 nan        	 nan        	 0          
	sin2pi_0_OW     	 0          	 0          	 nan        	 nan        	 0          
	sin4pi_0_OW     	 0          	 0          	 nan        	 nan        	 0          
	sin6pi_0_OW     	 0          	 0          	 nan        	 nan        	 0          
	cos2pi_0_OW     	 0          	 0          	 nan        	 nan        	 0          
	cos4pi_0_OW     	 0          	 0          	 nan        	 nan        	 0          
	cos6pi_0_OW     	 0          	 0          	 nan        	 nan        	 0          
	sin2pi_0_OB     	 0          	 0          	 nan        	 nan        	 0          
	sin4pi_0_OB     	 0          	 0          	 nan        	 nan        	 0          
	sin6pi_0_OB     	 0          	 0          	 nan        	 nan        	 0          
	cos2pi_0_OB     	 0          	 0          	 nan        	 nan        	 0          
	cos4pi_0_OB     	 0          	 0          	 nan        	 nan        	 0          
	cos6pi_0_OB     	 0          	 0          	 nan        	 nan        	 0          
	sin2pi_0_IB     	 0          	 0          	 nan        	 nan        	 0          
	sin4pi_0_IB     	 0          	 0          	 nan        	 nan        	 0          
	sin6pi_0_IB     	 0          	 0          	 nan        	 nan        	 0          
	cos2pi_0_IB     	 0          	 0          	 nan        	 nan        	 0          
	cos4pi_0_IB     	 0          	 0          	 nan        	 nan        	 0          
	cos6pi_0_IB     	 0          	 0          	 nan        	 nan        	 0          
	sin2pi_1_OW     	 0          	 0          	 nan        	 nan        	 0          
	sin4pi_1_OW     	 0          	 0          	 nan        	 nan        	 0          
	sin6pi_1_OW     	 0          	 0          	 nan        	 nan        	 0          
	cos2pi_1_OW     	 0          	 0          	 nan        	 nan        	 0          
	cos4pi_1_OW     	 0          	 0          	 nan        	 nan        	 0          
	cos6pi_1_OW     	 0          	 0          	 nan        	 nan        	 0          
	sin2pi_1_OB     	 0          	 0          	 nan        	 nan        	 0          
	sin4pi_1_OB     	 0          	 0          	 nan        	 nan        	 0          
	sin6pi_1_OB     	 0          	 0          	 nan        	 nan        	 0          
	cos2pi_1_OB     	 0          	 0          	 nan        	 nan        	 0          
	cos4pi_1_OB     	 0          	 0          	 nan        	 nan        	 0          
	cos6pi_1_OB     	 0          	 0          	 nan        	 nan        	 0          
	sin2pi_1_IB     	 0          	 0          	 nan        	 nan        	 0          
	sin4pi_1_IB     	 0          	 0          	 nan        	 nan        	 0          
	sin6pi_1_IB     	 0          	 0          	 nan        	 nan        	 0          
	cos2pi_1_IB     	 0          	 0          	 nan        	 nan        	 0          
	cos4pi_1_IB     	 0          	 0          	 nan        	 nan        	 0          
	cos6pi_1_IB     	 0          	 0          	 nan        	 nan        	 0          
	====================================================================================================
	...


Yup, looks good.  We have one of all of the common parameters, plus a set of the segmented parameters for
each segment.  Now let's estimate our model.  First we'll turn off the weight-choice rebalancing, because
we already did that manually for each submodel.  We'll also turn off the calculation of standard errors, because
that takes a bit of time and we're not interested in those results yet.

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> m.option.weight_choice_rebalance = False
	>>> m.option.calc_std_errors = False
	>>> r = m.maximize_loglike("SLSQP")
	>>> print(r)
			 ctol: 1.351...e-09
		  loglike: -27540.2525...
	 loglike_null: -27722.6710...
		  message: 'Optimization terminated successfully per computed tolerance. [SLSQP]'
			  nit: 48
			niter: [('SLSQP', 48)]
			stats: <larch.core.runstats, success in ...>
		   status: 0
		  success: True
				x: array([-0.14374193,  0.00833003,  0.03251612, ...])

	>>> print(m)
	====================================================================================================
	Model Parameter Estimates
	----------------------------------------------------------------------------------------------------
	Parameter       	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue   
	carrier=2       	 0          	-0.143742   	 nan        	 nan        	 0          
	carrier=3       	 0          	 0.00833003 	 nan        	 nan        	 0          
	carrier=4       	 0          	 0.0325161  	 nan        	 nan        	 0          
	carrier>=5      	 0          	 0.0464821  	 nan        	 nan        	 0          
	aver_fare_hy    	 0          	-0.00151544 	 nan        	 nan        	 0          
	aver_fare_ly    	 0          	-0.00111023 	 nan        	 nan        	 0          
	itin_num_cnxs   	 0          	-0.678798   	 nan        	 nan        	 0          
	itin_num_directs	 0          	-0.255871   	 nan        	 nan        	 0          
	sin2pi_0_OW     	 0          	-0.0908166  	 nan        	 nan        	 0          
	sin4pi_0_OW     	 0          	 0.030409   	 nan        	 nan        	 0          
	sin6pi_0_OW     	 0          	 0.00678247 	 nan        	 nan        	 0          
	cos2pi_0_OW     	 0          	-0.154815   	 nan        	 nan        	 0          
	cos4pi_0_OW     	 0          	-0.130647   	 nan        	 nan        	 0          
	cos6pi_0_OW     	 0          	-0.0811669  	 nan        	 nan        	 0          
	sin2pi_0_OB     	 0          	 0.00748833 	 nan        	 nan        	 0          
	sin4pi_0_OB     	 0          	 0.0386351  	 nan        	 nan        	 0          
	sin6pi_0_OB     	 0          	 0.0245834  	 nan        	 nan        	 0          
	cos2pi_0_OB     	 0          	-0.108559   	 nan        	 nan        	 0          
	cos4pi_0_OB     	 0          	-0.0573955  	 nan        	 nan        	 0          
	cos6pi_0_OB     	 0          	-0.0168128  	 nan        	 nan        	 0          
	sin2pi_0_IB     	 0          	-0.0779617  	 nan        	 nan        	 0          
	sin4pi_0_IB     	 0          	 0.0490604  	 nan        	 nan        	 0          
	sin6pi_0_IB     	 0          	-0.0037936  	 nan        	 nan        	 0          
	cos2pi_0_IB     	 0          	 0.00643352 	 nan        	 nan        	 0          
	cos4pi_0_IB     	 0          	-0.0437634  	 nan        	 nan        	 0          
	cos6pi_0_IB     	 0          	-0.0458394  	 nan        	 nan        	 0          
	sin2pi_1_OW     	 0          	-0.0896607  	 nan        	 nan        	 0          
	sin4pi_1_OW     	 0          	-0.0741084  	 nan        	 nan        	 0          
	sin6pi_1_OW     	 0          	-0.0221     	 nan        	 nan        	 0          
	cos2pi_1_OW     	 0          	 0.00213618 	 nan        	 nan        	 0          
	cos4pi_1_OW     	 0          	-0.0183732  	 nan        	 nan        	 0          
	cos6pi_1_OW     	 0          	 0.0319627  	 nan        	 nan        	 0          
	sin2pi_1_OB     	 0          	-0.0505187  	 nan        	 nan        	 0          
	sin4pi_1_OB     	 0          	-0.046858   	 nan        	 nan        	 0          
	sin6pi_1_OB     	 0          	-0.0286364  	 nan        	 nan        	 0          
	cos2pi_1_OB     	 0          	-0.18624    	 nan        	 nan        	 0          
	cos4pi_1_OB     	 0          	-0.113927   	 nan        	 nan        	 0          
	cos6pi_1_OB     	 0          	 0.00489613 	 nan        	 nan        	 0          
	sin2pi_1_IB     	 0          	-0.0591861  	 nan        	 nan        	 0          
	sin4pi_1_IB     	 0          	-0.0305713  	 nan        	 nan        	 0          
	sin6pi_1_IB     	 0          	 0.0151741  	 nan        	 nan        	 0          
	cos2pi_1_IB     	 0          	-0.0184999  	 nan        	 nan        	 0          
	cos4pi_1_IB     	 0          	-0.0291761  	 nan        	 nan        	 0          
	cos6pi_1_IB     	 0          	-0.0543221  	 nan        	 nan        	 0          
	====================================================================================================
	Model Estimation Statistics
	----------------------------------------------------------------------------------------------------
	Log Likelihood at Convergence     	-27540.25
	Log Likelihood at Null Parameters 	-27722.67
	----------------------------------------------------------------------------------------------------
	Rho Squared w.r.t. Null Parameters	0.007
	====================================================================================================
	...








.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(220)

