.. currentmodule:: larch

=================================================
306: Segmented Itinerary Choice using a MetaModel
=================================================

.. testsetup:: *

   import larch


This example is an itinerary choice model built using the example
itinerary choice dataset included with Larch.

For this example, we are going to estimate a segmented model with a MetaModel.
MetaModels are a way to estimate a group of models simultaneously. In this case,
we will estimate two models, one for each market segment, where some of the
parameters are estimated in common for both segments and some are segment-specific.

.. note::

	In this example, because our example dataset is just a small sliver of the much larger
	set of observations, we're just going to segment on one variable: equipment type. This probably
	isn't how you would want to segment an itinerary choice model in practice, but segmenting
	on much more with this small of a dataset will likely result in parameter instability,
	which we don't want to deal with in this demostration.

.. testcode::

	import larch
	import itertools
	from larch.metamodel import MetaModel

	db = larch.DB.Example('AIR', shared=True)

	traveler_segments = [1,2]
	direction_segments = ['east','west']

	common_vars = [
		"timeperiod=2",
		"timeperiod=3",
		"timeperiod=4",
		"timeperiod=5",
		"timeperiod=6",
		"timeperiod=7",
		"timeperiod=8",
		"timeperiod=9",
		"carrier=2",
		"carrier=3",
		"carrier=4",
		"carrier=5",
		"fare_hy",
		"fare_ly",    
		"elapsed_time",  
		"nb_cnxs",       
	]

	segmented_vars = [
		"equipment=2",
	]


We can construct a MetaModel in two general ways: manually (explicitly giving every submodel), or by using a
submodel factory function.  Such a function must take a segment descriptor, plus some set of other arguments,
and return a submodel for the given segment.

Here we show an example of a submodel factory function, which takes the (shared cache) db object, plus lists
of common and segemented variables, and returns a submodel.  The db object needs to have a shared cache
because we will use the NewConnection method of the DB class to spawn a new DB object that shares the same
underlying database.  This lets us have several different database connections, each with its own QuerySet,
so that each submodel can use a unique data set pulled from the master data.

The advantage of using the submodel factory is that the rest of the set up of the MetaModel can be handled
automatically.


.. testcode::

	def submodel_factory(segment_desciptor, db, common_vars, segmented_vars):
		# Create a new larch.DB based on the shared DB
		dx = larch.DB.NewConnection(db)
		# introduce a case filter to apply to the data table, to get the segment
		casefilter = " WHERE traveler=={0} AND CASE WHEN origin<destination THEN 'east' ELSE 'west' END=='{1}'".format(*segment_desciptor)
		dx.queries = qry = larch.core.QuerySetTwoTable(dx)
		qry.idco_query = "SELECT distinct id_case AS caseid, traveler, CASE WHEN origin<destination THEN 'east' ELSE 'west' END AS direction FROM data "+casefilter
		qry.idca_query = "SELECT id_case AS caseid, id_alt AS altid, * FROM data "+casefilter
		# The rest of the QuerySet is defined as usual and is the same for all segments
		qry.alts_query = "SELECT * FROM csv_alternatives "
		qry.choice = 'choice'
		qry.avail = '1'
		qry.weight = '1'
		dx.refresh_queries()
		# Create a new submodel using the filtered data DB
		submodel = larch.Model(dx)
		# If the submodel has no cases, skip the rest of setting it up
		if submodel.df.nCases()==0:
			return submodel
		# Populate the submodel with the common parameters
		for var in common_vars:
			submodel.utility.ca(var)
		# Populate the submodel with the segmented parameters
		for var in segmented_vars:
			built_par = var+"({},{})".format(*segment_desciptor)
			submodel.utility.ca(var, built_par)
		submodel.setUp()
		return submodel

Thst should be all we need to create our metamodel.  Then to actually create the MetaModel object,
we'll need to give an iterator over all the segements, the submodel factory, and a tuple with
the arguments for the factory:

.. testcode::

	m = MetaModel( itertools.product(traveler_segments, direction_segments), submodel_factory, args=(db, common_vars, segmented_vars) )



Let's confirm that we got a model that has the parameters we want.

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> print(m)
	====================================================================================================
	Model Parameter Estimates
	----------------------------------------------------------------------------------------------------
	Parameter         	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue
	timeperiod=2      	 0          	 0          	 nan        	 nan        	 0
	timeperiod=3      	 0          	 0          	 nan        	 nan        	 0          
	timeperiod=4      	 0          	 0          	 nan        	 nan        	 0          
	timeperiod=5      	 0          	 0          	 nan        	 nan        	 0          
	timeperiod=6      	 0          	 0          	 nan        	 nan        	 0          
	timeperiod=7      	 0          	 0          	 nan        	 nan        	 0          
	timeperiod=8      	 0          	 0          	 nan        	 nan        	 0          
	timeperiod=9      	 0          	 0          	 nan        	 nan        	 0          
	carrier=2         	 0          	 0          	 nan        	 nan        	 0          
	carrier=3         	 0          	 0          	 nan        	 nan        	 0          
	carrier=4         	 0          	 0          	 nan        	 nan        	 0          
	carrier=5         	 0          	 0          	 nan        	 nan        	 0          
	fare_hy           	 0          	 0          	 nan        	 nan        	 0          
	fare_ly           	 0          	 0          	 nan        	 nan        	 0          
	elapsed_time      	 0          	 0          	 nan        	 nan        	 0          
	nb_cnxs           	 0          	 0          	 nan        	 nan        	 0          
	equipment=2(1,east)	 0          	 0          	 nan        	 nan        	 0
	equipment=2(1,west)	 0          	 0          	 nan        	 nan        	 0
	equipment=2(2,east)	 0          	 0          	 nan        	 nan        	 0
	equipment=2(2,west)	 0          	 0          	 nan        	 nan        	 0
	====================================================================================================
	...



Yup, looks good.  We have one of all of the common parameters, plus a set of the segmented parameters for
each segment.  Now let's estimate our model.  We'll turn off the calculation of standard errors, because
that takes a bit of time and we're not interested in those results yet.

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> m.option.calc_std_errors = False
	>>> r = m.maximize_loglike("SLSQP")
	>>> print(r)
	messages: Optimization terminated successfully per computed tolerance...

	>>> print(m.report('txt', sigfigs=3))
	====================================================================================================
	Model Parameter Estimates
	----------------------------------------------------------------------------------------------------
	Parameter         	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue
	timeperiod=2      	 0.0        	 0.0974     	 nan        	 nan        	 0.0        
	timeperiod=3      	 0.0        	 0.134      	 nan        	 nan        	 0.0        
	timeperiod=4      	 0.0        	 0.0599     	 nan        	 nan        	 0.0        
	timeperiod=5      	 0.0        	 0.142      	 nan        	 nan        	 0.0        
	timeperiod=6      	 0.0        	 0.242      	 nan        	 nan        	 0.0        
	timeperiod=7      	 0.0        	 0.357      	 nan        	 nan        	 0.0        
	timeperiod=8      	 0.0        	 0.36       	 nan        	 nan        	 0.0        
	timeperiod=9      	 0.0        	-0.00302    	 nan        	 nan        	 0.0        
	carrier=2         	 0.0        	 0.153      	 nan        	 nan        	 0.0        
	carrier=3         	 0.0        	 0.67       	 nan        	 nan        	 0.0        
	carrier=4         	 0.0        	 0.473      	 nan        	 nan        	 0.0        
	carrier=5         	 0.0        	-0.587      	 nan        	 nan        	 0.0        
	fare_hy           	 0.0        	-0.00116    	 nan        	 nan        	 0.0        
	fare_ly           	 0.0        	-0.00152    	 nan        	 nan        	 0.0        
	elapsed_time      	 0.0        	-0.00606    	 nan        	 nan        	 0.0        
	nb_cnxs           	 0.0        	-2.86       	 nan        	 nan        	 0.0        
	equipment=2(1,east)	 0.0        	 0.392      	 nan        	 nan        	 0.0
	equipment=2(1,west)	 0.0        	 0.393      	 nan        	 nan        	 0.0
	equipment=2(2,east)	 0.0        	 0.688      	 nan        	 nan        	 0.0
	equipment=2(2,west)	 0.0        	 0.344      	 nan        	 nan        	 0.0
	====================================================================================================
	Model Estimation Statistics
	----------------------------------------------------------------------------------------------------
	Log Likelihood at Convergence     	-777603.28
	Log Likelihood at Null Parameters 	-953940.44
	----------------------------------------------------------------------------------------------------
	Rho Squared w.r.t. Null Parameters	0.185
	====================================================================================================
	...



.. tip::

	If you want access to the metamodel in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(306)
