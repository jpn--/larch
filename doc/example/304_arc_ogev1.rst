.. currentmodule:: larch

===============================================
304: Itinerary Choice using Ordered GEV
===============================================

.. testsetup:: *

   import larch


This example is an itinerary choice model built using the example
itinerary choice dataset included with Larch.
As usual, we first create the DB objects:

.. testcode::

	d = larch.DB.Example('AIR')

As with the nested logit, we need to renumber the alternatives.
In this example, we will be grouping on finer time of day groups,
so our numbering system must account for this:

.. testcode::

	from enum import IntEnum

	class time_of_day(IntEnum):
		tp1 = 1
		tp2 = 2
		tp3 = 3
		tp4 = 4
		tp5 = 5
		tp6 = 6
		tp7 = 7
		tp8 = 8
		tp9 = 9

	from larch.util.numbering import numbering_system
	ns = numbering_system(time_of_day)

Then we can use a special command on the DB object to assign new alternative
numbers.

.. testcode::

	d.recode_alts(ns, 'data', 'id_case', 'itinerarycode_ogev1',
		'timeperiod',
		newaltstable='itinerarycodes_ogev1',
	)

As arguments to this command, we provide the numbering system object, the
name of the table that contains the idca data to be numbered (here `data`),
the name of the column in that table that names the caseids, and the name of
a new column to be created (or overwritten) with the new code numbers.  We also
need to give a set of SQL expressions that can be evaluated on the rows of the
table to get the categorical values that we defined in the Enums above. In this example,
we give just one terms: `timeperiod`, which already contains
our 9 time periods exactly in the correct format.
Lastly, we can pass the name of a new table that will be created to identify every
observed alternative code.

Once we have completed the preparation of the data, we can build out model.


Now let's make our model.  The utility function we will use is the same as the one we used for
the MNL version of the model.

.. testcode::

	m = larch.Model(d)

	vars = [
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
		"equipment=2",
		"fare_hy",
		"fare_ly",    
		"elapsed_time",  
		"nb_cnxs",       
	]
	from larch.roles import PX
	m.utility.ca = sum(PX(i) for i in vars)


To build a simple OGEV model with balanced allocations, we can use a similar the looping structure
as we used in the NL model:

.. testcode::

	# Ensure time periods are sorted
	time_of_day_s = sorted(time_of_day)
	# Overlapping nests
	for tod1, tod2 in zip(time_of_day_s[:], time_of_day_s[1:]):
		tod_nest = m.new_nest(tod1.name+tod2.name, param_name="mu_tod", parent=m.root_id)
		for a in d.alternative_codes():
			if ns.code_matches_attributes(a, tod1) or ns.code_matches_attributes(a, tod2):
				m.link(tod_nest, a)
	# First and last nest
	for tod in (time_of_day_s[0], time_of_day_s[-1]):
		tod_nest = m.new_nest(tod.name+"only", param_name="mu_tod", parent=m.root_id)
		for a in d.alternative_codes():
			if ns.code_matches_attributes(a, tod):
				m.link(tod_nest, a)


To estimate the likelihood maximizing parameters, again we give:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike('SLSQP', metaoptions={'ftol': 1e-10}, ctol=1e-10)


.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> print(m.report('txt', sigfigs=3))
	============================================================================================...
	Model Parameter Estimates
	--------------------------------------------------------------------------------------------...
	timeperiod=2	 0.0        	 0.213      	 0.00914    	 23.3       	 0.0
	timeperiod=3	 0.0        	 0.248      	 0.00967    	 25.7       	 0.0        
	timeperiod=4	 0.0        	 0.193      	 0.01       	 19.3       	 0.0        
	timeperiod=5	 0.0        	 0.238      	 0.00956    	 24.9       	 0.0        
	timeperiod=6	 0.0        	 0.333      	 0.00948    	 35.1       	 0.0        
	timeperiod=7	 0.0        	 0.448      	 0.00961    	 46.6       	 0.0        
	timeperiod=8	 0.0        	 0.371      	 0.00963    	 38.5       	 0.0        
	timeperiod=9	 0.0        	 0.0572     	 0.0106     	 5.41       	 0.0        
	carrier=2   	 0.0        	 0.0952     	 0.00691    	 13.8       	 0.0        
	carrier=3   	 0.0        	 0.491      	 0.0083     	 59.2       	 0.0        
	carrier=4   	 0.0        	 0.438      	 0.0147     	 29.9       	 0.0        
	carrier=5   	 0.0        	-0.508      	 0.0111     	-45.6       	 0.0        
	equipment=2 	 0.0        	 0.379      	 0.00811    	 46.7       	 0.0        
	fare_hy     	 0.0        	-0.000941   	 2.41e-05   	-39.1       	 0.0        
	fare_ly     	 0.0        	-0.000972   	 6.87e-05   	-14.2       	 0.0        
	elapsed_time	 0.0        	-0.00476    	 0.000102   	-46.7       	 0.0        
	nb_cnxs     	 0.0        	-2.33       	 0.0307     	-75.9       	 0.0        
	mu_tod      	 1.0        	 0.782      	 0.00795    	-27.4       	 1.0        
	============================================================================================...
	Model Estimation Statistics
	--------------------------------------------------------------------------------------------...
	Log Likelihood at Convergence     	-777441.22
	Log Likelihood at Null Parameters 	-953940.44
	--------------------------------------------------------------------------------------------...
	Rho Squared w.r.t. Null Parameters	0.185
	============================================================================================...


.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(304)

