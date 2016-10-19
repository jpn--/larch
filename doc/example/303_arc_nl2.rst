.. currentmodule:: larch

===============================================
303: Itinerary Choice using Double Nested Logit
===============================================

.. testsetup:: *

   import larch


This example is an itinerary choice model built using the example
itinerary choice dataset included with Larch.
As usual, we first create the DB objects:

.. testcode::

	d = larch.DB.Example('AIR')

As with the simple nested logit, we need to renumber the alternatives.
In this more complex example, we will be nesting on both time of day and level of service,
so our numbering system must account for both:

.. testcode::

	from enum import Enum

	class levels_of_service(Enum):
		nonstop = 1
		withstop = 0

	class time_of_day(Enum):
		morning = 1
		midday = 2
		evening = 3

	from larch.util.numbering import numbering_system
	ns = numbering_system(levels_of_service, time_of_day)

Then we can use a special command on the DB object to assign new alternative
numbers.

.. testcode::

	d.recode_alts(ns, 'data', 'id_case', 'itinerarycode_nl2',
		'nb_cnxs==0', 'CASE WHEN timeperiod<=3 THEN 1 WHEN timeperiod>=7 THEN 3 ELSE 2 END',
		newaltstable='itinerarycodes_nl2',
	)

As arguments to this command, we provide the numbering system object, the
name of the table that contains the idca data to be numbered (here `data`),
the name of the column in that table that names the caseids, and the name of
a new column to be created (or overwritten) with the new code numbers.  We also
need to give a set of SQL expressions that can be evaluated on the rows of the
table to get the categorical values that we defined in the Enums above. In this example,
we give two terms: first `nb_cnxs==0`, which evaluates to 1 when the itinerary is nonstop, and 0 if
the itinerary has a stop, exactly as given in the levels_of_service Enum above, because it is the
first argument in defining the numbering system.
Second we give `CASE WHEN timeperiod<=3 THEN 1 WHEN timeperiod>=7 THEN 3 ELSE 2 END`, which divides
our 9 time periods into 3 nests using a
`standard "CASE WHEN ... END" conditional clause <http://www.sqlite.org/lang_expr.html>`_
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


To build a two level nested logit, we can simply expand the looping structure used in
the one level NL:

.. testcode::

	for tod in time_of_day:
		tod_nest = m.new_nest(tod.name, param_name="mu_tod", parent=m.root_id)
		for los in levels_of_service:
			los_nest = m.new_nest(tod.name+los.name, param_name="mu_los", parent=tod_nest)
			for a in d.alternative_codes():
				if ns.code_matches_attributes(a, los, tod):
					m.link(los_nest, a)


To estimate the likelihood maximizing parameters, again we give:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike('SLSQP', metaoptions={'ftol': 1e-10}, ctol=1e-10)

.. doctest::
	:hide: 

	>>> m.parameter('timeperiod=9', value=-0.07962, initial_value=0)
	ModelParameter('timeperiod=9', value=-0.07962)


.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> print(m.report('txt', sigfigs=3))
	============================================================================================...
	Model Parameter Estimates
	--------------------------------------------------------------------------------------------...
	Parameter   	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue
	timeperiod=2	 0.0        	 0.0814     	 0.00747    	 10.9       	 0.0        
	timeperiod=3	 0.0        	 0.102      	 0.00754    	 13.5       	 0.0        
	timeperiod=4	 0.0        	 0.0684     	 0.00824    	 8.3        	 0.0        
	timeperiod=5	 0.0        	 0.108      	 0.00833    	 13.0       	 0.0        
	timeperiod=6	 0.0        	 0.201      	 0.0084     	 23.9       	 0.0        
	timeperiod=7	 0.0        	 0.251      	 0.00972    	 25.8       	 0.0        
	timeperiod=8	 0.0        	 0.258      	 0.00994    	 26.0       	 0.0        
	timeperiod=9	 0.0        	-0.0796     	 0.00994    	-8.01       	 0.0
	carrier=2   	 0.0        	 0.0933     	 0.00687    	 13.6       	 0.0        
	carrier=3   	 0.0        	 0.491      	 0.00889    	 55.3       	 0.0        
	carrier=4   	 0.0        	 0.425      	 0.0152     	 28.0       	 0.0        
	carrier=5   	 0.0        	-0.497      	 0.0117     	-42.5       	 0.0        
	equipment=2 	 0.0        	 0.371      	 0.00852    	 43.6       	 0.0        
	fare_hy     	 0.0        	-0.000945   	 2.52e-05   	-37.4       	 0.0        
	fare_ly     	 0.0        	-0.000999   	 6.82e-05   	-14.6       	 0.0        
	elapsed_time	 0.0        	-0.00458    	 0.000107   	-42.7       	 0.0        
	nb_cnxs     	 0.0        	-2.79       	 0.0827     	-33.7       	 0.0        
	mu_tod      	 1.0        	 0.911      	 0.0236     	-3.76       	 1.0        
	mu_los      	 1.0        	 0.778      	 0.00955    	-23.3       	 1.0        
	============================================================================================...
	Model Estimation Statistics
	--------------------------------------------------------------------------------------------...
	Log Likelihood at Convergence     	-777495.43
	Log Likelihood at Null Parameters 	-953940.44
	--------------------------------------------------------------------------------------------...
	Rho Squared w.r.t. Null Parameters	0.185
	============================================================================================...


.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(303)

