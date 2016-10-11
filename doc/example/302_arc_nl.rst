.. currentmodule:: larch

===============================================
302: Itinerary Choice using Simple Nested Logit
===============================================

.. testsetup:: *

   import larch


This example is an itinerary choice model built using the example
itinerary choice dataset included with Larch.
As usual, we first create the DB objects:

.. testcode::

	d = larch.DB.Example('AIR')

We will be building a nested logit model, but in order to do so we need to rationalize the alternative
numbers.  As given, our raw itinerary choice data has a lot of alternatives, but they are not
ordered or numbered in a regular way; each elemental alternative has
an arbitrary code number assigned to it, and the code numbers for one case
are not comparable to another case. We need to renumber the alternatives in
a manner that is more suited for our application, such that based on the code
number we can programatically extract a the relevant features of the alternative
that we will want to use in building our nested logit model.  In this example
we want to test a model which has nests based on level of service.
To renumber, first we will define the relevant categories and values, and establish a numbering
system using a special object:

.. testcode::

	from enum import Enum

	class levels_of_service(Enum):
		nonstop = 1
		withstop = 0

	from larch.util.numbering import numbering_system
	ns = numbering_system(levels_of_service)

Then we can use a special command on the DB object to assign new alternative
numbers.

.. testcode::

	d.recode_alts(ns, 'data', 'id_case', 'itinerarycode_nl1',
		'nb_cnxs==0',
		newaltstable='itinerarycodes_nl1',
	)

As arguments to this command, we provide the numbering system object, the
name of the table that contains the idca data to be numbered (here `data`),
the name of the column in that table that names the caseids, and the name of
a new column to be created (or overwritten) with the new code numbers.  We also
need to give a set of SQL expressions that can be evaluated on the rows of the
table to get the categorical values that we defined in the Enums above. In this example,
we give `nb_cnxs==0`, which evaluates to 1 when the itinerary is nonstop, and 0 if
the itinerary has a stop, exactly as given in the levels_of_service Enum above.
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


If we just end our model specification here, we will have a plain MNL model.  To change to
a nested logit model, all we need to do is add the nests.  We can do this easily, leveraging
some special features of the numbering system class we used above.  First, we loop over the
levels of service, creating a new nest for each.  Then we loop over the alternatives, adding
all the relevant alternatives to each nest.

.. testcode::

	for los in levels_of_service:
		los_nest = m.new_nest(los.name, param_name="mu_los", parent=m.root_id)
		for a in d.alternative_codes():
			if ns.code_matches_attributes(a, los):
				m.link(los_nest, a)

For a simple static two level model, it might be nearly as easy to explicitly renumber
the alternatives and assign them to the appropriate nests.  The advantage of the
coding shown here is that it automatically keeps the renumbering and the nesting
structure in sync, even if the details of the breakdown of nests changes.  For instance,
to change to have LOS nests based on the number of connections (instead of just the presence
of any connection), we could simply change the `ns` object and the middle argument of the
recode_alts method, and the rest of the code
will adapt automatically.

To estimate the likelihood maximizing parameters, again we give:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike()
	>>> print(m.report('txt', sigfigs=3))
	============================================================================================...
	Model Parameter Estimates
	--------------------------------------------------------------------------------------------...
	Parameter   	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue
	timeperiod=2     0.0        	 0.0725     	 0.00767    	 9.45       	 0.0
	timeperiod=3     0.0        	 0.097      	 0.00795    	 12.2       	 0.0
	timeperiod=4     0.0        	 0.0471     	 0.00762    	 6.18       	 0.0
	timeperiod=5     0.0        	 0.107      	 0.00829    	 12.9       	 0.0
	timeperiod=6     0.0        	 0.182      	 0.00964    	 18.9       	 0.0
	timeperiod=7     0.0        	 0.269      	 0.0118     	 22.8       	 0.0
	timeperiod=8     0.0        	 0.27       	 0.0121     	 22.4       	 0.0
	timeperiod=9     0.0        	-0.00687    	 0.00837    	-0.821      	 0.0
	carrier=2    	 0.0        	 0.0883     	 0.00735    	 12.0       	 0.0
	carrier=3    	 0.0        	 0.486      	 0.0176     	 27.7       	 0.0
	carrier=4    	 0.0        	 0.436      	 0.0197     	 22.1       	 0.0
	carrier=5    	 0.0        	-0.483      	 0.0185     	-26.1       	 0.0
	equipment=2  	 0.0        	 0.36       	 0.0137     	 26.2       	 0.0
	fare_hy     	 0.0        	-0.000927   	 3.58e-05   	-25.9       	 0.0        
	fare_ly     	 0.0        	-0.000937   	 7.12e-05   	-13.2       	 0.0        
	elapsed_time	 0.0        	-0.00466    	 0.000177   	-26.3       	 0.0        
	nb_cnxs     	 0.0        	-3.11       	 0.028      	-1.11e+02   	 0.0        
	mu_los      	 1.0        	 0.763      	 0.0257     	-9.23       	 1.0        
	============================================================================================...
	Model Estimation Statistics
	--------------------------------------------------------------------------------------------...
	Log Likelihood at Convergence     	-777730.13
	Log Likelihood at Null Parameters 	-953940.44
	--------------------------------------------------------------------------------------------...
	Rho Squared w.r.t. Null Parameters	0.185
	============================================================================================...


.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(302)

