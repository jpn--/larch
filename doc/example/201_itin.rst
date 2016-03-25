.. currentmodule:: larch

=================================
201: Network GEV Itinerary Choice
=================================

.. testsetup:: *

   import larch



This example is an itinerary choice model built using the example
itinerary choice dataset included with Larch.
As usual, we first create the DB objects:

.. testcode::

	d = larch.DB.Example('ITINERARY')

Our itinerary choice data has a lot of alternatives, but they are not
ordered or numbered in a regular way; each elemental alternative has
an arbitrary code number assigned to it, and the code numbers for one case
are not comparable to another case. But we can renumber the alternatives in
a manner that is more suited for our application, such that based on the code
number we can programatically extract a few relevant features of the alternative
that we will want to use in building our Network GEV model.  Suppose for example
that we want to test a model which has level of service nested inside carriers on
one side, and carriers nested inside level of service on the other side.  To assign
each alternative into this structure, obviously we'll need to be able to easily
identify the level of service and carrier for each alternative.  To renumber,
first we will define the relevant categories and values, and establish a numbering
system using a special object:

.. testcode::

	from enum import Enum

	class levels_of_service(Enum):
		nonstop = 1
		withstop = 0
		
	class carriers(Enum):
		Robin = 1
		Cardinal = 2
		Bluejay = 3
		Heron = 4 
		Other = 5

	from larch.util.numbering import numbering_system
	ns = numbering_system(levels_of_service, carriers)

Then we can use a special command on the DB object to assign new alternative
numbers.

.. testcode::

	d.recode_alts(ns, 'data_ca', 'casenum', 'itinerarycode',
		'nonstop', 'MIN(carrier,5)',
		newaltstable='itinerarycodes',
	)

As arguments to this command, we provide the numbering system object, the
name of the table that contains the idca data to be numbered (here data_ca),
the name of the column in that table that names the caseids, and the name of
a new column to be created (or overwritten) with the new code numbers.  We also
need to give a set of SQL expressions that can be evaluated on the rows of the
table to get the categorical values that we defined in the Enums above. Lastly,
we can pass the name of a new table that will be created to identify every
observed alternative code.

Now let's make our model.  We'll use a few variables to define our
linear-in-parameters utility function.

.. testcode::

	m = larch.Model(d)

	vars = [
		"carrier=2",              
		"carrier=3",              
		"carrier=4",              
		"carrier>=5",
		"aver_fare_hy",
		"aver_fare_ly",           
		"itin_num_cnxs",
		"itin_num_directs",       
	]
	for var in vars:
		m.utility.ca(var)

Now it's time to build the network that will define our nesting structure.
First we'll need to have the list of all the alternative codes we created
above.

.. testcode::

	alts = d.alternative_codes()

Then we can build the network by looping over various categories to define
the nodes.

.. testcode::

	for los in levels_of_service:
		los_nest = m.new_nest(los.name, param_name="mu_los1", parent=m.root_id)
		for carrier in carriers:
			carrier_nest = m.new_nest(los.name+carrier.name, param_name="mu_carrier1", parent=los_nest)
			for a in alts:
				if ns.code_matches_attributes(a, los, carrier):
					m.link(carrier_nest, a)

In this first block, the outermost loop is over levels of service, and within that we
loop over carriers to create lower level nests, and then over the alternatives, linking
in those alternatives that match the level of service and carrier.  Note we reuse the same
`ns` object from when we did the alternative renumbering, as it knows how to quickly extract
the relevant attributes from the code number.

.. testcode::

	for carrier in carriers:
		carrier_nest = m.new_nest(carrier.name, param_name="mu_carrier2", parent=m.root_id)
		for los in levels_of_service:
			los_nest = m.new_nest(carrier.name+los.name, param_name="mu_los2", parent=carrier_nest)
			for a in alts:
				if ns.code_matches_attributes(a, los, carrier):
					m.link(los_nest, a)
					m.link[los_nest, a](data='1',param='PHI')

The second block mirrors the first block, except reversing the order of the categories,
so that the carrier nests are on top and the level of service nests are underneath them.
Also, we add one extra line after the link command, which associates a PHI parameter with
the link, to manage the allocation of the elemental alternatives to the nodes above them.
The PHI takes the place of alpha allocation parameters, and it is omitted in the earlier block
because one side needs to be implicitly normalized to zero.  For more details on this, check out
Newman (2008).

For this example, since we want it to run quickly, we'll limit the input
data to only a few cases, and turn off the the automatic calculation of standard errors.
We'll also tell the optimization engine to enforce logsum parameter ordering constraints.

.. testcode::

	filter = 'casenum < 20000'
	d.queries.idca_build(filter=filter)
	d.queries.idco_build(filter=filter)
	m.option.calc_std_errors = False
	m.option.enforce_constraints = True

By default, logsum parameters created automatically by the new_nest method have min/max bounds
set at 0.0/1.0.  But the network GEV can become numerically unstable if these parameters get too close
to zero, especially when the dataset is small (as it is here in this example). So we can set
a minimum value a little bit away from zero like this:

.. testcode::

	m['mu_los1'].minimum = 0.01
	m['mu_carrier1'].minimum = 0.01
	m['mu_los2'].minimum = 0.01
	m['mu_carrier2'].minimum = 0.01


Now it's time to run it and see what we get.  We'll use the SLSQP algorithm because it can
use the automatic parameter constraints (most of the available algorithms cannot):

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike('SLSQP', metaoptions={'ftol': 1e-08})
	>>> result.message
	'Optimization terminated successfully. [SLSQP]'

	>>> print(m.report('txt', sigfigs=3))
	=================================================================================================...
	Model Parameter Estimates
	-------------------------------------------------------------------------------------------------...
	Parameter       	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue   
	carrier=2       	 0.0        	-0.203      	 nan        	 nan        	 0.0        
	carrier=3       	 0.0        	 0.0054     	 nan        	 nan        	 0.0        
	carrier=4       	 0.0        	-0.04       	 nan        	 nan        	 0.0        
	carrier>=5      	 0.0        	 0.0294     	 nan        	 nan        	 0.0        
	aver_fare_hy    	 0.0        	-0.00216    	 nan        	 nan        	 0.0        
	aver_fare_ly    	 0.0        	-0.000162   	 nan        	 nan        	 0.0        
	itin_num_cnxs   	 0.0        	-0.477      	 nan        	 nan        	 0.0        
	itin_num_directs	 0.0        	-0.263      	 nan        	 nan        	 0.0        
	mu_los1         	 1.0        	 0.0505     	 nan        	 nan        	 1.0        
	mu_carrier1     	 1.0        	 0.045      	 nan        	 nan        	 1.0        
	mu_carrier2     	 1.0        	 1.0        	 nan        	 nan        	 1.0        
	mu_los2         	 1.0        	 1.0        	 nan        	 nan        	 1.0        
	PHI             	 0.0        	 1.21       	 nan        	 nan        	 0.0        
	=================================================================================================...
	Model Estimation Statistics
	-------------------------------------------------------------------------------------------------...
	Log Likelihood at Convergence     	-6086.89
	Log Likelihood at Null Parameters 	-6115.43
	-------------------------------------------------------------------------------------------------...
	Rho Squared w.r.t. Null Parameters	0.005
	=================================================================================================...


Don't worry that these results don't look great; we used a very tiny dataset here with a
complex model.  This example was just to walk you through the mechanics of specifying and
estimating this model, not showing you how to get a good model with the data, which is an
exercise left for the reader.

.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(201)

