.. currentmodule:: larch

================================
203: OGEV-NL Itinerary Choice
================================

.. testsetup:: *

   import larch



This example is an multi-level OGEV-NL itinerary choice model built using the example
itinerary choice dataset included with Larch.
As usual, we first create the DB objects:

.. testcode::

	db = larch.DB.Example('ITINERARY')

Our itinerary choice data has a lot of alternatives, but they are not
ordered or numbered in a regular way; each elemental alternative has
an arbitrary code number assigned to it, and the code numbers for one case
are not comparable to another case. But we can renumber the alternatives in
a manner that is more suited for our application, such that based on the code
number we can programatically extract a few relevant features of the alternative
that we will want to use in building our OGEV-NL model.  Suppose for example
that we want to test a model which has a departure time of day OGEV structure with
carriers nested inside time of day.  To assign
each alternative into this structure, obviously we'll need to be able to easily
identify the departure time (here we will group by hours) and carrier for each alternative.  To renumber,
first we will define the relevant categories and values, and establish a numbering
system using a special object:

.. testcode::

	from enum import Enum

	class departure_hours(Enum):
		h0 = 0
		h1 = 1
		h2 = 2
		h3 = 3
		h4 = 4
		h5 = 5
		h6 = 6
		h7 = 7
		h8 = 8
		h9 = 9
		h10 = 10
		h11 = 11
		h12 = 12
		h13 = 13
		h14 = 14
		h15 = 15
		h16 = 16
		h17 = 17
		h18 = 18
		h19 = 19
		h20 = 20
		h21 = 21
		h22 = 22
		h23 = 23

	class carriers(Enum):
		Robin = 1
		Cardinal = 2
		Bluejay = 3
		Heron = 4 
		Other = 5

	from larch.util.numbering import numbering_system
	ns = numbering_system(departure_hours, carriers)

Then we can use a special command on the DB object to assign new alternative
numbers.

.. testcode::

	db.recode_alts(ns, 'data_ca', 'casenum', 'ogevitinerarycode',
		'depart_time/60', 'MIN(carrier,5)',
		newaltstable='ogevitinerarycodes',
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

	m = larch.Model(db)

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

	alts = db.alternative_codes()

Then we can build the network by looping over various categories to define
the nodes.

.. testcode::

	hour_carrier_nests = {}
	for hour in departure_hours:
		for carrier in carriers:
			n = m.new_nest(hour.name+carrier.name, param_name="mu_carrier")
			hour_carrier_nests[hour,carrier] = n
			for a in alts:
				if ns.code_matches_attributes(a, hour, carrier):
					m.link(n, a)

In this block, we define a set of nests by hour and carrier, and allocate the elemental alternatives into
those nests. Note we reuse the same
`ns` object from when we did the alternative renumbering, as it knows how to quickly extract
the relevant attributes from the code number.


.. testcode::

	for hour in departure_hours:
		prev_hour = departure_hours((hour.value-1)%24)
		next_hour = departure_hours((hour.value+1)%24)
		time_nest = m.new_nest(hour.name, param_name="mu_ogev", parent=m.root_id)
		for carrier in carriers:
			m.link(time_nest, hour_carrier_nests[hour,carrier])
			m.link(time_nest, hour_carrier_nests[next_hour,carrier])
			m.link[time_nest, hour_carrier_nests[next_hour,carrier]](data='1',param='PHI_next')
			m.link(time_nest, hour_carrier_nests[prev_hour,carrier])
			m.link[time_nest, hour_carrier_nests[prev_hour,carrier]](data='1',param='PHI_prev')

In this second block, the outermost loop is over departure hour.  Within that, we define the
previous and next hours (this is easily expandible to include multiple hours in each OGEV nest)
and a nest to group together the three hours.  Then we loop over carriers link the
nests we created in the previous block, with one link to each current hour carrier nest, and
one link to each next hour carrier nest.  We also add a PHI parameter to the second link to
control the allocation of the hour-carrier nests to the OGEV level nests.


For this example, since we want it to run quickly, we'll limit the input
data to only a few cases, and turn off the the automatic calculation of standard errors.
We'll also tell the optimization engine to enforce logsum parameter ordering constraints.

.. testcode::

	filter = 'casenum < 20000'
	db.queries.idca_build(filter=filter)
	db.queries.idco_build(filter=filter)
	m.option.calc_std_errors = False
	m.option.enforce_constraints = False
	m.option.enforce_bounds = False

Now it's time to run it and see what we get.  We'll use the SLSQP algorithm because it can
use the automatic parameter constraints (most of the available algorithms cannot):

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike('SLSQP', metaoptions={'ftol': 1e-08})
	>>> result.message
	'Optimization terminated successfully...

	>>> print(m.report('txt', cats=['params'], param='< 12.3g'))
	====================================================================================================
	Model Parameter Estimates
	----------------------------------------------------------------------------------------------------
	Parameter       	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue   
	carrier=2       	 0          	-0.209      	 nan        	 nan        	 0          
	carrier=3       	 0          	 0.00105    	 nan        	 nan        	 0          
	carrier=4       	 0          	-0.0577     	 nan        	 nan        	 0          
	carrier>=5      	 0          	 0.046      	 nan        	 nan        	 0          
	aver_fare_hy    	 0          	-0.00231    	 nan        	 nan        	 0          
	aver_fare_ly    	 0          	-0.000213   	 nan        	 nan        	 0          
	itin_num_cnxs   	 0          	-0.442      	 nan        	 nan        	 0          
	itin_num_directs	 0          	-0.108      	 nan        	 nan        	 0          
	mu_carrier      	 1          	 0.844      	 nan        	 nan        	 1          
	mu_ogev         	 1          	 0.922      	 nan        	 nan        	 1          
	PHI_next        	 0          	 0.737      	 nan        	 nan        	 0          
	PHI_prev        	 0          	 0.00259    	 nan        	 nan        	 0          
	====================================================================================================

	>>> print(m.report('txt', cats=['LL']))
	================================================
	Model Estimation Statistics
	------------------------------------------------
	Log Likelihood at Convergence     	-6084.54
	Log Likelihood at Null Parameters 	-6115.43
	------------------------------------------------
	Rho Squared w.r.t. Null Parameters	0.005
	================================================



Don't worry that these results don't look great; we used a very tiny dataset here with a
complex model.  This example was just to walk you through the mechanics of specifying and
estimating this model, not showing you how to get a good model with the data, which is an
exercise left for the reader.

.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(203)

