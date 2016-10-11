.. currentmodule:: larch

=================================
301: Itinerary Choice using MNL
=================================

.. testsetup:: *

   import larch



This example is an itinerary choice model built using the example
itinerary choice dataset included with Larch.
As usual, we first create the DB objects:

.. testcode::

	d = larch.DB.Example('AIR')


Now let's make our model.  We'll use a few variables to define our
linear-in-parameters utility function.

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

The larch.roles module defines a few convenient classes for declaring data and parameter.
One we will use here is `PX` which creates a linear-in-parameter term that represents one data
element (a column from our data, or an expression that can be evaluated on the data alone) multiplied
by a parameter with the same name.

.. testcode::

	from larch.roles import PX
	m.utility.ca = sum(PX(i) for i in vars)

The one line of code above is equivalent to writing::

	m.utility.ca = P("tp2") * X("tp2") + P("tp3") * X("tp3") + ...

but obviously much simpler to write and maintain.

Since we are estimating just an MNL model in this example, this is all we need to do to build
our model, and we're ready to go.  To estimate the likelihood maximizing parameters, we give:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike()
	>>> result.message
	'Optimization terminated successfully per computed tolerance. [bhhh]'

	>>> print(m.report('txt', sigfigs=3))
	============================================================================================...
	Model Parameter Estimates
	--------------------------------------------------------------------------------------------...
	Parameter   	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue
	timeperiod=2     0.0        	 0.0959     	 0.00948    	 10.1       	 0.0
	timeperiod=3     0.0        	 0.126      	 0.00953    	 13.3       	 0.0
	timeperiod=4     0.0        	 0.0605     	 0.00978    	 6.19       	 0.0
	timeperiod=5     0.0        	 0.141      	 0.00973    	 14.5       	 0.0
	timeperiod=6     0.0        	 0.238      	 0.00973    	 24.5       	 0.0
	timeperiod=7     0.0        	 0.351      	 0.00996    	 35.3       	 0.0
	timeperiod=8     0.0        	 0.353      	 0.0105     	 33.8       	 0.0
	timeperiod=9     0.0        	-0.0104     	 0.011      	-0.945      	 0.0
	carrier=2    	 0.0        	 0.117      	 0.00869    	 13.5       	 0.0
	carrier=3    	 0.0        	 0.639      	 0.00813    	 78.6       	 0.0
	carrier=4    	 0.0        	 0.565      	 0.0176     	 32.2       	 0.0
	carrier=5    	 0.0        	-0.624      	 0.013      	-48.2       	 0.0
	equipment=2  	 0.0        	 0.466      	 0.00931    	 50.1       	 0.0
	fare_hy     	 0.0        	-0.00117    	 2.83e-05   	-41.5       	 0.0        
	fare_ly     	 0.0        	-0.00118    	 8.51e-05   	-13.8       	 0.0        
	elapsed_time	 0.0        	-0.00609    	 0.000111   	-54.6       	 0.0        
	nb_cnxs     	 0.0        	-2.95       	 0.0254     	-1.16e+02   	 0.0        
	============================================================================================...
	Model Estimation Statistics
	--------------------------------------------------------------------------------------------...
	Log Likelihood at Convergence     	-777770.07
	Log Likelihood at Null Parameters 	-953940.44
	--------------------------------------------------------------------------------------------...
	Rho Squared w.r.t. Null Parameters	0.185
	============================================================================================...



.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(301)

