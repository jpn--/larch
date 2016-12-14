.. currentmodule:: larch

=====================================
201: Exampville Mode Choice
=====================================

.. testsetup:: *

	import larch
	import os


Welcome to Exampville, the best simulated town in this here part of the internet!

Exampville is a simulation tool provided with Larch that can quickly simulate the
kind of data that a transportation planner might have available when building
a travel model.  We used the first Exampville builder to generate some
simulated data in the previous example, which we'll use here.

.. testcode::

	import larch, numpy, pandas, os
	from larch.roles import P,X
	d = larch.DT.Example(200)

In Exampville, there are only two kinds of trips: work (purpose=1) and non-work.
We want to estimate a mode choice model for work trips, so we'll begin by excluding
all the other trips:

.. testcode::

	d.exclude_idco("TOURPURP != 1")
	assert( d.nCases() == 1897 )

Now we are ready to create our model.

.. testcode::

	m = larch.Model(d)
	m.title = "Exampville Work Tour Mode Choice v1"


For this model, we want to ensure that the value of non-motorized travel time is
exactly double that of motorized in-vehicle time.  To do this, we can create
a "shadow parameter" for non-motorized time that is defined as 2 times the parameter
for motorized in-vehicle time.

.. testcode::

	m.parameter("InVehTime")                           # first create the original parameter...
	m.shadow_parameter.NonMotorTime = P.InVehTime * 2  # ...then we can create the shadow


Now we are ready to define some utility functions.

.. testcode::

	# For clarity, we can define numbers as names for modes
	DA = 1
	SR = 2
	Walk = 3
	Bike = 4
	Transit = 5

	m.utility.co[DA] = (
		+ P.InVehTime * X.AUTO_TIME
		+ P.Cost * X.CARCOST # dollars per mile
	)

	m.utility[SR] = (
		+ P.ASC_SR
		+ P.InVehTime * X.AUTO_TIME
		+ P.Cost * (X.CARCOST * 0.5) # dollars per mile, half share
		+ P("HighInc:SR") * X("INCOME>75000")
	)

Note in the SR utility that we use two different ways for writing parameters, with a dotted
name (`P.Cost`) and with a parenthesis (`P("HighInc:SR")`).  The dotted name version is neat and
concise, but it only works when the parameter name is a valid python identifier -- essentially, a
single word, beginning with a letter and containing only letter and numbers, and no spaces or punctuation.
Larch allows parameter names that are any string, including spaces and punctuation, but more interesting
names that are not python identifiers must be given using the second form.


.. testcode::

	m.utility[Walk] = (
		+ P.ASC_Walk
		+ P.NonMotorTime * X.WALKTIME 
		+ P("HighInc:Walk") * X("INCOME>75000")
	)

	m.utility[Bike] = (
		+ P.ASC_Bike
		+ P.NonMotorTime * X.BIKETIME
		+ P("HighInc:Bike") * X("INCOME>75000")
	)

	m.utility[Transit] = (
		+ P.ASC_Transit
		+ P.InVehTime * X.RAIL_TIME
		+ P.Cost * X.RAIL_FARE
		+ P("HighInc:Transit") * X("INCOME>75000")
	)


Let's create a nested logit model.  We'll nest together the two car modes, and the
two non-motorized modes, and then the car nest with the transit mode.

.. testcode::

	Car = m.new_nest('Nest:Car', children=[DA,SR])
	NonMotor = m.new_nest('Nest:NonMotor', children=[Walk,Bike])
	Motor = m.new_nest('Nest:Motorized', children=[Car,Transit])

We're also going to specify how we want to show parameters in the output.  We can group them into
categories using the parameter_groups attribute of Model objects, and the Categorizer class.
Categorizer instances are defined with a label as their first argument, and regular expressions (regex)
as other arguments.  The regex are evaluated against all parameter names, and those that match are
put into the category.

.. testcode::

	from larch.util.categorize import Categorizer

	m.parameter_groups = (
		Categorizer("Level of Service",
			".*Time.*",
			".*Cost.*",
		),
		Categorizer("Alternative Specific Constants",
			"ASC.*",
		),
		Categorizer("Income",
			".*HighInc.*",
			".*LowInc.*",
		),
		Categorizer("Logsum Parameters",
			"Nest.*",
		),
	)

Now we're ready to go.

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> m.maximize_loglike()
	messages: Optimization terminated successfully...

	>>> print(m.report('txt', sigfigs=3))
	=========================================================================================...
	Exampville Work Tour Mode Choice v1
	=========================================================================================...
	Model Parameter Estimates
	-----------------------------------------------------------------------------------------...
	Parameter      	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue   
	~~ Level of Service ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~...
	InVehTime      	 0.0        	-0.113      	 0.0136     	-8.31       	 0.0        
	Cost           	 0.0        	-0.341      	 0.181      	-1.88       	 0.0        
	~~ Alternative Specific Constants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~...
	ASC_SR         	 0.0        	-1.39       	 0.954      	-1.46       	 0.0        
	ASC_Walk       	 0.0        	 0.732      	 0.301      	 2.43       	 0.0        
	ASC_Bike       	 0.0        	-1.67       	 0.274      	-6.09       	 0.0        
	ASC_Transit    	 0.0        	-1.68       	 0.498      	-3.37       	 0.0        
	~~ Income ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~...
	HighInc:SR     	 0.0        	-1.76       	 1.29       	-1.36       	 0.0        
	HighInc:Walk   	 0.0        	-0.796      	 0.419      	-1.9        	 0.0        
	HighInc:Bike   	 0.0        	-1.05       	 0.463      	-2.27       	 0.0        
	HighInc:Transit	 0.0        	-1.24       	 0.451      	-2.75       	 0.0        
	~~ Logsum Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~...
	Nest:Car       	 1.0        	 0.742      	 0.538      	-0.48       	 1.0        
	Nest:NonMotor  	 1.0        	 0.87       	 0.183      	-0.71       	 1.0        
	Nest:Motorized 	 1.0        	 0.844      	 0.249      	-0.626      	 1.0        
	=========================================================================================...
	Model Estimation Statistics
	-----------------------------------------------------------------------------------------...
	Log Likelihood at Convergence     	-963.84
	Log Likelihood at Null Parameters 	-2592.70
	-----------------------------------------------------------------------------------------...
	Rho Squared w.r.t. Null Parameters	0.628
	=========================================================================================...



.. tip::

	If you want access to the data in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-use copy like this::

		m = larch.Model.Example(201)



