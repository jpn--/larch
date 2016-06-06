.. currentmodule:: larch

=======================================
101s: Swissmetro MNL, Stacked Variables
=======================================

.. testsetup:: *

   import larch



This example is a mode choice model built using the Swissmetro example dataset.
We will use a style for writing the utility functions that uses stacked variables,
which is a feature of the DT data format.
First we create the DT and Model objects, as usual:

.. testcode::

	d = larch.DT.Example('SWISSMETRO')
	m = larch.Model(d)

We can attach a title to the model. The title does not affect the calculations
as all; it is merely used in various output report styles.

.. testcode::

	m.title = "swissmetro example 01s (simple logit)"

Unlike Biogeme, the usual way of using Larch does not fill the main namespace
with all the parameters and data column references as distinct objects.  Instead, we
can use two master classes to fill those roles.

.. testcode::

	from larch.roles import P,X  # Parameters, Data

All of our parameter references can be written as instances of the P (:class:`roles.ParameterRef`) class,
and all of our data column references can be written as instances of the X (:class:`roles.DataRef`) class.

The swissmetro dataset, as with all Biogeme data, is only in `co` format.  But,
in our model most of the attributes are "generic", i.e. stuff like travel time, which
varies across alternatives, but for which we'll want to assign the same parameter
to for each alternative (so that a minute of travel time has the same value no
matter which alternative it is on).  So, here we will create the generic `ca` format
variables by stacking the relevant `co` variables.

.. testcode::

	d.stack_idco('traveltime', {1: X.TRAIN_TT, 2: X.SM_TT, 3: X.CAR_TT})
	d.stack_idco('cost', {1: X("TRAIN_CO*(GA==0)"), 2: X("SM_CO*(GA==0)"), 3: X("CAR_CO")})

Then we can use these stacked variables in the utility.ca function:

.. testcode::

	m.utility.ca = X.traveltime * P.Time + X.cost * P.Cost

Not all our variables are stack-able; the alternative specific constants are not because
we want to assign different parameters for each alternative.  So those we'll leave in
:ref:`idco` format (implicitly):

.. testcode::

	m.utility[1] = P.ASC_TRAIN
	m.utility[2] = 0
	m.utility[3] = P.ASC_CAR


We can estimate the models and check the results match up with those given by Biogeme:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike()
	>>> print(result.message)
	Optimization terminated successfully...
	>>> m.loglike()
	-5331.252...
	>>> m['Time'].value
	-0.01277...
	>>> m['Cost'].value
	-0.01083...
	>>> m['ASC_TRAIN'].value
	-0.7012...
	>>> m['ASC_CAR'].value
	-0.1546...



.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> print(m.report('txt', sigfigs=3))
	=========================================================================================...
	swissmetro example 01s (simple logit)
	=========================================================================================...
	Model Parameter Estimates
	-----------------------------------------------------------------------------------------...
	Parameter	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue
	Time     	 0.0        	-0.0128     	 0.000569   	-22.5       	 0.0        
	Cost     	 0.0        	-0.0108     	 0.000518   	-20.9       	 0.0        
	ASC_TRAIN	 0.0        	-0.701      	 0.0549     	-12.8       	 0.0
	ASC_CAR  	 0.0        	-0.155      	 0.0432     	-3.58       	 0.0        
	=========================================================================================...
	Model Estimation Statistics
	-----------------------------------------------------------------------------------------...
	Log Likelihood at Convergence     	-5331.25
	Log Likelihood at Null Parameters 	-6964.66
	-----------------------------------------------------------------------------------------...
	Rho Squared w.r.t. Null Parameters	0.235
	=========================================================================================...

	

.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example('101s')

