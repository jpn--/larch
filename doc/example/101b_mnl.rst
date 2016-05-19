.. currentmodule:: larch

===================================
101b: Swissmetro MNL, Biogeme Style
===================================

.. testsetup:: *

   import larch



This example is a mode choice model built using the Swissmetro example dataset.
We will use a style for writing the utility functions that is similar to the
style used in Biogeme.
First we create the DB and Model objects, as usual:

.. testcode::

	d = larch.DB.Example('SWISSMETRO')
	m = larch.Model(d)

We can attach a title to the model. The title does not affect the calculations
as all; it is merely used in various output report styles.

.. testcode::

	m.title = "swissmetro example 01b (simple logit)"


The swissmetro dataset, as with all Biogeme data, is only in `co` format.  Which is great,
because it lets us ignore the `ca` format and just write out the utility functions directly.

.. testcode::

	from larch.roles import P,X  # Parameters, Data
	m.utility[1] = ( P.ASC_TRAIN
	               + P.Time * X.TRAIN_TT
				   + P.Cost * X("TRAIN_CO*(GA==0)") )
	m.utility[2] = ( P.Time * X.SM_TT
	               + P.Cost * X("SM_CO*(GA==0)") )
	m.utility[3] = ( P.ASC_CAR
	               + P.Time * X.CAR_TT
				   + P.Cost * X("CAR_CO") )

Note that when the data field is too complex to be expressed as a single python
identifier, we can write it as a quoted string instead.

We can estimate the models and check the results match up with those given by Biogeme:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> m.estimate()
	<larch.core.runstats, success ...
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
	swissmetro example 01b (simple logit)
	=========================================================================================...
	Model Parameter Estimates
	-----------------------------------------------------------------------------------------...
	Parameter	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue
	ASC_TRAIN	 0.0        	-0.701      	 0.0549     	-12.8       	 0.0
	Time     	 0.0        	-0.0128     	 0.000569   	-22.5       	 0.0        
	Cost     	 0.0        	-0.0108     	 0.000518   	-20.9       	 0.0        
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

		m = larch.Model.Example('101b')

