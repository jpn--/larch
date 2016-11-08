.. currentmodule:: larch

========================================
102: Swissmetro Weighted MNL Mode Choice
========================================

.. testsetup:: *

   import larch



This example is a mode choice model built using the Swissmetro example dataset.
First we create the DB and Model objects.  When we create the DB object, we will
redefine the weight value:

.. testcode::

	d = larch.DB.Example('SWISSMETRO')
	d.queries.weight = "(1.0*(GROUPid==2)+1.2*(GROUPid==3))*0.8890991"
	m = larch.Model(d)

We can attach a title to the model. The title does not affect the calculations
as all; it is merely used in various output report styles.

.. testcode::

	m.title = "swissmetro example 02 (simple logit weighted)"


The swissmetro dataset, as with all Biogeme data, is only in `co` format.

.. testcode::

	from larch.roles import P,X
	m.utility[1] = ( P.ASC_TRAIN
	               + P.B_TIME * X.TRAIN_TT
	               + P.B_COST * X("TRAIN_CO*(GA==0)") )
	m.utility[2] = ( P.B_TIME * X.SM_TT
	               + P.B_COST * X("SM_CO*(GA==0)") )
	m.utility[3] = ( P.ASC_CAR
	               + P.B_TIME * X.CAR_TT
	               + P.B_COST * X("CAR_CO") )

Larch will find all the parameters in the model, but we'd like to output them in
a particular order, so we want to reorder the parameters.
We can use the reorder method to fix this:

.. testcode::

	m.reorder_parameters("ASC", "B_")

We can estimate the models and check the results match up with those given by Biogeme:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike()
	>>> print(result.message)
	Optimization terminated successfully...
	
	>>> print(m.report('txt', sigfigs=3))
	=========================================================================================...
	swissmetro example 02 (simple logit weighted)
	=========================================================================================...
	Model Parameter Estimates
	-----------------------------------------------------------------------------------------...
	Parameter	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue   
	ASC_TRAIN	 0.0        	-0.757      	 0.056      	-13.5       	 0.0
	ASC_CAR  	 0.0        	-0.114      	 0.0432     	-2.65       	 0.0
	B_TIME   	 0.0        	-0.0132     	 0.000569   	-23.2       	 0.0
	B_COST   	 0.0        	-0.0112     	 0.00052    	-21.5       	 0.0
	=========================================================================================...
	Model Estimation Statistics
	-----------------------------------------------------------------------------------------...
	Log Likelihood at Convergence     	-5273.74
	Log Likelihood at Null Parameters 	-7016.87
	-----------------------------------------------------------------------------------------...
	Rho Squared w.r.t. Null Parameters	0.248
	=========================================================================================...

	
.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(102)


