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

	m.utility.co("1",1,"ASC_TRAIN")
	m.utility.co("1",3,"ASC_CAR")
	m.utility.co("TRAIN_TT",1,"B_TIME")
	m.utility.co("SM_TT",2,"B_TIME")
	m.utility.co("CAR_TT",3,"B_TIME")
	m.utility.co("TRAIN_CO*(GA==0)",1,"B_COST")
	m.utility.co("SM_CO*(GA==0)",2,"B_COST")
	m.utility.co("CAR_CO",3,"B_COST")

We can estimate the models and check the results match up with those given by Biogeme:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike()
	>>> print(result.message)
	Optimization terminated successfully...
	
	>>> print(m.report('txt', sigfigs=4))
	=========================================================================================...
	swissmetro example 02 (simple logit weighted)
	=========================================================================================...
	Model Parameter Estimates
	-----------------------------------------------------------------------------------------...
	Parameter	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue   
	ASC_TRAIN	 0.0        	-0.7566     	 0.05604    	-13.5       	 0.0        
	ASC_CAR  	 0.0        	-0.1143     	 0.04315    	-2.65       	 0.0        
	B_TIME   	 0.0        	-0.01321    	 0.0005693  	-23.21      	 0.0        
	B_COST   	 0.0        	-0.0112     	 0.0005201  	-21.53      	 0.0        
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


