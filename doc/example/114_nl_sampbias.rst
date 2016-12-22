.. currentmodule:: larch

=========================================
114: Swissmetro Nested with Sampling Bias
=========================================

.. testsetup:: *

   import larch



This example is a mode choice model built using the Swissmetro example dataset.
First we create the DB and Model objects.  When we create the DB object, we will
redefine the weight value:

.. testcode::

	d = larch.DB.Example('SWISSMETRO')
	m = larch.Model(d)

We can attach a title to the model. The title does not affect the calculations
as all; it is merely used in various output report styles.

.. testcode::

	m.title = "swissmetro example 14 (nested logit with sampling bias)"


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


To create a new nest, we can use the new_nest command, although we'll need to know what the
alternative codes are for the alternatives in our dataset. To find out, we can do:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> m.df.alternatives()
	[(1, 'Train'), (2, 'SM'), (3, 'Car')]


For this example, we want to nest together the Train and Car modes into a "existing" modes nest.
It looks like those are modes 1 and 3, so we can use the new_nest command like this:

.. testcode::

	m.new_nest("existing", parent=m.root_id, children=[1,3])


We're also going to insert a sampling bias correction in this model.  To read more on this topic,
please see
Bierlaire, Bolduc, and McFadden (2008)
`The estimation of generalized extreme value models from choice-based samples <http://www.sciencedirect.com/science/article/pii/S0191261507000963>`_,
in `Transportation Research Part B: Methodological <http://www.sciencedirect.com/science/journal/01912615>`_.


.. testcode::

	m.samplingbias[1] = P.SB_TRAIN


Larch will find all the parameters in the model, but we'd like to output them in
a particular order, so we want to reorder the parameters.
We can use the reorder method to fix this:

.. testcode::

	m.reorder_parameters("ASC", "B_", "existing",)

We can estimate the models and check the results match up with those given by Biogeme:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike()
	>>> print(result.message)
	Optimization terminated successfully...
	
	>>> print(m.report('txt', sigfigs=3))
	=========================================================================================...
	swissmetro example 14 (nested logit with sampling bias)
	=========================================================================================...
	Model Parameter Estimates
	-----------------------------------------------------------------------------------------...
	Parameter	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue
	ASC_TRAIN	 0.0        	-10.2       	 3.13       	-3.27       	 0.0        
	ASC_CAR  	 0.0        	-0.317      	 0.0443     	-7.16       	 0.0        
	B_TIME   	 0.0        	-0.0113     	 0.000576   	-19.6       	 0.0        
	B_COST   	 0.0        	-0.0111     	 0.000511   	-21.6       	 0.0        
	SB_TRAIN 	 0.0        	 10.4       	 3.15       	 3.31       	 0.0        
	existing 	 1.0        	 0.874      	 0.035      	-3.61       	 1.0        
	=========================================================================================...
	Model Estimation Statistics
	-----------------------------------------------------------------------------------------...
	Log Likelihood at Convergence     	-5169.64
	Log Likelihood at Null Parameters 	-6964.66
	-----------------------------------------------------------------------------------------...
	Rho Squared w.r.t. Null Parameters	0.258
	=========================================================================================...

	
.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(114)


