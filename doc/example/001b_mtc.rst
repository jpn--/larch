.. currentmodule:: larch

===========================
1b: Using Loops
===========================

.. testsetup:: *

   import larch



This example is a mode choice model built using the MTC example dataset.
In fact, we are going to build the exact same model as Example 1, just a
bit more automagically. First we create the DB and Model objects:

.. testcode::

	d = larch.DB.Example('MTC')
	m = larch.Model(d)

Then, we will extract the set of alternatives from the data:

.. testcode::

	alts = d.alternatives()
	print(alts)

Which gives us

.. testoutput::

	[(1, 'DA'), (2, 'SR2'), (3, 'SR3+'), (4, 'Tran'), (5, 'Bike'), (6, 'Walk')]

You'll see that we have a list of 2-tuples, with each containing a code number and
a name.  We'll denote the code number (the first part of the tuple, indexed as zero)
of the first alternative as our reference alterative.

.. testcode::

	ref_id = alts[0][0]

To create the alternative specific constants, we loop over alternatives.

.. testcode::

	for code, name in alts:
		if code != ref_id:
			m.utility.co("1",code,"ASC_"+name)

To create the alternative specific parameters on income, we loop over alternatives
again. (We could also do this inside the same loop with the ASCs but then the
parameters would appear interleaved in the output, which we don't want.)

.. testcode::

	for code, name in alts:
		if code != ref_id:
			m.utility.co("hhinc",code)



The other two parameters are generic, so we don't need to do anything different from
the original example.

.. testcode::

	m.utility.ca("tottime")
	m.utility.ca("totcost")

Let's see what we get:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> m.option.calc_std_errors = True
	>>> m.estimate()
	<larch.core.runstats, success ...
	>>> m.loglike()
	-3626.18...

	>>> print(m)
	============================================================================================
	Model Parameter Estimates
	--------------------------------------------------------------------------------------------
	Parameter	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue   
	ASC_SR2  	 0          	-2.17804    	 0.104638   	-20.815     	 0          
	ASC_SR3+ 	 0          	-3.72513    	 0.177692   	-20.964     	 0
	ASC_Tran 	 0          	-0.670973   	 0.132591   	-5.06047    	 0
	ASC_Bike 	 0          	-2.37634    	 0.304502   	-7.80403    	 0
	ASC_Walk 	 0          	-0.206814   	 0.1941     	-1.0655     	 0
	hhinc#2  	 0          	-0.00217    	 0.00155329 	-1.39704    	 0          
	hhinc#3  	 0          	 0.000357656	 0.00253773 	 0.140935   	 0          
	hhinc#4  	 0          	-0.00528648 	 0.00182882 	-2.89064    	 0          
	hhinc#5  	 0          	-0.0128081  	 0.00532408 	-2.40568    	 0          
	hhinc#6  	 0          	-0.00968626 	 0.00303305 	-3.19358    	 0          
	tottime  	 0          	-0.0513404  	 0.00309941 	-16.5646    	 0          
	totcost  	 0          	-0.00492036 	 0.000238894	-20.5964    	 0          
	============================================================================================
	Model Estimation Statistics
	--------------------------------------------------------------------------------------------
	Log Likelihood at Convergence     	-3626.19
	Log Likelihood at Null Parameters 	-7309.60
	--------------------------------------------------------------------------------------------
	Rho Squared w.r.t. Null Parameters	0.504
	============================================================================================
	...


Exactly the same as before.  Awesome!

.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example('1b')

