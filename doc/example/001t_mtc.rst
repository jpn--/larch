.. currentmodule:: larch

==========================
1t: Troubleshooting
==========================

.. testsetup:: *

   import larch



In this example we are going to make some trouble, then do a little troubleshooting.
We'll start with the MTC example dataset, but we're going to impose a restriction that
nobody will want to bike more than 7 miles.

.. testcode::

	d = larch.DT.Example('MTC')
	# Bike (code 5) requires distance less than 7 to be available.
	d.idca._avail_[:,d._alternative_slot(5)] &= (d.idco.dist[:]<7)[:,None]


Then we can build up model as in the normal example.

.. testcode::

	m = larch.Model(d)
	from larch.roles import P, X, PX
	m.utility.co[2] = P("ASC_SR2")  + P("hhinc#2") * X("hhinc")
	m.utility.co[3] = P("ASC_SR3P") + P("hhinc#3") * X("hhinc")
	m.utility.co[4] = P("ASC_TRAN") + P("hhinc#4") * X("hhinc")
	m.utility.co[5] = P("ASC_BIKE") + P("hhinc#5") * X("hhinc")
	m.utility.co[6] = P("ASC_WALK") + P("hhinc#6") * X("hhinc")
	m.utility.ca = PX("tottime") + PX("totcost")
	m.option.calc_std_errors = True
	m.title = "MTC Example 1 (Troubleshooting)"


Having created this model, we can then estimate it.
(Since we happen to know we're
going to have problems, we can catch them and make sure we get the
right kind of problems.)

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> import warnings
	>>> with warnings.catch_warnings(record=True) as w:
	...		result = m.maximize_loglike()
	...		assert(len(w) == 1)
	...		assert(issubclass(w[-1].category, UserWarning))
	...		assert("did not succeed normally" in str(w[-1].message))
	>>> result.success
	False
	>>> result.message
	'convergence tolerance 0.00116'

Uh oh, something is wrong.  Let's find out what.

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> with warnings.catch_warnings(record=True) as w:
	...		m.doctor()
	'Model has 3 cases where the chosen alternative is unavailable'

The doctor method is not a comprehensive check up for every problem, but it can
catch a few of the obvious ones.  For instance, we've tried to estimate a model
where the chosen alternative isn't available for some of the cases, which is causing
trouble.  Larch doesn't pre-emptively check for this kind of problem, because it's
an unnecessary set of computations to run automatically every time you estimate a model,
assuming it's set up correctly.

For this particular problem, the doctor can actually fix the issue.  Setting `clash` equal to
'+' will make the troublesome alternatives available when they are actually chosen.  Or setting
`clash` to '-' will make them not chosen.

	>>> m.doctor(clash='+')
	'ok'

Now the problem is solved, and we can try estimating the model again:


.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike()
	>>> result.success
	True
	>>> result.message
	'Optimization terminated successfully per computed tolerance. [bhhh]'
	>>> m.reorder_parameters("ASC", "hhinc")
	>>> print(m.report('txt', sigfigs=3))
	============================================================================================
	MTC Example 1 (Troubleshooting)
	============================================================================================
	Model Parameter Estimates
	--------------------------------------------------------------------------------------------
	Parameter	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue   
	ASC_SR2  	 0.0        	-2.19       	 0.105      	-20.9       	 0.0        
	ASC_SR3P 	 0.0        	-3.74       	 0.178      	-21.0       	 0.0        
	ASC_TRAN 	 0.0        	-0.703      	 0.133      	-5.28       	 0.0        
	ASC_BIKE 	 0.0        	-2.23       	 0.303      	-7.38       	 0.0        
	ASC_WALK 	 0.0        	-0.248      	 0.194      	-1.28       	 0.0        
	hhinc#2  	 0.0        	-0.00219    	 0.00155    	-1.41       	 0.0        
	hhinc#3  	 0.0        	 0.000342   	 0.00254    	 0.135      	 0.0        
	hhinc#4  	 0.0        	-0.00533    	 0.00183    	-2.91       	 0.0        
	hhinc#5  	 0.0        	-0.0123     	 0.00528    	-2.34       	 0.0        
	hhinc#6  	 0.0        	-0.00969    	 0.00303    	-3.2        	 0.0        
	tottime  	 0.0        	-0.0498     	 0.0031     	-16.1       	 0.0        
	totcost  	 0.0        	-0.00495    	 0.00024    	-20.6       	 0.0        
	============================================================================================
	Model Estimation Statistics
	--------------------------------------------------------------------------------------------
	Log Likelihood at Convergence     	-3617.25
	Log Likelihood at Null Parameters 	-7193.34
	--------------------------------------------------------------------------------------------
	Rho Squared w.r.t. Null Parameters	0.497
	============================================================================================
	...


