.. currentmodule:: larch

===============================================
307: Itinerary Choice with Sampling Bias
===============================================

.. testsetup:: *

   import larch


We begin this example by picking up where we left off on :doc:`Example 303 <303_arc_nl2>`:

.. testcode::

	m = larch.Model.Example(303)

That model include a two level nested logit structure.  Let's suppose however,
that our source data does not quite look like a simple random sample of travelers.
Instead, suppose we have a stratified sample based on distribution channel (how travelers
purchased their tickets).  This stratification will manifest itself in different levels of
sampling intensity across carriers due to differences in marketing practices across those carriers.

Under an MNL model formulation, because we are including a complete set of carrier-specific
constants in the model, any bias in parameter estimates relating to the stratified sampling would
be isolated in the carrier-specific constants, allowing the other parameter estimates to
be unbiased by the sampling.  However, this does not hold for more complex GEV models unless
carrier-specific constants are processed outside of any GEV or nesting structure.

To do this, we can add a second set of carrier-specific constants in a special samplingbias term:

.. testcode::

	from larch.roles import P,X
	m.samplingbias.ca = sum(P("carrier_bias_{}".format(i))*X("carrier={}".format(i)) for i in [2,3,4,5])

Just like the "normal" carrier-specific constants, we hold out a carrier (#1) as the reference point, and
only add constants for 2 through 5.

We can now estimate the likelihood maximizing parameters:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike('SLSQP')

	>>> print(m.report('txt', sigfigs=3))
	============================================================================================...
	Model Parameter Estimates
	--------------------------------------------------------------------------------------------...
	Parameter     	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue
	timeperiod=2  	 0.0        	 0.0901     	 0.00775    	 11.6       	 0.0        
	timeperiod=3  	 0.0        	 0.104      	 0.00779    	 13.4       	 0.0        
	timeperiod=4  	 0.0        	 0.0887     	 0.00853    	 10.4       	 0.0        
	timeperiod=5  	 0.0        	 0.127      	 0.00869    	 14.6       	 0.0        
	timeperiod=6  	 0.0        	 0.229      	 0.00878    	 26.1       	 0.0        
	timeperiod=7  	 0.0        	 0.269      	 0.00986    	 27.3       	 0.0        
	timeperiod=8  	 0.0        	 0.293      	 0.0103     	 28.3       	 0.0        
	timeperiod=9  	 0.0        	-0.0223     	 0.011      	-2.04       	 0.0        
	carrier=2     	 0.0        	-0.82       	 0.113      	-7.26       	 0.0        
	carrier=3     	 0.0        	-2.04       	 0.153      	-13.3       	 0.0        
	carrier=4     	 0.0        	-1.7        	 0.231      	-7.35       	 0.0        
	carrier=5     	 0.0        	-6.84       	 8.48       	-0.807      	 0.0        
	equipment=2   	 0.0        	 0.387      	 0.0086     	 45.0       	 0.0        
	fare_hy       	 0.0        	-0.000948   	 2.52e-05   	-37.6       	 0.0        
	fare_ly       	 0.0        	-0.00104    	 7.02e-05   	-14.8       	 0.0        
	elapsed_time  	 0.0        	-0.00479    	 0.00011    	-43.4       	 0.0        
	nb_cnxs       	 0.0        	-2.45       	 0.0458     	-53.5       	 0.0        
	mu_tod        	 1.0        	 0.822      	 0.0128     	-13.9       	 1.0        
	mu_los        	 1.0        	 0.806      	 0.00911    	-21.3       	 1.0        
	carrier_bias_2	 0.0        	 1.13       	 0.138      	 8.23       	 0.0        
	carrier_bias_3	 0.0        	 3.16       	 0.169      	 18.6       	 0.0        
	carrier_bias_4	 0.0        	 2.64       	 0.279      	 9.44       	 0.0        
	carrier_bias_5	 0.0        	 7.86       	 10.5       	 0.747      	 0.0        
	============================================================================================...
	Model Estimation Statistics
	--------------------------------------------------------------------------------------------...
	Log Likelihood at Convergence     	-777321.75
	Log Likelihood at Null Parameters 	-953940.44
	--------------------------------------------------------------------------------------------...
	Rho Squared w.r.t. Null Parameters	0.185
	============================================================================================...

.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(307)

