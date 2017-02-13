.. currentmodule:: larch

=====================================
202: Exampville Mode Choice Logsums
=====================================

.. testsetup:: *

	import larch
	import os


Welcome to Exampville, the best simulated town in this here part of the internet!

In our previous example, we calculated mode choice model logsums.  Now we'll use those
here as data in a destination choice model.  We'll begin by preparing the data.

Data Prep
~~~~~~~~~

.. testcode::

	import larch, numpy, pandas, os
	from larch.roles import P,X

	d, nZones, omx = larch.examples.reproduce(202, ['d', 'nZones', 'omx'])

	d.set_alternatives( numpy.arange(1,nZones+1) )

	d.choice_idco = {
		taz: 'DTAZ=={}'.format(taz) for taz in numpy.arange(1,nZones+1)
	}

The choice array is created as a stack of arrays from the idco data.
But it may be advantageous to create the array explicitly once; it
is very sparse so it compresses well and it can be faster to run the model
if it's already computed, especially if there are a lot of zones.

.. testcode::

	ch = numpy.zeros([d.nAllCases(), nZones ], dtype=numpy.float32)
	dtaz = d.idco.DTAZ[:]
	ch[range(dtaz.shape[0]), dtaz-1] = 1
	d.new_idca_from_array('_choice_', ch, overwrite=True)

We're also gonna overwrite the _avail_ array, so that
all destinations are available for all tours.

.. testcode::

	d.new_idca_from_array('_avail_', numpy.ones([d.nAllCases(), nZones ], dtype=bool), overwrite=True)

Also, the data file contains all the skim values we plucked from the skims for
the particular destinations, but since now we'll be considering all destinations
we don't want that data anymore.

.. testcode::

	for i in omx.data:
		d.delete_data(i.name)
	for i in omx.lookup:
		d.delete_data(i.name)

We will replace it with entire rows from the OMX representing all destinations,

which we will attach as an external data souce. Plus, we need the lookups which are OTAZ-generic

.. testcode::

	d.idca.add_external_omx(omx, rowindexnode=d.idco.HOMETAZi, n_alts=nZones)

Destination Choice Model
~~~~~~~~~~~~~~~~~~~~~~~~

Now we are ready to create the destination choice model.

.. testcode::

	m = larch.Model(d)

	m.title = "Exampville Work Tour Destination Choice"

	from larch.util.piecewise import log_and_linear_function
	dist_func = log_and_linear_function('DIST', baseparam='Distance')

	m.utility.ca = (
		+ P.ModeChoiceLogSum * X.MODECHOICELOGSUM
		+ dist_func
	)

	m.quantity = (
		+ P("EmpRetail_HighInc") * X('EMP_RETAIL * (INCOME>50000)')
		+ P("EmpNonRetail_HighInc") * X('EMP_NONRETAIL') * X("INCOME>50000")
		+ P("EmpRetail_LowInc") * X('EMP_RETAIL') * X("INCOME<=50000")
		+ P("EmpNonRetail_LowInc") * X('EMP_NONRETAIL') * X("INCOME<=50000")

	)

	m.quantity_scale = P.Theta
	m.parameter.Theta(value=0.5, min=0.001, max=1.0, null_value=1.0)

	m.parameter.EmpRetail_HighInc(holdfast=1, value=0)
	m.parameter.EmpRetail_LowInc(holdfast=1, value=0)



.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE


	>>> m.maximize_loglike()
	messages: Optimization terminated successfully...

	>>> print(m.report('txt', sigfigs=3))
	======================================================================================================...
	Exampville Work Tour Destination Choice
	======================================================================================================...
	Model Parameter Estimates
	------------------------------------------------------------------------------------------------------...
	Parameter           	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue   
	ModeChoiceLogSum    	 0.0        	 1.05       	 0.0819     	 12.8       	 0.0        
	Distance            	 0.0        	-0.000936   	 0.025      	-0.0375     	 0.0        
	logDistanceP1       	 0.0        	-0.0521     	 0.116      	-0.447      	 0.0        
	Theta               	 1.0        	 0.812      	 0.0434     	-4.34       	 1.0        
	EmpRetail_HighInc   	 0.0        	 0.0        	 0.0        	 nan        	 0.0        	H
	EmpRetail_LowInc    	 0.0        	 0.0        	 0.0        	 nan        	 0.0        	H
	EmpNonRetail_HighInc	 0.0        	 0.464      	 0.218      	 2.13       	 0.0        
	EmpNonRetail_LowInc 	 0.0        	-0.818      	 0.305      	-2.68       	 0.0        
	------------------------------------------------------------------------------------------------------...
	H	Parameters held fixed at their initial values (not estimated)
	======================================================================================================...
	Model Estimation Statistics
	------------------------------------------------------------------------------------------------------...
	Log Likelihood at Convergence     	-3670.60
	Log Likelihood at Null Parameters 	-4493.08
	------------------------------------------------------------------------------------------------------...
	Rho Squared w.r.t. Null Parameters	0.183
	======================================================================================================...




