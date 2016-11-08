.. currentmodule:: larch

========================================
104: Swissmetro MNL with Modified Data
========================================

.. testsetup:: *

   import larch



This example is a mode choice model built using the Swissmetro example dataset.
First we create the DB and Model objects:

.. testcode::

	d = larch.DB.Example('SWISSMETRO')
	m = larch.Model(d)

We can attach a title to the model. The title does not affect the calculations
as all; it is merely used in various output report styles.

.. testcode::

	m.title = "swissmetro example 04 (modified data)"


The swissmetro dataset, as with all Biogeme data, is only in `co` format. To be consistent
with the Biogeme example, we divide travel time by 100.0.

.. note::
	We use 100.0 instead of 100 to ensure that the division is done with real (floating point) numbers
	and not integers. Larch internally uses integers only for nominal & ordinal values
	(i.e. identifying names and positions in arrays) and doesn't use integer values to
	represent cardinal data values, although SQLite sometimes does.  Math completed inside
	the SQLite kernel, such as that contained within a string used as a data item, can
	be impacted by this.

.. testcode::

	from larch.roles import P,X
	m.utility.co[1] = X("1") * P("ASC_TRAIN")
	m.utility.co[2] = 0
	m.utility.co[3] = X("1") * P("ASC_CAR")
	m.utility.co[1] += X("TRAIN_TT/100.0") * P("B_TIME")
	m.utility.co[2] += X("SM_TT/100.0") * P("B_TIME")
	m.utility.co[3] += X("CAR_TT/100.0") * P("B_TIME")


For this model, we will use the natural log of (cost/100.0), instead of cost.
But when cost is zero, this would give an error.  So, use the a "CASE ... WHEN ... THEN ... ELSE ... END"
construct from SQL to give us a non-error value (here, we set it to 0) when cost is
zero.

.. testcode::

	m.utility.co[1] += X("CASE TRAIN_CO*(GA==0) WHEN 0 THEN 0 ELSE LOG((TRAIN_CO/100.0)*(GA==0)) END") * P("B_LOGCOST")
	m.utility.co[2] += X("CASE SM_CO*(GA==0) WHEN 0 THEN 0 ELSE LOG((SM_CO/100.0)*(GA==0)) END") * P("B_LOGCOST")
	m.utility.co[3] += X("CASE CAR_CO WHEN 0 THEN 0 ELSE LOG(CAR_CO/100.0) END") * P("B_LOGCOST")

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
	swissmetro example 04 (modified data)
	=========================================================================================...
	Model Parameter Estimates
	-----------------------------------------------------------------------------------------...
	Parameter	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue   
	ASC_TRAIN	 0.0        	-0.851      	 0.056      	-15.2       	 0.0
	ASC_CAR  	 0.0        	-0.274      	 0.0457     	-6.0        	 0.0
	B_TIME   	 0.0        	-1.07       	 0.0547     	-19.6       	 0.0
	B_LOGCOST	 0.0        	-1.04       	 0.0595     	-17.4       	 0.0
	=========================================================================================...
	Model Estimation Statistics
	-----------------------------------------------------------------------------------------...
	Log Likelihood at Convergence     	-5423.30
	Log Likelihood at Null Parameters 	-6964.66
	-----------------------------------------------------------------------------------------...
	Rho Squared w.r.t. Null Parameters	0.221
	=========================================================================================...

	
.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(104)


