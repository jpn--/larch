.. currentmodule:: larch

=================================
17t: MTC MNL Mode Choice Using DT
=================================

.. testsetup:: *

   import larch

For this example, we're going to re-create model 17 from the
`Self Instructing Manual <http://www.caee.utexas.edu/prof/Bhat/COURSES/LM_Draft_060131Final-060630.pdf>`_,
but using the :class:`DT` data format.

Unlike for the :class:`DB` based model, we won't need to manipulate the data
in advance of creating our model, because combinations of :ref:`idca`
and :ref:`idco` variables can be done on-the-fly using broadcasting techniques for
numpy arrays.


To build that model, we are going to have to create some variables that
we don't already have: cost divided by income, and out of vehicle travel time
divided by distance.  The tricky part is that cost and time are :ref:`idca`
variables, and income and distance are :ref:`idco` variables, in a different table.
Fortunately, we can use SQL to pull the data from one table to the other,
but first we'll set ourselves up to do so efficiently.

.. testcode::

	d = larch.DT.Example('MTC')

We don't need to do anything more that open the example DT file and we are ready to build our model.

.. testcode::

	m = larch.Model(d)

	m.utility.ca("totcost/hhinc",  "costbyincome")

	m.utility.ca("tottime * (altnum <= 4)", "motorized_time")
	m.utility.ca("tottime * (altnum >= 5)", "nonmotorized_time")
	m.utility.ca("ovtt/dist * (altnum <= 4)", "motorized_ovtbydist")

The totcost/hhinc data is computed once as a new variable when loading the model data.
The same for tottime filtered by motorized modes (we harness the convenient fact
that all the motorized modes have identifying numbers 4 or less), and ovtt/dist.

.. testcode::

	m.utility.co("hhinc",4)
	m.utility.co("hhinc",5)
	m.utility.co("hhinc",6)

Since the model we want to create groups together DA, SR2 and SR3+ jointly as
reference alternatives with respect to income, we can simply omit all of these alternatives
from the block that applies to **hhinc**.

For vehicles per worker, the preferred model include a joint parameter on SR2 and SR3+,
but not including DA and not fixed at zero.  Here we might use an alias, which allows
us to specify one or more parameters that are simply a fixed proportion of another parameter.
For example, we can say that vehbywrk_SR2 will be equal to 1.0 times vehbywrk_SR.

.. testcode::

	m.parameter("vehbywrk_SR")
	m.alias("vehbywrk_SR2","vehbywrk_SR",1.0)
	m.alias("vehbywrk_SR3+","vehbywrk_SR",1.0)

Having defined these parameter aliases, we can then loop over all alternatives (skipping DA
in the index-zero position) to add vehicles per worker to the utility function:

.. testcode::

	for a,name in m.db.alternatives()[1:]:
		m.utility.co("vehbywrk",a,"vehbywrk_"+name)

We can also run similar loops over workplace in CBD, etc:

.. testcode::

	for a,name in m.db.alternatives()[1:]:
		m.utility.co("wkccbd+wknccbd",a,"wkcbd_"+name)

	for a,name in m.db.alternatives()[1:]:
		m.utility.co("wkempden",a,"wkempden_"+name)

	for a,name in m.db.alternatives()[1:]:
		m.utility.co("1",a,"ASC_"+name)

	m.option.calc_std_errors = True


Having created this model, we can then estimate it:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike()
	>>> result.message
	'Optimization terminated successfully...

	>>> m.loglike()
	-3444.1...

	>>> print(m)
	====================================================================================================
	Model Parameter Estimates
	----------------------------------------------------------------------------------------------------
	Parameter          	InitValue   	FinalValue  	StdError    	t-Stat      	NullValue   
	costbyincome       	 0          	-0.0524213  	 0.0104042  	-5.03849    	 0          
	motorized_time     	 0          	-0.0201867  	 0.00381463 	-5.2919     	 0          
	nonmotorized_time  	 0          	-0.045446   	 0.00576857 	-7.87821    	 0          
	motorized_ovtbydist	 0          	-0.132869   	 0.0196429  	-6.76423    	 0          
	hhinc#4            	 0          	-0.00532375 	 0.00197713 	-2.69266    	 0          
	hhinc#5            	 0          	-0.00864285 	 0.00515439 	-1.67679    	 0          
	hhinc#6            	 0          	-0.00599738 	 0.00314859 	-1.90478    	 0          
	vehbywrk_SR        	 0          	-0.316638   	 0.0666331  	-4.75196    	 0          
	vehbywrk_Tran      	 0          	-0.946257   	 0.118293   	-7.99925    	 0          
	vehbywrk_Bike      	 0          	-0.702149   	 0.258287   	-2.71849    	 0          
	vehbywrk_Walk      	 0          	-0.72183    	 0.169392   	-4.26131    	 0          
	wkcbd_SR2          	 0          	 0.259828   	 0.123353   	 2.10638    	 0          
	wkcbd_SR3+         	 0          	 1.06926    	 0.191275   	 5.59018    	 0          
	wkcbd_Tran         	 0          	 1.30883    	 0.165697   	 7.89889    	 0          
	wkcbd_Bike         	 0          	 0.489274   	 0.361098   	 1.35496    	 0          
	wkcbd_Walk         	 0          	 0.101732   	 0.252107   	 0.403529   	 0          
	wkempden_SR2       	 0          	 0.00157763 	 0.000390357	 4.04152    	 0          
	wkempden_SR3+      	 0          	 0.00225683 	 0.000451972	 4.9933     	 0          
	wkempden_Tran      	 0          	 0.00313243 	 0.00036073 	 8.68358    	 0          
	wkempden_Bike      	 0          	 0.00192791 	 0.00121547 	 1.58614    	 0          
	wkempden_Walk      	 0          	 0.00289023 	 0.000742102	 3.89465    	 0          
	ASC_SR2            	 0          	-1.80782    	 0.106123   	-17.035     	 0          
	ASC_SR3+           	 0          	-3.43374    	 0.151864   	-22.6106    	 0          
	ASC_Tran           	 0          	-0.684817   	 0.247816   	-2.7634     	 0          
	ASC_Bike           	 0          	-1.62885    	 0.427399   	-3.81108    	 0          
	ASC_Walk           	 0          	 0.0682096  	 0.348001   	 0.196004   	 0          
	====================================================================================================
	Model Estimation Statistics
	----------------------------------------------------------------------------------------------------
	Log Likelihood at Convergence     	-3444.19
	Log Likelihood at Null Parameters 	-7309.60
	----------------------------------------------------------------------------------------------------
	Rho Squared w.r.t. Null Parameters	0.529
	====================================================================================================
	...


.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example('17t')

