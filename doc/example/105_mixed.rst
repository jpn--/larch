.. currentmodule:: larch

============================================
105: Swissmetro Normal Mixed MNL Mode Choice
============================================

.. testsetup:: *

	import larch
	import larch.mixed

.. tip::

	Mixed logit models are under development.  The interface should not be considered stable
	and may change with future versions of Larch.  Use at your own risk.



This example is a mode choice model built using the Swissmetro example dataset.
First we create the DB and Model objects:

.. testcode::

	d = larch.DB.Example('SWISSMETRO')
	kernel = larch.Model(d)

We can attach a title to the model. The title does not affect the calculations
as all; it is merely used in various output report styles.

.. testcode::

	kernel.title = "swissmetro example 05 (normal mixture logit)"


The swissmetro dataset, as with all Biogeme data, is only in `co` format.  To add a mixed
parameter on time, we give the time parameter twice (once with \*1 added to give a distinctive name)
and add the plain (mean of the normal) and distributional (std dev of the normal).

.. testcode::

	kernel.utility.co("1",1,"ASC_TRAIN")
	kernel.utility.co("1",3,"ASC_CAR")
	kernel.utility.co("TRAIN_TT",1,"B_TIME")
	kernel.utility.co("SM_TT",2,"B_TIME")
	kernel.utility.co("CAR_TT",3,"B_TIME")
	kernel.utility.co("TRAIN_TT *1",1,"B_TIME_S")
	kernel.utility.co("SM_TT *1",2,"B_TIME_S")
	kernel.utility.co("CAR_TT *1",3,"B_TIME_S")
	kernel.utility.co("TRAIN_CO*(GA==0)",1,"B_COST")
	kernel.utility.co("SM_CO*(GA==0)",2,"B_COST")
	kernel.utility.co("CAR_CO",3,"B_COST")


From the kernel MNL model we create a normal mixed model.  We set the starting value for the
std dev to be nonzero to improve numerical stability:

.. testcode::

	m = larch.mixed.NormalMixedModel(kernel, ['B_TIME_S'], ndraws=100, seed=0)
	v = m.parameter_values()
	v[-1] = 0.01
	m.parameter_values(v)


We can estimate the models and check the results match up with those given by Biogeme:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> result = m.maximize_loglike()
	>>> m.loglike()
	-5213.34...


The reporting features of mixed logit models have not been developed yet.  A placeholder
simple report of the parameters is available for now:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> print(m)
	<larch.mixed.NormalMixedModel> Temporary Report
	============================================================
	ASC_TRAIN                	-0.396863
	ASC_CAR                  	 0.140273
	B_TIME                   	-0.0235885
	B_COST                   	-0.0128322
	Choleski_0               	 0.0160842
	============================================================

	

.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.Model.Example(105)

