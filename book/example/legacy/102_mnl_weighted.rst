.. currentmodule:: larch

========================================
102: Swissmetro Weighted MNL Mode Choice
========================================

.. testsetup:: *

	import larch
	import os
	import pandas
	pandas.set_option("display.max_columns", 999)
	pandas.set_option('expand_frame_repr', False)
	pandas.set_option('precision', 3)
	larch._doctest_mode_ = True

This example is a mode choice model built using the Swissmetro example dataset.
First we create the DB and Model objects:

.. testcode::

	import larch.examples
	d = larch.examples.SWISSMETRO()
	m = larch.Model(dataservice=d)

We can attach a title to the model. The title does not affect the calculations
as all; it is merely used in various output report styles.

.. testcode::

	m.title = "swissmetro example 02 (weighted logit)"

We need to identify the availability and choice variables. These have been conveniently
set up in the data.

.. testcode::

	m.availability_var = 'avail'
	m.choice_ca_var = 'choice'

This model adds a weighting factor.

.. testcode::

	m.weight_co_var = "1.0*(GROUP==2)+1.2*(GROUP==3)"

The swissmetro dataset, as with all Biogeme data, is only in `co` format.

.. testcode::

	from larch.roles import P,X
	m.utility_co[1] = P("ASC_TRAIN")
	m.utility_co[2] = 0
	m.utility_co[3] = P("ASC_CAR")
	m.utility_co[1] += X("TRAIN_TT") * P("B_TIME")
	m.utility_co[2] += X("SM_TT") * P("B_TIME")
	m.utility_co[3] += X("CAR_TT") * P("B_TIME")
	m.utility_co[1] += X("TRAIN_CO*(GA==0)") * P("B_COST")
	m.utility_co[2] += X("SM_CO*(GA==0)") * P("B_COST")
	m.utility_co[3] += X("CAR_CO") * P("B_COST")

Larch will find all the parameters in the model, but we'd like to output them in
a rational order.  We can use the ordering method to do this:

.. testcode::

	m.ordering = [
		("ASCs", 'ASC.*',),
		("LOS", 'B_.*',),
	]

The swissmetro example models exclude some observations.  We will use the selector
to identify the observations we would like to keep.  There are two selector criteria,
and only cases that evaluate `True` for both are selected.

.. testcode::

	m.dataservice.selector = ["PURPOSE in (1,3)", "CHOICE != 0"]

We can estimate the models and check the results match up with those given by Biogeme:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE, +REPORT_NDIFF

	>>> m.load_data()
	>>> m.maximize_loglike(method='SLSQP')
	┣ ...Optimization terminated successfully...
	>>> m.loglike()
	-5931.557...
	>>> m.calculate_parameter_covariance()
	>>> m.dataframes.weight_normalization
	1.124734...

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE, +REPORT_NDIFF

	>>> m.parameter_summary('xml')
	Category  Parameter     Value   Std Err  t Stat  Null Value
	    ASCs    ASC_CAR   -0.1143    0.0407   -2.81         0.0
	          ASC_TRAIN   -0.7565    0.0528  -14.32         0.0
	     LOS     B_COST   -0.0112  0.000490  -22.83         0.0
	             B_TIME  -0.01322  0.000537  -24.62         0.0


.. tip::

	If you want access to the model in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy like this::

		m = larch.example(102)
