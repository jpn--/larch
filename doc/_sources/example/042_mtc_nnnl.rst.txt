.. currentmodule:: larch

======================================================
42: MTC Non-Normalized Nested Mode Choice
======================================================

.. testsetup:: *

	import larch
	import larch.examples
	import pandas
	pandas.set_option("display.max_columns", 999)
	pandas.set_option('expand_frame_repr', False)
	pandas.set_option('precision', 3)
	larch._doctest_mode_ = True

For this example, we're going to create a non-normalized version of model 22 from the
`Self Instructing Manual <http://www.caee.utexas.edu/prof/Bhat/COURSES/LM_Draft_060131Final-060630.pdf>`.

This model is a nested logit model using the same utility function as model 17, so let's
start with that model and just add the NNNL structure.

.. testcode::

	umnl = larch.example(17)

We will create separate nests for the motorized and non-motorized alternatives.  Unlike for model 22,
we must give these two nests the same logsum parameter in order to achive consistent estimates with the NNNL model.

.. testcode::

	motorized = umnl.graph.new_node(parameter='mu', children=[1,2,3,4], name='Motorized')
	nonmotorized = umnl.graph.new_node(parameter='mu', children=[5,6], name='Nonmotorized')

Now we have a regular "correct" NL model.  But suppose for some reason we want to have a
NNNL model.  (Contrary to what you may read elsewhere, there are sometimes legitimate computational reasons
to choose such a model form, some of which will one day will be explained in this
documentation.)

.. testcode::

	import larch.model.nnnl
	nnnl = larch.NNNL(umnl)

This creates a new non-normalized nested logit model.

Having created this model, we can load it up and confirm it looks just like the UMNL equivalent:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> nnnl.finalize()
	>>> nnnl.load_data()
	>>> nnnl.loglike()
	-7309.60...

And we can estimation the model as usual:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> nnnl.maximize_loglike()
	┣ ...Optimization terminated successfully...
	>>> nnnl.loglike()
	-3441.69...

	>>> print(nnnl.pfo()[['value','initvalue','nullvalue','minimum','maximum','holdfast']])
	                               value  initvalue  nullvalue  minimum  maximum  holdfast
	Category  Parameter
	LOS       costbyincome        -0.053        0.0        0.0     -inf      inf         0
	          motorized_time      -0.020        0.0        0.0     -inf      inf         0
	          nonmotorized_time   -0.062        0.0        0.0     -inf      inf         0
	          motorized_ovtbydist -0.156        0.0        0.0     -inf      inf         0
	Zonal     wkcbd_BIKE           0.538        0.0        0.0     -inf      inf         0
	          wkcbd_SR2            0.266        0.0        0.0     -inf      inf         0
	          wkcbd_SR3            1.076        0.0        0.0     -inf      inf         0
	          wkcbd_TRANSIT        1.272        0.0        0.0     -inf      inf         0
	          wkcbd_WALK           0.151        0.0        0.0     -inf      inf         0
	          wkempden_BIKE        0.002        0.0        0.0     -inf      inf         0
	          wkempden_SR2         0.002        0.0        0.0     -inf      inf         0
	          wkempden_SR3         0.002        0.0        0.0     -inf      inf         0
	          wkempden_TRANSIT     0.003        0.0        0.0     -inf      inf         0
	          wkempden_WALK        0.003        0.0        0.0     -inf      inf         0
	Household hhinc#4             -0.005        0.0        0.0     -inf      inf         0
	          hhinc#5             -0.014        0.0        0.0     -inf      inf         0
	          hhinc#6             -0.008        0.0        0.0     -inf      inf         0
	          vehbywrk_BIKE       -0.984        0.0        0.0     -inf      inf         0
	          vehbywrk_SR         -0.311        0.0        0.0     -inf      inf         0
	          vehbywrk_TRANSIT    -0.973        0.0        0.0     -inf      inf         0
	          vehbywrk_WALK       -1.024        0.0        0.0     -inf      inf         0
	ASCs      ASC_BIKE            -1.618        0.0        0.0     -inf      inf         0
	          ASC_SR2             -1.825        0.0        0.0     -inf      inf         0
	          ASC_SR3             -3.452        0.0        0.0     -inf      inf         0
	          ASC_TRANSIT         -0.563        0.0        0.0     -inf      inf         0
	          ASC_WALK             0.437        0.0        0.0     -inf      inf         0
	Other     mu                   0.742        1.0        1.0    0.001    1.000         0

To contrast these results, let's also estimate the parameters on the utility maximizing nested
logit:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> umnl.load_data()
	>>> umnl.maximize_loglike()
	┣ ...Optimization terminated successfully...
	>>> umnl.loglike()
	-3441.69...

Interesting... the log likehood at convergence is the same.  Actually, it turns out the
entire model is the same, up to the scale of the parameters, which differs by exactly the
value of the logsum parameter.

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> joint = pandas.concat([nnnl.pf.value, umnl.pf.value], axis=1, keys=['nnnl','umnl'])
	>>> joint['ratio'] = joint['umnl']/joint['nnnl']
	>>> print(joint.sort_index())
	                      nnnl   umnl  ratio
	ASC_BIKE            -1.618 -1.201  0.742
	ASC_SR2             -1.825 -1.355  0.742
	ASC_SR3             -3.452 -2.562  0.742
	ASC_TRANSIT         -0.563 -0.418  0.742
	ASC_WALK             0.437  0.324  0.742
	costbyincome        -0.053 -0.039  0.742
	hhinc#4             -0.005 -0.004  0.742
	hhinc#5             -0.014 -0.010  0.742
	hhinc#6             -0.008 -0.006  0.742
	motorized_ovtbydist -0.156 -0.116  0.742
	motorized_time      -0.020 -0.015  0.742
	mu                   0.742  0.742  1.000
	nonmotorized_time   -0.062 -0.046  0.742
	vehbywrk_BIKE       -0.984 -0.731  0.742
	vehbywrk_SR         -0.311 -0.231  0.742
	vehbywrk_TRANSIT    -0.973 -0.722  0.742
	vehbywrk_WALK       -1.024 -0.760  0.742
	wkcbd_BIKE           0.538  0.400  0.742
	wkcbd_SR2            0.266  0.197  0.742
	wkcbd_SR3            1.076  0.799  0.742
	wkcbd_TRANSIT        1.272  0.944  0.742
	wkcbd_WALK           0.151  0.112  0.742
	wkempden_BIKE        0.002  0.002  0.742
	wkempden_SR2         0.002  0.001  0.742
	wkempden_SR3         0.002  0.002  0.742
	wkempden_TRANSIT     0.003  0.002  0.742
	wkempden_WALK        0.003  0.002  0.742

.. tip::

	If you want access to the models in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-estimate copy of each model like this::

		umnl = larch.example(42, 'umnl')
		nnnl = larch.example(42, 'nnnl')

