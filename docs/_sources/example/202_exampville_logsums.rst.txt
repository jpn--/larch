.. currentmodule:: larch

=====================================
202: Exampville Mode Choice Logsums
=====================================

.. testsetup:: *

	import larch
	import os
	import pandas
	import numpy
	pandas.set_option("display.max_columns", 999)
	pandas.set_option('expand_frame_repr', False)
	pandas.set_option('precision', 3)
	numpy.set_printoptions(precision=5, linewidth=200)
	larch._doctest_mode_ = True


Welcome to Exampville, the best simulated town in this here part of the internet!

In our previous example, we estimated a mode choice model.  We'll use that same model
here as the basis for creating logsums that we will use in the next example.  We'll
start from our mode choice model.

.. testcode::

	import numpy
	import larch.examples
	m = larch.example(201)
	m.load_data()
	m.maximize_loglike()

We're going to build the mode choice logsums, and store them in the DT
for use in the destination choice model.  

One important thing to recall is that we applied some filters to
the data, to get only work tours (as opposed to all tours).

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> m.selector_eval().sum()
	1897
	>>> m.dataservice.idco.shape
	(6123,)

In order to remain consistent, new arrays we create need to have the same number
of rows as existing arrays (i.e., 6123 rows, not just 1897).

Another concern is that the included cases are not a contiguous set; they
are spread around the list of all cases.  Fortunately, we have the :meth:`~Model.selector_eval` method to extract
the current active case indexes, which we will use to expand the logsums we create
and push them back into the final array in the correct places.

.. testcode::

	filter_idx = m.selector_eval()

We can check to see which cases have been included, and that we got the right
number of cases in our screen:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> filter_idx
	array([ True, False,  True, ..., False, False, False])
	>>> filter_idx.shape
	(6123,)

Now we're ready to generate our logsums.  First we'll create a blank array for the values.

.. testcode::

	nZones=15
	logsums = numpy.zeros([6123, nZones], dtype=numpy.float32)

We will mark several of the data servers with `durable_mask` markers, that tell Larch that
these data servers have data that will not change across destinations, and so will not need
to be reloaded repeatedly.

.. testcode::

	m.dataservice.idco[0].durable_mask = 0x1
	m.dataservice.idca[0].durable_mask = 0x2

Then we iterate across possible destinations, reloading the data as appropriate.

.. testcode::

	for zone in range(nZones):
		m.dataservice.idco[1].colindexes[:] = zone
		m.load_data(durable_mask=0xF)
		m.loglike()
		logsums[filter_idx, zone] = m.logsums()

Let's confirm we got the logsums we are looking for.

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> logsums
	array([[-2.16044, -1.75825, -1.25025, ..., -3.19883, -5.19884, -4.59501],
		   [ 0.     ,  0.     ,  0.     , ...,  0.     ,  0.     ,  0.     ],
		   [-2.90855, -2.44073, -2.15271, ..., -3.36426, -2.94129, -4.77848],
		   ...,
		   [ 0.     ,  0.     ,  0.     , ...,  0.     ,  0.     ,  0.     ],
		   [ 0.     ,  0.     ,  0.     , ...,  0.     ,  0.     ,  0.     ],
		   [ 0.     ,  0.     ,  0.     , ...,  0.     ,  0.     ,  0.     ]], dtype=float32)


.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE
	:hide:

	>>> from larch.examples import EXAMPVILLE
	>>> lib = EXAMPVILLE()
	>>> numpy.allclose(lib.logsums.MODECHOICELOGSUM[:], logsums)
	True
