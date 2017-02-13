.. currentmodule:: larch

=====================================
202: Exampville Mode Choice Logsums
=====================================

.. testsetup:: *

	import larch
	import os


Welcome to Exampville, the best simulated town in this here part of the internet!

In our previous example, we estimated a mode choice model.  We'll use that same model
here as the basis for creating logsums that we will use in the next example.  We'll

.. testcode::

	import larch, numpy, pandas, os
	from larch.roles import P,X
	d, nZones, omx = larch.examples.reproduce(200, ['d', 'nZones', 'omx'])
	m = larch.Model.Example(201, d=d)
	m.maximize_loglike()

We're going to build the mode choice logsums, and store them in the DT
for use in the destination choice model.  

One important thing to recall is that we applied some exclusion filters to
the DT, to get only work tours (as opposed to all tours).  In order to 
remain consistent, new arrays we add to the DT need to have the same number
of rows as existing DT arrays (and the same number of caseids).

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> d.nCases()
	1897
	>>> d.nAllCases()
	6123

The `nCases` method returns the number of active (non-excluded) cases,
while the `nAllCases` method counts _all_ of the cases in the DT, whether
they are active or not.

Another concern is that the non-excluded cases are not a contiguous set; they
are spread around the list of all cases.  Fortunately, we have the `get_screen_indexes` method to extract
the current active case indexes, which we will use to expand the logsums we create
and push them back into the DT in the correct places.

.. testcode::

	screen_idx = d.get_screen_indexes()

We can check to see which cases have been included, and that we got the right
number of cases in our screen:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> screen_idx
	array([   0,    2,    3, ..., 6112, 6116, 6119])
	>>> screen_idx.shape
	(1897,)

Now we're ready to generate our logsums.  First we'll create a blank array in the DT for the values.

.. testcode::

	d.new_idca('MODECHOICELOGSUM', numpy.zeros([d.nAllCases(), nZones], dtype=numpy.float32), original_source="mode choice model")

We will also create a seperate array in memory, to cache the logsums we calculate 
so that we can push all the calculated values into the DT on disk in one pass at the end.
Note the in-memory version is sized for nCases (the number of cases in the work tour mode
choice model), while the disk version used nAllCases (the total number of cases in the DT file
before any screening).

.. testcode::

	modechoicelogsums = numpy.zeros([d.nCases(), nZones], dtype=numpy.float32)

Now we'll `setUp` the model, which allocates the necessary memory for various computational parts.
We also set `preserve_casewise_logsums` to True, which will ensure that the logsums created
during the calculation of the loglikelihood are preserved after that calculation completes.
(Otherwise they can be discarded to save memory.)

.. testcode::

	m.setUp()
	m.preserve_casewise_logsums = True

.. note::
	If the model was just estimated, this is unnecessary; otherwise this is necessary
	to allocate the needed arrays for storing data and results.

The we will Loop over all TAZ indexes based on the number of zones, re-calculating the logsums for
every case as if the destination was each zone, instead of the actual destination zone.

.. testcode::

	for dtazi in range(nZones):
		# First we pull in replacement data from the skims,
		# replacing the data for the actually chosen destination with alternative
		# data from the proposed destination
		d.pluck_into_idco(omx, "HOMETAZi", dtazi, overwrite=True)
		
		# We need to refresh these derived variables too, as they are derived
		# from other data columns we just overloaded...
		d.new_idco("WALKTIME", "DIST / 2.5 * 60 * (DIST<=3)", overwrite=True) # 2.5 mph, 60 minutes per hour, max 3 miles
		d.new_idco("BIKETIME", "DIST / 12 * 60 * (DIST<=15)", overwrite=True)  # 12 mph, 60 minutes per hour, max 15 miles
		d.new_idco("CARCOST", "DIST * 0.20", overwrite=True)  # 20 cents per mile
		
		# Then we will load the relevant data into the Model
		m.provision()
		
		# Then we calculate and extract the logsums using the model and store
		# them in the correct indexes of the relevant column in the output array
		modechoicelogsums[:,dtazi] = m.logsums(hardway=True)

		# Note that the `hardway` argument is temporary, pending the resultion of a bug in
		# the more optimal handling of logsum calculations.


Once these calculation have been completed, we can push the array in memory into the data
file on disk.  Remember that they are different sizes, so we'll use the screen_idx array to
map which rows of the disk file are to get the rows from the in-memory array.

.. testcode::

	d.idca.MODECHOICELOGSUM[screen_idx, :] = modechoicelogsums


If we look at the results, we see that logsums are only generated for certain data rows
(the work tours with TOURPURP==1) and the other rows remain zeros.

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> d.idca.MODECHOICELOGSUM[:]
	array([[-2.16048551, -1.75828993, -1.25027871, ..., -3.19889712, -5.19893169, -4.59509802],
		   [ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ],
		   [-2.9085815 , -2.44075465, -2.15272021, ..., -3.36428213, -2.94131351, -4.77855158],
		   ..., 
		   [ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ],
		   [ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ],
		   [ 0.        ,  0.        ,  0.        , ...,  0.        ,  0.        ,  0.        ]], dtype=float32)
	>>> d.idco.TOURPURP[:]
	array([1, 2, 1, ..., 2, 2, 2])

