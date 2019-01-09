.. currentmodule:: larch

.. _example203:

=====================================
203: Exampville Destination Choice
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

In our previous example, we calculated mode choice model logsums.  Now we'll use those
here as data in a destination choice model.  We'll begin by preparing the data.

Data Prep
~~~~~~~~~

.. testcode::

	import larch, numpy, pandas, os
	from larch import P,X
	import larch.examples
	from larch.data_services.h5 import H5PodCS, H5Pod0A, H5PodGA

	nZones = 15
	lib = larch.examples.EXAMPVILLE()

Load up the logsums.

.. testcode::

	dest_stack = H5PodCS(
		[lib.tours],
		alts=lib.dest_ids,
		dest_choices={ taz: 'DTAZ=={}'.format(taz) for taz in lib.dest_ids },
		dest_availability={ taz: '1' for taz in lib.dest_ids },
	)
	skims_lookup = H5Pod0A(
		filename=lib.skims.lookup,
		ident='skims_l',
		n_cases=6123,
		n_alts=nZones,
	)
	skims_r = H5PodGA(
		rowindexes=lib.tours.get_data_item('HOMETAZi', dtype=int),
		filename=lib.skims_rc,
		ident='skims_r',
	)

	pods = [lib.tours, skims_r, dest_stack, lib.logsums, skims_lookup]

	dd = larch.DataService( *pods, altids=lib.dest_ids )

	m = larch.Model(dataservice=dd)

	m.title = "Exampville Work Tour Destination Choice"

	m.utility_ca = (
		+ P.ModeChoiceLogSum * X.MODECHOICELOGSUM
		+ X('log1p(DIST)') * P.logDistanceP1
		+ X('DIST') * P.Distance
	)

	m.quantity_ca = (
		+ P("EmpRetail_HighInc") * X('EMP_RETAIL * (INCOME>50000)')
		+ P("EmpNonRetail_HighInc") * X('EMP_NONRETAIL') * X("INCOME>50000")
		+ P("EmpRetail_LowInc") * X('EMP_RETAIL') * X("INCOME<=50000")
		+ P("EmpNonRetail_LowInc") * X('EMP_NONRETAIL') * X("INCOME<=50000")

	)

	m.quantity_scale = P.Theta

	m.availability_var = 'dest_availability'
	m.choice_ca_var = 'dest_choices'

	m.selector = 'TOURPURP==1'

	m.pf.loc['EmpRetail_HighInc', 'holdfast'] = 1
	m.pf.loc['EmpRetail_LowInc', 'holdfast'] = 1

	m.selector = numpy.where(m.selector_eval())[0]

	m.load_data()




.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE


	>>> m.maximize_loglike()
	┣ ...Optimization terminated successfully...
	>>> m.loglike()
	-3670.5976296...
	>>> print(m.pfo()[['value','initvalue','nullvalue','minimum','maximum','holdfast']])
	                                   value  initvalue  nullvalue  minimum  maximum  holdfast
	Category Parameter
	Other    Distance             -9.451e-04        0.0        0.0     -inf      inf         0
	         EmpNonRetail_HighInc  4.642e-01        0.0        0.0     -inf      inf         0
	         EmpNonRetail_LowInc  -8.177e-01        0.0        0.0     -inf      inf         0
	         EmpRetail_HighInc     0.000e+00        0.0        0.0     -inf      inf         1
	         EmpRetail_LowInc      0.000e+00        0.0        0.0     -inf      inf         1
	         ModeChoiceLogSum      1.053e+00        0.0        0.0     -inf      inf         0
	         Theta                 8.115e-01        1.0        1.0    0.001    1.000         0
	         logDistanceP1        -5.204e-02        0.0        0.0     -inf      inf         0




