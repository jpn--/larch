.. currentmodule:: larch

=====================================
200: Exampville Simulated Data
=====================================

.. testsetup:: *

	import larch
	import numpy, pandas, os


Welcome to Exampville, the best simulated town in this here part of the internet!

Exampville is a simulation tool provided with Larch that can quickly simulate the
kind of data that a transportation planner might have available when building
a travel model.  We will use the first Exampville builder to generate some
simulated data.

.. testcode::

	import larch.exampville
	from larch.roles import P,X
	directory, omx, f_hh, f_pp, f_tour = larch.exampville.builder_1(
		nZones=15,
		transit_scope=(4,15),
		n_HH=2000,
	)

The builder creates a temporary directory, and produces a set of files
representing some network skim matrixes in openmatrix (OMX) format, as well
as three DT files containing data on households, persons, and tours, respectively.
We can take a quick peek at what is inside each file:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> omx
	<larch.OMX> ...exampville.omx
	 |  shape:(15, 15)
	 |  data:
	 |    AUTO_TIME (float64)
	 |    DIST      (float64)
	 |    RAIL_FARE (float64)
	 |    RAIL_TIME (float64)
	 |  lookup:
	 |    EMPLOYMENT    (15 float64)
	 |    EMP_NONRETAIL (15 float64)
	 |    EMP_RETAIL    (15 float64)
	 |    LAT           (15 int64)
	 |    LON           (15 int64)
	 |    TAZID         (15 int64)

	>>> f_hh
	<larch.H5PodCO>
	 |  file: ...exampville_hh.h5
	 |  shape: (2000,)
	 |  data:
	 |    HHID    (int64)
	 |    HHSIZE  (int64)
	 |    HOMETAZ (int64)
	 |    INCOME  (int64)

	>>> f_pp
	<larch.H5PodCO>
	 |  file: ...exampville_person.h5
	 |  shape: (3462,)
	 |  data:
	 |    AGE          (int64)
	 |    HHID         (int64)
	 |    N_OTHERTOURS (int64)
	 |    N_TOTALTOURS (int64)
	 |    N_WORKTOURS  (int64)
	 |    PERSONID     (int64)
	 |    WORKS        (int64)

	>>> f_tour
	<larch.H5PodCO>
	 |  file: ...exampville_tours.h5
	 |  shape: (6123,)
	 |  data:
	 |    DTAZ     (int64)
	 |    HHID     (int64)
	 |    PERSONID (int64)
	 |    TOURID   (int64)
	 |    TOURMODE (int64)
	 |    TOURPURP (int64)



The Exampville data output contains a set of files similar to what we might
find for a real travel survey: network skims, and tables of households, persons,
and tours.  We'll need to merge these tables to create a composite dataset
for mode choice model estimation.


.. testcode::

	DA = 1
	SR = 2
	Walk = 3
	Bike = 4
	Transit = 5

	f_tour.merge_external_data(f_hh.astype('idco'), 'HHID', )
	f_tour.merge_external_data(f_pp.astype('idco'), 'PERSONID', )
	f_tour.add_expression("HOMETAZi", "HOMETAZ-1", dtype=numpy.int64)
	f_tour.add_expression("DTAZi", "DTAZ-1", dtype=numpy.int64)

Let's define some variables for clarity.  We could in theory call the complete formula
"DIST / 2.5 * 60 * (DIST<=3)" every time we want to refer to the walk time (given as
distance in miles, divided by 2.5 miles per hour, times 60 minutes per hour, but only up to 3 miles).
But it's much easier, and potentially faster, to pre-compute the walk time and use it directly.

.. testcode::

	from larch.data_services.h5 import H5Pod, H5PodRC, H5PodGroup, H5PodCS
	from larch.data_services import Pods
	from larch.data_services import DataService
	omx = omx.change_mode('a')
	omx_ = H5Pod(groupnode=omx.data)
	omx_.add_expression("WALKTIME", "DIST / 2.5 * 60 * (DIST<=3)") # 2.5 mph, 60 minutes per hour, max 3 miles
	omx_.add_expression("BIKETIME", "DIST / 12 * 60 * (DIST<=15)")  # 12 mph, 60 minutes per hour, max 15 miles
	omx_.add_expression("CARCOST", "DIST * 0.20")  # 20 cents per mile
	omx_.close()
	omx = omx.change_mode('r')
	rc = H5PodRC(f_tour.HOMETAZi[:], f_tour.DTAZi[:], groupnode=omx.data, )
	tours_stack = H5PodCS([f_tour,rc], storage=f_tour).set_alts([DA,SR,Walk,Bike,Transit])
	tours_stack.set_bunch('choices',{
	        DA: 'TOURMODE==1',
	        SR: 'TOURMODE==2',
	        Walk: 'TOURMODE==3',
	        Bike: 'TOURMODE==4',
	        Transit: 'TOURMODE==5',
	})
	tours_stack.set_bunch('availability',{
	        DA: '(AGE>=16)',
	        SR: '1',
	        Walk: 'DIST<=3',
	        Bike: 'DIST<=15',
	        Transit: 'RAIL_TIME>0',
	})
	d = DataService(pods=[f_tour, tours_stack, rc])
	d.set_alternatives( [DA,SR,Walk,Bike,Transit], ['DA','SR','Walk','Bike','Transit'] )

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> d
	<larch.DataService>
	 | idco:
	 |   <larch.H5PodCO>
	 |    |  file: ...exampville_tours.h5
	 |    |  shape: (6123,)
	 |    |  data:
	 |    |    AGE          (int64)
	 |    |    DTAZ         (int64)
	 |    |    DTAZi        (int64)
	 |    |    HHID         (int64)
	 |    |    HHSIZE       (int64)
	 |    |    HOMETAZ      (int64)
	 |    |    HOMETAZi     (int64)
	 |    |    INCOME       (int64)
	 |    |    N_OTHERTOURS (int64)
	 |    |    N_TOTALTOURS (int64)
	 |    |    N_WORKTOURS  (int64)
	 |    |    PERSONID     (int64)
	 |    |    TOURID       (int64)
	 |    |    TOURMODE     (int64)
	 |    |    TOURPURP     (int64)
	 |    |    WORKS        (int64)
	 |    |    _BUNCHES_    (<no dtype>)
	 |   <larch.H5PodRC>
	 |    |  file: ...exampville.omx
	 |    |  node: /data
	 |    |  shape: (6123,)
	 |    |  metashape: (15, 15)
	 |    |  data:
	 |    |    AUTO_TIME (float64)
	 |    |    BIKETIME  (float64)
	 |    |    CARCOST   (float64)
	 |    |    DIST      (float64)
	 |    |    RAIL_FARE (float64)
	 |    |    RAIL_TIME (float64)
	 |    |    WALKTIME  (float64)
	 | idca:
	 |   <larch.H5PodCOasCA>
	 |    |  file: ...exampville_tours.h5
	 |    |  shape: (6123, 5)
	 |    |  data:
	 |    |    AGE          (int64)
	 |    |    DTAZ         (int64)
	 |    |    DTAZi        (int64)
	 |    |    HHID         (int64)
	 |    |    HHSIZE       (int64)
	 |    |    HOMETAZ      (int64)
	 |    |    HOMETAZi     (int64)
	 |    |    INCOME       (int64)
	 |    |    N_OTHERTOURS (int64)
	 |    |    N_TOTALTOURS (int64)
	 |    |    N_WORKTOURS  (int64)
	 |    |    PERSONID     (int64)
	 |    |    TOURID       (int64)
	 |    |    TOURMODE     (int64)
	 |    |    TOURPURP     (int64)
	 |    |    WORKS        (int64)
	 |    |    _BUNCHES_    (<no dtype>)
	 |   <larch.H5PodCS>
	 |    |  file: ...exampville_tours.h5
	 |    |  node: /_BUNCHES_
	 |    |  shape: (6123, 5)
	 |    |  bunches:
	 |    |    choices: {1: 'TOURMODE==1',
	 |    |              2: 'TOURMODE==2',
	 |    |              3: 'TOURMODE==3',
	 |    |              4: 'TOURMODE==4',
	 |    |              5: 'TOURMODE==5'}
	 |    |    availability: {1: '(AGE>=16)', 2: '1', 3: 'DIST<=3', 4: 'DIST<=15', 5: 'RAIL_TIME>0'}
	 |   <larch.H5PodRCasCA>
	 |    |  file: ...exampville.omx
	 |    |  node: /data
	 |    |  shape: (6123, 5)
	 |    |  metashape: (15, 15)
	 |    |  data:
	 |    |    AUTO_TIME (float64)
	 |    |    BIKETIME  (float64)
	 |    |    CARCOST   (float64)
	 |    |    DIST      (float64)
	 |    |    RAIL_FARE (float64)
	 |    |    RAIL_TIME (float64)
	 |    |    WALKTIME  (float64)


.. tip::

	If you want access to the data in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-use copy like this::

		d = larch.examples.EXAMPVILLE()
