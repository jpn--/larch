.. currentmodule:: larch

=====================================
200: Exampville Simulated Data
=====================================

.. testsetup:: *

	import larch
	import os

.. testcode::
	:hide:

	os.chdir(os.path.join(larch._directory_,"data_warehouse"))

Welcome to Exampville, the best simulated town in this here part of the internet!

Exampville is a simulation tool provided with Larch that can quickly simulate the
kind of data that a transportation planner might have available when building
a travel model.  We will use the first Exampville builder to generate some
simulated data.

.. testcode::

	import larch, numpy, pandas, os
	from larch.roles import P,X
	from larch.examples.exampville import builder_1

	nZones = 15
	transit_scope = (4,15)
	n_HH = 2000
	directory, omx, f_hh, f_pp, f_tour = builder_1(
		nZones=nZones, 
		transit_scope=transit_scope, 
		n_HH=n_HH,
	)

The builder creates a temporary directory, and produces a set of files
representing some network skim matrixes in openmatrix (OMX) format, as well
as three DT files containing data on households, persons, and tours, respectively.
We can take a quick peek at what is inside each file:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> omx
	<larch.OMX> .../exampville.omx
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
	<larch.DT> .../exampville_hh.h5
	 |  > file is opened for read/write <
	 |  nCases: 2000
	 |  nAlts: <missing>
	 |  idco:
	 |    HHSIZE  	int64  
	 |    HOMETAZ 	int64  
	 |    INCOME  	int64  

	>>> f_pp
	<larch.DT> .../exampville_person.h5
	 |  > file is opened for read/write <
	 |  nCases: 3462
	 |  nAlts: <missing>
	 |  idco:
	 |    AGE         	int64  
	 |    HHID        	int64  
	 |    N_OTHERTOURS	int64  
	 |    N_TOTALTOURS	int64  
	 |    N_WORKTOURS 	int64  
	 |    WORKS       	int64  

	>>> f_tour
	<larch.DT> .../exampville_tours.h5
	 |  > file is opened for read/write <
	 |  nCases: 6123
	 |  nAlts: <missing>
	 |  idco:
	 |    DTAZ    	int64  
	 |    HHID    	int64  
	 |    PERSONID	int64  
	 |    TOURMODE	int64  
	 |    TOURPURP	int64  



The Exampville data output contains a set of files similar to what we might
find for a real travel survey: network skims, and tables of households, persons,
and tours.  We'll need to merge these tables to create a composite dataset
for mode choice model estimation.


.. testcode::

	### MODE CHOICE DATA

	# Define numbers and names for modes
	DA = 1
	SR = 2
	Walk = 3
	Bike = 4
	Transit = 5

	d = larch.DT() 
	# By omitting a filename here, we create a temporary HDF5 store.

	d.set_alternatives( [DA,SR,Walk,Bike,Transit], ['DA','SR','Walk','Bike','Transit'] )

	### merging survey datasets ###
	d.new_caseids( f_tour.caseids() )
	d.merge_into_idco(f_tour, "caseid")
	d.merge_into_idco(f_pp, "PERSONID")
	d.merge_into_idco(f_hh, "HHID")

	### merging skims ###
	# Create a new variables with zero-based home TAZ numbers
	d.new_idco("HOMETAZi", "HOMETAZ-1", dtype=int)
	d.new_idco("DTAZi", "DTAZ-1", dtype=int)

	# Pull in plucked data from Matrix file
	d.pluck_into_idco(omx, "HOMETAZi", "DTAZi")
	# This command is new as of Larch 3.3.15
	# It loads all the matrix DATA from an OMX based on OTAZ and DTAZ 
	# columns that already exist in the DT

	### prep data ###
	d.choice_idco = {
		DA: 'TOURMODE==1',
		SR: 'TOURMODE==2',
		Walk: 'TOURMODE==3',
		Bike: 'TOURMODE==4',
		Transit: 'TOURMODE==5',
	}
	# Alternately:   d.choice_idco = {i:'TOURMODE=={}'.format(i) for i in [1,2,3,4,5]}

	d.avail_idco = {
		DA: '(AGE>=16)',
		SR: '1',
		Walk: 'DIST<=3',
		Bike: 'DIST<=15',
		Transit: 'RAIL_TIME>0',
	}

Let's define some variables for clarity.  We could in theory call the complete formula
"DIST / 2.5 * 60 * (DIST<=3)" every time we want to refer to the walk time (given as
distance in miles, divided by 2.5 miles per hour, times 60 minutes per hour, but only up to 3 miles).
But it's much easier, and potentially faster, to pre-compute the walk time and use it directly.

.. testcode::

	d.new_idco("WALKTIME", "DIST / 2.5 * 60 * (DIST<=3)") # 2.5 mph, 60 minutes per hour, max 3 miles
	d.new_idco("BIKETIME", "DIST / 12 * 60 * (DIST<=15)")  # 12 mph, 60 minutes per hour, max 15 miles
	d.new_idco("CARCOST", "DIST * 0.20")  # 20 cents per mile


.. tip::

	If you want access to the data in this example without worrying about assembling all the code blocks
	together on your own, you can load a read-to-use copy like this::

		d = larch.DT.Example(200)



