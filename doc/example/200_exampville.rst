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
