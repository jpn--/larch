.. currentmodule:: larch

=======================================
100: Importing |idco| Data
=======================================

.. testsetup:: *

	import larch
	import os



In this example we will import the SWISSMETRO example dataset into a :class:`DT`, starting from a csv
text file in |idco| format.  Suppose that data file is named "swissmetro.csv" and
is located in the current directory (use :func:`os.getcwd` to see what is the
current directory).


.. testcode::
	:hide:

	os.chdir(os.path.join(larch._directory_,"data_warehouse"))


We can take a peek at the contents of the file, examining the first 10 lines:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> with open("swissmetro.csv", 'rt') as previewfile:
	...     print(*(next(previewfile) for x in range(10)))
	GROUP,SURVEY,SP,ID,PURPOSE,FIRST,TICKET,WHO,LUGGAGE,AGE,MALE,INCOME,GA,ORIGIN,DEST,TRAIN_AV,CAR_AV,SM_AV,TRAIN_TT,TRAIN_CO,TRAIN_HE,SM_TT,SM_CO,SM_HE,SM_SEATS,CAR_TT,CAR_CO,CHOICE
	2,0,1,1,1,0,1,1,0,3,0,2,0,2,1,1,1,1,112,48,120,63,52,20,0,117,65,2
	2,0,1,1,1,0,1,1,0,3,0,2,0,2,1,1,1,1,103,48,30,60,49,10,0,117,84,2
	2,0,1,1,1,0,1,1,0,3,0,2,0,2,1,1,1,1,130,48,60,67,58,30,0,117,52,2
	2,0,1,1,1,0,1,1,0,3,0,2,0,2,1,1,1,1,103,40,30,63,52,20,0,72,52,2
	2,0,1,1,1,0,1,1,0,3,0,2,0,2,1,1,1,1,130,36,60,63,42,20,0,90,84,2
	2,0,1,1,1,0,1,1,0,3,0,2,0,2,1,1,1,1,112,36,120,60,49,10,0,90,52,2
	2,0,1,1,1,0,1,1,0,3,0,2,0,2,1,1,1,1,103,48,120,67,58,10,0,72,65,2
	2,0,1,1,1,0,1,1,0,3,0,2,0,2,1,1,1,1,103,36,30,67,43,30,0,90,65,1
	2,0,1,1,1,0,1,1,0,3,0,2,0,2,1,1,1,1,130,40,60,60,46,10,0,72,65,2


The first line of the file contains column headers. After that, each line represents
a decision maker.  The idco data isn't quite as self-explanatory as idca data can be,
so we'll need to specify the factors that define whether an alternative is available.
We do that here in the swissmetro_alts dict, where the keys are the alternative codes,
and the values are tuples of (alternative name, availability expression).
Then we can import this data easily:

.. doctest::

	>>> swissmetro_alts = {
	... 	1:('Train','TRAIN_AV*(SP!=0)'),
	... 	2:('SM','SM_AV'),
	... 	3:('Car','CAR_AV*(SP!=0)'),
	... }
	>>> d = larch.DT.CSV_idco("swissmetro.csv", choice="CHOICE", alts=swissmetro_alts)

We can then look at some of the attibutes of the imported data:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> d.variables_co()
	['AGE', 'CAR_AV', 'CAR_CO', 'CAR_TT', 'CHOICE', 'DEST', 'FIRST', 'GA', 'GROUP', 'ID', 'INCOME', 'LUGGAGE', 'MALE', 'ORIGIN', 'PURPOSE', 'SM_AV', 'SM_CO', 'SM_HE', 'SM_SEATS', 'SM_TT', 'SP', 'SURVEY', 'TICKET', 'TRAIN_AV', 'TRAIN_CO', 'TRAIN_HE', 'TRAIN_TT', 'WHO']
	>>> d.variables_ca()
	['_avail_', '_choice_']
	>>> d.alternative_codes()
	(1, 2, 3)
	>>> d.alternative_names()
	('Train', 'SM', 'Car')


Larch automatically created |idca| format variables for availability and choice.

The swissmetro dataset, as with all Biogeme data, is only in `co` format.  But,
in the models we want to build some of the attributes are "generic", i.e. stuff like travel time, which
varies across alternatives, but for which we'll want to assign the same parameter
to for each alternative (so that a minute of travel time has the same value no
matter which alternative it is on).  So, we can create the generic `ca` format
variables by stacking the relevant `co` variables.

.. testcode::

	d.stack_idco('traveltime', {1: "TRAIN_TT", 2: "SM_TT", 3: "CAR_TT"})
	d.stack_idco('cost', {1: "TRAIN_CO*(GA==0)", 2: "SM_CO*(GA==0)", 3: "CAR_CO"})

Then these stacked variables become available for :ref:`idca` uses:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> d.variables_ca()
	['_avail_', '_choice_', 'cost', 'traveltime']

The data file we've loaded includes all the rows of the dataset.

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> d.nCases()
	10728

But most of the Biogeme examples employ data filtering.  This reduces the dataset
by dropping cases where the data is invalid or which we don't want to use for
whatever reason.  When the data is stored in a :class:`DB`, we can use a "WHERE"
in the queries to filted the data.  In the :class:`DT`, that function is
filled by the `screen` node.

We can define a screen array either manually, or we can just add some exclusion
factors, like this:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> d.exclude_idco("PURPOSE not in (1,3)")
	3960
	>>> d.exclude_idco("CHOICE == 0")
	0
	>>> d.nCases()
	6768

Now we're ready to use our data.




