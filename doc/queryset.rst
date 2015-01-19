.. currentmodule:: larch

=======================
Automatic Queries
=======================



.. class:: core.QuerySet

	To provide the ability to extract the correct data from the database, the
	:class:`DB` object has an attribute :attr:`DB.queries`, which is an object that is a subclass of
	this abstract base class.

	.. method:: tbl_idca()

		This method returns a SQL fragment that evaluates to an larch_idca table. The table
		should have the following features:

			* Column 1: caseid (integer) a key for every case observed in the sample
			* Column 2: altid (integer) a key for each alternative available in this case
			* Column 3+ can contain any explanatory data, typically numeric data, although non-numeric data is allowable.

		If no columns have the name *caseid* and *altid*, larch will use the first two columns, respectively.
		A query with less than two columns should raise an exception.

		For example, this method might return::

			(SELECT casenum AS caseid, altnum AS altid, * FROM data) AS larch_idca

		It would be perfectly valid for there to be an actual table in the database named "larch_idca", and
		for this function to return simply "larch_idca", although this would prohibit using the same
		underlying database to build different datasets.

	.. method:: tbl_idco()

		This method returns a SQL fragment that evaluates to an larch_idco table. The table
		should have the following features:

			* Column 1: caseid (integer) a key for every case observed in the sample
			* Column 2+ can contain any explanatory data, typically numeric data, although non-numeric data is allowable.

		If no column has the name 'caseid', larch will use the first column.
		A query with less than two columns should raise an exception.

		For example, this method might return::

			(SELECT _rowid_ AS caseid, * FROM data) AS larch_idco

	
	.. method:: tbl_alts()

		This method returns a SQL fragment that evaluates to an larch_alternatives table. The table
		should have the following features:

			* Column 1: id (integer) a key for every alternative observed in the sample
			* Column 2: name (text) a name for each alternative



	.. method:: tbl_caseids()

		This method returns a SQL fragment that evaluates to an larch_caseids table. The table
		should have the following features:

			* Column 1: caseid (integer) a key for every case observed in the sample



	.. method:: tbl_choice ()

		This method returns a SQL fragment that evaluates to an larch_choice table. The table
		should have the following features:

			* Column 1: caseid (integer) a key for every case observed in the sample
			* Column 2: altid (integer) a key for each alternative available in this case
			* Column 3: choice (numeric, typically 1.0 but could be other values)

		If an alternative is not chosen for a given case, it can have a zero choice value or
		it can simply be omitted from the result.



	.. method:: tbl_weight ()

		This method returns a SQL fragment that evaluates to an larch_weight table. The table
		should have the following features:

			* Column 1: caseid (integer) a key for every case observed in the sample
			* Column 2: weight (numeric) a weight associated with each case

		Alternatively, this method can return an empty string, in which case it is assumed that
		all cases are weighted equally.



	.. method:: tbl_avail  ()

		This method returns a SQL fragment that evaluates to an larch_avail table. The table
		should have the following features:

			* Column 1: caseid (integer) a key for every case observed in the sample
			* Column 2: altid (integer) a key for each alternative available in this case
			* Column 3: avail (boolean) evaluates as 1 or true when the alternative is available, 0 otherwise

		If an alternative is not available for a given case, it can have a zero avail value or
		it can simply be omitted from the result.

		Alternatively, this method can return an empty string, in which case it is assumed that
		all alternatives are available in all cases.






Queries for a Single *idco* Table
---------------------------------

.. class:: QuerySetSimpleCO

	This subclass of :class:`core.QuerySet` is used when the data consists exclusively of a single
	idco format table.

	.. note::

		This is similar to the data format required by Biogeme. Unlike Biogeme, the Larch
		:class:`DB` allows non-numeric values in the source data.


	.. attribute:: idco_query

		This attribute defines a SQL query that evaluates to an larch_idco table. The table
		should have the following features:

			* Column 1: caseid (integer) a key for every case observed in the sample
			* Column 2+ can contain any explanatory data, typically numeric data, although non-numeric data is allowable.

		If the main table is named "data" typically this query will be::

			SELECT _rowid_ AS caseid, * FROM data


	.. autoattribute:: alts_query

	.. method:: set_alts_values(alts)

		Defines a set of alternative codes and names.

		alts : dict
			A dictionary that contains `int`:`str` key/value pairs, where
			each key is an integer value corresponding to an alternative code, and each
			value is a string giving a descriptive name for the alternative.

		This method does not create a table in the :class:`DB`. Instead it defines a
		query that can be used with no table.

		.. warning:: Using this method will overwrite :attr:`alts_query`



	.. attribute:: choice

		This attributes defines the choices. It has two styles:

			* When set to a string, the string names the column of the main table that identifies
			  the choice for each case.  The indicated column should contain integer values
			  corresponding to the alternative codes.

			* When set to a dict, the dict should contain {integer:string} key/value pairs, where
			  each key is an integer value corresponding to an alternative code, and each
			  value is a string identifying a column in the main table; that column should
			  contain a value indication whether the alternative was chosen. Usually this will be
			  a binary dummy variable, although it need not be. For certain specialized models,
			  values other than 0 or 1 may be appropriate.

		The choice of style is a matter of convenience; the same data can be expressed with either
		style as long as the choices are binary.


	.. autoattribute:: avail



	.. attribute:: weight

		This attribute names the column in the main table that defines the weight for each case.
		Set it to an empty string, or 1.0, to assign all cases equal weight.











