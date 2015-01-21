.. currentmodule:: larch

======================================
Queries To Provision a Model with Data
======================================



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















