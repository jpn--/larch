.. currentmodule:: larch

=============
Data in Larch
=============

The default storage of data within Larch is handled using SQLite. This portable and
open source database system provides a common file format that is flexible and 
practical for storing data.

The interactions with data in Python take place through a :class:`DB` object, which
is derived from the :class:`apsw.Connection` class in APSW, the Python interface
wrapper for SQLite.

Creating :class:`DB` Objects
----------------------------

.. class:: DB(filename='file:larchdb?mode=memory', skip_initialization=False)
	
	This object wraps a :class:`apsw.Connection`, adding a number of methods designed
	specifically for working with choice-based data used in `larch`.

	The normal constructor creates a :class:`DB` object linked to an existing SQLite
	database file. Editing the object edits the file as well. There is currently no
	"undo" so be careful with this method.
	
In addition to opening an existing SQLite database directly, there are a number of
methods available to create a :class:`DB` object without having it linked to an
original database file.
	
.. method:: DB.Copy

	:param source: The source database.
	:type source:  str
	:param destination: The destination database.
	:type destination: str
	:returns: A DB object with an open connection to the destination DB.
	
	Create a copy of a database and link it to a DB object.
	It is often desirable to work on a copy of your data, instead of working
	with the original file. If you data file is not very large and you are 
	working with multiple models, there can be significant speed advantages
	to copying the entire database into memory first, and then working on it
	there, instead of reading from disk every time you want data.


.. method:: DB.Example

	Generate an example data object in memory.
	Larch comes with a few example data sets, which are used in documentation
	and testing. It is important that you do not edit the original data, so
	this function copies the data into an in-memory database, which you can
	freely edit without damaging the original data.




Importing Data
--------------

There are a variety of methods available to import data from external sources into
a SQLite table for use with the larch DB facility.

.. method:: DB.import_csv

	Import raw csv or tab-delimited data into SQLite.

	:param rawdata:     The absolute path to the raw csv or tab delimited data file.
	:param table:       The name of the table into which the data is to be imported
	:param drop_old:    Bool= drop old data table if it already exists?
	:param progress_callback: If given, this callback function takes a single integer
						as an argument and is called periodically while loading
						with the current precentage complete.
	
	:result:            A list of column headers from the imported csv file


.. method:: DB.import_dataframe

	Imports data from a pandas dataframe into an existing larch DB.

	:param rawdataframe: An existing pandas dataframe.
	:param table:        The name of the table into which the data is to be imported
	:param if_exists:    Should be one of {‘fail’, ‘replace’, ‘append’}. If the table
						 does not exist this parameter is ignored, otherwise,
						 *fail*: If table exists, raise a ValueError exception.
						 *replace*: If table exists, drop it, recreate it, and insert data.
						 *append*: If table exists, insert data.
	
	:result:             A list of column headers from imported pandas dataframe


.. method:: DB.import_dbf

	Imports data from a DBF file into an existing larch DB.
	
	:param rawdata:     The path to the raw DBF data file.
	:param table:       The name of the table into which the data is to be imported
	:param drop_old:    Bool= drop old data table if it already exists?
	
	:result:            A list of column headers from imported csv file
	
	Note: this method requires the dbfpy module (available using pip).




Exporting Data
--------------

Sometimes it will be necessary to get your data out of the database, for use in other
programs or for other sundry purposes. There will eventually be some documented methods to conviently allow you to export
data in a few standard formats.  Of course, since the :class:`DB` object links to a
standard SQLite database, it is possible to access your data directly from SQLite in
other programs, or through :mod:`apsw` (included as part of Larch)
or :mod:`sqlite3` (included in standard Python distributions).


