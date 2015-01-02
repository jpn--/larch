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
	
.. automethod:: DB.Copy

.. automethod:: DB.Example



Importing Data
--------------

There are a variety of methods available to import data from external sources into
a SQLite table for use with the larch DB facility.

.. automethod:: DB.import_csv

.. automethod:: DB.import_dataframe

.. automethod:: DB.import_dbf




Exporting Data
--------------

Sometimes it will be necessary to get your data out of the database, for use in other
programs or for other sundry purposes. There will eventually be some documented methods to conviently allow you to export
data in a few standard formats.  Of course, since the :class:`DB` object links to a
standard SQLite database, it is possible to access your data directly from SQLite in
other programs, or through :mod:`apsw` (included as part of Larch)
or :mod:`sqlite3` (included in standard Python distributions).


