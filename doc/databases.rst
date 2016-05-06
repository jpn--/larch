.. currentmodule:: larch

====================================
Data Storage and Access Using SQLite
====================================

The default storage of data within Larch is handled using SQLite. This portable and
open source database system provides a common file format that is flexible and 
practical for storing data.

The interactions with data in Python take place through a :class:`DB` object, which
is derived from the :class:`apsw.Connection` class in APSW, the Python interface
wrapper for SQLite.

Creating :class:`DB` Objects
----------------------------

.. autoclass:: DB(filename=None, readonly=False)
	

In addition to opening an existing SQLite database directly, there are a number of
methods available to create a :class:`DB` object without having it linked to an
original database file.
	
.. automethod:: DB.Copy

.. automethod:: DB.Example

.. automethod:: DB.CSV_idco

.. automethod:: DB.CSV_idca


Importing Data
--------------

There are a variety of methods available to import data from external sources into
a SQLite table for use with the larch DB facility.

.. automethod:: DB.import_csv

.. automethod:: DB.import_dataframe

.. automethod:: DB.import_xlsx

.. automethod:: DB.import_dbf





Exporting Data
--------------

Sometimes it will be necessary to get your data out of the database, for use in other
programs or for other sundry purposes. There will eventually be some documented methods to conviently allow you to export
data in a few standard formats.  Of course, since the :class:`DB` object links to a
standard SQLite database, it is possible to access your data directly from SQLite in
other programs, or through :mod:`apsw` (included as part of Larch)
or :mod:`sqlite3` (included in standard Python distributions).

.. automethod:: DB.export_idca

.. automethod:: DB.export_idco



Reviewing Data
--------------

.. automethod:: DB.seer


Loading Data into Arrays
------------------------

.. automethod:: DB.array_caseids

.. automethod:: DB.array_idca

.. automethod:: DB.array_idco


Convenience Methods
-------------------

.. automethod:: DB.attach

.. automethod:: DB.detach

.. automethod:: DB.crack_idca



Using Data in Models
--------------------

The :class:`DB` class primarily presents an interface between python and SQLite. The interface
between a :class:`DB` and a :class:`Model` is governed by a special attribute of the
:class:`DB` class:

.. attribute:: DB.queries

	This attribute defines the automatic queries used to provision a :class:`Model` with data.
	It should be an object that is a specialized subtype of the :class:`core.QuerySet` abstract
	base class.


.. toctree::

	queryset
	queryset_idco
	queryset_2




