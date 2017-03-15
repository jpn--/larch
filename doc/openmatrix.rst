.. currentmodule:: larch

==================================
Using Open Matrix
==================================

Larch embeds a python interface for interacting with
`open matrix (OMX) <https://github.com/osPlanning/omx/wiki>`_ files.
This data format rests on HDF5, the exact same underlying technology used in
:class:`DT` files.  This makes merging and linking to open matrix data easy and
fun. 

.. py:class:: OMX(filename)

	A subclass of the :class:`tables.File` class, adding an interface for openmatrix files.
	
	As suggested in the openmatrix documentation, the default when creating an OMX file
	is to use zlib compression level 1, although this can be overridden.


.. py:attribute:: OMX.shape

	The shape of the OMX file.

	As required by the standard, all OMX files must have a two dimensional shape. This 
	attribute accesses or alters that shape. Note that attempting to change the 
	shape of an existing file that already has data tables that would be incompatible
	with the new shape will raise an OMXIncompatibleShape exception.


Importing Data
--------------

.. automethod:: OMX.import_datatable

	Import a table in r,c,x,x,x... format into the matrix.

	The r and c columns need to be either 0-based or 1-based index values
	(this may be relaxed in the future). The matrix must already be set up
	with the correct size before importing the datatable.

	Parameters
	----------
	filepath : str or buffer
		This argument will be fed directly to the :func:`pandas.read_csv` function.
	one_based : bool
		If True (the default) it is assumed that zones are indexed sequentially starting with 1 
		(as is typical for travel demand forecasting applications).
		Otherwise, it is assumed that zones are indexed sequentially starting with 0 (typical for other c and python applications).
	chunksize : int
		The number of rows of the source file to read as a chunk.  Reading a giant file in moderate sized
		chunks can be much faster and less memory intensive than reading the entire file.
	column_map : dict or None
		If given, this dict maps columns of the input file to OMX tables, with the keys as
		the columns in the input and the values as the tables in the output.
	default_atom : str or dtype
		The default atomic type for imported data when the table does not already exist in this
		openmatrix.




.. automethod:: OMX.import_datatable_3d

	Import a table in r,c,x,x,x... format into the matrix.

	The r and c columns need to be either 0-based or 1-based index values
	(this may be relaxed in the future). The matrix must already be set up
	with the correct size before importing the datatable.

	This method is functionally the same as :meth:`import_datatable` but uses a different implementation.
	It is much more memory intensive but also much faster than the non-3d version.

	Parameters
	----------
	filepath : str or buffer
		This argument will be fed directly to the :func:`pandas.read_csv` function.
	one_based : bool
		If True (the default) it is assumed that zones are indexed sequentially starting with 1 
		(as is typical for travel demand forecasting applications).
		Otherwise, it is assumed that zones are indexed sequentially starting with 0 (typical for other c and python applications).
	chunksize : int
		The number of rows of the source file to read as a chunk.  Reading a giant file in moderate sized
		chunks can be much faster and less memory intensive than reading the entire file.
	default_atom : str or dtype
		The default atomic type for imported data when the table does not already exist in this
		openmatrix.





.. automethod:: OMX.import_datatable_as_lookups

	Import a table in r_or_c,x,x,x... format into the matrix.
	
	The r_or_c column needs to be either 0-based or 1-based index values
	(this may be relaxed in the future). The matrix must already be set up
	with the correct shape before importing the datatable.
	
	Parameters
	----------
	filepath : str or buffer
		This argument will be fed directly to the :func:`pandas.read_csv` function.
	chunksize : int
		The number of rows of the source file to read as a chunk.  Reading a giant file in moderate sized
		chunks can be much faster and less memory intensive than reading the entire file.
	column_map : dict or None
		If given, this dict maps columns of the input file to OMX tables, with the keys as
		the columns in the input and the values as the tables in the output.
	n_rows : int or None
		If given, this is the number of rows in the source file.  It can be omitted and will 
		be discovered automatically, but only for source files with consecutive zone numbering.
	zone_ix : str or None
		If given, this is the column name in the source file that gives the zone numbers.
	zone_ix1 : 1 or 0
		The smallest zone number.  Defaults to 1
	drop_zone : int or None
		If given, zones with this number (typically 0 or -1) will be ignored.




.. |idca| replace:: :ref:`idca <idca>`
.. |idco| replace:: :ref:`idco <idco>`




