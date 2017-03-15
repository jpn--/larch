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




.. automethod:: OMX.import_datatable_3d

	Import a table in r,c,x,x,x... format into the matrix.

	The r and c columns need to be either 0-based or 1-based index values
	(this may be relaxed in the future). The matrix must already be set up
	with the correct size before importing the datatable.

	This method is functionally the same as :meth:`import_datatable` but uses a different implementation.
	It is much more memory intensive but also much faster than the non-3d version.




.. automethod:: OMX.import_datatable_as_lookups

	Import a table in r_or_c,x,x,x... format into the matrix.
	
	The r_or_c column needs to be either 0-based or 1-based index values
	(this may be relaxed in the future). The matrix must already be set up
	with the correct shape before importing the datatable.
	




.. |idca| replace:: :ref:`idca <idca>`
.. |idco| replace:: :ref:`idco <idco>`




