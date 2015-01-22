.. currentmodule:: larch.core

======================================
Queries for a Single :ref:`idco` Table
======================================

.. class:: QuerySetSimpleCO

	This subclass of :class:`core.QuerySet` is used when the data consists exclusively of a single
	:ref:`idco` format table.

	.. note::

		This is similar to the data format required by Biogeme. Unlike Biogeme, the Larch
		:class:`DB` allows non-numeric values in the source data.


	.. autoattribute:: idco_query

	.. autoattribute:: alts_query

	.. autoattribute:: alts_values

	.. autoattribute:: choice

	.. autoattribute:: avail

	.. autoattribute:: weight












