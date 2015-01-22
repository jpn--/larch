.. currentmodule:: larch.core

======================================
Queries for a Pair of Tables
======================================

.. class:: QuerySetTwoTable

	This subclass of :class:`core.QuerySet` is used when the data consists of one
	:ref:`idco` format table and one :ref:`idca` format table.


	.. autoattribute:: idco_query

	.. autoattribute:: idca_query

	.. autoattribute:: alts_query


	.. autoattribute:: choice

	.. automethod:: set_choice_ca(expr)

	.. automethod:: set_choice_co(expr)

	.. autoattribute:: avail

	.. autoattribute:: weight












