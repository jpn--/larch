.. currentmodule:: larch.core

=======================
Abstract Data Interface
=======================

Both :class:`DT` and :class:`DB` classes are derived from a common abstract base class,
which defines a handful of important interface functions.

.. py:class:: Fountain()

	This object represents a source of data. It is an abstract base class from which both
	the :class:`DT` and :class:`DB` classes are derived.


	.. automethod:: Fountain.alternative_names

	.. automethod:: Fountain.alternative_codes

	.. automethod:: Fountain.alternative_name

	.. automethod:: Fountain.alternative_code

		

	.. py:method:: Fountain.array_idco(*vars, dtype='float64')

		Extract an array of :ref:`idco` data from the underlying data source.
		The `vars` arguments define what data columns to extract, although
		the exact format and implementation is left to the base class.


	.. py:method:: Fountain.array_idca(*vars, dtype='float64')

		Extract an array of :ref:`idca` data from the underlying data source.
		The `vars` arguments define what data columns to extract, although
		the exact format and implementation is left to the base class.


	.. py:method:: Fountain.check_co(column)

		Validate whether `column` is a legitimate input for :meth:`Fountain.array_idco`.

	.. py:method:: Fountain.check_ca(column)

		Validate whether `column` is a legitimate input for :meth:`Fountain.array_idca`.

	.. py:method:: Fountain.variables_co()

		Return a list of the natural columns of :ref:`idco` data available.

	.. py:method:: Fountain.variables_ca()

		Return a list of the natural columns of :ref:`idca` data available.


	.. automethod:: Fountain.export_all

	.. automethod:: Fountain.dataframe_all

	.. automethod:: Fountain.dataframe_idco

	.. automethod:: Fountain.dataframe_idca


