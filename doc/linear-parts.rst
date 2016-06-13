.. currentmodule:: larch.core


=============
Linear Parts
=============


Unlike other discrete choice tools (notably Biogeme), which allow for the
creation of a variety of arbitrary non-linear functions, Larch relies heavily
on linear functions.



.. autoclass:: LinearComponent(data="", param="", multiplier=1.0, category=None)


	.. py:attribute:: LinearComponent.data

		The data associated with this :class:`LinearComponent`, expressed as a
		:class:`.DataRef`.  You can assign a numerical value or a plain
		:class:`str` to this attribute as well.

	.. py:attribute:: LinearComponent.param

		The parameter associated with this :class:`LinearComponent`, expressed as a
		:class:`.ParameterRef`. You can assign a plain
		:class:`str` that names a parameter to this attribute as well.
		Parameter names are case-sensitive.


In addition to creating a :class:`LinearComponent` using the regular constructor,
you can also create these objects by multiplying a :class:`.ParameterRef` and a
:class:`.DataRef`.  For example:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> from larch.roles import P,X
	>>> P.TotalCost * X.totcost
	LinearComponent(data='totcost', param='TotalCost')





.. py:class:: LinearFunction

	This class is a specialize list of :class:`LinearComponent`, which are summed across
	during evaluation.


Instead of creating a :class:`LinearFunction` through a constructor, it is better to create one
simply by adding multiple :class:`LinearComponent` objects:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> from larch.roles import P,X
	>>> u1 = P.TotalCost * X.totcost
	>>> u2 = P.InVehTime * X.ivt
	>>> u1 + u2
	<LinearFunction with length 2>
	  = LinearComponent(data='totcost', param='TotalCost')
	  + LinearComponent(data='ivt', param='InVehTime')
	>>> lf = u1 + u2
	>>> lf += P.OutOfVehTime * X.ovt
	>>> lf
	<LinearFunction with length 3>
	  = LinearComponent(data='totcost', param='TotalCost')
	  + LinearComponent(data='ivt', param='InVehTime')
	  + LinearComponent(data='ovt', param='OutOfVehTime')

You can also add a :class:`.ParameterRef` by itself to a :class:`LinearFunction`,

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> lf += P.SomeConstant
	>>> lf
	<LinearFunction with length 4>
	  = LinearComponent(data='totcost', param='TotalCost')
	  + LinearComponent(data='ivt', param='InVehTime')
	  + LinearComponent(data='ovt', param='OutOfVehTime')
	  + LinearComponent(data='1', param='SomeConstant')

Although not just data by itself:


.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE, +IGNORE_EXCEPTION_DETAIL

	>>> lf += X.PlainData
	Traceback (most recent call last):
	  File "<stdin>", line 1, in <module>
	NotImplementedError: Wrong number or type of arguments...




.. py:class:: LinearBundle

	A LinearBundle represents a bundle of linear terms that, when combined,
	form a complete linear relationship from data to a modeled factor.

	.. py:attribute:: ca

		The `ca` attribute is a single :class:`LinearFunction` that can be
		applied for all alternatives.  Depending on the data source used,
		the data used in this function might need to be exclusively from the
		:ref:`idca` data (e.g. for :class:`DB`), or it could be a combination
		of :ref:`idca` and :ref:`idco` data (e.g. for :class:`DT`)

	.. py:attribute:: co

		The `co` attribute is a mapping (like a dict), where the keys are alternative codes
		and the values are :class:`LinearFunction` of :ref:`idco` data
		for each alternative.  If an alternative is omitted, the implied value
		of the :class:`LinearFunction` is zero.

	As a convenince, the :class:`LinearBundle` object also provides `__getitem__` and
	`__setitem__` functions that pass through to the `co` attribute.
