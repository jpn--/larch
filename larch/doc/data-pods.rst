.. currentmodule:: larch.data_services

=======================
Pods
=======================

A :class:`DataService` is essentially a collection of one or more separate but related
data :class:`Pod`.  Each pod can contain multiple data elements, stored in a variety of
formats. :class:`Pod` are roughly grouped into two families: those that provide :ref:`idco`
data, and those that provide :ref:`idca` data.  The abstract base class :class:`Pod`
provides some basic common functionality and defines a consistent interface.

.. autoclass:: Pod

	.. automethod:: get_data_item

	.. automethod:: load_data_item

	.. automethod:: names

	.. automethod:: nameset

	.. autoattribute:: shape

	.. autoattribute:: dims

	.. autoattribute:: n_cases

	.. autoattribute:: ident

	.. autoattribute:: durable_mask

	.. autoattribute:: filename



Creating a :class:`Pod` requires a particular subclass to instantiate.


:ref:`idco` Pods
----------------

.. autoclass:: H5PodCO

.. autoclass:: H5PodCS

.. autoclass:: H5PodRC



:ref:`idca` Pods
----------------

.. autoclass:: H5PodCA

.. autoclass:: H5PodGA

