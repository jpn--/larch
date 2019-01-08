.. currentmodule:: larch

=======================
Data Service
=======================

Larch (as of version 4) handles data through a :class:`DataService` object.

.. autoclass:: DataService

	**Read-Only Properties**

	.. autoattribute:: DataService.n_cases

	.. autoattribute:: DataService.n_alts

	.. autoattribute:: DataService.alternatives



Getting Data Arrays
-------------------

The usual method for external access to data is to call for an array. The `array_*`
methods assemble data into a new :class:`numpy.ndarray` and return that array directly.

.. automethod:: DataService.array_idco

.. automethod:: DataService.array_idca

Instead of creating a new array, it is also possible to [re]load data into an existing
array. The `load_*` methods take an existing array, and load the data into it.

.. automethod:: DataService.load_idco

.. automethod:: DataService.load_idca



