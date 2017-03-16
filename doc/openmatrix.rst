.. currentmodule:: larch

==================================
Using Open Matrix
==================================

Larch embeds a python interface for interacting with
`open matrix (OMX) <https://github.com/osPlanning/omx/wiki>`_ files.
This data format rests on HDF5, the exact same underlying technology used in
:class:`DT` files.  This makes merging and linking to open matrix data easy and
fun. :sup:`[citation needed]`

.. autoclass:: OMX(filename)

.. autoattribute:: OMX.shape

Importing Data
--------------

.. automethod:: OMX.import_datatable

.. automethod:: OMX.import_datatable_3d

.. automethod:: OMX.import_datatable_as_lookups


.. |idca| replace:: :ref:`idca <idca>`
.. |idco| replace:: :ref:`idco <idco>`




