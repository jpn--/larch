.. currentmodule:: larch

=======================
Data in Models
=======================

Larch offers two basic data file storage formats: SQLite and HDF5.

If you have experience with earlier version Larch (or its predecessor, ELM) then you have
been using the SQLite database interface. 

.. toctree::

	Using SQLite <databases>
	Using HDF5 <datatables>



The Abstract :class:`Fountain` Base Class
-----------------------------------------

.. py:class:: Fountain()

	This object represents a source of data. It is an abstract base class from which both
	the :class:`DT` and :class:`DB` classes are derived.
