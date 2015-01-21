.. currentmodule:: larch

=======================
Logit Models
=======================

The basic tool for analysis in Larch is a discrete choice model.

Creating :class:`Model` Objects
-------------------------------

.. py:class:: Model(db=None)
	
	:param db: The source database used to automatically populate model arrays.
	:type db:  :class:`DB`

	This object represents a discrete choice model. In addition to the methods
	described below, a :class:`Model` also acts like a list of :class:`Parameter`.


.. py:method:: Model.Example(number=1)

	Generate an example model object.

	:param number: The code number of the example model to load. Valid numbers
	               include {1,101,102,104,109,114}.
	:type number:  int

	Larch comes with a few example models, which are used in documentation
	and testing.  Models with numbers greater than 100 are designed to align with
	the `example models given for Biogeme <http://biogeme.epfl.ch/swissmetro/examples.html>`_.

.. py:method:: Model.estimate()

	Find the likelihood maximizing parameters of the model.



Model Parameters
----------------


.. autoclass:: Parameter(name='', value=0, null_value=0, holdfast=False)




GEV Network
-----------

Nested logit and Network GEV models have an underlying network structure.

.. autoattribute:: Model.nest(id, name=None, parameter=None)

.. autoattribute:: Model.node

.. autoattribute:: Model.link(up_id, down_id)

.. autoattribute:: Model.edge

.. autoattribute:: Model.root_id

