.. currentmodule:: larch

=======================
Logit Models
=======================

The basic tool for analysis in Larch is a discrete choice model.  A model is a
structure that interacts data with a set of :class:`Parameter`\ s.

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





Using :class:`Model` Objects
-------------------------------

.. py:method:: Model.estimate()

	Find the likelihood maximizing parameters of the model.


.. py:method:: Model.loglike([values])

	Find the log likelihood of the model.

	:param values: If given, an array-like vector of values should be provided that
                   will replace the current parameter values.  The vector must be exactly
                   as long as the number of parameters in the model (including holdfast
                   parameters).  If any holdfast parameter values differ in the provided
                   `values`, the new values are ignored and a warning is emitted to the
                   model logger.
	:type values:  array-like, optional



GEV Network
-----------

Nested logit and Network GEV models have an underlying network structure.

.. autoattribute:: Model.nest(id, name=None, parameter=None)

.. autoattribute:: Model.node

.. autoattribute:: Model.link(up_id, down_id)

.. autoattribute:: Model.edge

.. autoattribute:: Model.root_id



