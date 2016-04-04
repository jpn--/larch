.. currentmodule:: larch

=======================
Logit Models
=======================

The basic tool for analysis in Larch is a discrete choice model.  A model is a
structure that interacts data with a set of :class:`Parameter`\ s.

Creating :class:`Model` Objects
-------------------------------

.. py:class:: Model([d])
	
	:param d: The source data used to automatically populate model arrays. This can
	          be either a :class:`DB` or :class:`DT` object (or another data provider
	          that inherits from the abstract :class:`Fountain` class). This parameter
	          can be omitted, in which case data will not be loaded automatically and
	          validation checks will not be performed when specifying data elements of
	          the model.
	:type d:  :class:`Fountain`

	This object represents a discrete choice model. In addition to the methods
	described below, a :class:`Model` also acts like a list of :class:`Parameter`.


.. py:method:: Model.Example(number=1)

	Generate an example model object.

	:param number: The code number of the example model to load. Valid numbers
	               include {1,17,22,101,102,104,109,111,114}.
	:type number:  int

	Larch comes with a few example models, which are used in documentation
	and testing.  Models with numbers greater than 100 are designed to align with
	the `example models given for Biogeme <http://biogeme.epfl.ch/swissmetro/examples.html>`_.





Using :class:`Model` Objects
-------------------------------

.. py:method:: Model.maximize_loglike()

	Find the likelihood maximizing parameters of the model, using the scipy.optimize module.
	Depending on the model type and structure, various different optimization algorithms
	may be used.

	
.. automethod:: Model.roll


.. py:method:: Model.estimate()

	Find the likelihood maximizing parameters of the model using deprecated Larch optimization
	engine.  This engine has fewer algorithms available than the scipy.optimize and may perform
	poorly for some model types, particularly cross-nested and network GEV models.  Users should
	almost always prefer the :meth:`Model.maximize_loglike` function instead.




.. py:method:: Model.loglike([values])

	Find the log likelihood of the model.

	:param values: If given, an array-like vector of values should be provided that
                   will replace the current parameter values.  The vector must be exactly
                   as long as the number of parameters in the model (including holdfast
                   parameters).  If any holdfast parameter values differ in the provided
                   `values`, the new values are ignored and a warning is emitted to the
                   model logger.
	:type values:  array-like, optional


.. automethod:: Model.d_loglike([values])



GEV Network
-----------

Nested logit and Network GEV models have an underlying network structure.

.. autoattribute:: Model.nest(id, name=None, parameter=None)

.. autoattribute:: Model.node

.. automethod:: Model.new_nest

.. py:method:: Model.new_node

	an alias for :meth:`new_nest`

.. autoattribute:: Model.link(up_id, down_id)

.. autoattribute:: Model.edge

.. autoattribute:: Model.root_id



Descriptives
------------

.. py:attribute:: Model.title

	The is a descriptive title to attach to this model.  It is used in certain reports,
	and can be set to any string.  It has no bearing on the numerical representation of
	the model.



