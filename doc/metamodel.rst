.. currentmodule:: larch

=======================
MetaModels
=======================

A :class:`MetaModel` is an object that encapsulates a collection of :class:`Model` objects, for which the parameters
are to be estimated simultaneously.

The basic tool for analysis in Larch is a discrete choice model.  A model is a
structure that interacts data with a set of :class:`Parameter`\ s.

Creating :class:`MetaModel` Objects
-----------------------------------

.. autoclass:: MetaModel(segment_descriptors=None, submodel_factory=None, args=())
	


The :class:`MetaModel` is actually a subclass of :class:`Model`, so much of that class's functionality
(notably, interacting with parameters) is inherited here.  The :class:`MetaModel` overloads loglike,
d_loglike, d2_loglike, and bhhh methods of :class:`Model` in the manner you might expect: the overloaded
functions return the composite values, totalled across all submodels.



