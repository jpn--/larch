.. currentmodule:: larch

=======================
Model
=======================

A :class:`Model` is the core object used to represent a discrete choice model.

.. autoclass:: Model


Utility Function Definition
---------------------------

Note that these function definitions act like *properties* of
the Model object, instead of methods on Model objects.

.. autoattribute:: Model.utility_ca

.. autoattribute:: Model.utility_co

.. autoattribute:: Model.quantity_ca


Parameter Manipulation
----------------------

.. automethod:: Model.set_value

.. automethod:: Model.lock_value


Scikit-Learn Interface
----------------------

.. automethod:: Model.fit

.. automethod:: Model.predict

.. automethod:: Model.predict_proba


Reporting and Outputs
---------------------

.. automethod:: Model.parameter_summary

.. automethod:: Model.utility_functions

.. automethod:: Model.estimation_statistics


Visualization Tools
-------------------

.. automethod:: Model.distribution_on_continuous_idca_variable