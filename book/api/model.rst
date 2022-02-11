===========
Model
===========

.. currentmodule:: larch.numba


.. autosummary::
    :toctree: generated/

    Model


Attributes
==========

Data Connection
---------------

.. autosummary::
    :toctree: generated/

    Model.datatree
    Model.dataset
    Model.n_cases


Choice Definition
-----------------

.. autosummary::
    :toctree: generated/

    Model.choice_ca_var
    Model.choice_co_vars
    Model.choice_co_code


Alternative Availability
------------------------

.. autosummary::
    :toctree: generated/

    Model.availability_ca_var
    Model.availability_co_vars


Utility Definition
------------------

.. autosummary::
    :toctree: generated/

    Model.utility_ca
    Model.utility_co
    Model.quantity_ca


Parameters
----------

.. autosummary::
    :toctree: generated/

    Model.pf

Estimation Results
------------------

.. autosummary::
    :toctree: generated/

    Model.most_recent_estimation_result
    Model.possible_overspecification


Methods
=======

Setting Parameters
------------------

.. autosummary::
    :toctree: generated/

    Model.set_values
    Model.lock_value
    Model.set_cap
    Model.remove_unused_parameters


Parameter Estimation
--------------------

.. autosummary::
    :toctree: generated/

    Model.maximize_loglike
    Model.calculate_parameter_covariance


Model Fitness
-------------

.. autosummary::
    :toctree: generated/

    Model.loglike_nil
    Model.loglike_null
    Model.rho_sq_nil
    Model.rho_sq_null


Reporting
---------

.. autosummary::
    :toctree: generated/

    Model.parameter_summary
    Model.estimation_statistics
    Model.to_xlsx


Ancillary Computation
---------------------

.. autosummary::
    :toctree: generated/

    Model.bhhh
    Model.check_d_loglike
    Model.d_loglike
    Model.d_loglike_casewise
    Model.loglike
    Model.loglike_casewise
    Model.logsums
    Model.probability
    Model.quantity
    Model.total_weight
    Model.utility
