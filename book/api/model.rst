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


Estimation Results
------------------

.. autosummary::
    :toctree: generated/

    Model.most_recent_estimation_result
    Model.possible_overspecification


Methods
=======

.. autosummary::
    :toctree: generated/

    Model.bhhh
    Model.check_d_loglike
    Model.d_loglike
    Model.d_loglike_casewise
    Model.estimation_statistics
    Model.loglike
    Model.loglike_casewise
    Model.loglike_nil
    Model.loglike_null
    Model.logsums
    Model.maximize_loglike
    Model.probability
    Model.quantity
    Model.rho_sq_nil
    Model.rho_sq_null
    Model.set_values
    Model.total_weight
    Model.to_xlsx
    Model.utility
