

.. currentmodule:: larch.model.linear

=======================
Linear Functions
=======================

At the heart of most common discrete choice models is a linear-in-parameters utility
function.  Larch is written with this design specifically in mind.
It is designed to integrate with Panda and NumPy and facilitate fast processing of linear models.

.. note::

    If you want to estimate *non-linear* models, try `Biogeme <http://biogeme.epfl.ch/>`_,
    which is more flexible in form and can be used for almost any model structure.

The basic structure of any linear-in-parameters function is that it is a summation of a sequence
of terms, where each term is the product of a parameter and some data.  The data could be
a simple single named value (e.g. ``travel_cost``), or it could be some function of one or more other pieces of data,
but the important salient feature of data is that it can be computed without knowing the
value of any model parameter to be estimated (e.g. ``log(1+travel_cost)``).


.. autoclass:: UnicodeRef_C
    :show-inheritance:
    :members:


Parameters
----------

.. autoclass:: ParameterRef_C
    :show-inheritance:
    :members:

Data
----

.. autoclass:: DataRef_C
    :show-inheritance:
    :members:


