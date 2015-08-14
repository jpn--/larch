.. currentmodule:: larch.roles

.. testsetup:: *

   import larch

=======================
Roles
=======================

In some situations (particularly when dealing with multiple related models)
it can be advantageous to access model parameters and data via roles.
These are special references that are able to pull values from models, or
assume default values, without actually changing the models.


--------------------
Parameter References
--------------------

.. autoclass:: ParameterRef(name, default=None, fmt=None)

After a :class:`ParameterRef` object is created, its attributes can be
modified using these methods:

.. automethod:: ParameterRef.name

.. automethod:: ParameterRef.default_value

.. automethod:: ParameterRef.fmt

You can access a referenced parameter from a model using these methods:

.. automethod:: ParameterRef.value

.. automethod:: ParameterRef.valid

.. automethod:: ParameterRef.str


Math using Parameter References
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can do some simple math with :class:`ParameterRef` objects.  Currently supported
are addition, subtraction, multiplication, and division.  This allows, for example,
the automatic calculation of values of time for a model:

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> m = larch.Model.Example(1, pre=True)
	>>> from larch.roles import P
	>>> VoT_cents_per_minute = P.tottime / P.totcost
	>>> VoT_cents_per_minute.value(m)
	10.43427702270004
	>>> print("The implied value of time is", VoT_cents_per_minute.str(m, fmt="{:.1f}¢/minute"))
	The implied value of time is 10.4¢/minute
	>>> VoT_dollars_per_hour = (P.tottime * 60) / (P.totcost * 100)
	>>> print("The implied value of time is", VoT_dollars_per_hour.str(m, fmt="${:.2f}/hr"))
	The implied value of time is $6.26/hr




--------------------
Data References
--------------------

.. autoclass:: DataRef




----------------------------
Combinations of References
----------------------------

.. autoclass:: LinearComponent(data="", param="", multiplier=1.0, category=None)



.. autoclass:: LinearFunction
