.. currentmodule:: larch

=======================
DataFrames
=======================

A :class:`DataFrames` is essentially a collection of related :class:`pandas.DataFrame`\ s,
which represent :ref:`idco` and :ref:`idca` data features.

.. autoclass:: DataFrames

    .. automethod:: alternative_codes

    .. automethod:: alternative_names

    .. automethod:: set_alternative_names

    **Attributes**

    .. autoattribute:: data_co

    .. autoattribute:: data_ca

    **Read-only Attributes**

    .. autoattribute:: n_alts

    .. autoattribute:: n_cases

    .. autoattribute:: caseindex


.. |idca| replace:: :ref:`idca <idca>`
.. |idco| replace:: :ref:`idco <idco>`


