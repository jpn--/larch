.. currentmodule:: larch

=======================
DataService
=======================

Larch has two closely related interfaces to manage data: :class:`DataFrames` and
:class:`DataService`.  The former is a set of related data tables, while the
latter is an interface that generates instances of the former.  You can think
of a :class:`DataService` as a place to get data, and :class:`DataFrames` as the
data you get.

In fact, :class:`DataFrames` is itself a subclass of :class:`DataService`,
allowing you to pull subsets of the data you have stored, so you can use
:class:`DataFrames` in both places.

A Larch Model object will generally have a :class:`DataService` attached
by the user, and then it will use that :class:`DataService` to automatically
generate, format, and pre-process the particular :class:`DataFrames` it needs
for analysis.


.. py:class:: DataService

    An object that implements the :class:`DataService` interface must provide these
    methods.

    .. py:method:: make_dataframes(req_data, *, selector=None, float_dtype=numpy.float64)

        Create a DataFrames object that will satisfy a data request.

        :param req_data:
            The requested data. The keys for this dictionary may include {'ca', 'co',
            'choice_ca', 'choice_co', 'weight_co', 'avail_ca', 'standardize'}.
            Currently, the keys {'choice_co_code', 'avail_co'} are not implemented and
            will raise an error.
            Other keys are silently ignored.
        :type req_data: Dict or str
        :param selector:
            If given, the selector filters the cases. This argument can only be given
            as a keyword argument.
        :type selector: array-like[bool] or slice, optional
        :param float_dtype:
            The dtype to use for all float-type arrays.  Note that the availability
            arrays are always returned as int8 regardless of the float type.
            This argument can only be given
            as a keyword argument.
        :type float_dtype: dtype, default float64
        :rtype: DataFrames
        :return: This object should satisfy the request.

