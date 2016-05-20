.. currentmodule:: larch

=======================
Model Parameters
=======================


.. py:class:: ModelParameter(model, index)

	A ModelParameter is a reference object, referring to a :class:`Model` and
	a parameter index.  From this information, a number of parameter atrributes
	can be accessed and edited.

	.. autoattribute:: name
	.. autoattribute:: index
	.. autoattribute:: value
	.. autoattribute:: null_value
	.. autoattribute:: initial_value
	.. autoattribute:: min_value
	.. autoattribute:: max_value
	.. autoattribute:: holdfast
	.. autoattribute:: t_stat
	.. autoattribute:: std_err
	.. autoattribute:: robust_std_err
	.. autoattribute:: covariance
	.. autoattribute:: robust_covariance


.. py:class:: ParameterManager

	The ParameterManager class provides the interface to interact with
	various model parameters.  You can call a ParameterManager like a
	mathod, to add a new parameter to the model or to access an existing
	parameter.  You can also use it with square brackets, to get and
	set ModelParameter items.

	When called as a method, in addition to the required parameter name, you can
	specify other :class:`ModelParameter` attributes as keyword arguments.

	When getting or setting items (with square brackets) you can give the
	parameter name or integer index.

	See the :class:`Model` section for examples.


