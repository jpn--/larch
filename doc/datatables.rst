.. currentmodule:: larch

==================================
Data Storage and Access Using HDF5
==================================

An alternative data storage system is available for Larch, relying on the HDF5 format
and the pytables package.
This system is made available through a :class:`DT` object, which wraps a
:class:`tables.File` object.


Creating :class:`DT` Objects
----------------------------

.. autoclass:: DT(filename, mode='a')

Similar to the :class:`DB` class, the :class:`DT` class can be used with
example data files.

.. automethod:: DT.Example



Importing Data
--------------

There are methods available to import data from external sources into
the correct format for use with the larch DT facility.

.. automethod:: DT.import_idco

.. automethod:: DT.import_idca




Required HDF5 Structure
-----------------------

To be used with Larch, the HDF5 file must have a particular structure::

	════════════════════════════════════════════════════════════════════════════════
	larch.DT Validation for MTC.h5 (with mode 'w')
	─────┬──────────────────────────────────────────────────────────────────────────
	 >>> │ There should be a designated `larch` group node under which all other
	     │ nodes reside.
	─────┼──────────────────────────────────────────────────────────────────────────
	     │ CASES
	 >>> │ Under the top node, there must be an array node named `caseids`.
	 >>> │ The `caseids` array dtype should be Int64.
	 >>> │ The `caseids` array should be 1 dimensional.
	     ├ Case Filtering ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
	 >>> │ If there may be some data cases that are not to be included in the
	     │ processing of the discrete choice model, there should be a node named
	     │ `screen` under the top node.
	 >>> │ If it exists `screen` must be a Bool array.
	 >>> │ And `screen` must be have the same shape as `caseids`.
	─────┼──────────────────────────────────────────────────────────────────────────
	     │ ALTERNATIVES
	 >>> │ Under the top node, there should be a group named `alts` to hold
	     │ alternative data.
	 >>> │ Within the `alts` node, there should be an array node named `altids` to
	     │ hold the identifying code numbers of the alternatives.
	 >>> │ The `altids` array dtype should be Int64.
	 >>> │ The `altids` array should be one dimensional.
	 >>> │ Within the `alts` node, there should also be a VLArray node named `names`
	     │ to hold the names of the alternatives.
	 >>> │ The `names` node should hold unicode values.
	 >>> │ The `altids` and `names` arrays should be the same length, and this will
	     │ be the number of elemental alternatives represented in the data.
	─────┼──────────────────────────────────────────────────────────────────────────
	     │ IDCO FORMAT DATA
	 >>> │ Under the top node, there should be a group named `idco` to hold that
	     │ data.
	 >>> │ Every child node name in `idco` must be a valid Python identifer (i.e.
	     │ starts with a letter or underscore, and only contains letters, numbers,
	     │ and underscores) and not a Python reserved keyword.
	 >>> │ Every child node in `idco` must be (1) an array node with shape the same
	     │ as `caseids`, or (2) a group node with child nodes `_index_` as an array
	     │ with the correct shape and an integer dtype, and `_values_` such that
	     │ _values_[_index_] reconstructs the desired data array.
	     ├ Case Weights ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
	 >>> │ If the cases are to have non uniform weights, then there should a
	     │ `_weight_` node (or a name link to a node) within the `idco` group.
	 >>> │ If weights are given, they should be of Float64 dtype.
	─────┼──────────────────────────────────────────────────────────────────────────
	     │ IDCA FORMAT DATA
	 >>> │ Under the top node, there should be a group named `idca` to hold that
	     │ data.
	 >>> │ Every child node name in `idca` must be a valid Python identifer (i.e.
	     │ starts with a letter or underscore, and only contains letters, numbers,
	     │ and underscores) and not a Python reserved keyword.
	 >>> │ Every child node in `idca` must be an array node with the first dimension
	     │ the same as the length of `caseids`, and the second dimension the same as
	     │ the length of `altids`.
	     ├ Alternative Availability ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
	 >>> │ If there may be some alternatives that are unavailable in some cases,
	     │ there should be a node named `_avail_` under `idca`.
	 >>> │ If given, it should contain an appropriately sized Bool array indicating
	     │ the availability status for each alternative.
	     ├ Chosen Alternatives ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
	 >>> │ There should be a node named `_choice_` under `idca`.
	 >>> │ It should be a Float64 array indicating the chosen-ness for each
	     │ alternative. Typically, this will take a value of 1.0 for the alternative
	     │ that is chosen and 0.0 otherwise, although it is possible to have other
	     │ values, including non-integer values, in some applications.
	─────┼──────────────────────────────────────────────────────────────────────────
	     │ OTHER TECHNICAL DETAILS
	 >>> │ The set of child node names within `idca` and `idco` should not overlap
	     │ (i.e. there should be no node names that appear in both).
	═════╧══════════════════════════════════════════════════════════════════════════


To check if your file has the correct structure, you can use the validate function:

.. automethod:: DT.validate
