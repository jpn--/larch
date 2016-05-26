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

To be used with Larch, the HDF5 file must have a particular structure.  The group node structure is
created automatically when you open a new :class:`DT` object with a file that does not already have
the necessary structure.

.. digraph:: Required_HDF5_Structure

	larch [label="larch", shape="box"];
	caseids [label="caseids\n shape=(N) ", shape="box", color="#DD0000", style="rounded", penwidth=2];
	larch -> caseids;
	screen [label="screen\n shape=(N) ", shape="box", color="#01BB00", style="rounded,dashed", penwidth=2];
	larch -> screen;
	idco [label="idco", shape="box"];
	idco3 [label="...various...\n shape=(N) ", shape="box", style="rounded", penwidth=2];
	idco2 [label="...various...", shape="box"];
	idco2i [label="_index_\n shape=(N)", style="rounded", penwidth=2, color="#DD0000", shape="box"];
	idco2v [label="_values_\n shape=(?)", style="rounded", penwidth=2, shape="box"];
	idco2 -> idco2i;
	idco2 -> idco2v;
	idco -> idco2;
	wgt [label="_weight_\n shape=(N) ", shape="box", style="rounded,dashed", penwidth=2, color="#0000EE"];
	larch -> idco [minlen=2];
	idco -> idco3;
	idco -> wgt;
	idca [label="idca", shape="box"];
	idca3 [label="...various...\n shape=(N,A) ", shape="box", style="rounded", penwidth=2];
	idca2 [label="...various...", shape="box"];
	idca2i [label="_index_\n shape=(N)", style="rounded", penwidth=2, color="#DD0000", shape="box"];
	idca2v [label="_values_\n shape=(?,A)", style="rounded", penwidth=2, shape="box"];
	idca2 -> idca2i;
	idca2 -> idca2v;
	choice [label="_choice_\n shape=(N,A) ", shape="box", style="rounded", color="#0000EE", penwidth=2];
	avail [label="_avail_\n shape=(N,A) ", shape="box", style="rounded,dashed", color="#01BB00", penwidth=2];
	larch -> idca [minlen=2];
	idca -> idca3;
	idca -> idca2;
	idca -> choice;
	idca -> avail;
	larch -> alts;
	alts [label="alts", shape="box"];
	altids [label="altids\n shape=(A) ", shape="box", color="#DD0000", style="rounded", penwidth=2];
	names [label="names\n shape=(A) ", shape="box", color="#AAAA00", style="rounded", penwidth=2];
	alts -> altids;
	alts -> names;


.. digraph:: Required_HDF5_Structure_Legend

	subgraph clusterlegend {
		rank="same";
		shape="box";
		style="filled,rounded";
		color="#EEEEEE";
		label="Legend";
		int64 [label="dtype=Int64", color="#DD0000", shape="box", style="rounded", penwidth=2];
		float64 [label="dtype=Float64", color="#0000EE", shape="box", style="rounded", penwidth=2];
		unicode [label="dtype=Unicode", color="#AAAA00", shape="box", style="rounded", penwidth=2];
		bool [label="dtype=Bool", color="#01BB00", shape="box", style="rounded", penwidth=2];
		optional [label="optional", shape="box", style="rounded,dashed", penwidth=2];
		Group_Node [label="Group Node", shape="box", rank="sink"];
		Array_Node [label="Data Node", shape="box", style="rounded", penwidth=2];
	};



The details are as follows::

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
	     ├ Case Filtering ──────────────────────────────────────────────────────────
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
	     ├ Case Weights ────────────────────────────────────────────────────────────
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
	 >>> │ Every child node in `idca` must be (1) an array node with the first
	     │ dimension the same as the length of `caseids`, and the second dimension
	     │ the same as the length of `altids`, or (2) a group node with child nodes
	     │ `_index_` as a 1-dimensional array with the same length as the length of
	     │ `caseids` and an integer dtype, and a 2-dimensional `_values_` with the
	     │ second dimension the same as the length of `altids`, such that
	     │ _values_[_index_] reconstructs the desired data array.
		 ├ Alternative Availability ────────────────────────────────────────────────
	 >>> │ If there may be some alternatives that are unavailable in some cases,
	     │ there should be a node named `_avail_` under `idca`.
	 >>> │ If given as an array, it should contain an appropriately sized Bool array
	     │ indicating the availability status for each alternative.
	 >>> │ If given as a group, it should have an attribute named `stack` that is a
	     │ tuple of `idco` expressions indicating the availability status for each
	     │ alternative. The length and order of `stack` should match that of the
	     │ altid array.
		 ├ Chosen Alternatives ────────────────────────────────────────────────────
	 >>> │ There should be a node named `_choice_` under `idca`.
	 >>> │ If given as an array, it should be a Float64 array indicating the chosen-
	     │ ness for each alternative. Typically, this will take a value of 1.0 for
	     │ the alternative that is chosen and 0.0 otherwise, although it is possible
	     │ to have other values, including non-integer values, in some applications.
	 >>> │ If given as a group, it should have an attribute named `stack` that is a
	     │ tuple of `idco` expressions indicating the choice status for each
	     │ alternative. The length and order of `stack` should match that of the
	     │ altid array.
	─────┼──────────────────────────────────────────────────────────────────────────
	     │ OTHER TECHNICAL DETAILS
	 >>> │ The set of child node names within `idca` and `idco` should not overlap
	     │ (i.e. there should be no node names that appear in both).
	═════╧══════════════════════════════════════════════════════════════════════════

Note that the _choice_ and _avail_ nodes are special, they can be expressed as a
stack if idco expressions instead of as a single idca array.  To do so, replace the
array node with a group node, and attach a `stack` attribute that gives the list of
`idco` expressions.  The list should match the list of alternatives.  One way to do
this automatically is to use the avail_idco and choice_idco attributes of the
:class:`DT`.

To check if your file has the correct structure, you can use the validate function:

.. automethod:: DT.validate



.. autoattribute:: DT.choice_idco

.. autoattribute:: DT.avail_idco


