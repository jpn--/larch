.. currentmodule:: larch

=======================
Data Fundamentals
=======================

Larch requires data to be structured in one of two formats: the case-only ("idco")
format or the case-alternative ("idca") format. This are commonly referred to as
IDCase (each record contains all the information for mode choice over
alternatives for a single trip) or IDCase-IDAlt (each record contains all the
information for a single alternative available to each decision maker so there is one
record for each alternative for each choice).


.. _idco:

idco Format
-----------

	In the **idco** case-only format, each record provides all the relevant information
	about an individual choice, including the variables related to the decision maker
	or the choice itself, as well as alternative related variables for all available
	alternatives and a variable indicating which alternative was chosen.

	.. table:: Example of data in idco format

		====== ====== ========== ========== ========== ========== ========== ========== ==========
		caseid Income Alt 1 Time Alt 1 Cost Alt 2 Time Alt 2 Cost Alt 3 Time Alt 3 Cost Chosen Alt
		====== ====== ========== ========== ========== ========== ========== ========== ==========
		1      30,000 30         150        40         100        20         200        1
		2      30,000 25         125        35         100        0          0          2
		3      40,000 40         125        50         75         30         175        3
		4      50,000 15         225        20         150        10         250        3
		====== ====== ========== ========== ========== ========== ========== ========== ==========





.. _idca:

idca Format
-----------

	In the **idca** case-alternative format, each record can include information on the variables
	related to the decision maker or the choice itself, the attributes of that
	particular alternative, and a choice variable that indicates whether the
	alternative was or was not chosen.

	.. table:: Example of data in idca format

		====== ========== ============== ====== ==== ==== ======
		caseid Alt Number Number Of Alts Income Time Cost Chosen                                      
		====== ========== ============== ====== ==== ==== ======
		1      1          3              30,000 30   150  1
		1      2          3              30,000 40   100  0
		1      3          3              30,000 20   200  0
		2      1          2              30,000 25   125  0
		2      2          2              30,000 35   100  1
		3      1          3              40,000 40   125  0
		3      2          3              40,000 50   75   0
		3      3          3              40,000 30   175  1
		4      1          3              50,000 15   225  0
		4      2          3              50,000 20   150  0
		4      3          3              50,000 10   250  1
		====== ========== ============== ====== ==== ==== ======






Unlike most other tools for discrete choice analysis, Larch does not demand you
employ one or the other of these data formats.  You can use either, or both
simultaneously.





Using Data in Models
--------------------

The :class:`DB` class primarily presents an interface between python and SQLite. The interface
between a :class:`DB` and a :class:`Model` is governed by a special attribute of the
:class:`DB` class:

.. attribute:: DB.queries

	This attribute defines the automatic queries used to provision a :class:`Model` with data.
	It should be an object that is a specialized subtype of the :class:`core.QuerySet` abstract
	base class.


.. toctree::

	queryset
	queryset_idco
	queryset_2



