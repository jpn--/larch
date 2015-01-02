.. larch documentation master file

==============================
|treelogo| Larch Documentation
==============================

.. centered:: Larch |release|

.. |treelogo| image:: ../img/larch_favicon.png


This documentation is for the Python interface for Larch.

This project is very much under development.  There are plenty of undocumented functions
and features; use them at your own risk.  Undocumented features may be non-functional, 
not rigorously tested, deprecated or removed without notice in a future version.  If a
function or method is documented here, it is intended to be stable in future updates.

You may also find these links useful:

* `Python <http://www.python.org/>`_ 3.4: http://docs.python.org/3.4/index.html
* `NumPy <http://www.numpy.org>`_ |numpy_version|: http://docs.scipy.org/doc/numpy/
* `SQLite <http://www.sqlite.org/>`_ |sqlite_version|: http://www.sqlite.org/docs.html
* `APSW <https://github.com/rogerbinns/apsw>`_ |apsw_version|: http://rogerbinns.github.io/apsw/


Contents
========

.. toctree::
   :maxdepth: 4

   data
   math



Frequently Asked Questions
==========================

Why should I use Larch instead of `Biogeme <http://biogeme.epfl.ch/>`_ to estimate my logit models?

	`Biogeme <http://biogeme.epfl.ch/>`_ is a different free software package that also estimates
	parameters for logit models. If you know it, and how to use it, awesome!  If you don't, it's really not
	that hard to learn and in many ways quite similar to *Larch*. Depending on your particular application
	and needs, (especially if you want to explore more complex non-linear models) then Biogeme could be right
	for you.

	The principal systematic differences between Biogeme and Larch are in data and numerical processing:
	Larch uses NumPy to manage arrays, and invokes highly optimized linear algebra subroutines to
	compute certain matrix transformations. There is a loss in model flexibility (non-linear-in-parameters
	models are not supported in Larch) but a potentially significant gain in speed.

Did this software used to be called *ELM*?  Why the name change?

	Yes, earlier versions of this software were called ELM.  But it turns out ELM is a pretty common
	name for software.  Larch is not unique, but much less commonly used.  Also, it is the tradition of
	Python software to use names from Monty Python, particularly when you want to be able to identify
	your software `from quite a long way away <https://www.youtube.com/watch?v=ug8nHaelWtc>`_.




