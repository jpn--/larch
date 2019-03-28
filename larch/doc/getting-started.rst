.. larch documentation getting started

===============
Getting Started
===============

.. _installation:

Installation
------------

To run Larch, you'll need to have the 64 bit version of Python 3.7, plus a handful
of other useful statistical packages for Python.  The easiest way to get the basics
is to download and install the `Anaconda <https://www.anaconda.com/download>`_
version of Python 3.7 (64 bit). This comes with everything you'll need to get started,
and the Anaconda folks have helpfully curated a selection of useful tools for you,
so you don't have the sort through the huge pile of junk that is available for Python.

.. note::

	Python has two versions (2 and 3) that are available and currently maintained.
	Larch is compatible *only* with version 3.

Once you've installed Anaconda, to get Larch you can simply run::

	conda install larch -c jpn

This will install Larch in the active environment, which will be the "base" environment
if you have not set up another environment to work in. If you'd like to install
a suite of related tools relevant for transportation planning and discrete choice
analysis, you can create a new environment for Larch::

	conda env create jpn/taiga

Once you've got Larch installed, you might want to jump directly to some :ref:`examples`
to see how you might use it.




