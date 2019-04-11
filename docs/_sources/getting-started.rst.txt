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

If you've got conda already installed, to make sure that conda is up-to-date, you
can use its self-updating super-power::

	conda update -n base conda

And install the anaconda client too, which will let you directly install environments
from anaconda.org::

	conda install -n base anaconda-client

Once you've installed Anaconda, to get Larch you can simply run::

	conda install larch -c jpn

This will install Larch in the active environment, which will be the "base" environment
if you have not set up another environment to work in. If you'd like to install
a suite of related tools relevant for transportation planning and discrete choice
analysis, you can create a new environment for Larch::

	conda env create jpn/taiga

If you've already installed the *taiga* environment and want to update it to the latest
version, you can use::

	conda env update jpn/taiga --prune

The *prune* option here will remove packages that are not ordinarily included in the
*taiga* environment; omit that function if you've installed extra packages that you
want to keep.

Once you've got Larch installed, you might want to jump directly to some :ref:`examples`
to see how you might use it.




