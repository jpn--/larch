.. larch documentation getting started

===============
Getting Started
===============

.. _installation:

Installation
------------

To run Larch, you'll need to have the 64 bit version of Python 3.5, plus a handful
of other useful statistical packages for Python.  The easiest way to get everything
you need is to download and install the `Anaconda <http://www.continuum.io/downloads>`_
version of Python 3.5 (64 bit). This comes with everything you'll need to get started,
and the Anaconda folks have helpfully curated a selection of useful tools for you,
so you don't have the sort through the huge pile of junk that is available for Python.

Once you've installed Anaconda, to get Larch you can simply run::

	pip install larch

from your command prompt (Windows) or the Terminal (Mac OS X). It's possible that you may
get some kind of a permission error when running this command.  If so, try it again
as an admin (on windows, right click the command line program and choose "Run as Administrator").

Some of the graphical tools used to draw nested and network logit graphs may also not
be installed by default by Anaconda.  You don't need these tools to run any model in
Larch, just to draw pretty figures depicting the nests.  If you want to install these,
you can go to your command line to get the necessary tools::

	conda install graphviz
	pip install pygraphviz






