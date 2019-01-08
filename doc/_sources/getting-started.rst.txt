.. larch documentation getting started

===============
Getting Started
===============

.. _installation:

Installation
------------

To run Larch, you'll need to have the 64 bit version of Python 3.7, plus a handful
of other useful statistical packages for Python.  The easiest way to get everything
you need is to download and install the `Anaconda <http://www.continuum.io/downloads>`_
version of Python 3.7 (64 bit). This comes with everything you'll need to get started,
and the Anaconda folks have helpfully curated a selection of useful tools for you,
so you don't have the sort through the huge pile of junk that is available for Python.

.. note::

	Python has two versions (2 and 3) that are available and currently maintained.
	Larch is compatible *only* with version 3.

Because Larch will be using some packages from conda-forge, you will probably want to
do this, which will keep the main conda system from flopping back and forth between the
default channel and the conda-forge channel::

	conda config --system --add pinned_packages defaults::conda
	conda update conda

Once you've installed Anaconda, to get Larch you can simply run::

	conda config --append channels conda-forge
	conda install larch -c jpn

from your command prompt (Windows) or the Terminal (Mac OS X). It's possible that you may
get some kind of a permission error when running this command.  If so, try it again
as an admin (on windows, right click the command line program and choose "Run as Administrator").



Once you've got Larch installed, you might want to jump directly to some :ref:`examples`
to see how you might use it.




