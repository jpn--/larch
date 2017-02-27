.. larch documentation getting started

===============
Getting Started
===============

.. _installation:

Installation
------------

To run Larch, you'll need to have the 64 bit version of Python 3.5 or 3.6, plus a handful
of other useful statistical packages for Python.  The easiest way to get everything
you need is to download and install the `Anaconda <http://www.continuum.io/downloads>`_
version of Python 3.6 (64 bit). This comes with everything you'll need to get started,
and the Anaconda folks have helpfully curated a selection of useful tools for you,
so you don't have the sort through the huge pile of junk that is available for Python.

.. note::

	Python has two versions (2 and 3) that are available and currently maintained.
	Larch is currently compatible *only* with version 3.

Once you've installed Anaconda, to get Larch you can simply run::

	pip install larch

from your command prompt (Windows) or the Terminal (Mac OS X). It's possible that you may
get some kind of a permission error when running this command.  If so, try it again
as an admin (on windows, right click the command line program and choose "Run as Administrator").

.. note::

	You might want to create a new conda environment for running larch.  One thing you can try
	is installing a pre-defined environment from the anaconda cloud.  This method isn't
	well tested across macOS and Windows platforms yet, but if it works for you it'll be
	the easist way to get everything (or most of everything) that you might need.  Just go to
	your command line or the anaconda prompt and enter::

		conda env create jpn/taiga

	(The Taiga is the northern boreal forest environment where larch can be found.)  After you create
	the environment, activate it::

		[macOS] source activate taiga
		[Windows] activate taiga

	If instead you create a new blank environment yourself using the conda tool, you'll want to make sure you install
	at least these packages before larch::

		conda install numpy
		conda install scipy
		conda install pandas
		conda install matplotlib
		conda install numexpr
		conda install pytables

Some of the graphical tools used to draw nested and network logit graphs may also not
be installed by default by Anaconda.  You don't need these tools to run any model in
Larch, just to draw pretty figures depicting the nests.  If you want to install these,
you can go to your command line to get the necessary tools::

	conda install graphviz
	pip install pygraphviz

It's possible that pygraphviz still won't want to place nice.  On macOS, you can try this::

	pip install pygraphviz \
	--install-option="--include-path=/usr/local/include/graphviz/" \
	--install-option="--library-path=/usr/local/lib/graphviz"


Once you've got Larch installed, you might want to jump directly to some :ref:`examples`
to see how you might use it.




