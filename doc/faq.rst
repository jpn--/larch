.. larch documentation master file

==========================
Frequently Asked Questions
==========================

.. _whyNotBiogeme:

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





.. _usedToBeElm:

Did this software used to be called *ELM*?  Why the name change?
	Yes, earlier versions of this software were called ELM.  But it turns out ELM is a pretty common
	name for software.  Larch is not unique, but much less commonly used.  Also, it is the tradition of
	Python software to use names from Monty Python, particularly when you want to be able to identify
	your software `from quite a long way away <https://www.youtube.com/watch?v=ug8nHaelWtc>`_.





.. _windowsDownloadSize:

Why is the Windows wheel download so much larger than the Mac one?
	The Windows wheel include the openblas library for linear algebra computations.  The
	Mac version does not need an extra library because Mac OS X includes vector math libraries
	by default in the Accelerate framework.





.. _itsNotWorking:

It is not working. Can you troubleshoot for me?
	Are you using the 64 bit (amd64) version of Python?  Larch is only compiled for 64 bit at
	present.

	For some unknown reason, certain mathematical tools are not available on PyPI as wheels
	for Windows.  One way to get them is to download `numpy <http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy>`_,
	`scipy <http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy>`_, and
	`pandas <http://www.lfd.uci.edu/~gohlke/pythonlibs/#pandas>`_ and install them manually. But a
	better option if you can spare a few extra MB of installation disk space is to install the 64 bit
	`Anaconda <http://www.continuum.io/downloads>`_ version of Python 3.6.  This comes with a
	nice stack of other helpful statistical tools as well, and you'll probably not need
	any other libraries.  Once you've installed Anaconda, you can install Larch by typing
	"pip install larch" on the command line.





.. _troubleWithFilenames:

Why is Larch not recognizing my input file?
	If you are using windows, you might have a filename that looks like this::

		C:\path\to\my\file.csv

	Unfortunately, in Python the single backslash is an "escape character” and is interpreted
	differently depending on the next character.  For example, you might get::

		>>> filename = "C:\path\to\my\file.csv"
		>>> print(filename)
		C:\path	o\my♀ile.csv

	Obviously, that's not what you want.  Instead, you could use either of these for filenames::

		>>> filename = "C:\\path\\to\\my\\file.csv"
		>>> filename = "C:/path/to/my/file.csv"

	If you are using Mac OS X or Linux, your file pathnames use forward slashes and there
	should be no problem.




