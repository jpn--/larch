larch
=====

.. image:: https://img.shields.io/pypi/v/larch.svg
    :target: https://pypi.python.org/pypi/larch
    :class: statusbadge

.. image:: https://img.shields.io/badge/released-11%20February%202016-blue.svg
    :target: https://pypi.python.org/pypi/larch
    :class: statusbadge

.. image:: https://img.shields.io/pypi/l/larch.svg
    :target: https://github.com/jpn--/larch/blob/master/LICENSE
    :class: statusbadge

.. image:: https://readthedocs.org/projects/larch/badge/?version=latest&style=round
    :target: http://larch.readthedocs.org
    :class: statusbadge

**Larch**: the logit architect

This is a tool for the estimation and application of logit-based discrete choice models.
It is designed to integrate with NumPy and facilitate fast processing of linear models.
If you want to estimate *non-linear* models, try `Biogeme <http://biogeme.epfl.ch/>`_,
which is more flexible in form and can be used for almost any model structure.
If you don't know what the difference is, you probably want to start with linear models.

This project is very much under development.  There are plenty of undocumented functions
and features; use them at your own risk.  Undocumented features may be non-functional, 
not rigorously tested, deprecated or removed without notice in a future version.  If a
function or method is `documented <http://larch.readthedocs.org>`_, it is intended to be
stable in future updates.

FAQ
---

*Why is the Windows download so much larger than the Mac download?*

The Windows wheel include the openblas library for linear algebra computations.  The
Mac version does not need an extra library because Mac OS X includes vector math libraries
by default.

*It is not working. Can you troubleshoot for me?*

Are you using the 64 bit (amd64) version of Python?  Larch is only compiled for 64 bit at
present.

For some unknown reason, certain mathematical tools are not available on PyPI as wheels
for Windows.  You will need to download `numpy <http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy>`_,
`scipy <http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy>`_, and
`pandas <http://www.lfd.uci.edu/~gohlke/pythonlibs/#pandas>`_ and install them manually.

You may also need to install the
`Microsoft Visual C++ 2015 <https://www.microsoft.com/en-us/download/details.aspx?id=48145>`
redistributable libraries. Future versions of Larch may include these for you.