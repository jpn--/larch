larch
=====

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