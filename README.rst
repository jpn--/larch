larch
=====

.. image:: https://img.shields.io/conda/v/conda-forge/larch
    :target: https://anaconda.org/conda-forge/larch
    :class: statusbadge

.. image:: https://img.shields.io/conda/dn/conda-forge/larch
    :target: https://anaconda.org/conda-forge/larch
    :class: statusbadge

.. image:: https://img.shields.io/badge/source-github-yellow.svg
    :target: https://github.com/jpn--/larch
    :class: statusbadge

.. image:: https://img.shields.io/conda/l/conda-forge/larch
    :target: https://github.com/jpn--/larch/blob/master/LICENSE
    :class: statusbadge

**Larch**: the logit architect

This is a tool for the estimation and application of logit-based discrete choice models.
It is designed to integrate with NumPy and facilitate fast processing of linear models.
If you want to estimate *non-linear* models, try `Biogeme <http://biogeme.epfl.ch/>`_,
which is more flexible in form and can be used for almost any model structure.
If you don't know what the difference is, you probably want to start with linear models.

Larch is undergoing a transformation, with a new computational architecture
that can significantly improve performance when working with large datasets.
The new code relies on [numba](https://numba.pydata.org/),
[xarray](https://xarray.pydata.org/en/stable/), and
[sharrow](https://activitysim.github.io/sharrow) to enable super-fast estimation
of choice models.  Many (but not yet all) of the core features of Larch have been moved
over to this new platform.

You can still use the old version of Larch as normal, but to try out the new version
just import `larch.numba` instead of larch itself.

This project is very much under development.  There are plenty of undocumented functions
and features; use them at your own risk.  Undocumented features may be non-functional,
not rigorously tested, deprecated or removed without notice in a future version.
