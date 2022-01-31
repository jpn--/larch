# Larch Documentation

[![conda-forge](https://img.shields.io/conda/vn/conda-forge/larch.svg)](https://anaconda.org/conda-forge/larch)
[![conda-forge](https://img.shields.io/conda/dn/conda-forge/larch)](https://anaconda.org/conda-forge/larch)
[![conda-forge](https://img.shields.io/azure-devops/build/wire-paladin/larch/jpn--.larch/master)](https://dev.azure.com/wire-paladin/larch/_build?definitionId=1&_a=summary&repositoryFilter=1&branchFilter=5%2C5%2C5%2C5%2C5%2C5)

🏆︁ Winner of the [AGIFORS 56th Annual Symposium Best Innovation award](http://agifors.org/Symposium>).

This documentation is for the Python interface for Larch. If this is your first go
with Larch, or the first go on a new computer, you might want to start with [installation](installation).

Larch is undergoing a transformation, with a new computational architecture
that can significantly improve performance when working with large datasets.
The new code relies on [numba](https://numba.pydata.org/),
[xarray](https://xarray.pydata.org/en/stable/), and
[sharrow](https://activitysim.github.io/sharrow) to enable super-fast estimation
of choice models.  Many (but not yet all) of the core features of Larch have been moved
over to this new platform.

You can still use the old version of Larch as normal, but to try out the new version
just import `larch.numba` instead of larch itself.

:::{note}
This project is very much under development.  There are plenty of undocumented functions
and features; use them at your own risk.  Undocumented features may be non-functional,
not rigorously tested, deprecated or removed without notice in a future version.  If a
function or method is documented here, it is intended to be stable in future updates.
:::
