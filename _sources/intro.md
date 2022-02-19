# Larch Documentation

[![conda-forge](https://img.shields.io/conda/vn/conda-forge/larch.svg)](https://anaconda.org/conda-forge/larch)
[![conda-forge](https://img.shields.io/conda/dn/conda-forge/larch)](https://anaconda.org/conda-forge/larch)
[![conda-forge](https://img.shields.io/azure-devops/build/wire-paladin/larch/jpn--.larch/master)](https://dev.azure.com/wire-paladin/larch/_build?definitionId=1&_a=summary&repositoryFilter=1&branchFilter=5%2C5%2C5%2C5%2C5%2C5)

🏆︁ Winner of the [AGIFORS 56th Annual Symposium Best Innovation award](http://agifors.org/Symposium).

This documentation is for the Python interface for Larch. If this is your first go
with Larch, or the first go on a new computer, you might want to start with
[installation](larch-installation).

**Are you ready for something shiny, new, and *fast*?** Larch is undergoing
a transformation, with a new computational architecture
that can significantly improve performance when working with large datasets.
The old version of Larch used a carefully customized `DataFrames` object to
organize several different aspects of discrete choice data.
The new code uses a more standardized (although still enhanced) `xarray.Dataset`
interface for data, and relies on [numba](https://numba.pydata.org/),
[xarray](https://xarray.pydata.org/en/stable/), and
[sharrow](https://activitysim.github.io/sharrow) to enable super-fast estimation
of choice models.  Many (but not yet all) of the core features of Larch have been moved
over to this new platform.

If you want to try out the new version, just import `larch.numba` instead of `larch`
itself.  These docs adopt the convention of `import larch.numba as lx`.  All of the
compatible examples in this documentation are being migrated over to the new platform,
but the old examples remain available under the [Legacy Examples](deprecated-examples)
section. If you're not ready for all this awesomeness, or if you need to use some
features of Larch that are not yet in the new version,
**you can still use the legacy (i.e. "old") version of Larch as normal.**

:::{note}
This project is very much under development.  There are plenty of undocumented functions
and features; use them at your own risk.  Undocumented features may be non-functional,
not rigorously tested, deprecated or removed without notice in a future version.  If a
function or method is documented here, it is intended to be stable in future updates.
:::
