(larch-installation)=
# Installing Larch


## Quick Start

To install larch without building from source yourself, you'll need to use the
**conda** package manager. If you already have conda installed, you can use that,
otherwise you can download and install a [free version](https://github.com/conda-forge/miniforge).

Once you have conda installed, you can install Larch from the conda-forge
repository in a new environment called `arboretum` like this:

```shell
conda create -n arboretum -c conda-forge larch
```


## Installing Conda and Python

To install larch without building from source yourself, you'll
need to use the **conda** package manager. If you already have
conda installed, you can use that, otherwise you can download
and install a [free version](https://github.com/conda-forge/miniforge).

You should usually install conda for the local user,
which does not require administrator permissions.
You can also install conda system wide, which does require
administrator permissions -- but even if you have those permissions,
you may find that installing only for one user prevents problems
arising over multiple users editing common packages.

If you already have Python installed, either by itself or
as a companion to any one of a variety of common transportation planning
tools (e.g., ArcGIS), you can still install and use conda.
You do not need to uninstall, move, or change any existing
Python installation.  Just use the standard conda installer
and let the installer add the conda installation of Python
to your PATH environment variable. There is no need to set the
PYTHONPATH environment variable.

Once conda is installed, on Windows it can be accessed from a
preconfigured prompt (called "Anaconda Prompt", "Miniforge Prompt",
or something similar) that will be installed in the Start menu.
On linux and macOS, just use the regular terminal.

## Managing Environments

When you use conda to install Python, by default a `base` environment is
created and packages are installed in that environment.  However, in general you should
almost never undertake project work in the `base` environment, especially if your
project involves installing any custom Python packages.  Instead,
you should create a new environment for each project, and install the
necessary packages and dependencies in that environment.  This will help
prevent software conflicts, and ensure that tools installed for one project
will not break another project.

```{tip}
The instructions below provide only the most basic steps to
set up and use an environment.  Much more extensive documentation
on [managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
is available in the conda documentation
itself.
```

If you'd like one command to just install Larch and
a suite of related tools relevant for transportation planning and discrete choice
analysis, you can create a new environment for Larch with one line.

```shell
conda env create jpn/taiga
```

If you've already installed the *taiga* environment and want to update it to the latest
version, you can use:

```shell
conda env update jpn/taiga --prune
```

The *prune* option here will remove packages that are not ordinarily included in the
*taiga* environment; omit that function if you've installed extra packages that you
want to keep.


### Using an Environment

When using the terminal (MacOS/Linux) or a conda prompt (Windows), the
current environment name will be shown as part of the prompt:

```shell
(base) Computer:~ cfinley$
```

By default, when opening a new terminal the environment is set as the
``base`` environment, although this is typically not where you want to
be if you have followed the advice above.  Instead, to switch environments
use the ``conda activate`` command.  For example, to activate the ``taiga``
environment installed in the quick start, run:

```shell
(base) Computer:~ cfinley$ conda activate taiga
(taiga) Computer:~ cfinley$
```

## Running Jupyter

The most convenient interface for interactive use of Larch is within
[JupyterLab](https://jupyterlab.readthedocs.io/en/stable/).
If it's not already installed in your base or working
environments, you can install it using conda:

```shell
conda install -c conda-forge jupyterlab
```

Then to start JupyterLab,

```shell
jupyter lab
```

JupyterLab will open automatically in your browser.
