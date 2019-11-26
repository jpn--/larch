.. larch documentation getting started

===============
Getting Started
===============

.. _quickstart:

Quick Start
-----------

If you already have conda installed, you can install Larch from conda-forge in a
new environment called `arboretum` like this:

.. code-block:: console

    conda create -n arboretum -c conda-forge larch

Alternatively, if you would prefer to run in a container, which is potentially
a more stable and portable "go anywhere" solution across various machines,
you can install `Docker <https://www.docker.com/>`_ and run the following
command:

.. code-block:: console

    docker run -it -p 8857:8857 -v "%cd%":/home/paladin/work wirepaladin/sanfrancisco

This will install a container containing the entire conda package and everything
needed to run Larch, including JupyterLab.  Once the container is downloaded and
running, point your browser at `http://localhost:8857 <http://localhost:8857>`_
to access JupyterLab.

.. _installation:

Installing Python
-----------------

To run Larch, you'll need to have the 64 bit version of Python 3.7, plus a handful
of other useful statistical packages for Python.  The easiest way to get the basics
is to download and install the `Anaconda <https://www.anaconda.com/download>`_
version of Python 3.7 (64 bit). This comes with everything you'll need to get started,
and the Anaconda folks have helpfully curated a selection of useful tools for you,
so you don't have the sort through the huge pile of junk that is available for Python.

.. note::

    Python has two versions (2 and 3) that are available and currently maintained.
    Larch is compatible *only* with version 3.

You should usually install Anaconda for the local user,
which does not require administrator permissions.
You can also install Anaconda system wide, which does require
administrator permissions -- but even if you have those permissions,
you may find that installing only for one user prevents problems
arising over multiple users editing common packages.

If you already have Python installed, either by itself or
as a companion to any one of a variety of common transportation planning
tools (e.g., ArcGIS), you can still install and use Anaconda.
You do not need to uninstall, move, or change any existing
Python installation.  Just use the standard Anaconda installer
and let the installer add the conda installation of Python
to your PATH environment variable. There is no need to set the
PYTHONPATH environment variable.

Once Anaconda is installed, it can be accessed from the
Anaconda Prompt (on Windows) or the Terminal (linux and macOS).


Managing Environments
---------------------

When you use conda to install Python, by default a `base` environment is
created and packages are installed in that environment.  However, in general you should
almost never undertake project work in the `base` environment, especially if your
project involves installing any custom Python packages.  Instead,
you should create a new environment for each project, and install the
necessary packages and dependencies in that environment.  This will help
prevent software conflicts, and ensure that tools installed for one project
will not break another project.

.. note::

    The instructions below provide only the most basic steps to
    set up and use an environment.  Much more extensive documentation
    on :doc:`managing environments <conda:user-guide/tasks/manage-environments>`
    is available in the conda documentation
    itself.


Quick Start
~~~~~~~~~~~

.. note::

    If you installed the "Miniconda" version of the anaconda package, you
    may need to install or update the *conda* and *anaconda-client* packages
    before the remote environment installation below will work:

    .. code-block:: console

        conda install -n base -c defaults conda anaconda-client

If you'd like one command to just install Larch and
a suite of related tools relevant for transportation planning and discrete choice
analysis, you can create a new environment for Larch with one line.

.. code-block:: console

    conda env create jpn/taiga

If you've already installed the *taiga* environment and want to update it to the latest
version, you can use:

.. code-block:: console

    conda env update jpn/taiga --prune

The *prune* option here will remove packages that are not ordinarily included in the
*taiga* environment; omit that function if you've installed extra packages that you
want to keep.

Then you can skip directly to `using an environment <using_an_environment>`_.


Creating an Environment
~~~~~~~~~~~~~~~~~~~~~~~

Use the terminal (MacOS/Linux) or an Anaconda Prompt (Windows) to create an environment:

.. code-block:: console

    conda create --name your_environment_name

Be sure to replace ``your_environment_name`` with a suitable
name for the environment to create.

When conda asks you to proceed, type "y" or just hit enter:

.. code-block:: console

    proceed ([y]/n)?

This creates the ``your_environment_name`` environment. By default,
this new environment uses the same version of Python that you are
currently using.  If you want a specific version of Python you can
request it explicitly:

.. code-block:: console

    conda create --name your_environment_name python=3.7

You can also create an environment with one or more specific packages
installed, by giving them as well:

.. code-block:: console

    conda create --name your_environment_name python=3.7 numpy pandas

Clearly, if you have a lot of packages to install, this can become a long
command, and a bit unwieldy to use.  Fortunately, you can instead just
describe the environment you want to create in a YAML file instead of
doing so on the command line.  To do so, you would get or create a YAML
file that looks something like this:

.. code-block:: yaml

    name: your_environment_name

    channels:
    - conda-forge
    - defaults

    dependencies:
    - python=3.7
    - pip
    - numpy>=1.15.4
    - pandas>=0.23.4
    - scipy>=1.1
    - scikit-learn>=0.20.1
    - networkx
    - larch
    - pip:
      - specialty_package


And then create the environment using the file.

.. code-block:: console

    conda env create -f environment.yml

You may notice that the ``specialty_package`` in the environment.yml file
is installed using pip instead of conda.  This is
`strongly discouraged <https://www.anaconda.com/using-pip-in-a-conda-environment/>`_
if the package is also available from conda, but may be necessary to
install certain packages that are available only on PyPI.



.. _using_an_environment:

Using an Environment
~~~~~~~~~~~~~~~~~~~~

When using the terminal (MacOS/Linux) or an Anaconda Prompt (Windows), the
current environment name will be shown as part of the prompt:

.. code-block:: batch

    (base) C:\Users\cfinley>


.. code-block:: shell-session

    (base) Computer:~ cfinley$

By default, when opening a new terminal the environment is set as the
``base`` environment, although this is typically not where you want to
be if you have followed the advice above.  Instead, to switch environments
use the ``conda activate`` command.  For example, to activate the ``taiga``
environment installed in the quick start, run:

.. code-block:: batch

    (base) C:\Users\cfinley> conda activate taiga
    (taiga) C:\Users\cfinley>

.. code-block:: shell-session

    (base) Computer:~ cfinley$ conda activate taiga
    (taiga) Computer:~ cfinley$


Running Jupyter
---------------

The most convenient interface for interactive use of Larch is within
`JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`_.
If it's not already installed in your base or working
environments, you can install it using conda:

.. code-block:: console

    conda install -c conda-forge jupyterlab

Then to start JupyterLab,

.. code-block:: console

    jupyter lab

JupyterLab will open automatically in your browser.


Advanced Usage
--------------

It is highly recommended that you use the pre-built packages available through
conda.  However, if for some reason you want to compile Larch from source,
you might find some useful tips :ref:`here <compiling>`.

.. toctree::
    :maxdepth: 1

    compiling