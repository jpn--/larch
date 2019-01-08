from .h5data import *
from .h5vault import *

import tables
import sys, atexit
from distutils.version import StrictVersion

def my_close_open_files(verbose):
    open_files = tables.file._open_files

    are_open_files = len(open_files) > 0

    if verbose and are_open_files:
        sys.stderr.write("Closing remaining open files:")

    if StrictVersion(tables.__version__) >= StrictVersion("3.1.0"):
        # make a copy of the open_files.handlers container for the iteration
        handlers = list(open_files.handlers)
    else:
        # for older versions of pytables, setup the handlers list from the
        # keys
        keys = open_files.keys()
        handlers = []
        for key in keys:
            handlers.append(open_files[key])

    for fileh in handlers:
        if verbose:
            sys.stderr.write("\n%s..." % fileh.filename)

        fileh.close()

        if verbose:
            sys.stderr.write("done")

    if verbose and are_open_files:
        sys.stderr.write("\n")

atexit.register(my_close_open_files, False)