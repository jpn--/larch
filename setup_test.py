import sys, os

from setup_common import lib_folder
sys.path.insert(0,os.path.join(os.getcwd(),lib_folder()))

import larch.test
larch.test.run(exit=True, fail_fast=True)
