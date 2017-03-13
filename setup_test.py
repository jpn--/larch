import sys, os

from setup_common import lib_folder
os.chdir(lib_folder())
print(lib_folder())
sys.path.insert(0,lib_folder())

import larch.test
larch.test.run(exit=True, fail_fast=True)
