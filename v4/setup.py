from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy
import os

# os.environ['CFLAGS'] = '-fopenmp'

def find_pyx(path='.'):
    pyx_files = []
    for root, dirs, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.pyx'):
                pyx_files.append(os.path.join(root, fname))
    return pyx_files



setup(
  name = 'larch4',
  ext_modules = cythonize(find_pyx()), 
  include_dirs=[numpy.get_include()],
  packages=find_packages(),
)




###
# python setup.py build_ext --inplace
#

#
# extra_compile_args = ['-fopenmp'],
# extra_link_args = ['-fopenmp'],
#