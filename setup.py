from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize
import os
import platform
import re
import io

# For macOS, you need to  ...
#  xcode-select --install
#  conda install clangdev llvmdev openmp -c conda-forge
#
# On macOS 10.14 Mojave, it might be necessary to download additional libraries:
#  /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
# for more info on this topic see:
#  https://stackoverflow.com/questions/52509602/cant-compile-c-program-on-a-mac-after-upgrade-to-mojave


def read(path, encoding='utf-8'):
    path = os.path.join(os.path.dirname(__file__), path)
    with io.open(path, encoding=encoding) as fp:
        return fp.read()

def version(path):
    """Obtain the packge version from a python file e.g. pkg/__init__.py
    See <https://packaging.python.org/en/latest/single_source_version.html>.
    """
    version_file = read(path)
    version_match = re.search(r"""^__version__ = ['"]([^'"]*)['"]""",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = version('larch/__init__.py')

LARCH_COMPILER = "clang" #  "gcc"

os.environ["CC"] = LARCH_COMPILER

def find_pyx(path='.'):
    pyx_files = []
    for root, dirs, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.pyx'):
                pyx_files.append(os.path.join(root, fname))
    if platform.system() == 'Windows':
        pyx_files = [i for i in pyx_files if 'sqlite' not in i]
    return pyx_files

if platform.system() == 'Windows':
    os.environ['CFLAGS'] = '/openmp'
    try:
        include_dirs = ['.', os.environ['LIBRARY_INC']]
    except KeyError:
        include_dirs = ['.', ]
    try:
        library_dirs = [os.environ['LIBRARY_LIB']]
    except KeyError:
        library_dirs = []
    extra_compile_args = ("/openmp",) # ["/openmp"],
    extra_link_args = ("/openmp",) # ['/openmp']
    libraries = []
elif platform.system() == 'Linux':
    include_dirs = ['.', ]
    library_dirs = []
    extra_compile_args = [
        "-fopenmp",
    ]
    extra_link_args = [
        "-fopenmp",
    ]
    libraries = [
        # 'iomp5',
        # 'pthread',
    ]
else:
    if LARCH_COMPILER == 'gcc':
        ## notes : https://github.com/ContinuumIO/anaconda-issues/issues/8803
        os.environ['CFLAGS'] = '-openmp'
        include_dirs = ['.', ]
        library_dirs = []
        extra_compile_args = [
            # "-openmp",
        ]
        extra_link_args = [
           # '-openmp',
        ]
        libraries = [
            # 'iomp5',
            # 'pthread',
        ]
    else:
        os.environ['CFLAGS'] = '-fopenmp'
        include_dirs = ['.', ]
        library_dirs = []
        extra_compile_args = [
            "-fopenmp",
        ]
        extra_link_args = [
           '-fopenmp',
        ]
        libraries = [
            # 'iomp5',
            # 'pthread',
        ]

from pprint import pprint
pprint(os.environ)

extensions = [
    Extension(
        pyx.replace('.pyx','').replace("."+os.sep,'').replace(os.sep,'.'),  # os.path.basename(pyx).replace('.pyx',''),
        [pyx],
        include_dirs = include_dirs,
        libraries = libraries,
        library_dirs = library_dirs,
        extra_compile_args = extra_compile_args,
        extra_link_args = extra_link_args,
    )
    for pyx in find_pyx()
]

setup(
    name = 'larch',
    version = VERSION,
    ext_modules = cythonize(extensions, gdb_debug=True, annotate=True),
    include_dirs=include_dirs,
    packages=find_packages(),
    package_data={
        # If any package contains these types of files, include them:
        '': ['*.h5', '*.h5d', '*.omx', '*.h5.gz', '*.csv.gz', '*.sqlite', ],
        'larch': [
            'doc/*.rst','doc/*/*.rst','doc/*/*/*.rst','doc/*/*/*/*.rst',
            'doc/*.ipynb', 'doc/example/*.ipynb',
        ],
    },
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'tables >=3.4',
        'cloudpickle',
        'tqdm',
        'networkx >=2.0',
        'appdirs >=1.4',
        'docutils >=0.13.1',
        'ghalton',
        'jinja2 >=2.9.6', # for pandas styler
        'beautifulsoup4 >=4.6',
        'seaborn',
        # 'xmle >=0.1.3',
    ] + [
        # 'sqlite >=3.21',
    ] if platform.system() != 'Windows' else [],
    url='http://larch.newman.me',
    author='Jeffrey Newman',
    author_email='jeff@newman.me',
    description='A framework for estimating and applying discrete choice models.',
    license = 'GPLv3',
    classifiers = [
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    ],
    zip_safe=False,

)




###
# python setup.py build_ext --inplace
#

#
# extra_compile_args = ['-fopenmp'],
# extra_link_args = ['-fopenmp'],
#