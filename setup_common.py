import os, sysconfig, sys


def distutils_dir_name(dname):
    """Returns the name of a distutils build directory"""
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(dirname=dname,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info)

def lib_folder(basepath=None):
	if basepath is None:
		return os.path.join(os.environ.get('PROJECT_DIR', ''), 'build', distutils_dir_name('lib'))
	else:
		return os.path.join(basepath, distutils_dir_name('lib'))

def temp_folder(basepath=None):
	if basepath is None:
		return os.path.join(os.environ.get('PROJECT_DIR', ''), 'build', distutils_dir_name('temp'))
	else:
		return os.path.join(basepath, distutils_dir_name('temp'))

def shlib_folder(basepath=None):
	if basepath is None:
		return os.path.join(os.environ.get('PROJECT_DIR', ''), 'build', distutils_dir_name('lib'), 'larch')
	else:
		return os.path.join(basepath, distutils_dir_name('lib'), 'larch')
