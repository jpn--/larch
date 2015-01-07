import os, sysconfig, sys


def distutils_dir_name(dname):
    """Returns the name of a distutils build directory"""
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(dirname=dname,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info)

def lib_folder():
	return os.path.join('build', distutils_dir_name('lib'))

def shlib_folder():
	return os.path.join('build', distutils_dir_name('lib'), 'larch')

