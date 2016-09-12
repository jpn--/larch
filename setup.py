import setuptools
from setuptools import setup, Extension
import glob, time, platform, os, sysconfig, sys, shutil, io

VERSION = '3.3.10'



if os.environ.get('READTHEDOCS', None) == 'True':
	# hack for building docs on rtfd
	def read(*filenames, **kwargs):
		encoding = kwargs.get('encoding', 'utf-8')
		sep = kwargs.get('sep', '\n')
		buf = []
		for filename in filenames:
			with io.open(filename, encoding=encoding) as f:
				buf.append(f.read())
		return sep.join(buf)

	long_description = read('README.rst')
	
	setup(name='larch',
		  version=VERSION,
		  package_dir = {'larch': 'py'},
		  packages=['larch', 'larch.examples', 'larch.version', 'larch.util', 'larch.util.optimize', 'larch.model_reporter'],
		  url='http://larch.readthedocs.org',
		  download_url='http://github.com/jpn--/larch',
		  author='Jeffrey Newman',
		  author_email='jeff@newman.me',
		  description='A framework for estimating and applying discrete choice models.',
		  long_description=long_description,
		  license = 'GPLv3',
		  classifiers = [
			'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
			'Programming Language :: Python :: 3',
			'Programming Language :: Python :: 3.4',
			'Programming Language :: Python :: 3.5',
			'Operating System :: MacOS :: MacOS X',
			'Operating System :: Microsoft :: Windows',
		  ],
		 )


else:

	usedir = os.path.dirname(__file__)
	if sys.path[0] != usedir:
		sys.path.insert(0, usedir)

	def file_at(*arg):
		return os.path.join(usedir, *arg)

	def incl(*arg):
		return '-I'+os.path.join(usedir, *arg)

	def read(*filenames, **kwargs):
		encoding = kwargs.get('encoding', 'utf-8')
		sep = kwargs.get('sep', '\n')
		buf = []
		for filename in filenames:
			with io.open(filename, encoding=encoding) as f:
				buf.append(f.read())
		return sep.join(buf)

	long_description = read(file_at('README.rst'))
		

	import numpy

	if platform.system() == 'Darwin':
		os.environ['LDFLAGS'] = '-framework Accelerate'
		os.environ['CLANG_CXX_LIBRARY'] = 'libc++'
		os.environ['CLANG_CXX_LANGUAGE_STANDARD'] = 'gnu++0x'

	# To update the version, run `git tag -a 3.3.10 -m'version 3.3.10, August 2016'`



	compiletime = time.strftime("%A %B %d %Y - %I:%M:%S %p")


	elm_cpp_dirs = ['etk', 'model', 'data', 'sherpa', 'vascular']

	elm_cpp_fileset = set()
	for elm_cpp_dir in elm_cpp_dirs:
		#elm_cpp_fileset |= set(glob.glob('src/{}/*.cpp'.format(elm_cpp_dir)))
		elm_cpp_fileset |= set(glob.glob(file_at('src',elm_cpp_dir,'*.cpp')))


	elm_cpp_files = list(elm_cpp_fileset)

	for elm_cpp_dir in elm_cpp_dirs:
		#elm_cpp_fileset |= set(glob.glob('src/{}/*.h'.format(elm_cpp_dir)))
		elm_cpp_fileset |= set(glob.glob(file_at('src',elm_cpp_dir,'*.h')))

	elm_cpp_h_files = list(elm_cpp_fileset)

	from setup_common import lib_folder, shlib_folder




	if platform.system() == 'Darwin':
		openblas = None
		gfortran = None
		mingw64_libs = []
		local_swig_opts = []
		local_libraries = []
		local_library_dirs = []
		local_includedirs = []
		local_macros = [('I_AM_MAC','1'), ('SQLITE_ENABLE_RTREE','1'), ]
		local_extra_compile_args = ['-std=gnu++11', '-w', '-arch', 'i386', '-arch', 'x86_64']# +['-framework', 'Accelerate']
		local_apsw_compile_args = ['-w']
		local_extra_link_args =   ['-framework', 'Accelerate']
		local_data_files = [('/usr/local/bin', [file_at('bin','larch')]), ]
		local_sqlite_extra_postargs = []
		dylib_name_style = "lib{}.so"
		DEBUG = False
		buildbase = None
	elif platform.system() == 'Windows':
		#old openblas = 'OpenBLAS-v0.2.9.rc2-x86_64-Win', 'lib', 'libopenblas.dll'
		openblas = 'OpenBLAS-v0.2.15-Win64-int32', 'lib', 'libopenblas.dll'
		#gfortran = 'OpenBLAS-v0.2.9.rc2-x86_64-Win', 'lib', 'libgfortran-3.dll'
		gfortran = 'OpenBLAS-v0.2.15-Win64-int32', 'lib', 'libgfortran-3.dll'
		mingw64_path = 'OpenBLAS-v0.2.15-Win64-int32', 'lib',
		mingw64_dlls = ['libgfortran-3', 'libgcc_s_seh-1', 'libquadmath-0']
		mingw64_libs = [i+'.dll' for i in mingw64_dlls]
		local_swig_opts = []
		local_libraries = ['PYTHON35','libopenblas',]+mingw64_dlls+['PYTHON35',]
		if os.path.exists('Z:/CommonRepo/'):
			local_library_dirs = [
				'Z:/CommonRepo/{0}/{1}'.format(*openblas),
			#	'C:\\local\\boost_1_56_0\\lib64-msvc-10.0',
				]
			local_includedirs = [
				'Z:/CommonRepo/{0}/include'.format(*openblas),
			#	'C:/local/boost_1_56_0',
				 ]
		else:
			local_library_dirs = [
				'C:/Users/jnewman/Documents/GitHub/{0}/{1}'.format(*openblas),
				]
			local_includedirs = [
				'C:/Users/jnewman/Documents/GitHub/{0}/include'.format(*openblas),
				 ]
		local_macros = [('I_AM_WIN','1'),  ('SQLITE_ENABLE_RTREE','1'), ]
		local_extra_compile_args = ['/EHsc', '/W0', ]
		#  for debugging...
		#	  extra_compile_args=['/Zi' or maybe '/Z7' ?],
		#     extra_link_args=[])
		local_apsw_compile_args = ['/EHsc']
		local_extra_link_args =    ['/DEBUG']
		local_data_files = []
		buildbase = None # "Z:\LarchBuild"
		local_sqlite_extra_postargs = ['/IMPLIB:' + os.path.join(shlib_folder(buildbase), 'larchsqlite.lib'), '/DLL',]
		dylib_name_style = "{}.dll"
		DEBUG = False
	#	raise Exception("TURN OFF multithreading in OpenBLAS")
	else:
		openblas = None
		gfortran = None
		mingw64_libs = []
		local_swig_opts = []
		local_libraries = []
		local_library_dirs = []
		local_includedirs = []
		local_macros = [('I_AM_LINUX','1'),  ('SQLITE_ENABLE_RTREE','1'), ]
		local_extra_compile_args = []
		local_apsw_compile_args = []
		local_extra_link_args =    []
		local_data_files = []
		local_sqlite_extra_postargs = []
		dylib_name_style = "{}.so"
		DEBUG = False
		buildbase = None


	from setup_sqlite import build_sqlite
	build_sqlite(buildbase)


	from distutils.ccompiler import new_compiler

	# Create compiler with default options
	c = new_compiler()
#	c.add_include_dir("./sqlite")

	if os.path.exists('Z:/CommonRepo'):
		if openblas is not None:
			shutil.copyfile(os.path.join('Z:/CommonRepo',*openblas), os.path.join(shlib_folder(buildbase),openblas[-1]))
		for dll in mingw64_libs:
			shutil.copyfile(os.path.join('Z:/CommonRepo',*(mingw64_path+(dll,))), os.path.join(shlib_folder(buildbase),dll))
	else:
		if openblas is not None:
			shutil.copyfile(os.path.join('C:/Users/jnewman/Documents/GitHub',*openblas), os.path.join(shlib_folder(buildbase),openblas[-1]))
		for dll in mingw64_libs:
			shutil.copyfile(os.path.join('C:/Users/jnewman/Documents/GitHub',*(mingw64_path+(dll,))), os.path.join(shlib_folder(buildbase),dll))

	swig_files = [file_at('src/swig/elmcore.i'),]
	swig_opts = ['-modern', '-py3',
								#'-I../include',
								'-v', '-c++', '-outdir', file_at('py'),
								#'-I../include',
								incl('src/etk'),
								incl('src/model'),
								incl('src/data'),
								incl('src/sherpa'),
								incl('src/vascular'),
								incl('src/swig'),
								incl('sqlite'), ] + local_swig_opts
#	if platform.system() == 'Windows' and os.path.exists(file_at('src/swig/elmcore_wrap.cpp')) and os.path.exists(file_at('py/core.py')):
#		swig_opts = []
#		swig_files = [file_at('src/swig/elmcore_wrap.cpp')]

	core = Extension('larch._core',
					 swig_files + elm_cpp_files,
					 swig_opts=swig_opts,
					 libraries=local_libraries+['larchsqlite', ],
					 library_dirs=local_library_dirs+[shlib_folder(buildbase),],
					 define_macros=local_macros,
					 include_dirs=local_includedirs + [numpy.get_include(),
													   file_at('src'),
													   file_at('src/etk'),
													   file_at('src/model'),
													   file_at('src/data'),
													   file_at('src/sherpa'),
													   file_at('src/vascular'),
													   file_at('src/swig'),
													   file_at('sqlite'), ],
					 extra_compile_args=local_extra_compile_args,
					 extra_link_args=local_extra_link_args,
					 depends=swig_files + elm_cpp_h_files,
					 )

	apsw_extra_link_args = []
	if platform.system() == 'Darwin':
		apsw_extra_link_args += ['-install_name', '@loader_path/{}'.format(c.library_filename('apsw','shared'))]


	apsw = Extension('larch.apsw',
					 [file_at('sqlite/apsw/apsw.c'),],
					 libraries=['larchsqlite', ],
					 library_dirs=[shlib_folder(buildbase),],
					 define_macros=local_macros+[('EXPERIMENTAL','1')],
					 include_dirs= [ file_at('sqlite'), file_at('sqlite/apsw'), ],
					 extra_compile_args=local_apsw_compile_args,
	#				 extra_link_args=apsw_extra_link_args,
					 )


	import build_configuration
	build_configuration.write_build_info(build_dir=lib_folder(buildbase), packagename="larch")


	setup(name='larch',
		  version=VERSION,
		  package_dir = {'larch': file_at('py'), 'larch.examples.doc': file_at('doc/example')},
		  packages=['larch', 'larch.examples','larch.examples.doc', 'larch.test', 'larch.version', 'larch.util', 'larch.model_reporter', 'larch.util.optimize',],
		  ext_modules=[core, apsw, ],
		  package_data={'larch':['data_warehouse/*.sqlite', 'data_warehouse/*.csv', 'data_warehouse/*.csv.gz', 'data_warehouse/*.h5', 'data_warehouse/*.h5.gz'], 'larch.examples.doc':['*.rst']},
		  data_files=local_data_files,
		  install_requires=[
							"numpy >= 1.11",
							"scipy >= 0.17.0",
							"pandas >= 0.18",
							"tables >= 3.2.2",
						],
		  extras_require = {
			'docx':  ["python-docx >= 0.8.5",],
			'test': ["nose >= 1.3",],
			'network': ["networkx >= 1.10",],
			'graphing': ["pygraphviz >= 1.3", "matplotlib >= 1.5", ],
			#'docs': ["sphinx >= 1.2.3", "sphinxcontrib-napoleon >= 0.4",],
			'docs': ["sphinx >= 1.3", ],
		  },
		  url='http://larch.readthedocs.org',
		  download_url='http://github.com/jpn--/larch',
		  author='Jeffrey Newman',
		  author_email='jeff@newman.me',
		  description='A framework for estimating and applying discrete choice models.',
		  long_description=long_description,
		  license = 'GPLv3',
		  classifiers = [
			'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
			'Programming Language :: Python :: 3',
			'Programming Language :: Python :: 3.4',
			'Programming Language :: Python :: 3.5',
			'Operating System :: MacOS :: MacOS X',
			'Operating System :: Microsoft :: Windows',
		  ],
		 )


