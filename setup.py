import setuptools
from setuptools import setup, Extension
import glob, time, platform, os, sysconfig, sys, shutil, io

VERSION = '3.1.15'


def read(*filenames, **kwargs):
	encoding = kwargs.get('encoding', 'utf-8')
	sep = kwargs.get('sep', '\n')
	buf = []
	for filename in filenames:
		with io.open(filename, encoding=encoding) as f:
			buf.append(f.read())
	return sep.join(buf)

long_description = read('README.rst')



if os.environ.get('READTHEDOCS', None) == 'True':
	# hack for building docs on rtfd

	setup(name='larch',
		  version=VERSION,
		  package_dir = {'larch': 'py'},
		  packages=['larch', 'larch.examples', 'larch.version'],
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

	import numpy

	if platform.system() == 'Darwin':
		os.environ['LDFLAGS'] = '-framework Accelerate'
		os.environ['CLANG_CXX_LIBRARY'] = 'libc++'
		os.environ['CLANG_CXX_LANGUAGE_STANDARD'] = 'gnu++0x'

	# To update the version, run `git tag -a 3.1.1-JAN2015 -m'version 3.1.1, January 2015'`


	#
	## monkey-patch for parallel compilation
	#def parallelCCompile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
	#    # those lines are copied from distutils.ccompiler.CCompiler directly
	#    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
	#    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
	#    # parallel code
	#    N=2 # number of parallel compilations
	#    import multiprocessing.pool
	#    def _single_compile(obj):
	#        try: src, ext = build[obj]
	#        except KeyError: return
	#        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
	#    # convert to list, imap is evaluated on-demand
	#    list(multiprocessing.pool.ThreadPool(N).imap(_single_compile,objects))
	#    return objects
	#import distutils.ccompiler
	#distutils.ccompiler.CCompiler.compile=parallelCCompile
	#
	#
	#
	#
	#import distutils.debug
	#from distutils.debug import *


	from distutils.command.build_clib import build_clib


	compiletime = time.strftime("%A %B %d %Y - %I:%M:%S %p")

	simp_cpp_files = list((set(glob.glob('src/larch*.cpp')))-set(glob.glob('src/*_wrap.cpp')))


	elm_cpp_dirs = ['etk', 'model', 'data', 'sherpa', 'vascular']

	elm_cpp_fileset = set()
	for elm_cpp_dir in elm_cpp_dirs:
		elm_cpp_fileset |= set(glob.glob('src/{}/*.cpp'.format(elm_cpp_dir)))

	elm_cpp_files = list(elm_cpp_fileset)

	for elm_cpp_dir in elm_cpp_dirs:
		elm_cpp_fileset |= set(glob.glob('src/{}/*.h'.format(elm_cpp_dir)))

	elm_cpp_h_files = list(elm_cpp_fileset)


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


	sqlite3_exports=[
	'sqlite3_aggregate_context',
	'sqlite3_aggregate_count',
	'sqlite3_auto_extension',
	'sqlite3_backup_finish',
	'sqlite3_backup_init',
	'sqlite3_backup_pagecount',
	'sqlite3_backup_remaining',
	'sqlite3_backup_step',
	'sqlite3_bind_blob',
	'sqlite3_bind_double',
	'sqlite3_bind_int',
	'sqlite3_bind_int64',
	'sqlite3_bind_null',
	'sqlite3_bind_parameter_count',
	'sqlite3_bind_parameter_index',
	'sqlite3_bind_parameter_name',
	'sqlite3_bind_text',
	'sqlite3_bind_text16',
	'sqlite3_bind_value',
	'sqlite3_bind_zeroblob',
	'sqlite3_blob_bytes',
	'sqlite3_blob_close',
	'sqlite3_blob_open',
	'sqlite3_blob_read',
	'sqlite3_blob_reopen',
	'sqlite3_blob_write',
	'sqlite3_busy_handler',
	'sqlite3_busy_timeout',
	'sqlite3_cancel_auto_extension',
	'sqlite3_changes',
	'sqlite3_clear_bindings',
	'sqlite3_close',
	'sqlite3_close_v2',
	'sqlite3_collation_needed',
	'sqlite3_collation_needed16',
	'sqlite3_column_blob',
	'sqlite3_column_bytes',
	'sqlite3_column_bytes16',
	'sqlite3_column_count',
	#'sqlite3_column_database_name',
	#'sqlite3_column_database_name16',
	'sqlite3_column_decltype',
	'sqlite3_column_decltype16',
	'sqlite3_column_double',
	'sqlite3_column_int',
	'sqlite3_column_int64',
	'sqlite3_column_name',
	'sqlite3_column_name16',
	#'sqlite3_column_origin_name',
	#'sqlite3_column_origin_name16',
	#'sqlite3_column_table_name',
	#'sqlite3_column_table_name16',
	'sqlite3_column_text',
	'sqlite3_column_text16',
	'sqlite3_column_type',
	'sqlite3_column_value',
	'sqlite3_commit_hook',
	'sqlite3_compileoption_get',
	'sqlite3_compileoption_used',
	'sqlite3_complete',
	'sqlite3_complete16',
	'sqlite3_config',
	'sqlite3_context_db_handle',
	'sqlite3_create_collation',
	'sqlite3_create_collation16',
	'sqlite3_create_collation_v2',
	'sqlite3_create_function',
	'sqlite3_create_function16',
	'sqlite3_create_function_v2',
	'sqlite3_create_module',
	'sqlite3_create_module_v2',
	'sqlite3_data_count',
	'sqlite3_db_config',
	'sqlite3_db_filename',
	'sqlite3_db_handle',
	'sqlite3_db_mutex',
	'sqlite3_db_readonly',
	'sqlite3_db_release_memory',
	'sqlite3_db_status',
	'sqlite3_declare_vtab',
	'sqlite3_enable_load_extension',
	'sqlite3_enable_shared_cache',
	'sqlite3_errcode',
	'sqlite3_errmsg',
	'sqlite3_errmsg16',
	'sqlite3_errstr',
	'sqlite3_exec',
	'sqlite3_expired',
	'sqlite3_extended_errcode',
	'sqlite3_extended_result_codes',
	'sqlite3_file_control',
	'sqlite3_finalize',
	'sqlite3_free',
	'sqlite3_free_table',
	'sqlite3_get_autocommit',
	'sqlite3_get_auxdata',
	'sqlite3_get_table',
	'sqlite3_global_recover',
	'sqlite3_initialize',
	'sqlite3_interrupt',
	'sqlite3_last_insert_rowid',
	'sqlite3_libversion',
	'sqlite3_libversion_number',
	'sqlite3_limit',
	'sqlite3_load_extension',
	'sqlite3_log',
	'sqlite3_malloc',
	'sqlite3_memory_alarm',
	'sqlite3_memory_highwater',
	'sqlite3_memory_used',
	'sqlite3_mprintf',
	'sqlite3_mutex_alloc',
	'sqlite3_mutex_enter',
	'sqlite3_mutex_free',
	'sqlite3_mutex_leave',
	'sqlite3_mutex_try',
	'sqlite3_next_stmt',
	'sqlite3_open',
	'sqlite3_open16',
	'sqlite3_open_v2',
	'sqlite3_os_end',
	'sqlite3_os_init',
	'sqlite3_overload_function',
	'sqlite3_prepare',
	'sqlite3_prepare16',
	'sqlite3_prepare16_v2',
	'sqlite3_prepare_v2',
	'sqlite3_profile',
	'sqlite3_progress_handler',
	'sqlite3_randomness',
	'sqlite3_realloc',
	'sqlite3_release_memory',
	'sqlite3_reset',
	'sqlite3_reset_auto_extension',
	'sqlite3_result_blob',
	'sqlite3_result_double',
	'sqlite3_result_error',
	'sqlite3_result_error16',
	'sqlite3_result_error_code',
	'sqlite3_result_error_nomem',
	'sqlite3_result_error_toobig',
	'sqlite3_result_int',
	'sqlite3_result_int64',
	'sqlite3_result_null',
	'sqlite3_result_text',
	'sqlite3_result_text16',
	'sqlite3_result_text16be',
	'sqlite3_result_text16le',
	'sqlite3_result_value',
	'sqlite3_result_zeroblob',
	'sqlite3_rollback_hook',
	#'sqlite3_rtree_geometry_callback',
	#'sqlite3_rtree_query_callback',
	'sqlite3_set_authorizer',
	'sqlite3_set_auxdata',
	'sqlite3_shutdown',
	'sqlite3_sleep',
	'sqlite3_snprintf',
	'sqlite3_soft_heap_limit',
	'sqlite3_soft_heap_limit64',
	'sqlite3_sourceid',
	'sqlite3_sql',
	'sqlite3_status',
	'sqlite3_step',
	'sqlite3_stmt_busy',
	'sqlite3_stmt_readonly',
	'sqlite3_stmt_status',
	'sqlite3_strglob',
	'sqlite3_stricmp',
	'sqlite3_strnicmp',
	#'sqlite3_table_column_metadata',
	'sqlite3_test_control',
	'sqlite3_thread_cleanup',
	'sqlite3_threadsafe',
	'sqlite3_total_changes',
	'sqlite3_trace',
	'sqlite3_transfer_bindings',
	'sqlite3_update_hook',
	'sqlite3_uri_boolean',
	'sqlite3_uri_int64',
	'sqlite3_uri_parameter',
	'sqlite3_user_data',
	'sqlite3_value_blob',
	'sqlite3_value_bytes',
	'sqlite3_value_bytes16',
	'sqlite3_value_double',
	'sqlite3_value_int',
	'sqlite3_value_int64',
	'sqlite3_value_numeric_type',
	'sqlite3_value_text',
	'sqlite3_value_text16',
	'sqlite3_value_text16be',
	'sqlite3_value_text16le',
	'sqlite3_value_type',
	'sqlite3_vfs_find',
	'sqlite3_vfs_register',
	'sqlite3_vfs_unregister',
	'sqlite3_vmprintf',
	'sqlite3_vsnprintf',
	'sqlite3_vtab_config',
	'sqlite3_vtab_on_conflict',
	'sqlite3_wal_autocheckpoint',
	'sqlite3_wal_checkpoint',
	'sqlite3_wal_checkpoint_v2',
	'sqlite3_wal_hook',
	#'sqlite3_win32_is_nt',
	'sqlite3_win32_mbcs_to_utf8',
	'sqlite3_win32_set_directory',
	'sqlite3_win32_sleep',
	'sqlite3_win32_utf8_to_mbcs',
	'sqlite3_win32_write_debug',

	'sqlite3_haversine_autoinit',
	'sqlite3_bonus_autoinit',

	]


	libdir = shlib_folder()


	if platform.system() == 'Darwin':
		openblas = None
		gfortran = None
		local_swig_opts = []
		local_libraries = []
		local_library_dirs = []
		local_includedirs = []
		local_macros = [('I_AM_MAC','1'), ('SQLITE_ENABLE_RTREE','1'), ]
		local_extra_compile_args = ['-std=gnu++11', '-w', '-arch', 'i386', '-arch', 'x86_64']# +['-framework', 'Accelerate']
		local_apsw_compile_args = ['-w']
		local_extra_link_args =   ['-framework', 'Accelerate']
		local_data_files = [('/usr/local/bin', ['bin/larch']), ]
		local_sqlite_extra_postargs = []
		dylib_name_style = "lib{}.so"
		DEBUG = False
	elif platform.system() == 'Windows':
		openblas = 'OpenBLAS-v0.2.9.rc2-x86_64-Win', 'lib', 'libopenblas.dll'
		gfortran = 'OpenBLAS-v0.2.9.rc2-x86_64-Win', 'lib', 'libgfortran-3.dll'
		local_swig_opts = []
		local_libraries = ['PYTHON35','libopenblas','libgfortran-3','PYTHON35',]
		local_library_dirs = [
			'Z:/Larch/{0}/{1}'.format(*openblas),
		#	'C:\\local\\boost_1_56_0\\lib64-msvc-10.0',
			]
		local_includedirs = [
			'./{0}/include'.format(*openblas),
		#	'C:/local/boost_1_56_0',
			 ]
		local_macros = [('I_AM_WIN','1'),  ('SQLITE_ENABLE_RTREE','1'), ]
		local_extra_compile_args = ['/EHsc', '/W0', ]
		#  for debugging...
		#	  extra_compile_args=['/Zi' or maybe '/Z7' ?],
		#     extra_link_args=[])
		local_apsw_compile_args = ['/EHsc']
		local_extra_link_args =    ['/DEBUG']
		local_data_files = []
		local_sqlite_extra_postargs = ['/IMPLIB:' + os.path.join(libdir, 'larchsqlite.lib'), '/DLL',]
		dylib_name_style = "{}.dll"
		DEBUG = False
	#	raise Exception("TURN OFF multithreading in OpenBLAS")
	else:
		openblas = None
		gfortran = None
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




	#lib_sqlite = ('larchsqlite',           {'sources': ['sqlite/sqlite3.c']})
	#lib_sqlhav = ('larchsqlhaversine',     {'sources': ['sqlite/haversine.c']})
	#lib_sqlext = ('larchsqlite3extension', {'sources': ['sqlite/extension-functions.c']})



	shared_libs = [
	#('larchsqlite',           'sqlite/sqlite3.c'             ,sqlite3_exports,  local_sqlite_extra_postargs, []),
	#('larchsqlhaversine',     'sqlite/haversine.c'           ,None,             []                         , []),
	#('larchsqlite3extension', 'sqlite/extension-functions.c' ,None,             []                         , []),
	('larchsqlite', ['sqlite/sqlite3.c','sqlite/haversine.c','sqlite/bonus.c'] ,sqlite3_exports,  local_sqlite_extra_postargs, []),
	]


	from distutils.ccompiler import new_compiler

	# Create compiler with default options
	c = new_compiler()
	c.add_include_dir("./sqlite")

	for name, source, exports, extra_postargs, extra_preargs in shared_libs:
		
		if not isinstance(source,list):
			source = [source,]
		
		need_to_update = False
		for eachsource in source:
			try:
				need_to_update = need_to_update or (os.path.getmtime(eachsource) > os.path.getmtime(os.path.join(libdir, dylib_name_style.format(name))))
			except FileNotFoundError:
				need_to_update = True

		# change dynamic library install name
		if platform.system() == 'Darwin':
			extra_postargs += ['-install_name', '@loader_path/{}'.format(c.library_filename(name,'shared'))]
			extra_preargs  += ['-arch', 'i386', '-arch', 'x86_64']

		if need_to_update:
			# Compile into .o files
			objects = c.compile(source, extra_preargs=extra_preargs, debug=DEBUG, macros=local_macros,)
			# Create shared library
			c.link_shared_lib(objects, name, output_dir=libdir, export_symbols=exports, extra_preargs=extra_preargs, extra_postargs=extra_postargs, debug=DEBUG)


	if openblas is not None:
		shutil.copyfile(os.path.join('Z:/Larch',*openblas), os.path.join(shlib_folder(),openblas[-1]))
	if gfortran is not None:
		shutil.copyfile(os.path.join('Z:/Larch',*gfortran), os.path.join(shlib_folder(),gfortran[-1]))



	#simp = Extension('larch._larch',
	#				 ['src/larch_hello.i',] + simp_cpp_files,
	#				 swig_opts=['-modern', '-py3', '-I../include', '-v', '-c++', '-outdir', './py'] + local_swig_opts,
	#				 libraries=local_libraries+['larchsqlite', ],
	#				 library_dirs=local_library_dirs+[shlib_folder(),],
	#				 define_macros=local_macros,
	#				 include_dirs=local_includedirs + [numpy.get_include(), './src', './src/etk', './src/model', './sqlite', ],
	#				 extra_compile_args=local_extra_compile_args,
	#				 extra_link_args=local_extra_link_args,
	#				 )

	core = Extension('larch._core',
					 ['src/swig/elmcore.i',] + elm_cpp_files,
					 swig_opts=['-modern', '-py3', '-I../include', '-v', '-c++', '-outdir', './py',
								'-I../include', '-I./src/etk', '-I./src/model',
								'-I./src/data', '-I./src/sherpa', '-I./src/vascular', '-I./src/version',
								'-I./src/swig', '-I./sqlite', ] + local_swig_opts,
					 libraries=local_libraries+['larchsqlite', ],
					 library_dirs=local_library_dirs+[shlib_folder(),],
					 define_macros=local_macros,
					 include_dirs=local_includedirs + [numpy.get_include(), './src', './src/etk', './src/model',
														'./src/data', './src/sherpa', './src/vascular', './src/version',
														'./src/swig', './sqlite', ],
					 extra_compile_args=local_extra_compile_args,
					 extra_link_args=local_extra_link_args,
					 depends=['src/swig/elmcore.i',] + elm_cpp_h_files,
					 )

	apsw_extra_link_args = []
	if platform.system() == 'Darwin':
		apsw_extra_link_args += ['-install_name', '@loader_path/{}'.format(c.library_filename('apsw','shared'))]


	apsw = Extension('larch.apsw',
					 ['sqlite/apsw/apsw.c',],
					 libraries=['larchsqlite', ],
					 library_dirs=[shlib_folder(),],
					 define_macros=local_macros+[('EXPERIMENTAL','1')],
					 include_dirs= [ './sqlite', './sqlite/apsw', ],
					 extra_compile_args=local_apsw_compile_args,
	#				 extra_link_args=apsw_extra_link_args,
					 )


	import build_configuration
	build_configuration.write_build_info(build_dir=lib_folder(), packagename="larch")




	setup(name='larch',
		  version=VERSION,
		  package_dir = {'larch': 'py'},
		  packages=['larch', 'larch.examples', 'larch.test', 'larch.version', 'larch.util', 'larch.model_reporter'],
		  ext_modules=[core, apsw, ],
		  package_data={'larch':['data_warehouse/*.sqlite', 'data_warehouse/*.csv', 'data_warehouse/*.csv.gz']},
		  data_files=local_data_files,
		  install_requires=[
							"numpy >= 1.8.1",
							"scipy >= 0.14.0",
							"pandas >= 0.14.1",
							"python-docx >= 0.8.5",
							"sphinxcontrib-napoleon >= 0.4",
						],
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


