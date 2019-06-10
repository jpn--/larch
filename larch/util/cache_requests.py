# -*- coding: utf-8 -*-

import appdirs
import joblib
import requests

cache_dir = None
memory = None

def set_cache_dir(location=None, compress=True, verbose=0, **kwargs):
	"""
	Set up a cache directory for use with requests.

	Parameters
	----------
	location: str or None or False
		The path of the base directory to use as a data store
		or None or False.  If None, a default directory is created
		using appdirs.user_cache_dir.
		If False is given, no caching is done and
		the Memory object is completely transparent.

	compress: boolean, or integer, optional
		Whether to zip the stored data on disk. If an integer is
		given, it should be between 1 and 9, and sets the amount
		of compression.

	verbose: int, optional
		Verbosity flag, controls the debug messages that are issued
		as functions are evaluated.

	bytes_limit: int, optional
		Limit in bytes of the size of the cache.

	"""
	global memory, cache_dir

	if location is None:
		location = appdirs.user_cache_dir('cached_requests')

	if location is False:
		location = None

	memory = joblib.Memory(location, compress=compress, verbose=verbose, **kwargs)

	make_cache = (
		(requests, 'get'),
		(requests, 'post'),
	)

	for module, func_name in make_cache:
		try:
			func = getattr(module, f"_{func_name}_orig")
		except AttributeError:
			func = getattr(module, func_name)
			setattr(module, f"_{func_name}_orig", func)
		setattr(module, func_name, memory.cache(func))





set_cache_dir()


