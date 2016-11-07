import os

class dir_maker():
	def __init__(self, basedir):
		self._basedir = basedir
	def __getattr__(self,key):
		newdir = os.path.join(self._basedir, key)
		return dir_maker(newdir)
	def __call__(self, filename=''):
		newfile = os.path.join(self._basedir, filename)
		try:
			os.makedirs(self._basedir)
		except FileExistsError:
			pass
		return newfile


try:
	import appdirs
except ImportError:
	from .temporaryfile import TemporaryDirectory
	_cache = TemporaryDirectory()
else:
	_cache = appdirs.user_cache_dir('Larch')



cache = dir_maker(_cache)

project_cache = cache

