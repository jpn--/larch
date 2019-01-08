#!/usr/bin/env python
import os
import zipfile
import hashlib
import time


def _rec_split(s):
	rest, tail = os.path.split(s)
	if rest in ('', os.path.sep):
		return tail,
	return _rec_split(rest) + (tail,)


def _any_dot(s):
	for i in _rec_split(s):
		if len(i) > 0 and i[0] == '.':
			return True
	return False


def _zipdir(path, ziph, skip_dots=True, extra_layer=True, log=print):
	# ziph is zipfile handle
	keep_dots = not skip_dots
	for root, dirs, files in os.walk(path):
		folder = os.path.basename(root)
		if keep_dots or not _any_dot(folder):
			log(f'zipping folder: {folder} in {root}')
			for file in files:
				if keep_dots or not _any_dot(file):
					ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path,
																												'..' if extra_layer else '.')))
		else:
			log(f'not zipping folder: {folder} in {root}')


def zipdir(source_dir, zip_file_name=None, skip_dots=True, extra_layer=False, log=print):
	"""

	Parameters
	----------
	source_dir
	zip_file_name : str
		If not given, uses the name of the sourcedir.
	skip_dots : bool, defaults True
		Ignore files and dirs that start with a dot.

	Returns
	-------
	str
		zip_file_name
	"""
	if zip_file_name is None:
		if source_dir[-1] in ('/', '\\'):
			usepath = source_dir[:-1]
		else:
			usepath = source_dir
		zip_file_name = usepath + '.zip'
	with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
		_zipdir(source_dir, zipf, skip_dots=skip_dots, extra_layer=extra_layer, log=log)
	return zip_file_name


def zipmod(module, zip_file_name, skip_dots=True):
	"""
	Create a zipfile from a module

	Parameters
	----------
	module
	zip_file_name
	skip_dots

	Returns
	-------

	"""
	with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
		_zipdir(module.__path__[0], zipf, skip_dots=skip_dots)


def zipmod_temp(module, skip_dots=True):
	import tempfile
	tempdir = tempfile.TemporaryDirectory()
	zip_file_name = os.path.join(tempdir.name, module.__name__ + ".zip")
	zipmod(module, zip_file_name, skip_dots=skip_dots)
	return zip_file_name, tempdir


def make_hash_file(fname):
	hash256 = hashlib.sha256()
	if fname[-3:] == '.gz':
		import gzip
		with gzip.open(fname, "rb") as f:
			for chunk in iter(lambda: f.read(4096), b""):
				hash256.update(chunk)
		h = hash256.hexdigest()
		with open(fname[:-3] + ".sha256.txt", "w") as fh:
			fh.write(h)
	else:
		with open(fname, "rb") as f:
			for chunk in iter(lambda: f.read(4096), b""):
				hash256.update(chunk)
		h = hash256.hexdigest()
		with open(fname + ".sha256.txt", "w") as fh:
			fh.write(h)


def verify_hash_file(fname, hash_dir=None, max_retries=5):
	hash256 = hashlib.sha256()

	retries = 0
	while retries < max_retries:
		try:
			with open(fname, "rb") as f:
				for chunk in iter(lambda: f.read(4096), b""):
					hash256.update(chunk)
		except PermissionError:
			import time
			time.sleep(5)
			retries += 1
		except:
			raise
		else:
			break

	h = hash256.hexdigest()
	if hash_dir is None:
		with open(fname + ".sha256.txt", "r") as fh:
			h_x = fh.read()
	else:
		with open(os.path.join(hash_dir, os.path.basename(fname) + ".sha256.txt"), "r") as fh:
			h_x = fh.read()
	if h != h_x:
		if hash_dir:
			raise ValueError(f"bad hash on {fname} with hash_dir={hash_dir}")
		else:
			raise ValueError(f"bad hash on {fname}")


def archive_subdir(destination_dir):
	import time, os
	destination_dir = os.path.join(destination_dir, time.strftime("%Y_%m_%d-%H_%M"))
	os.makedirs(destination_dir, exist_ok=True)
	return destination_dir

def gzip_dir(source_dir, patterns=("*.*",), make_hash=True, exclude=".sha256.txt", destination_dir=None, archive_subdir=False, remove_original=True, log=None):
	"""Individually gzip every file matching patterns in source_dir."""

	import gzip, glob
	import shutil, os, time

	if destination_dir is None:
		destination_dir = source_dir
	else:
		os.makedirs(destination_dir, exist_ok=True)

	if archive_subdir:
		if isinstance(archive_subdir, str):
			destination_dir = os.path.join(destination_dir, archive_subdir)
		else:
			destination_dir = os.path.join(destination_dir, time.strftime("%Y_%m_%d-%H_%M"))
		os.makedirs(destination_dir, exist_ok=True)

	targets = set()

	for pattern in patterns:
		for f in glob.glob(os.path.join(source_dir, pattern)):
			if exclude in f:
				continue  # don't re-gzip the hash files by default
			targets.add(f)

	for f in targets:
		if make_hash:
			make_hash_file(f)
		f_relative = os.path.relpath(f, start=source_dir)

		if f[-3:] != '.gz':
			with open(f, 'rb') as f_in:
				f_to = os.path.join(destination_dir, f_relative)
				with gzip.open( f_to + '.gz', 'wb') as f_out:
					if log is not None:
						log(f"{f} --> {f_to}.gz")
					shutil.copyfileobj(f_in, f_out)
			if remove_original:
				os.remove(f)

def gzip_archive(source_dir, patterns=("*.*",), make_hash=False, exclude=".sha256.txt", destination_dir=None, archive_subdir=True, remove_original=False, log=None):
	"""Same as gzip_dir but different default inputs.

	Parameters
	----------
	source_dir
	patterns
	make_hash
	exclude
	destination_dir
	archive_subdir
	remove_original

	Returns
	-------

	"""
	return gzip_dir(source_dir=source_dir, patterns=patterns,
					make_hash=make_hash, exclude=exclude,
					destination_dir=destination_dir, archive_subdir=archive_subdir,
					remove_original=remove_original,
					log=log)



class Archiver():

	def __init__(self, *outdirs, log=None, handle=None):
		if handle is None:
			self._handle = time.strftime("%Y_%m_%d-%H_%M")
		else:
			self._handle = handle
		self.outdirs = []
		self.archive_dirs = []
		for od in outdirs:
			self.add_outdir(od)
		if log is None:
			from .logs.normal import log as _log
			self.log = _log.info
		elif log is False:
			self.log = None
		else:
			self.log = log

	def add_outdir(self, outdir):
		self.outdirs.append(outdir)
		d = os.path.join(outdir, self._handle)
		os.makedirs(d, exist_ok=True)
		self.archive_dirs.append(d)

	def gzip_by_glob(self, pattern, source_dir='.'):
		for arch in self.archive_dirs:
			gzip_archive(
				source_dir=source_dir,
				patterns=(pattern,),
				destination_dir=arch,
				archive_subdir=False,
				log=self.log,
			)

	def zip_dir(self, source_dir, zip_file_name=None, **kwargs):
		if zip_file_name is None:
			zip_file_name = os.path.basename(source_dir)+".zip"
		for arch in self.archive_dirs:
			zipdir(
				source_dir=source_dir,
				zip_file_name=os.path.join(arch, zip_file_name),
				log=self.log,
				**kwargs
			)

	zip_by_dir = zip_dir