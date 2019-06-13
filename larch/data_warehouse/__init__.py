
import os
import glob

warehouse_directory = os.path.dirname(__file__)

def file_list():
	return list(glob.glob(os.path.join(warehouse_directory,"*")))

def example_file(f, missing_ok=False, rel=False):
	"""
	Get the file path of a particular example file.

	All example files are stored in larch/data_warehouse.

	Parameters
	----------
	f : str
		The base filename to find.
	missing_ok : bool, default False
		Do not raise a FileNotFoundError if the file is missing. This
		is generally only useful in larch scripts that will create the
		example file, not for end users, who generally should not
		create new data files in the data_warehouse directory.
	rel : bool or str, default False
		Return the file path relative to this directory, or if given as
		True, relative to the current working directory.

	Returns
	-------
	str
	"""
	fi = os.path.join(warehouse_directory,f)
	if not os.path.exists(fi) and not missing_ok:
		candidates = list(glob.glob(os.path.join(warehouse_directory,f"*{f}*")))
		if len(candidates)==1:
			return candidates[0]
		raise FileNotFoundError(f)
	if rel is True:
		rel = os.getcwd()
	if rel:
		fi = os.path.relpath(fi, start=rel)
	return fi



