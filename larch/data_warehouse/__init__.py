
import os
import glob

warehouse_directory = os.path.dirname(__file__)

def file_list():
	return list(glob.glob(os.path.join(warehouse_directory,"*")))

def file_path(f):
	fi = os.path.join(warehouse_directory,f)
	if not os.path.exists(fi):
		raise FileNotFoundError(f)
	return fi



