
import os
import glob

warehouse_directory = os.path.dirname(__file__)

def file_list():
	return list(glob.glob(os.path.join(warehouse_directory,"*")))

def example_file(f, missing_ok=False):
	fi = os.path.join(warehouse_directory,f)
	if not os.path.exists(fi) and not missing_ok:
		candidates = list(glob.glob(os.path.join(warehouse_directory,f"*{f}*")))
		if len(candidates)==1:
			return candidates[0]
		raise FileNotFoundError(f)
	return fi



