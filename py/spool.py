import os
import time
import sys
import shutil
import imp
import importlib.util
import pathlib

_latest_spooled_dir = None

def new_dir(base_dir, date_fmt="%Y-%m-%d", make_working_dir=True):
	global _latest_spooled_dir, time
	n = 1
	while 1:
		try:
			local_dir = os.path.join(base_dir,'{}~{:03d}'.format(time.strftime(date_fmt),n))
			os.makedirs(local_dir)
			break
		except FileExistsError:
			n += 1
	if make_working_dir:
		os.chdir(local_dir)
		sys.path.insert(0,local_dir)
	_latest_spooled_dir = local_dir
	with open(os.path.join(local_dir,'_spool_info.py'), 'w') as f:
		import time, platform
		t = time.time()
		f.write("creation_time = {}\n# ".format(t))
		f.write(time.strftime("%A, %B %d, %Y at %I:%M:%S %p", time.localtime(t)))
		f.write("\n")
		f.write("platform = '{}'\n".format(platform.platform()))
		f.write("node = '{}'\n".format(platform.node()))
		f.write("system = '{}'\n".format(platform.system()))
		f.write("machine = '{}'\n".format(platform.machine()))
	return local_dir





def populate(imp_name,dir=None):
	global _latest_spooled_dir
	if dir is None:
		dir = _latest_spooled_dir
	dir = pathlib.Path(dir)
	spec = importlib.util.find_spec(imp_name)
	if spec is None:
		raise ImportError("module "+str(imp_name)+" not found")
	modulepath = pathlib.Path(spec.origin)
	if modulepath.stem == "__init__":
		# The populator is a package
		ignore = shutil.ignore_patterns("*__pycache__*")
		pkg_dir = shutil.copytree(str(pathlib.Path(*modulepath.parts[:-1])), str(dir/imp_name), ignore=ignore)
		return pkg_dir
	else:
		# The populator is a module
		shutil.copy2(spec.origin, str(dir/spec.name) )
		return dir/spec.name
	raise NotImplementedError("only packages are currently implemented")
