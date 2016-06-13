

import sys
import os
import os.path
import time
import subprocess



def execfile(filename):
	import __main__
	with open(filename) as f:
		code = compile(f.read(), filename, 'exec')
		exec(code, globals(), __main__.__dict__)




try:
	if sys.version_info[0] >= 3:
		ver = subprocess.check_output(['git','describe','--tags','--long']).decode("utf-8").strip()
	else:
		ver = subprocess.check_output(['git','describe','--tags','--long']).strip()
except subprocess.CalledProcessError:
	ver = '3.3.1'

if ver[0].lower() == 'v':
	ver = ver[1:]

try:
	version_file = os.path.join(os.environ['PROJECT_DIR'],'repository','py','version','__init__.py')
except KeyError:
	version_file = os.path.join(os.path.dirname(__file__),'py','version','__init__.py')

try:
	execfile(version_file)
except FileNotFoundError:
	version = None

if version != ver:
	with open(version_file,'w') as f:
		f.write("version='%s'\n" % ver)


