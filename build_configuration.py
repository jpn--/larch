#!/usr/bin/python
#
#  Copyright 2007-2017 Jeffrey Newman
#
#  This file is part of Larch.
#
#  Larch is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Larch is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Larch.  If not, see <http://www.gnu.org/licenses/>.
#

import sys
import os
import os.path
import time
import subprocess


def write_build_info(build_dir=None, packagename="larch"):

	if build_dir is None:
		build_dir = os.environ['CONFIGURATION_BUILD_DIR']
		built_file = os.path.join(build_dir, "built.py")
	else:
		built_file = os.path.join(build_dir, packagename, "built.py")

	try:
		build_configuration = os.environ['CONFIGURATION']
	except KeyError:
		build_configuration = "DEFAULT"


	try:
		f = open(built_file,'w')
		print ("writing built info to %s"%built_file)
	except IOError:
		built_file = os.path.join(build_dir, "built.py")
		f = open(built_file,'w')
		print ("writing built info to %s"%built_file)


	f.write("# -*- coding: utf-8 -*-\n")
	f.write("# This file automatically created while building Larch. Do not edit manually.\n")
	f.write("configuration='%s'\n"%build_configuration)
	f.write("time='%s'\n"%time.strftime("%I:%M:%S %p %Z"))
	f.write("date='%s'\n"%time.strftime("%d %b %Y"))
	f.write("day='%s'\n"%time.strftime("%A"))
	f.write("ymdh='%s'[3:]\n"%time.strftime("%Y%m%d%H"))

	f.write("from . import __version__ as version\nimport sys\n")
	f.write("""
	build='%s (%s, %s %s)'%(version,day,date,time)
	longversion='%s.%s%s'%(version,ymdh,sys.platform[0])
	from .apsw import apswversion, sqlitelibversion
	from .util import dicta
	versions = dicta({
	'larch':version,
	'apsw':apswversion(),
	'sqlite':sqlitelibversion(),
	})

	try:
		import numpy
		versions['numpy'] = numpy.version.version
	except:
		versions['numpy'] = 'failed'

	try:
		import scipy
		versions['scipy'] = scipy.version.version
	except:
		versions['scipy'] = 'failed'

	try:
		import pandas
		versions['pandas'] = pandas.__version__
	except:
		versions['pandas'] = 'failed'

	import sys
	versions['python'] = "{0}.{1}.{2} {3}".format(*(sys.version_info))
	build_config='Larch %s built on %s, %s %s'%(configuration,day,date,time)
	""".replace('\n\t','\n'))

	f.close()

# To update the version, run `git tag -a 3.10a-DEC2014 -m'version 3.0alpha, December 2014'`


if __name__=="__main__":
	import sys
	print("build_configuration:sys.argv=",sys.argv)
	if len(sys.argv)>=2:
		write_build_info(build_dir=sys.argv[1], packagename="larch")
