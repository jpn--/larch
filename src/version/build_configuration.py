#!/usr/bin/python
#
#  Copyright 2007-2015 Jeffrey Newman
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

try:
	build_configuration = os.environ['CONFIGURATION']
except KeyError:
	build_configuration = "DEFAULT"

if len(sys.argv) >= 2:
	packagename = sys.argv[1]
else:
	packagename = "larch"
print "packagename=",packagename

if len(sys.argv) >= 3:
	build_dir = sys.argv[2]
else:
	build_dir = os.environ['CONFIGURATION_BUILD_DIR']
print "build_dir=",build_dir


built_file = os.path.join(build_dir, packagename, "built.py")


try:
	f = open(built_file,'w')
	print ("writing built info to %s"%built_file)
except IOError:
	built_file = os.path.join(build_dir, "built.py")
	f = open(built_file,'w')
	print ("writing built info to %s"%built_file)


print sys.argv


f.write("# -*- coding: utf-8 -*-\n")
f.write("# This file automatically created while building Larch. Do not edit manually.\n")
f.write("configuration='%s'\n"%build_configuration)
f.write("time='%s'\n"%time.strftime("%I:%M:%S %p %Z"))
f.write("date='%s'\n"%time.strftime("%d %b %Y"))
f.write("day='%s'\n"%time.strftime("%A"))
if sys.version_info[0] >= 3:
	try:
		f.write("version='%s'\n" % subprocess.check_output(['git','describe','--long']).decode("utf-8").strip())
	except:
		f.write("version='%s'\n" % "XXX")
else:
	try:
		f.write("version='%s'\n" % subprocess.check_output(['git','describe','--long']).strip())
	except:
		f.write("version='%s'\n" % "XXX")
f.write("""
build='%s (%s, %s %s)'%(version,day,date,time)
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
	versions['pandas'] = pandas.version.version
except:
	versions['pandas'] = 'failed'

import sys
versions['python'] = "{0}.{1}.{2} {3}".format(*(sys.version_info))
build_config='Hangman %s built on %s, %s %s'%(configuration,day,date,time)
""")

f.close()

# To update the version, run `git tag -a 2.3a-NOV2013 -m'version 2.3alpha, November 2013'`