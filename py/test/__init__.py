#
#  Copyright 2007-2016 Jeffrey Newman
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
import os.path
import unittest, nose
import math

### Test Case Class
class ELM_TestCase(unittest.TestCase):
	def assertSuccess(self, stringResponder):
		self.assertEqual("success", stringResponder[0:7])
	def assertIgnored(self, stringResponder):
		self.assertEqual("ignored", stringResponder[0:7])
	def assertArrayEqual(self, fixedVal, testVal):
		self.assertEqual(fixedVal.shape, testVal.shape)
		fixedValFlat = fixedVal.flatten()
		testValFlat = testVal.flatten()
		for i in range(len(fixedValFlat)):
			self.assertAlmostEqual(fixedValFlat[i],testValFlat[i])
	def assertNearlyEqual(self, x, y, sigfigs=3):
		magnitude = (abs(x) + abs(y)) / 2
		if math.isinf(magnitude): magnitude = 1
		self.assertAlmostEqual(x, y, delta=magnitude*(10**(-sigfigs)))

### Testing Data Directory and Files
TEST_DIR = os.path.join(os.path.split(__file__)[0],"data_warehouse")
if not os.path.exists(TEST_DIR):
	uplevels = 0
	uppath = ""
	while uplevels < 20 and not os.path.exists(TEST_DIR):
		uppath = uppath+ ".."+os.sep
		uplevels += 1
		TEST_DIR = os.path.join(os.path.split(__file__)[0],uppath+"data_warehouse")


TEST_DATA = {
  'MTC WORK MODE CHOICE':os.path.join(TEST_DIR,"MTCwork.sqlite"),
#  'MTCWORK-CSV':os.path.join(TEST_DIR,"MTC_WorkMode.csv"),
#  'MTCWORK-CSV-SHORT':os.path.join(TEST_DIR,"MTC_WorkMode_Short.csv"),
#  'TWIN CITY DEST CHOICE':os.path.join(TEST_DIR,"TwinCityQ.sqlite"),
#  'MSOM SIMULATION':os.path.join(TEST_DIR,"msom_sim.elmdata"),
#  'MICRODATA':os.path.join(TEST_DIR,"microdata.sql"),
#  'MICRODATA SQUEEZE':os.path.join(TEST_DIR,"microdata_squeeze.sql"),
#  'REFERENCE OUTPUT':os.path.join(TEST_DIR,"reference_output.txt"),
  'SWISSMETRO-CSV':os.path.join(TEST_DIR,"swissmetro.csv"),
  }





NOSE_PROCESSES = 1
NOSE_VERBOSITY = 1
NOSE_STOP_ON_FAIL = False
NOSE_TESTS = None
DEEP_TEST = False
USE_NOSE = True

if NOSE_VERBOSITY>2:
	skriver_level(10, silently=False)

_multiprocess_shared_ = True

def simple():
	import nose
	return nose.run(argv=['-','--where='+os.path.split(__file__)[0],'--verbosity=3'])

def run(exit=False):
	print("<"*30,"larch.test",">"*30)
	print("FROM:",os.path.split(__file__)[0])
	try:
		from ..built import build, versions, build_config
	except (NameError, ImportError):
		build, versions, build_config = "",{},""
	print(versions)
	print("BUILD:", build)
	print("CONFIG:", build_config)
	print(">"*30,"larch.test","<"*30)
	result = simple()
	if exit:
		import sys
		sys.exit(0 if result else -1)
	else:
		return result
