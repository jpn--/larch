######################################################### encoding: utf-8 ######
#
#  ELM is free, open source software to estimate discrete choice models.
#  
#  Copyright 2007-2015 Jeffrey Newman
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
################################################################################

import sys
import os, os.path

if os.environ.get('READTHEDOCS', None) == 'True':
	# hack for building docs on rtfd
	from .mock_module import Mock
	MOCK_MODULES = ['numpy', 'pandas', 'larch._core', 'larch.apsw']
	sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)



info = """Larch is free, open source software to estimate discrete choice models.
Copyright 2007-2015 Jeffrey Newman
This program is licensed under GPLv3 and comes with ABSOLUTELY NO WARRANTY."""

status = ""


try:
	status += "Python %s" % sys.version

	import struct
	status += "\nPython is running in %i bit mode" % ( 8 * struct.calcsize("P"))

	try:
		from .larch import * # testing...
	except:
		pass
	from . import logging
	from . import core
	core.larch_initialize()
	from . import exceptions
	from .db import DB
	from .core import Model, Parameter
	from .model import ModelFamily
	from . import array
	core._set_array_module(array)
	try:
		from .built import version, build, versions, build_config
		del built
	except (NameError, ImportError):
		version, build, versions, build_config = "","",{},""

	try:
		from . import linalg
		core.set_linalg(linalg)
	except AttributeError as err:
		print(err)
	except ImportError:
		from .mock_module import Mock
		linalg = Mock()

	status += "\nLarch "+build
	_directory_ = os.path.split(__file__)[0]
	status += "\nLoaded from %s" % _directory_

	larch = sys.modules[__name__]

except:
	print ("Exception in initializing Larch")
	print (status)
	raise

finally:
	del sys
	del os
#	try:
#		del array
#	except NameError:
#		pass
# End of init
