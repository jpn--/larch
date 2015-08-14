######################################################### encoding: utf-8 ######
#
#  Copyright 2007-2015 Jeffrey Newman.
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
################################################################################
#
#  This package contains example files for elm. To see what example modules are
#  available, examine the dictionary elm.examples.examplefiles.  To see the
#  results of any given module, run elm.examples.show(n) with n as the module's
#  key number. To print the raw script of the module, run elm.examples.tell(n)
#  or use any Python-friendly text editor to examine the files in the examples 
#  directory. 
#
################################################################################

import sys, os.path

examplefiles = {
	  1:	"mtc01",
	100:	"swissmetro00data",
	101:	"swissmetro01logit",
	102:	"swissmetro02weighted",
	104:	"swissmetro04transforms",
	109:	"swissmetro09nested",
	111:	"swissmetro11cnl",
	114:	"swissmetro14selectionBias",
}

examplefiles_pre = {
	  1:	"mtc01e",
}

exampledir = os.path.dirname(__file__)

from .. import Model, DB, _directory_
class larch:
	DB = DB
	Model = Model
	_directory_ = _directory_

def load_example(n, pre=False):
	if pre:
		try:
			module_name = examplefiles_pre[n]
		except KeyError:
			from ..core import LarchError
			raise LarchError("Example no. %i (pre-estimated) not found"%n)
	else:
		try:
			module_name = examplefiles[n]
		except KeyError:
			from ..core import LarchError
			raise LarchError("Example no. %i not found"%n)
	filename = module_name+".py"
	with open(os.path.join(exampledir,filename)) as f:
		code = compile(f.read(), filename, 'exec')
		exec(code, globals(), globals())

def show(n):
	'''Show the computed results of example file number <n>.'''
	load_example(n)
	m = model(data())
	m.estimate()
	print(m)
	
def tell(n):
	'''Print the raw script of example file number <n>.'''
	try:
		module_name = examplefiles[n]
	except KeyError:
		from ..core import LarchError
		raise LarchError("Example no. %i not found"%n)
	filename = module_name+".py"
	with open(os.path.join(exampledir,filename)) as f:
		print(f.read())

def LL(n):
	'''Return the estimated log likelihood for example file number <n>.'''
	load_example(n)
	m = model(data())
	m.estimate()
	return m.LL()

def CheckLL():
	assert(round(LL(  1),3)==-3626.186)
	assert(round(LL(101),3)==-5331.252)
	assert(round(LL(102),3)==-5273.743)
	#assert(round(LL(104),3)==-5423.299)
	assert(round(LL(109),3)==-5236.900)
	assert(round(LL(114),3)==-5169.642)
