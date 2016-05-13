######################################################### encoding: utf-8 ######
#
#  Copyright 2007-2016 Jeffrey Newman.
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

import sys, os.path, glob

exampledir = os.path.dirname(__file__)
exampledocdir = os.path.join(os.path.dirname(__file__),'doc')

examplefiles = {
	  1:	"mtc01",
	 17:	"mtc17",
	 22:	"mtc22",
	 80:	"itin80",
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


examplefiles_glob = glob.glob(os.path.join(exampledocdir,"[0-9][0-9][0-9]_*.rst"))
examplefiles_glob_x = glob.glob(os.path.join(exampledocdir,"[0-9][0-9][0-9][a-z]_*.rst"))

examplefiles_doc = { int(os.path.basename(g)[:3]): os.path.basename(g) for g in examplefiles_glob }
examplefiles_doc.update({ (os.path.basename(g)[:4]): os.path.basename(g) for g in examplefiles_glob_x })

class UnknownExampleCode(KeyError):
	pass

def get_examplefile(n):
	if isinstance(n,str):
		try:
			n = int(n)
		except ValueError:
			n = "{:0>4s}".format(n.casefold())
	if n in examplefiles_doc:
		return examplefiles_doc[n]
	else:
		raise UnknownExampleCode(n)



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
		f = os.path.join(exampledocdir, get_examplefile(n))
		print(_testcode_parsed(f))
	except UnknownExampleCode:
		try:
			module_name = examplefiles[n]
		except KeyError:
			from ..core import LarchError
			raise LarchError("Example no. %i not found"%n)
		filename = module_name+".py"
		with open(os.path.join(exampledir,filename)) as f:
			print(f.read())

def pseudofile(n):
	import io
	pf = io.StringIO()
	try:
		f = os.path.join(exampledocdir, get_examplefile(n))
		print(_testcode_parsed(f), file=pf)
	except UnknownExampleCode:
		pass
	pf.seek(0)
	return pf

def LL(n):
	'''Return the estimated log likelihood for example file number <n>.'''
	load_example(n)
	m = model(data())
	m.estimate()
	return m.loglike(cached=2)

def CheckLL():
	assert(round(LL(  1),3)==-3626.186)
	assert(round(LL(101),3)==-5331.252)
	assert(round(LL(102),3)==-5273.743)
	#assert(round(LL(104),3)==-5423.299)
	assert(round(LL(109),3)==-5236.900)
	assert(round(LL(114),3)==-5169.642)


def _testcode_iter(sourcefile):
	active = False
	with open(sourcefile, 'r') as f:
		for line in f:
			line = line.rstrip()
			if line == '.. testcode::':
				active = True
				continue
			if line == '':
				continue
			if line[0] not in (' ','\t'):
				active = False
			if active:
				if line[0]=='\t':
					yield line[1:]
				else:
					yield line[5:]
				continue

def _testcode_parsed(sourcefile):
	return "\n".join(_testcode_iter(sourcefile))

def _exec_example(sourcefile, d = None, extract='m'):
	_global = {}
	_local = {}
	from .. import larch
	_global['larch'] = larch
	if d is None:
		rawcode = _testcode_parsed(sourcefile)
		code = compile(rawcode, "<::testcode:{!s}>".format(sourcefile), 'exec')
		exec(code, _global, _local)
	else:
		_local['d'] = d
		for n, line in enumerate(_testcode_iter(sourcefile)):
			if len(line)>2 and line[:2]=='d=':
				continue
			if len(line)>3 and line[:3]=='d =':
				continue
			code = compile(line, sourcefile+":"+str(n), 'exec')
			exec(code, _global, _local)
	if isinstance(extract, str):
		return _local[extract]
	else:
		return tuple(_local[i] for i in extract)

def _exec_example_n(n, *arg, **kwarg):
	f = os.path.join(exampledocdir, get_examplefile(n))
	return _exec_example(f, *arg, **kwarg)

def reproduce(n, extract='m'):
	'''Run an example code section (from the documentation) and give the result.
	
	Parameters
	----------
	n : int
		The number of the example to reproduce.
	extract : str or iterable of str
		The name of the object of the example to extract.  By default `m`
		but it any named object that exists in the example namespace can
		be returned.  Give a list of `str`s to get multiple objects.
	'''
	return _exec_example_n(n, extract=extract)


