
import sys, os, os.path, glob

from ..data_services.examples import MTC, EXAMPVILLE, SWISSMETRO


exampledir = os.path.dirname(__file__)

exampledocdir = os.path.join(os.path.dirname(__file__),'doc','example')
if not os.path.exists(exampledocdir):
	exampledocdir = os.path.join(os.path.dirname(__file__),'..','doc','example')
if not os.path.exists(exampledocdir):
	exampledocdir = os.path.join(os.path.dirname(__file__),'..','..','doc','example')
if not os.path.exists(exampledocdir):
	exampledocdir = os.path.join(os.path.dirname(__file__),'..','..','..','doc','example')

examplefiles = {}

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
			n = "{:0>4s}".format(n.lower())
	if n in examplefiles_doc:
		return examplefiles_doc[n]
	else:
		raise UnknownExampleCode(n)


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
			if line == '\t:hide:':
				continue
			if line[0] not in (' ', '\t'):
				active = False
			if active:
				if line[0] == '\t':
					yield line[1:]
				else:
					yield line[5:]
				continue


def _testcode_parsed(sourcefile):
	return "\n".join(_testcode_iter(sourcefile))


def _exec_example(sourcefile, d=None, extract='m', echo=False):
	_global = {}
	_local = {}
	larch = __import__(__name__.split('.')[0])
	_global['larch'] = larch
	_global['sys'] = sys
	_global['os'] = os
	from ..roles import P, X, PX
	_global['P'] = P
	_global['X'] = X
	_global['PX'] = PX
	if d is None:
		rawcode = _testcode_parsed(sourcefile)
		if echo:
			print("#" * 80)
			print(rawcode)
			print("#" * 80)
		code = compile(rawcode, "<::testcode:{!s}>".format(sourcefile), 'exec')
		exec(code, _global, _local)
	else:
		_local['d'] = d
		prev_line = ""
		for n, line in enumerate(_testcode_iter(sourcefile)):
			if len(line) > 2 and line[:2] == 'd=':
				continue
			if len(line) > 3 and line[:3] == 'd =':
				continue
			try:
				code = compile(prev_line + line, sourcefile + ":" + str(n), 'exec')
			except SyntaxError as syntax:
				if 'unexpected EOF' in syntax.msg:
					prev_line += line + "\n"
			else:
				prev_line = ""
			exec(code, _global, _local)
	if isinstance(extract, str):
		return _local[extract]
	else:
		return tuple(_local[i] for i in extract)


def _exec_example_n(n, *arg, **kwarg):
	f = os.path.join(exampledocdir, get_examplefile(n))
	return _exec_example(f, *arg, **kwarg)


def example(n, extract='m', echo=False):
	'''Run an example code section (from the documentation) and give the result.

	Parameters
	----------
	n : int
		The number of the example to reproduce.
	extract : str or iterable of str
		The name of the object of the example to extract.  By default `m`
		but it any named object that exists in the example namespace can
		be returned.  Give a list of `str` to get multiple objects.
	echo : bool, optional
		If True, will echo the commands used.
	'''
	return _exec_example_n(n, extract=extract, echo=echo)
