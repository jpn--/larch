
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
examplefiles_glob_i = glob.glob(os.path.join(exampledocdir,"[0-9][0-9][0-9]_*.ipynb"))
examplefiles_glob_j = glob.glob(os.path.join(exampledocdir,"[0-9][0-9][0-9][a-z]_*.ipynb"))

examplefiles_doc = { int(os.path.basename(g)[:3]): os.path.basename(g) for g in examplefiles_glob }
examplefiles_doc.update({ (os.path.basename(g)[:4]): os.path.basename(g) for g in examplefiles_glob_x })
examplefiles_doc.update({ int(os.path.basename(g)[:3]): os.path.basename(g) for g in examplefiles_glob_i })
examplefiles_doc.update({ (os.path.basename(g)[:4]): os.path.basename(g) for g in examplefiles_glob_j })

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
	if sourcefile.endswith('.rst'):
		return "\n".join(_testcode_iter(sourcefile))
	elif sourcefile.endswith('.ipynb'):
		import yaml
		with open(sourcefile) as s:
			sourcefilecontent = s.read()
		y = yaml.safe_load(sourcefilecontent)
		sourcecode = "\n\n".join(["".join(i['source']) for i in y['cells'] if
								  (i['cell_type'] == 'code' and not i['metadata'].get('doc_only', False))])
		return sourcecode
	else:
		raise ValueError('unknown file type')

def _exec_example(sourcefile, d=None, extract='m', echo=False, larch=None):
	_global = {}
	_local = {}
	if larch is None:
		import importlib
		larch = importlib.import_module(__name__.split('.')[0])
	_global['larch'] = larch
	_global['lx'] = larch
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
			code = None
			try:
				code = compile(prev_line + line, sourcefile + ":" + str(n), 'exec')
			except SyntaxError as syntax:
				if 'unexpected EOF' in syntax.msg:
					prev_line += line + "\n"
			else:
				prev_line = ""
			if code is not None:
				exec(code, _global, _local)
	if isinstance(extract, str):
		try:
			return _local[extract]
		except KeyError:
			print("Known keys:", ", ".join(str(i) for i in _local.keys()))
			raise
	elif extract is not None:
		return tuple(_local[i] for i in extract)


def _exec_example_n(n, *arg, larch=None, **kwarg):
	f = os.path.join(exampledocdir, get_examplefile(n))
	return _exec_example(f, *arg, larch=larch, **kwarg)


def example(n, extract='m', echo=False, output_file=None, larch=None):
	'''Run an example code section (from the documentation) and give the result.

	Parameters
	----------
	n : int
		The number of the example to reproduce.
	extract : str or Sequence[str]
		The name of the object of the example to extract.  By default `m`
		but it any named object that exists in the example namespace can
		be returned.  Give a sequence of `str` to get multiple objects.
	echo : bool, optional
		If True, will echo the commands used.
	output_file : str, optional
		If given, check whether this named file exists.  If so,
		return the filename, otherwise run the example code section
		(which should as a side effect create this file).  Then
		check that the file now exists, raising a FileNotFoundError
		if it still does not exist.
	'''
	if output_file is not None:
		if os.path.exists(output_file):
			return output_file
		_exec_example_n(n, extract=None, echo=echo, larch=larch)
		if os.path.exists(output_file):
			return output_file
		else:
			raise FileNotFoundError(output_file)
	else:
		return _exec_example_n(n, extract=extract, echo=echo, larch=larch)
