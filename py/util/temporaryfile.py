
import tempfile, webbrowser, types, os, shutil



TemporaryBucket = []

def TemporaryBucketCleanUp():
	global TemporaryBucket
	for i in TemporaryBucket:
		try:
			os.remove(os.path.realpath(i.name))
		except PermissionError:
			try:
				shutil.rmtree(i.name)
			except:
				import traceback
				traceback.print_exc()
		except:
			import traceback
			traceback.print_exc()
	del TemporaryBucket

import atexit
atexit.register(TemporaryBucketCleanUp)

def _open_in_chrome_or_something(url):
	# MacOS
	if os.path.exists('/Applications/Google Chrome.app'):
		chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
	# Windows
	elif os.path.exists('C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'):
		chrome_path = 'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe %s'
	# Linux
	elif os.path.exists('/usr/bin/google-chrome'):
		chrome_path = '/usr/bin/google-chrome %s'
	else:
		chrome_path = None
	if chrome_path is None:
		webbrowser.open(url)
	else:
		webbrowser.get(chrome_path).open(url)

# Trap __exit__ to ensure the file does not get
# deleted when used in a with statement; to keep temporary files open
# until intentionally cleaned up
def _exit_without_closing(self, *arg, **kwarg):
	return self.file.__exit__(*arg, **kwarg)


def TemporaryFile(suffix='', mode='w+', use_chrome=True):
	t = tempfile.NamedTemporaryFile(suffix=suffix,mode=mode,delete=False)
	t.__exit__ = types.MethodType( _exit_without_closing, t )
	if use_chrome:
		t.view = types.MethodType( lambda self: _open_in_chrome_or_something('file://'+os.path.realpath(self.name)), t )
	else:
		t.view = types.MethodType( lambda self: webbrowser.open('file://'+os.path.realpath(self.name)), t )
	global TemporaryBucket
	TemporaryBucket.append(t)
	return t


def _try_write(self, content):
	try:
		self.write_(content)
	except:
		try:
			self.write_(str(content))
		except:
			self.write_(str(content).encode('utf-8'))


def TemporaryHtml(style=None, *, nohead=False, mode='wb+', content=None, **tagheads):
	t = TemporaryFile(suffix='.html', mode=mode)
	if 'b' in mode:
		t.write_ = t.write
		t.write = lambda x: _try_write(t,x)
	if not nohead and (style or len(tagheads)>0):
		t.write("<head>")
		if style:
			t.write("<style>{}</style>".format(style))
		for tag, content in tagheads.items():
			t.write("<{0}>{1}</{0}>".format(tag.lower(),content))
		t.write("</head>")
	if content is not None:
		t.write(content)
		t.view()
	return t


def TemporaryGzipInflation(gzfile):
	t = TemporaryFile(mode='wb')
	import gzip
	with gzip.open(gzfile, 'rb') as previewfile:
		t.write(previewfile.read())
	return t.name

_tempdir = None

def TemporaryDirectory(common=True):
	global _tempdir
	if common and _tempdir is not None:
		return _tempdir
	t = tempfile.mkdtemp()
	global TemporaryBucket
	from .attribute_dict import dictal
	TemporaryBucket.append(dictal(name=t))
	if common:
		_tempdir = t
	return t

def TemporaryCopy(sourcefile):
	tdir = TemporaryDirectory()
	shutil.copy2(sourcefile,tdir)
	return os.path.join(tdir, os.path.basename(sourcefile))

