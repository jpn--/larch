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

import os.path, os, pickle, zlib, glob
import numpy
from . import logging
try:
	from . import apsw
except ImportError:
	from .mock_module import Mock
	apsw = Mock()
	class Dummy(): pass

from .core import LarchError

_uidn = 0

def uid():
	global _uidn
	_uidn += 1
	return "rx{}".format(_uidn)

from .util.persistent import stored_dict
from .util.temporaryfile import TemporaryFile, TemporaryHtml
from .util.attribute_dict import dicta
from .util.pmath import category, rename


def random_sample(a, size=None, replace=False, p=None):
	""" Generates a random sample from a given 1-D array, using numpy.random.choice

		Parameters
		----------
		a : 1-D array-like or int
			If an ndarray, a random sample is generated from its elements. 
			If an int, the random sample is generated as if a was np.arange(n)
		size : int or tuple of ints, optional
			Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. 
			Default is None, in which case a single value is returned.
		replace : boolean, optional
			Whether the sample is with or without replacement
		p : 1-D array-like, optional
			The weights associated with each entry in a. If not given the sample assumes 
			a uniform distribution over all entries in a.
			Unlike numpy.random.choice, the sum of p need not equal 1; the weights
			are automatically scaled into probabilities.
			
		Returns
		-------
		samples : 1-D ndarray, shape (size,)
			The generated random samples
		
		Raises
		------
		ValueError
			If a is an int and less than zero, 
			if a or p are not 1-dimensional, 
			if a is an array-like of size 0, 
			if p contains any negative weights,
			if a and p have different lengths, 
			or if replace=False and the sample size is greater than the population size
	"""
	if p is not None:
		return numpy.random.choice(a, size, replace, p=numpy.array(p)/numpy.array(p).sum())
	else:
		return numpy.random.choice(a, size, replace)





def webpage(content, *, file=None, title=None):
	'''A convenience function for writing a html file.
		
		:param content:   The HTML content.
		:param file:      A file-like object whereupon to write the HTML table. If None
		(the default), a temporary named html file is created.
		:param file:      An optional title to put at the top of the page as an <h1>.
		'''
	if file is None:
		file = TemporaryHtml("table {border-collapse:collapse;} table, th, td {border: 1px solid #999999; padding:2px;}")
	file.write("<body>\n")
	if title:
		file.write("<h1>{}</h1>\n".format(title))
	file.write(content)
	file.write("</body>\n")
	file.flush()
	try:
		file.view()
	except AttributeError:
		pass
	return file




class FrozenClass(object):
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "cannot create the new attribute '%s' for %s" % (str(key),str(type(self))) )
        super(FrozenClass, self).__setattr__(key, value)
    def _freeze(self):
        self.__isfrozen = True


def priority_iterator(unsorted, priorities=[]):
	for key in priorities:
		if key in unsorted:
			yield key
	for key in unsorted:
		if key not in priorities:
			yield key
	raise StopIteration
	
def path_shrink_until_exists(pth):
	import os.path
	from .util.filemanager import filename_split
	cap = 0
	while cap < 30 and not os.path.exists(pth):
		cap += 1
		pth = os.path.normpath( os.path.join(pth,"..") )
	p,b,ex = filename_split(pth)
	if ex==".zip":
		pth = os.path.normpath( os.path.join(pth,"..") )
	return pth



from collections import OrderedDict
import inspect

def latex_table(keys, source, topline=None, bottomline=None, heads=True, headunderline=None):
	if not isinstance(keys,OrderedDict):
		_k = OrderedDict()
		for key in keys:
			_k[key] = None
		keys = _k
	z = "\\begin{tabular}{"
	for key, val in keys.items():
		try:
			z += val['align']
		except (KeyError, TypeError):
			z += "c"
	z += "}\n"
	if topline is not None:
		z += topline + "\n"
	if heads:
		for key, val in keys.items():
			try:
				h = val['head']
			except (KeyError, TypeError):
				h = ""
			z += h + ' & '
		z = z[:-3]+" \\\\\n"
		if headunderline is not None:
			z += headunderline + "\n"
	for row in source:
		for key, val in keys.items():
			try:
				fmt = val['format']
			except (KeyError, TypeError):
				fmt = "{}"
			i = row[key]
			if inspect.isroutine(i): i = i()
			if isinstance(i,basestring): i = i.replace("_","\_")
			z += fmt.format(i) + ' & '
		z = z[:-3]+" \\\\\n"
	if bottomline is not None:
		z += bottomline + "\n"
	z += "\\end{tabular}"	
	return z

_sqlite_keywords = set([ 'ABORT', 'ACTION', 'ADD', 'AFTER', 'ALL', 'ALTER',
	'ANALYZE', 'AND', 'AS', 'ASC', 'ATTACH', 'AUTOINCREMENT', 'BEFORE', 'BEGIN',
	'BETWEEN', 'BY', 'CASCADE', 'CASE', 'CAST', 'CHECK', 'COLLATE', 'COLUMN',
	'COMMIT', 'CONFLICT', 'CONSTRAINT', 'CREATE', 'CROSS', 'CURRENT_DATE',
	'CURRENT_TIME', 'CURRENT_TIMESTAMP', 'DATABASE', 'DEFAULT', 'DEFERRABLE',
	'DEFERRED', 'DELETE', 'DESC', 'DETACH', 'DISTINCT', 'DROP', 'EACH', 'ELSE',
	'END', 'ESCAPE', 'EXCEPT', 'EXCLUSIVE', 'EXISTS', 'EXPLAIN', 'FAIL', 'FOR',
	'FOREIGN', 'FROM', 'FULL', 'GLOB', 'GROUP', 'HAVING', 'IF', 'IGNORE',
	'IMMEDIATE', 'IN', 'INDEX', 'INDEXED', 'INITIALLY', 'INNER', 'INSERT',
	'INSTEAD', 'INTERSECT', 'INTO', 'IS', 'ISNULL', 'JOIN', 'KEY', 'LEFT',
	'LIKE', 'LIMIT', 'MATCH', 'NATURAL', 'NO', 'NOT', 'NOTNULL', 'NULL', 'OF',
	'OFFSET', 'ON', 'OR', 'ORDER', 'OUTER', 'PLAN', 'PRAGMA', 'PRIMARY', 'QUERY',
	'RAISE', 'REFERENCES', 'REGEXP', 'REINDEX', 'RELEASE', 'RENAME', 'REPLACE',
	'RESTRICT', 'RIGHT', 'ROLLBACK', 'ROW', 'SAVEPOINT', 'SELECT', 'SET',
	'TABLE', 'TEMP', 'TEMPORARY', 'THEN', 'TO', 'TRANSACTION', 'TRIGGER',
	'UNION', 'UNIQUE', 'UPDATE', 'USING', 'VACUUM', 'VALUES', 'VIEW', 'VIRTUAL',
	'WHEN', 'WHERE'
])

def sqlite_keyword_check(trial_word):
	if trial_word.upper() in _sqlite_keywords:
		return "{:s}_".format(trial_word)
	return trial_word

def sqlite_keyword_check_list(trial_list):
	for i in range(len(trial_list)):
		trial_list[i] = sqlite_keyword_check(trial_list[i])




class SmartFileReader(object):
	def __init__(self, file, *args, **kwargs):
		if file[-3:]=='.gz':
			import gzip
			import struct
			with open(file, 'rb') as f:
				f.seek(-4, 2)
				self._filesize = struct.unpack('I', f.read(4))[0]
			self.file = gzip.open(file, 'rt', *args, **kwargs)
		else:
			import os
			self.file = open(file, 'rt', *args, **kwargs)
			self._filesize = os.fstat(self.file.fileno()).st_size
	def __getattr__(self, name):
		return getattr(self.file, name)
	def __setattr__(self, name, value):
		if name in ['file', 'percentread', '_filesize']:
			return object.__setattr__(self, name, value)
		return setattr(self.file, name, value)
	def __delattr__(self, name):
		return delattr(self.file, name)
	def percentread(self):
		return (float(self.file.tell())/float(self._filesize)*100)
	def __iter__(self):
		return self.file.__iter__()


def interpret_header_name(h):
	eL = logging.getScriber("util")
	if h.lower() == "case" or h[1:].lower() == "case":
		eL.warning("'case' is a reserved keyword, changing column header to 'caseid'")
		h = h+"id"
	if h.lower() == "group" or h[1:].lower() == "group":
		eL.warning("'group' is a reserved keyword, changing column header to 'groupid'")
		h = h+"id"
	if h.upper() in _sqlite_keywords:
		eL.warning("'%s' is a reserved keyword, changing column header to '%s_'"%(h,h))
		h = h+"_"
	if '.' in h:
		h = h.replace('.','_')
		eL.warning("dots are not allowed in column headers, changing column header to '%s'"%(h,))
	if len(h)==0:
		eL.error("zero length header")
		return " "
	if h[0] == '@':
		return h[1:]+" TEXT"
	elif h[0] == '#':
		return h[1:]+" INTEGER"
	return h+" NUMERIC"



def prepare_import_headers(rawfilename, headers=None):
	'''
	Identify the file type and read the column headers from the first row.
	For each column, data type is presumed to be FLOAT unless the column name is
	prepended with '@' or '#' which makes the column format TEXT or INT respectively.
	Since data is stored in SQLite, format is just a suggestion anyhow.
	
	Input: 
	  rawfile: the filename (absolute or relative path) to the text file with data.
	Output:
	  headers: a series of 2-tuples identifying column names and data types
	  r: the csv reader object, primed to begin reading at the second line of the file
	'''
	import csv
	eL = logging.getScriber("util")
	if isinstance(rawfilename,str):
		raw = SmartFileReader(rawfilename)
	else:
		raw = rawfilename
	sample = raw.read(28192)
	raw.seek(0)
	if isinstance(sample, bytes):
		sample = sample.decode('UTF-8')
	try:
		dialect = csv.Sniffer().sniff(sample)
	except csv.Error:
		sample = raw.read(512000)
		raw.seek(0)
		if isinstance(sample, bytes):
			sample = sample.decode('UTF-8')
		dialect = csv.Sniffer().sniff(sample)
	r = csv.reader(raw, dialect)
	eL.debug("DIALECT = %s",str(dialect))
	eL.debug("TYPE OF READER = %s",str(type(r)))
	if headers is None:
		try:
			headers = r.next()
		except AttributeError:
			headers = next(r) ##
	# fix header items for type
	num_cols = len(headers)
	for h in range(len(headers)):
		headers[h] = interpret_header_name(headers[h])	
	return headers, r, raw



def format_seconds(seconds):
	if seconds < 60.0:
		txt = "{:.3f}".format(seconds)
		fmt = "seconds"
	else:
		minutes, seconds = divmod(seconds, 60)
		seconds, fracsec = divmod(seconds, 1)
		if minutes < 60.0:
			txt = "{:.0f}:{:02.0f}{}".format(minutes,seconds,("%.3f" % fracsec).lstrip('0'))
			fmt = "minutes:seconds"
		else:
			hours, minutes = divmod(minutes, 60)
			txt = "{:.0f}:{:02.0f}:{:02.0f}{}".format(hours,minutes,seconds,("%.3f" % fracsec).lstrip('0'))
			fmt = "hours:minutes:seconds"
	return (txt,fmt)




## Storage

from enum import Enum
class storage_format(Enum):
	plain = 0
	pickle = 1
	picklezip = 2
	string = 3
	stringzip = 4

def storage_encode(item, form):
	assert( isinstance(form, storage_format) )
	if form==storage_format.plain:
		return item
	if form==storage_format.pickle:
		return pickle.dumps(item)
	if form==storage_format.picklezip:
		return zlib.compress(pickle.dumps(item))
	if form==storage_format.string:
		assert( isinstance(item, str) )
		return item
	if form==storage_format.stringzip:
		assert( isinstance(item, str) )
		return zlib.compress(item.encode())

def storage_decode(encoded_item, form):
	assert( isinstance(form, storage_format) )
	if form==storage_format.plain:
		return encoded_item
	if form==storage_format.pickle:
		return pickle.loads(encoded_item)
	if form==storage_format.picklezip:
		try:
			try:
				return pickle.loads(zlib.decompress(encoded_item))
			except:
				return pickle.loads(zlib.decompress(encoded_item).replace(b'elm.',b'larch.'))
		except:
			print("error in processing:")
			print(zlib.decompress(encoded_item))
			raise
	if form==storage_format.string:
		return encoded_item
	if form==storage_format.stringzip:
		return zlib.decompress(encoded_item).decode()

class storage():
	def __init__(self, db, name=None, *, reverse_index=False, key_format="", value_format=""):
		if name is None:
			name = "larch_genericstorage"
		if db is None:
			from . import DB
			db = DB()
		if isinstance(db,str):
			from . import DB
			db = DB(db)
		self._db = db
		self._name = name
		self._key_format = key_format
		self._value_format = value_format
		self._cur = db.cursor()
		self._create_table()
		if reverse_index:
			self.reverse_index()
	def _create_table(self):
		self._cur.execute("""CREATE TABLE IF NOT EXISTS {} (
		id {} PRIMARY KEY, 
		value {}, 
		form INTEGER DEFAULT 0)""".format(self._name,self._key_format,self._value_format))
		if "elm_store" in self._db.all_table_names():
			self._cur.execute("INSERT OR IGNORE INTO {} SELECT * FROM elm_store;".format(self._name))
	def __getattr__(self, item):
		return self[item]
	def __setattr__(self, item, value):
		if item[0]=='_':
			super().__setattr__(item, value)
			return
		self[item] = value
	def __delitem__(self, key):
		try:
			for row in self._cur.execute("SELECT EXISTS (SELECT id FROM {} WHERE id=?);".format(self._name), (key,)):
				if not row[0]:
					raise KeyError("key '{}' not found".format(key))
			self._cur.execute("DELETE FROM {} WHERE id=?".format(self._name), (key,))
		except apsw.SQLError as err:
			if "no such table: larch_genericstorage" in str(err):
				self._create_table()
				raise KeyError("key '{}' not found".format(key))
			else:
				raise
	def __getitem__(self, key):
		if len(key) > 2 and key[:2]=='?!':
			key = key[2:]
			try:
				for row in self._cur.execute("SELECT value, form FROM {} WHERE id=? LIMIT 1".format(self._name), (key,)):
					return (row[0], storage_format(row[1]))
				raise KeyError("key '{}' not found".format(key))
			except apsw.SQLError as err:
				if "no such table: larch_genericstorage" in str(err):
					self._create_table()
					for row in self._cur.execute("SELECT value, form FROM {} WHERE id=? LIMIT 1".format(self._name), (key,)):
						return (row[0], storage_format(row[1]))
					raise KeyError("key '{}' not found".format(key))
				else:
					raise
		else:
			try:
				for row in self._cur.execute("SELECT value, form FROM {} WHERE id=? LIMIT 1".format(self._name), (key,)):
					return storage_decode(row[0], storage_format(row[1]))
				raise KeyError("key '{}' not found".format(key))
			except apsw.SQLError as err:
				if "no such table: larch_genericstorage" in str(err):
					self._create_table()
					for row in self._cur.execute("SELECT value, form FROM {} WHERE id=? LIMIT 1".format(self._name), (key,)):
						return storage_decode(row[0], storage_format(row[1]))
					raise KeyError("key '{}' not found".format(key))
				else:
					raise
	def __setitem__(self, key, value):
		f = storage_format.plain
		if isinstance(value, str):
			f = storage_format.string
			zvalue = storage_encode(value, storage_format.stringzip)
			if len(zvalue) < len(value):
				f = storage_format.stringzip
				value = zvalue
		elif not isinstance(value, (int, float)):
			try:
				pkl = pickle.dumps(value)
				zpkl = zlib.compress(pkl)
				if len(zpkl) < len(pkl):
					f = storage_format.picklezip
					value = zpkl
				else:
					f = storage_format.pickle
					value = pkl
			except pickle.PickleError:
				pass
				raise
			except zlib.error:
				f = storage_format.pickle
				value = pkl
		try:
			self._cur.execute("INSERT OR REPLACE INTO {} VALUES (?,?,?)".format(self._name),(key,value,f.value))
		except apsw.SQLError as err:
			if "no such table: larch_genericstorage" in str(err):
				self._create_table()
				self._cur.execute("INSERT OR REPLACE INTO {} VALUES (?,?,?)".format(self._name),(key,value,f.value))
			else:
				raise
	def begin_transaction(self):
		self._cur.execute("BEGIN TRANSACTION;")
	def end_transaction(self):
		self._cur.execute("END TRANSACTION;")
	def reverse_index(self):
		self._cur.execute("CREATE INDEX IF NOT EXISTS {name}_reverse ON {name} (value)".format(name=self._name))
	def reverse_lookup(self, value, all=False):
		cur = self._cur.execute("SELECT id FROM {0} WHERE value=?".format(self._name),(value,))
		if all:
			return [i[0] for i in cur]
		else:
			return next(cur,[None])[0]
	def keys_like(self, category):
		if "'" in category:
			category = category.replace("'","\'")
		try:
			cur = self._cur.execute("SELECT SUBSTR(id,{2}) FROM {0} WHERE id LIKE '{1}:%'".format(self._name,category,len(category)+2))
		except apsw.SQLError as err:
			if "no such table: larch_genericstorage" in str(err):
				self._create_table()
				return []
			else:
				raise
		return [i[0] for i in cur]
	def __len__(self):
		for row in self._cur.execute("SELECT count(*) FROM {}".format(self._name)):
			return row[0]
	def __contains__(self,key):
		for row in self._cur.execute("SELECT EXISTS(SELECT 1 FROM {} WHERE id=? LIMIT 1)".format(self._name), (key,)):
			return row[0]
	def keys(self):
		cur = self._db.cursor()
		for row in cur.execute("SELECT id FROM {}".format(self._name)):
			yield row[0]



def globt(*arg, **kwarg):
	"glob sorted by file modification time"
	return sorted(glob.glob(*arg, **kwarg),  key=lambda x: os.stat(x).st_mtime)




class pmath():
	def __init__(self, name):
		self._p = name
		self._fmt = "{}"
		self._name = ""
	def value(self,m):
		if self._p in m.alias_names():
			als = m.alias(self._p)
			return m.metaparameter(self._p).value
		if self._p in m:
			return m[self._p].value
		raise LarchError("parameter {} not in model".format(self._p))
	def name(self, n):
		self._name = n
		return self
	def fmt(self,format):
		self._fmt = format
		return self
	def str(self,m):
		return self._fmt.format(self.value(m))
	def getname(self):
		return self._name
	def valid(self,m):
		if self._p in m:
			return True
		return False
	def __add__(self, other):
		return _param_add(self,other)
	def __radd__(self, other):
		return _param_add(other,self)
	def __sub__(self, other):
		return _param_subtract(self,other)
	def __rsub__(self, other):
		return _param_subtract(other,self)
	def __mul__(self, other):
		return _param_multiply(self,other)
	def __rmul__(self, other):
		return _param_multiply(other,self)
	def __truediv__(self, other):
		return _param_divide(self,other)
	def __rtruediv__(self, other):
		return _param_divide(other,self)
	def __neg__(self):
		return _param_negate(self)

class _param_add(pmath):
	def __init__(self,left,right):
		self._p = None
		self._left = left
		self._right = right
		self._fmt = "{}"
		self._name = ""
	def value(self,m):
		if isinstance(self._left, pmath):
			x = self._left.value(m)
		else:
			x = self._left
		if isinstance(self._right, pmath):
			x += self._right.value(m)
		else:
			x += self._right
		return x
	def valid(self,m):
		if isinstance(self._left, pmath):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, pmath):
			x &= self._right.valid(m)
		return x

class _param_subtract(pmath):
	def __init__(self,left,right):
		self._p = None
		self._left = left
		self._right = right
		self._fmt = "{}"
		self._name = ""
	def value(self,m):
		if isinstance(self._left, pmath):
			x = self._left.value(m)
		else:
			x = self._left
		if isinstance(self._right, pmath):
			x -= self._right.value(m)
		else:
			x -= self._right
		return x
	def valid(self,m):
		if isinstance(self._left, pmath):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, pmath):
			x &= self._right.valid(m)
		return x


class _param_multiply(pmath):
	def __init__(self,left,right):
		self._p = None
		self._left = left
		self._right = right
		self._fmt = "{}"
		self._name = ""
	def value(self,m):
		if isinstance(self._left, pmath):
			x = self._left.value(m)
		else:
			x = self._left
		if isinstance(self._right, pmath):
			x *= self._right.value(m)
		else:
			x *= self._right
		return x
	def valid(self,m):
		if isinstance(self._left, pmath):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, pmath):
			x &= self._right.valid(m)
		return x


class _param_divide(pmath):
	def __init__(self,left,right):
		self._p = None
		self._left = left
		self._right = right
		self._fmt = "{}"
		self._name = ""
	def value(self,m):
		if isinstance(self._left, pmath):
			x = self._left.value(m)
		else:
			x = self._left
		try:
			if isinstance(self._right, pmath):
				x /= self._right.value(m)
			else:
				x /= self._right
		except ZeroDivisionError:
			return float('NaN')
		return x
	def valid(self,m):
		if isinstance(self._left, pmath):
			x = self._left.valid(m)
		else:
			x = True
		if isinstance(self._right, pmath):
			x &= self._right.valid(m)
		return x


class _param_negate(pmath):
	def __init__(self,orig):
		self._p = None
		self._orig = orig
		self._fmt = "{}"
		self._name = ""
	def value(self,m):
		if isinstance(self._orig, pmath):
			return -self._orig.value(m)
		else:
			return -self._orig
	def valid(self,m):
		if isinstance(self._orig, pmath):
			return self._orig.valid(m)
		else:
			return True










