

import gzip, os, csv, struct, zipfile, io
from .. import logging




class SmartFileReader(object):
	def __init__(self, file, *args, **kwargs):
		if file[-3:]=='.gz':
			with open(file, 'rb') as f:
				f.seek(-4, 2)
				self._filesize = struct.unpack('I', f.read(4))[0]
			self.file = gzip.open(file, 'rt', *args, **kwargs)
		if file[-4:]=='.zip':
			zf = zipfile.ZipFile(file, 'r')
			zf_info = zf.infolist()
			if len(zf_info)!=1:
				raise TypeError("zip archive files must contain a single member file for SmartFileReader")
			zf_info = zf_info[0]
			self.file = zf.open(zf_info.filename, 'r', *args, **kwargs)
			self._filesize = zf_info.file_size
		else:
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
		try:
			return (float(self.file.tell())/float(self._filesize)*100)
		except io.UnsupportedOperation:
			return 1.0-(float(self.file._left)/float(self._filesize)*100)
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
	for h in range(len(headers)):
		headers[h] = interpret_header_name(headers[h])	
	return headers, r, raw




def prepare_import_headers_plain(rawfilename, headers=None):
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
	return headers, r, raw
