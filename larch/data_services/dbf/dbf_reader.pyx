# cython: language_level=3, embedsignature=True

from __future__ import print_function

from libc.stdio cimport FILE, fopen, fwrite, fscanf, fclose, fseek, SEEK_END, ftell, stdout, stderr, fread, SEEK_SET
cimport numpy as np
cimport cython
from cython.parallel cimport prange, parallel, threadid


ctypedef np.int64_t int64_t

cdef enum:
	maxfld = 256

cdef enum:
	maxrecln = 4000

cdef enum CType:
	DOUBLE,
	STRING,
	INTEGER,
	OTHER


cdef struct dbfhrecord:
	int nrecs               # number of records
	int nflds               # number of fields
	int recln               # record length
	char[maxfld][11] fname  # field names
	char[maxfld]     ftyp   # field types
	char[maxfld]     flen   # field lengths
	char[maxfld]     fdec   # field decimals
	int[maxfld]      fdisp  # field displacement in a record
	CType[maxfld]    fctype # field c-types for conversion to useful values


cdef void copy_dbfhrecord(dbfhrecord* source, dbfhrecord* dest):
	dest.nrecs=source.nrecs               # number of records
	dest.nflds=source.nflds               # number of fields
	dest.recln=source.recln               # record length
	for j in range(maxfld):
		for i in range(11):
			dest.fname[j][i]=source.fname[j][i]  # field names
		dest.ftyp[j] =source.ftyp[j]    # field types
		dest.flen[j] =source.flen[j]    # field lengths
		dest.fdec[j] =source.fdec[j]    # field decimals
		dest.fdisp[j]=source.fdisp[j]  # field displacement in a record
		dest.fctype[j]=source.fctype[j]  # field ctype


@cython.cdivision(True)
cdef void _readDBFFileHeader(
		char* filename,
		dbfhrecord* dbhrec,
		bint log_info
) except *:
	cdef:
		char b
		int i,j,posfr,nread
		# char[32] hb
	with open(filename, mode='rb') as f:
		# reset to beginning of file
		f.seek(0)
		# read in first 32 bytes of files
		hb = f.read(32)  #todo check we correctly read into the array
		if log_info:
			print('dBase file last updated: '+str(hb[2])+'/'+str(hb[3])+'/'+str(hb[1]+1900))
		dbhrec.nrecs = round(hb[4]+256.0*hb[5]+256*256.0*hb[6]+256*256*256.0*hb[7])
		posfr = hb[8]+256*hb[9]
		dbhrec.nflds =(posfr-33) // 32
		dbhrec.recln =hb[10]+256*hb[11]
		if log_info:
			print('File has '+str(dbhrec.nflds)+' data fields  and '+str(dbhrec.nrecs)+' records of '+str(dbhrec.recln)+' bytes')

		posfr = 1
		# parse out field names, types, lengths, and decimals, and calculate displacements
		for i in range(dbhrec.nflds):
			hb = f.read(32)
			# print()
			# print('row=',i,'  len(hb)=',len(hb))
			# print('hb[:11]=',hb[:11])
			# print('hb[11:]=',hb[11:])
			for j in range(11):
				dbhrec.fname[i][j]=0
				if hb[j]>0:
					dbhrec.fname[i][j] = hb[j]
			dbhrec.ftyp[i] =hb[11]
			dbhrec.flen[i] =hb[16]
			dbhrec.fdec[i] =hb[17]
			dbhrec.fdisp[i] = posfr
			posfr += dbhrec.flen[i]

			if dbhrec.ftyp[i]==b'N':
				if dbhrec.fdec[i]==0:
					dbhrec.fctype[i] = CType.INTEGER
				else:
					dbhrec.fctype[i] = CType.DOUBLE
			elif dbhrec.ftyp[i]==b'F':
				dbhrec.fctype[i] = CType.DOUBLE
			elif dbhrec.ftyp[i]==b'C':
				dbhrec.fctype[i] = CType.STRING
			else:
				dbhrec.fctype[i] = CType.OTHER

			if log_info:
				print('Field '+str(i)+': '+str(dbhrec.fname[i])+' : Format '+str(dbhrec.ftyp[i])+str(dbhrec.flen[i])+'.'+str(dbhrec.fdec[i])+'    <'+str(dbhrec.fname[i])+'>');

		#read last header byte
		f.read(1)

def readDBFFileHeader(
		filename,
		log = True
):
	cdef dbfhrecord header
	header.nrecs=0               # number of records
	header.nflds=0               # number of fields
	header.recln=0               # record length
	for j in range(maxfld):
		for i in range(11):
			header.fname[j][i]=0  # field names
		header.ftyp[j]=0   # field types
		header.flen[j]=0   # field lengths
		header.fdec[j]=0   # field decimals
		header.fdisp[j]=0  # field displacement in a record
		header.fctype[j]=CType.OTHER   # field types
	#print("BLANK HEADER")
	#from pprint import pprint
	#pprint(header)
	_readDBFFileHeader( filename, &header, log)
	return header


# FILE *fopen( const char * filename, const char * mode )

# size_t fread ( void * ptr, size_t size, size_t count, FILE * stream );
#
# Read block of data from stream
# Reads an array of count elements, each one with a size of size bytes, from the stream and stores them in the block of memory specified by ptr.
# The position indicator of the stream is advanced by the total amount of bytes read.
# The total amount of bytes read if successful is (size*count).

cdef void pull_record_data(FILE* fileobj, dbfhrecord* dbf_header, int rownum, char* buffer) nogil:
	cdef:
		int64_t i
		int nread
	i = (rownum)*dbf_header.recln
	i += 32*(dbf_header.nflds+1)+1
	fseek(fileobj,i,SEEK_SET)
	#print(f"READING {dbf_header.recln} bytes from FILE rownum {rownum} OFFSET {i}")
	fread(<void*>buffer, sizeof(char), dbf_header.recln, fileobj )
	# cdef int j
	# print(f"BUFFER IS [",end="")
	# for j in range(dbf_header.recln):
	# 	try:
	# 		k = chr(buffer[j])
	# 	except ValueError:
	# 		k = "?"
	# 	if k in ('\r','\n'):
	# 		print("ß", end="")
	# 	else:
	# 		try:
	# 			print(k, end="")
	# 		except ValueError:
	# 			print("?", end="")
	# print(']')



# size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
# Parameters
# 	ptr − This is the pointer to the array of elements to be written.
# 	size − This is the size in bytes of each element to be written.
# 	nmemb − This is the number of elements, each one with a size of size bytes.
# 	stream − This is the pointer to a FILE object that specifies an output stream.

cdef void push_record_data(FILE* fileobj, dbfhrecord* dbf_header, int rownum, char* buffer) nogil:
	cdef:
		int64_t i
		int nread
	i = (rownum)*dbf_header.recln
	i += 32*(dbf_header.nflds+1)+1
	fseek(fileobj,i,SEEK_SET)
	fwrite(<void*>buffer, sizeof(char), dbf_header.recln, fileobj )



from libc.stdlib cimport malloc, free, atoi, atoll, atof
from libc.stdio cimport sprintf

cdef int get_dbf_int(dbfhrecord* dbf_header, int fnum, char* buffer):
	cdef char* s
	cdef int j
	cdef int k = dbf_header.flen[fnum]
	cdef int offset = dbf_header.fdisp[fnum]
	s = <char*>malloc((k+1) * sizeof(char))
	for j in range(k):
		s[j]=buffer[offset+j]
	s[k] = 0
	cdef int result = atoi(s)
	free(s)
	return result


cdef int64_t get_dbf_int64(dbfhrecord* dbf_header, int fnum, char* buffer) nogil:
	cdef char* s
	cdef int j
	cdef int k = dbf_header.flen[fnum]
	cdef int offset = dbf_header.fdisp[fnum]
	s = <char*>malloc((k+1) * sizeof(char))
	for j in range(k):
		s[j]=buffer[offset+j]
	s[k] = 0
	cdef int64_t result = atoll(s)
	free(s)
	return result


cdef double get_dbf_double(dbfhrecord* dbf_header, int fnum, char* buffer) nogil:
	cdef char* s
	cdef int j
	cdef int k = dbf_header.flen[fnum]
	cdef int offset = dbf_header.fdisp[fnum]
	s = <char*>malloc((k+1) * sizeof(char))
	for j in range(k):
		s[j]=buffer[offset+j]
	s[k] = 0
	cdef double result = atof(s)
	free(s)
	return result


cdef double get_dbf_double_v2(int fieldlen, int fieldoffset, char* buffer) nogil:
	cdef char* s
	cdef int j
	s = <char*>malloc((fieldlen+1) * sizeof(char))
	for j in range(fieldlen):
		s[j]=buffer[fieldoffset+j]
	s[fieldlen] = 0
	cdef double result = atof(s)
	free(s)
	return result

cdef unicode get_dbf_string_v2(int fieldlen, int fieldoffset, char* buffer, bint strip_whitespace=False):
	cdef char* s
	cdef int j
	s = <char*>malloc((fieldlen+1) * sizeof(char))
	for j in range(fieldlen):
		s[j]=buffer[fieldoffset+j]
	s[fieldlen] = 0
	cdef unicode result = s.decode('UTF-8')
	free(s)
	if strip_whitespace:
		return result.strip()
	return result

# cdef void set_dbf_double_v2(double value, int fieldlen, int fieldprecision, int fieldoffset, char* buffer) nogil:
# 	cdef char* s
# 	cdef int j
# 	s = <char*>malloc((fieldlen+20) * sizeof(char))
# 	sprintf(s, "%*.*f", fieldlen, fieldprecision, value)
# 	for j in range(fieldlen):
# 		buffer[fieldoffset+j]=s[j]
# 	with gil:
# 		print('set_dbf_double_v2',value,fieldlen,fieldprecision,fieldoffset,s)
# 		print(buffer)
# 	free(s)


cdef class DBF:

	#cdef FILE* _file_ptr
	cdef dbfhrecord _header
	cdef bytes _filename
	cdef object _tempdir
	cdef char* _filename_c
	cdef void* _read_buffer

	def __init__(self, filename, make_copy=False):
		"""
		Open a DBF file for read access.

		Parameters
		----------
		filename : str
			Name of file to open. If it has a .gz extension, an inflated temporary copy will automatically be created.
		make_copy : bool, optional
			If True, a temp copy of the file will be made before opening.  This is useful to prevent file-locking, so
			that other processes can edit or delete the original DBF file even while it remains open in Python.
		"""

		if isinstance(filename, str):
			filename = filename.encode()

		if filename[-3:] == b'.gz':
			import tempfile, os, shutil, gzip
			self._tempdir = tempfile.TemporaryDirectory()
			tempfilename = os.path.join(self._tempdir.name,os.path.basename(filename[:-3].decode()))
			with open(tempfilename, "wb") as tmp:
				shutil.copyfileobj(gzip.open(filename.decode()), tmp)
			filename = tempfilename.encode()
		elif make_copy:
			import tempfile, os, shutil, gzip
			self._tempdir = tempfile.TemporaryDirectory()
			tempfilename = os.path.join(self._tempdir.name,os.path.basename(filename.decode()))
			with open(tempfilename, "wb") as tmp:
				shutil.copyfileobj(open(filename.decode(), 'rb'), tmp)
			filename = tempfilename.encode()

		_readDBFFileHeader( filename, &self._header, False)
		#self._file_ptr = fopen( filename, 'r' )
		self._filename = filename
		self._filename_c = self._filename
		self._read_buffer = <void*>malloc(self._header.recln)

	def __del__(self):
		#fclose(self._file_ptr)
		free(self._read_buffer)

	def __repr__(self):
		s = []
		for i in range(self._header.nflds):
			if self._header.fctype[i] == CType.DOUBLE:
				ct = "FLOAT"
			elif self._header.fctype[i] == CType.INTEGER:
				ct = "INTEGER"
			elif self._header.fctype[i] == CType.STRING:
				ct = "STRING"
			else:
				ct = "OTHER"
			s.append(f' [{i: 3}] {self._header.fname[i].decode():11s}: Format {chr(self._header.ftyp[i])} {self._header.flen[i]}.{self._header.fdec[i]} {ct}')
		return f"<larch.DBF> {self._header.nflds} data fields and {self._header.nrecs} records of {self._header.recln} bytes\n" + "\n".join(s)

	def fieldnames(self):
		result = []
		for i in range(self._header.nflds):
			result.append(self._header.fname[i].decode())
		return result

	def header_definition(self):
		result = []
		for i in range(self._header.nflds):
			if self._header.fdec[i] >0:
				result.append(
					(
						self._header.fname[i].decode(),
						chr(self._header.ftyp[i]),
						self._header.flen[i],
						self._header.fdec[i],
					)
				)
			else:
				result.append(
					(
						self._header.fname[i].decode(),
						chr(self._header.ftyp[i]),
						self._header.flen[i],
					)
				)
		return tuple(result)

	def fieldnames_integer(self):
		result = []
		for i in range(self._header.nflds):
			if self._header.fctype[i] == CType.INTEGER:
				result.append(self._header.fname[i].decode())
		return result

	def fieldnames_float(self):
		result = []
		for i in range(self._header.nflds):
			if self._header.fctype[i] == CType.DOUBLE:
				result.append(self._header.fname[i].decode())
		return result

	def fieldnames_char(self):
		result = []
		for i in range(self._header.nflds):
			if self._header.fctype[i] == CType.STRING:
				result.append(self._header.fname[i].decode())
		return result

	cdef void _get_int_array(self, int fieldnum, int startrow, int stoprow, int[:] arr):
		"""
		Extract an int64 array of values from the DBF file.

		Parameters
		----------
		fieldnum : int
			Which field (zero-based number) to get
		startrow, stoprow: int
			The start and stop row numbers, in usual Python format (first row, one past last row).
			These values must be given explicitly

		Returns
		-------
		ndarray
			dtype is int64
		"""
		cdef FILE* local_file_handle
		cdef int i
		local_file_handle = fopen( self._filename_c, 'r' )
		for i,row in enumerate(range(startrow, stoprow)):
			pull_record_data(local_file_handle, &self._header, row, <char*>self._read_buffer)
			q = get_dbf_int64(&self._header, fieldnum, <char*>self._read_buffer)
			arr[i] = q
		fclose(local_file_handle)

	def get_int_array(self, fieldnum, startrow, stoprow):
		"""
		Extract an int64 array of values from the DBF file.

		Parameters
		----------
		fieldnum : int
			Which field (zero-based number) to get
		startrow, stoprow: int
			The start and stop row numbers, in usual Python format (first row, one past last row).
			These values must be given explicitly

		Returns
		-------
		ndarray
			dtype is int64
		"""
		import numpy
		arr = numpy.zeros([stoprow-startrow], dtype=numpy.int64)
		self._get_int_array(fieldnum, startrow, stoprow, arr)
		return arr

	@cython.boundscheck(False)
	cdef void _load_dataframe_arr_int(self, int startrow, int stoprow, int64_t[:,:] arr):
		cdef int i=0
		cdef int row=startrow
		cdef int nrows = stoprow - startrow
		cdef int j, k
		cdef FILE* local_file_handle
		cdef void* local_read_buffer
		with nogil, parallel():
			local_file_handle = fopen( self._filename_c, 'r' )
			local_read_buffer = <void*>malloc(self._header.recln)
			for i in prange(nrows):
				row = startrow+i
				pull_record_data(local_file_handle, &self._header, row, <char*>local_read_buffer)
				k = 0
				for j in range(self._header.nflds):
					if self._header.fctype[j] == CType.INTEGER:
						arr[i,k] = get_dbf_int64(&self._header, j, <char*>local_read_buffer)
						k = k + 1
			free(local_read_buffer)
			fclose(local_file_handle)

	@cython.boundscheck(False)
	cdef void _load_dataframe_arr_float(self, int startrow, int stoprow, double[:,:] arr):
		cdef int i=0
		cdef int row=startrow
		cdef int nrows = stoprow - startrow
		cdef int j, k, xx1, xx2
		cdef FILE* local_file_handle
		cdef void* local_read_buffer
		cdef double xx3
		with nogil, parallel():
			local_file_handle = fopen( self._filename_c, 'r' )
			local_read_buffer = <void*>malloc(self._header.recln)
			for i in prange(nrows):
				row = startrow+i
				pull_record_data(local_file_handle, &self._header, row, <char*>local_read_buffer)
				k = 0
				for j in range(self._header.nflds):
					if self._header.fctype[j] == CType.DOUBLE:
						arr[i,k] = get_dbf_double_v2(self._header.flen[j], self._header.fdisp[j], <char*>local_read_buffer)
						k = k + 1
			free(local_read_buffer)
			fclose(local_file_handle)

	@cython.boundscheck(False)
	cdef void _load_dataframe_arr_string(self, int startrow, int stoprow, unicode[:,:] arr, bint strip_whitespace=False):
		cdef int i=0
		cdef int row=startrow
		cdef int nrows = stoprow - startrow
		cdef int j, k, xx1, xx2
		cdef FILE* local_file_handle
		cdef void* local_read_buffer
		cdef double xx3
		cdef unicode s

		local_file_handle = fopen( self._filename_c, 'r' )
		local_read_buffer = <void*>malloc(self._header.recln)
		for i in range(nrows):
			row = startrow+i
			pull_record_data(local_file_handle, &self._header, row, <char*>local_read_buffer)
			k = 0
			for j in range(self._header.nflds):
				if self._header.fctype[j] == CType.STRING:
					s = get_dbf_string_v2(self._header.flen[j], self._header.fdisp[j], <char*>local_read_buffer, strip_whitespace)
					arr[i,k] = s
					k = k + 1
		free(local_read_buffer)
		fclose(local_file_handle)

	# @cython.boundscheck(False)
	# cdef void _write_arr_float(self, int startrow, int stoprow, int fieldnum, double[:] arr):
	# 	cdef int i=0
	# 	cdef int j=fieldnum
	# 	cdef int row=startrow
	# 	cdef int nrows = stoprow - startrow
	# 	cdef int k, xx1, xx2
	# 	cdef FILE* local_file_handle
	# 	cdef void* local_read_buffer
	# 	cdef double xx3
	# 	if self._header.fctype[j] != CType.DOUBLE:
	# 		raise TypeError("field fctype is not DOUBLE")
	# 	print("_write_arr_float:",fieldnum)
	# 	with nogil:
	# 		local_file_handle = fopen( self._filename_c, 'a' )
	# 		fseek(local_file_handle,0,SEEK_SET)
	# 		local_read_buffer = <void*>malloc(self._header.recln)
	# 		for i in range(nrows):
	# 			row = startrow+i
	# 			pull_record_data(local_file_handle, &self._header, row, <char*>local_read_buffer)
	# 			set_dbf_double_v2(arr[i], self._header.flen[j], self._header.fdec[j], self._header.fdisp[j], <char*>local_read_buffer)
	# 			push_record_data(local_file_handle, &self._header, row, <char*>local_read_buffer)
	# 		free(local_read_buffer)
	# 		fclose(local_file_handle)
	#
	# def write_column_float(self, int fieldnum, double[:] arr, int startrow=0, int stoprow=-1):
	# 	if stoprow<0 or stoprow>self._header.nrecs:
	# 		stoprow=self._header.nrecs
	# 	return self._write_arr_float(startrow, stoprow, fieldnum, arr)

	# cdef void _load_dataframe_arr(self, int startrow, int stoprow, double[:,:] arr):
	# 	cdef int i=0
	# 	cdef int row=startrow
	# 	cdef int nrows = stoprow - startrow
	# 	cdef int j
	# 	for i in range(nrows):
	# 		row = startrow+i
	# 		pull_record_data(self._file_ptr, &self._header, row, <char*>self._read_buffer)
	# 		for j in range(self._header.nflds):
	# 			arr[i,j] = get_dbf_double(&self._header, j, <char*>self._read_buffer)

	def _load_dataframe(self, int startrow, int stoprow,
						bint preserve_order=False,
						bint strip_whitespace=False,
						):
		import pandas, numpy
		fields_integer = self.fieldnames_integer()
		fields_float = self.fieldnames_float()
		fields_string = self.fieldnames_char()
		if stoprow<0 or stoprow>self._header.nrecs:
			stoprow=self._header.nrecs
		if startrow<0 or startrow>self._header.nrecs:
			raise IndexError(f'startrow {startrow} out of range for file with {self._header.nrecs} rows')
		if fields_integer:
			df = pandas.DataFrame(0, index=numpy.arange(startrow, stoprow), columns=fields_integer, dtype=numpy.int64 )
			self._load_dataframe_arr_int(startrow, stoprow, df.values)
		else:
			df = pandas.DataFrame(index=numpy.arange(startrow, stoprow), columns=fields_integer, dtype=numpy.int64 )
		if fields_float:
			df_float = pandas.DataFrame(index=numpy.arange(startrow, stoprow), columns=fields_float, dtype=numpy.float64 )
			self._load_dataframe_arr_float(startrow, stoprow, df_float.values)
			df = pandas.concat([df, df_float], axis=1).reindex(index=df.index)
		if fields_string:
			df_string = pandas.DataFrame(index=numpy.arange(startrow, stoprow), columns=fields_string, dtype=str )
			self._load_dataframe_arr_string(startrow, stoprow, df_string.values, strip_whitespace)
			df = pandas.concat([df, df_string], axis=1).reindex(index=df.index)
		if preserve_order:
			cols = [i for i in self.fieldnames() if i in df.columns]
			df = df[cols]
		return df

	def load_dataframe(self, start=0, stop=-1, preserve_order=True, strip_whitespace=True):
		"""
		Load a DataFrame from the DBF file.

		Parameters
		----------
		start : int, default 0
			The index of the row number to begin loading data. Only non-negative
			values are permitted.
		stop : int, default -1
			One past the index of the row number to stop loading data.
			Negative or out-of-range values are interpreted as an instruction
			to read to the end of the file.
		preserve_order: bool, default True
			Preserve the order of columns when loading.  If False,
			columns are re-ordered to group together similar data types, which
			slightly improves efficiency.
		strip_whitespace: bool, default True
			Strip white space from text fields, which are stored generally as
			space-padded fixed-length strings in the raw file.

		Returns
		-------
		pandas.DataFrame

		Raises
		------
		IndexError
			If `startrow` is negative or out-of-range.
		"""
		return self._load_dataframe(start, stop, preserve_order, strip_whitespace)

	def load_dataframe_iter(self, chunksize=100000, *, return_slice=False):
		"""
		Load rows from this DBF file in chunks.

		Parameters
		----------
		chunksize : int, default 100000
		    Number of rows to read in each chunk.
		return_slice : bool, default False
		    Instead of yielding loaded dataframes only, also yield
		    slice objects that indicate the start and stop marks
		    for the given chunk.

        Yields
        ------
        pandas.DataFrame or (pandas.DataFrame, slice)
		"""
		start = 0
		stop = chunksize
		while start < self._header.nrecs:
			if stop > self._header.nrecs:
				stop = self._header.nrecs
			if return_slice:
				yield (self._load_dataframe(start, stop), slice(start,stop))
			else:
				yield self._load_dataframe(start, stop)
			start += chunksize
			stop += chunksize

	@property
	def filename(self):
		return self._filename.decode()

	@property
	def nrecs(self):
		return self._header.nrecs

	@nrecs.setter
	def nrecs(self, value):
		if value != self._header.nrecs:
			import warnings
			warnings.warn(f"changing the number of records from {self._header.nrecs} to {value} is dangerous and may cause program instability")
			self._header.nrecs = int(value)

	def convert_to_hdf5(self, h5filename=None, groupnode=None, show_progress=True, identify=None, **kwargs):
		from ..h5.h5pod import H5Pod
		import numpy
		import os

		if h5filename is None:
			# convert if needed based on dbf filename
			h5filename_auto = os.path.splitext(self.filename)[0] + '.h5d'
			if os.path.exists(h5filename_auto):
				# check if file creation time is newer than dbf creation time
				def which_file_created_more_recently(filename1, filename2):
					(mode1, ino1, dev1, nlink1, uid1, gid1, size1, atime1, mtime1, ctime1) = os.stat(filename1)
					(mode2, ino2, dev2, nlink2, uid2, gid2, size2, atime2, mtime2, ctime2) = os.stat(filename2)
					if ctime1<ctime2:
						return 1
					if ctime1>=ctime2:
						return 0
				if which_file_created_more_recently(self.filename, h5filename_auto)==1:
					# h5d file created more recently, so just return it
					return H5Pod(filename=h5filename_auto, mode='r')
				else:
					# dbf file created more recently, so rename the older h5d
					if os.path.exists(h5filename_auto+'.oldbackup'):
						raise FileExistsError('do you really want to overwrite this file again?')
					os.rename(h5filename_auto, h5filename_auto+'.oldbackup')
					h5filename = h5filename_auto
			else:
				h5filename = h5filename_auto

		pod = H5Pod(filename=h5filename, mode='a', groupnode=groupnode, **kwargs)
		pod.shape = (self._header.nrecs, )
		for fi in self.fieldnames_integer():
			pod.add_blank(fi, dtype=numpy.int64)
		for ff in self.fieldnames_float():
			pod.add_blank(ff, dtype=numpy.float64)
		from tqdm import tqdm
		with tqdm(total=self._header.nrecs, unit='rows') as pbar:
			pbar.set_description(identify or self.filename)
			for d,slc in self.load_dataframe_iter(return_slice=True):
				if show_progress:
					pbar.update(slc.stop-slc.start)
					#print(f"\rDBF Reading {100.0*slc.stop/self._header.nrecs:.2f}% Complete")
				for fi in self.fieldnames_integer():
					pod.__getattr__(fi)[slc] = d.loc[:,fi].values
				for ff in self.fieldnames_float():
					pod.__getattr__(ff)[slc] = d.loc[:,ff].values
		return pod.change_mode('r')



# from larch.h5.dbf_reader import DBF

'''
program dbf_hacker;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}{$IFDEF UseCThreads}
  cthreads,
  {$ENDIF}{$ENDIF}
  Classes, SysUtils, CustApp, GMap, GUtil
  { you can add units after this };

type

  { TDbfHacker }

  TDbfHacker = class(TCustomApplication)
  protected
    procedure DoRun; override;
  public
    constructor Create(TheOwner: TComponent); override;
    destructor Destroy; override;
    procedure WriteHelp; virtual;
  end;



 type
     less_int = specialize TLess<Int64>;
     TMapIntInt = specialize TMap<Int64, Int64, less_int>;





  {//////////////// routines to read and write dBase IV files ////////////}

  {Data types}
  const {globals}
      maxfld=256;
      maxrecln=4000;
  type
      dbfftype=file;
      dbfhrecord=record
        nrecs, nflds, recln:integer;  {number of records, fields, record length}
        fname:array[1..maxfld] of string[11]; {field names}
        ftyp:array[1..maxfld] of byte; {field types}
        flen:array[1..maxfld] of byte; {field lengths}
        fdec:array[1..maxfld] of byte; {field decimals}
        fdisp:array[1..maxfld] of integer; {field displacement in a record}
        currec:array[1..maxrecln] of char;
      end;

  {//// Procedure to read header record information}
  procedure readDBFFileHeader( var dbfil:dbfftype; var dbhrec:dbfhrecord; var number_of_fields: integer;
                               var target_field_name: string; var target_field_num: integer;
                               log_info:boolean);
  var {locals}
      b:byte; i,j,posfr,nread:integer; hb:array[0..31] of byte;
      target_field_name_lower:string;
  begin
   target_field_name_lower := AnsiLowerCase(target_field_name);
   with dbhrec do begin
  {reset to beginning of file}
    seek(dbfil,0);
  {read in first 32 bytes of files}
    blockread(dbfil,hb,sizeof(hb),nread);
    if log_info then writeln('');
    if log_info then writeln('dBase file last updated: '+inttostr(hb[2])+'/'+inttostr(hb[3])+'/'+inttostr(hb[1]+1900));
  {parse out number of records and fields, and record length, in bytes}
    nrecs:=round(hb[4]+256.0*hb[5]+256*256.0*hb[6]+256*256*256.0*hb[7]);
    posfr:=hb[8]+256*hb[9];
    nflds:=(posfr-33) div 32;
    recln:=hb[10]+256*hb[11];
    if log_info then writeln('File has '+inttostr(nflds)+' data fields  and '+inttostr(nrecs)+' records of '+inttostr(recln)+' bytes');
    number_of_fields := nflds;

  {parse out field names, types, lengths, and decimals, and calculate displacements}
    posfr:=1;
    for i:=1 to nflds do begin
      fname[i]:='';
      blockread(dbfil,hb,sizeof(hb),nread);
      for j:=0 to 10 do if hb[j]>32 then fname[i]:=fname[i]+char(hb[j]);
      ftyp[i]:=hb[11];
      flen[i]:=hb[16];
      fdec[i]:=hb[17];
      fdisp[i]:=posfr;
      posfr:=posfr+flen[i];
      if log_info then writeln('Field '+inttostr(i)+': '+fname[i]+' : Format '+char(ftyp[i])+inttostr(flen[i])+'.'+inttostr(fdec[i])+'    <'+AnsiLowerCase(fname[i])+'>');
      if target_field_name_lower=AnsiLowerCase(fname[i]) then begin
         target_field_num := i;
      end;
    end;

  {read last header byte}
    blockread(dbfil,b,sizeof(b),nread);

   end;
  end;


  {////Procedure to write header record information}
  procedure writeDBFFileHeader( var dbfil:dbfftype; var dbfhrec:dbfhrecord);
  var {locals}
      b:byte; i,j,nwrit:integer;
      hb:array[0..31] of byte;
      y,m,d:word;
  begin
   with dbfhrec do begin
  {reset to beginning of file}
    seek(dbfil,0);
  {set first 32 bytes and write to file}
    for j:=0 to 31 do hb[j]:=0;
  {file type}
    hb[0]:=3;
  {date}
    decodedate(now,y,m,d);
    hb[1]:=y-2000; {year}
    hb[2]:=m;  {month}
    hb[3]:=d;  {day}
  {number of records}
    hb[4]:=nrecs mod 256;
    hb[5]:=(nrecs div 256) mod 256;
    hb[6]:=(nrecs div (256*256)) mod 256;
    hb[7]:=(nrecs div (256*256*256)) mod 256;
  {header length}
    hb[8]:=(32*(nflds+1)+1) mod 256;
    hb[9]:=(32*(nflds+1)+1) div 256;
  {record length}
    hb[10]:=recln mod 256;
    hb[11]:=recln div 256;
  {write}
    blockwrite(dbfil,hb,sizeof(hb),nwrit);
    //writeln('    wrote bytes '+IntToStr(nwrit)+' for HEAD now at '+IntToStr(FilePos(dbfil)));

  {for each field, set 32 bytes and write to file}
    for i:=1 to nflds do begin
      for j:=0 to 31 do hb[j]:=0;
  {field name}
      for j:=1 to length(fname[i]) do hb[j-1]:=ord(fname[i][j]);
  {field type}
      hb[11]:=ftyp[i];
  {field length}
      hb[16]:=flen[i];
  {field decimals}
      hb[17]:=fdec[i];
  {write}
      blockwrite(dbfil,hb,sizeof(hb),nwrit);
      //writeln('    wrote bytes '+IntToStr(nwrit)+' for field '+IntToStr(i)+' now at '+IntToStr(FilePos(dbfil)));
    end;

  {write end of header byte}
    b:=13;
    blockwrite(dbfil,b,sizeof(b),nwrit);
    //writeln('    wrote bytes '+IntToStr(nwrit)+' for TAIL now at '+IntToStr(FilePos(dbfil)));
   end;
  end;

  {//// Procedure to get record rnum from the file}
  procedure readDBFRecord(var dbfil:dbfftype; var dbfhrec:dbfhrecord; rnum:integer);

  var {locals}
      i:longint; nread:integer;
  begin
   with dbfhrec do begin
    i:=round(32.0*(nflds+1)+(rnum-1.0)*recln+1);
    //writeln('read record at byte '+IntToStr(i));
    seek(dbfil,i);
    blockread(dbfil,currec,recln,nread);
    //writeln('    read bytes '+IntToStr(nread));
   end;
  end;

  {//// Procedure to get an integer value from the current record}
  procedure getDBFInt(dbfhrec:dbfhrecord; fnum:integer; var x:integer);
  var {locals}
      e,j:integer; s:string;
  begin
   with dbfhrec do begin
    setlength(s,flen[fnum]);
    for j:=1 to flen[fnum] do s[j]:=currec[fdisp[fnum]+j];
    val(s,x,e);
   end;
  end;

  {//// Procedure to get an int64 value from the current record}
  procedure getDBFInt64(dbfhrec:dbfhrecord; fnum:integer; var x:Int64);
  var {locals}
      j:integer; s:string;
  begin
   with dbfhrec do begin
    setlength(s,flen[fnum]);
    for j:=1 to flen[fnum] do s[j]:=currec[fdisp[fnum]+j];
    x:=StrToInt64(s);
   end;
  end;

  {//// Procedure to get a real value from the current record}
  procedure getDBFReal(dbfhrec:dbfhrecord; fnum:integer; var x:single);
  var {locals}
      e,j:integer; s:string;
  begin
   with dbfhrec do begin
    setlength(s,flen[fnum]);
    for j:=1 to flen[fnum] do s[j]:=currec[fdisp[fnum]+j];
    val(s,x,e);
   end;
  end;

  {//// Procedure to put an integer value into the current record}
  procedure putDBFInt(var dbfhrec:dbfhrecord; fnum:integer; x:integer);
  var {locals}
      j:integer; s:string;
  begin
   with dbfhrec do begin
    str(x:flen[fnum],s);
    for j:=1 to flen[fnum] do currec[fdisp[fnum]+j]:=s[j];
   end;
  end;

  {//// Procedure to put a real value into the current record}
  procedure putDBFReal(var dbfhrec:dbfhrecord; fnum:integer; x:single);
  var {locals}
      j:integer; s:string;
  begin
   with dbfhrec do begin
    str(x:flen[fnum]:fdec[fnum],s);
    for j:=1 to flen[fnum] do currec[fdisp[fnum]+j]:=s[j];
   end;
  end;

  {//// Procedure to write the current record to the file}
  procedure writeDBFRecord(var dbfil:dbfftype; dbfhrec:dbfhrecord; rnum:integer);
  var {locals}
      i:longint; nwrit:integer;
  begin

   with dbfhrec do begin
    i:=round(32.0*(nflds+1)+(rnum-1.0)*recln+1);
    //writeln('write record at byte '+IntToStr(i));
    seek(dbfil,i);
    currec[1]:=char(32);
    blockwrite(dbfil,currec,recln,nwrit);
    //writeln('    wrote bytes '+IntToStr(nwrit)+' with i='+IntToStr(i)+' now at '+IntToStr(FilePos(dbfil)));
   end;
  end;

  {//// Procedure to write end-of-file character}
  procedure writeDBFFileEOF(var dbfil:dbfftype; dbfhrec:dbfhrecord);
  var {locals}
      b:byte; nwrit:integer; i:longint;
  begin
   with dbfhrec do begin
    i:=round(32.0*(nflds+1)+(nrecs)*recln+1);
    seek(dbfil,i);
   end;
   b:=26;
   blockwrite(dbfil,b,sizeof(b),nwrit);
  end;
  {//////////////// end of routines to read and write dBase IV files ////////////}





procedure spewfile( source_file:string);
var
  tfIn: TextFile;
  s: string;
begin
  // Give some feedback
  writeln('Reading the contents of file: ', source_file);
  writeln('=========================================');

  // Set the name of the file that will be read
  AssignFile(tfIn, source_file);

  // Embed the file handling in a try/except block to handle errors gracefully
  try
    // Open the file for reading
    reset(tfIn);

    // Keep reading lines until the end of the file is reached
    while not eof(tfIn) do
    begin
      readln(tfIn, s);
      writeln(s);
    end;

    // Done so close the file
    CloseFile(tfIn);

  except
    on E: EInOutError do
     writeln('File handling error occurred. Details: ', E.Message);
  end;

  // Wait for the user to end the program
  writeln('=========================================');
  writeln('File ', source_file, ' was probably read.');

end;





{ TDbfHacker }

procedure TDbfHacker.DoRun;
var
  ErrorMsg: String;
  source_file, source_file_base, keys_file: String;
  split_count: Integer;
  dbfil:dbfftype;
  dbhrec:dbfhrecord;
  number_of_fields:Integer;
  outdbfil:array of dbfftype;
  rnum:array of integer;
  i,j,k:Integer;
  target_field_name:String;
  target_field_num:Integer;
  personid_to_hhid:TMapIntInt;
  iterator:TMapIntInt.TIterator;
  tempHH,tempPer,temp:Int64;
  use_pid:boolean;
begin

  writeln('This executable is '+ParamStr(0));
  writeln('Compiled at '+{$I %TIME%}+' on '+{$I %DATE%}+' for target CPU '+{$I %FPCTARGET%});
  writeln(DateTimeToStr(Now)+' Start');

  use_pid := False;

  // quick check parameters
  ErrorMsg:=CheckOptions('hf:n:', 'help from: number: hhid: pid: key: inspect: fusion');
  if ErrorMsg<>'' then begin
    ShowException(Exception.Create(ErrorMsg));
    Terminate;
    Exit;
  end;

  // parse parameters
  if HasOption('h', 'help') then begin
    WriteHelp;
    Terminate;
    Exit;
  end;

  // inspect and nothing else
  if HasOption('inspect') then begin
    keys_file := GetOptionValue('inspect');
    if keys_file='' then begin
      WriteHelp;
      Terminate;
      Exit;
    end;
    assignFile(dbfil, keys_file);
    reset(dbfil, 1);
    target_field_name:='';
    readDBFFileHeader(dbfil, dbhrec, number_of_fields, target_field_name, target_field_num, True);
    Terminate;
    Exit;
  end;
  personid_to_hhid := TMapIntInt.Create;

  { populate personid_to_hhid }
  keys_file := GetOptionValue('key');
  if (keys_file<>'') and (not HasOption('fusion')) then begin
    writeln('keys file is ',keys_file);
    assignFile(dbfil, keys_file);
    reset(dbfil, 1);
    target_field_name:='';
    readDBFFileHeader(dbfil, dbhrec, number_of_fields, target_field_name, target_field_num, False);
    for j:=1 to dbhrec.nrecs do begin
      readDBFRecord(dbfil,dbhrec,j);
      getDBFInt64(dbhrec, 2, tempHH);
      getDBFInt64(dbhrec, 1, tempPer);
      //writeln('PER '+IntToStr(tempPer)+' HH '+IntToStr(tempHH));
      personid_to_hhid[tempPer] := tempHH;
      if j mod 500000 = 0 then begin
          writeln(DateTimeToStr(Now)+' Read '+FloatToStr(j/1000000)+' million keys ');
      end;
    end;
    writeln(DateTimeToStr(Now)+' keys file reading complete');
    CloseFile(dbfil);
  end else begin
    writeln('no keys file (this is ok if you are not splitting a file or are splitting on hhid)');
  end;




  source_file := GetOptionValue('f', 'from');
  if AnsiLowerCase(RightStr(source_file,4))='.dbf' then begin
    source_file_base := Copy(source_file,1,Length(source_file)-4);
  end else begin
    source_file_base := source_file;
  end;
  writeln('source file is '+source_file);

  split_count := StrToInt(GetOptionValue('n', 'number'));
  SetLength(outdbfil, split_count);
  SetLength(rnum, split_count);

  target_field_num := -1;
  target_field_name := GetOptionValue('hhid');

  if (target_field_name='') then begin
    { try using person id instead }
    target_field_name := GetOptionValue('pid');
    if (target_field_name<>'') then use_pid := True;
  end;

  if HasOption('fusion') then begin
    { --- FUSION --- }
    writeln('fusion to '+source_file_base+'_fusion.dbf');
    assignFile(dbfil, source_file_base+'_fusion.dbf');
    rewrite(dbfil, 1);
    k := 0;

    { open the split files for reading }
    for i := 0 to split_count-1 do begin
     assignFile(outdbfil[i], source_file_base+'_'+IntToStr(i)+'of'+IntToStr(split_count)+'.dbf');
     reset(outdbfil[i], 1);

     readDBFFileHeader(outdbfil[i], dbhrec, number_of_fields, target_field_name, target_field_num, False);

     { run through records, write to common masterfile }
     for j:=1 to dbhrec.nrecs do begin
       readDBFRecord(outdbfil[i],dbhrec,j);
       if k mod 500000 = 0 then begin
             writeln(DateTimeToStr(Now)+' Fused '+FloatToStr(k/1000000)+' million rows ');
       end;
       inc(k);
       writeDBFRecord( dbfil, dbhrec, k);
     end;
     CloseFile(outdbfil[i]);
    end;

    { write head and tail for fused file }

    dbhrec.nrecs:=k;
    writeDBFFileHeader(dbfil, dbhrec);
    writeDBFFileEOF(dbfil, dbhrec);
    CloseFile(dbfil);





    { --- END FUSION --- }
  end else begin
    { --- SPLITTER --- }
    assignFile(dbfil, source_file);
    reset(dbfil, 1);
    readDBFFileHeader(dbfil, dbhrec, number_of_fields, target_field_name, target_field_num, False);


    if (target_field_name='') then begin
      writeln('=============================================================');
      writeln('ERROR: target_field_name not given!');
      writeln('=============================================================');
      WriteHelp;
      SetLength(outdbfil, 0);
      SetLength(rnum, 0);
      Terminate;
      Halt(-1);
    end;


    if target_field_num<0 then begin
      writeln('=============================================================');
      writeln('ERROR: field ',target_field_name,' not found!');
      writeln('=============================================================');
      SetLength(outdbfil, 0);
      SetLength(rnum, 0);
      Terminate;
      Halt(-2);
    end;


    { open the split files for writing }
    for i := 0 to split_count-1 do begin
     assignFile(outdbfil[i], source_file_base+'_'+IntToStr(i)+'of'+IntToStr(split_count)+'.dbf');
     rewrite(outdbfil[i], 1);
    end;

    { run through records, write to correct splitfile }
    for j:=1 to dbhrec.nrecs do begin
      readDBFRecord(dbfil,dbhrec,j);
      getDBFInt64(dbhrec, target_field_num, temp);
      if j mod 500000 = 0 then begin
            writeln(DateTimeToStr(Now)+' Split '+FloatToStr(j/1000000)+' million rows ');
      end;
      if use_pid then begin
        //writeln('temp is ',IntToStr(temp));
        iterator := personid_to_hhid.find(temp);
        if iterator<>nil then begin
            temp := iterator.Value;
            //writeln('temp '+IntToStr(temp)+' is changed to '+IntToStr(tempHH));
            //temp := tempHH;
        end else begin
            writeln('WARNING: PID '+IntToStr(temp)+' does not map to a HHID');
            continue;
        end;
      end;
      i := temp mod split_count;
      //
      inc(rnum[i]);
      writeDBFRecord( outdbfil[i], dbhrec, rnum[i]);
    end;

    { write head and tail for split files }
    for i := 0 to split_count-1 do begin
      writeln('Wrote '+IntToStr(rnum[i])+' records to file '+IntToStr(i)+'of'+IntToStr(split_count));
      dbhrec.nrecs:=rnum[i];
      writeDBFFileHeader(outdbfil[i], dbhrec);
      writeDBFFileEOF(outdbfil[i], dbhrec);
      CloseFile(outdbfil[i]);
    end;

    CloseFile(dbfil);

  end;

  writeln(DateTimeToStr(Now)+' Clean Finish');
  // stop program loop
  Terminate;
end;

constructor TDbfHacker.Create(TheOwner: TComponent);
begin
  inherited Create(TheOwner);
  StopOnException:=True;
end;

destructor TDbfHacker.Destroy;
begin
  inherited Destroy;
end;

procedure TDbfHacker.WriteHelp;
begin
  { add your help code here }
  writeln('Usage: ', ExeName, ' -h');
  writeln(ExeName, ' -f sourcefile -n number_of_splits [--hhid=HHID] [--pid=PERSONID] [--key=PERSONFILE]');
  writeln('  note that field_to_split_on must be an integer field in the source file');
end;

var
  Application: TDbfHacker;
begin
  Application:=TDbfHacker.Create(nil);
  Application.Title:='DbfHacker';
  Application.Run;
  Application.Free;
end.
'''


def df2dbf(df, dbf_path, my_specs=None):
	'''


	Convert a pandas.DataFrame into a dbf.
	__author__  = "Dani Arribas-Bel <darribas@asu.edu> "
	...
	Arguments
	---------
	df          : DataFrame
				  Pandas dataframe object to be entirely written out to a dbf
	dbf_path    : str
				  Path to the output dbf. It is also returned by the function
	my_specs    : list
				  List with the field_specs to use for each column.
				  Defaults to None and applies the following scheme:
					* int: ('N', 14, 0)
					* float: ('N', 14, 14)
					* str: ('C', 14, 0)

	from: https://github.com/GeoDaSandbox/sandbox/blob/master/pyGDsandbox/dataIO.py

	Copyright (c) 2007-2011, GeoDa Center for Geospatial Analysis and Computation
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:

	* Redistributions of source code must retain the above copyright notice, this
	  list of conditions and the following disclaimer.

	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.

	* Neither the name of the GeoDa Center for Geospatial Analysis and Computation
	  nor the names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
	CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
	INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
	MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
	CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
	SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
	LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
	USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
	LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
	ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
	POSSIBILITY OF SUCH DAMAGE.
	'''

	import pysal as ps
	import numpy as np
	if my_specs:
		specs = my_specs
	else:
		type2spec = {int: ('N', 20, 0),
					 np.int64: ('N', 20, 0),
					 float: ('N', 36, 15),
					 np.float64: ('N', 36, 15),
					 str: ('C', 14, 0)
					 }
		types = [type(df[i].iloc[0]) for i in df.columns]
		specs = [type2spec[t] for t in types]
	db = ps.open(dbf_path, 'w')
	db.header = list(df.columns)
	db.field_spec = specs
	for i, row in df.T.iteritems():
		db.write(row)
	db.close()
	return dbf_path

