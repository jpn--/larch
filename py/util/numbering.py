from enum import Enum

def enum_bitmask(enumeration):
	mask = 0
	for i in enumeration:
		mask |= i.value
	return mask

def enum_bitmask_length(enumeration):
	return enum_bitmask(enumeration).bit_length()

class numbering_system():
	def __init__(self, *enumerations):
		self.enumerations = enumerations
		self.bitmasks = [ enum_bitmask(i) for i in enumerations ]
		self.bitmask_lengths = [ enum_bitmask_length(i) for i in enumerations ]
		self.shifts = [ ]
		self.total_shift = 0
		for j in range(len(self.enumerations)):
			shift = self.bitmask_lengths[j]
			self.bitmasks[j+1:] = [k<<shift for k in self.bitmasks[j+1:]]
			self.shifts += [self.total_shift,]
			self.total_shift += shift
	def code_from_attributes(self, instance_number, *attributes):
		assert( isinstance(instance_number, int) )
		assert( instance_number>=0 )
		assert( len(attributes)==len(self.enumerations) )
		code = instance_number<<self.total_shift
		for j in range(len(self.enumerations)):
			if attributes[j] in self.enumerations[j]:
				code += attributes[j].value<<self.shifts[j]
			else:
				# pass value through enum to make sure it's valid
				code += self.enumerations[j](attributes[j]).value<<self.shifts[j]
		return code
	def attributes_from_code(self, code):
		attributes = (self.enumerations[j]((code&self.bitmasks[j])>>self.shifts[j]) for j in range(len(self.enumerations)))
		attributes = (code>>self.total_shift,) + tuple(attributes)
		return attributes
	def code_matches_attributes(self, code, *attributes):
		code_attrrib = self.attributes_from_code(code)
		return code_attrrib[1:] == attributes

import itertools

def recode_alts(self, ns, tablename, casecol, newaltcol, *cat_columns, logfreq=200, newaltstable=None):
	"""
	This function renumbers the alternatives to use a new numbering_system.
	
	Parameters
	----------
	ns : numbering_system
		A :class:`numbering_system` object that defines the categories that will be
		used to renumber the alternatives.
	tablename : str
		The name of the table containing the alternatives to be renumbered. Probably an
		:ref:`idca` format table. It needs to be a bonafide table and not a view as the
		content of the table will be altered by this method.
	casecol : str
		The name of the column in the table that contains the caseids.
	newaltcol : str
		The name of a column in the table that will contain the new alternative codes.
		If this column does not yet exist it will be added to the table.  If it does 
		exist it will be overwritten.
	cat_columns : tuple of str
		One expression for each enumeration contained in `ns`, that will evaluate to
		an integer that can be passed to each enumeration (in order) to initialize a 
		member.
	
	Other Parameters
	----------------
	logfreq : int, optional
		If given, a message will be logged each time this many cases have been processed.
		Since the recoding may proceed slowly, this can give the user a hint at how
		long it might take to finish.
	newaltstable : str, optional
		If given, all newly created alternative codes will be assembled in a table of the
		given name, in a format suited for use in an alternatives query. If a :attr:`queries`
		object is set up, the :attr:`queries.alts_query` will be adjusted accordingly.
	
	"""
	assert(isinstance(ns, numbering_system))
	assert(isinstance(tablename, str))
	assert(isinstance(casecol, str))
	assert(isinstance(newaltcol, str))
	db = self
	log = db.logger()
	if len(cat_columns)!=len(ns.enumerations):
		raise TypeError("must specify {} category columns".format(len(ns.enumerations)))
	cat_vals = None
	db.add_column(tablename, newaltcol+" INTEGER")
	caseids = db.values("SELECT DISTINCT {1} FROM {0}".format(tablename,casecol))
	reportnum = 1
	try:
		for c in caseids:
			qry = "SELECT rowid, {2} FROM {0} WHERE {1}=?".format(tablename,casecol,", ".join(cat_columns))
			rows = [i for i in db.execute(qry,(c,))]
			ticker = {j:1 for j in itertools.product(*ns.enumerations)}
			db.execute("begin transaction;")
			for row in rows:
				cat_vals = tuple(converter(row[1+i]) for i,converter in enumerate(ns.enumerations))
				db.execute("UPDATE data_ca SET {}=? WHERE rowid=?".format(newaltcol), (ns.code_from_attributes(ticker[cat_vals],*cat_vals), row[0]))
				ticker[cat_vals] += 1
			db.execute("end transaction;")
			reportnum += 1
			if log and logfreq:
				if reportnum%logfreq==0:
					log("updating case %i",c)
	except:
		raise
	if newaltstable is not None:
		qry = """
		CREATE TABLE IF NOT EXISTS {0} (id PRIMARY KEY, name TEXT);
		INSERT OR IGNORE INTO {0}
		SELECT DISTINCT {1}, 'a'||{1} FROM {2};
		""".format(newaltstable, newaltcol, tablename)
		db.execute(qry)
		if self.queries:
			try:
				self.queries.alts_query = 'SELECT * FROM {}'.format(newaltstable)
			except:
				pass
			try:
				self.queries.idca_build(altcol=newaltcol)
			except:
				pass
	db.uncache_alternatives()



