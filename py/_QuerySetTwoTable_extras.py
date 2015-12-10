
from .core import SQLiteError as _SQLiteError


def info(self, format=None):
	if format=='html':
		s = "<table>"
		from .core import IntStringDict
		d = lambda x: x if not isinstance(x,IntStringDict) else dict(x)
		p = lambda x: str(d(x)).replace("\n","\n           \t")
		s += "<tr>"
		s += "<th>idco query:</th><td>{}</td>\n".format(p(self.get_idco_query()))
		s += "</tr><tr>"
		s += "<th>idca query:</th><td>{}</td>\n".format(p(self.get_idca_query()))
		s += "</tr><tr>"
		s += "<th>alts query:</th><td>{}</td>\n".format(p(self.get_alts_query()))
		try:
			ch = self.choice
		except AttributeError:
			ch = "(error/not implemented)"
		try:
			wg = self.weight
		except AttributeError:
			wg = "(error/not implemented)"
		try:
			av = self.avail
		except AttributeError:
			av = "(error)"
		s += "</tr><tr>"
		s += "<th>choice:</th><td>{}</td>\n".format(p(ch))
		s += "</tr><tr>"
		s += "<th>weight:</th><td>{}</td>\n".format(p(wg))
		s += "</tr><tr>"
		s += "<th>avail:</th><td>{}</td>".format(p(av))
		s += "</tr>"
		s += "</table>"
		return s
	else:
		s = ""
		from .core import IntStringDict
		d = lambda x: x if not isinstance(x,IntStringDict) else dict(x)
		p = lambda x: str(d(x)).replace("\n","\n           \t")
		s += "idco query:\t{}\n".format(p(self.get_idco_query()))
		s += "idca query:\t{}\n".format(p(self.get_idca_query()))
		s += "alts query:\t{}\n".format(p(self.get_alts_query()))
		try:
			ch = self.choice
		except AttributeError:
			ch = "<error/not implemented>"
		try:
			wg = self.weight
		except AttributeError:
			wg = "<error/not implemented>"
		try:
			av = self.avail
		except AttributeError:
			av = "<error>"
		s += "choice:    \t{}\n".format(p(ch))
		s += "weight:    \t{}\n".format(p(wg))
		s += "avail:     \t{}".format(p(av))
		return s



_alts_query_doc = """\
This attribute defines a SQL query that evaluates to an larch_alternatives table. The table
should have the following features:

	* Column 1: id (integer) a key for every alternative observed in the sample
	* Column 2: name (text) a name for each alternative
"""

_idco_query_doc = """\
This attribute defines a SQL query that evaluates to an larch_idco table. The table
should have the following features:

	* Column 1: caseid (integer) a key for every case observed in the sample
	* Column 2+ can contain any explanatory data, typically numeric data, although non-numeric data is allowable.
"""


_idca_query_doc = """\
This attribute defines a SQL query that evaluates to an larch_idca table. The table
should have the following features:

	* Column 1: caseid (integer) a key for every case observed in the sample
	* Column 2: altid (integer) a key for each alternative available in this case
	* Column 3+ can contain any explanatory data, typically numeric data, although non-numeric data is allowable.
"""

alts_query = property(lambda self: self.get_alts_query(), lambda self,w: self.set_alts_query(w), None, _alts_query_doc)
idco_query = property(lambda self: self.get_idco_query(), lambda self,w: self.set_idco_query(w), None, _idco_query_doc)
idca_query = property(lambda self: self.get_idca_query(), lambda self,w: self.set_idca_query(w), None, _idca_query_doc)








_alts_values_doc = """\
This attribute defines a set of alternative codes and names, as a dictionary
that contains {integer:string} key/value pairs, where
each key is an integer value corresponding to an alternative code, and each
value is a string giving a descriptive name for the alternative.
When assigning to this attribute, a
query is defined that can be used with no table.

.. warning:: Using this method will overwrite :attr:`alts_query`
"""




alts_values = property(lambda self: self._get_alts_values(), lambda self,w: self.set_alts_values(w), None, _alts_values_doc)



def set_avail(self, x):
	if x is None:
		self.set_avail_ca_column("1")
	elif isinstance(x,(int,float)) and x==1:
		self.set_avail_ca_column("1")
	elif isinstance(x,bool) and x is True:
		self.set_avail_all()
	elif isinstance(x,str):
		self.set_avail_ca_column(x)
	else:
		self.set_avail_co_column_map(x)

def get_avail(self):
	if self.all_alts_always_available():
		return True
	if self.get_avail_ca_column()!="" and len(self.get_avail_co_column_map())==0:
		return self.get_avail_ca_column()
	if self.get_avail_ca_column()=="" and len(self.get_avail_co_column_map())>0:
		return self.get_avail_co_column_map()
	raise LarchError("bad avail specification")

_avail_doc ="""\
This attributes defines the availability of alternatives for each case.  If set to `True`,
then all alternatives are available to all cases.  Otherwise, this attribute should be either:

	* a dict that contains {integer:string} key/value pairs, where
	  each key is an integer value corresponding to an alternative code, and each
	  value is a string identifying an expression evaluated on the idco table; that expression should
	  evaluate to a value indicating whether the keyed alternative is available. This must be
	  a binary dummy variable.
	* a string identifying an expression evaluated on the idca table; that expression should
	  evaluate to a value indicating whether the alternative for that row is available. This must be
	  a binary dummy variable.
	* If set to 1 or None, then the string "1" is used.

"""

avail = property(lambda self: self.get_avail(), lambda self,w: self.set_avail(w), None, _avail_doc)


def _get_choice_plus(self):
	x = self.get_choice_co()
	if x!="":
		return x, 'co'
	x = self.get_choice_ca()
	if x!="":
		return x, 'ca'
	x = self.get_choice_co_map()
	if len(x)>0:
		return x, 'map'
	return None, 'blank'


def set_choice(self, x):
	current, currenttable = self._get_choice_plus()
	if isinstance(x,dict):
		self.set_choice_co_map(x)
		return
	if not isinstance(x,str):
		raise TypeError("choice must be str or dict, not %s"%str(type(x)))
	try:
		self.set_choice_ca(x)
	except _SQLiteError:
		# the choice is not a valid expression in the idca query
		try:
			self.set_choice_co(x)
			return
		except _SQLiteError:
			raise ValueError("choice not a valid expression for either idca or idco queries")
	else:
		# the choice is a valid expression in the idca query
		try:
			self.set_choice_co(x)
		except _SQLiteError:
			return
		except:
			raise
		else:
			if currenttable=='ca':
				self.set_choice_ca(x)
				return
			elif currenttable=='co':
				self.set_choice_co(x)
				return
			elif currenttable=='map':
				self.set_choice_co_map(x)
				return
			elif currenttable=='blank':
				raise ValueError("choice is a valid expression for both idca or idco queries, must set using set_choice_co() or set_choice_ca()")
			else:
				raise ValueError("fatal error in setting choice, the queries may be invalid")
			raise ValueError("choice is a valid expression for both idca or idco queries, must set using set_choice_co() or set_choice_ca()")

def _get_choice(self):
	return self._get_choice_plus()[0]

_doc_choice = """\
This attribute defines the choices. It has two styles:

	* When set to a string, the string gives an expression evaluated on one of
	  the two main tables that identifies the choice for each case.  If the expression
	  is evaluated on the idco table, it should result in integer values
	  corresponding to the alternative codes. If the expression
	  is evaluated on the idca table, it should evaluate to 1 if the alternative for the
	  particular row was chosen, and 0 otherwise. (For certain specialized models,
	  values other than 0 or 1 may be appropriate.) If the expression is part of a valid query on
	  both main tables, this attribute cannot be set directly, and must instead be set using
	  either :meth:`set_choice_ca` or :meth:`set_choice_co`.

	* When set to a dict, the dict should contain {integer:string} key/value pairs, where
	  each key is an integer value corresponding to an alternative code, and each
	  value is a string identifying an expression evaluated on the idco table; the result should
	  contain a value indicating whether the alternative was chosen. Usually this will be
	  a binary dummy variable, although it need not be. For certain specialized models,
	  values other than 0 or 1 may be appropriate.

The choice of style is a matter of convenience; the same data can be expressed with either
style as long as the choices are binary, or if the first style names an expression in the idca table.

"""

choice = property(lambda self: self._get_choice(), lambda self,w: self.set_choice(w), None, _doc_choice)



_weight_doc = """\
This attribute gives an expression that is evaluated on the idco table to defines the weight for each case.
Set it to an empty string, or 1.0, to assign all cases equal weight. Te weight cannot be set based on the 
idca table.
"""

weight = property(lambda self: self.get_weight_co_column(), lambda self,w: self.set_weight_co_column(str(w)), None, _weight_doc)


def quality_check(self):
	warns = []
#	validator = self.get_validator()
#	if validator is None:
#		return
#	if not (self.avail is True) and not isinstance(self.avail, str):
#		import warnings
#		for altnum, altavail in dict(self.avail).items():
#			try:
#				altname = "{} (Alt {})".format( self.alts_values[altnum], altnum )
#			except (KeyError, IndexError):
#				altname = "Alt {}".format(altnum)
#			bads = validator.value("SELECT count(*) FROM larch_idco WHERE NOT ({0}) AND ({1}={2})".format(altavail,self.choice,altnum),cte=True)
#			if bads > 0:
#				warns += ["Warning: {} has {} instances where it is chosen but explicitly not available".format(altname, bads),]
#				warnings.warn(warns[-1], stacklevel=2)
#			nulls = validator.value("SELECT count(*) FROM larch_idco WHERE ({0}) IS NULL AND ({1}={2})".format(altavail,self.choice,altnum),cte=True)
#			if nulls > 0:
#				warns += ["Warning: {} has {} instances where it is chosen but implicitly not available (criteria evaluates NULL)".format(altname, nulls),]
#				warnings.warn(warns[-1], stacklevel=2)
#	elif not (self.avail is True) and isinstance(self.avail, str) and validator is not None:
#		pass
#		# m.Data("Choice")[~m.Data("Avail")].sum() > 0 means problems
	return warns


spork = lambda self: print("SPORK")

def _spong(self, n):
	print(n, "_splong!")


## Load these methods into core.QuerySetTwoTable
import_these = dict(locals())

from .core import QuerySetTwoTable
_private_methods = ['_get_choice_plus', '_get_choice', ]

for k,f in import_these.items():
	if len(k)>0 and (k[0]!='_' or k in _private_methods):
		setattr(QuerySetTwoTable,k,f)


