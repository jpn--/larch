#
#  Copyright 2007-2015 Jeffrey Newman
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

if __name__ == "__main__" and __package__ is None:
    __package__ = "larch.test.test_data"


import nose, unittest
from nose.tools import *
from ..test import TEST_DATA, ELM_TestCase, DEEP_TEST
from ..db import DB
from ..model import Model
from ..core import LarchError, SQLiteError, FacetError, darray_req, LinearComponent
from ..exceptions import *
from ..array import Array, ArrayError
import shutil, os
import numpy, pandas

class TestSwigCommmands(ELM_TestCase):
	def test_dictionaries(self):
		from ..core import _swigtest_alpha_dict, _swigtest_empty_dict
		self.assertEqual({'a':1.0, 'b':2.0, 'c':3.0}, _swigtest_alpha_dict())
		self.assertEqual({}, _swigtest_empty_dict())


class TestData1(unittest.TestCase):
	def test_basic_stats(self):
		d = DB.Example('MTC')

		self.assertEqual(5029, d.nCases())		
		self.assertEqual(6, d.nAlts())
		
		AltCodes = d.alternative_codes()
		self.assertTrue( len(AltCodes) == 6 )
		self.assertTrue( 1 in AltCodes )  
		self.assertTrue( 2 in AltCodes )  
		self.assertTrue( 3 in AltCodes )  
		self.assertTrue( 4 in AltCodes )  
		self.assertTrue( 5 in AltCodes )  
		self.assertTrue( 6 in AltCodes )  

		AltNames = d.alternative_names()
		self.assertTrue( len(AltNames) == 6 )
		self.assertTrue( 'DA' in AltNames )  
		self.assertTrue( 'SR2' in AltNames )  
		self.assertTrue( 'SR3+' in AltNames )  
		self.assertTrue( 'Tran' in AltNames )  
		self.assertTrue( 'Bike' in AltNames )  
		self.assertTrue( 'Walk' in AltNames )  

	def test_StoredDict(self):
		from ..utilities import stored_dict
		d = DB()
		s1 = stored_dict(d,'hello')
		s1.add('a')
		s1.add('b')
		self.assertEqual( 1, s1.a )
		self.assertEqual( 2, s1.b )
		s2 = stored_dict(d,'hello')
		self.assertEqual( 1, s2.a )
		self.assertEqual( 2, s2.b )
		s1['c'] = 5
		self.assertEqual( 5, s2.c )

	@raises(NoResultsError)
	def test_no_results(self):
		db = DB()
		db.value("select 1 limit 0;")

	@raises(TooManyResultsError)
	def test_too_many_results(self):
		db = DB()
		db.execute("CREATE TEMP TABLE zz(a); INSERT INTO zz VALUES (1);INSERT INTO zz VALUES (2);")
		db.value('''SELECT a FROM zz;''')

	def test_dataframe(self):
		cols = ['a', 'b', 'c', 'd', 'e']
		df1 = pandas.DataFrame(numpy.random.randn(10, 5), columns=cols)
		db = DB()
		db.import_dataframe(df1, table="staging", if_exists='append')
		df2 = db.dataframe("SELECT * FROM staging")
		self.assertEqual( 5, len(df2.columns) )
		self.assertEqual( 10, len(df2) )
		for c in cols:
			for i in range(10):
				self.assertEqual( df1[c][i], df2[c][i])


	def test_percentiles(self):
		db = DB.Example()
		self.assertEqual( 2515, db.value("SELECT median(casenum) FROM "+db.tbl_idco()) )
		self.assertEqual( 2515, db.value("SELECT percentile(casenum, 50) FROM "+db.tbl_idco()) )
		self.assertEqual( 3772, db.value("SELECT upper_quartile(casenum) FROM "+db.tbl_idco()))
		self.assertEqual( 3772, db.value("SELECT percentile(casenum, .75) FROM "+db.tbl_idco()))
		self.assertEqual( 5029, db.value("SELECT percentile(casenum, 1.00) FROM "+db.tbl_idco()))
		self.assertEqual( 5029, db.value("SELECT percentile(casenum, 100) FROM "+db.tbl_idco()))
		self.assertEqual( 4979, db.value("SELECT percentile(casenum, 0.99) FROM "+db.tbl_idco()))
		self.assertEqual( 4979, db.value("SELECT percentile(casenum, 99) FROM "+db.tbl_idco()))

	def test_ldarray(self):
		import numpy
		z = numpy.ones([3,3])
		with self.assertRaises(ArrayError):
			q = Array(z, vars=['a','b'])
		q = Array(z, vars=['a','b','c'])
		w = Array(z, vars=['x','b','c'])
		req = darray_req(2,numpy.dtype('float64'))
		req.set_variables(['a','b','c'])
		self.assertTrue(req.satisfied_by(q)==0)
		self.assertFalse(req.satisfied_by(w)==0)
		self.assertTrue(req.satisfied_by(z)==0)

	def test_export_import_idca(self):
		from io import StringIO
		f = StringIO()
		d = DB.Example('MTC')
		d.queries.idco_query += " WHERE casenum < 100"
		d.queries.idca_query += " WHERE casenum < 100"
		m1 = Model.Example()
		m1.db = d
		m1.provision()
		self.assertAlmostEqual( -147.81203484134275, m1.loglike(), delta= 0.000001)
		self.assertAlmostEqual( -472.8980609887492, m1.loglike((-0.01,0,0,0,0.1,0,0,0.1,0,0,0,0)), delta= 0.000001)
		d.export_idca(f)
		f.seek(0)
		x = DB.CSV_idca(f, caseid='caseid', altid='altid', choice='chose', weight=None, avail=None, tablename='data', tablename_co='_co', savename=None, alts={}, safety=True)
		self.assertEqual( 99, x.nCases() )
		self.assertEqual( 6, x.nAlts() )
		self.assertEqual( ('caseid', 'altid', 'altnum', 'chose', 'ivtt', 'ovtt', 'tottime', 'totcost'), x.variables_ca() )
		self.assertEqual( ('caseid', 'casenum', 'hhid', 'perid', 'numalts', 'dist', 'wkzone', 'hmzone', 'rspopden', 'rsempden', 'wkpopden', 'wkempden', 'vehavdum', 'femdum', 'age', 'drlicdum', 'noncadum', 'numveh', 'hhsize', 'hhinc', 'famtype', 'hhowndum', 'numemphh', 'numadlt', 'nmlt5', 'nm5to11', 'nm12to16', 'wkccbd', 'wknccbd', 'corredis', 'vehbywrk', 'vocc', 'wgt'), x.variables_co() )
		m = Model.Example()
		m.db = x
		m.provision()
		self.assertAlmostEqual( -147.81203484134275, m.loglike(), delta= 0.000001)
		self.assertAlmostEqual( -472.8980609887492, m.loglike((-0.01,0,0,0,0.1,0,0,0.1,0,0,0,0)), delta= 0.000001)


	def test_component(self):
		nullcode = -9997999
	
		c = LinearComponent()
		self.assertEqual( nullcode, c._altcode )
		self.assertEqual( nullcode, c._upcode )
		self.assertEqual( nullcode, c._dncode )
		self.assertEqual( ""      , c._altname )
		self.assertEqual( ""      , c.data )
		self.assertEqual( ""      , c.param )

		c = LinearComponent(data="123", param="par", category=(3,4))
		self.assertEqual( nullcode, c._altcode )
		self.assertEqual( 3, c._upcode )
		self.assertEqual( 4, c._dncode )
		self.assertEqual( "", c._altname )
		self.assertEqual( "123", c.data )
		self.assertEqual( "par", c.param )

		c = LinearComponent(data="123", param="par", category=5)
		self.assertEqual( 5, c._altcode )
		self.assertEqual( nullcode, c._upcode )
		self.assertEqual( nullcode, c._dncode )
		self.assertEqual( "", c._altname )
		self.assertEqual( "123", c.data )
		self.assertEqual( "par", c.param )

		c = LinearComponent(data="123", param="PAR", category="five")
		self.assertEqual( nullcode, c._altcode )
		self.assertEqual( nullcode, c._upcode )
		self.assertEqual( nullcode, c._dncode )
		self.assertEqual( "five", c._altname )
		self.assertEqual( "123", c.data )
		self.assertEqual( "PAR", c.param )

	def test_numbering_system(self):
		from ..util.numbering import numbering_system
		from enum import Enum
		class levels_of_service(Enum):
			nonstop = 1
			withstop = 2
		class carriers(Enum):
			DL = 1
			US = 2
			UA = 3
			AA = 4 
			Other = 5
		class things(Enum):
			Apple = 1
			Orange = 2
			Hat = 3
			Boot = 4
			Camera = 5
			Box = 10
			Squid = 12
			Dog = 13
			Cat = 14
			Sun = 15
		ns = numbering_system(levels_of_service, carriers, things)
		self.assertEqual( ['0b11', '0b11100', '0b111100000'], [bin(a) for a in ns.bitmasks] )
		self.assertEqual( [0, 2, 5], [s for s in ns.shifts] )
		nn = ns.code_from_attributes(1, levels_of_service.withstop, carriers.UA, things.Cat)
		self.assertEqual(974, nn)
		x = ns.attributes_from_code(nn)
		self.assertEqual( (1, levels_of_service.withstop, carriers.UA, things.Cat), x)

