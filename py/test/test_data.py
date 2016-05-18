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

if __name__ == "__main__" and __package__ is None:
    __package__ = "larch.test.test_data"


import nose, unittest
from nose.tools import *
from ..test import TEST_DATA, ELM_TestCase, DEEP_TEST
from ..db import DB
from ..dt import DT
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
		from ..util import stored_dict
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


	def test_pytables_examples(self):
		dts = DT.Example('SWISSMETRO')
		ms = Model.Example(101)
		ms.db = dts
		ms.provision()
		x = [-0.7012268762617896, -0.15465520761303447, -0.01277806274978315, -0.01083774419411773]
		self.assertAlmostEqual(  -5331.252007380466 , ms.loglike(x,cached=False))
		dt = DT.Example()
		dt.h5top.screen[:10] = False
		rr = dt.array_idca('_avail_*hhinc')
		self.assertEqual( rr.shape, (5019, 6, 1) )
		self.assertTrue( numpy.allclose( rr[0], numpy.array([[ 42.5],[ 42.5],[ 42.5],[ 42.5],[ 42.5],[  0. ]]) ))
		rr1 = dt.array_idca('_avail_')
		rr2 = dt.array_idca('hhinc')
		self.assertEqual( rr1.shape, (5019, 6, 1) )
		self.assertEqual( rr2.shape, (5019, 6, 1) )
#		m = Model.Example()
#		m.db = dt
#		m.utility.ca('exp(log(ivtt))+ovtt+altnum')
#		m.maximize_loglike()
#		self.assertAlmostEqual(   -3616.461567801068 , m.loglike())
#		self.assertEqual(   5019 , m.nCases())

	def test_pytables_examples_validate(self):
		d1=DT.Example('MTC')
		self.assertEqual( 0, d1.validate_hdf5(log=(lambda y: None), errlog=print) )
		del d1
		d2=DT.Example('SWISSMETRO')
		self.assertEqual( 0, d2.validate_hdf5(log=(lambda y: None), errlog=print) )
		del d2
		d3=DT.Example('ITINERARY')
		self.assertEqual( 0, d3.validate_hdf5(log=(lambda y: None), errlog=print) )
		del d3
		d4=DT.Example('MINI')
		self.assertEqual( 0, d4.validate_hdf5(log=(lambda y: None), errlog=print) )
		del d4


	def test_autoindex_string(self):
		from ..core import autoindex_string
		a = autoindex_string( ['Hello','World!'] )
		self.assertEqual( 1, a['World!'] )
		self.assertEqual( 0, a['Hello'] )
		self.assertEqual( 2, a['Earth!'] )
		self.assertEqual( 1, a.drop('World!') )
		self.assertEqual( 1, a['Earth!'] )
		self.assertEqual( 1, a[-1] )
		with self.assertRaises(IndexError):
			a[-3]
		with self.assertRaises(IndexError):
			a[3]
		a.extend(['a','b','c'])
		self.assertEqual( 5, len(a) )
		self.assertEqual( 3, a['b'] )


	def test_html_reporting(self):
		m = Model.Example(1, pre=True)
		from ..util.pmath import category, rename
		from ..util.xhtml import XHTML
		import re
		param_groups = [
			category('Level of Service', 
					 rename('Total Time', 'tottime'),
					 rename('Total Cost', 'totcost'),
					),
			category('Alternative Specific Constants', 
						'ASC_SR2',
						'ASC_SR3P',
						'ASC_TRAN',
						'ASC_BIKE',
						'ASC_WALK',
					),
			category('Income', 
						'hhinc#2',
						'hhinc#3',
						'hhinc#4',
						'hhinc#5',
						'hhinc#6',
					),
		]
		with XHTML(quickhead=m) as f:
			f << m.xhtml_title()
			f << m.xhtml_params(param_groups)
			s = f.dump()
		self.assertTrue(re.compile(b'<td.*class="parameter_category".*>Level of Service</td>').search(s) is not None)
		self.assertTrue(re.compile(b'<td.*class="parameter_category".*>Alternative Specific Constants</td>').search(s) is not None)
		self.assertTrue(re.compile(b'<td><a.*></a>Total Cost</td><td.*>-0.00492\s*</td>').search(s) is not None)


	def test_pytables_import_idco_with_nulls(self):
		dt = DT()
		tinytest = os.path.join( DT.ExampleDirectory(), 'tinytest.csv' )
		dt.import_idco(tinytest)
		self.assertEqual( 2, dt.h5idco.Banana[0] )
		self.assertTrue( numpy.isnan(dt.h5idco.Banana[1]) )
		self.assertTrue( numpy.isnan(dt.h5idco.Banana[-1]) )
		purch = numpy.array([b'Apple', b'Cookie', b'Apple', b'Banana', b'Cookie', b'Apple',
							b'Apple', b'Cookie', b'Cookie'],
							dtype='|S8')
		self.assertTrue(numpy.array_equal( purch, dt.h5idco.Purchase[:] ))
		apple = numpy.array([1, 1, 2, 1, 1, 2, 1, 1, 2])
		self.assertTrue( numpy.array_equal(apple, dt.h5idco.Apple[:]) )

	def test_pytables_import_idca_with_nulls(self):
		dt = DT()
		tinytest = os.path.join( DT.ExampleDirectory(), 'tinytest_idca.csv' )
		dt.import_idca(tinytest, caseid_col='Customer', altid_col='Product')
		self.assertTrue(numpy.array_equal( ['Apple','Banana','Cookie'], dt.alternative_names() ))
		self.assertEqual( 2, dt.h5idca.Price[0,1] )
		self.assertEqual( 0, dt.h5idca.Price[1,1] ) # missing values are 0 not NAN in idca load
		self.assertTrue( numpy.isnan(dt.h5idca.Price[-1,1]) )
		purch = numpy.array([  [ 1.,  0.,  0.],
							   [ 0.,  0.,  1.],
							   [ 1.,  0.,  0.],
							   [ 0.,  1.,  0.],
							   [ 0.,  0.,  1.],
							   [ 1.,  0.,  0.],
							   [ 1.,  0.,  0.],
							   [ 0.,  0.,  1.],
							   [ 0.,  0.,  1.]])
		self.assertTrue(numpy.array_equal( purch, dt.h5idca.Purchased[:] ))
		apple = numpy.array([1, 1, 2, 1, 1, 2, 1, 1, 2])
		self.assertTrue( numpy.array_equal(apple, dt.h5idca.Price[:,0]) )


