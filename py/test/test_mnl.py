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


import os
import unittest
import numpy

if __name__ == "__main__" and __package__ is None:
    __package__ = "larch.test.test_mnl"

from ..test import TEST_DATA, ELM_TestCase, DEEP_TEST
from ..core import Parameter, Model, DB, LarchError, SQLiteError, LarchCacheError
from ..model import ModelFamily
from ..roles import ParameterRef

class TestMTC(ELM_TestCase):

	_multiprocess_shared_ = True # nose will run setUpClass once for all tests if true

	@classmethod
	def setUpClass(cls):
		# Expensive fixture setup method goes here
		cls._db = DB.Copy(TEST_DATA['MTC WORK MODE CHOICE'])

	@classmethod
	def tearDownClass(cls):
		# Expensive fixture teardown method goes here
		del cls._db

	def test_sqlite_math_extensions(self):
		d = self._db
		self.assertEqual(7.69, d.eval_float("select dist from "+d.tbl_idco()+" limit 1"))
		self.assertEqual(2.0399207835175526, d.eval_float("select log(dist) from "+d.tbl_idco()+" limit 1"))
		self.assertEqual(2186.374562228241, d.eval_float("select exp(dist) from "+d.tbl_idco()+" limit 1"))
		self.assertAlmostEqual(0.885926339801431, d.eval_float("select log10(dist) from "+d.tbl_idco()+" limit 1"))
		self.assertAlmostEqual(163.989427419834, d.eval_float("select power(dist,2.5) from "+d.tbl_idco()+" limit 1"))
		self.assertEqual(1, d.eval_float("select sign(dist) from "+d.tbl_idco()+" limit 1"))
		self.assertAlmostEqual(2.77308492477241, d.eval_float("select sqrt(dist) from "+d.tbl_idco()+" limit 1"))
		self.assertAlmostEqual(7.69*7.69, d.eval_float("select square(dist) from "+d.tbl_idco()+" limit 1"))
		self.assertEqual(8, d.eval_float("select ceil(dist) from "+d.tbl_idco()+" limit 1"))
		self.assertEqual(7, d.eval_float("select floor(dist) from "+d.tbl_idco()+" limit 1"))

	def test_parameter_math(self):
		m = Model()
		m.parameter('P1', value=1.23)
		m.parameter('P2', value=2.0)
		
		P1 = ParameterRef('P1')
		P1x = ParameterRef('P1', fmt="{:0.1f}")
		P2 = ParameterRef('P2')
		P3 = ParameterRef('P3', default=10.0)
		P4 = ParameterRef('P4')
		
		self.assertEqual( 1.23, P1.value(m) )
		self.assertEqual( 2.0,  P2.value(m) )
		self.assertEqual( 10.0,  P3.value(m) )
		with self.assertRaises(LarchError):
			P4.value(m)

		self.assertAlmostEqual( 3.23   , (P1+P2).value(m) )
		self.assertAlmostEqual( 3.23   , (P2+P1).value(m) )
		self.assertAlmostEqual(1.23-2.0, (P1-P2).value(m) )
		self.assertAlmostEqual( 0.77   , (P2-P1).value(m) )
		self.assertAlmostEqual( 2.46   , (P1*P2).value(m) )
		self.assertAlmostEqual( 2.46   , (P2*P1).value(m) )
		self.assertAlmostEqual( 0.615  , (P1/P2).value(m) )
		self.assertAlmostEqual(2.0/1.23, (P2/P1).value(m) )
		self.assertAlmostEqual(-1.23   , (-P1).value(m) )
	
		self.assertEqual( '1.2', P1x.str(m) )
		self.assertEqual( '2.5', (P1x*P2).str(m) )
		self.assertEqual( '2.5', (P2*P1x).str(m) )
		self.assertEqual( '3.2', (P1x+P2).str(m) )
		self.assertEqual( '3.2', (P2+P1x).str(m) )
		self.assertEqual( '0.6', (P1x/P2).str(m) )
		self.assertEqual( '1.6', (P2/P1x).str(m) )
		self.assertEqual( '-0.8',(P1x-P2).str(m) )
		self.assertEqual( '0.8', (P2-P1x).str(m) )
		self.assertEqual( '-1.2',(-P1x).str(m) )

	def test_reporting(self):
		m = Model.Example(1, pre=True)
		r1 = m.report(style='html')
		r2 = m.report(style='xml')
		r3 = m.report(style='txt')
		self.assertEqual(bytes, type(r1))
		from ..util.xhtml import Elem
		self.assertEqual(Elem, type(r2))
		self.assertEqual(str, type(r3))

	def test_model2_options(self):		
		d = self._db
		m = Model (d)
		m.option.gradient_diagnostic = 1
		m.option.threads = 4
		self.assertTrue(m.option.gradient_diagnostic)
		self.assertEqual(4, m.option.threads)

	def test_model2_tally_chosen_and_avail(self):
		d = self._db
		m = Model (d)
		avail = numpy.ascontiguousarray([ 4755.,  5029.,  5029.,  4003.,  1738.,  1479.])
		#self.assertArrayEqual(avail, m.tally_avail())
		chose = numpy.ascontiguousarray([ 3637.,   517.,   161.,   498.,    50.,   166.])
		#self.assertArrayEqual(chose, m.tally_chosen())

	def test_model2_mnl(self):
		d = self._db
		m = Model (d)
		m.option.calc_std_errors = False
		m.parameter("cost",-0.01) 
		m.parameter("tottime",0) 
		m.parameter("con2",0) 
		m.parameter("con3",0) 
		m.parameter("con4",0.1) 
		m.parameter("con5",0) 
		m.parameter("con6",0) 
		m.parameter("inc2",0.1) 
		m.parameter("inc3",0) 
		m.parameter("inc4",0) 
		m.parameter("inc5",0) 
		m.parameter("inc6",0) 
		m.utility.ca("tottime")
		m.utility.ca("totcost","cost")
		m.utility.co("HHINC",2,"inc2") 
		m.utility.co("HHINC",3,"inc3") 
		m.utility.co("HHINC",4,"inc4") 
		m.utility.co("HHINC",5,"inc5") 
		m.utility.co("HHINC",6,"inc6") 
		m.utility.co("1","SR2","con2") 
		m.utility.co("1","SR3+","con3") 
		m.utility.co("1","Tran","con4") 
		m.utility.co("1","Bike","con5") 
		m.utility.co("1","Walk","con6") 
		with self.assertRaises(Exception):
			m.utility.co("HHINC","AltNameWhichDoesNotExist")
		with self.assertRaises(Exception):
			m.utility.co("HHINC",666)
		del m["HHINC#666"]
		del m.utility.co[666]
		with self.assertRaises(Exception):
			m.utility.ca("CA_ColumnNameWhichDoesNotExist")
		del m.utility.ca[-1]
		del m["CA_ColumnNameWhichDoesNotExist"]
		with self.assertRaises(Exception):
			m.utility.co("CO_ColumnNameWhichDoesNotExist","Tran")
		del m["CO_ColumnNameWhichDoesNotExist#4"]
		m.utility.clean(d)
		m.setUp()
		self.assertEqual(['cost','tottime','con2','con3','con4','con5','con6','inc2','inc3','inc4','inc5','inc6'],m.parameter_names())
		self.assertEqual((-0.01,0,0,0,0.1,0,0,0.1,0,0,0,0),m.parameter_values())
		self.assertAlmostEqual(-28993.7, m.loglike(), delta=0.05)
		g = m.negative_d_loglike()
		self.assertAlmostEqual(   -226530, g[0], delta=1. )
		self.assertAlmostEqual(   11196.5, g[1], delta=.05 )
		self.assertAlmostEqual(   4084.42, g[2], delta=.005)
		self.assertAlmostEqual(  0.375794, g[3], delta=.0000005 )
		self.assertAlmostEqual(  -409.382, g[4], delta=.0005 )
		self.assertAlmostEqual(   4.78092, g[5], delta=.000005 )
		self.assertAlmostEqual(  -102.757, g[6], delta=.0005 )
		self.assertAlmostEqual(    253613, g[7], delta=.5 )
		self.assertAlmostEqual(  -5561.31, g[8], delta=.005 )
		self.assertAlmostEqual(  -24352.9, g[9], delta=.05 )
		self.assertAlmostEqual(  -820.592, g[10],delta=.0005 )
		self.assertAlmostEqual(  -5794.31, g[11],delta=.005 )
		m.tearDown()
		m.option.gradient_diagnostic = 2 
		z = m.estimate()
		self.assertAlmostEqual( -3626.1862548451313, m.loglike(cached=2))

	def test_model2_mnl_with_constant_parameter(self):
		d = self._db
		m = Model (d)
		m.option.calc_std_errors = False
		m.parameter("cost",-0.01) 
		m.parameter("con2",0) 
		m.parameter("con3",0) 
		m.parameter("con4",0.1) 
		m.parameter("con5",0) 
		m.parameter("con6",0) 
		m.parameter("inc2",0.1) 
		m.parameter("inc3",0) 
		m.parameter("inc4",0) 
		m.parameter("inc5",0) 
		m.parameter("inc6",0) 
		m.utility.ca("tottime","ConstanT",-0.001) 
		m.utility.ca("totcost","cost")
		m.utility.co("HHINC","SR2","inc2") 
		m.utility.co("HHINC","SR3+","inc3") 
		m.utility.co("HHINC","Tran","inc4") 
		m.utility.co("HHINC","Bike","inc5") 
		m.utility.co("HHINC","Walk","inc6") 
		m.utility.co("1","SR2","con2") 
		m.utility.co("1","SR3+","con3") 
		m.utility.co("1","Tran","con4") 
		m.utility.co("1","Bike","con5") 
		m.utility.co("1","Walk","con6") 
		m.setUp()
		self.assertAlmostEqual(-28982.6, m.loglike(), delta=0.05)
		g = m.negative_d_loglike()
		self.assertAlmostEqual(   -226372, g[0], delta=.5 )
		self.assertAlmostEqual(   4087.18, g[1], delta=.005 )
		self.assertAlmostEqual(  0.567559, g[2], delta=.0000005 )
		self.assertAlmostEqual(  -410.803, g[3], delta=.0005 )
		self.assertAlmostEqual(   4.36171, g[4], delta=.000005 )
		self.assertAlmostEqual(  -104.307, g[5], delta=.0005 )
		self.assertAlmostEqual(    253697, g[6], delta=.5 )
		self.assertAlmostEqual(  -5558.95, g[7], delta=.005 )
		self.assertAlmostEqual(  -24395.3, g[8], delta=.05 )
		self.assertAlmostEqual(  -835.116, g[9], delta=.0005 )
		self.assertAlmostEqual(  -5833.57, g[10],delta=.005)
		m.tearDown()

	def test_model2_mnl_with_holdfast_parameter(self):
		d = self._db
		m = Model (d)
		m.option.calc_std_errors = False
		m.parameter("cost",-0.01)
		m.parameter("time",-0.001, holdfast=1)
		m.parameter("con2",0) 
		m.parameter("con3",0) 
		m.parameter("con4",0.1) 
		m.parameter("con5",0) 
		m.parameter("con6",0) 
		m.parameter("inc2",0.1) 
		m.parameter("inc3",0) 
		m.parameter("inc4",0) 
		m.parameter("inc5",0) 
		m.parameter("inc6",0) 
		m.utility.ca("tottime","time")
		m.utility.ca("totcost","cost")
		m.utility.co("HHINC","SR2","inc2") 
		m.utility.co("HHINC","SR3+","inc3") 
		m.utility.co("HHINC","Tran","inc4") 
		m.utility.co("HHINC","Bike","inc5") 
		m.utility.co("HHINC","Walk","inc6") 
		m.utility.co("1","SR2","con2") 
		m.utility.co("1","SR3+","con3") 
		m.utility.co("1","Tran","con4") 
		m.utility.co("1","Bike","con5") 
		m.utility.co("1","Walk","con6") 
		m.setUp()
		self.assertAlmostEqual(-28982.6, m.loglike(), delta=0.05)
		g = m.negative_d_loglike()
		self.assertAlmostEqual(   -226372, g[0], delta=.5 )
		self.assertAlmostEqual(   4087.18, g[2], delta=.005 )
		self.assertAlmostEqual(  0.567559, g[3], delta=.0000005 )
		self.assertAlmostEqual(  -410.803, g[4], delta=.0005 )
		self.assertAlmostEqual(   4.36171, g[5], delta=.000005 )
		self.assertAlmostEqual(  -104.307, g[6], delta=.0005 )
		self.assertAlmostEqual(    253697, g[7], delta=.5 )
		self.assertAlmostEqual(  -5558.95, g[8], delta=.005 )
		self.assertAlmostEqual(  -24395.3, g[9], delta=.05 )
		self.assertAlmostEqual(  -835.116, g[10], delta=.0005 )
		self.assertAlmostEqual(  -5833.57, g[11],delta=.005)
		m.tearDown()
		m.parameter("time").holdfast = False
		m.setUp()
		self.assertAlmostEqual(-28982.6, m.loglike(), delta=0.05)
		g = m.negative_d_loglike()
		self.assertAlmostEqual(   -226372, g[0], delta=.5 )
		self.assertAlmostEqual(11054.3015, g[1], delta=.005 )
		self.assertAlmostEqual(   4087.18, g[2], delta=.005 )
		self.assertAlmostEqual(  0.567559, g[3], delta=.0000005 )
		self.assertAlmostEqual(  -410.803, g[4], delta=.0005 )
		self.assertAlmostEqual(   4.36171, g[5], delta=.000005 )
		self.assertAlmostEqual(  -104.307, g[6], delta=.0005 )
		self.assertAlmostEqual(    253697, g[7], delta=.5 )
		self.assertAlmostEqual(  -5558.95, g[8], delta=.005 )
		self.assertAlmostEqual(  -24395.3, g[9], delta=.05 )
		self.assertAlmostEqual(  -835.116, g[10], delta=.0005 )
		self.assertAlmostEqual(  -5833.57, g[11],delta=.005)
		m.tearDown()
		m.parameter("time").holdfast = True
		m.estimate()


	def test_model2_mnl_with_alias_parameter(self):
		d = self._db
		m = Model (d)
		m.option.calc_std_errors = False
		m.parameter("cost",-0.01)
		m.parameter("time",-0.001)
		m.parameter("con2",0) 
		m.parameter("con3",0) 
		m.parameter("con4",0.1) 
		m.parameter("con5",0) 
		m.parameter("con6",0) 
		m.parameter("inc2",0.1) 
		m.parameter("inc3",0) 
		m.parameter("inc4",0) 
		m.parameter("inc5",0) 
		m.parameter("inc6",0) 
		m.utility.ca("tottime","time")
		m.utility.ca("totcost","cost")
		m.utility.co("HHINC","SR2","inc2") 
		m.utility.co("HHINC","SR3+","inc3") 
		m.utility.co("HHINC","Tran","inc4") 
		m.utility.co("HHINC","Bike","inc5") 
		m.utility.co("HHINC","Walk","inc6") 
		m.utility.co("1","SR2","con2") 
		m.utility.co("1","SR3+","con3") 
		m.utility.co("1","Tran","con4") 
		m.utility.co("1","Bike","con5") 
		m.utility.co("1","Walk","con6") 
		m.alias("cost", "time", 1.0)
		m.setUp()
		self.assertAlmostEqual(-27082.08468362813, m.loglike(), delta=0.05)
		g = m.negative_d_loglike()
		self.assertEqual(11, len(g))
		self.assertAlmostEqual(   -187015.11988655446, g[0 ], delta=.5 )
		self.assertAlmostEqual(   4130.665795197024,   g[1 ], delta=.5 )
		self.assertAlmostEqual(   -43.15184299217523,  g[2 ], delta=.005 )
		self.assertAlmostEqual(  -400.2936592090151,   g[3 ], delta=.0000005 )
		self.assertAlmostEqual(  -16.881507840220,     g[4 ], delta=.0005 )
		self.assertAlmostEqual(   -124.63236050685,    g[5 ], delta=.000005 )
		self.assertAlmostEqual(  255383.056358671,     g[6 ], delta=.0005 )
		self.assertAlmostEqual(    -6796.808724221511, g[7 ], delta=.5 )
		self.assertAlmostEqual(  -24443.198884079553,  g[8 ], delta=.005 )
		self.assertAlmostEqual(  -1553.1927312531757,  g[9 ], delta=.05 )
		self.assertAlmostEqual(  -6347.310737255839,   g[10], delta=.0005 )


	def test_model2_mnl_overspecified(self):
		d = self._db
		m1 = Model (d)
		m1.utility.ca("tottime","time")
		m1.utility.ca("totcost","cost")
		m1.utility.co("1","DA","con1")
		m1.utility.co("1","SR2","con2")
		m1.utility.co("1","SR3+","con3")
		m1.utility.co("1","Tran","con4")
		m1.utility.co("1","Bike","con5")
		m1.utility.co("1","Walk","con6")
		m1.setUp()
		r = m1.maximize_loglike()
		self.assertTrue('possible_overspecification' in r)
		self.assertTrue(r.possible_overspecification[0][1] == ['con1','con2','con3','con4','con5','con6'])
		del m1


class TestMNL(ELM_TestCase):

	def setUp(self):
		pass
		
#	def test_model2_mnl_simulate_probabilities(self):
#		d=DB.Example()
#		m=Model(d)
#		m.option.calc_std_errors = False
#		m.parameter("cost",-0.00491995) 
#		m.parameter("time",-0.0513384) 
#		m.parameter("con2", -2.17804) 
#		m.parameter("con3",-3.7251337) 
#		m.parameter("con4",-0.67097311159) 
#		m.parameter("con5",-2.37636) 
#		m.parameter("con6",-0.206814) 
#		m.parameter("inc2",-0.00216993) 
#		m.parameter("inc3",0.0003576561990) 
#		m.parameter("inc4",-0.00528647693) 
#		m.parameter("inc5",-0.012808051) 
#		m.parameter("inc6",-0.00968604) 
#		m.utility.ca("tottime","time")
#		m.utility.ca("totcost","cost")
#		m.utility.co("hhinc","SR2","inc2") 
#		m.utility.co("hhinc","SR3+","inc3")
#		m.utility.co("hhinc","Tran","inc4")
#		m.utility.co("hhinc","Bike","inc5") 
#		m.utility.co("hhinc","Walk","inc6") 
#		m.utility.co("1","SR2","con2") 
#		m.utility.co("1","SR3+","con3") 
#		m.utility.co("1","Tran","con4")
#		m.utility.co("1","Bike","con5") 
#		m.utility.co("1","Walk","con6") 
#		m.provision()
#		m.simulate_probability('simprob')
#		x=d.execute("select * from simprob order by caseid limit 2")
#		x0 = (1, 0.8174600870190675, 0.07770903163375573, 0.01790556295759778, 0.07142751074925907, 0.015497807640319872, 0.0)
#		for z1,z2 in zip(x0, next(x)): self.assertAlmostEqual(z1,z2,8)
#		x1 = (2, 0.3369635864515776, 0.07434203412431624, 0.05206877880356903, 0.49807830721999424, 0.03854729340054295,  0.0)
#		for z1,z2 in zip(x1, next(x)): self.assertAlmostEqual(z1,z2,8)
#		x=d.execute("select * from simprob order by caseid limit 9999999 offset 5027;")
#		x0 = (5028, 0.7106958426535986, 0.06287714163729226, 0.014048614223318392, 0.15747292709818003, 0.0, 0.05490547438761054)
#		for z1,z2 in zip(x0, next(x)): self.assertAlmostEqual(z1,z2,8)
#		x1= (5029, 0.5816188285670169, 0.04728081160774693, 0.01036514008391914, 0.12086299032327717, 0.031085619661209166, 0.2087866097568307)
#		for z1,z2 in zip(x1, next(x)): self.assertAlmostEqual(z1,z2,8)

	def test_swissmetro_01logit(self):
		swissmetro_alts = {
		1:('Train','TRAIN_AV*(SP!=0)'),
		2:('SM','SM_AV'),
		3:('Car','CAR_AV*(SP!=0)'),
		}
		d = DB.CSV_idco(filename=TEST_DATA['SWISSMETRO-CSV'],
				   choice="CHOICE", weight="_equal_",
				   tablename="data", savename=None, alts=swissmetro_alts, safety=True)
		d.queries.set_idco_query(d.queries.get_idco_query()+" WHERE CHOICE!=0 AND (PURPOSE==1 OR PURPOSE==3)")
		reweight_factor = 0.8890991
		#d.refresh()
		m = Model(d);
		m.logger(False)
		m.parameter("ASC_TRAIN",0)
		m.parameter("B_TIME"   ,0)
		m.parameter("B_COST"   ,0)
		m.parameter("ASC_CAR"  ,0)
		m.utility.co("1",1,"ASC_TRAIN") 
		m.utility.co("1",3,"ASC_CAR") 
		m.utility.co("TRAIN_TT",1,"B_TIME")
		m.utility.co("SM_TT",2,"B_TIME") 
		m.utility.co("CAR_TT",3,"B_TIME") 
		m.utility.co("TRAIN_CO*(GA==0)",1,"B_COST")
		m.utility.co("SM_CO*(GA==0)"   ,2,"B_COST") 
		m.utility.co("CAR_CO",        3,"B_COST") 
		m.option.gradient_diagnostic=0
		m.option.calc_std_errors=True
		m.setUp()
		m.estimate()
		self.assertAlmostEqual( -5331.252006978, m.loglike(cached=2), 6 )
		self.assertAlmostEqual( -0.7012,   m.parameter("ASC_TRAIN").value,4 )
		self.assertAlmostEqual( -0.012778, m.parameter("B_TIME").value   ,4 )
		self.assertAlmostEqual( -0.010838, m.parameter("B_COST").value   ,4 )
		self.assertAlmostEqual( -0.15463,  m.parameter("ASC_CAR").value  ,4 )
		self.assertAlmostEqual( 0.0549,    m.parameter("ASC_TRAIN").std_err,4 )
		self.assertAlmostEqual( 0.00056883,m.parameter("B_TIME").std_err   ,6 )
		self.assertAlmostEqual( 0.000518,    m.parameter("B_COST").std_err ,6 )
		self.assertAlmostEqual( 0.0432,    m.parameter("ASC_CAR").std_err  ,4 )
		self.assertAlmostEqual( 0.0826,    m.parameter("ASC_TRAIN").robust_std_err,4 )
		self.assertAlmostEqual( 0.0010425386137449327,  m.parameter("B_TIME").robust_std_err   ,6 )
		self.assertAlmostEqual( 0.000682,  m.parameter("B_COST").robust_std_err   ,6 )
		self.assertAlmostEqual( 0.0582,    m.parameter("ASC_CAR").robust_std_err  ,4 )


	def test_swissmetro_02weight_double(self):
		m = Model.Example(102, DB.Example('swissmetro'))
		m.db.queries.weight = '(1.0*(GROUPid==2)+1.2*(GROUPid==3))*0.8890991*2.0'
		m.option.calc_std_errors=False
		m.maximize_loglike()
		m.loglike()
		self.assertAlmostEqual( -10547.48518638403, m.loglike(), 3 )
		self.assertAlmostEqual( -10547.48518638403, m.loglike(cached=False), 3 )

	def test_automatic_family(self):
		d = DB.Example('swissmetro')
		f = ModelFamily()
		f.add(Model.Example(101, d))
		f.constants_only_model()
		self.assertNearlyEqual(-5864.998303233962, f['constants_only'].loglike())
		saved_model = f['constants_only'].save(None)
		m = Model.loads(saved_model)
		self.assertEqual(m.parameter_values(), f['constants_only'].parameter_values())

	def test_save_and_load(self):
		m = Model.Example(114)
		m['ASC_TRAIN'].value = -.123
		m['ASC_CAR'].value = .123
		m['B_TIME'].value = -.0045
		m['B_COST'].value = -.0067
		m['existing'].value = 0.5
		m['SB_TRAIN'].value = 0.1
		m.root_id = 999
		m.provision()
		m.covariance_matrix[1,1] = 9.9
		m2 = Model.loads(m.save(None))
		m2.db = m.db
		m2.provision()
		self.assertEqual(m.parameter_values(), m2.parameter_values())
		self.assertEqual(m.parameter_names(), m2.parameter_names())
		self.assertEqual(m.root_id, m2.root_id)
		self.assertTrue( numpy.allclose(numpy.asarray(m.covariance_matrix),numpy.asarray(m2.covariance_matrix), equal_nan=True) )
		self.assertNearlyEqual(m.loglike(), m2.loglike(), 12)

	def test_single_row_probability(self):
		m = Model.Example()
		needco = m.utility.co.needs()
		needca = m.utility.ca.needs()
		xa = m.db.array_idca(*needca)[0][0:1,:,:]
		#xa = m.db.ask_idca(needca,1)
		xo = m.db.array_idco(*needco)[0][0:1,:]
		#xo = m.db.ask_idco(needco,1)
		av = m.db.array_avail()[0][0:1,:,:]
		#av = m.db.ask_avail(1)
		self.assertEqual(1, xa.shape[0])
		self.assertEqual(6, xa.shape[1])
		self.assertEqual(2, xa.shape[2])
		self.assertEqual(1, xo.shape[0])
		self.assertEqual(2, xo.shape[1])
		m.freshen()
		m.parameter_values([-2.1780392286038217, -3.725133748807042, -0.6709731115935808,
							-2.3763431244580198, -0.20681363746347237, -0.002170001687300809,
							0.00035765619931134817, -0.005286476936260372, -0.012808051997631317,
							-0.009686263668346087, -0.05134043022343194, -0.004920362964462176])
		pr = numpy.array([[ 0.8174641 ,  0.07770958,  0.01790577,  0.0714228 ,  0.01549774, 0.        ]])
		m.freshen()
		self.assertArrayEqual( pr, m.calc_probability(m.calc_utility(xo,xa,av)) )
		av = m.db.array_avail_blind()[0][0:1,:,:]
		self.assertArrayEqual( pr, m.calc_probability(m.calc_utility(xo,xa,av)) )



	def test_qmnl_loglikelihoods_theta1(self):

		m1 = Model.Example()
		m1.parameter('quant_cost', value=1)
		m1.parameter('quant_time', value=2)
		m1.quantity('totcost+10', 'quant_cost')
		m1.quantity('tottime',    'quant_time')
		m1.provision()

		m2 = Model.Example()
		m2.parameter('theta', value=1)
		m2.utility.ca('log(((totcost+10)*exp(1))+(tottime*exp(2)))','theta')
		m2.provision()

		self.assertAlmostEqual(-7823.232043623507, m1.loglike(), 7)
		self.assertAlmostEqual(-7823.232043623507, m2.loglike(), 7)

		m3 = Model.Example()
		m3.parameter('quant_cost', value=2)
		m3.parameter('quant_time', value=1)
		m3.quantity('totcost+10', 'quant_cost')
		m3.quantity('tottime',    'quant_time')
		m3.provision()

		m4 = Model.Example()
		m4.parameter('theta', value=1)
		m4.utility.ca('log(((totcost+10)*exp(2))+(tottime*exp(1)))','theta')
		m4.provision()

		self.assertAlmostEqual(-6974.993686737006, m3.loglike(), 7)
		self.assertAlmostEqual(-6974.993686737006, m4.loglike(), 7)
		
		correct_g = (468.2545410105376, 526.8256449198816, 1026.0599226967315, 76.18605295647285, 6.164706155686871, 28238.215664896412, 30259.2409306726, 60449.75569443522, 5132.286505795491, 2090.2918481169763, 52188.194044174445, 46679.47009934323, -209.09066088279357, 209.09066097374304)
		check_g = m3.negative_d_loglike()
		
		
		
		for correct_gi,check_gi in zip(correct_g,check_g):
			self.assertNearlyEqual(correct_gi,check_gi, 5)

		m3.parameter('ASC_TRAN', value = 1.0)
		m4.parameter('ASC_TRAN', value = 1.0)
		self.assertAlmostEqual(-8458.368386815826, m3.loglike(), 7)
		self.assertAlmostEqual(-8458.368386815826, m4.loglike(), 7)

		correct_g = (214.33071392819235, 342.49202810627287, 1939.683594154003, 31.456485424894787, -59.88434201479634, 13741.611224029959, 19766.022834585823, 113507.44438243657, 2430.4592427702332, -1517.4853508812657, 77160.21217390696, 74488.37299910728, -171.13372819264637, 171.13372819264632)
		check_g = m3.negative_d_loglike()
		for correct_gi,check_gi in zip(correct_g,check_g):
			self.assertNearlyEqual(correct_gi,check_gi, 5)

		m3.parameter('tottime', value = -0.1)
		m4.parameter('tottime', value = -0.1)
		self.assertAlmostEqual(-5511.9881882709515, m3.loglike(), 7)
		self.assertAlmostEqual(-5511.9881882709515, m4.loglike(), 7)

		correct_g = (522.5184853830001, 463.6990251864535, 247.38042830373828, 24.114642455255556, -147.3307132317191, 30984.541140176854, 26322.255483618563, 14156.810114419472, 2004.3211709394827, -6283.239683176233, 3973.4674946145788, 63361.91323563814, -17.98436648149563, 17.98436648149564)
		check_g = m3.negative_d_loglike()
		for correct_gi,check_gi in zip(correct_g,check_g):
			self.assertNearlyEqual(correct_gi,check_gi, 5)

		m3.tearDown()
		m3.parameter('nonmotor', null_value=1.0, value=0.5)
		m3.nest(9,'nonmotor')
		m3.link(9,6)
		m3.link(9,5)
		m3.setUp()
		self.assertAlmostEqual(-5618.048268386866, m3.loglike(), 7)

		correct_g = (523.7844907321696, 464.5988173284788, 250.592567002237, 81.94457436295288, -212.93147800862212, 31060.03073633408, 26376.030177959692, 14345.888386315535, 5056.735849911448, -9803.002436338618, 2615.0622375662056, 63693.22471443263, -3.5640720507976535, 3.5640720507976895, -426.7707472245791)
		check_g = m3.negative_d_loglike()
		for correct_gi,check_gi in zip(correct_g,check_g):
			self.assertNearlyEqual(correct_gi,check_gi, 5)


	def test_bhhh(self):
		m1 = Model.Example()
		m1.provision()
		with self.assertRaises(LarchCacheError):
			m1.bhhh_cached()
		bhhh = m1.bhhh()
		correct_bhhh = numpy.asarray([
		[  5.63081667e+02,   1.33448333e+02,  -1.22030556e+01,
          2.41933333e+01,  -7.89083333e+00,   3.24177229e+04,
          7.98426458e+03,  -9.21576389e+01,   1.70104847e+03,
         -1.42165278e+01,   5.80371684e+03,  -4.49907787e+04],
       [  1.33448333e+02,   3.81815000e+02,   4.87469444e+01,
          4.79266667e+01,   1.34925000e+01,   7.98426458e+03,
          2.24538062e+04,   3.19150903e+03,   2.99467347e+03,
          1.09128347e+03,   8.48690017e+03,  -5.22980411e+04],
       [ -1.22030556e+01,   4.87469444e+01,   4.52296944e+02,
          2.48683333e+01,  -9.92833333e+00,  -9.21576389e+01,
          3.19150903e+03,   2.52170757e+04,   1.75526514e+03,
          4.17251389e+01,   1.27795749e+04,  -3.84332102e+04],
       [  2.41933333e+01,   4.79266667e+01,   2.48683333e+01,
          9.51933333e+01,   5.06666667e+00,   1.70104847e+03,
          2.99467347e+03,   1.75526514e+03,   5.33984014e+03,
          4.70413889e+02,   3.46529816e+03,  -6.65734413e+03],
       [ -7.89083333e+00,   1.34925000e+01,  -9.92833333e+00,
          5.06666667e+00,   1.52042500e+02,  -1.42165278e+01,
          1.09128347e+03,   4.17251389e+01,   4.70413889e+02,
          7.24522514e+03,   4.58237876e+03,  -6.72174352e+03],
       [  3.24177229e+04,   7.98426458e+03,  -9.21576389e+01,
          1.70104847e+03,  -1.42165278e+01,   2.45688771e+06,
          6.04142441e+05,   1.49097691e+04,   1.42915626e+05,
          1.44434785e+04,   3.61407837e+05,  -2.81463189e+06],
       [  7.98426458e+03,   2.24538062e+04,   3.19150903e+03,
          2.99467347e+03,   1.09128347e+03,   6.04142441e+05,
          1.73983117e+06,   2.42776477e+05,   2.33265730e+05,
          9.62607702e+04,   5.10380039e+05,  -3.31693869e+06],
       [ -9.21576389e+01,   3.19150903e+03,   2.52170757e+04,
          1.75526514e+03,   4.17251389e+01,   1.49097691e+04,
          2.42776477e+05,   1.87644755e+06,   1.48191524e+05,
          3.09099577e+04,   7.59242223e+05,  -2.63029100e+06],
       [  1.70104847e+03,   2.99467347e+03,   1.75526514e+03,
          5.33984014e+03,   4.70413889e+02,   1.42915626e+05,
          2.33265730e+05,   1.48191524e+05,   4.04281563e+05,
          4.37127292e+04,   2.16170363e+05,  -3.87075339e+05],
       [ -1.42165278e+01,   1.09128347e+03,   4.17251389e+01,
          4.70413889e+02,   7.24522514e+03,   1.44434785e+04,
          9.62607702e+04,   3.09099577e+04,   4.37127292e+04,
          5.01334908e+05,   2.43447066e+05,  -2.76571448e+05],
       [  5.80371684e+03,   8.48690017e+03,   1.27795749e+04,
          3.46529816e+03,   4.58237876e+03,   3.61407837e+05,
          5.10380039e+05,   7.59242223e+05,   2.16170363e+05,
          2.43447066e+05,   9.23597714e+05,  -2.04018963e+06],
       [ -4.49907787e+04,  -5.22980411e+04,  -3.84332102e+04,
         -6.65734413e+03,  -6.72174352e+03,  -2.81463189e+06,
         -3.31693869e+06,  -2.63029100e+06,  -3.87075339e+05,
         -2.76571448e+05,  -2.04018963e+06,   4.64208243e+07]
		 ])
		self.assertTrue( numpy.allclose(correct_bhhh, numpy.asarray(bhhh)) )
		m1.bhhh_cached()
		m1.d_loglike_cached()

	def test_gradient_check(self):
		m1 = Model.Example()
		gc = m1.gradient_check(disp=False)
		self.assertTrue( gc[0] < -6 )

	def test_gradient_holdfast_switching(self):
		m = Model.Example()
		m.parameter('ASC_SR2').holdfast = True
		m.parameter('ASC_SR3P').holdfast = True
		m.setUp()
		m.parameter_values((0.0, 0.0, 0.5305777095119981, -1.3804808428294322, 1.1023754107604156, -0.03289104850834897, -0.058267030048137054, -0.015279104463811553, -0.02184376051734762, -0.019312533988295785, -0.06930477008694798, -0.003905346765501799))
		g = m.d_loglike()
		self.assertEqual(0, g[0])
		self.assertEqual(0, g[1])
		self.assertTrue(0 != g[2])
		m.parameter('ASC_SR2').holdfast = False
		m.parameter('ASC_SR3P').holdfast = False
		g = m.d_loglike_nocache()
		self.assertTrue(0 != g[0])
		self.assertTrue(0 != g[1])

