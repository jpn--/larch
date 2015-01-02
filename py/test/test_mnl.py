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


import os
import unittest
import numpy

if __name__ == "__main__" and __package__ is None:
    __package__ = "larch.test.test_mnl"

from ..test import TEST_DATA, ELM_TestCase, DEEP_TEST
from ..core import Parameter, Model, DB, LarchError, SQLiteError
from ..model import ModelFamily

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
		m.option.calculate_std_err = 0
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
		del m.utility.co[-1]
		del m["HHINC@AltNameWhichDoesNotExist"]
		with self.assertRaises(Exception):
			m.utility.co("HHINC",666)
		del m["HHINC#666"]
		del m.utility.co[-1]
		with self.assertRaises(Exception):
			m.utility.ca("CA_ColumnNameWhichDoesNotExist")
		del m.utility.ca[-1]
		del m["CA_ColumnNameWhichDoesNotExist"]
		with self.assertRaises(Exception):
			m.utility.co("CO_ColumnNameWhichDoesNotExist","Tran")
		del m["CO_ColumnNameWhichDoesNotExist@Tran"]
		m.utility.clean(d)
		m.setUp()
		self.assertEqual((-0.01,0,0,0,0.1,0,0,0.1,0,0,0,0),m.parameter_values())
		self.assertAlmostEqual(-28993.7, m.loglike(), delta=0.05)
		g = m.d_loglike()
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
		self.assertAlmostEqual( -3626.1862548451313, m.LL())

	def test_model2_mnl_with_constant_parameter(self):
		d = self._db
		m = Model (d)
		m.option.calculate_std_err = 0
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
		g = m.d_loglike()
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
		m.option.calculate_std_err = 0
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
		g = m.d_loglike()
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
		g = m.d_loglike()
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

#	def test_save_and_load_model(self):
#		d = self._db
#		m = Model (d)
#		m.option.calculate_std_err = 0
#		m.parameter("cost",-0.01)
#		m.parameter("time",-0.001, holdfast=1)
#		m.parameter("con2",0) 
#		m.parameter("con3",0) 
#		m.parameter("con4",0.1) 
#		m.parameter("con5",0) 
#		m.parameter("con6",0) 
#		m.parameter("inc2",0.1) 
#		m.parameter("inc3",0) 
#		m.parameter("inc4",0) 
#		m.parameter("inc5",0) 
#		m.parameter("inc6",0) 
#		m.utility.ca("tottime","time")
#		m.utility.ca("totcost","cost")
#		m.utility.co("HHINC","SR2","inc2") 
#		m.utility.co("HHINC","SR3+","inc3") 
#		m.utility.co("HHINC","Tran","inc4") 
#		m.utility.co("HHINC","Bike","inc5") 
#		m.utility.co("HHINC","Walk","inc6") 
#		m.utility.co("1","SR2","con2") 
#		m.utility.co("1","SR3+","con3") 
#		m.utility.co("1","Tran","con4") 
#		m.utility.co("1","Bike","con5") 
#		m.utility.co("1","Walk","con6") 
#		import tempfile
#		t = tempfile.TemporaryFile()
#		m.Save(t)
#		t.seek(0)
#		m2 = Model.Load(d,t)


class TestMNL(ELM_TestCase):

	def setUp(self):
		pass
		
	def test_model2_mnl_simulate_probabilities(self):
		d=DB.Example()
		m=Model(d)
		m.option.calculate_std_err = 0
		m.parameter("cost",-0.00491995) 
		m.parameter("time",-0.0513384) 
		m.parameter("con2", -2.17804) 
		m.parameter("con3",-3.7251337) 
		m.parameter("con4",-0.67097311159) 
		m.parameter("con5",-2.37636) 
		m.parameter("con6",-0.206814) 
		m.parameter("inc2",-0.00216993) 
		m.parameter("inc3",0.0003576561990) 
		m.parameter("inc4",-0.00528647693) 
		m.parameter("inc5",-0.012808051) 
		m.parameter("inc6",-0.00968604) 
		m.utility.ca("tottime","time")
		m.utility.ca("totcost","cost")
		m.utility.co("hhinc","SR2","inc2") 
		m.utility.co("hhinc","SR3+","inc3")
		m.utility.co("hhinc","Tran","inc4")
		m.utility.co("hhinc","Bike","inc5") 
		m.utility.co("hhinc","Walk","inc6") 
		m.utility.co("1","SR2","con2") 
		m.utility.co("1","SR3+","con3") 
		m.utility.co("1","Tran","con4")
		m.utility.co("1","Bike","con5") 
		m.utility.co("1","Walk","con6") 
		m.provision()
		m.simulate_probability('simprob')
		x=d.execute("select * from simprob order by caseid limit 2")
		x0 = (1, 0.8174600870190675, 0.07770903163375573, 0.01790556295759778, 0.07142751074925907, 0.015497807640319872, 0.0)
		for z1,z2 in zip(x0, next(x)): self.assertAlmostEqual(z1,z2,8)
		x1 = (2, 0.3369635864515776, 0.07434203412431624, 0.05206877880356903, 0.49807830721999424, 0.03854729340054295,  0.0)
		for z1,z2 in zip(x1, next(x)): self.assertAlmostEqual(z1,z2,8)
		x=d.execute("select * from simprob order by caseid limit 9999999 offset 5027;")
		x0 = (5028, 0.7106958426535986, 0.06287714163729226, 0.014048614223318392, 0.15747292709818003, 0.0, 0.05490547438761054)
		for z1,z2 in zip(x0, next(x)): self.assertAlmostEqual(z1,z2,8)
		x1= (5029, 0.5816188285670169, 0.04728081160774693, 0.01036514008391914, 0.12086299032327717, 0.031085619661209166, 0.2087866097568307)
		for z1,z2 in zip(x1, next(x)): self.assertAlmostEqual(z1,z2,8)

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
		m.option.calculate_std_err=1
		m.setUp()
		m.estimate()
		self.assertAlmostEqual( -5331.252006978, m.LL(), 6 )
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


	def test_swissmetro_02weight(self):
		swissmetro_alts = {
		1:('Train','TRAIN_AV*(SP!=0)'),
		2:('SM','SM_AV'),
		3:('Car','CAR_AV*(SP!=0)'),
		}
		d = DB.CSV_idco(filename=TEST_DATA['SWISSMETRO-CSV'],
				   choice="CHOICE", weight="(1.0*(GROUPid==2)+1.2*(GROUPid==3))*0.8890991",
				   tablename="data", savename=None, alts=swissmetro_alts, safety=True)
		d.queries.set_idco_query(d.queries.get_idco_query()+" WHERE CHOICE!=0 AND (PURPOSE==1 OR PURPOSE==3)")
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
		m.option.calculate_std_err=1
		m.estimate()
		self.assertAlmostEqual( -5273.743, m.LL(), 3 )
		self.assertAlmostEqual( -0.7565,   m.parameter("ASC_TRAIN").value,4 )
		self.assertAlmostEqual( -0.0132,  m.parameter("B_TIME").value   ,4 )
		self.assertAlmostEqual( -0.0112,  m.parameter("B_COST").value   ,4 )
		self.assertAlmostEqual( -0.114,   m.parameter("ASC_CAR").value  ,3 )
		self.assertAlmostEqual( 0.0560,   m.parameter("ASC_TRAIN").std_err,4 )
		self.assertAlmostEqual( 0.000569, m.parameter("B_TIME").std_err   ,6 )
		self.assertAlmostEqual( 0.000520, m.parameter("B_COST").std_err   ,6 )
		self.assertAlmostEqual( 0.0432,   m.parameter("ASC_CAR").std_err  ,4 )
		self.assertNearlyEqual( 0.08325742000250282,m.parameter("ASC_TRAIN").robust_std_err )
		self.assertNearlyEqual( 0.0010365818243247937,  m.parameter("B_TIME").robust_std_err    )
		self.assertNearlyEqual( 0.0006767859433091413,m.parameter("B_COST").robust_std_err    )
		self.assertNearlyEqual( 0.05831177412562568,   m.parameter("ASC_CAR").robust_std_err   )


	def test_automatic_family(self):
		d = DB.Example('swissmetro')
		f = ModelFamily()
		f.add(Model.Example(101, d))
		f.constants_only_model()
		self.assertNearlyEqual(-5864.998303233962, f['constants_only'].loglike())
		m = Model.loads(f['constants_only'].save(None))
		self.assertEqual(m.parameter_values(), f['constants_only'].parameter_values())

	def test_single_row_probability(self):
		m = Model.Example()
		needco = m.utility.co.needs()
		needca = m.utility.ca.needs()
		xa = m.db.array_idca(needca)[0][0:1,:,:]
		#xa = m.db.ask_idca(needca,1)
		xo = m.db.array_idco(needco)[0][0:1,:]
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
