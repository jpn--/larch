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
    __package__ = "larch.test.test_nl"

from ..test import TEST_DATA, ELM_TestCase, DEEP_TEST
from ..core import Parameter, Model, DB, LarchError, SQLiteError


class TestNL(ELM_TestCase):


	def test_nl2_single_cycle(self):		
		d = DB.Example('MTC');
		m = Model(d);
		m.parameter ("cost",0)
		m.parameter("time",0)
		m.parameter("con2",0)
		m.parameter("con3",0)
		m.parameter("con4",0)
		m.parameter("con5",0)
		m.parameter("con6",0)
		m.parameter("inc2",0)
		m.parameter("inc3",0)
		m.parameter("inc4",0)
		m.parameter("inc5",0)
		m.parameter("inc6",0)
		m.parameter("autoNest", 1.0, 1.0)
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
		m.nest("auto", 8, "autoNest")
		m.link(8, 1)
		m.link(8, 2)
		m.link(8, 3)
		m.link(8, 4)
		m.option.gradient_diagnostic = 2
		m.provision()
		m.setUp()
		self.assertAlmostEqual( -7309.600971749863, m.loglike(), delta=0.00000001 )	
		g = m.d_loglike()
		self.assertAlmostEqual( -127397.53666666638, g[ 0], delta=0.000001 )
		self.assertAlmostEqual( 42104.2, g[ 1], delta=0.01 )
		self.assertAlmostEqual( 687.7  , g[ 2], delta=0.01 )
		self.assertAlmostEqual( 1043.7 , g[ 3], delta=0.01 )
		self.assertAlmostEqual( 380.78333333332864, g[ 4], delta=0.00000001 )
		self.assertAlmostEqual( 279.8  , g[ 5], delta=0.01 )
		self.assertAlmostEqual( 113.65 , g[ 6], delta=0.001 )
		self.assertAlmostEqual( 41179.541666666715, g[ 7], delta=0.0000001 ) 
		self.assertAlmostEqual( 60691.541666666686, g[ 8], delta=0.0000001 )
		self.assertAlmostEqual( 24028.374999999985, g[ 9], delta=0.0000001 )
		self.assertAlmostEqual( 17374.8, g[10], delta=0.01 )
		self.assertAlmostEqual( 7739.108333333339, g[11], delta=0.0000001 )
		self.assertAlmostEqual(-537.598179908304, g[12], delta=0.00000001 )
		v =(-0.0053047 ,
			-0.0746418 ,
			-2.07691   ,
			-9.2709    ,
			-0.185096  ,
			-1.65575   ,
			1.86265    ,
			-0.00427335,
			0.0126937  ,
			-0.00710494,
			-0.0160185 ,
			-0.00716963,
			2.32074    ,
			)
		self.assertAlmostEqual( -4213.0122967116695, m.loglike(v), delta=0.00000001 )
		m.freshen()
		g = m.d_loglike()
		self.assertAlmostEqual( -5213.854558101514, g[ 0], delta=0.0000001 )
		self.assertAlmostEqual(  12836.151998017447, g[ 1], delta=0.0000001 )
		self.assertAlmostEqual(  629.8310448361855, g[ 2], delta=0.001 )
		self.assertAlmostEqual(  -74.76717809020077, g[ 3], delta=0.001 )
		self.assertAlmostEqual(  263.166080293258, g[ 4], delta=0.0001 )
		self.assertAlmostEqual( -24.52301491338605, g[ 5], delta=0.00000001 )
		self.assertAlmostEqual(  84.57760963703537, g[ 6], delta=0.00000001 )
		self.assertAlmostEqual(  36931.80015794064, g[ 7], delta=0.00000001 )
		self.assertAlmostEqual( -3989.2630876959906, g[ 8], delta=0.00000001 )
		self.assertAlmostEqual(  15728.304436116596, g[ 9], delta=0.01 )
		self.assertAlmostEqual( -1252.58, g[10], delta=0.01 )
		self.assertAlmostEqual(  4531.231730902182, g[11], delta=0.00000001 )
		self.assertAlmostEqual(  1741.9031614953105, g[12], delta=0.00000001 )
		m.tearDown()

	def test_nl2_single_cycle_multithread(self):		
		d = DB.Example('MTC');
		m = Model(d);
		import platform
		if platform.system() == "Darwin":
			m.option.threads = 4
		m.parameter ("cost",0)
		m.parameter("time",0)
		m.parameter("con2",0)
		m.parameter("con3",0)
		m.parameter("con4",0)
		m.parameter("con5",0)
		m.parameter("con6",0)
		m.parameter("inc2",0)
		m.parameter("inc3",0)
		m.parameter("inc4",0)
		m.parameter("inc5",0)
		m.parameter("inc6",0)
		m.parameter("autoNest", 1.0, 1.0)
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
		m.nest("auto", 8, "autoNest")
		m.link(8, 1)
		m.link(8, 2)
		m.link(8, 3)
		m.link(8, 4)
		m.option.gradient_diagnostic = 2
		m.setUp()
		self.assertAlmostEqual( -7309.600971749863, m.loglike(), delta=0.00000001 )	
		g = m.d_loglike()
		self.assertAlmostEqual( -127397.53666666638, g[ 0], delta=0.000001 )
		self.assertAlmostEqual( 42104.2, g[ 1], delta=0.01 )
		self.assertAlmostEqual( 687.7  , g[ 2], delta=0.01 )
		self.assertAlmostEqual( 1043.7 , g[ 3], delta=0.01 )
		self.assertAlmostEqual( 380.78333333332864, g[ 4], delta=0.00000001 )
		self.assertAlmostEqual( 279.8  , g[ 5], delta=0.01 )
		self.assertAlmostEqual( 113.65 , g[ 6], delta=0.001 )
		self.assertAlmostEqual( 41179.541666666715, g[ 7], delta=0.0000001 ) 
		self.assertAlmostEqual( 60691.541666666686, g[ 8], delta=0.0000001 )
		self.assertAlmostEqual( 24028.374999999985, g[ 9], delta=0.0000001 )
		self.assertAlmostEqual( 17374.8, g[10], delta=0.01 )
		self.assertAlmostEqual( 7739.108333333339, g[11], delta=0.0000001 )
		self.assertAlmostEqual(-537.598179908304, g[12], delta=0.00000001 )
		v =(-0.0053047 ,
			-0.0746418 ,
			-2.07691   ,
			-9.2709    ,
			-0.185096  ,
			-1.65575   ,
			1.86265    ,
			-0.00427335,
			0.0126937  ,
			-0.00710494,
			-0.0160185 ,
			-0.00716963,
			2.32074    ,
			)
		self.assertAlmostEqual( -4213.0122967116695, m.loglike(v), delta=0.00000001 )
		m.freshen()
		g = m.d_loglike()
		self.assertAlmostEqual( -5213.854558101514, g[ 0], delta=0.0000001 )
		self.assertAlmostEqual(  12836.151998017447, g[ 1], delta=0.0000001 )
		self.assertAlmostEqual(  629.8310448361855, g[ 2], delta=0.001 )
		self.assertAlmostEqual(  -74.76717809020077, g[ 3], delta=0.001 )
		self.assertAlmostEqual(  263.166080293258, g[ 4], delta=0.0001 )
		self.assertAlmostEqual( -24.52301491338605, g[ 5], delta=0.00000001 )
		self.assertAlmostEqual(  84.57760963703537, g[ 6], delta=0.00000001 )
		self.assertAlmostEqual(  36931.80015794064, g[ 7], delta=0.00000001 )
		self.assertAlmostEqual( -3989.2630876959906, g[ 8], delta=0.00000001 )
		self.assertAlmostEqual(  15728.304436116596, g[ 9], delta=0.01 )
		self.assertAlmostEqual( -1252.58, g[10], delta=0.01 )
		self.assertAlmostEqual(  4531.231730902182, g[11], delta=0.00000001 )
		self.assertAlmostEqual(  1741.9031614953105, g[12], delta=0.00000001 )
		m.tearDown()

	def test_swissmetro_09nested(self):
		d = DB.Example('swissmetro')
		m = Model(d)
		m.logger(False)
		m.parameter("ASC_TRAIN",0)
		m.parameter("B_TIME"   ,0)
		m.parameter("B_COST"   ,0)
		m.parameter("ASC_CAR"  ,0)
		m.parameter("existing" ,1,1)
		m.utility.co("1",1,"ASC_TRAIN") 
		m.utility.co("1",3,"ASC_CAR") 
		m.utility.co("TRAIN_TT",1,"B_TIME")
		m.utility.co("SM_TT",2,"B_TIME") 
		m.utility.co("CAR_TT",3,"B_TIME") 
		m.utility.co("TRAIN_CO*(GA==0)",1,"B_COST")
		m.utility.co("SM_CO*(GA==0)"   ,2,"B_COST") 
		m.utility.co("CAR_CO",        3,"B_COST") 
		m.option.gradient_diagnostic = 0
		m.option.calculate_std_err = 1
		m.nest("existing", 4, "existing") 
		m.link(4, 1)
		m.link(4, 3)
		m.estimate()
		self.assertAlmostEqual( -5236.90001, m.LL(), 4 )
		self.assertAlmostEqual( -0.51194,   m.parameter("ASC_TRAIN").value,4 )
		self.assertAlmostEqual( -0.0089868, m.parameter("B_TIME").value   ,4 )
		self.assertAlmostEqual( -0.0085669, m.parameter("B_COST").value   ,4 )
		self.assertAlmostEqual( -0.16714,   m.parameter("ASC_CAR").value  ,4 )
		self.assertAlmostEqual(  0.48685,   m.parameter("existing").value ,4 )
		self.assertAlmostEqual(  0.0451798,   m.parameter("ASC_TRAIN").std_err,6 )
		self.assertAlmostEqual(  0.000569908, m.parameter("B_TIME").std_err   ,6 )
		self.assertAlmostEqual(  0.000462736, m.parameter("B_COST").std_err   ,6 )
		self.assertAlmostEqual(  0.0371364,   m.parameter("ASC_CAR").std_err  ,6 )
		self.assertAlmostEqual(  0.0278974,   m.parameter("existing").std_err ,6 )
		self.assertAlmostEqual(  0.0791147,   m.parameter("ASC_TRAIN").robust_std_err,5 )
		self.assertAlmostEqual(  0.00107113,  m.parameter("B_TIME").robust_std_err   ,5 )
		self.assertAlmostEqual(  0.000600363, m.parameter("B_COST").robust_std_err   ,5 )
		self.assertAlmostEqual(  0.0545294,   m.parameter("ASC_CAR").robust_std_err  ,5 )
		self.assertAlmostEqual(  0.038919,    m.parameter("existing").robust_std_err ,5 )

	def test_swissmetro_09nested_weighted(self):
		d = DB.Example('swissmetro')
		d.load_queries('weighted')
		m = Model(d)
		m.logger(False)
		m.parameter("ASC_TRAIN",0)
		m.parameter("B_TIME"   ,0)
		m.parameter("B_COST"   ,0)
		m.parameter("ASC_CAR"  ,0)
		m.parameter("existing" ,1,1)
		m.utility.co("1",1,"ASC_TRAIN") 
		m.utility.co("1",3,"ASC_CAR") 
		m.utility.co("TRAIN_TT",1,"B_TIME")
		m.utility.co("SM_TT",2,"B_TIME") 
		m.utility.co("CAR_TT",3,"B_TIME") 
		m.utility.co("TRAIN_CO*(GA==0)",1,"B_COST")
		m.utility.co("SM_CO*(GA==0)"   ,2,"B_COST") 
		m.utility.co("CAR_CO",        3,"B_COST") 
		m.option.gradient_diagnostic = 0
		m.option.calculate_std_err = 1
		m.nest("existing", 4, "existing") 
		m.link(4, 1)
		m.link(4, 3)
		m.estimate()
		self.assertAlmostEqual( -5181.79616, m.LL(), 4 )
		self.assertAlmostEqual( -0.5323,   m.parameter("ASC_TRAIN").value,4 )
		self.assertAlmostEqual( -0.0095934, m.parameter("B_TIME").value   ,4 )
		self.assertAlmostEqual( -0.0089907, m.parameter("B_COST").value   ,4 )
		self.assertAlmostEqual( -0.13352,   m.parameter("ASC_CAR").value  ,4 )
		self.assertAlmostEqual(  0.49833,   m.parameter("existing").value ,4 )
		self.assertAlmostEqual(  0.046564596603103144,   m.parameter("ASC_TRAIN").std_err,7 )
		self.assertAlmostEqual(  0.00057001, m.parameter("B_TIME").std_err   ,5 )
		self.assertAlmostEqual(  0.00046797, m.parameter("B_COST").std_err   ,5 )
		self.assertAlmostEqual(  0.037606,   m.parameter("ASC_CAR").std_err  ,5 )
		self.assertAlmostEqual(  0.027673,   m.parameter("existing").std_err ,5 )

	def test_swissmetro_14selectionBias(self):
		d = DB.Example('swissmetro')
		m = Model(d);
		m.logger(False)
		m.parameter("ASC_TRAIN",0)
		m.parameter("B_TIME"   ,0)
		m.parameter("B_COST"   ,0)
		m.parameter("ASC_CAR"  ,0)
		m.parameter("existing" ,1,1)
		m.parameter("SB_TRAIN" ,0)
		m.utility.co("1",1,"ASC_TRAIN") 
		m.utility.co("1",3,"ASC_CAR") 
		m.utility.co("TRAIN_TT",1,"B_TIME")
		m.utility.co("SM_TT",2,"B_TIME") 
		m.utility.co("CAR_TT",3,"B_TIME") 
		m.utility.co("TRAIN_CO*(GA==0)",1,"B_COST")
		m.utility.co("SM_CO*(GA==0)"   ,2,"B_COST") 
		m.utility.co("CAR_CO",        3,"B_COST") 
		m.option.gradient_diagnostic = 5
		m.option.calculate_std_err = 1
		m.nest("existing", 4, "existing") 
		m.link(4, 1)
		m.link(4, 3)
		m.samplingbias.co("1",1,"SB_TRAIN")
		m.estimate()
		self.assertNearlyEqual( -5169.641515645088, m.LL(), 5 )
		self.assertNearlyEqual( -10.25184105080805,   m.parameter("ASC_TRAIN").value, 2 )
		self.assertNearlyEqual( -0.0112766758, m.parameter("B_TIME").value ,3   )
		self.assertNearlyEqual( -0.011067, m.parameter("B_COST").value  ,3  )
		self.assertNearlyEqual( -0.3171095912683884,   m.parameter("ASC_CAR").value  ,3 )
		self.assertNearlyEqual(  0.8737,   m.parameter("existing").value ,3 )
		self.assertNearlyEqual(  10.411834188552463,   m.parameter("SB_TRAIN").value ,3 )
		self.assertNearlyEqual(  3.13,   m.parameter("ASC_TRAIN").std_err,3 )
		self.assertNearlyEqual(  0.0005758, m.parameter("B_TIME").std_err  ,3  )
		self.assertNearlyEqual(  0.00051123, m.parameter("B_COST").std_err   ,3 )
		self.assertNearlyEqual(  0.0443,   m.parameter("ASC_CAR").std_err   ,3)
		self.assertNearlyEqual(  0.035,   m.parameter("existing").std_err  ,2)
		self.assertNearlyEqual(  3.1463539084087366,   m.parameter("SB_TRAIN").std_err ,2 )
		self.assertNearlyEqual(  3.15,   m.parameter("ASC_TRAIN").robust_std_err,2 )
		self.assertNearlyEqual(  0.0010336527461594616,  m.parameter("B_TIME").robust_std_err   ,2 )
		self.assertNearlyEqual(  0.000672, m.parameter("B_COST").robust_std_err   ,2 )
		self.assertNearlyEqual(  0.0597,   m.parameter("ASC_CAR").robust_std_err  ,2 )
		self.assertNearlyEqual(  3.16,    m.parameter("SB_TRAIN").robust_std_err ,2 )


	def test_swissmetro_14selectionBias_convenience(self):
		d = DB.Example('swissmetro')
		m = Model(d);
		m.logger(False)
		m.parameter("ASC_TRAIN",-10)
		m.parameter("B_TIME"   ,0)
		m.parameter("B_COST"   ,0)
		m.parameter("ASC_CAR"  ,0)
		m.parameter("existing" ,1,1)
		m.parameter("SB_TRAIN" ,10)
		m.utility.co("1",1,"ASC_TRAIN") 
		m.utility.co("1",3,"ASC_CAR") 
		m.utility.co("TRAIN_TT",1,"B_TIME")
		m.utility.co("SM_TT",2,"B_TIME") 
		m.utility.co("CAR_TT",3,"B_TIME") 
		m.utility.co("TRAIN_CO*(GA==0)",1,"B_COST")
		m.utility.co("SM_CO*(GA==0)"   ,2,"B_COST") 
		m.utility.co("CAR_CO",        3,"B_COST") 
		m.option.gradient_diagnostic = 0
		m.option.calculate_std_err = 1
		m.nest("existing", 4, "existing") 
		m.link(4, 1)
		m.link(4, 3)
		m.samplingbias(1,"SB_TRAIN")
		m.samplingbias(2)
		self.assertEqual("1", m.samplingbias.co[0].data)
		self.assertEqual("SB_TRAIN", m.samplingbias.co[0].param)
		self.assertEqual("1", m.samplingbias.co[1].data)
		self.assertEqual("samplingbias#2", m.samplingbias.co[1].param)
		self.assertEqual(2, len(m.samplingbias.co))







	