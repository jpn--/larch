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
    __package__ = "larch.test.test_nl"

from ..test import TEST_DATA, ELM_TestCase, DEEP_TEST
from ..core import Parameter, Model, DB, LarchError, SQLiteError


class TestNL(ELM_TestCase):


	def test_nl2_single_cycle(self):		
		d = DB.Example('MTC');
		m = Model(d)
		m.option.threads = 1
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
		g = m.negative_d_loglike()
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
		g = m.negative_d_loglike(v)
		self.assertAlmostEqual( -4140.9722837573108, g[ 0], delta=0.00001 )
		self.assertAlmostEqual(  6437.577630163296, g[ 1], delta=0.0000001 )
		self.assertAlmostEqual(  263.62687005016875, g[ 2], delta=0.001 )
		self.assertAlmostEqual(  -32.73528233531812, g[ 3], delta=0.001 )
		self.assertAlmostEqual(  105.44094790643184, g[ 4], delta=0.0001 )
		self.assertAlmostEqual( -24.52301491338605, g[ 5], delta=0.000001 )
		self.assertAlmostEqual(  84.57760963703537, g[ 6], delta=0.000001 )
		self.assertAlmostEqual(  15532.154389060192, g[ 7], delta=0.0001 )
		self.assertAlmostEqual( -1748.1507300171445, g[ 8], delta=0.00001 )
		self.assertAlmostEqual(  6338.605865710019, g[ 9], delta=0.01 )
		self.assertAlmostEqual( -1252.58, g[10], delta=0.01 )
		self.assertAlmostEqual(  4531.231730902182, g[11], delta=0.000001 )
		self.assertAlmostEqual(  294.42765717636166, g[12], delta=0.000001 )
#       These used to work, but they appear to be wrong...
#		self.assertAlmostEqual( -5213.854558101514, g[ 0], delta=0.0000001 )
#		self.assertAlmostEqual(  12836.151998017447, g[ 1], delta=0.0000001 )
#		self.assertAlmostEqual(  629.8310448361855, g[ 2], delta=0.001 )
#		self.assertAlmostEqual(  -74.76717809020077, g[ 3], delta=0.001 )
#		self.assertAlmostEqual(  263.166080293258, g[ 4], delta=0.0001 )
#		self.assertAlmostEqual( -24.52301491338605, g[ 5], delta=0.00000001 )
#		self.assertAlmostEqual(  84.57760963703537, g[ 6], delta=0.00000001 )
#		self.assertAlmostEqual(  36931.80015794064, g[ 7], delta=0.00000001 )
#		self.assertAlmostEqual( -3989.2630876959906, g[ 8], delta=0.00000001 )
#		self.assertAlmostEqual(  15728.304436116596, g[ 9], delta=0.01 )
#		self.assertAlmostEqual( -1252.58, g[10], delta=0.01 )
#		self.assertAlmostEqual(  4531.231730902182, g[11], delta=0.00000001 )
#		self.assertAlmostEqual(  1741.9031614953105, g[12], delta=0.00000001 )
		m.tearDown()


	def test_nl2_single_cycle_aliased_paramaters(self):
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
		m.alias("inc5","inc3",2.0)
		m.alias("inc6","inc2",1.0)
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
		g = m.negative_d_loglike()
		self.assertAlmostEqual( -127397.53666666638, g[ 0], delta=0.000001 )
		self.assertAlmostEqual( 42104.2, g[ 1], delta=0.01 )
		self.assertAlmostEqual( 687.7  , g[ 2], delta=0.01 )
		self.assertAlmostEqual( 1043.7 , g[ 3], delta=0.01 )
		self.assertAlmostEqual( 380.78333333332864, g[ 4], delta=0.00000001 )
		self.assertAlmostEqual( 279.8  , g[ 5], delta=0.01 )
		self.assertAlmostEqual( 113.65 , g[ 6], delta=0.001 )
		self.assertAlmostEqual( 41179.541666666715+7739.108333333339, g[ 7], delta=0.0000001 ) ### inc2
		self.assertAlmostEqual( 60691.541666666686+17374.8*2, g[ 8], delta=0.1 ) ### inc3
		self.assertAlmostEqual( 24028.374999999985, g[ 9], delta=0.0000001 )
		#self.assertAlmostEqual( 17374.8, g[10], delta=0.01 ) ### inc5
		#self.assertAlmostEqual( 7739.108333333339, g[11], delta=0.0000001 ) ### inc6
		self.assertAlmostEqual(-537.598179908304, g[10], delta=0.00000001 )
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
			2.32074    ,
			)
		#		self.assertAlmostEqual( -4213.0122967116695, m.loglike(v), delta=0.00000001 )
		#		m.freshen()
		#		g = m.negative_d_loglike()
		#		self.assertAlmostEqual( -5213.854558101514, g[ 0], delta=0.0000001 )
		#		self.assertAlmostEqual(  12836.151998017447, g[ 1], delta=0.0000001 )
		#		self.assertAlmostEqual(  629.8310448361855, g[ 2], delta=0.001 )
		#		self.assertAlmostEqual(  -74.76717809020077, g[ 3], delta=0.001 )
		#		self.assertAlmostEqual(  263.166080293258, g[ 4], delta=0.0001 )
		#		self.assertAlmostEqual( -24.52301491338605, g[ 5], delta=0.00000001 )
		#		self.assertAlmostEqual(  84.57760963703537, g[ 6], delta=0.00000001 )
		#		self.assertAlmostEqual(  36931.80015794064, g[ 7], delta=0.00000001 )
		#		self.assertAlmostEqual( -3989.2630876959906, g[ 8], delta=0.00000001 )
		#		self.assertAlmostEqual(  15728.304436116596, g[ 9], delta=0.01 )
		#		self.assertAlmostEqual( -1252.58, g[10], delta=0.01 )
		m.tearDown()


	def test_nl2_single_cycle_multithread(self):		
		d = DB.Example('MTC');
		m = Model(d);
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
		g = m.negative_d_loglike()
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
		g = m.negative_d_loglike()
		self.assertAlmostEqual( -4140.9722837573108, g[ 0], delta=0.00001 )
		self.assertAlmostEqual(  6437.577630163296, g[ 1], delta=0.00001 )
		self.assertAlmostEqual(  263.62687005016875, g[ 2], delta=0.001 )
		self.assertAlmostEqual(  -32.73528233531812, g[ 3], delta=0.001 )
		self.assertAlmostEqual(  105.44094790643184, g[ 4], delta=0.0001 )
		self.assertAlmostEqual( -24.52301491338605, g[ 5], delta=0.000001 )
		self.assertAlmostEqual(  84.57760963703537, g[ 6], delta=0.000001 )
		self.assertAlmostEqual(  15532.154389060192, g[ 7], delta=0.0001 )
		self.assertAlmostEqual( -1748.1507300171445, g[ 8], delta=0.00001 )
		self.assertAlmostEqual(  6338.605865710019, g[ 9], delta=0.01 )
		self.assertAlmostEqual( -1252.58, g[10], delta=0.01 )
		self.assertAlmostEqual(  4531.231730902182, g[11], delta=0.000001 )
		self.assertAlmostEqual(  294.42765717636166, g[12], delta=0.000001 )
#		These used to work, but appear to be wrong...
#		self.assertAlmostEqual( -5213.854558101514, g[ 0], delta=0.0000001 )
#		self.assertAlmostEqual(  12836.151998017447, g[ 1], delta=0.0000001 )
#		self.assertAlmostEqual(  629.8310448361855, g[ 2], delta=0.001 )
#		self.assertAlmostEqual(  -74.76717809020077, g[ 3], delta=0.001 )
#		self.assertAlmostEqual(  263.166080293258, g[ 4], delta=0.0001 )
#		self.assertAlmostEqual( -24.52301491338605, g[ 5], delta=0.00000001 )
#		self.assertAlmostEqual(  84.57760963703537, g[ 6], delta=0.00000001 )
#		self.assertAlmostEqual(  36931.80015794064, g[ 7], delta=0.00000001 )
#		self.assertAlmostEqual( -3989.2630876959906, g[ 8], delta=0.00000001 )
#		self.assertAlmostEqual(  15728.304436116596, g[ 9], delta=0.01 )
#		self.assertAlmostEqual( -1252.58, g[10], delta=0.01 )
#		self.assertAlmostEqual(  4531.231730902182, g[11], delta=0.00000001 )
#		self.assertAlmostEqual(  1741.9031614953105, g[12], delta=0.00000001 )
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
		m.option.calc_std_errors = True
		m.nest("existing", 4, "existing") 
		m.link(4, 1)
		m.link(4, 3)
		m.estimate()
		self.assertAlmostEqual( -5236.90001, m.loglike(cached=2), 4 )
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
		m.option.calc_std_errors = True
		m.nest("existing", 4, "existing") 
		m.link(4, 1)
		m.link(4, 3)
		m.estimate()
		self.assertAlmostEqual(	   -5181.7961550603186, m.loglike(cached=2)         ,7 )
		self.assertAlmostEqual(	  -0.53231563866755072, m.parameter("ASC_TRAIN").value  ,7 )
		self.assertAlmostEqual(	-0.0095924003306561544, m.parameter("B_TIME").value     ,7 )
		self.assertAlmostEqual(	-0.0089907063608799867, m.parameter("B_COST").value     ,7 )
		self.assertAlmostEqual(	  -0.13353404397337604, m.parameter("ASC_CAR").value    ,7 )
		self.assertAlmostEqual(	    0.4982901555360098, m.parameter("existing").value   ,7 )
		self.assertAlmostEqual(	  0.046564597147382529, m.parameter("ASC_TRAIN").std_err,7 )
		self.assertAlmostEqual(	0.00056998804078429901, m.parameter("B_TIME").std_err   ,7 )
		self.assertAlmostEqual(	0.00046796689761018697, m.parameter("B_COST").std_err   ,7 )
		self.assertAlmostEqual(	  0.037604832136669264, m.parameter("ASC_CAR").std_err  ,7 )
		self.assertAlmostEqual(	  0.027670801001024776, m.parameter("existing").std_err ,7 )

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
		m.option.calc_std_errors = True
		m.nest("existing", 4, "existing") 
		m.link(4, 1)
		m.link(4, 3)
		m.samplingbias.co("1",1,"SB_TRAIN")
		m.estimate()
		self.assertNearlyEqual( -5169.641515645088, m.loglike(cached=2), 5 )
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
		m.option.calc_std_errors = True
		m.nest("existing", 4, "existing") 
		m.link(4, 1)
		m.link(4, 3)
		m.samplingbias[1]("1","SB_TRAIN")
		self.assertEqual("1", m.samplingbias.co[1][0].data)
		self.assertEqual("SB_TRAIN", m.samplingbias.co[1][0].param)
#		self.assertEqual("1", m.samplingbias.co[1].data)
#		self.assertEqual("samplingbias#2", m.samplingbias.co[1].param)
#		self.assertEqual(2, len(m.samplingbias.co))


	def test_out_of_order_nests(self):
		d = DB.Example('mini')
		m = Model(d);
		m.logger(False)
		m.option.weight_autorescale = False
		m.utility.ca("COST","B_COST")
		m.node(100, 'Walk', 'mu_walking')
		m.node(200, 'notWalking', 'mu_notwalking')
		m.node(300, 'car', 'mu_car')
		m.node(400, 'publicTransport', 'mu_public')
		m.node(500, 'walking', 'mu_walk')
		m.link(0,500)
		m.link(0,200)
		m.link(500,100)
		m.link(100,1)
		m.link(200,300)
		m.link(200,400)
		m.link(300,2)
		m.link(300,3)
		m.link(400,4)
		m.link(400,5)
		m.parameter('B_COST').value = -0.10
		m.parameter('mu_walking').value = 1
		m.parameter('mu_notwalking').value = 0.8
		m.parameter('mu_car').value = 0.8
		# m.parameter('mu_public').value = 0.8 
		m.parameter('mu_walk').value = 1
		m.parameter('B_COST').holdfast = 1
		m.parameter('mu_walking').holdfast = 1
		m.parameter('mu_notwalking').holdfast = 1
		m.parameter('mu_car').holdfast = 1
		#m.parameter('mu_public').holdfast = 1
		m.parameter('mu_walk').holdfast = 1
		m.provision()
		self.assertNearlyEqual(-17.042185953864063, m.loglike([-0.1, 1, .8, .8, -0.0779, 1]), 7)
		self.assertNearlyEqual(-21.789514514441944, m.loglike(([0,1,1,1,1,1])), 7)
		#m.estimate([larch.core.OptimizationMethod(honey=100, thresh=1e-12, max=400, ext=1.1)])


	def test_cnl_loglike(self):
		m = Model.Example(111)
		m.provision()
		m['ASC_TRAIN'].value = 0.0983
		m['B_TIME'].value = -0.00777
		m['B_COST'].value = -0.00819
		m['phi_et'].value =-0.02
		m['existing'].value = 1.0/2.51
		m['ASC_CAR'].value =-0.240
		m['public'].value = 1.0/4.11
		self.assertNearlyEqual(-5214.049, m.loglike(),sigfigs=6)

	def test_ngev(self):
		d = DB.Example('MTC')
		m = Model(d)
		m.option.threads = 4
		m.option.calc_std_errors = False
		m.parameter("cost",0)
		m.parameter("tottime",0) 
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
		m.parameter("motoNest", 0.9, 1.0) 
		m.parameter("greenNest",0.8, 1.0) 
		m.parameter("carNest",  0.7, 1.0) 
		m.parameter("PhiINC", 0.0) 
		m.parameter("PhiCON", 0.0) 
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
		m.nest("moto", 8, "motoNest")
		m.nest("green", 10, "greenNest")
		m.nest("cars", 9, "carNest")
		m.link(8, 9)
		m.link(8, 4)
		m.link(9, 1)
		m.link(9, 2)
		m.link(9, 3)
		m.link(10, 4)
		m.link(10, 5)
		m.link(10, 6)
		m.link[10, 4](data="HHINC", param="PhiINC")
		m.link[10, 4](data="1", param="PhiCON")
		m.setUp()
		ll =  m.loglike()
		g = m.negative_d_loglike().copy()
		v =(-0.000423724,
			-0.00974341 ,
			-0.136579   ,
			-0.740303   ,
			1.19811     ,
			0.288508    ,
			0.679871    ,
			-0.00051391 ,
			0.000961995 ,
			-0.000331558,
			-0.00245688 ,
			-0.00166543 ,
			0.0287018   ,
			0.695356    ,
			1.17513     ,
			-0.0239607  ,
			2.33664     ,
			)

		ll2 =  m.loglike(v)
		m.option.force_recalculate = True
		g2 = m.negative_d_loglike(v)
		self.assertNearlyEqual(-7500.712699231124, ll,)
		self.assertNearlyEqual(-203148.818120826, g[0 ], )
		self.assertNearlyEqual( 55620.21857904255, g[1 ], )
		self.assertNearlyEqual( 1007.8,  g[2 ], )		
		self.assertNearlyEqual( 1516.37, g[3 ], )	
		self.assertNearlyEqual( 518.005, g[4 ], )	
		self.assertNearlyEqual( 342.357, g[5 ], )	
		self.assertNearlyEqual( 151.047, g[6 ], )	
		self.assertNearlyEqual( 60950.5, g[7 ], )		
		self.assertNearlyEqual( 88824.8, g[8 ], )
		self.assertNearlyEqual( 32281.1, g[9 ], )
		self.assertNearlyEqual( 21146.5, g[10], )
		self.assertNearlyEqual( 9743.75, g[11], )
		self.assertNearlyEqual( 339.24,  g[12], )
		self.assertNearlyEqual( 420.713, g[13], )			
		self.assertNearlyEqual( -1142.84,g[14], )			
		self.assertNearlyEqual( -1089.43,g[15], )									
		self.assertNearlyEqual( -16.8988,g[16], )
		self.assertNearlyEqual(-6619.19437740866, ll2     )
		self.assertNearlyEqual( -44078.1,         g2[0 ], )
		self.assertNearlyEqual( 45385.9,          g2[1 ], )		
		self.assertNearlyEqual( 455.611,          g2[2 ], )		
		self.assertNearlyEqual( 441.68,           g2[3 ], )	
		self.assertNearlyEqual( 888.106,          g2[4 ], )	
		self.assertNearlyEqual( 198.114,          g2[5 ], )	
		self.assertNearlyEqual( 65.9350316424698, g2[6 ], )	
		self.assertNearlyEqual( 26252.9503317513, g2[7 ], )		
		self.assertNearlyEqual( 25587.8552481395, g2[8 ], )
		self.assertNearlyEqual( 49404.8394509919, g2[9 ], )
		self.assertNearlyEqual( 12898.5,          g2[10], )
		self.assertNearlyEqual( 6240.038886725008,g2[11], )
		self.assertNearlyEqual( -620.796083378989,g2[12], )
		self.assertNearlyEqual( 528.361336255419, g2[13], )			
		self.assertNearlyEqual( -630.085253427633,g2[14], )			
		self.assertNearlyEqual( 9125.88491483512, g2[15], )									
		self.assertNearlyEqual( 174.008,          g2[16], )




	def test_building_nl_sequentially(self):

		m = Model.Example()
		m.setUp()

		r = m.maximize_loglike('SLSQP')
		self.assertNearlyEqual(-7309.600971749682,  r.loglike_null)
		self.assertEqual([('SLSQP', 38)], r.niter)
		self.assertNearlyEqual(6.012024227765063e-07, r.ctol)
		self.assertTrue(r.success)
		self.assertNearlyEqual(-3626.1862547377136,  r.loglike)

		nonmotorized = m.new_nest("nonmotorized", children=[5,6])
		shareride = m.new_nest("shared ride", children=[2,3])
		automobile = m.new_nest("automobile", children=[1,shareride])
		motorized = m.new_nest("motorized", children=[automobile,4])

		r1 = m.maximize_loglike('SLSQP')

		self.assertNearlyEqual(-7309.600971749682,  r1.loglike_null)
		self.assertNearlyEqual(-3609.5435783723583,  r1.loglike)
		#self.assertEqual([('SLSQP', 26)], r1.niter)
		x_correct = [ -2.01351786e+00,  -2.86642895e+00,  -5.54027095e-01,
				-2.46100890e+00,  -4.90688219e-01,  -1.83655214e-03,
				-7.51691241e-04,  -3.56551747e-03,  -1.27448065e-02,
				-9.84707826e-03,  -3.94120262e-02,  -4.09608576e-03,
				 1.00000000e+00,   5.23238992e-01,   1.00000000e+00,
				 7.33693038e-01]
		for x_c, x_o in zip (x_correct, r1.x):
			self.assertNearlyEqual(x_c,x_o)

		m.option.enforce_constraints = True
		try:
			import networkx
		except ImportError:
			self.skipTest('networkx package not installed')
		r2 = m.maximize_loglike('SLSQP')
		self.assertNearlyEqual(-3623.8414797211444, r2.loglike)
		self.assertNearlyEqual(-7309.600971749633, r2.loglike_null)
		x2_correct = [ -2.10038566e+00,  -3.16519485e+00,  -6.71644352e-01,
				-2.36951756e+00,  -2.05697845e-01,  -1.84931282e-03,
				-5.87837571e-04,  -5.16635470e-03,  -1.27779660e-02,
				-9.67688866e-03,  -5.10725736e-02,  -4.80853420e-03,
				 1.00000000e+00,   6.56146997e-01,   1.00000000e+00,
				 1.00000000e+00]
		for x_c, x_o in zip (x2_correct, r2.x):
			self.assertNearlyEqual(x_c,x_o, sigfigs=2.5)


		self.assertNearlyEqual(-7309.600971749633, m.loglike([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1], cached=False))
		self.assertNearlyEqual(-numpy.inf, m.loglike([-10,0,0,0,0,0,0,0,0,10,0,0,1,1,1,1], cached=False))

	def test_cnl_probability_features(self):
		m = Model.Example(111, DB.Example('swissmetro'))
		m.setUp()
		m.loglike([ 0.09827949, -0.24044757, -0.0077686 , -0.00818902,  0.39764565,
				0.24308742, -0.01971338])
		p1 = m.probability()
		p2 = m.Probability()
		self.assertTrue(p1 is p2)
		self.assertTrue(len(m._xylem().all_codes()) == p1.shape[1])
		self.assertEqual((6768, 6), p1.shape)
		p1top = [ 0.15184372,  0.62716611,  0.22099017,  0.35932659,  0.64067341, 1.        ]
		for p_c, p_o in zip (p1top, p1[0,:]):
			self.assertNearlyEqual(p_c,p_o)



	def test_utility_ce_example80(self):
		m = Model.Example(80)
		m.setup_utility_ce()
		m.parameter_values([-0.13923, 0.00272, 0.02866, 0.029358, -0.00142, -0.0010364, -0.64088,
							-0.2469074, 0.455194, 0.45519, 0.993614, 0.99361, 1.244206])
		self.assertNearlyEqual(-6090.183759158203, m.loglike(), sigfigs=8)
		d_ll_correct = [ -1.24588068e+01,   6.42281963e+00,  -7.69257754e+00,   6.62163599e-01,
		                 -1.18375540e+03,   6.86147005e+02,   1.45742634e+01,   2.20817939e-01,
						 -1.42346504e+00,   8.27116755e-01,   2.08385451e+00,   1.10369816e+01,
						 2.44998020e-01]
		for x_c, x_o in zip (d_ll_correct, m.d_loglike()):
			self.assertNearlyEqual(x_c,x_o)
		self.assertTrue(m.Data("UtilityCA") is None)


	def test_utility_ce_manual(self):
		d = DB.Example('MTC')
		m = Model(d)
		m.utility.ca("tottime")
		m.utility.ca("totcost")
		m.setup_utility_ce()
		rm = m.maximize_loglike()
		self.assertEqual([('bhhh', 7)], rm.niter)
		self.assertNearlyEqual(-5902.369810160076, rm.loglike)
		self.assertNearlyEqual(-5902.369810160076, m.loglike())
		self.assertNearlyEqual(-0.10022946, rm.x[0])
		self.assertNearlyEqual(0.00190891, rm.x[1])


	def test_utility_ce_automatic(self):
		d = DB.Example('MTC')
		# with idca
		m = Model(d)
		m.utility.ca("tottime")
		m.utility.ca("totcost")
		m.option.idca_avail_ratio_floor = 0.0
		rm = m.maximize_loglike()
		self.assertEqual([('bhhh', 7)], rm.niter)
		self.assertNearlyEqual(-5902.369810160076, rm.loglike)
		self.assertNearlyEqual(-5902.369810160076, m.loglike())
		self.assertNearlyEqual(-0.10022946, rm.x[0])
		self.assertNearlyEqual(0.00190891, rm.x[1])
		self.assertTrue(m.Data("UtilityCA") is not None)
		self.assertTrue(not m.Data_UtilityCE_builtin.active())

		# with idce
		m2 = Model(d)
		m2.utility.ca("tottime")
		m2.utility.ca("totcost")
		m2.option.idca_avail_ratio_floor = 1.0
		rm2 = m2.maximize_loglike()
		self.assertEqual([('bhhh', 7)], rm2.niter)
		self.assertNearlyEqual(-5902.369810160076, rm2.loglike)
		self.assertNearlyEqual(-5902.369810160076, m2.loglike())
		self.assertNearlyEqual(-0.10022946, rm2.x[0])
		self.assertNearlyEqual(0.00190891, rm2.x[1])
		self.assertTrue(m2.Data("UtilityCA") is None)
		self.assertTrue(m2.Data_UtilityCE_builtin.active())

		for z1,z2 in zip(m.d_loglike(), m2.d_loglike()):
			self.assertNearlyEqual(z1,z2)
		for z1,z2 in zip(m.d_loglike([0,0]), m2.d_loglike([0,0])):
			self.assertNearlyEqual(z1,z2)
		for z1,z2 in zip(m.d_loglike([0.1,0.1]), m2.d_loglike([0.1,0.1])):
			self.assertNearlyEqual(z1,z2)

