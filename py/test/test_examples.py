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
    __package__ = "larch.test.test_examples"

from ..test import TEST_DATA, ELM_TestCase, DEEP_TEST
from ..core import Parameter, Model, DB, LarchError, SQLiteError

class TestSwissmetroExamples(ELM_TestCase):

	_multiprocess_shared_ = True

	@classmethod
	def setUpClass(cls):
		from ..examples import swissmetro00data as ed
		cls.ed = ed

	def test_swissmetro01logit(self):		
		from ..examples import load_example
		load_example(101)
		from ..examples import model
		M = model()
		M.estimate()
		self.assertAlmostEqual(   -5331.2520069781867 , M.LL())
		self.assertAlmostEqual( -0.010837854591997336 , M.parameter("B_COST").value)
		self.assertAlmostEqual(0.00051830069613924713 , M.parameter("B_COST").std_err)
		self.assertAlmostEqual(0.00068224682164246908 , M.parameter("B_COST").robust_std_err)
		self.assertAlmostEqual(                     0 , M.parameter("B_COST").null_value)
		self.assertAlmostEqual(                     0 , M.parameter("B_COST").initial_value)
		self.assertAlmostEqual( -0.012778463167925237 , M.parameter("B_TIME").value)
		self.assertAlmostEqual(0.00056883173483269665 , M.parameter("B_TIME").std_err)
		self.assertAlmostEqual( 0.0010425386260355324 , M.parameter("B_TIME").robust_std_err)
		self.assertAlmostEqual(                     0 , M.parameter("B_TIME").null_value)
		self.assertAlmostEqual(                     0 , M.parameter("B_TIME").initial_value)
		self.assertAlmostEqual(  -0.70120149931593501 , M.parameter("ASC_TRAIN").value)
		self.assertAlmostEqual(  0.054873938851410259 , M.parameter("ASC_TRAIN").std_err)
		self.assertAlmostEqual(   0.08256185065647649 , M.parameter("ASC_TRAIN").robust_std_err)
		self.assertAlmostEqual(                     0 , M.parameter("ASC_TRAIN").null_value)
		self.assertAlmostEqual(                     0 , M.parameter("ASC_TRAIN").initial_value)
		self.assertAlmostEqual(  -0.15463265994157291 , M.parameter("ASC_CAR").value)
		self.assertAlmostEqual(  0.043235420404501367 , M.parameter("ASC_CAR").std_err)
		self.assertAlmostEqual(  0.058163317878822646 , M.parameter("ASC_CAR").robust_std_err)
		self.assertAlmostEqual(                     0 , M.parameter("ASC_CAR").null_value)
		self.assertAlmostEqual(                     0 , M.parameter("ASC_CAR").initial_value)
		self.assertAlmostEqual(5.4989364384274582e-08 , M.parameter("B_COST").covariance['B_TIME'    ])
		self.assertAlmostEqual(8.2206996770805741e-08 , M.parameter("B_COST").covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( 4.847594599639639e-06 , M.parameter("B_COST").covariance['ASC_CAR'   ])
		self.assertAlmostEqual(2.6863561161842814e-07 , M.parameter("B_COST").covariance['B_COST'    ])
		self.assertAlmostEqual( 3.235695425527753e-07 , M.parameter("B_TIME").covariance['B_TIME'    ])
		self.assertAlmostEqual(-2.2539121766964457e-05, M.parameter("B_TIME").covariance['ASC_TRAIN' ])
		self.assertAlmostEqual(-1.4377375327182157e-05, M.parameter("B_TIME").covariance['ASC_CAR'   ])
		self.assertAlmostEqual(5.4989364384274582e-08 , M.parameter("B_TIME").covariance['B_COST'    ])
		self.assertAlmostEqual(-1.4377375327182157e-05, M.parameter("ASC_CAR").covariance['B_TIME'    ])
		self.assertAlmostEqual( 0.0013769280691442523 , M.parameter("ASC_CAR").covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( 0.0018693015775539733 , M.parameter("ASC_CAR").covariance['ASC_CAR'   ])
		self.assertAlmostEqual( 4.847594599639639e-06 , M.parameter("ASC_CAR").covariance['B_COST'    ])
		self.assertAlmostEqual(-2.2539121766964457e-05, M.parameter("ASC_TRAIN").covariance['B_TIME'    ])
		self.assertAlmostEqual( 0.0030111491650683126 , M.parameter("ASC_TRAIN").covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( 0.0013769280691442523 , M.parameter("ASC_TRAIN").covariance['ASC_CAR'   ])
		self.assertAlmostEqual(8.2206996770805741e-08 , M.parameter("ASC_TRAIN").covariance['B_COST'    ])
		self.assertAlmostEqual(2.1979623403851048e-07 , M.parameter("B_COST").robust_covariance['B_TIME'    ])
		self.assertAlmostEqual(-8.305452063886241e-06 , M.parameter("B_COST").robust_covariance['ASC_TRAIN' ])
		self.assertAlmostEqual(2.8640528329606779e-07 , M.parameter("B_COST").robust_covariance['ASC_CAR'   ])
		self.assertAlmostEqual(4.6546072564125103e-07 , M.parameter("B_COST").robust_covariance['B_COST'    ])
		self.assertAlmostEqual(1.0868867867760557e-06 , M.parameter("B_TIME").robust_covariance['B_TIME'    ])
		self.assertAlmostEqual(-7.6022516448137793e-05, M.parameter("B_TIME").robust_covariance['ASC_TRAIN' ])
		self.assertAlmostEqual(-4.824198234263382e-05 , M.parameter("B_TIME").robust_covariance['ASC_CAR'   ])
		self.assertAlmostEqual(2.1979623403851048e-07 , M.parameter("B_TIME").robust_covariance['B_COST'    ])
		self.assertAlmostEqual(-4.824198234263382e-05 , M.parameter("ASC_CAR").robust_covariance['B_TIME'    ])
		self.assertAlmostEqual( 0.0039013058305926744 , M.parameter("ASC_CAR").robust_covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( 0.0033829715466729702 , M.parameter("ASC_CAR").robust_covariance['ASC_CAR'   ])
		self.assertAlmostEqual(2.8640528329606779e-07 , M.parameter("ASC_CAR").robust_covariance['B_COST'    ])
		self.assertAlmostEqual(-7.6022516448137793e-05, M.parameter("ASC_TRAIN").robust_covariance['B_TIME'    ])
		self.assertAlmostEqual( 0.0068164591838223276 , M.parameter("ASC_TRAIN").robust_covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( 0.0039013058305926744 , M.parameter("ASC_TRAIN").robust_covariance['ASC_CAR'   ])
		self.assertAlmostEqual(-8.305452063886241e-06 , M.parameter("ASC_TRAIN").robust_covariance['B_COST'    ])
						  
	def test_swissmetro02weighted(self):		
		from ..examples import load_example
		load_example(102)
		from ..examples import model
		M = model()
		M.estimate()
		self.assertAlmostEqual(   -5273.7424612016184, M.LL()                                  ,3)
		self.assertAlmostEqual( -0.011196551959180576, M.parameter("B_COST").value             ,7)
		self.assertAlmostEqual(0.00052009766901624634, M.parameter("B_COST").std_err           ,7)
		self.assertAlmostEqual(0.00067678573779952661, M.parameter("B_COST").robust_std_err    ,7)
		self.assertAlmostEqual(                     0, M.parameter("B_COST").null_value        ,7)
		self.assertAlmostEqual(                     0, M.parameter("B_COST").initial_value     ,7)
		self.assertAlmostEqual( -0.013215230457827699, M.parameter("B_TIME").value             ,7)
		self.assertAlmostEqual(0.00056930596508046117, M.parameter("B_TIME").std_err           ,7)
		self.assertAlmostEqual( 0.0010365818509383274, M.parameter("B_TIME").robust_std_err    ,7)
		self.assertAlmostEqual(                     0, M.parameter("B_TIME").null_value        ,7)
		self.assertAlmostEqual(                     0, M.parameter("B_TIME").initial_value     ,7)
		self.assertAlmostEqual(  -0.75653171259560081, M.parameter("ASC_TRAIN").value          ,7)
		self.assertAlmostEqual(  0.056038547832057167, M.parameter("ASC_TRAIN").std_err        ,7)
		self.assertAlmostEqual(  0.083257424005508493, M.parameter("ASC_TRAIN").robust_std_err ,7)
		self.assertAlmostEqual(                     0, M.parameter("ASC_TRAIN").null_value     ,7)
		self.assertAlmostEqual(                     0, M.parameter("ASC_TRAIN").initial_value  ,7)
		self.assertAlmostEqual(  -0.11431063366935797, M.parameter("ASC_CAR").value            ,7)
		self.assertAlmostEqual(  0.043151665433859974, M.parameter("ASC_CAR").std_err          ,7)
		self.assertAlmostEqual(  0.058311781077102311, M.parameter("ASC_CAR").robust_std_err   ,7)
		self.assertAlmostEqual(                     0, M.parameter("ASC_CAR").null_value       ,7)
		self.assertAlmostEqual(                     0, M.parameter("ASC_CAR").initial_value    ,7)

	def test_swissmetro04transforms(self):		
		from ..examples import load_example
		load_example(104)
		from ..examples import model
		M = model()
		M.estimate()
		self.assertNearlyEqual(-5423.299182871572, M.LL(), 6)
		self.assertNearlyEqual(-1.036332141437882   , M.parameter("B_LOGCOST").value,4)
		self.assertNearlyEqual(-0.010716067643111737, M.parameter("B_TIME").value,4)
		self.assertNearlyEqual(-0.8513254051831449  , M.parameter("ASC_TRAIN").value,4)
		self.assertNearlyEqual(-0.2744308889363716  , M.parameter("ASC_CAR").value,3)

	def test_swissmetro11crossnested(self):
		# Check values vs Biogeme                           Robust  Robust
		#Name	        Value	Std err	t-test	p-value     Std err	t-test	p-value
		#ALPHA_EXISTING	0.495	0.0289	17.11	0.00		0.0348	14.25	0.00	
		#ASC_CAR	   -0.240	0.0384	-6.26	0.00		0.0534	-4.50	0.00
		#ASC_TRAIN	    0.0983	0.0563	1.74	0.08	*	0.0700	1.40	0.16	*
		#B_COST	       -0.819	0.0446	-18.36	0.00		0.0590	-13.89	0.00
		#B_TIME	       -0.777	0.0558	-13.93	0.00		0.102	-7.59	0.00
		#MU_EXISTING	2.51	0.175	14.40	0.00		0.248	10.13	0.00	
		#MU_PUBLIC	    4.11	0.569	7.23	0.00		0.497	8.28	0.00
		from ..examples import load_example
		load_example(111)
		from ..examples import model
		M = model()
		M.setUp()
		self.assertEqual(['ASC_TRAIN', 'ASC_CAR', 'B_TIME', 'B_COST', 'MU_EXISTING', 'MU_PUBLIC', 'PHI_EXISTING'], M.parameter_names())
		ll = M.loglike([0.0983, -0.240, -0.00777, -0.00819, 1/2.51, 1/4.11, -0.0200])
		M.estimate()
		self.assertNearlyEqual(    -5214.050277246205, ll                                         ,5)
		self.assertNearlyEqual(   -5214.0491950955357, M.LL()                                     ,5)
		self.assertNearlyEqual( -0.008188710196615958, M.parameter("B_COST").value                ,3)
		self.assertNearlyEqual(-0.0077680935404387729, M.parameter("B_TIME").value                ,3)
		self.assertNearlyEqual(  0.098275590091527643, M.parameter("ASC_TRAIN").value             ,3)
		self.assertNearlyEqual(   -0.2404680225222213, M.parameter("ASC_CAR").value               ,3)
		self.assertNearlyEqual(   0.39762776886073403, M.parameter("MU_EXISTING").value           ,3)
		self.assertNearlyEqual(   0.24307631828377166, M.parameter("MU_PUBLIC").value             ,3)
		self.assertNearlyEqual( -0.019714708365512132, M.parameter("PHI_EXISTING").value          ,3)
		self.assertNearlyEqual(0.00044600500287571504, M.parameter("B_COST").std_err              ,3)
		self.assertNearlyEqual(0.00055762769364630846, M.parameter("B_TIME").std_err              ,3)
		self.assertNearlyEqual(  0.056338434373479053, M.parameter("ASC_TRAIN").std_err           ,3)
		self.assertNearlyEqual(  0.038437965787190519, M.parameter("ASC_CAR").std_err             ,3)
		self.assertNearlyEqual(  0.027606316034293359, M.parameter("MU_EXISTING").std_err         ,3)
		self.assertNearlyEqual(  0.033602461452014731, M.parameter("MU_PUBLIC").std_err           ,3)
		self.assertNearlyEqual(   0.11571579085493591, M.parameter("PHI_EXISTING").std_err        ,3)
		self.assertNearlyEqual(0.00058970877273927501, M.parameter("B_COST").robust_std_err       ,3)
		self.assertNearlyEqual( 0.0010237850873481284, M.parameter("B_TIME").robust_std_err       ,3)
		self.assertNearlyEqual(   0.06997929869399587, M.parameter("ASC_TRAIN").robust_std_err    ,3)
		self.assertNearlyEqual(  0.053449732684130648, M.parameter("ASC_CAR").robust_std_err      ,3)
		self.assertNearlyEqual(   0.03926320793927153, M.parameter("MU_EXISTING").robust_std_err  ,3)
		self.assertNearlyEqual(  0.029349015935364681, M.parameter("MU_PUBLIC").robust_std_err    ,3)
		self.assertNearlyEqual(   0.13902172458108558, M.parameter("PHI_EXISTING").robust_std_err ,3)

	def test_swissmetro11crossnested_slsqp(self):
		# Check values vs Biogeme                           Robust  Robust
		#Name	        Value	Std err	t-test	p-value     Std err	t-test	p-value
		#ALPHA_EXISTING	0.495	0.0289	17.11	0.00		0.0348	14.25	0.00	
		#ASC_CAR	   -0.240	0.0384	-6.26	0.00		0.0534	-4.50	0.00
		#ASC_TRAIN	    0.0983	0.0563	1.74	0.08	*	0.0700	1.40	0.16	*
		#B_COST	       -0.819	0.0446	-18.36	0.00		0.0590	-13.89	0.00
		#B_TIME	       -0.777	0.0558	-13.93	0.00		0.102	-7.59	0.00
		#MU_EXISTING	2.51	0.175	14.40	0.00		0.248	10.13	0.00	
		#MU_PUBLIC	    4.11	0.569	7.23	0.00		0.497	8.28	0.00
		from ..examples import load_example
		load_example(111)
		from ..examples import model
		M = model()
		M.setUp()
		r = M.maximize_loglike('SLSQP')
		self.assertNearlyEqual(   -5214.0491950955357, M.LL()                                     ,5)
		self.assertNearlyEqual( -0.008188710196615958, M.parameter("B_COST").value                ,3)
		self.assertNearlyEqual(-0.0077680935404387729, M.parameter("B_TIME").value                ,3)
		self.assertNearlyEqual(  0.098275590091527643, M.parameter("ASC_TRAIN").value             ,3)
		self.assertNearlyEqual(   -0.2404680225222213, M.parameter("ASC_CAR").value               ,3)
		self.assertNearlyEqual(   0.39762776886073403, M.parameter("MU_EXISTING").value           ,3)
		self.assertNearlyEqual(   0.24307631828377166, M.parameter("MU_PUBLIC").value             ,3)
		self.assertNearlyEqual( -0.019714708365512132, M.parameter("PHI_EXISTING").value          ,3)
		self.assertNearlyEqual(0.00044600500287571504, M.parameter("B_COST").std_err              ,3)
		self.assertNearlyEqual(0.00055762769364630846, M.parameter("B_TIME").std_err              ,3)
		self.assertNearlyEqual(  0.056338434373479053, M.parameter("ASC_TRAIN").std_err           ,3)
		self.assertNearlyEqual(  0.038437965787190519, M.parameter("ASC_CAR").std_err             ,3)
		self.assertNearlyEqual(  0.027606316034293359, M.parameter("MU_EXISTING").std_err         ,3)
		self.assertNearlyEqual(  0.033602461452014731, M.parameter("MU_PUBLIC").std_err           ,3)
		self.assertNearlyEqual(   0.11571579085493591, M.parameter("PHI_EXISTING").std_err        ,3)
		self.assertNearlyEqual(0.00058970877273927501, M.parameter("B_COST").robust_std_err       ,3)
		self.assertNearlyEqual( 0.0010237850873481284, M.parameter("B_TIME").robust_std_err       ,3)
		self.assertNearlyEqual(   0.06997929869399587, M.parameter("ASC_TRAIN").robust_std_err    ,3)
		self.assertNearlyEqual(  0.053449732684130648, M.parameter("ASC_CAR").robust_std_err      ,3)
		self.assertNearlyEqual(   0.03926320793927153, M.parameter("MU_EXISTING").robust_std_err  ,3)
		self.assertNearlyEqual(  0.029349015935364681, M.parameter("MU_PUBLIC").robust_std_err    ,3)
		self.assertNearlyEqual(   0.13902172458108558, M.parameter("PHI_EXISTING").robust_std_err ,3)



