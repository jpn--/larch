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
		self.assertAlmostEqual(-5331.252006978187, M.LL())
		self.assertAlmostEqual(-1.0837854589547063e-02, M.parameter("B_COST").value)
		self.assertAlmostEqual( 5.1830069605943250e-04, M.parameter("B_COST").std_err)
		self.assertAlmostEqual( 6.8224682140162821e-04, M.parameter("B_COST").robust_std_err)		
		self.assertAlmostEqual( 0.0000000000000000e+00, M.parameter("B_COST").null_value)
		self.assertAlmostEqual( 0.0000000000000000e+00, M.parameter("B_COST").initial_value)
		self.assertAlmostEqual(-1.2778463153730711e-02, M.parameter("B_TIME").value)
		self.assertAlmostEqual( 5.6883173474293924e-04, M.parameter("B_TIME").std_err)
		self.assertAlmostEqual( 1.0425386257158060e-03, M.parameter("B_TIME").robust_std_err)		
		self.assertAlmostEqual( 0.0000000000000000e+00, M.parameter("B_TIME").null_value)
		self.assertAlmostEqual( 0.0000000000000000e+00, M.parameter("B_TIME").initial_value)
		self.assertAlmostEqual(-7.0120150002982684e-01, M.parameter("ASC_TRAIN").value)
		self.assertAlmostEqual( 5.4873938849339082e-02, M.parameter("ASC_TRAIN").std_err)
		self.assertAlmostEqual( 8.2561850648792984e-02, M.parameter("ASC_TRAIN").robust_std_err)		
		self.assertAlmostEqual( 0.0000000000000000e+00, M.parameter("ASC_TRAIN").null_value)
		self.assertAlmostEqual( 0.0000000000000000e+00, M.parameter("ASC_TRAIN").initial_value)
		self.assertAlmostEqual(-1.5463266009806520e-01, M.parameter("ASC_CAR").value)
		self.assertAlmostEqual( 4.3235420408219878e-02, M.parameter("ASC_CAR").std_err)
		self.assertAlmostEqual( 5.8163317897743642e-02, M.parameter("ASC_CAR").robust_std_err)		
		self.assertAlmostEqual( 0.0000000000000000e+00, M.parameter("ASC_CAR").null_value)
		self.assertAlmostEqual( 0.0000000000000000e+00, M.parameter("ASC_CAR").initial_value)
		self.assertAlmostEqual( 5.498936430191042e-08,   M.parameter("B_COST").covariance['B_TIME'    ])
		self.assertAlmostEqual( 8.220700089527862e-08,   M.parameter("B_COST").covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( 4.847594597637228e-06,   M.parameter("B_COST").covariance['ASC_CAR'   ])
		self.assertAlmostEqual( 2.686356115356922e-07,   M.parameter("B_COST").covariance['B_COST'    ])
		self.assertAlmostEqual( 3.235695424506616e-07,   M.parameter("B_TIME").covariance['B_TIME'    ])
		self.assertAlmostEqual( -2.2539121763960147e-05, M.parameter("B_TIME").covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( -1.4377375333984807e-05, M.parameter("B_TIME").covariance['ASC_CAR'   ])
		self.assertAlmostEqual( 5.498936430191042e-08,   M.parameter("B_TIME").covariance['B_COST'    ])
		self.assertAlmostEqual( -1.4377375333984807e-05, M.parameter("ASC_CAR").covariance['B_TIME'    ])
		self.assertAlmostEqual( 0.0013769280697752,      M.parameter("ASC_CAR").covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( 0.001869301577875516,    M.parameter("ASC_CAR").covariance['ASC_CAR'   ])
		self.assertAlmostEqual( 4.847594597637228e-06 ,  M.parameter("ASC_CAR").covariance['B_COST'    ])
		self.assertAlmostEqual( -2.2539121763960147e-05, M.parameter("ASC_TRAIN").covariance['B_TIME'    ])
		self.assertAlmostEqual( 0.003011149164841005,    M.parameter("ASC_TRAIN").covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( 0.0013769280697752,      M.parameter("ASC_TRAIN").covariance['ASC_CAR'   ])
		self.assertAlmostEqual( 8.220700089527862e-08 ,  M.parameter("ASC_TRAIN").covariance['B_COST'    ])
		self.assertAlmostEqual( 2.1979623359675466e-07,  M.parameter("B_COST").robust_covariance['B_TIME'    ])
		self.assertAlmostEqual( -8.305452040438645e-06,  M.parameter("B_COST").robust_covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( 2.8640528395451224e-07,  M.parameter("B_COST").robust_covariance['ASC_CAR'   ])
		self.assertAlmostEqual( 4.6546072531262517e-07 , M.parameter("B_COST").robust_covariance['B_COST'    ])
		self.assertAlmostEqual( 1.0868867861094013e-06,  M.parameter("B_TIME").robust_covariance['B_TIME'    ])
		self.assertAlmostEqual( -7.602251642020505e-05,  M.parameter("B_TIME").robust_covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( -4.82419823628773e-05,   M.parameter("B_TIME").robust_covariance['ASC_CAR'   ])
		self.assertAlmostEqual( 2.1979623359675466e-07 , M.parameter("B_TIME").robust_covariance['B_COST'    ])
		self.assertAlmostEqual( -4.82419823628773e-05,   M.parameter("ASC_CAR").robust_covariance['B_TIME'    ])
		self.assertAlmostEqual( 0.0039013058327078704,   M.parameter("ASC_CAR").robust_covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( 0.003382971548873986,    M.parameter("ASC_CAR").robust_covariance['ASC_CAR'   ])
		self.assertAlmostEqual( 2.8640528395451224e-07 , M.parameter("ASC_CAR").robust_covariance['B_COST'    ])
		self.assertAlmostEqual( -7.602251642020505e-05,  M.parameter("ASC_TRAIN").robust_covariance['B_TIME'    ])
		self.assertAlmostEqual( 0.006816459182553598,    M.parameter("ASC_TRAIN").robust_covariance['ASC_TRAIN' ])
		self.assertAlmostEqual( 0.0039013058327078704,   M.parameter("ASC_TRAIN").robust_covariance['ASC_CAR'   ])
		self.assertAlmostEqual( -8.305452040438645e-06 , M.parameter("ASC_TRAIN").robust_covariance['B_COST'    ])
						  
	def test_swissmetro02weighted(self):		
		from ..examples import load_example
		load_example(102)
		from ..examples import model
		M = model()
		M.estimate()
		self.assertAlmostEqual(-5273.74259          , M.LL(), 5)
		self.assertNearlyEqual(-0.011196            , M.parameter("B_COST").value)
		self.assertNearlyEqual(0.0005201014058006726, M.parameter("B_COST").std_err)
		self.assertNearlyEqual(0.0006767857292272   , M.parameter("B_COST").robust_std_err)
		self.assertAlmostEqual(0.0                  , M.parameter("B_COST").null_value)
		self.assertAlmostEqual(0.0                  , M.parameter("B_COST").initial_value)
		self.assertNearlyEqual(-0.013215340567778556, M.parameter("B_TIME").value)
		self.assertNearlyEqual(0.0005693070009628066, M.parameter("B_TIME").std_err)
		self.assertNearlyEqual(0.0010327853238187572, M.parameter("B_TIME").robust_std_err    ,2)
		self.assertAlmostEqual(0.0                  , M.parameter("B_TIME").null_value)
		self.assertAlmostEqual(0.0                  , M.parameter("B_TIME").initial_value)
		self.assertNearlyEqual(-0.7564616727277894  , M.parameter("ASC_TRAIN").value          ,2)
		self.assertNearlyEqual(0.05603796211533885  , M.parameter("ASC_TRAIN").std_err        ,2)
		self.assertNearlyEqual(0.08325742298643872  , M.parameter("ASC_TRAIN").robust_std_err ,2)
		self.assertAlmostEqual(0.0                  , M.parameter("ASC_TRAIN").null_value)
		self.assertAlmostEqual(0.0                  , M.parameter("ASC_TRAIN").initial_value)
		self.assertNearlyEqual(-0.11434225215207203 , M.parameter("ASC_CAR").value            ,2)
		self.assertNearlyEqual(0.04315184534133326  , M.parameter("ASC_CAR").std_err          ,2)
		self.assertNearlyEqual(0.05856746057118884  , M.parameter("ASC_CAR").robust_std_err   ,2)
		self.assertAlmostEqual(0.0                  , M.parameter("ASC_CAR").null_value)
		self.assertAlmostEqual(0.0                  , M.parameter("ASC_CAR").initial_value)

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



