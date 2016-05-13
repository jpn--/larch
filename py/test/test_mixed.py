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
    __package__ = "larch.test.test_mixed"

from ..test import TEST_DATA, ELM_TestCase, DEEP_TEST
from ..core import Parameter, Model, DB, LarchError, SQLiteError, LarchCacheError, QuerySetTwoTable
from ..model import ModelFamily
from ..roles import ParameterRef
from ..mixed import NormalMixedModel
import scipy.optimize
import itertools

class TestMixed(ELM_TestCase):

	_multiprocess_shared_ = True # nose will run setUpClass once for all tests if true

	@classmethod
	def setUpClass(cls):
		# Expensive fixture setup method goes here
		cls._db = DB.Example('SWISSMETRO')

	@classmethod
	def tearDownClass(cls):
		# Expensive fixture teardown method goes here
		del cls._db

	def test_mixed_cnl_model(self):
		d = self._db
		m = Model.Example(111, d=d)
		m.db.queries.idco_query = 'SELECT _rowid_ AS caseid, * FROM data WHERE CHOICE!=0 AND (PURPOSE==1 OR PURPOSE==3) AND _rowid_<=100'
		x = numpy.array([ 0.09827949, -0.24044757, -0.0077686 , -0.00818902,  0.39764564,  0.24308742, -0.01971338])
		m.parameter_array[:] = x[:]
		m.parameter("B_TIME_S", holdfast=0, value=0.00)
		m.utility.co("TRAIN_TT *1",1,"B_TIME_S")
		m.utility.co("SM_TT *1",2,"B_TIME_S")
		m.utility.co("CAR_TT *1",3,"B_TIME_S")
		m.parameter("B_COST_S", holdfast=0, value=0.00)
		m.utility.co("TRAIN_CO*(GA==0) *1",1,"B_COST_S")
		m.utility.co("SM_CO*(GA==0) *1",2,"B_COST_S")
		m.utility.co("CAR_CO *1",3,"B_COST_S")
		m.reprovision()
		m.setUp()
		mm = NormalMixedModel(m, ['B_TIME_S','B_COST_S'], ndraws=10, seed=0)
		v = mm.parameter_values()
		v[-1] = 0.01
		v[-2] = 0.01
		v[-3] = 0.01
		mm.parameter_values(v)
		self.assertNearlyEqual(-77.89535520953255, mm.loglike(), sigfigs=3)
		ag = mm.d_loglike()
		fdg = scipy.optimize.approx_fprime(mm.parameter_values(), mm.loglike, mm.parameter_values()*1e-4)
		for g1,g2 in zip(ag,fdg):
			self.assertNearlyEqual(g1,g2, sigfigs=3)


	def test_mixed_mnl_model(self):
		d = self._db
		m = Model.Example(101, d=d)
		m.db.queries.idco_query = 'SELECT _rowid_ AS caseid, * FROM data WHERE CHOICE!=0 AND (PURPOSE==1 OR PURPOSE==3) AND _rowid_<=100'
		x = numpy.array([-0.70122688, -0.15465521, -0.01277806, -0.01083774])
		m.parameter_array[:] = x[:]
		m.parameter("B_TIME_S", holdfast=0, value=0.00)
		m.utility.co("TRAIN_TT *1",1,"B_TIME_S")
		m.utility.co("SM_TT *1",2,"B_TIME_S")
		m.utility.co("CAR_TT *1",3,"B_TIME_S")
		m.parameter("B_COST_S", holdfast=0, value=0.00)
		m.utility.co("TRAIN_CO*(GA==0) *1",1,"B_COST_S")
		m.utility.co("SM_CO*(GA==0) *1",2,"B_COST_S")
		m.utility.co("CAR_CO *1",3,"B_COST_S")
		m.reprovision()
		m.setUp()
		mm = NormalMixedModel(m, ['B_TIME_S','B_COST_S'], ndraws=10, seed=0)
		v = mm.parameter_values()
		v[-1] = 0.01
		v[-2] = 0.01
		v[-3] = 0.01
		mm.parameter_values(v)
		self.assertNearlyEqual(-64.45886575007995, mm.loglike(), sigfigs=7)
		ag = mm.d_loglike()
		fdg = scipy.optimize.approx_fprime(mm.parameter_values(), mm.loglike, mm.parameter_values()*1e-4)
		for g1,g2 in zip(ag,fdg):
			self.assertNearlyEqual(g1,g2, sigfigs=3)


	def test_metamodel(self):

		from ..metamodel import MetaModel

		db = DB.Example('ITINERARY', shared=True)

		casefilter = ""
		db.queries.idco_query = "SELECT distinct(casenum) AS caseid, 1 as weight FROM data_ca "+casefilter
		db.queries.idca_query = "SELECT casenum AS caseid, itinerarycode AS altid, * FROM data_ca "+casefilter

		dow_set = {1,0}
		type_set = {'OW','OB','IB'}

		common_vars = [
			"carrier=2",
			"carrier=3",
			"carrier=4",
			"carrier>=5",
			"aver_fare_hy",
			"aver_fare_ly",
			"itin_num_cnxs",
			"itin_num_directs",
		]

		segmented_vars = [
			"sin2pi",
			"sin4pi",
			"sin6pi",
			"cos2pi",
			"cos4pi",
			"cos6pi",
		]

		consolidated_m = Model()

		for var in common_vars:
			consolidated_m.parameter(var)
			consolidated_m.utility.ca(var)

		meta = MetaModel()

		for segment_desciptor in itertools.product(dow_set, type_set):
			dx = DB.NewConnection(db)
			casefilter = " WHERE dow=={0} AND direction=='{1}'".format(*segment_desciptor)
			dx.queries = qry = QuerySetTwoTable(dx)
			qry.idco_query = "SELECT distinct casenum AS caseid, dow, direction FROM data_ca "+casefilter
			qry.idca_query = "SELECT casenum AS caseid, itinerarycode AS altid, * FROM data_ca "+casefilter
			qry.alts_query = "SELECT * FROM itinerarycodes "
			qry.choice = 'pax_count'
			qry.avail = '1'
			qry.weight = '1'

			m = meta.sub_model[segment_desciptor] = Model(dx)
			m.option.calc_std_errors = False
			m.option.calc_null_likelihood = False

			m.db.refresh_queries()
			
			if m.db.nCases()==0:
				meta.sub_ncases[segment_desciptor] = 0
				meta.sub_weight[segment_desciptor] = 0
				continue
			
			for var in common_vars:
				m.parameter(var)
				m.utility.ca(var)
				meta.parameter(var)

			for var in segmented_vars:
				built_par = var+"_{}_{}".format(*segment_desciptor)
				m.parameter(built_par)
				consolidated_m.parameter(built_par)
				m.utility.ca(var, built_par)
				meta.parameter(built_par)
				consolidated_m.utility(var+"*(dow={0})*(direction=='{1}')".format(*segment_desciptor), built_par)

			m.option.idca_avail_ratio_floor = 0
			m.setUp()
			m.weight_choice_rebalance()
			meta.sub_weight[segment_desciptor] = partwgt = m.Data("Weight").sum()
			meta.total_weight += partwgt
			meta.sub_ncases[segment_desciptor] = partncase = m.nCases()
			meta.total_ncases += partncase

		consolidated_m.db = db
		consolidated_m.option.idca_avail_ratio_floor = 0
		consolidated_m.setUp()
		consolidated_m.provision(idca_avail_ratio_floor=0)

		rr = None

		self.assertNearlyEqual(consolidated_m.loglike_null(), meta.loglike_null())

		meta.option.weight_choice_rebalance = False
		meta.option.calc_std_errors = False

		meta.parameter_array[:] = 0.0
		consolidated_m.parameter_array[:] = 0.0
		d1 = meta.d_loglike()
		d2 = consolidated_m.d_loglike()
		for dd1,dd2 in zip(d1,d2):
			self.assertNearlyEqual(dd1,dd2)

		meta.parameter_array[:] = 0.01
		consolidated_m.parameter_array[:] = 0.01

		self.assertNearlyEqual(consolidated_m.loglike(), meta.loglike())

		d1 = meta.d_loglike()
		d2 = consolidated_m.d_loglike()
		for dd1,dd2 in zip(d1,d2):
			self.assertNearlyEqual(dd1,dd2)

