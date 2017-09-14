
import larch

from ..nesting.tree import NestingTree
from ..nesting.nl_utility import exp_util_of_nests
from ..nesting.nl_prob import conditional_logprob_from_tree, elemental_logprob_from_conditional_logprob
import numpy

def test_utility_spec_changes():
	from ..parameter_collection import ParameterCollection
	from ..data_collection import DataCollection
	from ..workspace_collection import WorkspaceCollection
	from ..roles import P, X

	param_names = ['costbyincome',
	               'motorized_time',
	               'nonmotorized_time',
	               'motorized_ovtbydist',
	               'vehbywrk_SR',
	               'mu_motor',
	               'mu_nonmotor',
	               'wkcbd_SR2',
	               'wkempden_SR2',
	               'ASC_SR2',
	               'wkcbd_SR3+',
	               'wkempden_SR3+',
	               'ASC_SR3+',
	               'hhinc#4',
	               'vehbywrk_Tran',
	               'wkcbd_Tran',
	               'wkempden_Tran',
	               'ASC_Tran',
	               'hhinc#5',
	               'vehbywrk_Bike',
	               'wkcbd_Bike',
	               'wkempden_Bike',
	               'ASC_Bike',
	               'hhinc#6',
	               'vehbywrk_Walk',
	               'wkcbd_Walk',
	               'wkempden_Walk',
	               'ASC_Walk']

	altcodes = (1, 2, 3, 4, 5, 6)

	p = ParameterCollection(param_names,
	                        altcodes)

	p.utility_ca = (
		+ (P('costbyincome') * X('totcost/hhinc'))
		+ (P('motorized_time') * X('tottime * (altnum <= 4)'))
		+ (P('nonmotorized_time') * X('tottime * (altnum >= 5)'))
		+ (P('motorized_ovtbydist') * X('ovtt/dist * (altnum <= 4)'))
	)

	u_co = {
		2:((P('vehbywrk_SR2') * X('vehbywrk'))+(P('wkcbd_SR2') * X('wkccbd+wknccbd'))+(P('wkempden_SR2') * X('wkempden'))+(P('ASC_SR2') )),
		3:((P('vehbywrk_SR3+') * X('vehbywrk'))+(P('wkcbd_SR3+') * X('wkccbd+wknccbd'))+(P('wkempden_SR3+') * X('wkempden'))+(P('ASC_SR3+') )),
		4:((P('hhinc#4') * X('hhinc'))+(P('vehbywrk_Tran') * X('vehbywrk'))+(P('wkcbd_Tran') * X('wkccbd+wknccbd'))+(P('wkempden_Tran') * X('wkempden'))+(P('ASC_Tran') * X('1'))),
		5:((P('hhinc#5') * X('hhinc'))+(P('vehbywrk_Bike') * X('vehbywrk'))+(P('wkcbd_Bike') * X('wkccbd+wknccbd'))+(P('wkempden_Bike') * X('wkempden'))+(P('ASC_Bike') * X('1'))),
		6:((P('hhinc#6') * X('hhinc'))+(P('vehbywrk_Walk') * X('vehbywrk'))+(P('wkcbd_Walk') * X('wkccbd+wknccbd'))+(P('wkempden_Walk') * X('wkempden'))+(P('ASC_Walk') * X('1')))
	}

	p.utility_co = u_co

	param_values = (-0.03861527921549844,
					 -0.014523038624767998,
					 -0.046215369431426574,
					 -0.11379697567750172,
					 -0.2256677903743352,
					 0.7257824243960934,
					 0.768934053868761,
					 0.19313965705114092,
					 0.0011489703923458934,
					 -1.3250178603913265,
					 0.7809570669175268,
					 0.0016377717591202284,
					 -2.5055361149071476,
					 -0.003931169283310062,
					 -0.7070553074051729,
					 0.9213037533865539,
					 0.002236618465113466,
					 -0.4035506229859652,
					 -0.010045677860597766,
					 -0.7347879705726603,
					 0.40770600782114,
					 0.0016743955255113922,
					 -1.2012147989034374,
					 -0.0062076142199297265,
					 -0.7638904490769256,
					 0.11414028210681003,
					 0.0021704169426275174,
					 0.34548388394786805)

	for p_name, p_value in zip(param_names, param_values):
		p.set_value(p_name, p_value)

	p.set_value('vehbywrk_SR3+', p.get_value('vehbywrk_SR'))
	p.set_value('vehbywrk_SR2', p.get_value('vehbywrk_SR'))

	p.unmangle()

	dt = larch.DT.Example()

	d = DataCollection(
		dt.caseids().copy(), altcodes,
		p.utility_ca_vars,
		p.utility_co_vars,
	)

	d.load_data(dt)

	t = NestingTree()
	t.add_nodes([1, 2, 3, 4, 5, 6])

	t.add_node(7, children=(1, 2, 3, 4), parameter='mu_motor')
	t.add_node(8, children=(5, 6), parameter='mu_nonmotor')

	assert( not p._mangled )

	p.utility_ca[0] *= 2
	assert( p._mangled )
	p.unmangle()
	assert( not p._mangled )

	try:
		p.utility_co[7] += (P('hhinc#7') * X('hhinc'))
	except KeyError:
		pass
	else:
		raise AssertionError

	assert( not p._mangled )

	p.utility_co[1] += (P('hhinc#1') * X('hhinc'))
	assert( p._mangled )
	p.unmangle()
	assert( not p._mangled )

	assert( len(p.utility_co)==6 )

	work = WorkspaceCollection(d, p, t)

	d._calculate_exp_utility_elemental(p, work.util_elementals)


	exp_util_of_nests(work.util_elementals, work.util_nests, t, p)


	conditional_logprob_from_tree(
		work.util_elementals,
		work.util_nests,
		t,
		p,
		work.log_conditional_prob
	)

	elemental_logprob_from_conditional_logprob(
		work.log_conditional_prob,
		t,
		work.log_prob

	)

	assert( numpy.isclose(d._calculate_log_like(work.log_prob), -3441.672527074124 ))
