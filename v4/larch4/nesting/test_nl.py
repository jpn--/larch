
import larch

from .tree import NestingTree
from .nl_utility import exp_util_of_nests
from .nl_prob import conditional_logprob_from_tree, elemental_logprob_from_conditional_logprob
from ..model import Model
import numpy
from larch4.math.optimize import approx_fprime


def nl_straw_man_model_1():
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

	dt = larch.DT.Example()

	t = NestingTree()
	t.add_nodes([1, 2, 3, 4, 5, 6])
	t.add_node(7, children=(1, 2, 3, 4), parameter='mu_motor')
	t.add_node(8, children=(5, 6), parameter='mu_nonmotor')

	p = Model(parameters=param_names, alts=altcodes, datasource=dt, graph=t)

	from ..roles import P,X

	p.utility_ca = (
		+ (P('costbyincome') * X('totcost/hhinc'))
		+ (P('motorized_time') * X('tottime * (altnum <= 4)'))
		+ (P('nonmotorized_time') * X('tottime * (altnum >= 5)'))
		+ (P('motorized_ovtbydist') * X('ovtt/dist * (altnum <= 4)'))
	)

	u_co = {
		2:((P('vehbywrk_SR2') * X('vehbywrk'))+(P('wkcbd_SR2') * X('wkccbd+wknccbd'))+(P('wkempden_SR2') * X('wkempden'))+(P('ASC_SR2') * X('1'))),
		3:((P('vehbywrk_SR3+') * X('vehbywrk'))+(P('wkcbd_SR3+') * X('wkccbd+wknccbd'))+(P('wkempden_SR3+') * X('wkempden'))+(P('ASC_SR3+') * X('1'))),
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

	p.load_data()
	return p


def test_nl_simple_loglike():
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
		2:((P('vehbywrk_SR2') * X('vehbywrk'))+(P('wkcbd_SR2') * X('wkccbd+wknccbd'))+(P('wkempden_SR2') * X('wkempden'))+(P('ASC_SR2') * X('1'))),
		3:((P('vehbywrk_SR3+') * X('vehbywrk'))+(P('wkcbd_SR3+') * X('wkccbd+wknccbd'))+(P('wkempden_SR3+') * X('wkempden'))+(P('ASC_SR3+') * X('1'))),
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

	work = WorkspaceCollection(d, p, t)

	d._calculate_exp_utility_elemental(p, work.util_elementals)


	exp_util_of_nests(work.util_elementals, work.util_nests, t, p)


	conditional_logprob_from_tree(
		work.util_elementals,
		work.util_nests,
		t,
		p,
		work.log_conditional_prob_dict
	)

	elemental_logprob_from_conditional_logprob(
		work.log_conditional_prob_dict,
		t,
		work.log_prob

	)

	assert( numpy.isclose(d._calculate_log_like(work.log_prob), -3441.672527074124 ))



def test_case_dUtility_dFusedParameters():
	p = nl_straw_man_model_1()
	p.loglike()

	for c in range(0,100,10):

		func = lambda x: p.calculate_utility_values(x)[0][c]
		func2 = lambda x: p.calculate_utility_values(x)[1][c]


		xk = p.frame['value'].values.copy()

		axp = approx_fprime(xk, func, 1e-5)
		axp2 = approx_fprime(xk, func2, 1e-5)

		from .nl_deriv import case_dUtility_dFusedParameters
		from ..linalg.contiguous_group import Blocker

		p.unmangle()

		edge_slot_arrays = p.graph.edge_slot_arrays()
		dU = Blocker([len(p.graph)], [
			p.coef_utility_ca.shape,
			p.coef_utility_co.shape,
			p.coef_quantity_ca.shape,
			p.coef_logsums.shape,
		])

		case_dUtility_dFusedParameters(
			c, # int c,

			p.data.n_cases,  # int n_cases,
			p.data.n_alts,   # int n_elementals,
			len(p.graph) - p.data.n_alts, #  int n_nests,
			p.graph.n_edges, # int n_edges,

			p.work.log_prob,        # double[:,:] log_prob_elementals,  # shape = [cases,alts]
			p.work.util_elementals, # double[:,:] util_elementals,      # shape = [cases,alts]
			p.work.util_nests,      # double[:,:] util_nests,           # shape = [cases,nests]

			p.coef_utility_ca,  # double[:] coef_u_ca,               # shape = (vars_ca)
			p.coef_utility_co,  # double[:,:] coef_u_co,             # shape = (vars_co,alts)
			p.coef_quantity_ca, # double[:] coef_q_ca,               # shape = (qvars_ca)
			p.coef_logsums,     # double[:] coef_mu,                 # shape = (nests)

			dU.outer,     # double[:,:] d_util_meta,           # shape = (nodes,n_meta_coef)  refs same memory as the d_util_coef_* arrays...
			dU.inners[0], # double[:,:] d_util_coef_u_ca,      # shape = (nodes,vars_ca)
			dU.inners[1], # double[:,:,:] d_util_coef_u_co,    # shape = (nodes,vars_co,alts)
			dU.inners[2], # double[:,:] d_util_coef_q_ca,      # shape = (nodes,qvars_ca)
			dU.inners[3], # double[:,:] d_util_coef_mu,        # shape = (nodes,nests)

			p.data._u_ca, # double[:,:,:] data_u_ca,           # shape = (cases,alts,vars_ca)
			p.data._u_co, # double[:,:] data_u_co,             # shape = (cases,vars_co)

			p.work.log_conditional_probability, # double[:,:] conditional_prob,      # shape = (cases,edges)

			edge_slot_arrays[1], # int[:] edge_child_slot,            # shape = (edges)
			edge_slot_arrays[0], # int[:] edge_parent_slot,           # shape = (edges)
			edge_slot_arrays[2], # int[:] first_visit_to_child,       # shape = (edges)
		)

		dUp = p.push_to_parameterlike(dU)

		try:
			assert(numpy.allclose(axp, dUp.T[:,:6], rtol=1e-02, atol=1e-06,))
		except:
			print("Error in Elementals  c=",c)
			for row in range(30):
				if not numpy.allclose(axp[row], dUp.T[row,:6], rtol=1e-02, atol=1e-06,):
					print("ROW",row,"  -- ",p.frame.index[row])
					print(axp[row])
					print(dUp.T[row,:6])
			raise

		try:
			assert(numpy.allclose(axp2, dUp.T[:,6:], rtol=1e-02, atol=1e-06,))
		except:
			print("Error in Nests  c=",c)
			for row in range(30):
				if not numpy.allclose(axp2[row], dUp.T[row,6:], rtol=1e-02, atol=1e-06,):
					print("ROW",row,"  -- ",p.frame.index[row])
					print(axp2[row])
					print(dUp.T[row,6:])
			raise
