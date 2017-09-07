#
# import larch
# import numpy
#
# m = larch.Model.Example(22)
# m.maximize_loglike()
#
# d_u_ca = m.data.UtilityCA.copy()
# d_u_co = m.data.UtilityCO.copy()
#
# coef_u_ca = m.Coef("UtilityCA").copy()
# coef_u_co = m.Coef("UtilityCO").copy()
#
#
#
# from larch4.parameter_collection import ParameterCollection
# from larch4.data_collection import DataCollection
# from larch4.workspace_collection import WorkspaceCollection
#
#
# d = DataCollection(
# 	m.df.caseids().copy(), m.df.alternative_codes(),
# 	m.needs()['UtilityCA'].get_variables(), m.needs()['UtilityCO'].get_variables(),
# )
#
# d.load_data(m.df)
#
# coef_ca = m.Coef("UtilityCA").squeeze().copy()
# coef_co = m.Coef("UtilityCO").squeeze().copy()
#
# p = ParameterCollection(
# 	m.df.alternative_codes(),
# 	coef_ca, m.needs()['UtilityCA'].get_variables(),
# 	coef_co, m.needs()['UtilityCO'].get_variables())
#
#
# from larch4.nesting.tree import NestingTree
#
# t = NestingTree()
# t.add_nodes([1,2,3,4,5,6])
#
# t.add_node(7, children=(1,2,3,4), parameter='mu_motor')
# t.add_node(8, children=(5,6), parameter='mu_nonmotor')
#
#
# work = WorkspaceCollection(d,p,t)
#
#
# d._calculate_exp_utility_elemental(p, work.exp_util_elementals)
#
# from larch4.nesting.nl_utility import exp_util_of_nests
#
# exp_util_of_nests(work.exp_util_elementals,work.exp_util_nests,t,p)
#
#
# from larch4.nesting.nl_prob import conditional_logprob_from_tree, elemental_logprob_from_conditional_logprob
#
# # cp = {
# # 	code:numpy.zeros([len(d._caseindex), code_out_degree])
# # 	for code, code_out_degree in t.out_degree_iter()
# # 	if code_out_degree > 0
# # }
#
# conditional_logprob_from_tree(
# 	work.exp_util_elementals,
# 	work.exp_util_nests,
# 		t,
# 		p,
# 		work.log_conditional_prob
# 	)
# print("?"*30, 'Prob?')
# print(m.work.probability[0])
# print(work.log_conditional_prob[0][0])
# print(work.log_conditional_prob[7][0])
# print(work.log_conditional_prob[8][0])
#
#
# print("!"*30, 'Prob!')
#
# elemental_logprob_from_conditional_logprob(
# 	work.log_conditional_prob,
# 	t,
# 	work.log_prob
#
# )
#
#
#
# print("-----LL-----")
# print(d._calculate_log_like(work.log_prob))
# print(m.loglike())
#
# # -3441.6725270750367
# # -3441.6725270750376
#
#
# print('done')
#
# # %timeit exp_util_of_nests_threaded(u,u1,t,p,7)
# # %timeit exp_util_of_nests(u,u1,t,p)
# # %prun exp_util_of_nests(u,u1,t,p)
#
# # %prun m.loglike(cached=False)
#
#
# # %timeit d._calculate_log_like(work.log_prob)
#
#

from larch.roles import P,X
from larch4.parameter_collection import ParameterCollection

names = ['aa',]

uca = P.aa * X.aaa + P.bb * X.bbb

uco = {
	1: P.co1 * X.coo1,
	2: P.co2 * X.coo2,
}

p = ParameterCollection(names, utility_ca=uca, utility_co=uco, altindex=[1,2,3,4,5,6])



