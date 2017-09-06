
import larch
import numpy

m = larch.Model.Example(22)
m.maximize_loglike()

d_u_ca = m.data.UtilityCA.copy()
d_u_co = m.data.UtilityCO.copy()

coef_u_ca = m.Coef("UtilityCA").copy()
coef_u_co = m.Coef("UtilityCO").copy()



from larch4.parameter_collection import ParameterCollection
from larch4.data_collection import DataCollection

d = DataCollection(
	m.df.caseids().copy(), m.df.alternative_codes(),
	m.needs()['UtilityCA'].get_variables(), m.needs()['UtilityCO'].get_variables(),
	m.data.UtilityCA.copy(), m.data.UtilityCO.copy(), m.data.Avail.squeeze().copy())


coef_ca = m.Coef("UtilityCA").squeeze().copy()
coef_co = m.Coef("UtilityCO").squeeze().copy()

p = ParameterCollection(
	m.df.alternative_codes(),
	coef_ca, m.needs()['UtilityCA'].get_variables(),
	coef_co, m.needs()['UtilityCO'].get_variables())



from larch4.nesting.tree import NestingTree

t = NestingTree()
t.add_nodes([1,2,3,4,5,6])

t.add_node(7, children=(1,2,3,4), parameter='mu_motor')
t.add_node(8, children=(5,6), parameter='mu_nonmotor')

print('t.topological_sorted_no_elementals', t.topological_sorted_no_elementals)

print('t.standard_sort', t.standard_sort)

u = numpy.zeros([len(d._caseindex), len(d._altindex)])
print(u.shape)
d._calculate_exp_utility_elemental(p, u)
print(u.shape)

u1 = numpy.zeros([len(d._caseindex), len(t)-len(d._altindex)])


from larch4.nesting.nl_utility import exp_util_of_nests
# from larch4.nesting import exp_util_of_nests_threaded

exp_util_of_nests(u,u1,t,p)

print("numpy.exp(m.work.utility).sum(0)")
print(numpy.exp(m.work.utility).sum(0))

print('u.sum[0]\n',u.sum(0))
print('u1.sum[0]\n',u1.sum(0))

# 'mu_motor', value=0.7257824244230557
# ModelParameter('mu_nonmotor', value=0.7689340538871795)


# %timeit exp_util_of_nests_threaded(u,u1,t,p,7)
# %timeit exp_util_of_nests(u,u1,t,p)