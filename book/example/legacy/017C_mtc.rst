.. currentmodule:: larch

===========================================================
17C: MTC MNL Mode Choice, Segmented for Males
===========================================================

.. testsetup:: *

	import larch
	import larch.examples
	import pandas
	pandas.set_option('display.max_columns',999)
	pandas.set_option('expand_frame_repr',False)
	pandas.set_option('display.precision',4)
	larch._doctest_mode_ = True

Model 17C segments the market by gender for males. (`pp. 135 <http://www.caee.utexas.edu/prof/Bhat/COURSES/LM_Draft_060131Final-060630.pdf>`_)

.. testcode::

	d = larch.examples.MTC()
	m = larch.Model(dataservice=d)

	m.dataservice = m.dataservice.selector_co("femdum == 0")

.. testcode::

	m.availability_var = '_avail_'
	m.choice_ca_var = '_choice_'

.. testcode::

	from larch.roles import P, X, PX
	m.utility_ca = (
		+ P("costbyincome") * X("totcost/hhinc")
		+ P("motorized_time") * X("(altnum <= 4) * tottime")
		+ P("nonmotorized_time") * X("(altnum > 4) * tottime")
		+ P("motorized_ovtbydist") * X("(altnum <=4) * ovtt/dist")
		)

.. testcode::

	for a in [4,5,6]:
		m.utility_co[a] = P("hhinc#{}".format(a)) * X("hhinc")

	for a,name in m.dataservice.alternative_pairs[1:3]:
		m.utility_co[a] += (
			+ P("vehbywrk_SR") * X("vehbywrk")
			+ P("wkcbd_"+name) * X("wkccbd+wknccbd")
			+ P("wkempden"+name) * X("wkempden")
			+ P("ASC_"+name)
			)

	for a,name in m.dataservice.alternative_pairs[3:]:
		m.utility_co[a] += (
			+ P("vehbywrk_"+name) * X("vehbywrk")
			+ P("wkcbd_"+name) * X("wkccbd + wknccbd")
			+ P("wkempden"+name) * X("wkempden")
			+ P("ASC_"+name)
			)

.. testcode::

	m.ordering = (
		("CostbyInc","costbyincome",),
		("TravelTime",".*time.*",".*dist.*", ),
		("Household","hhinc.*","vehbywrk.*", ),
		("Zonal","wkcbd.*","wkempden.*", ),
		("ASCs","ASC.*", ),
	)

.. doctest::
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE, +REPORT_NDIFF

	>>> m.load_data()
	>>> m.maximize_loglike()
	┣ ...Optimization terminated...
	>>> m.calculate_parameter_covariance()
	>>> m.loglike()
	-1889.783...

	>>> print(m.pfo()[['value','std_err','t_stat','robust_std_err','robust_t_stat']])
	                                 value  std_err   t_stat  robust_std_err  robust_t_stat
	Category   Parameter
	CostbyInc  costbyincome        -0.0640   0.0147  -4.3596          0.0203        -3.1488
	TravelTime motorized_time      -0.0195   0.0053  -3.6913          0.0052        -3.7440
	           nonmotorized_time   -0.0244   0.0074  -3.2718          0.0067        -3.6522
	           motorized_ovtbydist -0.1866   0.0312  -5.9763          0.0383        -4.8744
	Household  hhinc#4             -0.0021   0.0027  -0.7758          0.0028        -0.7457
	           hhinc#5             -0.0014   0.0057  -0.2454          0.0061        -0.2259
	           hhinc#6             -0.0050   0.0046  -1.0988          0.0051        -0.9771
	           vehbywrk_BIKE       -0.9900   0.3210  -3.0847          0.3093        -3.2014
	           vehbywrk_SR         -0.2104   0.0756  -2.7813          0.0836        -2.5157
	           vehbywrk_TRANSIT    -0.8331   0.1571  -5.3039          0.1713        -4.8622
	           vehbywrk_WALK       -0.6100   0.2234  -2.7305          0.2713        -2.2485
	Zonal      wkcbd_BIKE           0.3082   0.4669   0.6601          0.4803         0.6417
	           wkcbd_SR2            0.0279   0.1742   0.1599          0.1767         0.1576
	           wkcbd_SR3            1.4238   0.2364   6.0219          0.2380         5.9834
	           wkcbd_TRANSIT        1.1967   0.2413   4.9591          0.2278         5.2522
	           wkcbd_WALK           0.2238   0.3747   0.5973          0.3969         0.5640
	           wkempdenBIKE         0.0005   0.0015   0.3481          0.0015         0.3378
	           wkempdenSR2          0.0009   0.0005   1.7463          0.0006         1.6229
	           wkempdenSR3          0.0006   0.0006   0.9555          0.0006         0.9522
	           wkempdenTRANSIT      0.0025   0.0005   5.4708          0.0005         5.2918
	           wkempdenWALK         0.0013   0.0010   1.2415          0.0010         1.2577
	ASCs       ASC_BIKE            -1.9311   0.5343  -3.6140          0.5951        -3.2452
	           ASC_SR2             -1.9123   0.1347 -14.1951          0.1471       -12.9975
	           ASC_SR3             -3.5514   0.1983 -17.9090          0.1960       -18.1173
	           ASC_TRANSIT         -0.8651   0.3534  -2.4477          0.3620        -2.3902
	           ASC_WALK            -1.2275   0.5101  -2.4063          0.4521        -2.7148
