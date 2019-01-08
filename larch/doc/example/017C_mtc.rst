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
	pandas.set_option('precision',4)
	larch._doctest_mode_ = True

Model 17C segments the market by gender for males. (`pp. 135 <http://www.caee.utexas.edu/prof/Bhat/COURSES/LM_Draft_060131Final-060630.pdf>`_)

.. testcode::
	
	d = larch.examples.MTC()
	m = larch.Model(dataservice=d)
	
	m.selector = "femdum == 0"
	
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
	
	for a,name in m.dataservice.alternatives[1:3]:
		m.utility_co[a] += (
			+ P("vehbywrk_SR") * X("vehbywrk")
			+ P("wkcbd_"+name) * X("wkccbd+wknccbd")
			+ P("wkempden"+name) * X("wkempden")
			+ P("ASC_"+name)
			)
			
	for a,name in m.dataservice.alternatives[3:]:	
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
	:options: +ELLIPSIS, +NORMALIZE_WHITESPACE

	>>> m.load_data()
	>>> m.maximize_loglike()
	┣ ...Optimization terminated successfully...
	>>> m.calculate_parameter_covariance()
	>>> m.loglike()
	-1889.783...
	
	>>> print(m.pfo()[['value','std err','t stat','robust std err','robust t stat']])
	                                 value  std err   t stat  robust std err  robust t stat
	Category   Parameter
	CostbyInc  costbyincome        -0.0639   0.0147  -4.3597          0.0203        -3.1489
	TravelTime motorized_time      -0.0195   0.0053  -3.6928          0.0052        -3.7459
	           nonmotorized_time   -0.0244   0.0074  -3.2820          0.0067        -3.6624
	           motorized_ovtbydist -0.1865   0.0312  -5.9725          0.0383        -4.8725
	Household  hhinc#4             -0.0021   0.0027  -0.7762          0.0028        -0.7461
	           hhinc#5             -0.0014   0.0057  -0.2439          0.0061        -0.2246
	           hhinc#6             -0.0050   0.0046  -1.1006          0.0051        -0.9787
	           vehbywrk_BIKE       -0.9920   0.3212  -3.0887          0.3097        -3.2033
	           vehbywrk_SR         -0.2104   0.0756  -2.7823          0.0836        -2.5165
	           vehbywrk_TRANSIT    -0.8329   0.1570  -5.3033          0.1713        -4.8625
	           vehbywrk_WALK       -0.6112   0.2235  -2.7351          0.2714        -2.2518
	Zonal      wkcbd_BIKE           0.3110   0.4668   0.6662          0.4801         0.6477
	           wkcbd_SR2            0.0278   0.1742   0.1599          0.1767         0.1575
	           wkcbd_SR3            1.4238   0.2365   6.0215          0.2380         5.9828
	           wkcbd_TRANSIT        1.1965   0.2413   4.9594          0.2278         5.2532
	           wkcbd_WALK           0.2242   0.3747   0.5983          0.3969         0.5648
	           wkempdenBIKE         0.0005   0.0015   0.3426          0.0015         0.3321
	           wkempdenSR2          0.0009   0.0005   1.7457          0.0006         1.6223
	           wkempdenSR3          0.0006   0.0006   0.9555          0.0006         0.9523
	           wkempdenTRANSIT      0.0025   0.0005   5.4703          0.0005         5.2917
	           wkempdenWALK         0.0013   0.0010   1.2349          0.0010         1.2505
	ASCs       ASC_BIKE            -1.9277   0.5344  -3.6069          0.5953        -3.2383
	           ASC_SR2             -1.9121   0.1347 -14.1932          0.1471       -12.9958
	           ASC_SR3             -3.5513   0.1983 -17.9079          0.1960       -18.1159
	           ASC_TRANSIT         -0.8651   0.3534  -2.4479          0.3619        -2.3907
	           ASC_WALK            -1.2209   0.5101  -2.3932          0.4521        -2.7003