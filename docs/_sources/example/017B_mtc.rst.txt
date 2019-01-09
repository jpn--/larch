.. currentmodule:: larch

===========================================================
17B: MTC MNL Mode Choice, Segmented for 2 or more cars
===========================================================

.. testsetup:: *

	import larch
	import larch.examples
	import pandas
	pandas.set_option('display.max_columns',999)
	pandas.set_option('expand_frame_repr',False)
	pandas.set_option('precision',4)
	larch._doctest_mode_ = True

Model 17B segments the market by automobile ownership for households
that have more than one car. (`pp. 133 <http://www.caee.utexas.edu/prof/Bhat/COURSES/LM_Draft_060131Final-060630.pdf>`_)

.. testcode::
	
	d = larch.examples.MTC()
	m = larch.Model(dataservice=d)
	
	m.selector = "numveh >= 2"
	
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
	-2296.667...
	
	>>> print(m.pfo()[['value','std err','t stat','robust std err','robust t stat']])
	                                 value  std err   t stat  robust std err  robust t stat
	Category   Parameter
	CostbyInc  costbyincome        -0.0981   0.0162  -6.0686          0.0192        -5.1154
	TravelTime motorized_time      -0.0187   0.0051  -3.6422          0.0052        -3.6065
	           nonmotorized_time   -0.0450   0.0087  -5.1639          0.0083        -5.4290
	           motorized_ovtbydist -0.1944   0.0331  -5.8770          0.0323        -6.0176
	Household  hhinc#4              0.0004   0.0026   0.1385          0.0027         0.1368
	           hhinc#5             -0.0019   0.0065  -0.2960          0.0080        -0.2409
	           hhinc#6              0.0007   0.0041   0.1646          0.0043         0.1545
	           vehbywrk_BIKE       -0.1906   0.3285  -0.5801          0.3786        -0.5033
	           vehbywrk_SR         -0.2413   0.0737  -3.2745          0.0792        -3.0452
	           vehbywrk_TRANSIT    -0.2399   0.1359  -1.7656          0.1268        -1.8919
	           vehbywrk_WALK       -0.0975   0.2111  -0.4618          0.2186        -0.4460
	Zonal      wkcbd_BIKE           0.4867   0.5032   0.9672          0.4945         0.9842
	           wkcbd_SR2            0.1627   0.1487   1.0944          0.1507         1.0798
	           wkcbd_SR3            1.3301   0.2209   6.0202          0.2177         6.1110
	           wkcbd_TRANSIT        1.2785   0.2444   5.2322          0.2377         5.3782
	           wkcbd_WALK           0.1114   0.3872   0.2876          0.4154         0.2681
	           wkempdenBIKE         0.0016   0.0017   0.9573          0.0016         0.9873
	           wkempdenSR2          0.0011   0.0005   2.2213          0.0005         2.1166
	           wkempdenSR3          0.0013   0.0005   2.4516          0.0005         2.5022
	           wkempdenTRANSIT      0.0029   0.0004   6.4238          0.0004         6.4486
	           wkempdenWALK        -0.0009   0.0021  -0.4168          0.0021        -0.4156
	ASCs       ASC_BIKE            -3.2179   0.7337  -4.3860          0.8648        -3.7208
	           ASC_SR2             -1.9786   0.1286 -15.3817          0.1366       -14.4879
	           ASC_SR3             -3.7196   0.1857 -20.0259          0.1816       -20.4842
	           ASC_TRANSIT         -2.1627   0.3838  -5.6351          0.3729        -5.7992
	           ASC_WALK            -1.5346   0.5749  -2.6694          0.5311        -2.8898