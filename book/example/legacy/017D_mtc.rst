.. currentmodule:: larch

===========================================================
17D: MTC MNL Mode Choice, Segmented for Females
===========================================================

.. testsetup:: *

	import larch
	import larch.examples
	import pandas
	pandas.set_option('display.max_columns',999)
	pandas.set_option('expand_frame_repr',False)
	pandas.set_option('display.precision',4)
	larch._doctest_mode_ = True

Model 17D segments the market by gender for females. (`pp. 135 <http://www.caee.utexas.edu/prof/Bhat/COURSES/LM_Draft_060131Final-060630.pdf>`_)

.. testcode::

	d = larch.examples.MTC()
	m = larch.Model(dataservice=d)

	m.dataservice = m.dataservice.selector_co("femdum == 1")

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
	┣ ...Optimization terminated successfully...
	>>> m.calculate_parameter_covariance()
	>>> m.loglike()
	-1511.319...

The parameters can be found in the `pf` DataFrame::

	>>> print(m.pfo()[['value','std_err','t_stat','robust_std_err','robust_t_stat']])
	                                value    std_err  t_stat  robust_std_err  robust_t_stat
	Category   Parameter
	CostbyInc  costbyincome        -0.044  1.486e-02  -2.939       1.723e-02         -2.535
	TravelTime motorized_time      -0.019  5.592e-03  -3.413       5.865e-03         -3.254
	           nonmotorized_time   -0.070  9.383e-03  -7.494       9.972e-03         -7.051
	           motorized_ovtbydist -0.090  2.530e-02  -3.556       3.084e-02         -2.917
	Household  hhinc#4             -0.009  2.997e-03  -2.966       3.076e-03         -2.890
	           hhinc#5             -0.038  1.392e-02  -2.713       1.681e-02         -2.247
	           hhinc#6             -0.005  4.455e-03  -1.102       4.673e-03         -1.051
	           vehbywrk_BIKE       -0.056  4.021e-01  -0.140       5.337e-01         -0.105
	           vehbywrk_SR         -0.607  1.358e-01  -4.466       1.468e-01         -4.132
	           vehbywrk_TRANSIT    -1.173  1.892e-01  -6.196       2.326e-01         -5.042
	           vehbywrk_WALK       -0.904  2.700e-01  -3.348       3.242e-01         -2.789
	Zonal      wkcbd_BIKE           1.039  5.944e-01   1.748       5.566e-01          1.866
	           wkcbd_SR2            0.454  1.809e-01   2.512       1.831e-01          2.482
	           wkcbd_SR3            0.377  3.364e-01   1.120       3.252e-01          1.159
	           wkcbd_TRANSIT        1.380  2.320e-01   5.951       2.253e-01          6.127
	           wkcbd_WALK          -0.008  3.496e-01  -0.023       3.444e-01         -0.023
	           wkempdenBIKE         0.004  2.184e-03   1.884       1.772e-03          2.322
	           wkempdenSR2          0.003  6.605e-04   4.525       7.685e-04          3.889
	           wkempdenSR3          0.005  7.723e-04   6.641       8.680e-04          5.909
	           wkempdenTRANSIT      0.005  6.527e-04   6.987       7.813e-04          5.837
	           wkempdenWALK         0.005  1.163e-03   4.695       1.204e-03          4.533
	ASCs       ASC_BIKE            -1.153  7.664e-01  -1.504       7.275e-01         -1.584
	           ASC_SR2             -1.564  1.820e-01  -8.598       1.921e-01         -8.142
	           ASC_SR3             -3.199  2.452e-01 -13.047       2.613e-01        -12.246
	           ASC_TRANSIT         -0.477  3.594e-01  -1.327       4.087e-01         -1.167
	           ASC_WALK             1.305  5.058e-01   2.580       5.433e-01          2.402
