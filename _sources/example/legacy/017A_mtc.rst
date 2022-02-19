.. currentmodule:: larch

===========================================================
17A: MTC MNL Mode Choice, Segmented for 1 or fewer cars
===========================================================

.. testsetup:: *

	import larch
	import larch.examples
	import pandas
	pandas.set_option('display.max_columns',999)
	pandas.set_option('expand_frame_repr',False)
	pandas.set_option('display.precision',4)
	larch._doctest_mode_ = True

Market segmentation can be used to determine whether the impact of other variables is
different among population groups. The most common approach to market segmentation is
for the analyst to consider sample segments which are mutually exclusive and
collectively exhaustive (that is, each case is included in one and only one segment).
Models are estimated for the sample associated with each segment and compared to the
pooled model (all segments represented by a single model) to determine if there are
statistically significant and important differences among the market segments.

Model 17A segments the market by automobile ownership for households that have one or
fewer cars. It is appealing to include a distinct segment for households with no cars
since the mode choice behavior of this segment is very different from the rest of the
population due to their dependence on non-automobile modes. However, the size of this
segment in the dataset is too small, so it is joined with the one car group. (`pp. 129-133 <http://www.caee.utexas.edu/prof/Bhat/COURSES/LM_Draft_060131Final-060630.pdf>`_)

.. testcode::

	d = larch.examples.MTC()
	m = larch.Model(dataservice=d)

	m.dataservice = m.dataservice.selector_co("numveh <= 1")

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
	-1049.279...

	>>> print(m.pfo()[['value','std_err','t_stat','robust_std_err','robust_t_stat']])
	                                 value  std_err   t_stat  robust_std_err  robust_t_stat
	Category   Parameter
	CostbyInc  costbyincome        -0.0227   0.0138  -1.6408          0.0172        -1.3178
	TravelTime motorized_time      -0.0211   0.0060  -3.4888          0.0063        -3.3546
	           nonmotorized_time   -0.0440   0.0081  -5.4261          0.0086        -5.1318
	           motorized_ovtbydist -0.1131   0.0260  -4.3588          0.0355        -3.1862
	Household  hhinc#4             -0.0064   0.0036  -1.8066          0.0036        -1.7838
	           hhinc#5             -0.0116   0.0095  -1.2272          0.0086        -1.3496
	           hhinc#6             -0.0120   0.0060  -2.0084          0.0057        -2.1147
	           vehbywrk_BIKE       -2.6641   0.6620  -4.0240          0.6861        -3.8831
	           vehbywrk_SR         -3.0145   0.3488  -8.6438          0.3532        -8.5357
	           vehbywrk_TRANSIT    -3.9632   0.3758 -10.5458          0.3829       -10.3497
	           vehbywrk_WALK       -3.3420   0.4446  -7.5166          0.4800        -6.9621
	Zonal      wkcbd_BIKE           0.3958   0.5370   0.7371          0.5601         0.7067
	           wkcbd_SR2            0.3724   0.2410   1.5454          0.2383         1.5627
	           wkcbd_SR3            0.2294   0.4065   0.5644          0.4235         0.5418
	           wkcbd_TRANSIT        1.1065   0.2593   4.2667          0.2541         4.3548
	           wkcbd_WALK           0.0297   0.3505   0.0847          0.3646         0.0814
	           wkempdenBIKE         0.0015   0.0019   0.8015          0.0018         0.8151
	           wkempdenSR2          0.0020   0.0007   2.8046          0.0008         2.5928
	           wkempdenSR3          0.0035   0.0009   3.8880          0.0010         3.4788
	           wkempdenTRANSIT      0.0032   0.0007   4.7292          0.0008         4.1510
	           wkempdenWALK         0.0038   0.0010   3.9143          0.0010         3.8821
	ASCs       ASC_BIKE             0.9765   0.7023   1.3905          0.7043         1.3865
	           ASC_SR2              0.5929   0.3025   1.9596          0.3187         1.8600
	           ASC_SR3             -0.7852   0.3538  -2.2196          0.3660        -2.1452
	           ASC_TRANSIT          2.2579   0.4442   5.0835          0.4932         4.5781
	           ASC_WALK             2.9070   0.5621   5.1715          0.6133         4.7402
