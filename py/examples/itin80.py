################################################################################
#
#  Copyright 2007-2016 Jeffrey Newman.
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
################################################################################
#
#  This example file shows commands needed to replicate the models given in 
#  the Field Guide to Discrete Choice Models.
#
################################################################################

from .. import larch

def data():
	# Define which data set to use with this example.
	return larch.DB.Example('ITINERARY')

def model(d=None): # Define a function to create a data object.
	if d is None: d = data()
	m = larch.Model(d)

	from enum import Enum

	class levels_of_service(Enum):
			nonstop = 1
			withstop = 0

	class carriers(Enum):
			Robin = 1
			Cardinal = 2
			Bluejay = 3
			Heron = 4
			Other = 5

	from larch.util.numbering import numbering_system
	ns = numbering_system(levels_of_service, carriers)

	vars = [
			"carrier=2",
			"carrier=3",
			"carrier=4",
			"carrier>=5",
			"aver_fare_hy",
			"aver_fare_ly",
			"itin_num_cnxs",
			"itin_num_directs",
	]
	for var in vars:
		m.utility.ca(var)
	
	alts = d.alternative_codes()

	for los in levels_of_service:
		los_nest = m.new_nest(los.name, param_name="mu_los1", parent=m.root_id)
		for carrier in carriers:
				carrier_nest = m.new_nest(los.name+carrier.name, param_name="mu_carrier1", parent=los_nest)
				for a in alts:
						if ns.code_matches_attributes(a, los, carrier):
								m.link(carrier_nest, a)

	for carrier in carriers:
		carrier_nest = m.new_nest(carrier.name, param_name="mu_carrier2", parent=m.root_id)
		for los in levels_of_service:
				los_nest = m.new_nest(carrier.name+los.name, param_name="mu_los2", parent=carrier_nest)
				for a in alts:
						if ns.code_matches_attributes(a, los, carrier):
								m.link(los_nest, a)
								m.link[los_nest, a](data='1',param='PHI')

	filter = 'casenum < 20000'
	d.queries.idca_build(filter=filter)
	d.queries.idco_build(filter=filter)
	m.option.calc_std_errors = False
	m.option.enforce_constraints = True
	return m

############################# END OF EXAMPLE FILE ##############################

