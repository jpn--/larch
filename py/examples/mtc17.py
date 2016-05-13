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
	return larch.DB.Example('MTC')

def model(d=None): # Define a function to create a data object.
	if d is None:
		d = data()
	
	d.execute("CREATE INDEX IF NOT EXISTS data_co_casenum ON data_co (casenum);")
	d.add_column("data_ca", "costbyincome FLOAT")
	qry1="UPDATE data_ca SET costbyincome = totcost/(SELECT hhinc FROM data_co WHERE data_co.casenum=data_ca.casenum)"
	d.execute(qry1)
	d.add_column("data_ca", "ovtbydist FLOAT")
	qry2="UPDATE data_ca SET ovtbydist = ovtt/(SELECT dist FROM data_co WHERE data_co.casenum=data_ca.casenum)"
	d.execute(qry2)
	d.refresh_queries()

	m = larch.Model(d)

	m.utility.ca("costbyincome")

	m.utility.ca("tottime * (altnum IN (1,2,3,4))", "motorized_time")
	m.utility.ca("tottime * (altnum IN (5,6))", "nonmotorized_time")
	m.utility.ca("ovtbydist * (altnum IN (1,2,3,4))", "motorized_ovtbydist")

	m.utility.co("hhinc",4)
	m.utility.co("hhinc",5)
	m.utility.co("hhinc",6)
	
	m.parameter("vehbywrk_SR")
	m.alias("vehbywrk_SR2","vehbywrk_SR",1.0)
	m.alias("vehbywrk_SR3+","vehbywrk_SR",1.0)

	for a,name in m.db.alternatives()[1:]:
		m.utility.co("vehbywrk",a,"vehbywrk_"+name)

	for a,name in m.db.alternatives()[1:]:
		m.utility.co("wkccbd+wknccbd",a,"wkcbd_"+name)

	for a,name in m.db.alternatives()[1:]:
		m.utility.co("wkempden",a,"wkempden_"+name)

	for a,name in m.db.alternatives()[1:]:
		m.utility.co("1",a,"ASC_"+name)

	m.option.calc_std_errors = True
	return m

############################# END OF EXAMPLE FILE ##############################

