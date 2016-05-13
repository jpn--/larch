######################################################### encoding: utf-8 ######
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
#  This example file shows commands needed to replicate the model given in 
#  BIOGEME's example file: 02weight.py
#  http://biogeme.epfl.ch/swissmetro/examples.html
#
################################################################################

from .. import larch

def data(): # Define a function to create a data object.
	# Data object
	#	Larch comes pre-loaded with several free example data sets, including the
	#	SWISSMETRO data set that is used as the example data for BIOGEME.
	#	The SWISSMETRO data set contains an idCO (case only) data table as the master
	#	table. The example data also contains a SQL view that represents the
	#	availability and choice variables in an idCA (case-alt) format.
	swissmetro_alts = {
		1:('Train','TRAIN_AV*(SP!=0)'),
		2:('SM','SM_AV'),
		3:('Car','CAR_AV*(SP!=0)'),
	}
	d = larch.DB.CSV_idco(filename=os.path.join(larch._directory_,"data_warehouse","swissmetro.csv"),
					choice="CHOICE", weight="(1.0*(GROUPid==2)+1.2*(GROUPid==3))*0.8890991",
					tablename="data", savename=None, alts=swissmetro_alts, safety=True)
					
	# Filtering the data
	d.queries.set_idco_query(d.queries.get_idco_query()+" WHERE CHOICE!=0 AND (PURPOSE==1 OR PURPOSE==3)")

	return d # Returns the data object from the data() function.


def model(d=None): # Define a function to create a model object.
	if d is None: d = data()
	m = larch.Model(d)
	m.title = "swissmetro example 02 (simple logit weighted)"
	
	# ModelObject.utility.co(<idCO data column>,<alternative code>,[<parameter name>])
	#	Adds a linear component to the utility of the indicated alternative.
	#	The data column is a string and can be any idCO data column or a pre-calculated
	#	value derived from one or more idCO data columns, or no data columns at all.
	#	Note: there is no need to declare a parameter name seperately from this
	#	command. Default values will be assumed for parameters that are not previously
	#	declared.
	m.utility.co("1",1,"ASC_TRAIN")
	m.utility.co("1",3,"ASC_CAR")
	m.utility.co("TRAIN_TT",1,"B_TIME")
	m.utility.co("SM_TT",2,"B_TIME")
	m.utility.co("CAR_TT",3,"B_TIME")
	m.utility.co("TRAIN_CO*(GA==0)",1,"B_COST")
	m.utility.co("SM_CO*(GA==0)",2,"B_COST")
	m.utility.co("CAR_CO",3,"B_COST")
	
	# ModelObject.option
	#	A structure that defines certain options to be applied when estimating
	#	models.
	m.option.calc_std_errors = True

	return m # Returns the model object from the model() function.

############################# END OF EXAMPLE FILE ##############################
