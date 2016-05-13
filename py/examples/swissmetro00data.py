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
#  This example file shows commands needed to create a data object used to 
#  replicate the models given in BIOGEME's example files.
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
	d = DB.CSV_idco(filename=os.path.join(larch._directory_,"data_warehouse","swissmetro.csv"),
					choice="CHOICE", weight="1.0",
					tablename="data", savename=None, alts=swissmetro_alts, safety=False)

	# Filtering the data
	d.queries.set_idco_query(d.queries.get_idco_query()+" WHERE CHOICE!=0 AND (PURPOSE==1 OR PURPOSE==3)")

	return d # Returns the data object from the data() function.

############################# END OF EXAMPLE FILE ##############################
