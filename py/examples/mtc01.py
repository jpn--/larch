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
	if d is None: d = data()
	m = larch.Model(d)
	############################################################################
	# ModelObject.utility.co(<idCO data column>,<alternative code>,[<parameter name>])
	#	Adds a linear component to the utility of the indicated alternative.
	#	The data column is a string and can be any idCO data column or a pre-calculated 
	#	value derived from one or more idCO data columns, or no data columns at all.
	#	Note: there is no need to declare a parameter name seperately from this 
	#	command. Default values will be assumed for parameters that are not previously
	#	declared.
	m.utility.co("1",2,"ASC_SR2") 
	m.utility.co("1",3,"ASC_SR3P") 
	m.utility.co("1",4,"ASC_TRAN") 
	m.utility.co("1",5,"ASC_BIKE") 
	m.utility.co("1",6,"ASC_WALK") 
	m.utility.co("hhinc",2)
	m.utility.co("hhinc",3)
	m.utility.co("hhinc",4)
	m.utility.co("hhinc",5)
	m.utility.co("hhinc",6)
	############################################################################
	# ModelObject.utility.ca(<idCA data column>,[<parameter name>])
	#	Adds a linear component to the utility of the indicated alternative.
	#	The data column is a string and can be any idCA data column or a pre-calculated 
	#	value derived from one or more idCA data columns, or no data columns at all.
	#	Note: there is no need to declare a parameter name seperately from this 
	#	command. Default values will be assumed for parameters that are not previously
	#	declared.
	m.utility.ca("tottime")
	m.utility.ca("totcost")
	############################################################################
	# ModelObject.option
	#	A structure that defines certain options to be applied when estimating
	#	models.
	m.option.calc_std_errors = True
	return m

############################# END OF EXAMPLE FILE ##############################

