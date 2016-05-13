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
#  BIOGEME's example file: 14selectionBias.py
#  http://biogeme.epfl.ch/swissmetro/examples.html
#
################################################################################

from .. import larch
import os

# Set example data for this model
with open(os.path.join(larch._directory_,"examples","swissmetro00data.py")) as f:
	code = compile(f.read(), "swissmetro00data.py", 'exec')
	exec(code, globals(), globals())

def model(d=None): # Define a function to create a model object.
	if d is None: d = data()
	m = larch.Model(d)
	m.title = "swissmetro example 14 (nested logit with sampling bias)"

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

	# ModelObject.nest(<name of nest>, <altcode of nest>, <parameter name>)
	m.nest("existing", 4, "existing") 

	# ModelObject.link(<altcode of upstream node>, <altcode of downstream node>)
	m.link(4, 1)
	m.link(4, 3)

	# ModelObject.samplingbias.co(<idCO data column>,<alternative code>,[<parameter name>])
	m.samplingbias.co("1",1,"SB_TRAIN")

	# ModelObject.option
	#	A structure that defines certain options to be applied when estimating
	#	models.
	m.option.calc_std_errors = True

	return m # Returns the model object from the model() function.

############################# END OF EXAMPLE FILE ##############################
