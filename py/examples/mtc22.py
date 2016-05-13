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

def model(d=None):
	from . import mtc17
	m = mtc17.model(d)
	
	m.new_nest(nest_name='motorized', param_name="mu_motorized", parent=m.root_id, children=[1,2,3,4])
	m.new_nest(nest_name='nonmotorized', param_name="mu_nonmotorized", parent=m.root_id, children=[5,6])
	
	return m

############################# END OF EXAMPLE FILE ##############################

