/*
 *  elm_queryset_twotable.i
 *
 *  Copyright 2007-2015 Jeffrey Newman
 *
 *  Larch is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  Larch is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with Larch.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


%feature("docstring") elm::QuerySetTwoTable::set_choice_ca "\
Set the choice expression that will evaluate on the idca table. \n\
 \n\
Parameters \n\
---------- \n\
expr : str \n\
	The expression to be evaluated. It should evaluate to 1 if the alternative for the \n\
	particular row was chosen, and 0 otherwise. (For certain specialized models,     \n\
	values other than 0 or 1 may be appropriate.)\n\
";

%feature("docstring") elm::QuerySetTwoTable::set_choice_co "\
Set the choice expression that will evaluate on the idco table. \n\
 \n\
Parameters \n\
---------- \n\
expr : str \n\
	The expression to be evaluated. It should result in integer values \n\
	corresponding to the alternative codes.\n\
";



