/*
 *  etk_ndarray.cpp
 *
 *  Copyright 2007-2017 Jeffrey Newman
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


#include "etk.h"
#include "etk_ndarray_func.h"
#include <numpy/arrayobject.h> 

using namespace etk;

#include <iostream>
#include <cmath>
#include <climits>
#include <cstring>


etk::ndarray* etk::ndarray_make()
{
	etk::ndarray* result = new etk::ndarray(5,5);
	return result;
}



/////////////////////////////////

void etk::ndarray_exp(etk::ndarray* self)
{
	self->exp();
}

void etk::ndarray_log(etk::ndarray* self)
{
	self->log();
}

void etk::ndarray_init(etk::ndarray* self)
{
	self->initialize();
}

void etk::SymmetricArray_use_upper_triangle(etk::symmetric_matrix* self)
{
	self->copy_uppertriangle_to_lowertriangle();
}




