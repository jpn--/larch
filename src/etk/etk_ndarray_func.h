/*
 *  etk_ndarray.h
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


#ifndef __TOOLBOX_NDARRAY_FUNC__
#define __TOOLBOX_NDARRAY_FUNC__

#ifndef SWIG
#include "etk_python.h"
#include "etk_ndarray.h"
#include <climits>
#include <numpy/arrayobject.h> 
#endif // ndef SWIG

namespace etk {

	etk::ndarray* ndarray_make();

	void ndarray_exp(etk::ndarray* self);
	void ndarray_log(etk::ndarray* self);
	void ndarray_init(etk::ndarray* self);
	void SymmetricArray_use_upper_triangle(etk::symmetric_matrix* self);

} // end namespace etk


#ifdef SWIG	
%pythoncode %{

from .array import Array
from .array import SymmetricArray

%}
#endif // ndef SWIG



#endif // __TOOLBOX_NDARRAY_FUNC__
