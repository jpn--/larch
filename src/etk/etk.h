/*
 *  etk.h
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


#ifndef __etk_headers__
#define __etk_headers__

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <map>

//typedef std::string string;

#define PY_ARRAY_UNIQUE_SYMBOL _ETK_PY_ARRAY_UNIQUE_SYMBOL_
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define NO_IMPORT_ARRAY

#include "larch_portable.h"

#include "etk_messenger.h"
#include "etk_memory.h"
#include "etk_ndarray.h"
#include "etk_ndarray_bool.h"
#include "etk_math.h"
#include "etk_object.h"
#include "etk_exception.h"
#include "etk_sqlite.h"
#include "etk_autoindex.h"
#include "etk_random.h"
#include "etk_index.h"
#include "etk_vectors.h"
#include "etk_time.h"
#include "etk_cat.h"
#include "etk_basic.h"
#include "etk_pydict.h"
#include "etk_test_swig.h"
#include "etk_refcount.h"
#include "etk_resultcodes.h"


#ifdef __APPLE__
// C++11 compiler functions
#define noexcept_ noexcept
#else
#define noexcept_ 
#endif // __APPLE__


#endif // __etk_headers__

