

%{
#include "etk_ndarray.h"
#include "etk_ndarray_func.h"
%}


%typecheck(1) etk::ndarray*,  const etk::ndarray*
{
  if (PyArray_Check($input)) {
	$1 = ((PyArray_TYPE((PyArrayObject*)$input)== NPY_DOUBLE)||(PyArray_TYPE((PyArrayObject*)$input)== NPY_BOOL)) ? 1 : 0;
  } else {
	if (PySequence_Check($input)) {
	  for (Py_ssize_t seqi=0; seqi<PySequence_Length($input); seqi++) {
	    PyObject* seqitem = PySequence_GetItem($input, seqi);
		PyObject* floattest = PyNumber_Float(seqitem);
		if (!floattest) $1 = 0; else $1 = 1;
		Py_CLEAR(floattest);
		Py_CLEAR(seqitem);
		if ($1 == 0) break;
	  }
	} else {
		if ($input == Py_None) {
		  $1 = 1;
		} else {
		  $1 = 0;
		}
	}
  }
}

%typecheck(2) etk::symmetric_matrix*,  const etk::symmetric_matrix*
{
  if (PyArray_Check($input)) {
	if (PyArray_TYPE((PyArrayObject*)$input)== NPY_DOUBLE) {
		$1 = (PyArray_NDIMS($input)==2 && PyArray_DIM($input,0)==PyArray_DIM($input,1)) ? 1 : 0;
	} else { $1 = 0; }
  } else {
		if ($input == Py_None) {
		  $1 = 1;
		} else {
		  $1 = 0;
		}
  }
}

%typecheck(3) etk::ndarray_bool*,  const etk::ndarray_bool*
{
  if (PyArray_Check($input)) {
	$1 = (PyArray_TYPE((PyArrayObject*)$input)== NPY_BOOL) ? 1 : 0;
  } else {
		if ($input == Py_None) {
		  $1 = 1;
		} else {
		  $1 = 0;
		}
  }
}

%typecheck(4) long long*
{
	$1 = PyLong_Check($input) ? 1 : 0 ;
}

%typecheck(5) elm::darray*,  const elm::darray*
{
	if (PyArray_Check($input)) {
		$1 = ( (PyArray_TYPE((PyArrayObject*)$input)== NPY_DOUBLE)
			  ||(PyArray_TYPE((PyArrayObject*)$input)== NPY_BOOL  )
			  ||(PyArray_TYPE((PyArrayObject*)$input)== NPY_INT64 )
			  ) ? 1 : 0;
	} else {
		if ($input == Py_None) {
		  $1 = 1;
		} else {
		  $1 = 0;
		}
	}
}

%typecheck(6) int dtype {
	if (PyLong_Check($input)) {
		$1 = 1;
	} else {
		if (PyObject_HasAttrString($input, "num")) {
			$1 = 1;
		} else {
			$1 = 0;
		}
	}
}



/* Convert from Python --> C */
%typemap(in) etk::ndarray* (boosted::shared_ptr<etk::ndarray> temp) {
	if ($input == Py_None) {
		$1 = nullptr;
	} else {
		if (PyArray_Check($input)) {
			if (  (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE)
			    &&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_BOOL)
			    &&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_INT64)
			    &&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_INT8)
				) {
				PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE or BOOL or INT64 or INT8"));
				SWIG_fail;
			}
			
			try {
				temp = boosted::make_shared<etk::ndarray>($input);
			} catch (const etk::SQLiteError& e) {
				PyErr_SetString(ptrToSQLError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const etk::FacetError& e) {
				PyErr_SetString(ptrToFacetError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const std::exception& e) {
				PyErr_SetString(ptrToLarchError, const_cast<char*>(e.what()));
				SWIG_fail;
			}
			$1 = &(*temp);
		} else {
			temp = boosted::make_shared<etk::ndarray>(PyArray_ContiguousFromAny($input, NPY_DOUBLE, 0, 0));
			$1 = &(*temp);
		}
	}
}

%typemap(in) const etk::ndarray* (boosted::shared_ptr<const etk::ndarray> temp) {
	if ($input == Py_None) {
		$1 = nullptr;
	} else {
    if (PyArray_Check($input)) {
    if (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE"));
		SWIG_fail;
	}
		
			try {
				temp = boosted::make_shared<const etk::ndarray>($input);
			} catch (const etk::SQLiteError& e) {
				PyErr_SetString(ptrToSQLError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const etk::FacetError& e) {
				PyErr_SetString(ptrToFacetError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const std::exception& e) {
				PyErr_SetString(ptrToLarchError, const_cast<char*>(e.what()));
				SWIG_fail;
			}
	$1 = const_cast<etk::ndarray*>( &(*temp) );
	} else {
    temp = boosted::make_shared<const etk::ndarray>(PyArray_ContiguousFromAny($input, NPY_DOUBLE, 0, 0));
	$1 = const_cast<etk::ndarray*>( &(*temp) );
	}
	}
}

/* Convert from C --> Python */

%typemap(out) std::shared_ptr<etk::ndarray> {
	if (!*&$1) {
		Py_RETURN_NONE;
	} else {
		$result = (*(&($1)))->get_object();
	}
}

%typemap(out) etk::ndarray* {
	if (!$1) {
		Py_RETURN_NONE;
	} else {
		$result = $1->get_object();
	}
}




/* Convert from Python --> C */
%typemap(in) etk::symmetric_matrix* (boosted::shared_ptr<etk::symmetric_matrix> temp) {
	if ($input == Py_None) {
		$1 = nullptr;
	} else {
    if (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE"));
		SWIG_fail;
	}
		
			try {
				temp = boosted::make_shared<etk::symmetric_matrix>($input);
			} catch (const etk::SQLiteError& e) {
				PyErr_SetString(ptrToSQLError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const etk::FacetError& e) {
				PyErr_SetString(ptrToFacetError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const std::exception& e) {
				PyErr_SetString(ptrToLarchError, const_cast<char*>(e.what()));
				SWIG_fail;
			}

	$1 = &(*temp);
	}
}

%typemap(in) const etk::symmetric_matrix* (boosted::shared_ptr<const etk::symmetric_matrix> temp) {
	if ($input == Py_None) {
		$1 = nullptr;
	} else {
    if (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE"));
		SWIG_fail;
	}
		
			try {
				temp = boosted::make_shared<const etk::symmetric_matrix>($input);
			} catch (const etk::SQLiteError& e) {
				PyErr_SetString(ptrToSQLError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const etk::FacetError& e) {
				PyErr_SetString(ptrToFacetError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const std::exception& e) {
				PyErr_SetString(ptrToLarchError, const_cast<char*>(e.what()));
				SWIG_fail;
			}
	$1 = const_cast<etk::symmetric_matrix*>( &(*temp) );
	}
}

/* Convert from C --> Python */
%typemap(out) etk::symmetric_matrix* {
	if (!$1) {
		Py_RETURN_NONE;
	} else {
		$result = $1->get_object();
	}
}

%typemap(out) std::shared_ptr<etk::symmetric_matrix> {
	if (!*&$1) {
		Py_RETURN_NONE;
	} else {
		$result = (*(&($1)))->get_object();
	}
}



/* Convert from Python --> C */
%typemap(in) etk::ndarray_bool* (boosted::shared_ptr<etk::ndarray_bool> temp) {
	if ($input == Py_None) {
		$1 = nullptr;
	} else {
    if (PyArray_TYPE((PyArrayObject*)$input)!= NPY_BOOL) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type BOOL"));
		SWIG_fail;
	}
	
			try {
				temp = boosted::make_shared<etk::ndarray_bool>($input);
			} catch (const etk::SQLiteError& e) {
				PyErr_SetString(ptrToSQLError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const etk::FacetError& e) {
				PyErr_SetString(ptrToFacetError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const std::exception& e) {
				PyErr_SetString(ptrToLarchError, const_cast<char*>(e.what()));
				SWIG_fail;
			}
	$1 = &(*temp);
	}
}

/* Convert from C --> Python */
%typemap(out) etk::ndarray_bool* {
	if (!$1) {
		Py_RETURN_NONE;
	} else {
		$result = $1->get_object();
	}
}




/* Convert from Python --> C */
%typemap(in) etk::symmetric_matrix* (boosted::shared_ptr<etk::symmetric_matrix> temp) {
 	if ($input == Py_None) {
		$1 = nullptr;
	} else {
    if (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE"));
		SWIG_fail;
	}
	
			try {
				temp = boosted::make_shared<etk::symmetric_matrix>($input);
			} catch (const etk::SQLiteError& e) {
				PyErr_SetString(ptrToSQLError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const etk::FacetError& e) {
				PyErr_SetString(ptrToFacetError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const std::exception& e) {
				PyErr_SetString(ptrToLarchError, const_cast<char*>(e.what()));
				SWIG_fail;
			}
	$1 = &(*temp);
	}
}

/* Convert from C --> Python */
%typemap(out) etk::symmetric_matrix* {
	if (!$1) {
		Py_RETURN_NONE;
	} else {
		$result = $1->get_object();
	}
}




/* Convert from Python --> C */
%typemap(in) long long* (boosted::shared_ptr<long long> temp) {
	
			try {
				temp = boosted::make_shared<long long>(PyLong_AsLongLong($input));
			} catch (const etk::SQLiteError& e) {
				PyErr_SetString(ptrToSQLError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const etk::FacetError& e) {
				PyErr_SetString(ptrToFacetError, const_cast<char*>(e.what()));
				SWIG_fail;
			} catch (const std::exception& e) {
				PyErr_SetString(ptrToLarchError, const_cast<char*>(e.what()));
				SWIG_fail;
			}
	$1 = &(*temp);
}



/* Convert from C --> Python */
//%typemap(out) std::list<larch::datamatrix> {
// 
//obj = SWIG_NewPointerObj(f, $1_descriptor, 0);	
//	   
//		     $result = $1;
//}
//











%include "etk_ndarray.h"
%include "etk_ndarray_func.h"







%typemap(in) int dtype {
	if (PyLong_Check($input)) {
		$1 = PyLong_AsLong($input);
	} else {
		if (PyObject_HasAttrString($input, "num")) {
			PyObject* num = PyObject_GetAttrString($input, "num");
			$1 = PyLong_AsLong(num);
			Py_CLEAR(num);
		} else {
			PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires a type number or a numpy.dtype"));
			SWIG_fail;
		}
	}
}




/* Convert from Python --> C */
%typemap(in) elm::darray* (boosted::shared_ptr<elm::darray> temp) {
	
	if ($input == Py_None) {
		$1 = nullptr;
	} else {
		if (PyArray_Check($input)) {
			if (  (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE)
				&&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_BOOL  )
				&&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_INT64 )
				&&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_INT8  )
				) {
				PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE or BOOL or INT64 or INT8"));
				SWIG_fail;
			}
			try {
				temp = boosted::make_shared<elm::darray>($input);
			} catch (const std::exception& e) {
				PyErr_SetString(ptrToLarchError, const_cast<char*>(e.what()));
				SWIG_fail;
			}
			$1 = &(*temp);
		} else {
			PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array"));
			SWIG_fail;
		}
	}
} 

%typemap(in) const elm::darray* (boosted::shared_ptr<const elm::darray> temp) {
	
	if ($input == Py_None) {
		$1 = nullptr;
	} else {
		if (PyArray_Check($input)) {
		if (  (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE)
			&&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_BOOL  )
			&&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_INT64 )
			&&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_INT8  )
			) {
			PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE or BOOL or INT64 or INT8"));
			SWIG_fail;
		}
		try {
			temp = boosted::make_shared<const elm::darray>($input);
		} catch (const std::exception& e) {
			PyErr_SetString(ptrToLarchError, const_cast<char*>(e.what()));
			SWIG_fail;
		}
		$1 = const_cast<elm::darray*>( &(*temp) );
		} else {
			PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array"));
			SWIG_fail;
		}
	}
}



%typecheck(7) const std::map< std::string, boosted::shared_ptr<const elm::darray> >&
{
	if (!PyDict_Check($input)) {
		$1 = 0;
	} else {
		$1 = 1;
		PyObject *thekey, *thearray;
		Py_ssize_t pos = 0;
		while (PyDict_Next($input, &pos, &thekey, &thearray)) {
			if (PyArray_Check(thearray)) {
				if (  (PyArray_TYPE((PyArrayObject*)thearray)!= NPY_DOUBLE)
					&&(PyArray_TYPE((PyArrayObject*)thearray)!= NPY_BOOL  )
					&&(PyArray_TYPE((PyArrayObject*)thearray)!= NPY_INT64 )
					) {
					$1 = 0;
				}
			} else {$1 = 0;}
			if (!PyUnicode_Check(thekey)) { $1 = 0; }
		}
	}
}


%typemap(in) const std::map< std::string, boosted::shared_ptr<const elm::darray> >& ( std::map< std::string, boosted::shared_ptr<const elm::darray> > temp)
{
	if (!PyDict_Check($input)) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires a dict"));
		SWIG_fail;
	}
	PyObject *thekey, *thearray;
	Py_ssize_t pos = 0;
	while (PyDict_Next($input, &pos, &thekey, &thearray)) {

		if (PyArray_Check(thearray)) {
		if (  (PyArray_TYPE((PyArrayObject*)thearray)!= NPY_DOUBLE)
			&&(PyArray_TYPE((PyArrayObject*)thearray)!= NPY_BOOL  )
			&&(PyArray_TYPE((PyArrayObject*)thearray)!= NPY_INT64 )
			) {
			std::string explain = "function requires all array types to be DOUBLE or BOOL or INT64, '";
			explain+= PyString_ExtractCppString(thekey);
			explain+= "' is not";
			PyErr_SetString(ptrToLarchError, const_cast<char*>(explain.c_str()));
			SWIG_fail;
		}
		try {
			temp[PyString_ExtractCppString(thekey)] = boosted::make_shared<const elm::darray>(thearray);
		} catch (const std::exception& e) {
			std::string explain = e.what();
			explain+= "(on '";
			explain+= PyString_ExtractCppString(thekey);
			explain+= "')";
			PyErr_SetString(ptrToLarchError, const_cast<char*>(explain.c_str()));
			SWIG_fail;
		}
		$1 = &temp;
		} else {
			PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array"));
			SWIG_fail;
		}


	}
	
}






/* Convert from C --> Python */

%typemap(out) std::shared_ptr<elm::darray> {
    $result = (*(&($1)))->get_array();
	
	
	if (PyObject_HasAttrString($result, "vars")) {
		PyObject_DelAttrString($result, "vars");
	}
	
	PyObject* py_vars = PyTuple_New(($1)->get_variables().size());
	
	for (Py_ssize_t i = 0; i<PySequence_Size(py_vars); i++) {
		PyObject* item = PyString_FromString((($1)->get_variables()[i]).c_str());
		PyTuple_SetItem(py_vars, i, item);
	}
	
	int errstatus = PyObject_SetAttrString($result, "vars", py_vars);
	if (errstatus!=0) {
		PyErr_Clear();
	}
	
	Py_CLEAR(py_vars);

	//PyRun_SimpleString("print('%typemap(out) std::shared_ptr<elm::darray>')");
	
}

%typemap(out) elm::darray* {
	
	if (!$1) {
		//std::cerr << "typemap(out) elm::darray* return none\n";
		Py_RETURN_NONE;
	} else {
		//std::cerr << "typemap(out) elm::darray* return something\n";
	}
	
	$result = $1->get_array();
	
	if (PyObject_HasAttrString($result, "vars")) {
		PyObject_DelAttrString($result, "vars");
	}
	
	PyObject* py_vars = PyTuple_New(($1)->get_variables().size());
	
	for (Py_ssize_t i = 0; i<PySequence_Size(py_vars); i++) {
		PyObject* item = PyString_FromString((($1)->get_variables()[i]).c_str());
		PyTuple_SetItem(py_vars, i, item);
	}
	
	int errstatus = PyObject_SetAttrString($result, "vars", py_vars);
	if (errstatus!=0) {
		PyErr_Clear();
	}
	
	Py_CLEAR(py_vars);

	//PyRun_SimpleString("print('%typemap(out) elm::darray*')");
}


%typemap(out) const elm::darray* {
	
	if (!$1) {
		//std::cerr << "typemap(out) elm::darray* return none\n";
		Py_RETURN_NONE;
	} else {
		//std::cerr << "typemap(out) elm::darray* return something\n";
	}
	
	PyObject* temp_result = $1->get_array();

	$result = PyArray_View((PyArrayObject*) temp_result, NULL, NULL);
	Py_CLEAR(temp_result);
	
	PyArray_CLEARFLAGS((PyArrayObject*)$result, NPY_ARRAY_WRITEABLE);
	
	if (PyObject_HasAttrString($result, "vars")) {
		PyObject_DelAttrString($result, "vars");
	}
	
	PyObject* py_vars = PyTuple_New(($1)->get_variables().size());
	
	for (Py_ssize_t i = 0; i<PySequence_Size(py_vars); i++) {
		PyObject* item = PyString_FromString((($1)->get_variables()[i]).c_str());
		PyTuple_SetItem(py_vars, i, item);
	}
	
	int errstatus = PyObject_SetAttrString($result, "vars", py_vars);
	if (errstatus!=0) {
		PyErr_Clear();
	}
	
	Py_CLEAR(py_vars);

	//PyRun_SimpleString("print('%typemap(out) elm::darray*')");
}






%typecheck(SWIG_TYPECHECK_POINTER) elm::LinearComponent *, elm::LinearComponent & {



	// check if input is LinearComponent
	int res = SWIG_ConvertPtr($input, 0, $1_descriptor, 0);
	$1 = SWIG_CheckState(res);

	// if not, check if input is ParameterRef (can build component)
	if (!($1)) {
		if (PyUnicode_Check($input) && PyObject_HasAttrString($input, "_role")) {
			PyObject* role = PyObject_GetAttrString($input, "_role");
			$1 = (PyString_ExtractCppString(role)=="parameter");
			Py_CLEAR(role);
		}
		if ($1 == 0) PyErr_Clear();
	}

//	PyObject* type_o = PyObject_Type($input);
//	if (type_o) {
//		PyObject* type_o_str = PyObject_Str(type_o);
//		if (type_o_str)
//			std::cerr << "check "<<$1<<": type is " << PyString_ExtractCppString(type_o_str) << "\n";
//		Py_CLEAR(type_o_str);
//	}
//	Py_CLEAR(type_o);

}




%typemap(in) elm::LinearComponent* (int res, void* argp, bool use_temp, elm::LinearComponent temp_comp) {
	
	
	use_temp = false;
	res = SWIG_ConvertPtr($input, &argp, $1_descriptor,  0  | 0);
	if (!SWIG_IsOK(res)) {
		if (PyUnicode_Check($input) && PyObject_HasAttrString($input, "_role")) {
			PyObject* role = PyObject_GetAttrString($input, "_role");
			use_temp = (PyString_ExtractCppString(role)=="parameter");
			if (use_temp) {
				PyObject* pname = PyObject_Str($input);
				temp_comp.param_name = PyString_ExtractCppString(pname);
				Py_CLEAR(pname);
			}
			Py_CLEAR(role);
		}
		if (!use_temp) {
			SWIG_exception_fail(SWIG_ArgError(res), "in method '" "$symname" "', argument " "$argnum" " of type 'larch.LinearComponent' (*)");
		}
	}
	if (!argp && !use_temp) {
		SWIG_exception_fail(SWIG_ValueError, "invalid null reference " "in method '" "$symname" "', argument " "$argnum" " of type 'larch.LinearComponent'");
	}
	if (use_temp) {
		$1 = &temp_comp;
	} else {
		$1 = reinterpret_cast< elm::LinearComponent * >(argp);
	}
	
	
}


//	snipped from function below
//	PyObject* type_o = PyObject_Type($input);
//	if (type_o) {
//		PyObject* type_o_str = PyObject_Str(type_o);
//		if (type_o_str)
//			std::cerr << "le type is " << PyString_ExtractCppString(type_o_str) << "\n";
//		Py_CLEAR(type_o_str);
//	}
//	Py_CLEAR(type_o);



%typemap(in) elm::LinearComponent& (int res, void* argp, bool use_temp, elm::LinearComponent temp_comp) {
	
	
	use_temp = false;
	res = SWIG_ConvertPtr($input, &argp, $1_descriptor,  0  | 0);
	if (!SWIG_IsOK(res)) {

		if (PyUnicode_Check($input) && PyObject_HasAttrString($input, "_role")) {
			PyObject* role = PyObject_GetAttrString($input, "_role");
			
//			std::cerr << "converting role "<<PyString_ExtractCppString(role) << "\n";
			
			use_temp = (PyString_ExtractCppString(role)=="parameter");
			if (use_temp) {
				PyObject* pname = PyObject_Str($input);
				temp_comp.param_name = PyString_ExtractCppString(pname);
				temp_comp.data_name = "1";
				Py_CLEAR(pname);
			}
			Py_CLEAR(role);
		}
		if (!use_temp) {
			SWIG_exception_fail(SWIG_ArgError(res), "in method '$symname', argument $argnum of type 'larch.LinearComponent' (&)");
		}
	}
	if (!argp && !use_temp) {
		SWIG_exception_fail(SWIG_ValueError, "invalid null reference " "in method '" "$symname" "', argument " "$argnum" " of type 'larch.LinearComponent'");
	}
	if (use_temp) {
		$1 = &temp_comp;
	} else {
		$1 = reinterpret_cast< elm::LinearComponent * >(argp);
	}
	
	
}





%typecheck(SWIG_TYPECHECK_POINTER) elm::ComponentList *, elm::ComponentList & {

	// check if input is ComponentList
	int res = SWIG_ConvertPtr($input, 0, $1_descriptor, 0);
	$1 = SWIG_CheckState(res);
	
//	if (!($1)) {
//		std::cerr<<"Not A ComponentList";
//	}

//	// if not, check if input is ParameterRef (can build component)
//	if (!($1)) {
//		if (PyUnicode_Check($input) && PyObject_HasAttrString($input, "_role")) {
//			PyObject* role = PyObject_GetAttrString($input, "_role");
//			$1 = (PyString_ExtractCppString(role)=="parameter");
//			Py_CLEAR(role);
//		}
//		if ($1 == 0) PyErr_Clear();
//	}

}













/* Set the input argument to point to a temporary variable */
%typemap(in, numinputs=0) elm::darray** result_array (elm::darray* temp) {
	temp = nullptr;
	$1 = &temp;
}

%typemap(in, numinputs=0) elm::darray** result_caseids (elm::darray* temp) {
	temp = nullptr;
	$1 = &temp;
}

// Return the buffer.  Discarding any previous return result
%typemap(argout) (elm::darray** result_array, elm::darray** result_caseids) {
   Py_XDECREF($result);   /* Blow away any previous result */
	
	PyObject* ret1 = nullptr;
	PyObject* ret2 = nullptr;
	
	if (!$1 || !(*$1)) {
		Py_INCREF(Py_None);
		ret1 = Py_None;
	} else {
		ret1 = (*$1)->get_array();
		
		if (PyObject_HasAttrString(ret1, "vars")) {
			PyObject_DelAttrString(ret1, "vars");
		}
		
		PyObject* py_vars = PyTuple_New(((*$1))->get_variables().size());
		
		for (Py_ssize_t i = 0; i<PySequence_Size(py_vars); i++) {
			PyObject* item = PyString_FromString((((*$1))->get_variables()[i]).c_str());
			PyTuple_SetItem(py_vars, i, item);
		}
		
		
		int errstatus = PyObject_SetAttrString(ret1, "vars", py_vars);
		if (errstatus!=0) {
			PyErr_Clear();
		}
		
		Py_CLEAR(py_vars);
		delete (*$1);
	}

	if (!$2 || !(*$2)) {
		Py_INCREF(Py_None);
		ret2 = Py_None;
	} else {
		ret2 = (*$2)->get_array();
		
		if (PyObject_HasAttrString(ret2, "vars")) {
			PyObject_DelAttrString(ret2, "vars");
		}
		
		PyObject* py_vars = PyTuple_New(((*$2))->get_variables().size());
		
		for (Py_ssize_t i = 0; i<PySequence_Size(py_vars); i++) {
			PyObject* item = PyString_FromString((((*$2))->get_variables()[i]).c_str());
			PyTuple_SetItem(py_vars, i, item);
		}
		
		
		int errstatus = PyObject_SetAttrString(ret2, "vars", py_vars);
		if (errstatus!=0) {
			PyErr_Clear();
		}
		
		Py_CLEAR(py_vars);
		delete (*$2);
	}
	
	$result = PyTuple_New(2);
	PyTuple_SetItem($result, 0, ret1);
	PyTuple_SetItem($result, 1, ret2);
}

