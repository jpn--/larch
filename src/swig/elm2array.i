

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
	  $1 = 0;
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
	$1 = 0;
  }
}

%typecheck(3) etk::ndarray_bool*,  const etk::ndarray_bool*
{
  if (PyArray_Check($input)) {
	$1 = (PyArray_TYPE((PyArrayObject*)$input)== NPY_BOOL) ? 1 : 0;
  } else {
	$1 = 0;
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
    $1 = 0;
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
    if (PyArray_Check($input)) {
	if ((PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE)&&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_BOOL)) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE or BOOL"));
		SWIG_fail;
	}
    temp = boosted::make_shared<etk::ndarray>($input);
	$1 = &(*temp);
	} else {
    temp = boosted::make_shared<etk::ndarray>(PyArray_ContiguousFromAny($input, NPY_DOUBLE, 0, 0));
	$1 = &(*temp);		
	}
}

%typemap(in) const etk::ndarray* (boosted::shared_ptr<const etk::ndarray> temp) {
    if (PyArray_Check($input)) {
    if (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE"));
		SWIG_fail;
	}
    temp = boosted::make_shared<const etk::ndarray>($input);
	$1 = const_cast<etk::ndarray*>( &(*temp) );
	} else {
    temp = boosted::make_shared<const etk::ndarray>(PyArray_ContiguousFromAny($input, NPY_DOUBLE, 0, 0));
	$1 = const_cast<etk::ndarray*>( &(*temp) );
	}
}

/* Convert from C --> Python */

%typemap(out) std::shared_ptr<etk::ndarray> {
    $result = (*(&($1)))->get_object();
}

%typemap(out) etk::ndarray* {
    $result = $1->get_object();
}




/* Convert from Python --> C */
%typemap(in) etk::symmetric_matrix* (boosted::shared_ptr<etk::symmetric_matrix> temp) {
    if (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE"));
		SWIG_fail;
	}
    temp = boosted::make_shared<etk::symmetric_matrix>($input);
	$1 = &(*temp);
}

%typemap(in) const etk::symmetric_matrix* (boosted::shared_ptr<const etk::symmetric_matrix> temp) {
    if (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE"));
		SWIG_fail;
	}
    temp = boosted::make_shared<const etk::symmetric_matrix>($input);
	$1 = const_cast<etk::symmetric_matrix*>( &(*temp) );
}

/* Convert from C --> Python */
%typemap(out) etk::symmetric_matrix* {
    $result = $1->get_object();
}






/* Convert from Python --> C */
%typemap(in) etk::ndarray_bool* (boosted::shared_ptr<etk::ndarray_bool> temp) {
    if (PyArray_TYPE((PyArrayObject*)$input)!= NPY_BOOL) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type BOOL"));
		SWIG_fail;
	}
	temp = boosted::make_shared<etk::ndarray_bool>($input);
	$1 = &(*temp);
}

/* Convert from C --> Python */
%typemap(out) etk::ndarray_bool* {
    $result = $1->get_object();
}




/* Convert from Python --> C */
%typemap(in) etk::symmetric_matrix* (boosted::shared_ptr<etk::symmetric_matrix> temp) {
    if (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE"));
		SWIG_fail;
	}
	temp = boosted::make_shared<etk::symmetric_matrix>($input);
	$1 = &(*temp);
}

/* Convert from C --> Python */
%typemap(out) etk::symmetric_matrix* {
    $result = $1->get_object();
}




/* Convert from Python --> C */
%typemap(in) long long* (boosted::shared_ptr<long long> temp) {
	temp = boosted::make_shared<long long>(PyLong_AsLongLong($input));
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
    if (PyArray_Check($input)) {
	if (  (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE)
		&&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_BOOL  )
		&&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_INT64 )
		) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE or BOOL or INT64"));
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

%typemap(in) const elm::darray* (boosted::shared_ptr<const elm::darray> temp) {
    if (PyArray_Check($input)) {
    if (  (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE)
		&&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_BOOL  )
		&&(PyArray_TYPE((PyArrayObject*)$input)!= NPY_INT64 )
		) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires array type DOUBLE or BOOL or INT64"));
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
			PyErr_SetString(ptrToLarchError, const_cast<char*>("function requires all array types to be DOUBLE or BOOL or INT64"));
			SWIG_fail;
		}
		try {
			temp[PyString_ExtractCppString(thekey)] = boosted::make_shared<const elm::darray>(thearray);
		} catch (const std::exception& e) {
			PyErr_SetString(ptrToLarchError, const_cast<char*>(e.what()));
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
	
	PyObject_SetAttrString($result, "vars", py_vars);
	
	Py_CLEAR(py_vars);

	PyRun_SimpleString("print('%typemap(out) std::shared_ptr<elm::darray>')");
	
}

%typemap(out) elm::darray* {
    $result = $1->get_array();

	
	
	if (PyObject_HasAttrString($result, "vars")) {
		PyObject_DelAttrString($result, "vars");
	}
	
	PyObject* py_vars = PyTuple_New(($1)->get_variables().size());
	
	for (Py_ssize_t i = 0; i<PySequence_Size(py_vars); i++) {
		PyObject* item = PyString_FromString((($1)->get_variables()[i]).c_str());
		PyTuple_SetItem(py_vars, i, item);
	}
	
	PyObject_SetAttrString($result, "vars", py_vars);
	
	Py_CLEAR(py_vars);

	PyRun_SimpleString("print('%typemap(out) elm::darray*')");
}
