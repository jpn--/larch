

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


/* Convert from Python --> C */
%typemap(in) etk::ndarray* (boosted::shared_ptr<etk::ndarray> temp) {
    if (PyArray_Check($input)) {
	if (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE) {
		PyErr_SetString(ptrToElmError, const_cast<char*>("function requires array type DOUBLE"));
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
		PyErr_SetString(ptrToElmError, const_cast<char*>("function requires array type DOUBLE"));
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
		PyErr_SetString(ptrToElmError, const_cast<char*>("function requires array type DOUBLE"));
		SWIG_fail;
	}
    temp = boosted::make_shared<etk::symmetric_matrix>($input);
	$1 = &(*temp);
}

%typemap(in) const etk::symmetric_matrix* (boosted::shared_ptr<const etk::symmetric_matrix> temp) {
    if (PyArray_TYPE((PyArrayObject*)$input)!= NPY_DOUBLE) {
		PyErr_SetString(ptrToElmError, const_cast<char*>("function requires array type DOUBLE"));
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
		PyErr_SetString(ptrToElmError, const_cast<char*>("function requires array type BOOL"));
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
		PyErr_SetString(ptrToElmError, const_cast<char*>("function requires array type DOUBLE"));
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
