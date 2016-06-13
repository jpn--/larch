//
//  elm_inputstorage.i
//
//  Created by Jeffrey Newman on 12/21/12.
//
//

#ifndef __ELM_INPUTSTORAGE_I__
#define __ELM_INPUTSTORAGE_I__


%rename(LinearBundle) elm::LinearBundle_1;
%rename(LinearCOBundle) elm::LinearCOBundle_1;


%rename(LinearComponent) elm::LinearComponent;
%rename(_data) elm::LinearComponent::data_name;
%rename(_param) elm::LinearComponent::param_name;

%rename(LinearFunction) elm::ComponentList;
%rename(LinearFunctionPair) elm::ComponentListPair;

%feature("kwargs", 1) elm::LinearComponent::LinearComponent;
%feature("kwargs", 1) elm::ComponentList::receive_utility_ca;
%feature("kwargs", 1) elm::ComponentList::receive_allocation;
%feature("kwargs", 1) elm::ComponentList::receive_utility_co_kwd;



%feature("docstring") elm::LinearComponent
"A combination of a parameter and data.

This class represents a single term of a linear function, i.e. a parameter
multiplied by some data.  The data may be a single column of raw data
from a data :class:`Fountain`, or it may be some prescribed function of
raw data (e.g. logarithm of cost, or cost divided by income); the principal
requirement is that the data function contains only data and no parameters
to be estimated, other than the single linear coefficient.

Parameters
----------
param : :class:`str` or :class:`.ParameterRef`
	The name of, or reference to, a parameter.
data : :class:`str` or :class:`.DataRef`
	The name of, or reference to, some data.  This may be a raw column in
	a data :class:`Fountain`, or an expression that can be evaluated, including
	a number expressed as a string. To express a constant (i.e. a parameter
	with no data) give 1.0.
multiplier : float
	A convenient method to multiply the data by a constant, which can
	be given as a float instead of a string.
category : None or int or string or tuple
	Some LinearComponent's apply only to certain things.
";




/* Convert cellcodepair from Python --> C */
%typemap(in) const elm::cellcodepair& (elm::cellcodepair temp) {
	if (!PyArg_ParseTuple($input, "LL", &(temp.up), &(temp.dn))) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("a cellcode pair must be a 2-tuple of integers"));
		SWIG_fail;
	};
	$1 = &temp;
}

/* Convert cellcodepair from C --> Python */
%typemap(out) elm::cellcodepair {
    $result = Py_BuildValue("LL", &(($1).up), &(($1).dn)));
}

%typemap(out) ::std::vector<elm::cellcodepair> {

	$result = PyTuple_New((&($1))->size());
	for (size_t j=0; j<(&($1))->size(); j++) {
		//std::cerr << "OUTX j="<<j<<": "<<(((*&($1))[j].up)<<","<<(((*&($1))[j].dn)));
		PyTuple_SetItem($result, j, Py_BuildValue("LL", ((*&($1))[j].up), ((*&($1))[j].dn)));
	}
}

%{
#include "elm_model2.h"
%}



%template(ComponentVector) std::vector<elm::LinearComponent>;
%template(_base_LinearSubBundle_1) std::map< elm::cellcode, elm::ComponentList >;






#endif // __ELM_INPUTSTORAGE_I__
