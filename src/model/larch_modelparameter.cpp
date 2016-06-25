

#include "larch_modelparameter.h"
#include "sherpa.h"
#include "elm_model2.h"


elm::ModelParameter::ModelParameter(sherpa* model, const size_t& slot)
: model(model)
, slot(slot)
, model_as_pyobject(nullptr)
{
	if (!model) OOPS("cannot create a ModelParameter without a Model");
	
	model_as_pyobject = model->weakself;
	Py_XINCREF(model_as_pyobject);

}

elm::ModelParameter::ModelParameter(const elm::ModelParameter& original)
: model(original.model)
, slot(original.slot)
, model_as_pyobject(original.model_as_pyobject)
{
	if (!original.model) OOPS("cannot create a ModelParameter without a Model");
	
	Py_XINCREF(model_as_pyobject);

}


elm::ModelParameter::~ModelParameter()
{
//	if (model_as_pyobject) {
//		std::cerr<<"DESTROY ModelParameter "<<(void*)this<<" on "<< (void*)model_as_pyobject << " refcount="<< model_as_pyobject->ob_refcnt <<"\n";
//	} else {
//		std::cerr<<"DESTROY ModelParameter "<<(void*)this<<" on NULLPTR\n";
//	}
	Py_CLEAR(model_as_pyobject);
}

double elm::ModelParameter::_get_value() const
{
	if (slot>=model->FCurrent.size()) OOPS_IndexError("slot ",slot," exceeds allocated size of ",model->FCurrent.size());
	return model->FCurrent[slot];
}

void elm::ModelParameter::_set_value(const double& value)
{
	if (slot>=model->FCurrent.size()) OOPS_IndexError("slot ",slot," exceeds allocated size of ",model->FCurrent.size());
	 model->FCurrent[slot] = value;
}




double elm::ModelParameter::_get_min() const
{
	if (slot>=model->FMin.size()) OOPS_IndexError("slot ",slot," exceeds allocated size of ",model->FMin.size());
	return model->FMin[slot];
}

void elm::ModelParameter::_set_min(const double& value)
{
	if (slot>=model->FMin.size()) OOPS_IndexError("slot ",slot," exceeds allocated size of ",model->FMin.size());
	 model->FMin[slot] = value;
}

void elm::ModelParameter::_del_min()
{
	if (slot>=model->FMin.size()) OOPS_IndexError("slot ",slot," exceeds allocated size of ",model->FMin.size());
	 model->FMin[slot] = -INF;
}






double elm::ModelParameter::_get_max() const
{
	if (slot>=model->FMax.size()) OOPS_IndexError("slot ",slot," exceeds allocated size of ",model->FMax.size());
	return model->FMax[slot];
}

void elm::ModelParameter::_set_max(const double& value)
{
	if (slot>=model->FMax.size()) OOPS_IndexError("slot ",slot," exceeds allocated size of ",model->FMax.size());
	 model->FMax[slot] = value;
}

void elm::ModelParameter::_del_max()
{
	if (slot>=model->FMax.size()) OOPS_IndexError("slot ",slot," exceeds allocated size of ",model->FMax.size());
	 model->FMax[slot] = INF;
}


signed char elm::ModelParameter::_get_holdfast() const
{
	if (slot>=model->FHoldfast.size()) OOPS_IndexError("slot slot ",slot," exceeds allocated size");
	return model->FHoldfast.int8_at(slot);
}

void elm::ModelParameter::_set_holdfast(const bool& value)
{
	if (slot>=model->FHoldfast.size()) OOPS_IndexError("slot slot ",slot," exceeds allocated size");
	if (value) {
		model->FHoldfast.int8_at(slot) = 1;
	} else {
		model->FHoldfast.int8_at(slot) = 0;
	}
}

void elm::ModelParameter::_set_holdfast(const signed char& value)
{
	if (slot>=model->FHoldfast.size()) OOPS_IndexError("slot slot ",slot," exceeds allocated size");
	model->FHoldfast.int8_at(slot) = value;
}

void elm::ModelParameter::_del_holdfast()
{
	if (slot>=model->FHoldfast.size()) OOPS_IndexError("slot slot ",slot," exceeds allocated size");
	model->FHoldfast.int8_at(slot) = 0;
}





double elm::ModelParameter::_get_nullvalue() const
{
	if (slot>=model->FNullValues.size()) OOPS_IndexError("slot ",slot," exceeds allocated size");
	return model->FNullValues[slot];
}

void elm::ModelParameter::_set_nullvalue(const double& value)
{
	if (slot>=model->FNullValues.size()) OOPS_IndexError("slot ",slot," exceeds allocated size");
	model->FNullValues[slot] = value;
}

double elm::ModelParameter::_get_initvalue() const
{
	if (slot>=model->FInitValues.size()) OOPS_IndexError("slot ",slot," exceeds allocated size");
	return model->FInitValues[slot];
}

void elm::ModelParameter::_set_initvalue(const double& value)
{
	if (slot>=model->FInitValues.size()) OOPS_IndexError("slot ",slot," exceeds allocated size");
	model->FInitValues[slot] = value;
}

void elm::ModelParameter::_set_std_err(const double& value)
{
	if (slot >= model->invHess.size1() || slot >= model->invHess.size2()) OOPS_IndexError("slot ",slot," exceeds allocated covariance array size");
	model->invHess(slot,slot) = value*value;
}



void elm::ModelParameter::_set_t_stat(const double& value)
{
	_set_std_err((_get_value() - _get_nullvalue()) / value );
}



double elm::ModelParameter::_get_std_err() const
{
	if (slot >= model->invHess.size1() || slot >= model->invHess.size2()) OOPS_IndexError("slot ",slot," exceeds allocated covariance array size");
	double x = model->invHess(slot,slot);
	if (x>0) {
		return sqrt( x );
	}
	return -sqrt(-x);
}

double elm::ModelParameter::_get_t_stat() const
{
	return (_get_value() - _get_nullvalue()) / _get_std_err();
}

double elm::ModelParameter::_get_robust_std_err() const
{
	if (slot >= model->robustCovariance.size1() || slot >= model->robustCovariance.size2()) OOPS_IndexError("slot ",slot," exceeds allocated robust covariance array size");
	double x = model->robustCovariance(slot,slot);
	if (x>0) {
		return sqrt( x );
	}
	return -sqrt(-x);
}


std::string elm::ModelParameter::_get_name() const
{
	return model->FNames.string_from_index(slot);
}

size_t elm::ModelParameter::_get_index() const
{
	return slot;
}

etk::symmetric_matrix* elm::ModelParameter::_get_complete_covariance_matrix() const
{
	return model->_get_inverse_hessian_array();
}

PyObject* elm::ModelParameter::_get_model()
{
	if (model_as_pyobject) {
		Py_XINCREF( model_as_pyobject );
		return model_as_pyobject;
	} else {
		Py_RETURN_NONE;
	}
}


