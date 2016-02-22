

#include "larch_modelparameter.h"
#include "elm_model2.h"


elm::ModelParameter::ModelParameter(Model2* model, const size_t& slot)
: model(model)
, slot(slot)
{
	if (!model) OOPS("cannot create a ModelParameter without a Model");
	Py_XINCREF(model->weakself);
}

elm::ModelParameter::~ModelParameter()
{
	Py_XDECREF(model->weakself);
}

double elm::ModelParameter::_get_value() const
{
	if (slot>=model->FCurrent.size()) OOPS_IndexError("slot exceeds allocated size");
	return model->FCurrent[slot];
}

void elm::ModelParameter::_set_value(const double& value)
{
	if (slot>=model->FCurrent.size()) OOPS_IndexError("slot exceeds allocated size");
	 model->FCurrent[slot] = value;
}




double elm::ModelParameter::_get_min() const
{
	if (slot>=model->FMin.size()) OOPS_IndexError("slot exceeds allocated size");
	return model->FMin[slot];
}

void elm::ModelParameter::_set_min(const double& value)
{
	if (slot>=model->FMin.size()) OOPS_IndexError("slot exceeds allocated size");
	 model->FMin[slot] = value;
}

void elm::ModelParameter::_del_min()
{
	if (slot>=model->FMin.size()) OOPS_IndexError("slot exceeds allocated size");
	 model->FMin[slot] = -INF;
}






double elm::ModelParameter::_get_max() const
{
	if (slot>=model->FMax.size()) OOPS_IndexError("slot exceeds allocated size");
	return model->FMax[slot];
}

void elm::ModelParameter::_set_max(const double& value)
{
	if (slot>=model->FMax.size()) OOPS_IndexError("slot exceeds allocated size");
	 model->FMax[slot] = value;
}

void elm::ModelParameter::_del_max()
{
	if (slot>=model->FMax.size()) OOPS_IndexError("slot exceeds allocated size");
	 model->FMax[slot] = INF;
}


bool elm::ModelParameter::_get_holdfast() const
{
	if (slot>=model->FHoldfast.size()) OOPS_IndexError("slot exceeds allocated size");
	return model->FHoldfast.bool_at(slot);
}

void elm::ModelParameter::_set_holdfast(const bool& value)
{
	if (slot>=model->FHoldfast.size()) OOPS_IndexError("slot exceeds allocated size");
	model->FHoldfast.bool_at(slot) = value;
}

void elm::ModelParameter::_del_holdfast()
{
	if (slot>=model->FHoldfast.size()) OOPS_IndexError("slot exceeds allocated size");
	model->FHoldfast.bool_at(slot) = false;
}





double elm::ModelParameter::_get_nullvalue() const
{
	if (slot>=model->FNullValues.size()) OOPS_IndexError("slot exceeds allocated size");
	return model->FNullValues[slot];
}

void elm::ModelParameter::_set_nullvalue(const double& value)
{
	if (slot>=model->FNullValues.size()) OOPS_IndexError("slot exceeds allocated size");
	model->FNullValues[slot] = value;
}

double elm::ModelParameter::_get_initvalue() const
{
	if (slot>=model->FInitValues.size()) OOPS_IndexError("slot exceeds allocated size");
	return model->FInitValues[slot];
}

void elm::ModelParameter::_set_initvalue(const double& value)
{
	if (slot>=model->FInitValues.size()) OOPS_IndexError("slot exceeds allocated size");
	model->FInitValues[slot] = value;
}





double elm::ModelParameter::_get_std_err() const
{
	if (slot >= model->invHess.size1() || slot >= model->invHess.size2()) OOPS_IndexError("slot exceeds allocated size");
	double x = model->invHess(slot,slot);
	if (x>0) {
		return sqrt( x );
	}
	return -sqrt(-x);
}


double elm::ModelParameter::_get_robust_std_err() const
{
	if (slot >= model->robustCovariance.size1() || slot >= model->robustCovariance.size2()) OOPS_IndexError("slot exceeds allocated size");
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
