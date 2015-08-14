//
//  elm_inputstorage.i
//
//  Created by Jeffrey Newman on 12/21/12.
//
//

#ifndef __ELM_INPUTSTORAGE2_I__
#define __ELM_INPUTSTORAGE2_I__




%pythoncode %{
def __LinearFunction__call(self, *args, **kwargs):
	try:
		if (self._receiver_type==0):
			raise LarchError("LinearFunction improperly initialized")
		elif (self._receiver_type & COMPONENTLIST_TYPE_UTILITYCA):
			self.receive_utility_ca(*args, **kwargs)
		elif (self._receiver_type & COMPONENTLIST_TYPE_UTILITYCO):
			if len(kwargs)>0 and len(args)==0:
				self.receive_utility_co_kwd(**kwargs)
			elif len(kwargs)==0 and len(args)>0:
				if len(args)<2: raise LarchError("LinearFunction for co type requires at least two arguments: data and alt")
				self.receive_utility_co(*args)
			else:
				raise LarchError("LinearFunction for co type requires all-or-none use of keyword arguments")
		elif (self._receiver_type & COMPONENTLIST_TYPE_EDGE):
			self.receive_allocation(*args, **kwargs)
		else:
			raise LarchError("LinearFunction Not Implemented for type %i list"%self._receiver_type)
		####if self.parentmodel:
		####	self.parentmodel.freshen()
	except TypeError:
		for arg in args:
			print('type=',type(arg), 'value=',arg)
		for key,arg in kwargs.items():
			print('key=',key,'type=',type(arg), 'value=',arg)
		raise
LinearFunction.__call__ = __LinearFunction__call
del __LinearFunction__call
LinearFunction.__long_len = LinearFunction.__len__
LinearFunction.__len__ = lambda self: int(self.__long_len())

def __ComponentCellcodeMap__call(self, nest_name, nest_code=None, param_name="", multiplier=1.0, parent=None, parents=None, children=None):
	if isinstance(nest_name,(int,)) and nest_code is None:
		nest_name, nest_code = "nest%i"%nest_name, nest_name
	if isinstance(nest_name,(int,)) and isinstance(nest_code,(str,)):
		nest_name, nest_code = nest_code, nest_name
	self._create(nest_name, nest_code, param_name, multiplier)
	if parent is not None:
		self._link(parent,nest_code)
	if parents is not None:
		for p in parents: self._link(p,nest_code)
	if children is not None:
		for c in children: self._link(nest_code,c)
	####if self.parentmodel:
	####	self.parentmodel.freshen()
	return self[nest_code]
ComponentCellcodeMap.__call__ = __ComponentCellcodeMap__call
del __ComponentCellcodeMap__call
%}



#endif // __ELM_INPUTSTORAGE_I__
