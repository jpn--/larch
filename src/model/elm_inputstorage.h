//
//  elm_inputstorage.h
//  Yggdrasil
//
//  Created by Jeffrey Newman on 12/21/12.
//
//

#ifndef __ELM_INPUTSTORAGE_H__
#define __ELM_INPUTSTORAGE_H__


#ifdef SWIG
%rename(Component) elm::InputStorage;
%rename(data) elm::InputStorage::apply_name;
%rename(param) elm::InputStorage::param_name;

%feature("kwargs", 1) elm::InputStorage::InputStorage;
%feature("kwargs", 1) elm::ComponentList::receive_utility_ca;
%feature("kwargs", 1) elm::ComponentList::receive_utility_co_kwd;

/* Convert cellcodepair from Python --> C */
%typemap(in) const elm::cellcodepair& (elm::cellcodepair temp) {
	if (!PyArg_ParseTuple($input, "KK", &(temp.up), &(temp.dn))) {
		PyErr_SetString(ptrToLarchError, const_cast<char*>("a cellcode pair must be a 2-tuple of non-negative integers"));
		SWIG_fail;
	};
	$1 = &temp;
}

/* Convert cellcodepair from C --> Python */
%typemap(out) elm::cellcodepair {
    $result = Py_BuildValue("KK", &(($1).up), &(($1).dn)));
}

%{
#include "elm_model2.h"
%}
#endif // def SWIG

#include "elm_cellcode.h"

namespace elm {
	
	class Model2;
	class Facet;
		
	struct InputStorage {
		std::string		apply_name;
		std::string     param_name;
		elm::cellcode	altcode;
		std::string	    altname;
		double			multiplier;

		std::string __repr__() const;
		
		InputStorage(std::string data="", std::string param="",
					 elm::cellcode altcode=cellcode_empty, std::string altname="",
					 double multiplier=1.0);
		static InputStorage Create(PyObject* obj);
//		InputStorage(const InputStorage& obj);

		#ifdef SWIG
		%pythoncode %{
		def altcallsign(self):
			return "%i: %s"%(self.altcode, self.altname)
		def set_from_callsign(self, x):
			c,n = x.split(": ")
			self.altcode = long(c)
			self.altname = n
		%}
		#endif // SWIG

	};
	

};

#ifdef SWIG
%template(ComponentVector) std::vector<elm::InputStorage>;
#endif

#define COMPONENTLIST_TYPE_UTILITYCA 0x1
#define COMPONENTLIST_TYPE_UTILITYCO 0x2
#define COMPONENTLIST_TYPE_LOGSUM    0x4
#define COMPONENTLIST_TYPE_EDGE      0x8

namespace elm {
	
	class ComponentList
	: public ::std::vector<InputStorage>
	{
		public:
		int _receiver_type;
		Model2* parentmodel;
		ComponentList(int type=0, Model2* parentmodel=nullptr);
		
		
		void receive_utility_ca(const std::string& data,
							    std::string param="",
							    const double& multiplier=1.0);

		void receive_utility_co (const std::string& data,
							   const std::string& altname,
							   std::string param="", 
							   const double& multiplier=1.0);
		void receive_utility_co (const std::string& data, 
							   const elm::cellcode& altcode,
							   std::string param="", 
							   const double& multiplier=1.0);
		void receive_utility_co_kwd(const std::string&		data="",
									const std::string&		altname="",
									const elm::cellcode&	altcode=-9,
									std::string				param="",
									const double&			multiplier=1.0);

		std::string __repr__() const;

		std::vector<std::string> needs() const;

	};



	class ComponentCellcodeMap
	#ifndef SWIG
	: public ::std::map<elm::cellcode, elm::InputStorage>
	#endif
	{
		public:
		int _receiver_type;
		Model2* parentmodel;
		ComponentCellcodeMap(int type=0, Model2* parentmodel=nullptr);

		std::string __repr__() const;

		#ifdef SWIG
        unsigned int size() const;
        bool empty() const;
        void clear();
		%extend {
            elm::InputStorage& __getitem__(const elm::cellcode& key) throw (std::out_of_range) {
                std::map<elm::cellcode,elm::InputStorage >::iterator i = self->find(key);
                if (i != self->end())
                    return i->second;
                else
                    throw std::out_of_range("key not found");
            }
            void __setitem__(const elm::cellcode& key, const elm::InputStorage& x) {
                (*self)[key] = x;
            }
            void __delitem__(const elm::cellcode& key) throw (std::out_of_range) {
                std::map<elm::cellcode,elm::InputStorage >::iterator i = self->find(key);
                if (i != self->end())
                    self->erase(i);
					else
						throw std::out_of_range("key not found");
            }
            bool __contains__(const elm::cellcode& key) {
                std::map<elm::cellcode,elm::InputStorage >::iterator i = self->find(key);
                return i != self->end();
            }
			int __len__() const {
				return self->size();
			}
			void _create(const std::string& nest_name, const elm::cellcode& nest_code,
						 std::string param_name="", const double& param_multiplier=1.0)
						  throw (etk::exception_t) {
				if (self->parentmodel) {
					self->parentmodel->nest(nest_name, nest_code, param_name, param_multiplier);
				} else {
					throw etk::exception_t("not linked to a model");
				}
			}
			void _link(const elm::cellcode& parent, const elm::cellcode& child) {
				if (self->parentmodel) {
					self->parentmodel->link(parent,child);
				} else {
					throw etk::exception_t("not linked to a model");
				}
			}
        }
		#endif // def SWIG
	};
	
	
	struct ComponentListPair {
		std::string descrip;
		ComponentList ca;
		ComponentList co;
		ComponentListPair(int type1=0, int type2=0, std::string descrip="", Model2* parentmodel=nullptr);
		
		void __call__(const elm::cellcode& altcode, std::string param="", const double& multiplier=1.0);
		void clean(Facet& db);
	};
	
	
	
	typedef elm::ComponentList EdgeValue;

	class ComponentEdgeMap
	#ifndef SWIG
	: public ::std::map<elm::cellcodepair, elm::EdgeValue>
	#endif
	{
		public:
		int _receiver_type;
		Model2* parentmodel;
		ComponentEdgeMap(Model2* parentmodel=nullptr);

		std::string __repr__() const;

		#ifdef SWIG
        unsigned int size() const;
        bool empty() const;
        void clear();
		%extend {
            elm::EdgeValue& __getitem__(const elm::cellcodepair& key) throw (std::out_of_range) {
                std::map<elm::cellcodepair, elm::EdgeValue>::iterator i = self->find(key);
                if (i != self->end())
                    return i->second;
                else
                    throw std::out_of_range("key not found");
            }
            void __setitem__(const elm::cellcodepair& key, const elm::EdgeValue& x) {
                (*self)[key] = x;
            }
            void __delitem__(const elm::cellcodepair& key) throw (std::out_of_range) {
                std::map<elm::cellcodepair, elm::EdgeValue>::iterator i = self->find(key);
                if (i != self->end())
                    self->erase(i);
					else
						throw std::out_of_range("key not found");
            }
            bool __contains__(const elm::cellcodepair& key) {
                std::map<elm::cellcodepair, elm::EdgeValue>::iterator i = self->find(key);
                return i != self->end();
            }
			int __len__() const {
				return self->size();
			}
			void __call__(elm::cellcode upcode, elm::cellcode dncode) {
				elm::cellcodepair x (upcode,dncode);
                (*self)[x] = elm::EdgeValue(COMPONENTLIST_TYPE_EDGE, self->parentmodel);
			}
        }
		#endif // def SWIG
	};
	

	struct ComponentGraphDNA {
	
		const Facet* db;
		const ComponentCellcodeMap* nodes;
		const ComponentEdgeMap* edges;
		
		ComponentGraphDNA(const ComponentCellcodeMap* nodes=nullptr, const ComponentEdgeMap* edges=nullptr, const Facet* db=nullptr);
		ComponentGraphDNA(const ComponentGraphDNA&);
		bool operator==(const ComponentGraphDNA&) const;
		
		bool valid() const {return (nodes && edges);}
		
		std::string node_name(const elm::cellcode& node_code) const;
		elm::cellcode node_code(const std::string& node_name) const;

		std::string __repr__() const;
		
		elm::cellcodeset elemental_codes() const;
		elm::cellcodeset all_node_codes() const;
		elm::cellcodeset nest_node_codes() const;
		
		std::vector<std::string> elemental_names() const;
		std::vector<std::string> all_node_names() const;
		std::vector<std::string> nest_node_names() const;
		
		elm::cellcodeset dn_node_codes(const elm::cellcode& node_code) const;
		elm::cellcodeset up_node_codes(const elm::cellcode& node_code, bool include_implicit_root=true) const;

		elm::cellcodeset chain_dn_node_codes(const elm::cellcode& node_code) const;
		elm::cellcodeset chain_up_node_codes(const elm::cellcode& node_code) const;

		#ifndef SWIG
		std::list<elm::cellcode> branches_ascending_order(etk::logging_service* msg=nullptr) const;
		std::list<elm::cellcode> nodes_ascending_order() const;
		#endif
		
		#ifdef SWIG
		%pythoncode %{
		def node_callsign(self, altcode):
			try:
				return "%i: %s"%(altcode, self.node_name(altcode))
			except LarchError:
				return "%i: %s"%(altcode, "alt_%i"%(altcode))
		def elemental_callsigns(self):
			return [self.node_callsign(j) for j in self.elemental_codes()]
		def all_node_callsigns(self):
			return [self.node_callsign(j) for j in self.all_node_codes()]
		def nest_node_callsigns(self):
			return [self.node_callsign(j) for j in self.nest_node_codes()]
		def dn_node_callsigns(self,code):
			return [self.node_callsign(j) for j in self.dn_node_codes(code)]
		def up_node_callsigns(self,code):
			return [self.node_callsign(j) for j in self.up_node_codes(code)]
		def up_node_candidate_callsigns(self,code):
			candidates = self.nest_node_codes()
			candidates -= self.chain_dn_node_codes(code)
			candidates -= code
			candidates += 0
			return [self.node_callsign(j) for j in candidates]
		def dn_node_candidate_callsigns(self,code):
			candidates = self.all_node_codes()
			candidates -= 0
			candidates -= self.chain_up_node_codes(code)
			candidates -= code
			return [self.node_callsign(j) for j in candidates]
		%}
		#endif
	
	};
		
	
	
};





#ifdef SWIG
%pythoncode %{
def __ComponentList__call(self, *args, **kwargs):
	if (self._receiver_type==0):
		raise LarchError("ComponentList improperly initialized")
	elif (self._receiver_type & COMPONENTLIST_TYPE_UTILITYCA):
		self.receive_utility_ca(*args, **kwargs)
	elif (self._receiver_type & COMPONENTLIST_TYPE_UTILITYCO):
		if len(kwargs)>0 and len(args)==0:
			self.receive_utility_co_kwd(**kwargs)
		elif len(kwargs)==0 and len(args)>0:
			if len(args)<2: raise LarchError("ComponentList for co type requires at least two arguments: data and alt")
			self.receive_utility_co(*args)
		else:
			raise LarchError("ComponentList for co type requires all-or-none use of keyword arguments")
	else:
		raise LarchError("ComponentList Not Implemented for type %i list"%self._receiver_type)
	####if self.parentmodel:
	####	self.parentmodel.freshen()
ComponentList.__call__ = __ComponentList__call
del __ComponentList__call
ComponentList.__long_len = ComponentList.__len__
ComponentList.__len__ = lambda self: int(self.__long_len())

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





#endif


#endif // __ELM_INPUTSTORAGE_H__
