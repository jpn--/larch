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
%include "elm_inputstorage.i"
#endif // def SWIG

#include "elm_cellcode.h"

namespace elm {
	
	class Model2;
	class Facet;
	class Fountain;
	class ComponentList;
		
	struct LinearComponent {
		std::string		data_name;
		std::string     param_name;
		
		elm::cellcode	_altcode;
		std::string	    _altname;

		elm::cellcode	_upcode;
		elm::cellcode	_dncode;

		double			multiplier;

		std::string __repr__() const;
		std::string __str__() const;
		
		LinearComponent(std::string data="", std::string param="", double multiplier=1.0, PyObject* category=nullptr);
		static LinearComponent Create(PyObject* obj);
		~LinearComponent();
//		LinearComponent(const LinearComponent& obj);

		ComponentList operator+(const LinearComponent& other) const;
		ComponentList operator+(const ComponentList& other) const;
		ComponentList operator+(const int& other) const;

		#ifdef SWIG
		%pythoncode %{
		def altcallsign(self):
			return "%i: %s"%(self.altcode, self.altname)
		def set_from_callsign(self, x):
			c,n = x.split(": ")
			self.altcode = long(c)
			self.altname = n
		
		@property
		def data(self):
			from .roles import DataRef
			return DataRef(self._data)
		@data.setter
		def data(self, value):
			self._data = value
			
		@property
		def param(self):
			from .roles import ParameterRef
			return ParameterRef(self._param)
		@param.setter
		def param(self, value):
			self._param = value

		def __pos__(self):
			return self
		
		_add = __add__
		def __add__(self, other):
			if other==():
				return self
			return self._add(other)
			
		def __mul__(self,other):
			from .roles import DataRef
			if isinstance(other,(int,float,DataRef)):
				return LinearComponent(data=self.data * other, param=self.param)
			else:
				raise TypeError('unsupported operand type(s) for LinearComponent*: {}'.format(type(other)))
		
		def __rmul__(self,other):
			from .roles import DataRef
			if isinstance(other,(int,float,DataRef)):
				return LinearComponent(data=self.data * other, param=self.param)
			else:
				raise TypeError('unsupported operand type(s) for LinearComponent*: {}'.format(type(other)))

		def __imul__(self,other):
			from .roles import DataRef
			if isinstance(other,(int,float,DataRef)):
				self.data = self.data * other
			else:
				raise TypeError('unsupported operand type(s) for LinearComponent*: {}'.format(type(other)))

		def __iter__(self):
			return iter(LinearFunction() + self)

		%}
		#endif // SWIG

		
		

	};
	

};



#define COMPONENTLIST_TYPE_UTILITYCA 0x1
#define COMPONENTLIST_TYPE_UTILITYCO 0x2
#define COMPONENTLIST_TYPE_LOGSUM    0x4
#define COMPONENTLIST_TYPE_EDGE      0x8
#define COMPONENTLIST_TYPE_SIMPLECO  0x8

namespace elm {
	
	class ComponentList
	: public ::std::vector<LinearComponent>
	{
		public:
		int _receiver_type;
		Model2* parentmodel;
		ComponentList(int type=0, Model2* parentmodel=nullptr);
		
		
		void receive_utility_ca(const std::string& data,
							    std::string param="",
							    const double& multiplier=1.0);
		void receive_allocation(const std::string& data,
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

		std::string __str__() const;
		std::string __repr__() const;
		std::string __indent_repr__(int indent) const;

		std::vector<std::string> needs() const;

		ComponentList _add(const LinearComponent& x) const;
		ComponentList _add(const ComponentList& x) const;
		ComponentList _add(const int& x) const;
		#ifndef SWIG
		ComponentList operator+(const LinearComponent& x);
		ComponentList operator+(const ComponentList& x);
		ComponentList& operator+=(const LinearComponent& x);
		ComponentList& operator+=(const ComponentList& x);
		#endif // ndef SWIG
		
		void _inplace_add(const LinearComponent& x);
		void _inplace_add(const ComponentList& x);

		#ifdef SWIG
		%pythoncode %{
		def __add__(self, other):
			if other==():
				return self
			return self._add(other)
		def __iadd__(self, other):
			if other==():
				return self
			self._inplace_add(other)
			return self
		def __radd__(self, other):
			return LinearFunction() + other + self
		def __pos__(self):
			return self
		def __mul__(self, other):
			trial = LinearFunction()
			for component in self:
				trial += component * other
			return trial
		def __rmul__(self, other):
			trial = LinearFunction()
			for component in self:
				trial += other * component
			return trial
		def evaluate(self, dataspace, model):
			if len(self)>0:
				i = self[0]
				y = i.data.eval(**dataspace) * i.param.default_value(0).value(model)
			for i in self[1:]:
				y += i.data.eval(**dataspace) * i.param.default_value(0).value(model)
			return y
		def evaluator1d(self, factorlabel='U', dimlabel=None):
			if dimlabel is None:
				try:
					dimlabel = self._dimlabel
				except AttributeError:
					raise TypeError('a dimlabel must be given')
			if dimlabel is None:
				raise TypeError('a dimlabel must be given')
			from .util.plotting import ComputedFactor
			return ComputedFactor( label=factorlabel, func=lambda x,m: self.evaluate({dimlabel:x}, m) )
		def __contains__(self, val):
			from .roles import ParameterRef, DataRef
			if isinstance(val, ParameterRef):
				for i in self:
					if i.param==val:
						return True
				return False
			if isinstance(val, DataRef):
				for i in self:
					if i.data==val:
						return True
				return False
			raise TypeError("the searched for content must be of type ParameterRef or DataRef")
		%}
		#endif // def SWIG


	};



	class ComponentCellcodeMap
	#ifndef SWIG
	: public ::std::map<elm::cellcode, elm::LinearComponent>
	#endif
	{
		public:
		#ifndef SWIG
		int _receiver_type;
		Model2* parentmodel;
		#endif // ndef swig
		ComponentCellcodeMap(int type=0, Model2* parentmodel=nullptr);

		std::string __repr__() const;

		void _create(const std::string& nest_name, const elm::cellcode& nest_code,
						 std::string param_name="", const double& param_multiplier=1.0);

		#ifdef SWIG
        unsigned int size() const;
        bool empty() const;
        void clear();
		
		
		%extend {
            elm::LinearComponent& __getitem__(const elm::cellcode& key) throw (std::out_of_range) {
                std::map<elm::cellcode,elm::LinearComponent >::iterator i = self->find(key);
                if (i != self->end())
                    return i->second;
                else
                    throw std::out_of_range("key not found");
            }
            void __setitem__(const elm::cellcode& key, const elm::LinearComponent& x) {
                (*self)[key] = x;
            }
            void __delitem__(const elm::cellcode& key) throw (std::out_of_range) {
                std::map<elm::cellcode,elm::LinearComponent >::iterator i = self->find(key);
                if (i != self->end())
                    self->erase(i);
					else
						throw std::out_of_range("key not found");
            }
            bool __contains__(const elm::cellcode& key) {
                std::map<elm::cellcode,elm::LinearComponent >::iterator i = self->find(key);
                return i != self->end();
            }
			int __len__() const {
				return self->size();
			}
			void _link(const elm::cellcode& parent, const elm::cellcode& child) {
				if (self->parentmodel) {
					self->parentmodel->Input_Edges.__call__(parent,child);
				} else {
					throw etk::exception_t("not linked to a model");
				}
			}
			std::vector<elm::cellcode> nodes() const {
				std::vector<elm::cellcode> it;
				for (auto i=self->begin(); i!=self->end(); i++) {
					it.push_back(i->first);
				}
				return it;
			}
        }
		#endif // def SWIG
	};




	class LinearCOBundle_1
	: public ::std::map<elm::cellcode, elm::ComponentList>
	{
		
	public:
		elm::Model2* parentmodel;
	
		LinearCOBundle_1(elm::Model2* parent=nullptr);
		LinearCOBundle_1(const LinearCOBundle_1& other);
	
		size_t metasize() const;
		std::string __str__() const;
		std::string __repr__() const;
		std::vector<std::string> needs() const;
		elm::ComponentList& add_blank(const elm::cellcode& i);
		
		void _call (const elm::cellcode& altcode,
					   const std::string& data,
					   std::string param="",
					   const double& multiplier=1.0) ;

		#ifdef SWIG
		%pythoncode %{
		def __call__(self, altcode, data, param="", multiplier=1.0):
			if isinstance(altcode, str) and isinstance(data, int):
				_ = data
				data = altcode
				altcode = _
			if isinstance(altcode, str) and isinstance(data, str):
				try:
					a = DB.alternatives(self.parentmodel, 'reversedict')
					if altcode in a:
						altcode = a[altcode]
					elif data in a:
						_ = a[data]
						data = altcode
						altcode = _
				except AttributeError:
					raise TypeError('cannot identify alternative')
			self._call(altcode, data, param, multiplier)
			
		def __setitem__(self, key, value):
			parent = self.parentmodel
			if parent is not None and not parent.option.autocreate_parameters:
				if isinstance(value, LinearFunction):
					for i in value:
						if i.param not in parent:
							raise KeyError("Parameter '{}' is not found in model and autocreate_parameters is off".format(i.param))
				if isinstance(value, LinearComponent):
					if value.param not in parent:
						raise KeyError("Parameter '{}' is not found in model and autocreate_parameters is off".format(value.param))
			from .roles import ParameterRef, DataRef
			if value is None:
				value = LinearFunction()
			if isinstance(value, (int,float)):
				if value==0:
					value = LinearFunction()
				else:
					raise TypeError("Assigning a nonzero fixed value to a utility function is not supported, try using a holdfast parameter instead")
			if isinstance(value, DataRef):
				value = LinearComponent(data=str(value))
			if isinstance(value, ParameterRef):
				value = LinearComponent(param=value, data='1')
			if isinstance(value, LinearComponent):
				value = LinearFunction() + value
			return super().__setitem__(key, value)
		%}
		#endif // def SWIG
		
		
	};


	struct LinearBundle_1 {
		std::string descrip;
		
		#ifndef SWIG
		ComponentList     ca;
		LinearCOBundle_1 co;
		#endif
		
		LinearBundle_1(std::string descrip="", Model2* parentmodel=nullptr);
		
		void __call__(std::string data="", std::string param="", const double& multiplier=1.0);
		void clean(Facet& db);

		std::string __baserepr__() const;
		
		void _set_ca(const LinearComponent& x);
		void _set_ca(const ComponentList& x);
		void _set_co(const LinearCOBundle_1& x);
		ComponentList&  _get_ca();
		LinearCOBundle_1*       _get_co();
		
		#ifdef SWIG
		%pythoncode %{
		ca = property(lambda self: self._get_ca(), lambda self,x: self._set_ca(x))
		co = property(lambda self: self._get_co(), lambda self,x: self._set_co(x))
		
		def __getitem__(self, key):
			return self._get_co().add_blank(key)
		def __setitem__(self, key, value):
			self._get_co()[key] = value
		def __delitem__(self, key):
			del self._get_co()[key]
		def __repr__(self):
			r = self.__baserepr__()
			r += "\n ca: "+"\n     ".join(repr(self.ca).split("\n"))
			r += "\n co: "+"\n     ".join(repr(self.co).split("\n"))
			return r
		%}
		#endif // def SWIG

		
	};










	
	
	struct ComponentListPair {
		std::string descrip;
		
		#ifndef SWIG
		ComponentList ca;
		ComponentList co;
		#endif
		
		ComponentListPair(int type1=0, int type2=0, std::string descrip="", Model2* parentmodel=nullptr);
		
		void __call__(const elm::cellcode& altcode, std::string param="", const double& multiplier=1.0);
		void clean(Facet& db);

		std::string __repr__() const;
		
		void _set_ca(const LinearComponent& x);
		void _set_co(const LinearComponent& x);
		void _set_ca(const ComponentList& x);
		void _set_co(const ComponentList& x);
		ComponentList&  _get_ca();
		ComponentList*  _get_co();
		
		#ifdef SWIG
		%pythoncode %{
		def _set_ca_1(self,x):
			#//if type(x) is ParameterRef: x = LinearComponent(param=str(x), data="1")
			self._set_ca(x)
		def _set_co_1(self,x):
			#//if type(x) is ParameterRef: x = LinearComponent(param=str(x), data="1")
			self._set_co(x)
		ca = property(lambda self: self._get_ca(), lambda self,x: self._set_ca_1(x))
		co = property(lambda self: self._get_co(), lambda self,x: self._set_co_1(x))
		%}
		#endif // def SWIG

		
	};
	
	
	
	typedef elm::ComponentList EdgeValue;

	class LinearCOBundle_2
	#ifndef SWIG
	: public ::std::map<elm::cellcodepair, elm::EdgeValue>
	#endif
	{
		public:
		int _receiver_type;
		Model2* parentmodel;
		LinearCOBundle_2(Model2* parentmodel=nullptr);

		std::string __repr__() const;
		void __call__(elm::cellcode upcode, elm::cellcode dncode);

		std::vector< elm::cellcode > downlinks(const elm::cellcode& upcode) const;

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
			
			::std::vector<elm::cellcodepair> links () const {
				::std::vector<elm::cellcodepair> lks;
				std::map<elm::cellcodepair, elm::EdgeValue>::const_iterator i = self->begin();
				while (i!=self->end()) {
					lks.push_back(i->first);
					i++;
				}
				return lks;
			}
        }
		#endif // def SWIG
	};
	

	struct ComponentGraphDNA {
	
		const Fountain* db;
		const ComponentCellcodeMap* nodes;
		const LinearCOBundle_2* edges;
		
		ComponentGraphDNA(const ComponentCellcodeMap* nodes=nullptr, const LinearCOBundle_2* edges=nullptr, const Fountain* db=nullptr, const elm::cellcode* root=nullptr);
		ComponentGraphDNA(const ComponentGraphDNA&);
		bool operator==(const ComponentGraphDNA&) const;
		
		bool valid() const {return (nodes && edges);}
		
		std::string node_name(const elm::cellcode& node_code) const;
		elm::cellcode node_code(const std::string& node_name) const;
		elm::cellcode root_code;

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
%include "elm_inputstorage2.i"
#endif // def SWIG


#endif // __ELM_INPUTSTORAGE_H__
