//
//  elm_inputstorage.h
//  Yggdrasil
//
//  Created by Jeffrey Newman on 12/21/12.
//
//

#include "etk.h"
#include "elm_inputstorage.h"
#include "elm_model2.h"
#include "elm_sql_scrape.h"
#include "elm_names.h"

#ifndef SIZE_T_MAX
#define	SIZE_T_MAX	ULONG_MAX	/* max value for a size_t */
#endif // ndef SIZE_T_MAX

#include <iostream>
#include <iomanip>

elm::Component::Component(std::string data, std::string param, double multiplier, PyObject* category)
: data_name(data)
, param_name(param)
, _altcode(cellcode_empty)
, _altname("")
, _upcode(cellcode_empty)
, _dncode(cellcode_empty)
, multiplier(multiplier)
{
	if (category) {
		Py_XINCREF(category);
		
		if (category == Py_None) {
			// do nothing
		}
	
		else if PyLong_Check(category) {
			_altcode = PyLong_AsLongLong(category);
		}
		
		else if PyUnicode_Check(category) {
			_altname = PyString_ExtractCppString(category);
		}
	
		else if PyTuple_Check(category) {
			int i = PyArg_ParseTuple(category, "LL", &_upcode, &_dncode);
			if (!i) OOPS("incompatible category type, tuple must be (int,int)");
		}
		
		else {
			OOPS("incompatible category type, must be int, str, or tuple(int,int)");
		}

	
		Py_XDECREF(category);
	}
}

elm::Component::~Component()
{
}

//elm::Component::Component(const elm::Component& x)
//: data_name(x.data_name)
//, param_name(x.param_name)
//, altcode(x.altcode)
//, altname(x.altname)
//, multiplier(x.multiplier)
//{}

elm::Component elm::Component::Create(PyObject* obj)
{
	elm::Component x;
	
	const char* d =nullptr;
	const char* p =nullptr;
	const char* a =nullptr;
		
	if (!PyArg_ParseTuple(obj, "s|sKsd", &d, &p, &(x._altcode), &a, &(x.multiplier))) OOPS("Error reading ModelComponent");
	if (d) x.data_name = d;
	if (p) x.param_name = p;
	if (a) x._altname = a;
	
	return x;
}

std::string elm::Component::__repr__() const
{
	std::ostringstream x;
	x << "Component(";
	bool comma = false;
	if (!data_name.empty()) {
		x<<"data='"<<data_name<<"'";
		comma = true;
	}
	if (!_altname.empty()) {
		if (comma) x << ", ";
		x<<"altname='"<<_altname<<"'";
		comma = true;
	}
	if (_altcode != cellcode_empty) {
		if (comma) x << ", ";
		x<<"altcode="<<_altcode;
		comma = true;
	}
	if (!param_name.empty()) {
		if (comma) x << ", ";
		x<<"param='"<<param_name<<"'";
		comma = true;
	}
	if (multiplier != 1.0) {
		if (comma) x << ", ";
		x<<"multiplier="<<multiplier;
		comma = true;
	}
	x << ")";
	return x.str();
}

elm::ComponentList::ComponentList(int type, elm::Model2* parentmodel)
: _receiver_type (type)
, parentmodel(parentmodel)
{

}

std::string elm::ComponentList::__repr__() const
{
	std::ostringstream x;
	x << "<ComponentList with length "<< size() <<">";
	return x.str();
}

void elm::ComponentList::receive_utility_ca(const std::string& column_name, 
							    std::string freedom_name, 
							    const double& freedom_multiplier)

{
	if (freedom_name=="") freedom_name = column_name;
	if (parentmodel && etk::to_uppercase(freedom_name)!="CONSTANT" && !parentmodel->__contains__(freedom_name)) {
		parentmodel->parameter(freedom_name);
	}
	
	Component x;
	x.data_name = column_name;
	x.param_name = freedom_name;
	x.multiplier = freedom_multiplier;
	push_back( x );
	if (parentmodel && parentmodel->_Data) {
		if (parentmodel) BUGGER(parentmodel->msg) << "checking for validity of "<<column_name<<" in idCA data";
		parentmodel->_Data->check_ca(column_name);
	}
	if (parentmodel) {
		INFO(parentmodel->msg) << "success: added "<<column_name;
		//parentmodel->freshen();
	}
	
}















void elm::ComponentList::receive_utility_co
(const std::string&    column_name,
 const std::string&    alt_name, 
 std::string           freedom_name, 
 const double&         freedom_multiplier)
{
	return receive_utility_co_kwd(column_name, alt_name, cellcode_empty, freedom_name, freedom_multiplier);
}

void elm::ComponentList::receive_utility_co
(const std::string&    column_name, 
 const elm::cellcode&  alt_code, 
 std::string           freedom_name, 
 const double&        freedom_multiplier)
{
	return receive_utility_co_kwd(column_name, "", alt_code, freedom_name, freedom_multiplier);
}


void elm::ComponentList::receive_utility_co_kwd
(const std::string&    column_name, 
 const std::string&    alt_name, 
 const elm::cellcode&      alt_code,
 std::string           freedom_name, 
 const double&         freedom_multiplier)
{
	unsigned slot;
	std::string variable_name = column_name;
	if (!alt_name.empty()) {
		variable_name = etk::cat(column_name, "@", alt_name);
	} else {
		variable_name = etk::cat(column_name, "#", alt_code);
	}
	if (freedom_name=="") {
		freedom_name = variable_name;
	}
	if (parentmodel && etk::to_uppercase(freedom_name)!="CONSTANT" && !parentmodel->__contains__(freedom_name)) {
		parentmodel->parameter(freedom_name);
	}

	Component x;
	x.data_name = column_name;
	x.param_name = freedom_name;
	x.multiplier = freedom_multiplier;
	x._altcode = alt_code;
	x._altname = alt_name;
	push_back( x );
	
	if (parentmodel && parentmodel->_Data) {
		BUGGER(parentmodel->msg) << "checking for validity of "<<column_name<<" in idCO data";
		parentmodel->_Data->check_co(column_name);
		if (!alt_name.empty()) {
			slot = parentmodel->_Data->DataDNA()->slot_from_name(alt_name);
		} else {
			if (alt_code==cellcode_empty) {
				OOPS("utility.co input requires that you specify an alternative.");
			}
			BUGGER(parentmodel->msg) << "parentmodel->_Data->source_filename=" << parentmodel->_Data->source_filename;
			slot = parentmodel->_Data->DataDNA()->slot_from_code(alt_code);
		}
	}

	if (parentmodel) {
		INFO(parentmodel->msg) << "success: added "<<column_name;
		//parentmodel->freshen();
	}

}


std::vector<std::string> elm::ComponentList::needs() const
{
	std::vector<std::string> u_ca;
	
	for (unsigned b=0; b<size(); b++) {
		etk::push_back_if_unique(u_ca, (*this)[b].data_name);
	}
	
	return u_ca;
}



elm::ComponentCellcodeMap::ComponentCellcodeMap(int type, elm::Model2* parentmodel)
: _receiver_type (type)
, parentmodel(parentmodel)
{

}

std::string elm::ComponentCellcodeMap::__repr__() const
{
	if (size()==0) {
		return "<larch.core.ComponentCellcodeMap (empty)>";
	}
	
	std::ostringstream x;
	
	int minwide_code = 0;
	int minwide_name = 0;
	
	for (elm::ComponentCellcodeMap::const_iterator a=begin(); a!=end(); a++) {
		std::string temp = etk::cat(a->first);
		if (temp.size() > minwide_code) minwide_code = temp.size();
		temp = a->second._altname;
		if (temp.size() > minwide_name) minwide_name = temp.size();
	}
	
	for (elm::ComponentCellcodeMap::const_iterator a=begin(); a!=end(); a++) {
		x << "\n["<< std::setw(minwide_code) <<a->first<<"] "
		  << std::setw(minwide_name) << std::left
		  << a->second._altname << std::right
		  << " { mu = "<< a->second.param_name;
		if (a->second.multiplier != 1.0) {
			x << " * "<<a->second.multiplier;
		}
		x << " }";
	}
	std::string z = x.str();
	
	return z.substr(1);
}


elm::ComponentListPair::ComponentListPair(int type1, int type2, std::string descrip, elm::Model2* parentmodel)
: ca (type1, parentmodel)
, co (type2, parentmodel)
, descrip(descrip)
{

}




void elm::ComponentListPair::__call__(const elm::cellcode& altcode, std::string param, const double& multiplier)
{
	if (param=="") {
		param = etk::cat(descrip,"#",altcode);
	}
	co.receive_utility_co ("1", altcode, param, multiplier);
}


std::string elm::ComponentListPair::__repr__() const
{
	std::ostringstream x;
	x<< "<LinearFunction with length ("<< ca.size()<<"," << co.size() <<")>";
	return x.str();
}

void elm::ComponentListPair::clean(elm::Facet& db)
{
	
	for (elm::ComponentList::iterator i=ca.begin(); i!=ca.end(); i++) {
		try {
			db.check_ca(i->data_name);
		} SPOO {
			i = ca.erase(i);
			i--;
		}
	}
	for (elm::ComponentList::iterator i=co.begin(); i!=co.end(); i++) {
		try {
			db.check_co(i->data_name);
		} SPOO {
			i = co.erase(i);
			i--;
		}
	}

}


elm::ComponentEdgeMap::ComponentEdgeMap(elm::Model2* parentmodel)
: _receiver_type (COMPONENTLIST_TYPE_EDGE)
, parentmodel(parentmodel)
{

}

std::string elm::ComponentEdgeMap::__repr__() const
{
	if (size()==0) {
		return "<larch.core.ComponentEdgeMap (empty)>";
	}

	std::ostringstream x;
	
	int minwide_code1 = 0;
	int minwide_code2 = 0;
	int minwide_name = 0;
	
	for (elm::ComponentEdgeMap::const_iterator a=begin(); a!=end(); a++) {
		std::string temp = etk::cat(a->first.up);
		if (temp.size() > minwide_code1) minwide_code1 = temp.size();
		temp = etk::cat(a->first.dn);
		if (temp.size() > minwide_code2) minwide_code2 = temp.size();
	}
	
	for (elm::ComponentEdgeMap::const_iterator a=begin(); a!=end(); a++) {
		x	<< "\n["<< std::setw(minwide_code1) <<a->first.up<<" --> "
			<< std::setw(minwide_code2) << std::left << a->first.dn << std::right << "] "
			<< a->second.__repr__();
	}
	std::string z = x.str();
	
	return z.substr(1);
}








elm::ComponentGraphDNA::ComponentGraphDNA(const ComponentCellcodeMap* nodes, const ComponentEdgeMap* edges, const Facet* db, const elm::cellcode* root)
: db(db)
, nodes(nodes)
, edges(edges)
, root_code(root ? *root : cellcode_null)
{}

elm::ComponentGraphDNA::ComponentGraphDNA(const ComponentGraphDNA& x)
: db(x.db)
, nodes(x.nodes)
, edges(x.edges)
, root_code(x.root_code)
{}

bool elm::ComponentGraphDNA::operator==(const ComponentGraphDNA& x) const
{
	if (db!=x.db || nodes!=x.nodes || edges!=x.edges || root_code!=x.root_code) return false;
	return true;
}


std::string elm::ComponentGraphDNA::node_name(const elm::cellcode& node_code) const
{
	if (node_code==root_code) return "ROOT";
	
	// Look in nodes
	if (nodes) {
		elm::ComponentCellcodeMap::const_iterator i = nodes->find(node_code);
		if (i != nodes->end()) {
			return i->second._altname;
		}
	}
	
	// Look in db
	if (db) {
		size_t k = etk::find_first(node_code, db->alternative_codes());
		if (k != SIZE_T_MAX) {
			return db->alternative_names()[k];
		}
	}
	
	OOPS("node code ",node_code," not found");
}

elm::cellcode elm::ComponentGraphDNA::node_code(const std::string& node_name) const
{
	if (node_name=="ROOT") return root_code;

	// Look in nodes
	if (nodes) {
		elm::ComponentCellcodeMap::const_iterator i = nodes->begin();
		while (i != nodes->end()) {
			if (i->second._altname == node_name) return i->first;
			i++;
		}
	}
	
	// Look in db
	if (db) {
		size_t k = etk::find_first(node_name, db->alternative_names());
		if (k != SIZE_T_MAX) {
			return db->alternative_codes()[k];
		}
	}
	
	OOPS("node name '",node_name,"' not found");
}


elm::cellcodeset elm::ComponentGraphDNA::elemental_codes() const
{
	elm::cellcodeset ret;
	
	if (db) {
		std::vector<elm::cellcode> x = db->alternative_codes();
		ret.insert(x.begin(), x.end());
		return ret;
	}
	
	if (edges) {
		elm::cellcodeset candidates;
		elm::cellcodeset noncandidates;
		ComponentEdgeMap::const_iterator i = edges->begin();
		while (i!=edges->end()) {
			// up nodes become non-candidates
			noncandidates.insert(i->first.up);
			candidates.erase(i->first.up);
			// dn nodes become candidates if they are not already non-candidates
			if (!noncandidates.count(i->first.dn)) {
				candidates.insert(i->first.dn);
			}
			i++;
		}
		ret.insert_set(candidates);
		return ret;
	}
	
	OOPS("error in finding elemental_codes");
}

std::vector<std::string> elm::ComponentGraphDNA::elemental_names() const
{
	elm::cellcodeset x = elemental_codes();
	std::vector<std::string> ret;
	
	for (std::set<cellcode>::iterator i=x.begin(); i!=x.end(); i++) {
		ret.push_back(node_name(*i));
	}
	return ret;
}


elm::cellcodeset elm::ComponentGraphDNA::all_node_codes() const
{
	elm::cellcodeset ret = elemental_codes();
	
	if (nodes) {
		ComponentCellcodeMap::const_iterator i = nodes->begin();
		while (i!=nodes->end()) {
			ret.insert(i->first);
			i++;
		}
	}
	
	if (edges) {
		ComponentEdgeMap::const_iterator i = edges->begin();
		while (i!=edges->end()) {
			ret.insert(i->first.up);
			ret.insert(i->first.dn);
			i++;
		}
	}

	ret.insert(root_code);
	return ret;
	
	OOPS("error in finding all_node_codes");
}

std::vector<std::string> elm::ComponentGraphDNA::all_node_names() const
{
	elm::cellcodeset x = all_node_codes();
	std::vector<std::string> ret;
	
	for (std::set<cellcode>::iterator i=x.begin(); i!=x.end(); i++) {
		ret.push_back(node_name(*i));
	}
	return ret;
}

elm::cellcodeset elm::ComponentGraphDNA::nest_node_codes() const
{
	elm::cellcodeset all = all_node_codes();
	elm::cellcodeset elem = elemental_codes();
	elm::cellcodeset ret;
	for (std::set<cellcode>::iterator i=all.begin(); i!=all.end(); i++) {
		if (!elem.contains(*i) && *i!=root_code) ret.append(*i);
	}
	return ret;
}

std::vector<std::string> elm::ComponentGraphDNA::nest_node_names() const
{
	elm::cellcodeset x = nest_node_codes();
	std::vector<std::string> ret;
	
	for (std::set<cellcode>::iterator i=x.begin(); i!=x.end(); i++) {
		ret.push_back(node_name(*i));
	}
	return ret;
}


elm::cellcodeset elm::ComponentGraphDNA::dn_node_codes(const elm::cellcode& node_code) const
{
	elm::cellcodeset ret;
	
	if (elemental_codes().count(node_code)) return ret;
	
	if (edges) {
		if (node_code==0) {
			elm::cellcodeset candidates;
			elm::cellcodeset noncandidates;
			ComponentEdgeMap::const_iterator i = edges->begin();
			while (i!=edges->end()) {
				if (i->first.up == node_code) {
					ret.insert(i->first.dn);
				} else {
					// down nodes become non-candidates
					noncandidates.insert(i->first.dn);
					candidates.erase(i->first.dn);
					// up nodes become candidates if they are not already non-candidates
					if (!noncandidates.count(i->first.up)) {
						candidates.insert(i->first.up);
					}
				}
				i++;
			}
			ret.insert_set(candidates);
			return ret;
		} else {
			ComponentEdgeMap::const_iterator i = edges->begin();
			while (i!=edges->end()) {
				if (i->first.up == node_code) ret.insert(i->first.dn);
				i++;
			}
			return ret;
		}
	}
	
	OOPS("error in finding down codes");
}


elm::cellcodeset elm::ComponentGraphDNA::up_node_codes(const elm::cellcode& node_code, bool include_implicit_root) const
{
	elm::cellcodeset ret;
	
	if (node_code==root_code) {
		return ret;
	}
	
	if (edges) {
		ComponentEdgeMap::const_iterator i = edges->begin();
		while (i!=edges->end()) {
			if (i->first.dn == node_code) ret.insert(i->first.up);
			i++;
		}
	}
	
	if (ret.size()==0 && include_implicit_root) ret.insert(root_code);
	return ret;
	
	OOPS("error in finding up codes");
}

elm::cellcodeset elm::ComponentGraphDNA::chain_dn_node_codes(const elm::cellcode& node_code) const
{
	elm::cellcodeset dns = dn_node_codes(node_code);
	elm::cellcodeset ret = dn_node_codes(node_code);
	
	for (std::set<cellcode>::iterator i=dns.begin(); i!=dns.end(); i++) {
		ret.insert_set(chain_dn_node_codes(*i));
	}
	
	return ret;
}
elm::cellcodeset elm::ComponentGraphDNA::chain_up_node_codes(const elm::cellcode& node_code) const
{
	elm::cellcodeset ups = up_node_codes(node_code);
	elm::cellcodeset ret = up_node_codes(node_code);
	
	for (std::set<cellcode>::iterator i=ups.begin(); i!=ups.end(); i++) {
		ret.insert_set(chain_up_node_codes(*i));
	}
	
	return ret;
}


std::string elm::ComponentGraphDNA::__repr__() const
{
	std::ostringstream x;
	
	int minwide_code = 0;
	int minwide_name = 0;
	
	std::list<elm::cellcode> all = nodes_ascending_order();
	for (std::list<elm::cellcode>::reverse_iterator a=all.rbegin(); a!=all.rend(); a++) {
		std::string temp = etk::cat(*a);
		if (temp.size() > minwide_code) minwide_code = temp.size();
		temp = node_name(*a);
		if (temp.size() > minwide_name) minwide_name = temp.size();
	}
	
	
	elm::cellcodeset elem = elemental_codes();
	elm::cellcodeset implieds;
	for (std::list<elm::cellcode>::reverse_iterator a=all.rbegin(); a!=all.rend(); a++) {
		x << "\n["<< std::setw(minwide_code) <<*a<<"] "<< std::setw(minwide_name) << std::left <<node_name(*a) << std::right;
		if (elem.count(*a)) {
			x << " <*>";
		} else {
			x << " --> (";
			elm::cellcodeset dns = dn_node_codes(*a);
			if (dns.size()) {
				std::set<cellcode>::iterator d = dns.begin();
				x << *d;
				d++;
				while (d!=dns.end()) {
					x << ", "<<*d;
					d++;
				}
			}
			if (*a==root_code) {
				for (std::list<elm::cellcode>::iterator y=all.begin(); y!=all.end(); y++) {
					if (!dns.contains(*y) && up_node_codes(*y).contains(0)) {
						implieds += *y;
					}
				}
			}
			if (implieds.size()) {
				if (dns.size()) {
					x << ", ";
				}
				x << "[";
				std::set<elm::cellcode>::iterator y = implieds.begin();
				x << *y;
				y++;
				while (y!=implieds.end()) {
					x << ", " << *y;
					y++;
				}
				x << "]";
				implieds.clear();
			}
			x << ")";
		}
		
	}
	std::string z = x.str();
	
	if (z.size()<=1) {
		return "<larch.core.ComponentGraphDNA (empty)>";
	}
	
	return z.substr(1);
}




std::list<elm::cellcode> elm::ComponentGraphDNA::branches_ascending_order(etk::logging_service* msg) const
{
	BUGGER_(msg, "Getting branches in ascending order...");
	cellcodeset Elementals = elemental_codes();
	cellcodeset all = all_node_codes();
	ComponentCellcodeMap::const_iterator b;
	std::list<elm::cellcode> Branches;
	std::list<elm::cellcode>::iterator hop;
	
	std::map< elm::cellcode, elm::cellcodeset > down_nodes;
	BUGGER_(msg, "Mapping all down nodes...");
	for (std::set<elm::cellcode>::iterator a=all.begin(); a!=all.end(); a++) {
		down_nodes[*a] = dn_node_codes(*a);
	}
	
	// Sort Branches into an ascending order
	BUGGER_(msg, "Setting default root node...");
	ComponentCellcodeMap::const_iterator root_iter = nodes->end();
	BUGGER_(msg, "Sorting "<<nodes->size()<<" branches...");
	int b_count = 0;
	for (b = nodes->begin(); b!=nodes->end(); b++) {
		BUGGER_(msg, "Branch:"<<b_count++);
		BUGGER_(msg, "    is:"<<b->first);
		if (b->first == 0) {
			root_iter = b;
			continue;
		}
		if (Elementals.contains(b->first)) continue;
		hop = Branches.begin();
		while ( hop!=Branches.end() && !down_nodes[*hop].contains(b->first) ) hop++;
		BUGGER_(msg, "  inserting "<<b->first);
		Branches.insert(hop,b->first);
	}
	if (Branches.size()==0 || Branches.back() != root_code) {
		Branches.insert(Branches.end(),root_code);
	}
	return Branches;
}

std::list<elm::cellcode> elm::ComponentGraphDNA::nodes_ascending_order() const
{	
	cellcodeset Elementals = elemental_codes();
	std::list<elm::cellcode> Branches = branches_ascending_order();
	
	for (std::set<elm::cellcode>::reverse_iterator i=Elementals.rbegin(); i!=Elementals.rend(); i++) {
		Branches.insert(Branches.begin(), *i);
	}

	return Branches;
}

