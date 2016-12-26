/*
 *  elm_vascular.cpp
 *
 *  Copyright 2007-2016 Jeffrey Newman
 *
 *  Larch is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  Larch is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with Larch.  If not, see <http://www.gnu.org/licenses/>.
 *  
 */

#include "etk.h"
#include <list>
//#include <queue>
#include <climits>
#include "elm_vascular.h"
#include "etk_resultcodes.h"
#include "elm_sql_facet.h"

using namespace elm;
using namespace etk;
using namespace std;

bool VAS_dna_info::is_elemental() const
{
	if (dns.empty()) {
		return !is_branch;
	}
	return false;
}


VAS_dna_info::VAS_dna_info(const std::string& input)
: dns()
, name()
, is_branch (false)
//, color ()
//, fontcolor ()
//, fontname ()
//, type_name ()
//, shape ()

{ 

	std::string i = input;
	etk::trim(i,"[];,");
	std::map<std::string,std::string> parsed = etk::parse_option_string(i);
	
	name = parsed["label"];
//	color  = parsed["color"];
//	fontcolor  = parsed["fontcolor"];
//	fontname  = parsed["fontname"];
//	type_name  = parsed["type"];
//	shape = parsed["shape"];

	
}

VAS_dna::VAS_dna()
: _known_downs ()
{}

void VAS_dna::read_sequence (const string& seq)
{
	clear();
	add_sequence (seq);
}
void VAS_dna::add_sequence (const string& seq)
{
	string u,d;
	cellcode cu,cd;
	istringstream sq (seq);
	while (sq) {
		sq >> u;
		if (!sq) break;
		sq >> d;
		if (u == "0*") continue;
		cu = cellcode_from_string(u);
		cd = cellcode_from_string(d);
		(*this)[cu].dns.insert(cd);
		(*this)[cd].dns.noop();
		_known_downs.insert(cd);
	}	
}

void VAS_dna::add_sequence (const VAS_dna& dna)
{
	for (const_iterator t=dna.begin(); t!=dna.end(); t++) {
		add_cell(t->first,t->second.name,t->second.is_branch);
		for (elm::cellcode_set_citer j=t->second.dns.begin(); j!=t->second.dns.end(); j++) {
			add_edge(t->first,*j);
		}
	}
}

/*
 void VAS_dna::add_sequence_from_images()
 {
 cellcodeset::iterator i;
 map<cellcode,Vascular_Cell_Image*>::iterator im = _known_images.begin();
 while (im != _known_images.end()) {
 for (i= im->second->_dn_DNA.begin(); i!=im->second->_dn_DNA.end(); i++) {
 add_edge(im->first,*i);
 }
 for (i= im->second->_up_DNA.begin(); i!=im->second->_up_DNA.end(); i++) {
 add_edge(*i,im->first);
 }
 im++;
 }
 }
 */



string VAS_dna::generate_sequence() const
{
	ostringstream sq;
	for (const_iterator i=begin(); i!=end(); i++) {
		for (elm::cellcode_set_citer j=i->second.dns.begin(); j!=i->second.dns.end(); j++) {
			sq << CELLCODEasCSTR(i->first) << "\t" << CELLCODEasCSTR(*j) << "\n";
		}
	}
	return sq.str();
}
string VAS_dna::generate_phenotype() const
{
	ostringstream sq;
	for (const_iterator i=begin(); i!=end(); i++) {
		for (elm::cellcode_set_citer j=i->second.dns.begin(); j!=i->second.dns.end(); j++) {
			sq << "(" << CELLCODEasCSTR(i->first) << ")->(" << CELLCODEasCSTR(*j) << ")\n";
		}
	}
	for (const_iterator i=begin(); i!=end(); i++) {
		elm::cellcode_set_citer j = _known_downs.find(i->first);
		if (j==_known_downs.end()) {
			sq << "(0*)->(" << CELLCODEasCSTR(i->first) << ")\n";
		}
	}
	return sq.str();
}

string VAS_dna::add_edge(const cellcode& source, const cellcode& sink)
{
	ostringstream ret;
	iterator i = find(source);
	if (i==end()) {
		(*this)[source].dns.insert(sink);
		_known_downs.insert(sink);
		ret<<"success: created cell ("<<CELLCODEasCSTR(source)<<
				   ") and added edge ("<<CELLCODEasCSTR(source)<<")->("<<
				   CELLCODEasCSTR(sink)<<")";
		return ret.str();
	}
	std::set<cellcode>::iterator j = i->second.dns.find(sink);
	if (j==i->second.dns.end()) {
		i->second.dns.insert(sink);
		_known_downs.insert(sink);
		ret<<"success: added edge ("<<CELLCODEasCSTR(source)<<")->("<<
				   CELLCODEasCSTR(sink)<<")";
		return ret.str();
	}
	ret<<"ignored: edge ("<<CELLCODEasCSTR(source)<<")->("<<CELLCODEasCSTR(sink)<<") already exists";
	return ret.str();
}
string VAS_dna::remove_edge(const cellcode& source, const cellcode& sink)
{
	iterator i = find(source);
	if (i==end()) {
		return cat("ignored: source (",CELLCODEasCSTR(source),") absent");
	}
	std::set<cellcode>::iterator j = i->second.dns.find(sink);
	if (j==i->second.dns.end()) {
		return cat("ignored: edge (",CELLCODEasCSTR(source),")->(",CELLCODEasCSTR(sink),") already absent");
	}
	i->second.dns.erase(j);
	// TODO: Check for remove from known_downs.
	return cat("success: edge (",CELLCODEasCSTR(source),")->(",CELLCODEasCSTR(sink),") removed");
}

int VAS_dna::add_cell(const cellcode& cod, const string& nam, const bool& must_be_branch)
{
	iterator i = find(cod);
	if (i==end()) {
		(*this)[cod].name = nam;
		(*this)[cod].is_branch = must_be_branch;
		string s ("success: added cell ");
		s+= nam;
		s+= " (";
		s+= CELLCODEasCSTR(cod);
		s+= ")";
		return ELM_CREATED;
	}
	i->second.is_branch = must_be_branch;
	if (i->second.name != nam) {
		i->second.name = nam;
		string s ("success: changed name of existing cell (");
		s+= CELLCODEasCSTR(cod);
		s+= ") to ";
		s+= nam;
		return ELM_UPDATED;
	}
	return ELM_IGNORED;
}



void VAS_dna::clear()
{
	map<cellcode,VAS_dna_info>::clear();
	_known_downs.clear();
}

cellcodeset VAS_dna::elemental_codes() const
{
	cellcodeset Elements;
	for (VAS_dna::const_iterator f = begin(); f != end(); f++) {
		if (f->second.is_elemental()) Elements.insert(f->first);
	}
	return Elements;
}

cellcodeset VAS_dna::all_known_codes() const
{
	cellcodeset All;
	for (VAS_dna::const_iterator f = begin(); f!=end(); f++) {
		All.insert(f->first);
	}
	return All;
}



VAS_Edge::VAS_Edge (VAS_Cell*& uedge, VAS_Cell*& dedge)
:	_up (uedge)
,	_dn (dedge)
,	_edge_slot  (UINT_MAX)
,	_alloc_slot (UINT_MAX)
{ }

VAS_Edge::VAS_Edge (const VAS_Edge& x)
:	_up (x._up)
,	_dn (x._dn)
,	_edge_slot  (x._edge_slot)
,	_alloc_slot (x._alloc_slot)
{ }

const unsigned& VAS_Edge::alloc_slot() const 
{ 
	if (_alloc_slot==UINT_MAX) {
		OOPS("VAS_Edge::alloc_slot(): bad call");
	}
	return _alloc_slot; 
}





VAS_Edge& VAS_EdgeVec::operator[](const unsigned& i)
{
	return *(at(i));
}
const VAS_Edge& VAS_EdgeVec::operator[](const unsigned& i) const
{
	return *(at(i));
}
void VAS_EdgeVec::add_back(VAS_Cell*& uedge, VAS_Cell*& dedge)
{
	push_back(new VAS_Edge(uedge, dedge));
}
void VAS_EdgeVec::clear()
{
	for (unsigned i=0; i<size(); i++) {
		delete at(i);
		at(i) = NULL;
	}
	vector<VAS_Edge*>::clear();
}








VAS_Cell::VAS_Cell (const cellcode& c, const unsigned& slot_)
:	_cell_name ()
,	_cell_code (c)
,	_cell_slot (slot_)
,	_parameter (NULL)
,	_parameter_offset (UINT_MAX)
,	_contained_elementals()
{  }



VAS_Cell* VAS_Cell::upcell(const unsigned& i)
{
	if (i >= _ups.size()) OOPS("VAS_Cell::upcell: out of range");
	return _ups[i]->u();
}
VAS_Cell* VAS_Cell::dncell(const unsigned& i)
{
	if (i >= _dns.size()) OOPS("VAS_Cell::dncell: out of range");
	return _dns[i]->d();
}
VAS_Edge* VAS_Cell::upedge(const unsigned& i)
{
	if (i >= _ups.size()) {
		OOPS("VAS_Cell::upedge: index ",i," out of range that tops out at ",_ups.size());
	}
	return _ups[i];
}
VAS_Edge* VAS_Cell::dnedge(const unsigned& i)
{
	if (i >= _dns.size()) OOPS("VAS_Cell::dnedge: index ",i," out of range that tops out at ",_dns.size());
	return _dns[i];
}

const VAS_Cell* VAS_Cell::upcell(const unsigned& i) const 
{
	if (i >= _ups.size()) OOPS("VAS_Cell::upcell: out of range");
	return _ups[i]->u();
}
const VAS_Cell* VAS_Cell::dncell(const unsigned& i) const 
{
	if (i >= _dns.size()) OOPS("VAS_Cell::dncell: out of range");
	return _dns[i]->d();
}
const VAS_Edge* VAS_Cell::upedge(const unsigned& i) const 
{
	if (i >= _ups.size()) {
		OOPS("VAS_Cell::upedge: index ",i," out of range that tops out at ",_ups.size());
	}
	return _ups[i];
}
const VAS_Edge* VAS_Cell::dnedge(const unsigned& i) const 
{
	if (i >= _dns.size()) OOPS("VAS_Cell::dnedge: index ",i," out of range that tops out at ",_dns.size());
	return _dns[i];
}

string VAS_Cell::dncodes() const
{
	ostringstream os;
	if (dnsize()==0) return os.str();
	os << dncell(0)->code();
	for (unsigned i=1; i<dnsize(); i++) {
		os << "\t" << dncell(i)->code();
	}
	return os.str();
}

double VAS_Cell::mu() const
{
	if (_parameter) return *_parameter;
	return 1.0;
}

const unsigned& VAS_Cell::mu_offset() const
{
	if (_parameter_offset==UINT_MAX) OOPS("VAS_Cell::mu_offset: no offset available for cellcode "+_cell_code);
	return _parameter_offset;
}

const unsigned& VAS_Cell::mu_slot() const
{
	if (_parameter_offset==UINT_MAX) OOPS("VAS_Cell::mu_offset: no offset available for cellcode "+_cell_code);
	return _parameter_offset;
}

string VAS_Cell::display() const
{
	ostringstream disp;
	disp << "CELL:\t\"" << CELLCODEasCSTR(_cell_code) << "\"\n"
	<< " | name:\t" << name() << "\n";
	if (upsize()) {
		disp << (" | upcodes\n");
		for (unsigned i=0; i<upsize(); i++) {
			disp << " |  |\t" << CELLCODEasCSTR(upcell(i)->code()) << "\n";
		}
	}
	if (dnsize()) {
		disp << (" | dncodes\n");
		for (unsigned i=0; i<dnsize(); i++) {
			disp << " |  |\t" << CELLCODEasCSTR(dncell(i)->code()) << "\n";
		}
	}
	disp << " | slot:\t" << (_cell_slot) << "\n";
	if (upsize()) {
		disp << (" | upslots\n");
		for (unsigned i=0; i<upsize(); i++) {
			disp << " |  |\t" << (upcell(i)->slot()) << "\n";
		}
	}
	if (dnsize()) {
		disp << (" | dnslots\n");
		for (unsigned i=0; i<dnsize(); i++) {
			disp << " |  |\t" << (dncell(i)->slot()) << "\n";
		}
	}
	if (_parameter_offset != UINT_MAX) {
		disp << " | mu offset:\t" << (_parameter_offset) << "\n";
	}
	return disp.str();
}

string VAS_Cell::display_compact() const
{
	unsigned i;
	ostringstream os;
	os  << name() << " [" << CELLCODEasCSTR(_cell_code) << "]: ";
	
	if (upsize()) {
		os << "UP={";
		i=0;
		os << CELLCODEasCSTR(upcell(i)->code());
		for (i++; i<upsize(); i++) {
			os << "," << CELLCODEasCSTR(upcell(i)->code());
		}
		os << "}";
	}
	if (dnsize()) {
		os << "DN={";
		i=0;
		os << CELLCODEasCSTR(dncell(i)->code());
		for (i++; i<dnsize(); i++) {
			os << "," << CELLCODEasCSTR(dncell(i)->code());
		}
		os << "}";
	}
	return os.str();
}



const VAS_Cell* VAS_System::operator[] (const unsigned& i) const
{
	if (i >= _cells.size()) {
		OOPS("VAS_System[]: out of range");
	}
	return &(_cells[i]);
}

const VAS_Cell* VAS_System::at (const unsigned& i) const
{
	if (i >= _cells.size()) OOPS("VAS_System: out of range");
	return &(_cells[i]);
}

const VAS_Cell* VAS_System::operator[] (const cellcode& i) const
{
	cellmap::const_iterator ce = _anatomy.find(i);
	if ( ce == _anatomy.end() )
		OOPS("VAS_System::cell_from_code: cell code "+string(CELLCODEasCSTR(i))+" not found");
	return ce->second;
}


const VAS_Cell* VAS_System::cell_from_code(const cellcode& cod) const
{
	cellmap::const_iterator ce = _anatomy.find(cod);
	if ( ce == _anatomy.end() ) {
		if (_anatomy.size() < 20) {
			std::ostringstream em;
			em << "Cell code " << CELLCODEasCSTR(cod) << " not found in {";
			for (auto iter=_anatomy.begin(); iter!=_anatomy.end(); iter++) {
				if (iter!=_anatomy.begin()) em << ",";
				em << iter->first;
			}
			em << "}";
			OOPS(em.str());
		} else {
			OOPS("Cell code ",CELLCODEasCSTR(cod)," not found among ",_anatomy.size()," possibilities");
		}
	}
	return ce->second;
}

const unsigned& VAS_System::slot_from_code(const cellcode& cod) const
{
	return cell_from_code(cod)->slot();
}

const VAS_Cell* VAS_System::cell_from_name(const string& nam) const
{
	for (unsigned i=0; i<size(); i++) {
		if (_cells[i].name() == nam) return &(_cells[i]);
	}
	ostringstream cellnames;
	for (unsigned i=0; i<size(); i++) {
		cellnames << _cells[i].name() << "\n";
	}
	OOPS("Cell name "+nam+" not found among known names:\n"+cellnames.str());
	return 0;
}

const unsigned& VAS_System::slot_from_name(const string& nam) const
{
	if (nam.empty()) {
		OOPS("searching for a cell slot using an empty name");
	}
	return cell_from_name(nam)->slot();
}

const VAS_Edge* VAS_System::edge_from_codes(const cellcode& upcode, const cellcode& dncode) const
{
	const VAS_Edge* ret = NULL;
	const VAS_Cell* dn_ce = cell_from_code(dncode);
	for (unsigned i=0; i<dn_ce->upsize(); i++) {
		if (dn_ce->upcell(i)->code()==upcode) {
			ret = dn_ce->upedge(i);
			break;
		}
	}
	return ret;
}

unsigned VAS_System::alloc_slot_from_codes(const cellcode& upcode, const cellcode& dncode) const
{
	const VAS_Edge* ed = edge_from_codes(upcode, dncode);
	if (ed) {
		try {
			return ed->alloc_slot();
		} SPOO {
			OOPS("Error on alloc_slot for (",CELLCODEasCSTR(upcode),")->(",CELLCODEasCSTR(dncode),"):",oops.what());
		}
	} 
	return UINT_MAX;
}


void VAS_System::add_dna_sequence (const VAS_dna& dna)
{
	_genome.add_sequence(dna);
	_touch = true;
}

void VAS_System::add_dna_sequence (const VAS_System& dna)
{
	_genome.add_sequence(dna._genome);
	_touch = true;
}

string VAS_System::add_edge(const cellcode& source, const cellcode& sink)
{
	string s = _genome.add_edge(source,sink);
	if (s.size() && s[0]=='s') _touch = true;
	return s;
}

string VAS_System::remove_edge(const cellcode& source, const cellcode& sink)
{
	string s = _genome.remove_edge(source,sink);
	if (s.size() && s[0]=='s') _touch = true;
	return s;
}

int VAS_System::add_cell(const cellcode& cod, const string& nam, const bool& must_be_branch)
{
	int s = _genome.add_cell(cod,nam,must_be_branch);
	if (s & ELM_CREATED) _touch = true;
	return s;
}

void VAS_System::ungrow()
{
	_cells.clear();
	_edges.clear();
	_anatomy.clear();
	_touch = true;
}

std::list<elm::cellcode> elm::VAS_dna::branches_in_ascending_order(const elm::cellcode& root_cellcode) const
{
	cellcodeset Elementals = elemental_codes();
	list<VAS_dna::const_iterator> Branches;

	// Sort Branches into an ascending order
	VAS_dna::const_iterator root_iter = end();
	for (VAS_dna::const_iterator b = begin(); b!=end(); b++) {
		if (b->first == root_cellcode) {
			root_iter = b;
			continue;
		}
		if (Elementals.contains(b->first)) continue;
		list<VAS_dna::const_iterator>::const_iterator hop = Branches.begin();
		while ( hop!=Branches.end() && !(*hop)->second.dns.contains(b->first) ) hop++;
		Branches.insert(hop,b);
	}
//	if (root_iter != cend()) { // If found the root, tack it on the end.
//		Branches.insert(Branches.end(),root_iter);
//	}
	
	std::list<elm::cellcode> ret;
	for (list<VAS_dna::const_iterator>::const_iterator hop = Branches.begin(); hop!=Branches.end(); hop++) {
		ret.push_back((*hop)->first);
	}
		ret.push_back(0);
	return ret;
}


elm::cellcode VAS_System::root_cellcode() const
{
	return _graph.root_code;
}


void VAS_System::root_cellcode(const elm::cellcode& r, etk::logging_service* msg)
{
	if (_graph.root_code != r) {
		_touch = true;
		_graph.root_code = r;
		regrow(nullptr,nullptr,nullptr,nullptr,msg);
	}
}


#define REGROW_LOG BUGGER_


void VAS_System::regrow( ComponentCellcodeMap* nodes, LinearCOBundle_2* edges, Fountain* db, elm::cellcode* root, etk::logging_service* msg )
{
	boosted::shared_ptr< const std::vector<std::string> > _altnames;
	boosted::shared_ptr< const std::vector<long long  > > _altcodes;
	
	if (db) {
		_altnames = db->cache_alternative_names();
		_altcodes = db->cache_alternative_codes();
	}
	
	if (nodes||edges||db||root) {
		ComponentGraphDNA new_graph = ComponentGraphDNA(nodes, edges, db, root);
		if (!(_graph == new_graph)) {
			_graph = new_graph;
			_touch = true;
		}
	}
	if (!_touch) {
		REGROW_LOG(msg, "Not regrowing vascular system, touch is false.");
		REGROW_LOG(msg, display());
		return;
	}
	REGROW_LOG(msg, "Deleting old vascular system...");
	ungrow();
	
	REGROW_LOG(msg, "Identifying elemental alternatives...");
	cellcodeset Elementals;
	if (_graph.valid()) {
		Elementals.insert_set(_graph.elemental_codes());
	} else {
		Elementals.insert_set(_genome.elemental_codes());
	}
	
	cellcodeset KnownNodes;
	KnownNodes.insert_set(Elementals);
	
	VAS_dna::iterator b;
//	list<VAS_dna::iterator> Branches;
	list<VAS_dna::iterator>::iterator hop;
	unsigned i,j;
	
	
	// Sort Branches into an ascending order
	REGROW_LOG(msg, "Sorting branches in ascending order...");
	std::list< elm::cellcode > BranchesAscend;
	if (_graph.valid()) {
		REGROW_LOG(msg, "  _graph is valid");
		BranchesAscend = _graph.branches_ascending_order(msg);
		if (BranchesAscend.size()==0 || BranchesAscend.back()!=_graph.root_code) BranchesAscend.push_back(_graph.root_code);
	} else {
		REGROW_LOG(msg, "  _graph is not valid");
		BranchesAscend = _genome.branches_in_ascending_order(_graph.root_code);
	}
	
	// Assemble cells
	REGROW_LOG(msg, "Assembling cells...");
	unsigned s=0;
	for (std::set<cellcode>::iterator e=Elementals.begin(); e!=Elementals.end(); e++) {
		_cells.push_back(VAS_Cell(*e,s++));
		KnownNodes.insert(*e);
		REGROW_LOG(msg, "  elemental alternative:" << *e);
	}
	for (std::list<cellcode>::iterator b_iter = BranchesAscend.begin(); b_iter!=BranchesAscend.end(); b_iter++) {
		_cells.push_back(VAS_Cell(*b_iter,s++));
		KnownNodes.insert(*b_iter);
		REGROW_LOG(msg, "  branch:" << *b_iter);
	}
	
	// Build anatomy and name cells
	REGROW_LOG(msg, "Building anatomy and name cells...");
	if (_graph.valid()) {
		for ( i=0; i<_cells.size(); i++) {
			REGROW_LOG(msg, cat("     anatomy for ",_cells[i].code()));
			_anatomy[_cells[i].code()] = &(_cells[i]);
			_cells[i]._cell_name = _graph.node_name(_cells[i].code());
		}
	} else {
		Vasc_CellVec::iterator ce;
		for ( i=0; i<_cells.size(); i++) {
			_anatomy[_cells[i].code()] = &(_cells[i]);
			b = _genome.find(_cells[i].code());
			if (b != _genome.end()) _cells[i]._cell_name = (b->second.name);
		}
	}

	
	// Create and attach edges
	REGROW_LOG(msg, "Creating and attaching edges...");
	i=0;
	if (_graph.valid()) {
		std::list<cellcode> nao = _graph.nodes_ascending_order();
		for (std::list<cellcode>::iterator x=nao.begin(); x!=nao.end(); x++) {
			elm::cellcodeset x_dn = _graph.dn_node_codes(*x);
			for (std::set<elm::cellcode>::iterator y=x_dn.begin(); y!=x_dn.end(); y++) {
				if (!KnownNodes.contains(*x)) {
					OOPS("Unknown code ",*x," found in edge ",*x," -> ",*y);
				}
				if (!KnownNodes.contains(*y)) {
					OOPS("Unknown code ",*y," found in edge ",*x," -> ",*y);
				}
				_edges.add_back(_anatomy[*x],_anatomy[*y]);
				_edges[i]._edge_slot = i;
				_edges[i].u()->_dns.push_back(&_edges[i]);
				_edges[i].d()->_ups.push_back(&_edges[i]);
				i++;
			}
		}
	} else {
		for (b = _genome.begin(); b!=_genome.end(); b++) {
			for (std::set<elm::cellcode>::iterator e= b->second.dns.begin(); e!= b->second.dns.end(); e++) {
				_edges.add_back(_anatomy[b->first],_anatomy[*e]);
				_edges[i]._edge_slot = i;
				_edges[i].u()->_dns.push_back(&_edges[i]);
				_edges[i].d()->_ups.push_back(&_edges[i]);
				i++;
			}
		}
	}
	// Connect no-ups to the root
	REGROW_LOG(msg, "Connecting loose nodes to the root...");
	for (j=0; j<_cells.size()-1; j++) {
		if (_cells[j].upsize()==0) {
			_edges.add_back(_anatomy[_graph.root_code],_anatomy[_cells[j].code()]);
			_edges[i]._edge_slot = i;
			_edges[i].u()->_dns.push_back(&_edges[i]);
			_edges[i].d()->_ups.push_back(&_edges[i]);
			i++;
		}
	}
	
	
	// Counting
	REGROW_LOG(msg, "Calculating counts...");
	_n_elemental = Elementals.size();
	if (_n_elemental >= _cells.size()) {
		OOPS("in vascular::regrow, cell count (",_cells.size(),") too small for elementals count (",_n_elemental,")");
	}
	_n_branches = _cells.size() - _n_elemental - 1;
	_n_edges = _edges.size();
		
	// Setup cascading data system
	if (_use_cascading_data) {
		REGROW_LOG(msg, "Setting up cascading data system...");
		for (i=0; i<_n_elemental; i++) {
			_cells[i]._contained_elementals.insert(_cells[i]._cell_code);
			_cells[i]._elemental_slots.insert(_cells[i]._cell_slot);
		}
		for (; i<_cells.size()-1; i++) {
			for ( j=0; j<_cells[i].dnsize(); j++) {
				_cells[i]._contained_elementals.insert_set(_cells[i].dncell(j)->_contained_elementals);
				_cells[i]._elemental_slots.insert(_cells[i].dncell(j)->_elemental_slots.begin(),
												  _cells[i].dncell(j)->_elemental_slots.end()   );
			}
		}
	}
	
	
	// Repoint MU parameters and offsets
	REGROW_LOG(msg, "re-pointing mu parameters and offsets...");
	unsigned offset (0);
	if (_mu_offset) offset = *_mu_offset;
	double* par_ptr = NULL;
	if (_mu_begins) par_ptr = _mu_begins;
	for ( i=_n_elemental; i<size(); i++) {
		if (_mu_offset) {
			_cells[i]._parameter_offset = offset;
			offset++;
		} else {
			_cells[i]._parameter_offset = offset;
			offset++;
		}
		if (_mu_begins) {
			_cells[i]._parameter = par_ptr;
			par_ptr++;
		} else {
			_cells[i]._parameter = NULL;
		}
	}
	
	// Assign PHI allocation slots
	REGROW_LOG(msg, "Assigning phi allocation slots...");
	_n_competitive_allocations = 0;
	_allocation_breaks.clear();
	for (i=0; i<size(); i++) {
		if (_cells[i].upsize() > 1) {
			_allocation_breaks.push_back(_n_competitive_allocations);
			for (j=0; j<_cells[i].upsize(); j++) {
				_cells[i].upedge(j)->_alloc_slot = _n_competitive_allocations;
				_n_competitive_allocations++;
			}
		}
	}
	_allocation_breaks.push_back(_n_competitive_allocations);
	
	REGROW_LOG(msg, "Vascular system regrow complete.");
	
	REGROW_LOG(msg, display());
	
	
	_touch = false;
}


void VAS_System::repoint_parameters (double* mu_ptr, unsigned* mu_offset) 
//double* phi_ptr, unsigned* phi_offset, unsigned n_phi_vars)
{
	_mu_begins = mu_ptr;
	_mu_offset = mu_offset;
	//	_phi_begins = phi_ptr;
	//	_phi_offset = phi_offset;
	//	_n_phi_vars = n_phi_vars;
	_touch = true;
	regrow();
}

class __cell_info {
public:
	unsigned upfound;
	unsigned upknown;
	
	bool ready() { return (upfound == upknown); }
	
	__cell_info (int known=UINT_MAX)
	: upfound (0)
	, upknown (known) { }
};

string VAS_System::evaluate_system_integrity(bool allow_multi_ups)
{
	regrow();
	string s;
	int warn_no_dn=0;
	int warn_multi_up=0;
	unsigned i,j;
	
	// ELEMENTALS
	for (i=0; i<_n_elemental; i++) {
		if (!allow_multi_ups) { // NetGEV won't want this
			if (_cells[i].upsize()>1) {
				warn_multi_up++;
				s+= "node ";
				s+= CELLCODEasCSTR( _cells[i].code() );
				s+= " has multiple upward links\n";
			}
		}
	}	
	
	// NESTS
	for (i=_n_elemental; i<size(); i++) {
		if (_cells[i].dnsize()==0) {
			warn_no_dn++;
			s+= "node ";
			s+= CELLCODEasCSTR( _cells[i].code() );
			s+= " has no downward links\n";
		}
		if (!allow_multi_ups) { // NetGEV won't want this
			if (_cells[i].upsize()>1) {
				warn_multi_up++;
				s+= "node ";
				s+= CELLCODEasCSTR( _cells[i].code() );
				s+= " has multiple upward links\n";
			}
		}
	}	
	
	// CHECK FOR CYCLES
	i = _cells.size()-1;
	map<cellcode,__cell_info> candidates;
	map<cellcode,__cell_info>::iterator candidate;
	// Initialize candidates
	for (j=0; j<i; j++) {
		candidates[_cells[j].code()] = __cell_info(_cells[j].upsize());
	}
	
	cellcodeset north;
	north.insert(_cells[i].code());
	for (j=0; j<_cells[i].dnsize(); j++) {
		candidates[_cells[i].dncell(j)->code()].upfound++;
	}
	while (candidates.size()) {
		candidate = candidates.begin();
		while ((!(candidate->second.ready())) && (candidate!=candidates.end())) {
			//	HEAVY("bypass %s with",candidate->first);
			//	HEAVY(" %u",candidate->second.upfound);
			//	HEAVY(" %u\n",candidate->second.upknown);
			candidate++;
		}
		if ((candidate==candidates.end()) && (candidates.size())) {
			s+= "unknown cyclic error\n";
			break;
		}
		
		VAS_Cell* con = _anatomy[candidate->first];
		// Check for edges into the north (which would make a cycle)
		for (j=0; j<con->dnsize(); j++) {
			candidates[con->dncell(j)->code()].upfound++;
			if (north.contains(con->dncell(j)->code())) {
				s+= "node "; 
				s+= CELLCODEasCSTR( con->code() ); 
				s+= " is in a cycle\n";
				break;
			}
		}
		// Move considered to the north
		north.insert(candidate->first);
		candidates.erase(candidate);
	}
	
	return s;
}

bool VAS_System::check_single_predecessor()
{
	if (evaluate_system_integrity(false).empty()) return true;
	return false;
}


void VAS_System::use_cascading_data(const bool& flag)
{
	if (flag != _use_cascading_data) {
		_use_cascading_data = flag;
		_touch = true;
	}
}

void VAS_System::use_params_on_elementals(const bool& flag)
{
	if (flag != _use_params_on_elementals) {
		_use_params_on_elementals = flag;
		_touch = true;
	}
}


string VAS_System::display() const
{
	ostringstream disp;
	for (unsigned i=0; i<size(); i++) {
		disp << _cells[i].display();
	}	
	disp << display_edges();
	return disp.str();
}

string VAS_System::display_nodes_skinny(etk::ndarray* valuearray, etk::ndarray* valuearray2, elm::darray_ptr* valuearr3) const
{
	int show_value = -1;
	if (valuearray) {
		if (valuearray->size1()==size()) {
			show_value = 0;
		}
		if (valuearray->size2()==size()) {
			show_value = 1;
		}
		if (valuearray->size3()==size()) {
			show_value = 2;
		}
	}

	int show_value2 = -1;
	if (valuearray2) {
		if (valuearray2->size1()==size()) {
			show_value2 = 0;
		}
		if (valuearray2->size2()==size()) {
			show_value2 = 1;
		}
		if (valuearray2->size3()==size()) {
			show_value2 = 2;
		}
	}

	ostringstream disp;
	for (unsigned i=0; i<size(); i++) {
		disp << "CELL:\t\"" << CELLCODEasCSTR(_cells[i]._cell_code) << "\"\t name:\t" << _cells[i].name() ;

		
		if (show_value==0) {
			disp << "\t U:\t" << (valuearray->at(i)) ;
		}
		if (show_value==1) {
			disp << "\t U:\t" << (valuearray->at(0,i)) ;
		}
		if (show_value==2) {
			disp << "\t U:\t" << (valuearray->at(0,0,i));
		}


		if (show_value2==0) {
			disp << "\t Pr:\t" << (valuearray2->at(i)) ;
		}
		if (show_value2==1) {
			disp << "\t Pr:\t" << (valuearray2->at(0,i)) ;
		}
		if (show_value2==2) {
			disp << "\t Pr:\t" << (valuearray2->at(0,0,i)) ;
		}

		if (valuearr3) {
			if (i<n_elemental()) {
				disp << "\t AV:\t" << ((*valuearr3)->boolvalue(0, i)) ;
			}
		}

		disp << "\n";
	}
	return disp.str();
}

string VAS_System::display_edges(etk::ndarray* valuearray) const
{
	int show_value = -1;
	if (valuearray) {
		if (valuearray->size1()==_edges.size()) {
			show_value = 0;
		}
		if (valuearray->size2()==_edges.size()) {
			show_value = 1;
		}
		if (valuearray->size3()==_edges.size()) {
			show_value = 2;
		}
	}

	ostringstream s;
	s<< ("Edge\t");
	s<< ("Up Code       \t");
	s<< ("Dn Code       \t");
	if (show_value) {
		s<< ("Allo\t");
		s<< ("VALUE\n");
	} else {
		s<< ("Allo\n");
	}
	
	for (unsigned e=0; e<_edges.size(); e++) {
		s<< _edges[e].edge_slot() << "\t";
		s<< _edges[e].u()->code() << "\t";
		s<< _edges[e].d()->code() << "\t";
		try {
			s<< ("-?-");
//			s<< _edges[e].alloc_slot() << "\n";
		} SPOO {
			s<< ("---");
		}
		if (show_value==0) {
			s << "\t" << valuearray->at(e);
		}
		if (show_value==1) {
			s << "\t" << valuearray->at(0,e);
		}
		if (show_value==2) {
			s << "\t" << valuearray->at(0,0,e);
		}
		s<<"\n";
	}
	
	return s.str();
}

std::vector< std::pair<cellcode, cellcode> > VAS_System::list_edges() const
{
	std::vector< std::pair<cellcode, cellcode> > List;
	
	for (unsigned e=0; e<_edges.size(); e++) {
		List.push_back( std::pair<cellcode, cellcode>(_edges[e].u()->code(), _edges[e].d()->code()) );
	}

	return List;

}

std::vector< std::pair<cellcode, cellcode> > VAS_System::list_edges_dna() const
{
	std::vector< std::pair<cellcode, cellcode> > List;
	
//	for (unsigned e=0; e<_edges.size(); e++) {
//		auto u = _edges[e].u()->code();
//		auto d = _edges[e].d()->code();
//		if (u==0) {
//			auto i = _genome.find(0);
//			if (i==_genome.end()) continue; // there are no explicit root node links
//			if (!(i->second.dns.contains(d))) continue; // this is not an explicit root link
//		}
//		List.push_back( std::pair<cellcode, cellcode>(u, d) );
//	}
	
	size_t gs = _genome.size();
	
	for (auto uu=_genome.begin(); uu!=_genome.end(); uu++) {
		size_t dnsz = uu->second.dns.size();
		for (auto dd=uu->second.dns.begin(); dd!=uu->second.dns.end(); dd++) {
			List.push_back( std::pair<cellcode, cellcode>(uu->first, *dd) );
		}
	}
	return List;
	
}


string VAS_System::display_phenotype() const
{
	ostringstream S;
	S << _genome.generate_phenotype();
	return S.str();
}

std::vector<std::string> VAS_System::elemental_names() const
{
	std::vector<std::string> ret;
	for (unsigned a=0; a<n_elemental(); a++) {
		ret.push_back(operator[](a)->name());
	}
	return ret;
}

cellcodevec VAS_System::elemental_codes() const
{
	cellcodevec ret;
	for (unsigned a=0; a<n_elemental(); a++) {
		ret.push_back(operator[](a)->code());
	}
	return ret;
}

cellcodevec VAS_System::all_codes() const
{
	cellcodevec ret;
	for (unsigned a=0; a<size(); a++) {
		ret.push_back(operator[](a)->code());
	}
	return ret;
}

void VAS_System::clear()
{
	_genome.clear();
	_cells.clear();
	_edges.clear();
	_anatomy.clear();
	_touch = true;
		
	_mu_offset=NULL;
	_mu_begins=NULL;
		
	// Flags to determine how extensively to grow the tree.
	_use_cascading_data=false;
	_use_params_on_elementals=false;
		
	// Counting
	_n_competitive_allocations=0;
	_n_elemental=0;
	_n_branches=0;
	_n_edges=0;
	_allocation_breaks.clear();
	
	
}

VAS_System::VAS_System ()
:	_touch			(true)
,	_n_elemental	(0)
,	_n_branches		(0)
,	_n_edges		(0)
,	_use_cascading_data (false)
,	_use_params_on_elementals (false)
,	_mu_offset		(NULL)
,	_mu_begins		(NULL)
,	_graph			()
{ }

