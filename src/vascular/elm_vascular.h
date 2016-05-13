/*
 *  elm_vascular.h
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



#ifndef __VASCULAR_SYSTEM_H__
#define __VASCULAR_SYSTEM_H__

#ifndef SWIG
#include <climits>
#include "elm_cellcode.h"
#include "etk.h"
#include "elm_inputstorage.h"
#endif // ndef SWIG

#ifdef SWIG
%include "std_map.i"
%include "std_list.i"
class elm::VAS_dna_info;
%template(cellcode_infodict) std::map<elm::cellcode,elm::VAS_dna_info>;
%template(cellcode_list) std::list<elm::cellcode>;
#endif // def SWIG


namespace elm {

	#ifndef SWIG
	
	// forward declarations
	class VAS_Cell;
	class ComponentGraphDNA;
	class Fountain;

	#endif // ndef SWIG
	
	// Vascular DNA information
	//  this structure holds the most basic information about a vascular cell:
	//  the name of the cell, and the set of successor cells.
	class VAS_dna_info {
	public:
		cellcodeset dns;
		std::string name;
		bool is_branch;

		// Graphviz parameters
//		std::string color;
//		std::string fontcolor;
//		std::string fontname;
//		std::string type_name;
//		std::string shape;
		
		VAS_dna_info(const std::string& input="");		
		bool is_elemental() const;
	};

	
	// Vascular DNA
	//  This is the blueprint for the vascular networking system. It contains a
	//  map of cellcodes and the names and successors of those cellcodes. When
	//  a vascular system is regrown, the existing network is wiped clean and
	//  a new network is rebuilt from this blueprint. Changes to the vascular
	//  system should always be made in the DNA, not in the vascular structure
	//  itself.
	class VAS_dna: public std::map<cellcode,VAS_dna_info>
	{
	protected:
		cellcodeset _known_downs;
		// this is a set of all known successor cells.
			
	public:
		void read_sequence (const std::string& seq);
		// re-sequences this dna, erasing previous information
		
		void add_sequence (const std::string& seq);
		void add_sequence (const VAS_dna& dna);
		// adds edges to the existing dna, either from a seq string or an
		// existing dna strand
		
		std::string generate_sequence() const;
		// creates a program-readable dna sequence as a string, excluding auto-connections
		// to the root node
		
		std::string generate_phenotype() const;
		// creates a human-readable dna sequence as a string, including auto-connections
		// to the root node
				
		std::string add_edge(const cellcode& source, const cellcode& sink);
		// adds an individual edge to the dna
		
		std::string remove_edge(const cellcode& source, const cellcode& sink);
		// removes an individual edge to the dna, if it exists

		int add_cell(const cellcode& cod, const std::string& nam="", const bool& must_be_branch=false);
		// adds a cell to the dna, and names it. no edges are created, but the
		// new cell is auto-connected to the root node
		
		void clear();
		
		cellcodeset elemental_codes() const;
		cellcodeset all_known_codes() const;
		
		std::list<cellcode> branches_in_ascending_order(const elm::cellcode& root_cellcode=cellcode_null) const;
		
		VAS_dna();
	}; // end class VAS_dna


	#ifndef SWIG


	class VAS_Edge
	{
		VAS_Cell* _up;
		VAS_Cell* _dn;
		unsigned  _edge_slot;	// All edges, in order
		unsigned  _alloc_slot;	// Only competitive edges, grouped by terminals
		
		friend class VAS_dna;
		friend class VAS_System;
		
	public:
		VAS_Cell* u() { return _up; }
		VAS_Cell* d() { return _dn; }
		const VAS_Cell* u() const { return _up; }
		const VAS_Cell* d() const { return _dn; }
		const unsigned& edge_slot()  const { return _edge_slot; }
		const unsigned& alloc_slot() const ;
		
		bool is_competitive() const { return (_alloc_slot != UINT_MAX); }
		
		VAS_Edge (VAS_Cell*& uedge, VAS_Cell*& dedge);
		VAS_Edge (const VAS_Edge& x);
	}; // end class VAS_Edge
	
	typedef std::vector<VAS_Edge*> Vasc_EdgePVec;
	//typedef vector<VAS_Edge>  VAS_EdgeVec;

	class VAS_EdgeVec: public std::vector<VAS_Edge*> { public:
		VAS_Edge& operator[](const unsigned& i);
		const VAS_Edge& operator[](const unsigned& i) const;
		void add_back(VAS_Cell*& uedge, VAS_Cell*& dedge);
		void clear();
	};





	class VAS_Cell
	{
		// Info on Self
		std::string		_cell_name;
		cellcode		_cell_code;
		unsigned		_cell_slot;
		
		// Mu parameter on this node
	protected:
		double*			_parameter;
		unsigned		_parameter_offset;
		
		cellcodeset		_contained_elementals;
		etk::uintset			_elemental_slots;
		
	protected:
		Vasc_EdgePVec _ups;
		Vasc_EdgePVec _dns;
	public:
		size_t upsize() const { return _ups.size(); }
		size_t dnsize() const { return _dns.size(); }
		
		const cellcode& code() const { return _cell_code; }
		const unsigned& slot() const { return _cell_slot; }
		std::string name() const { return _cell_name; }
		
		VAS_Cell* upcell(const unsigned& i);
		VAS_Cell* dncell(const unsigned& i);
		VAS_Edge* upedge(const unsigned& i);
		VAS_Edge* dnedge(const unsigned& i);
		
		const VAS_Cell* upcell(const unsigned& i) const ;
		const VAS_Cell* dncell(const unsigned& i) const ;
		const VAS_Edge* upedge(const unsigned& i) const ;
		const VAS_Edge* dnedge(const unsigned& i) const ;
		
		const etk::uintset& elemental_slots() const { return _elemental_slots; }
		std::string dncodes() const;
		
		double mu() const;
		const unsigned& mu_offset() const;
		const unsigned& mu_slot() const;
		
		std::string display() const;
		std::string display_compact() const;
		
		VAS_Cell (const cellcode& c, const unsigned& slot);
		friend class VAS_System;
	};
	typedef std::vector<VAS_Cell*> Vasc_CellPVec;
	typedef std::vector<VAS_Cell>  Vasc_CellVec;



	typedef std::map<cellcode,VAS_Cell*> cellmap;

	class VAS_System
	{
		VAS_dna					_genome;
		Vasc_CellVec			_cells;
		VAS_EdgeVec				_edges;
		cellmap					_anatomy;
		bool					_touch;
		elm::ComponentGraphDNA	_graph;
		
		// Pointers to model information that will be incorporated into the
		//  fully grown vascular tree
		unsigned*               _mu_offset;
		double*					_mu_begins;
		
		// Flags to determine how extensively to grow the tree.
		bool		_use_cascading_data;
		bool		_use_params_on_elementals;
		
		// Counting
		unsigned	_n_elemental; 
		unsigned	_n_branches;  // nesting nodes other than the root
		unsigned	_n_edges;
		unsigned    _n_competitive_allocations;
		std::vector<unsigned> _allocation_breaks;
		
	public:
		VAS_dna genome() const {return _genome;}
	
		void touch() {_touch = true;}
	
		const unsigned& n_elemental() const { return _n_elemental; }
		const unsigned& n_branches()  const { return _n_branches; }
		const unsigned& n_edges()     const { return _n_edges; }
		const unsigned& n_compet_alloc() const { return _n_competitive_allocations; }
		const unsigned& alloc_break(const unsigned& i) const { return _allocation_breaks.at(i); }
		const std::vector<unsigned>& alloc_breaks() const { return _allocation_breaks; }
		size_t n_alloc_break() const { return _allocation_breaks.size(); }
		size_t        size()        const { return _cells.size(); }
		
		const VAS_Cell* operator[] (const unsigned& i) const;
		const VAS_Cell* operator[] (const cellcode& i) const;
		const VAS_Cell* at(const unsigned& i) const;
		
		// LOOKUP ACCESS - CELLS
		const VAS_Cell* cell_from_code(const cellcode& cod) const;
		const unsigned& slot_from_code(const cellcode& cod) const;
		const VAS_Cell* cell_from_name(const std::string& nam) const;
		const unsigned& slot_from_name(const std::string& nam) const;
		
		// LOOKUP ACCESS - EDGES
		const VAS_Edge* edge_from_codes(const cellcode& upcode, const cellcode& dncode) const;
		unsigned alloc_slot_from_codes(const cellcode& upcode, const cellcode& dncode) const;
		
		// DNA
		std::string add_edge(const cellcode& source, const cellcode& sink);
		std::string remove_edge(const cellcode& source, const cellcode& sink);
		int add_cell(const cellcode& cod, const std::string& nam="", const bool& must_be_branch=false);
		void add_dna_sequence (const VAS_dna& dna);
		void add_dna_sequence (const VAS_System& dna);
		
		void use_cascading_data(const bool& flag);
		void use_params_on_elementals(const bool& flag);
		
		void ungrow();
		// UNGROW flushes the existing cells and edges, leaving the genome intact.
		
		void regrow( ComponentCellcodeMap* nodes=nullptr, LinearCOBundle_2* edges=nullptr, Fountain* db=nullptr, elm::cellcode* root=nullptr, etk::logging_service* msg=nullptr );
		// REGROW flushes the existing cells and edges, and regrows new ones based
		//  on the genome.

		elm::cellcode root_cellcode() const;
		void root_cellcode(const elm::cellcode& r, etk::logging_service* msg=nullptr);
		
		void clear();
		
		void repoint_parameters (double* mu_ptr, unsigned* mu_offset); 
		//double* phi_ptr=NULL, unsigned* phi_offset=NULL, unsigned n_phi_vars=UINT_MAX);
		// Resets the parameter pointers of the nesting nodes.
		
		std::string evaluate_system_integrity(bool allow_multi_ups);
		// Checks for cycles, multi-ups in non-NetGEV models.
		
		bool check_single_predecessor();
		// Checks if the vascular system is a Nested logit-style strict hierarchy.
		
		std::string display() const;
		std::string display_phenotype() const;
		std::string display_edges() const;
		std::vector< std::pair<cellcode, cellcode> > list_edges() const;
		std::vector< std::pair<cellcode, cellcode> > list_edges_dna() const;
		// DISPLAY prints a representation of the network.
		
		std::vector<std::string> elemental_names() const;
		std::vector<long long> elemental_codes() const;
		std::vector<long long> all_codes() const;
		
		VAS_System();
	};

	#endif // ndef SWIG
	
} // end namespace elm

#endif // __VASCULAR_SYSTEM_H__
