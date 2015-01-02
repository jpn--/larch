/*
 *  elm_cellcode.h
 *
 *  Copyright 2007-2015 Jeffrey Newman
 *
Larch is free software: you can redistribute it and/or modify
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


#ifndef __ELM_CELLCODE_H__
#define __ELM_CELLCODE_H__

#ifndef SWIG
#include "etk.h"
#include <vector>
#include <list>
#include <set>
#include <string>
#include <map>
#endif // not SWIG

#ifdef SWIG
//%include <std_shared_ptr.i>
//%shared_ptr(cellcodeset)
#endif // SWIG

namespace elm {
	
	// CELLCODE
	//  The cellcode type is defined to hold code-values for cells. It is
	//  defined specially here so that changing the underlying type is all
	//  done in one place, if the underlying type ever needs to be changed
	//  from "string"
	

	typedef long long                 cellcode;
	
	
	typedef	std::vector<cellcode>     cellcodevec;
	typedef std::vector<cellcodevec>  cellcodevecs;
	#ifndef SWIG

	long long longlong_from_string(const std::string& s);

	#define cellcode_null			  (0)
	#define cellcode_empty			  (-9)
	#define is_cellcode_empty(x)      ((x) == ULLONG_MAX)
	typedef std::list<cellcode>		  cellcodelist;

	#define CELLCODEasCSTR(x)         (etk::thing_as_string(x).c_str())

	cellcodevec cellcodevec_from_uintvec (const std::vector<unsigned int>& x);

	#define cellcode_from_string(x)		(longlong_from_string(x))

	#endif // ndef SWIG

	class cellcodeset;

	class cellcodeset_iterator
	{
	private:
		std::set<cellcode>::iterator it;
		std::set<cellcode>::iterator ender;
		boosted::shared_ptr< std::set<cellcode> > parent;
	public:
		cellcodeset_iterator __iter__();
		elm::cellcode next();
		elm::cellcode __next__();
		cellcodeset_iterator(cellcodeset* parent);
	};

	typedef std::set<cellcode>::iterator cellcode_set_iter;
	typedef std::set<cellcode>::const_iterator cellcode_set_citer;
	  

	class cellcodeset
	{
		friend class cellcodeset_iterator;
		#ifndef SWIG
	  private:
		boosted::shared_ptr< std::set<cellcode> > _codes;
	  public:
		cellcodeset (const cellcodevec& iv);
		
		inline std::pair<cellcode_set_iter,bool> insert (const cellcode& val)
		{ return _codes->insert(val); }

		template <class InputIterator>
		void insert (InputIterator first, InputIterator last)
		{ _codes->insert(first, last); }

		
		inline cellcode_set_iter  erase (cellcode_set_citer position)
		{ return _codes->erase(position); }
		inline size_t erase (const cellcode& val)
		{ return _codes->erase(val); }
		inline cellcode_set_iter  erase (cellcode_set_citer first, cellcode_set_citer last)
		{ return _codes->erase(first,last); }
		
		inline cellcode_set_citer find (const cellcode& val) const
		{ return _codes->find(val); }
		inline cellcode_set_iter  find (const cellcode& val)
		{ return _codes->find(val); }
		
		inline bool empty() const noexcept_
		{ return _codes->empty(); }
		
		inline void clear() noexcept_
		{ _codes->clear(); }
				
		inline size_t size() const noexcept_
		{ return _codes->size(); }
		
		inline size_t count (const cellcode& val) const
		{ return _codes->count(val); }


		inline cellcode_set_iter begin() noexcept_
		{ return _codes->begin(); }
		inline cellcode_set_citer begin() const noexcept_
		{ return _codes->begin(); }
		inline cellcode_set_iter end() noexcept_
		{ return _codes->end(); }
		inline cellcode_set_citer end() const noexcept_
		{ return _codes->end(); }
		
		inline std::set<cellcode>::reverse_iterator rbegin() noexcept_
		{ return _codes->rbegin(); }
		inline std::set<cellcode>::const_reverse_iterator rbegin() const noexcept_
		{ return _codes->rbegin(); }
		inline std::set<cellcode>::reverse_iterator rend() noexcept_
		{ return _codes->rend(); }
		inline std::set<cellcode>::const_reverse_iterator rend() const noexcept_
		{ return _codes->rend(); }
		#endif // ndef SWIG

	  public:
		cellcodeset ();
		cellcodeset (const cellcode& i);
		cellcodeset (const cellcodeset&);
		
		bool contains (const cellcode& j) const;
		void insert_set (const cellcodeset& i);
		void append(const cellcode& i);
		bool remove(const cellcode& i);
		void noop() const { }
		
		cellcodeset& operator+=(const cellcodeset& i);
		cellcodeset& operator-=(const cellcodeset& i);
		cellcodeset& operator+=(const cellcode& i);
		cellcodeset& operator-=(const cellcode& i);
		
		int __len__() const;
		std::string __repr__() const;
		elm::cellcodeset_iterator __iter__();
				
	};

	#ifndef SWIG

	typedef std::map<cellcode,int>  cellcodei_map;
	typedef std::pair<cellcode,int> cellcodei_pair;
		
	cellcodevec read_cellcodevec (const std::string& s);
	
	#endif // ndef SWIG
	
	elm::cellcode max_cellcode();

	struct string_and_cellcode {
		std::string text;
		cellcode code;
	};
	
	struct cellcodepair {
		cellcode up;
		cellcode dn;
		cellcodepair(cellcode up=cellcode_empty, cellcode dn=cellcode_empty): up(up), dn(dn) {}
		cellcodepair(const cellcodepair& x): up(x.up), dn(x.dn) {}
		
		bool operator<(const cellcodepair& x) const {if (up<x.up || (up==x.up&&dn<x.dn)) return true; else return false;}
		bool operator==(const cellcodepair& x) const {if (up==x.up&&dn==x.dn) return true; else return false;}
		bool operator>(const cellcodepair& x) const {return !(*this<x || *this==x);}
		bool operator>=(const cellcodepair& x) const {return (*this>x || *this==x);}
		bool operator<=(const cellcodepair& x) const {return (*this<x || *this==x);}
	};
}

#endif // __ELM_CELLCODE_H__


