/*
 *  etk_memory.h
 *
 *  Copyright 2007-2013 Jeffrey Newman
 *
 *  This file is part of ELM.
 *  
 *  ELM is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  ELM is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with ELM.  If not, see <http://www.gnu.org/licenses/>.
 *  
 */


#ifndef __TOOLBOX_MEMORY__
#define __TOOLBOX_MEMORY__

#include <string>
#include <vector>
#include <climits>

#include "etk_ndarray.h"

#define NUMPY_ARRAY_MEMORY

#ifndef NUMPY_ARRAY_MEMORY
 #define memarray memarray_raw
 #define triangle triangle_raw
 #define SYMMETRIC_PACKED
#else
 #define memarray ndarray
 #define triangle symmetric_matrix
#endif

namespace etk {

	class puddle { 
	protected:
		unsigned siz;
		double*  pool;
	public:
		puddle(const unsigned& s=0);
		puddle(const puddle& p);
		virtual ~puddle();
		
		double& operator[](const unsigned& i);
		const double& operator[](const unsigned& i) const;
	//	virtual double* ptr(const unsigned& i=0);
		inline double* operator* () { return pool; }
		inline const double* operator* () const { return pool; }
		
		virtual void resize(const unsigned& s, bool init = true);
		void purge();
		
		// Math
		void exp ();
		void is_exponential_of (const puddle& that);
		void log ();
		double sum() const;
		
		double operator*(const puddle& that) const; // dot product
		
		// Assign and Evaluate
		bool operator==(const puddle& that) const;
		virtual void operator= (const puddle& that);
		const unsigned& size() const;
		
		void operator+=(const puddle& that);
		void operator-=(const puddle& that);
		
		void projection(const puddle& fixture, const puddle& beam, const double& distance);
		
		void initialize(const double& init=0);
		void scale(const double& scal=1);
		
		std::string print(unsigned start=0, unsigned stop=UINT_MAX) const;
		std::vector<double> vectorize(unsigned start=0, unsigned stop=UINT_MAX) const;
		std::vector<double> negative_vectorize(unsigned start=0, unsigned stop=UINT_MAX) const;
	};


	class memarray_raw;

	class triangle_raw: public puddle {
		unsigned side;
		friend class memarray_raw;
	public:
		triangle_raw(const unsigned& s=0);
		virtual ~triangle_raw();
		
		double& operator()(const unsigned& i, const unsigned& j);
		const double& operator()(const unsigned& i, const unsigned& j) const;

		virtual void resize(const unsigned& s, bool init = true);
		virtual void operator= (const triangle_raw& that);
		
	//	void inv(const bool& rescale=false, bool squeeze=true);
		void initialize_identity();
		
		std::string printSquare();
		void unpack(memarray_raw& destination) const;
		void repack(const memarray_raw& source);
		
		void operator= (const memarray_raw& that) {repack(that);}
		const unsigned& size1() const { return side; }
	};

	class three_array: public three_dim {
	protected:
		const double* start;
	//	unsigned rows;
	//	unsigned cols;
	//	unsigned deps;
	public:
		three_array(const double* starter=NULL, const unsigned& r=0, const unsigned& c=1, const unsigned& d=1);
		virtual ~three_array();
		
		const double* deref() const { return start; }
		const double& operator()(const unsigned& i, const unsigned& j=0, const unsigned& k=0) const;		
		const double* ptr(const unsigned& i=0, const unsigned& j=0, const unsigned& k=0) const;
		const double& at(const unsigned& i, const unsigned& j=0, const unsigned& k=0) const;		

		void reset(const double* starter=NULL, const unsigned& r=0, const unsigned& c=1, const unsigned& d=1);
		void destroy() {reset();}
	};


	class memarray_raw: public puddle, public three_dim {
	public:
		memarray_raw(const unsigned& r=0, const unsigned& c=1, const unsigned& d=1);
		memarray_raw(const triangle_raw& t);
		virtual ~memarray_raw();
		
		const double& operator()(const unsigned& i, const unsigned& j=0, const unsigned& k=0) const;		
		double& operator()(const unsigned& i, const unsigned& j=0, const unsigned& k=0);

		double* ptr(const unsigned& i=0, const unsigned& j=0, const unsigned& k=0);
		const double* ptr(const unsigned& i=0, const unsigned& j=0, const unsigned& k=0) const;
		
		void resize(const unsigned& r, const unsigned& c=1, const unsigned& d=1);
		virtual void operator= (const memarray_raw& that);
		virtual void operator= (const ndarray& that);
		
		void prob_scale_2 ();
		std::string printrow(const unsigned& r) const;
		std::string printrows(unsigned rstart, const unsigned& rfinish) const;
		std::string printall() const;
		std::string printSize() const;

		void operator= (const triangle_raw& that) {that.unpack(*this);}

		void initialize(const double& init=0);
	};

	class memarray_symmetric: public memarray_raw {
	public:
		memarray_symmetric(const unsigned& r=0, const unsigned& c=0);
		void resize(const unsigned& r, const unsigned& c=0);
		void copy_uppertriangle_to_lowertriangle();
		std::string printSquare();
		void operator= (const memarray_symmetric& that);
		void operator= (const symmetric_matrix& that);
	
	};

	typedef std::vector<memarray_raw> memarrays;
	
	class bitarray {
	private:
		std::vector<bool> pool;
		
		unsigned rows;
		unsigned cols;
		unsigned deps;
		
	public:
		bitarray(const unsigned& r=0, const unsigned& c=0, const unsigned& d=1);
		~bitarray();
		
		const unsigned& size1() const { return rows; }
		const unsigned& size2() const { return cols; }
		const unsigned& size3() const { return deps; }

		void input(const bool& val, const unsigned& i, const unsigned& j=0, const unsigned& k=0);
		bool operator()(const unsigned& i, const unsigned& j=0, const unsigned& k=0) const;
		
		void resize(const unsigned& r, const unsigned& c=1, const unsigned& d=1, bool init = true);
		void purge();
		
		// Assign and Evaluate
		bool operator==(const bitarray& that) const;
		void operator= (const bitarray& that);
		void operator= (const std::vector<bool>& that);
		size_t size() const;
		void initialize(const bool& init=false);
		std::vector<bool> contents() const;
	
		std::string printrow(const unsigned& r) const;		
		std::string printrows(unsigned rstart, const unsigned& rfinish) const;
	};
	
	

} // end namespace etk

#endif
