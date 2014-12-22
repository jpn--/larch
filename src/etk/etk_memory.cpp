/*
 *  etk_memory.cpp
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


#include "etk.h"
using namespace etk;

#include <iostream>
#include <cmath>
#include <climits>
#include <cstring>


puddle::puddle(const unsigned& s)
: siz  (0)
, pool (NULL)
{
 	resize(s);
}
puddle::puddle(const etk::puddle& that)
: siz  (0)
, pool (NULL)
{
	if (siz != that.size()) {
		resize(that.size(),false);
	}
	memcpy(pool, that.pool, siz*sizeof(double));
}

puddle::~puddle()
{
	if (pool) {
		delete [] pool;
		pool = NULL;
	}
}

double& puddle::operator[](const unsigned& i)
{
	if (i<siz) return pool[i];
	OOPS(cat("puddle access out of range: ",i," in ",siz));
	return pool[0];
}

const double& puddle::operator[](const unsigned& i) const
{
	if (i<siz) return pool[i];
	OOPS(cat("const puddle access out of range: ",i," in ",siz));
	return pool[0];
}

/*
double* puddle::ptr(const unsigned& i)
{
	if (i<siz) return pool+i;
	OOPS(cat("puddle ptr access out of range: ",i," in ",siz));
	return pool;
}
 */

void puddle::resize(const unsigned& s, bool init)
{
	if (pool) delete [] pool;
	pool = NULL;
	if (s) try {
		pool = new double [s];
		siz = s;
		if (init) memset(pool, 0, s*sizeof(double));
	}
	SPOO {
        pool = NULL;
		OOPS("out of memory");
    }
}

void puddle::initialize(const double& init)
{
	if (!init) memset(pool, 0, siz*sizeof(double));
	else {
		for (unsigned i=0; i<siz; i++) pool[i]=init;
	}
}

void puddle::scale(const double& scal)
{
	cblas_dscal(siz, scal, pool, 1);
}

void puddle::purge()
{
	if (pool) delete [] pool;
	pool = NULL;
	siz = 0;
}

double puddle::operator*(const puddle& that) const // dot product
{
	if (siz != that.size()) OOPS("puddle dot-product of different sized puddles");
	return cblas_ddot(siz, pool,1, that.pool,1);
}

void puddle::exp () {
	unsigned i;
	for (i=0; i<siz; i++) {
		pool[i] = ::exp(pool[i]);
	}
}

void puddle::is_exponential_of (const puddle& that) {
	if (siz != that.size()) OOPS("puddle is_exponential_of different sized puddle");
	unsigned i;
	for (i=0; i<siz; i++) {
		pool[i] = ::exp(that[i]);
	}
}

void puddle::log () {
	unsigned i;
	for (i=0; i<siz; i++) {
		pool[i] = ::log(pool[i]);
	}
}

double puddle::sum () const {
	unsigned i;
	double s (0.0);
	for (i=0; i<siz; i++) {
		s += pool[i];
	}
	return s;
}


bool puddle::operator==(const puddle& that) const
{
	if (siz != that.size()) return false;
	if (!pool || !that.pool) return false;
	return !(memcmp(pool, that.pool, siz*sizeof(double)));
}
void puddle::operator= (const puddle& that)
{
	if (siz != that.size()) {
		resize(that.size(),false);
	}
	memcpy(pool, that.pool, siz*sizeof(double));
}
void puddle::operator+=(const puddle& that)
{
	if (siz != that.size()) OOPS("puddle addition of different sized puddles");
	cblas_daxpy(siz, 1, that.pool, 1, pool, 1);
}
void puddle::operator-=(const puddle& that)
{
	if (siz != that.size()) OOPS("puddle subtraction of different sized puddles");
	cblas_daxpy(siz, -1, that.pool, 1, pool, 1);
}

void puddle::projection(const puddle& fixture, const puddle& beam, const double& distance)
{
	if (siz != fixture.size()) OOPS("puddle projection fixture wrong size");
	if (siz != beam.size()) OOPS("puddle beam fixture wrong size");
	memcpy(pool, fixture.pool, siz*sizeof(double));
	cblas_daxpy(siz, distance, beam.pool, 1, pool, 1);
}


const unsigned& puddle::size() const 
{ 
	return siz; 
}

std::string puddle::print(unsigned start, unsigned stop) const
{
	std::ostringstream ret;
	if (UINT_MAX == stop) stop = size();
	if (start > stop) return ret.str();
	while (start < stop) {
		ret << operator[](start) << "\t";
		start++;
	}
	return ret.str();
}

std::vector<double> puddle::vectorize(unsigned start, unsigned stop) const
{
	std::vector<double> ret;
	if (UINT_MAX == stop) stop = size();
	if (start > stop) return ret;
	while (start < stop) {
		ret.push_back( operator[](start) );
		start++;
	}
	return ret;
}

std::vector<double> puddle::negative_vectorize(unsigned start, unsigned stop) const
{
	std::vector<double> ret;
	if (UINT_MAX == stop) stop = size();
	if (start > stop) return ret;
	while (start < stop) {
		ret.push_back( - operator[](start) );
		start++;
	}
	return ret;
}

triangle_raw::triangle_raw(const unsigned& s)
: puddle (s*(s+1)/2)
, side (s)
{ }

triangle_raw::~triangle_raw()
{ }

double& triangle_raw::operator()(const unsigned& i, const unsigned& j)
{
	if (i>=side) {
		OOPS("triangle_raw access out of range");
	}
	if (j>=side) {
		OOPS("triangle_raw access out of range");
	}
	if (i<j) {
		unsigned col_top = (i*side)-((i*(i-1))/2);
		return pool[col_top+j-i];
	}
	unsigned col_top = (j*side)-((j*(j-1))/2);
	return pool[col_top+i-j];
}

const double& triangle_raw::operator()(const unsigned& i, const unsigned& j) const
{
	if (i>=side) {
		OOPS("triangle_raw access out of range");
	}
	if (j>=side) {
		OOPS("triangle_raw access out of range");
	}
	if (i<j) {
		unsigned col_top = (i*side)-((i*(i-1))/2);
		return pool[col_top+j-i];
	}
	unsigned col_top = (j*side)-((j*(j-1))/2);
	return pool[col_top+i-j];
}

void triangle_raw::resize(const unsigned& s, bool init)
{
	puddle::resize((s*(s-1)/2)+s, init);
	side = s;
}

void triangle_raw::operator= (const triangle_raw& that)
{
	puddle::operator= (that);
	side=that.side;
}

/*
void triangle_raw::inv(const bool& rescale, bool squeeze)
{
	if (squeeze) {
		// First check if a squeeze is even needed
		squeeze = false;
		std::vector<int> zerow (side, 1);
		// Identify all rows with non-zero values
		for (unsigned i=0; i<side; i++) {
			for (unsigned j=0; j<side; j++) {
				if (this->operator()(i,j)) zerow[i] = 0;
				break;
			}
			// If a zeros-only row is found, squeeze is needed
			if (zerow[i]) squeeze = true;
		}
		if (squeeze) {
			unsigned shortside (0);
			for (unsigned i=0; i<side; i++) {if (!zerow[i]) shortside++;}
			triangle_raw squeezed(shortside);
			unsigned i(0);
			unsigned ix(0);
			unsigned j(0);
			unsigned jx(0);
			while (i<shortside) {
				while (zerow[ix]) { ix++; }
				while (j<shortside) {
					while (zerow[jx]) { jx++; }
					squeezed(i,j) = this->operator()(ix,jx);
					j++; jx++;
				}
				i++; ix++; j=0; jx=0;
			}
			squeezed.inv(rescale,false);
			i=0; ix=0; j=0; jx=0;
			while (i<shortside) {
				while (zerow[ix]) { ix++; }
				while (j<shortside) {
					while (zerow[jx]) { jx++; }
					this->operator()(ix,jx) = squeezed(i,j);
					j++; jx++;
				}
				i++; ix++; j=0; jx=0;
			}
		}
	}
	if (!squeeze) {
		if (rescale) {
			puddle scales (side);
			unsigned i,j;
			for (i=0; i<side; i++) {
				scales[i] = sqrt(this->operator()(i,i));
			}
			for (i=0;i<side;i++){
				for (j=i;j<side;j++){
					this->operator()(i,j) /= scales[i];
					this->operator()(i,j) /= scales[j];
				}
			}
			
			etk::dppinv (pool, side);
			
			for (i=0;i<side;i++){
				for (j=i;j<side;j++){
					this->operator()(i,j) /= scales[i];
					this->operator()(i,j) /= scales[j];
				}
			}
			// scales.purge();
		} else {
			//	cerr << "pool=" << pool << "\n";
			//	cerr << "pool*=" << *pool << "\n";
			//	cerr << "side=" << side << "\n";
			etk::dppinv (pool, side);
		}
	}
}
*/

void triangle_raw::initialize_identity()
{
	initialize();
	for (unsigned i=0; i<side; i++) operator()(i,i) = 1;
}



std::string triangle_raw::printSquare()
{
    std::ostringstream pr;
	for (unsigned i=0; i<side; i++) {
		for (unsigned j=0; j<side; j++) {
			pr.width(12);
			pr << this->operator()(i, j) << "\t";
		}
		pr << "\n";
	}
	return pr.str();
}
std::string memarray_symmetric::printSquare()
{
    std::ostringstream pr;
	for (unsigned i=0; i<size1(); i++) {
		for (unsigned j=0; j<size2(); j++) {
			pr.width(12);
			pr << this->operator()(i, j) << "\t";
		}
		pr << "\n";
	}
	return pr.str();
}

void memarray_symmetric::operator= (const memarray_symmetric& that)
{
	memarray_raw::operator=(that);
}


void memarray_symmetric::operator= (const symmetric_matrix& that)
{
	if (siz != that.size()) {
		resize(that.size1());
	}
	memcpy(pool, that.ptr(), siz*sizeof(double));
	rows=that.size1();
	cols=(that.ndim()>=2 ? that.size2() : 1);
	deps=(that.ndim()>=3 ? that.size3() : 1);
}


memarray_symmetric::memarray_symmetric(const unsigned& r, const unsigned& c)
: memarray_raw(r,r)
{
}

void memarray_symmetric::resize(const unsigned& r, const unsigned& c)
{
	puddle::resize(r*r);
	rows=r;
	cols=r;
	deps=1;
}

void memarray_symmetric::copy_uppertriangle_to_lowertriangle()
{
	for (size_t i=0; i<size1(); i++) {
		for (size_t j=i+1; j<size1(); j++) {
			pool[j*cols+i] = pool[i*cols+j];
		}
	}
}


void triangle_raw::unpack(memarray_raw& destination) const
{
	destination.resize(side,side);
	for (unsigned i=0; i<side; i++) {
		for (unsigned j=0; j<side; j++) {
			destination(i,j) = this->operator()(i, j);
		}
	}
}

void triangle_raw::repack(const memarray_raw& source)
{
	if (source.size1()!=source.size2()) OOPS("repack triangle_raw failed, source not square");
	resize(source.size1());
	for (unsigned i=0; i<side; i++) {
		for (unsigned j=i; j<side; j++) {
			this->operator()(i, j) = source(i,j);
		}
	}
}


/////// MARK: THREE_ARRAY

three_array::three_array(const double* starter, const unsigned& r, const unsigned& c, const unsigned& d)
: etk::three_dim (r,c,d)
, start (starter)
{
}

void three_array::reset(const double* starter, const unsigned& r, const unsigned& c, const unsigned& d)
{
	start = starter;
	rows= (r);
	cols= (c);
	deps= (d);
}
three_array::~three_array()
{ }

const double& memarray_raw::operator()(const unsigned& r, const unsigned& c, const unsigned& d) const
{
	if (r>=rows) {
		OOPS("rectangle row access out of range, asking ",r," but having only ",rows);
	}
	if (c>=cols) {
		OOPS("rectangle col access out of range, asking ",c," but having only ",cols);
	} 
	if (d>=deps) {
		OOPS("rectangle dep access out of range, asking ",d," but having only ",deps);
	}
	return pool[r*cols*deps+c*deps+d];
}

double& memarray_raw::operator()(const unsigned& r, const unsigned& c, const unsigned& d)
{
	if (r>=rows) {
		OOPS("rectangle row access out of range, asking ",r," but having only ",rows);
	}
	if (c>=cols) {
		OOPS("rectangle col access out of range, asking ",c," but having only ",cols);
	} 
	if (d>=deps) {
		OOPS("rectangle dep access out of range, asking ",d," but having only ",deps);
	}
	return pool[r*cols*deps+c*deps+d];
}

const double& three_array::operator()(const unsigned& r, const unsigned& c, const unsigned& d) const
{
	if (r>=rows) {
		OOPS("const rectangle row access out of range, asking ",r," but having only ",rows);
	}
	if (c>=cols) {
		OOPS("const rectangle col access out of range, asking ",c," but having only ",cols);
	} 
	if (d>=deps) {
		OOPS("const rectangle dep access out of range, asking ",d," but having only ",deps);
	}
	return start[r*cols*deps+c*deps+d];
}

const double& three_array::at(const unsigned& r, const unsigned& c, const unsigned& d) const
{
	return three_array::operator()(r,c,d);
}


const double* memarray_raw::ptr(const unsigned& r, const unsigned& c, const unsigned& d) const
{
	if (r>=rows) {
		OOPS("rectangle pointer row access out of range, asking ",r," but having only ",rows);
	}
	if (c>=cols) {
		OOPS("rectangle pointer col access out of range, asking ",c," but having only ",cols);
	}
	if (d>=deps) {
		OOPS("rectangle pointer dep access out of range, asking ",d," but having only ",deps);
	} 
	return pool+r*cols*deps+c*deps+d;
}
double* memarray_raw::ptr(const unsigned& r, const unsigned& c, const unsigned& d)
{
	if (r>=rows) {
		OOPS("rectangle pointer row access out of range, asking ",r," but having only ",rows);
	}
	if (c>=cols) {
		OOPS("rectangle pointer col access out of range, asking ",c," but having only ",cols);
	}
	if (d>=deps) {
		OOPS("rectangle pointer dep access out of range, asking ",d," but having only ",deps);
	} 
	return pool+r*cols*deps+c*deps+d;
}

const double* three_array::ptr(const unsigned& r, const unsigned& c, const unsigned& d) const
{
	if (r>=rows) {
		OOPS("const ptr rectangle row access out of range, asking ",r," but having only ",rows);
	}
	if (c>=cols) {
		OOPS("const ptr rectangle col access out of range, asking ",c," but having only ",cols);
	}
	if (d>=deps) {
		OOPS("const ptr rectangle dep access out of range, asking ",d," but having only ",deps);
	}
	return start+r*cols*deps+c*deps+d;
}



/////// MARK: MEMARRAY NEW

memarray_raw::memarray_raw(const unsigned& r, const unsigned& c, const unsigned& d)
: puddle (r*c*d)
, three_dim (r,c,d)
{
}

memarray_raw::memarray_raw(const triangle_raw& t)
: puddle (t.side*t.side)
, three_dim (t.side,t.side,1)
{
	for (unsigned i=0; i<t.side; i++) for (unsigned j=0; j<t.side; j++) {
		operator()(i,j) = t(i,j);
	}
}

memarray_raw::~memarray_raw()
{ }

void memarray_raw::resize(const unsigned& r, const unsigned& c, const unsigned& d)
{
	puddle::resize(r*c*d);
	rows=r;
	cols=c;
	deps=d;
}


void memarray_raw::initialize(const double& init)
{
	if (siz!=rows*cols*deps) {
		OOPS("Error! This memarray has been stomped");
	}
	puddle::initialize(init);
}

void memarray_raw::operator= (const memarray_raw& that)
{
	puddle::operator= (that);
	rows=that.rows;
	cols=that.cols;
	deps=that.deps;
}

void memarray_raw::operator= (const ndarray& that)
{
	if (siz != that.size()) {
		resize(that.size(),false);
	}
	memcpy(pool, that.ptr(), siz*sizeof(double));
	rows=that.size1();
	cols=(that.ndim()>=2 ? that.size2() : 1);
	deps=(that.ndim()>=3 ? that.size3() : 1);
}

void memarray_raw::prob_scale_2 () {
	unsigned x1, x2, x3; double temp;
	for ( x1=0; x1<rows; x1++ ) { 
		for ( x3=0; x3<deps; x3++ ) { 
			temp = 0;
			for ( x2=0; x2<cols; x2++ ) { temp += this->operator()(x1,x2,x3); }
			if (!temp) break;
			for ( x2=0; x2<cols; x2++ ) { this->operator()(x1,x2,x3) /= temp; }
		}
	}
}

std::string memarray_raw::printrow(const unsigned& r) const
{
	std::ostringstream ret;
	unsigned x2, x3;
	char depMarker, colMarker, rowMarker;
	if (deps==1) {
		depMarker = ' ';
		colMarker = '\t';
		rowMarker = '\n';
	} else {
		depMarker = '\t';
		colMarker = '\n';
		rowMarker = '\n';
	}
	for ( x2=0; x2<cols; x2++ ) { 
		for ( x3=0; x3<deps; x3++ ) { 
			ret << operator()(r,x2,x3) << depMarker;
		}
		ret << colMarker;
	}
	ret << rowMarker;
	return ret.str();
}

std::string memarray_raw::printrows(unsigned rstart, const unsigned& rfinish) const
{
	std::ostringstream ret;
	for (; rstart < rfinish; rstart++) {
		ret << printrow(rstart);
	}
	return ret.str();
}

std::string memarray_raw::printall() const
{
	return printrows(0,rows);
}

std::string memarray_raw::printSize() const
{
	std::ostringstream ret;
	ret << "("<< rows<< ","<<cols<<","<<deps<<")";
	return ret.str();
}


/////// MARK: BITARRAY

bitarray::bitarray(const unsigned& r, const unsigned& c, const unsigned& d)
: pool (r*c*d)
, rows (r)
, cols (c)
, deps (d)
{ }
bitarray::~bitarray()
{
	pool.clear();
}

void bitarray::input(const bool& val, const unsigned& i, const unsigned& j, const unsigned& k)
{
	if (i>=rows) {
		OOPS("bitarray row access out of range");
	}
	if (j>=cols) {
		OOPS("bitarray col access out of range");
	}
	if (k>=deps) {
		OOPS("bitarray dep access out of range");
	}
	pool[i*cols*deps+j*deps+k] = val;
}
bool bitarray::operator()(const unsigned& i, const unsigned& j, const unsigned& k) const
{
	if (i>=rows) {
		OOPS("const bitarray row access out of range");
	}	
	if (j>=cols) {
		OOPS("const bitarray col access out of range");
	}
	if (k>=deps) {
		OOPS("const bitarray dep access out of range");		
	}
	return pool[i*cols*deps+j*deps+k];	
}

void bitarray::resize(const unsigned& r, const unsigned& c, const unsigned& d, bool init)
{
	if (init) 
		pool.assign(r*c*d,false);
	else
		pool.resize(r*c*d);	
	rows=r;
	cols=c;
	deps=d;
}
void bitarray::purge()
{
	pool.clear();
}

// Assign and Evaluate
bool bitarray::operator==(const bitarray& that) const
{
	if (rows != that.rows) return false;
	if (cols != that.cols) return false;
	if (deps != that.deps) return false;
	return (pool==that.pool);
}
void bitarray::operator= (const bitarray& that)
{
	pool=that.pool;
	rows=that.rows;
	cols=that.cols;
	deps=that.deps;
}
void bitarray::operator= (const std::vector<bool>& that)
{
	if (that.size() != rows*cols*deps) OOPS("reassignment of bitarray using badly sized bool vector");
	pool=that;
}

std::vector<bool> bitarray::contents() const
{
	return pool;
}
size_t bitarray::size() const
{
	return pool.size();
}
void bitarray::initialize(const bool& init)
{
	pool.assign(pool.size(),init);
}

std::string bitarray::printrow(const unsigned& r) const
{
	std::ostringstream ret;
	unsigned x2, x3;
	std::string colMarker, rowMarker;
	if (deps==1) {
		colMarker = "";
		rowMarker = "\n";
	} else {
		colMarker = "\n";
		rowMarker = "\n";
	}
	for ( x2=0; x2<cols; x2++ ) { 
		for ( x3=0; x3<deps; x3++ ) { 
			ret << this->operator()(r,x2,x3);
		}
		ret << colMarker;
	}
	ret << rowMarker;
	return ret.str();
}

std::string bitarray::printrows(unsigned rstart, const unsigned& rfinish) const
{
	std::ostringstream ret;
	for (; rstart < rfinish; rstart++) {
		ret << printrow(rstart);
	}
	return ret.str();
}

