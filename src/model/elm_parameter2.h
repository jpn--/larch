/*
 *  elm_parameter.h
 *
 *  Copyright 2007-2015 Jeffrey Newman
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



#ifndef __ELM_PARAMETEX_H__
#define __ELM_PARAMETEX_H__

#ifdef SWIG
%{
#include "elm_parameter2.h"
%}
%rename(ParameterLinkArray) elm::paramArray;
#endif // SWIG

#ifndef SWIG
#include "etk.h"
#include "elm_parameter2.h"
#endif // not SWIG

namespace elm {
			
	#ifndef SWIG
	class ParameterList;
	
	class parametex {
	
	/////////// DEFAULT ////////////	
		
	protected:
		std::string freedom;
		size_t freedom_slot;
		// Each parametex is a transform of a freedom.  The most basic transform is 
		// equals.  Gradients and values both get transformed.
	public:
		inline const std::string& freedomName() const { return freedom; }
		virtual double freedomScale() const { return 0.0; }
	protected:
		elm::ParameterList* mdl;
		
	public:
		virtual double pullvalue(const double* pullSource) const;
		virtual void pushvalue(double* pushDest, const double& q) const;
		virtual std::string print() const;
		virtual std::string smallprint() const;
		
		parametex(const std::string& f, elm::ParameterList* mdl); // constructor
	};
		
	///////////// CONSTANT /////////////
	
	class parametex_constant: public parametex {
		double _value;
	public:
		virtual double freedomScale() const { return _value; }
		virtual double pullvalue(const double* pullSource) const;
		virtual void pushvalue(double* pushDest, const double& q) const;
		virtual std::string print() const;
		virtual std::string smallprint() const;
		parametex_constant(const double& value); // constructor
	};
	
	///////////// EQUAL /////////////
	
	class parametex_equal: public parametex {
	public:
		virtual double pullvalue(const double* pullSource) const;
		virtual void pushvalue(double* pushDest, const double& q) const;
		virtual std::string print() const;
		virtual std::string smallprint() const;
		parametex_equal(const std::string& f, elm::ParameterList* mdl); // constructor
	};
	
	///////////// SCALE //////////////

	class parametex_scale: public parametex {
		double _multiplier;
	public:
		virtual double freedomScale() const { return _multiplier; }
		virtual double pullvalue(const double* pullSource) const;
		virtual void pushvalue(double* pushDest, const double& q) const;
		virtual std::string print() const;
		virtual std::string smallprint() const;
		parametex_scale(const std::string& f, elm::ParameterList* mdl, const double& multiplier); // constructor
	};
	
	class paramArray;
	
	void pull_from_freedoms2(const paramArray& par,       double* ops, const double* fr);
	void push_to_freedoms2  (const paramArray& par, const double* ops,       double* fr);
	std::string push_to_freedoms2_ (const paramArray& par, const double* ops,       double* fr);

	class DBArray;
	class LinearComponent;
	class logging_service;

	
	///////////// PARAM ARRAY //////////////
	
	typedef boosted::shared_ptr<parametex> parametexr;
	
	#endif // not SWIG

	#ifndef SWIG
	class paramArray
	: public etk::three_dim
	, etk::object
	{
	protected:
		std::vector<parametexr> z;

	public:
		unsigned length() const { return rows*cols*deps; }
		
		parametexr& operator()(const unsigned& i, const unsigned& j=0, const unsigned& k=0);
		const parametexr operator()(const unsigned& i, const unsigned& j=0, const unsigned& k=0) const;
		parametexr& operator[](const unsigned& i);
		const parametexr operator[](const unsigned& i) const;

		void resize(const unsigned& r=0, const unsigned& c=1, const unsigned& d=1);
		void clear();
		void delete_and_clear();
		
		paramArray(const unsigned& r, const unsigned& c=1, const unsigned& d=1);

	#else // SWIG
	class paramArray: public etk::three_dim {
	#endif // SWIG
	
	public:
		paramArray();
		virtual ~paramArray();
		
		std::string __str__() const;
		std::string __repr__() const;
		
		void pull(const etk::ndarray* listorder, etk::ndarray* apporder);
		void push(etk::ndarray* listorder, const etk::ndarray* apporder);
	};

	

}



#endif // __ELM_PARAMETEX_H__

