/*
 *  etk_thread.h
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


#ifndef __TOOLBOX_THREADING__
#define __TOOLBOX_THREADING__

#ifndef NOTHREAD

#ifdef min
# undef min
#endif
#ifdef max
# undef max
#endif

#ifndef I_AM_WIN
#ifndef __APPLE__
#define I_AM_WIN 1
#endif
#endif

// On non-Apple systems, use boost::thread
#ifdef I_AM_WIN
//#include "etk_thread_windows.h"
#define _ELM_USE_THREADS_ 1
#include <boost/config.hpp>
#include <boost/thread.hpp>
#include <boost/functional.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/chrono.hpp>
#include <boost/enable_shared_from_this.hpp>

#define boosted boost
#define nullptr NULL 
namespace boost {
//#if !defined( BOOST_NO_FUNCTION_TEMPLATE_ORDERING )
//# define BOOST_SP_MSD( T ) boost::detail::sp_inplace_tag< boost::detail::sp_ms_deleter< T > >()
//#else
//# define BOOST_SP_MSD( T ) boost::detail::sp_ms_deleter< T >()
//#endif

template< class T, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10 >
boost::shared_ptr< T > make_shared( A1 const & a1, A2 const & a2, A3 const & a3, A4 const & a4, A5 const & a5, A6 const & a6, A7 const & a7, A8 const & a8, A9 const & a9, A10 const & a10 )
{
    boost::shared_ptr< T > pt( static_cast< T* >( 0 ), detail::sp_ms_deleter< T >() );

    detail::sp_ms_deleter< T > * pd = boost::get_deleter< detail::sp_ms_deleter< T > >( pt );

    void * pv = pd->address();

    ::new( pv ) T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 );
    pd->set_initialized();

    T * pt2 = static_cast< T* >( pv );

    boost::detail::sp_enable_shared_from_this( &pt, pt2, pt2 );
    return boost::shared_ptr< T >( pt, pt2 );
}


template< class T, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11 >
boost::shared_ptr< T > make_shared( A1 const & a1, A2 const & a2, A3 const & a3, A4 const & a4, A5 const & a5, A6 const & a6, A7 const & a7, A8 const & a8, A9 const & a9, A10 const & a10, A11 const & a11 )
{
    boost::shared_ptr< T > pt( static_cast< T* >( 0 ), detail::sp_ms_deleter< T >() );

    detail::sp_ms_deleter< T > * pd = boost::get_deleter< detail::sp_ms_deleter< T > >( pt );

    void * pv = pd->address();

    ::new( pv ) T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 );
    pd->set_initialized();

    T * pt2 = static_cast< T* >( pv );

    boost::detail::sp_enable_shared_from_this( &pt, pt2, pt2 );
    return boost::shared_ptr< T >( pt, pt2 );
}



template< class T, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14 >
boost::shared_ptr< T > make_shared( A1 const & a1, A2 const & a2, A3 const & a3, A4 const & a4, A5 const & a5, A6 const & a6, A7 const & a7, A8 const & a8, A9 const & a9, A10 const & a10, A11 const & a11, A12 const & a12, A13 const & a13, A14 const & a14 )
{
    boost::shared_ptr< T > pt( static_cast< T* >( 0 ), detail::sp_ms_deleter< T >() );

    detail::sp_ms_deleter< T > * pd = boost::get_deleter< detail::sp_ms_deleter< T > >( pt );

    void * pv = pd->address();

    ::new( pv ) T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14 );
    pd->set_initialized();

    T * pt2 = static_cast< T* >( pv );

    boost::detail::sp_enable_shared_from_this( &pt, pt2, pt2 );
    return boost::shared_ptr< T >( pt, pt2 );
}

template< class T, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15 >
boost::shared_ptr< T > make_shared( A1 const & a1, A2 const & a2, A3 const & a3, A4 const & a4, A5 const & a5, A6 const & a6, A7 const & a7, A8 const & a8, A9 const & a9, A10 const & a10, A11 const & a11, A12 const & a12, A13 const & a13, A14 const & a14, A15 const & a15 )
{
    boost::shared_ptr< T > pt( static_cast< T* >( 0 ), detail::sp_ms_deleter< T >() );

    detail::sp_ms_deleter< T > * pd = boost::get_deleter< detail::sp_ms_deleter< T > >( pt );

    void * pv = pd->address();

    ::new( pv ) T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 );
    pd->set_initialized();

    T * pt2 = static_cast< T* >( pv );

    boost::detail::sp_enable_shared_from_this( &pt, pt2, pt2 );
    return boost::shared_ptr< T >( pt, pt2 );
}




} // end namespace boost
//#undef BOOST_SP_MSD



#endif // def _WINDOWS

















#ifdef __APPLE__
# define _ELM_USE_THREADS_ 1
# define boosted std
# include <thread>
# include <functional>
#endif // def __APPLE__



namespace etk {

	class reading_lockout;
	typedef boosted::shared_ptr<reading_lockout> readlock;

	class reading_lockout
	: public boosted::enable_shared_from_this<reading_lockout>
	{
	public:
		readlock lock() {
			return shared_from_this();
		}
		
		inline bool check_use_count() {
			return ((shared_from_this().use_count()-2)>0);
		}
		
		static readlock open() {
			return boosted::make_shared<reading_lockout>();
		}
		
	public:
		reading_lockout(){}
	};


	template <class T>
	class ptr_lockout
	{
		T* _ref;
		readlock _lock;
	public:
		ptr_lockout(T* ref, readlock& lock)
		: _ref(ref)
		, _lock(lock)
		{ }
	
		T* operator->() {return _ref;}
		//T* operator*() {return _ref;}
		operator T*() {return _ref;}

        T& operator[](long idx) { return _ref[idx]; }
		const T& operator[](long idx) const { return _ref[idx]; }
		
		T* operator+(const int& b) {return _ref+b;}
	
	};






	
/*	class threadString: boost::noncopyable{
	protected:
		std::string _s;
		boost::mutex _mutex;
	public:
		threadString(const std::string& s = "");
		std::string read();
		void write(const std::string& s);
	};
	
	void spew(etk::threadString* tS);
	void replacer(etk::threadString* tS);
	void threadtester();
	
	
	
	
	class threadCounter: boost::noncopyable{
	protected:
		unsigned _dex;
		unsigned _stop;
		unsigned _start;
		unsigned _step;
		boost::mutex _mutex;
	public:
		threadCounter(const unsigned& start, const unsigned& stop, const unsigned& step=1);
		bool get(unsigned& cntr);
		bool incomplete() const;
	};
	
	*/
}


#endif // NOTHREAD
#endif // __TOOLBOX_THREADING__
