/*
 *  etk_thread.h
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


#ifndef __TOOLBOX_THREADING_WINDOWS__
#define __TOOLBOX_THREADING_WINDOWS__


#define _ELM_USE_THREADS_ 0  

#include <boost/thread.hpp>
#include <boost/functional.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/chrono.hpp>

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



} // end namespace boost
//#undef BOOST_SP_MSD

#endif // __TOOLBOX_THREADING_WINDOWS__
