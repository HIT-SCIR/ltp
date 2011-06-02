//////////////////////////////////////////////////////////////////////
//
// Author: Oskar Wieland (oskar.wieland@gmx.de)
//         You can modifiy and distribute this file freely.
//
// Creation: 31.05.2000
//
// Purpose: Declarations for using STL without warning noise.
//
// Usage: Include this file and define at least one of the STL_USING_xxx
//        macros. Currently supported data types from the STL:
//
//        // defines for using the STL
//        #define STL_USING_ALL
//        #define STL_USING_MAP
//        #define STL_USING_VECTOR
//        #define STL_USING_LIST
//        #define STL_USING_STRING
//        #define STL_USING_STREAM
//        #define STL_USING_ASSERT
//        #define STL_USING_MEMORY
//        #define STL_USING_STACK
//
// Sample:
//        #define STL_USING_ALL
//        #include "STL.h"
//
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// include guards
//////////////////////////////////////////////////////////////////////

#ifndef STLHELPER_INCLUDED_
#define STLHELPER_INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif


//////////////////////////////////////////////////////////////////////
// handy define to include all stuff
//////////////////////////////////////////////////////////////////////

#ifdef STL_USING_ALL

#define STL_USING_MAP
#define STL_USING_VECTOR
#define STL_USING_LIST
#define STL_USING_STRING
#define STL_USING_STREAM
#define STL_USING_ASSERT
#define STL_USING_MEMORY
#define STL_USING_STACK

#endif


//////////////////////////////////////////////////////////////////////
// STL neccessary declaration for map
//////////////////////////////////////////////////////////////////////

#ifdef STL_USING_MAP

#pragma warning(push)

#include <yvals.h>              // warning numbers get enabled in yvals.h 

#pragma warning(disable: 4018)  // signed/unsigned mismatch
#pragma warning(disable: 4100)  // unreferenced formal parameter
#pragma warning(disable: 4245)  // conversion from 'type1' to 'type2', signed/unsigned mismatch
#pragma warning(disable: 4512)  // 'class' : assignment operator could not be generated
#pragma warning(disable: 4663)  // C++ language change: to explicitly specialize class template 'vector'
#pragma warning(disable: 4710)  // 'function' : function not inlined
#pragma warning(disable: 4786)  // identifier was truncated to 'number' characters in the debug information

// BUG: C4786 Warning Is Not Disabled with #pragma Warning
// STATUS: Microsoft has confirmed this to be a bug in the Microsoft product. This warning can be ignored.
// This occured only in the <map> container.

#include <map>

#pragma warning(pop)

#endif


//////////////////////////////////////////////////////////////////////
// STL neccessary declaration for vector
//////////////////////////////////////////////////////////////////////

#ifdef STL_USING_VECTOR

#pragma warning(push)

#include <yvals.h>              // warning numbers get enabled in yvals.h 

#pragma warning(disable: 4018)  // signed/unsigned mismatch
#pragma warning(disable: 4100)  // unreferenced formal parameter
#pragma warning(disable: 4245)  // conversion from 'type1' to 'type2', signed/unsigned mismatch
#pragma warning(disable: 4663)  // C++ language change: to explicitly specialize class template 'vector'
#pragma warning(disable: 4702)  // unreachable code
#pragma warning(disable: 4710)  // 'function' : function not inlined
#pragma warning(disable: 4786)  // identifier was truncated to 'number' characters in the debug information

#include <vector>

#pragma warning(pop)

#endif


//////////////////////////////////////////////////////////////////////
// STL neccessary declaration for list
//////////////////////////////////////////////////////////////////////

#ifdef STL_USING_LIST

#pragma warning(push)

#include <yvals.h>              // warning numbers get enabled in yvals.h 

#pragma warning(disable: 4100)  // unreferenced formal parameter
#pragma warning(disable: 4284)  // return type for 'identifier::operator ->' is not a UDT or reference 
                                // to a UDT. Will produce errors if applied using infix notation
#pragma warning(disable: 4710)  // 'function' : function not inlined
#pragma warning(disable: 4786)  // identifier was truncated to 'number' characters in the debug information

#include <list>

#pragma warning(pop)

#endif


//////////////////////////////////////////////////////////////////////
// STL neccessary declaration for string
//////////////////////////////////////////////////////////////////////

#ifdef STL_USING_STRING

#pragma warning(push)

#include <yvals.h>              // warning numbers get enabled in yvals.h 

#pragma warning(disable: 4018)  // signed/unsigned mismatch
#pragma warning(disable: 4100)  // unreferenced formal parameter
#pragma warning(disable: 4146)  // unary minus operator applied to unsigned type, result still unsigned
#pragma warning(disable: 4244)  // 'conversion' conversion from 'type1' to 'type2', possible loss of data
#pragma warning(disable: 4245)  // conversion from 'type1' to 'type2', signed/unsigned mismatch
#pragma warning(disable: 4511)  // 'class' : copy constructor could not be generated
#pragma warning(disable: 4512)  // 'class' : assignment operator could not be generated
#pragma warning(disable: 4663)  // C++ language change: to explicitly specialize class template 'vector'
#pragma warning(disable: 4710)  // 'function' : function not inlined
#pragma warning(disable: 4786)  // identifier was truncated to 'number' characters in the debug information

#include <string>

#pragma warning(pop)

#pragma warning(disable: 4514)  // unreferenced inline/local function has been removed
#pragma warning(disable: 4710)  // 'function' : function not inlined
#pragma warning(disable: 4786)  // identifier was truncated to 'number' characters in the debug information

#endif


//////////////////////////////////////////////////////////////////////
// STL neccessary declaration for streams
//////////////////////////////////////////////////////////////////////

#ifdef STL_USING_STREAM

#pragma warning(push)

#include <yvals.h>              // warning numbers get enabled in yvals.h 

#pragma warning(disable: 4097)  // typedef-name 'identifier1' used as synonym for class-name 'identifier2'
#pragma warning(disable: 4127)  // conditional expression is constant

#include <sstream>
#include <fstream>

#pragma warning(pop)

#endif


//////////////////////////////////////////////////////////////////////
// STL neccessary declaration for memory
//////////////////////////////////////////////////////////////////////

#ifdef STL_USING_MEMORY

// The STL library provides a type called auto_ptr for managing pointers.  
// This template class acts as a stack variable for dynamically allocated 
// memory.  When the variable goes out of scope, its destructor gets called.  
// In its de-structor, it calls delete on the contained pointer, making sure 
// that the memory is returned to the heap.

#pragma warning(push)

#include <yvals.h>              // warning numbers get enabled in yvals.h 

#pragma warning(disable: 4018)  // signed/unsigned mismatch
#pragma warning(disable: 4100)  // unreferenced formal parameter
#pragma warning(disable: 4245)  // conversion from 'type1' to 'type2', signed/unsigned mismatch
#pragma warning(disable: 4710)  // 'function' : function not inlined
#pragma warning(disable: 4786)  // identifier was truncated to 'number' characters in the debug information

#include <memory>

#pragma warning(pop)

#endif


//////////////////////////////////////////////////////////////////////
// STL neccessary declaration for stack
//////////////////////////////////////////////////////////////////////

#ifdef STL_USING_STACK

#pragma warning(push)

#include <yvals.h>              // warning numbers get enabled in yvals.h 

#pragma warning(disable: 4018)  // signed/unsigned mismatch
#pragma warning(disable: 4100)  // unreferenced formal parameter
#pragma warning(disable: 4245)  // conversion from 'type1' to 'type2', signed/unsigned mismatch
#pragma warning(disable: 4710)  // 'function' : function not inlined
#pragma warning(disable: 4786)  // identifier was truncated to 'number' characters in the debug information

#include <stack>

#pragma warning(pop)

#endif


//////////////////////////////////////////////////////////////////////
// STL neccessary declaration for assert
//////////////////////////////////////////////////////////////////////

#ifdef STL_USING_ASSERT

// avoid macro redefinition when using MFC
#ifndef ASSERT

#include <cassert>

// macros for tracking down errors
#ifdef _DEBUG

#define ASSERT( exp )           assert( exp )
#define VERIFY( exp )           assert( exp )
#define TRACE                   ::OutputDebugString

#else

#define ASSERT( exp )           ((void)0)
#define VERIFY( exp )           ((void)(exp))
#define TRACE                   1 ? (void)0 : ::OutputDebugString

#endif  // _DEBUG

#endif  // ASSERT

// additional macros 
#define ASSERT_BREAK( exp )             { ASSERT(exp); if( !(exp) ) break; }
#define ASSERT_CONTINUE( exp )          { ASSERT(exp); if( !(exp) ) continue; }
#define ASSERT_RETURN( exp )            { ASSERT(exp); if( !(exp) ) return; }
#define ASSERT_RETURN_NULL( exp )       { ASSERT(exp); if( !(exp) ) return 0; }
#define ASSERT_RETURN_FALSE( exp )      { ASSERT(exp); if( !(exp) ) return false; }

#endif  // STL_USING_ASSERT


//////////////////////////////////////////////////////////////////////
// verify proper use of macros
//////////////////////////////////////////////////////////////////////

#if defined STL_USING_MAP || defined STL_USING_VECTOR || defined STL_USING_LIST || defined STL_USING_STRING || defined STL_USING_STREAM || defined STL_USING_ASSERT || defined STL_USING_MEMORY || defined STL_USING_STACK
using namespace std;
#else
#pragma message( "Warning: You included <STL.H> without using any STL features!" )
#endif

#endif  // STLHELPER_INCLUDED_
