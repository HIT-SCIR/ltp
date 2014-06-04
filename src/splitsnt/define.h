// NOTICE: 
//  define _WIN32 for Microsoft Visual C++ 6.0 or higher
//  define _GCC for g++ compiler

#ifndef __LUCID_DEFINE_H__
#define __LUCID_DEFINE_H__

//////////////////////////////////////////////////////////////////////////
/// definition for ASCII or wide-char code
/// _ASCII -- ASCII code
/// _UCS2  -- wide code
#if defined(_UCS2)
#pragma message ("==================Using UCS2 mode!!!==================")

	#define string_t wstring
	#define TCHAR wchar_t
 #define _tcslen wcslen //get length of a string
#else //if defined(_ASCII)
#pragma message ("==================Using ASCII mode!!!==================")

  	#define string_t string
	#define TCHAR char
   #define _tcslen strlen
#endif	// end of _UCS2

#ifdef _WIN32		// VC6 platform
typedef __int64	int64_t;
typedef unsigned __int64 uint64_t;
#else	// other platforms
// typedef long long int64_t;
// typedef unsigned long long uint64_t;
#include <sys/types.h>
#endif
typedef int int32_t;
typedef unsigned int uint32_t;

#define LUCENE_MAX_FILELENGTH 0x7FFFFFFFL

#endif	//__LUCID_DEFINE_H__
