#ifndef _lucene_util_Reader_
#define _lucene_util_Reader_

#if defined(_LUCENE_PRAGMA_ONCE)
# pragma once
#endif

#include "define.h"
#include "stddef.h"

namespace util {
	
	//todo: create buffered reader, split buffereing into this class
	//move encoding down lower, so different types of readers can use the encoding
	//i think there are a lot of slow parts to this code... look into that
	
	class Reader
	{
	public:
		virtual void close() = 0;
		virtual int32_t read(TCHAR* b, const int64_t start, const int32_t length) = 0;
		/* the available value may be greater than the actual value if the encoding
		* is a variable one (such as utf8 or unicode) */
		virtual int64_t available () = 0;
		virtual TCHAR readChar() = 0;
		virtual TCHAR peek() = 0;
		virtual int64_t position() = 0;
		virtual void seek(int64_t position) = 0;
		virtual ~Reader(){
		}
	};
	
	class StringReader:public Reader
	{
	private:
		const TCHAR* data;
		uint32_t pt;
		size_t len;
		bool delVal;
	public:
		StringReader ( const TCHAR* value );
		StringReader ( const TCHAR* value, const size_t length, const bool deletevalue=false );
		~StringReader();
		
		int64_t available ();
		void seek(int64_t position);
		int64_t position();
		void close();
		TCHAR readChar();
		int32_t read(TCHAR* b, const int64_t start, const int32_t length);
		TCHAR peek();
	};
	
}
#endif
