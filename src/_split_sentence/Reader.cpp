#include "Reader.h"
#include <string>
#include <cstring>

using namespace std;

namespace util {

StringReader::StringReader ( const TCHAR* value, const size_t length, const bool deletevalue ):
data(value),
len(length),
delVal(deletevalue)
{
	pt = 0;
}

StringReader::StringReader( const TCHAR* value )
{
	this->data = value;
	this->len = _tcslen(value);
	delVal = false;
	pt = 0;
}

StringReader::~StringReader()
{
	close();
}

int64_t StringReader::available()
{
	return len-pt;
}

int32_t StringReader::read ( TCHAR* buf, const int64_t start, const int32_t length )
{
	if ( pt >= len )
		return -1;
	int32_t rd = 0;
	while ( pt < len && rd < length )
	{
		buf[start+rd] = data[pt];
		rd ++;
		pt ++;
	}
	return rd;
}

TCHAR StringReader::readChar()
{
	if ( pt>=len )
	{
		if (pt==len)
			return 0;
//		printf("StringReader throwing EOF %d/%d\n", pt, len);fflush(NULL); //todo: some printf debug code here...
		throw string("String reader EOF");
	}
	TCHAR ret = data[pt];
	pt++;
	return ret;
}

TCHAR StringReader::peek()
{
	if ( pt>=len )
	{
		if (pt==len)
			return 0;
//		printf("StringReader throwing EOF %d/%d\n", pt, len);fflush(NULL);
		throw string("String reader EOF");
	}
	return data[pt];
}

void StringReader::close()
{
	if (data != NULL && delVal)
	{
		delete [] (char*)data;
	}
}

int64_t StringReader::position()
{
	return pt;
}

void StringReader::seek(int64_t position)
{
	if (position > LUCENE_MAX_FILELENGTH ) {
/*  	     _CLTHROWA(CL_ERR_IO,"position parameter to StringReader::seek exceeds theoretical"
			 " capacity of StringReader's internal buffer."
			 );*/
		throw string("position parameter to StringReader::seek exceeds theoretical");
	}
	pt=position;
}

}

