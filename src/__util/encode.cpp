#include "encode.hpp"
#include "decode_gbk.h"
#include <iostream>
#include <cstdlib>

using namespace std;

ENC_NS_BEG

#include "conversion_utf.h"

enum _ENCODE_CONST_
{
	MAX_CODE_NUM = 65536
};

wchar_t _gbk2uni[MAX_CODE_NUM];
wchar_t _uni2gbk[MAX_CODE_NUM];
bool _bGbkUnicodeInit;

void InitGBKDecoder()
{
	if( _bGbkUnicodeInit ) return;
	memset(_gbk2uni, ' ', sizeof(_gbk2uni));
	memset(_uni2gbk, ' ', sizeof(_uni2gbk));
	int i;
	for(i=0; i<=256; ++i)
	{
		_gbk2uni[i] = i;
		_uni2gbk[i] = i;
	}
	const int size=sizeof(gbk_utf)/sizeof(gbk_utf[0]);
	for(i=0; i<size; ++i)
	{
		_gbk2uni[gbk_utf[i][0]] = gbk_utf[i][1];
		_uni2gbk[gbk_utf[i][1]] = gbk_utf[i][0];
	}
	_bGbkUnicodeInit=true;
}

void init()
{
	InitGBKDecoder();
}

size_t _multi_wide(const std::string &sMulti, std::wstring &wsWide)
{
	unsigned char c;
	wsWide.reserve( sMulti.size() );
	wsWide.resize(0);
	for(size_t i=0; i<sMulti.size(); ++i)
	{
		if( ((c = sMulti[i]) & 0x80) == 0x80 )
		{
			i++;
			if( i<sMulti.size() )
				wsWide.append(1, ((wchar_t)(c<<8) & 0xFF00) | ((unsigned char)sMulti[i] & 0xFF));
			else
			{
				wsWide.append(1, (wchar_t)(c & 0xFF));
				i--; // backward
			}
		}
		else
			wsWide.append(1, (wchar_t)( c & 0xFF ) );
	}
	return wsWide.size();
}

size_t _wide_multi(const std::wstring &wsWide, std::string &sMulti)
{
	wstring::const_iterator i;
	sMulti.reserve(wsWide.size());
	sMulti.resize(0);
	wchar_t c=0;
	for(i=wsWide.begin(); i!=wsWide.end(); ++i)
	{
		if( ((c=*i) & 0xff00) != 0 )
		{
			sMulti.append(1, (char)( (c >> 8 ) & 0xff) );
		}
		sMulti.append(1, (char)( c & 0xff ) );
	}
	return (unsigned)sMulti.length();
}

wstring& decode_gbk(const std::string& s, std::wstring &ws)
{
	ws.resize(0);
	_multi_wide(s, ws);
	wstring::iterator i;
	for(i=ws.begin(); i!=ws.end(); ++i)
	{
		*i = _gbk2uni[ *i & 0xFFFF ];
	}
#ifdef NEW_VERSION
	setlocale(LC_ALL, "chs"); 
	const char* _Source = s.c_str();
	size_t _Dsize = s.size() + 1;
	wchar_t *_Dest = new wchar_t[_Dsize];
	wmemset(_Dest, 0, _Dsize);
	mbstowcs(_Dest,_Source,_Dsize);
	ws.assign(_Dest);
	delete []_Dest;
	setlocale(LC_ALL, "C");
#endif
	return ws;
}

std::wstring decode_gbk(const string& s)
{
	wstring ws;
	decode_gbk(s, ws);
	return ws;
}

std::wstring decode_gbk(const char* s)
{
	wstring ws;
	decode_gbk(string(s), ws);
	return ws;
}

// convert unicode to GBK
string& encode_gbk(const std::wstring &src, std::string &dst)
{
	wstring ws;
	ws.resize(src.size());
	wstring::const_iterator i;
	size_t j=0;
	for(i=src.begin(); i!=src.end(); ++i, ++j)
	{
		ws[j] = _uni2gbk[*i & 0xFFFF];
	}
	_wide_multi(ws, dst);
#ifdef NEW_VERSION
	string curLocale(setlocale(LC_ALL, NULL));        // curLocale = "C";
	setlocale(LC_ALL, "chs");
	const wchar_t* _Source = src.c_str();
	size_t _Dsize = 2 * src.size() + 1;
	char *_Dest = new char[_Dsize];
	memset(_Dest,0,_Dsize);
	wcstombs(_Dest,_Source,_Dsize);
	dst.assign(_Dest);
	delete []_Dest;
	setlocale(LC_ALL, curLocale.c_str());
#endif
	return dst;
}

std::string encode_gbk(const std::wstring &ws)
{
	string dst;
	encode_gbk(ws, dst);
	return dst;
}

std::string encode_gbk(const wchar_t* ws)
{
	string dst;
	encode_gbk(wstring(ws), dst);
	return dst;
}


// converting UTF8 to UTF16
wstring& decode_utf8(const string& u8, wstring &u16)
{
	wchar_t w;
	const unsigned char *pu8 = (const unsigned char*)u8.c_str();
	size_t len=0;

	u16.resize(0);
	for(size_t i=0; i<u8.length(); i+=len)
	{
		len = g_f_u8towc(w, pu8+i);
		if( len == (size_t)-1 )
			break;
		u16.append(1, w);
	}
	return u16;
}

std::wstring decode_utf8(const std::string& s)
{
	wstring ws;
	decode_utf8(s, ws);
	return ws;
}

std::wstring decode_utf8(const char* s)
{
	wstring ws;
	decode_utf8(string(s), ws);
	return ws;
}

string& encode_utf8(const wstring &u16, string &u8)
{
	const int buf_size = 10;
	char buf[buf_size];
	size_t len;

	u8.reserve(u16.length() * 2);
	u8.resize(0);
	wstring::const_iterator i;
	for(i=u16.begin(); i!=u16.end(); ++i)
	{
		len = g_f_wctou8(buf, *i);
		if( len == (size_t)-1 )
			break;
		u8.append(buf, len);
	}
	return u8;
}

std::string encode_utf8(const std::wstring &ws)
{
	string s;
	encode_utf8(ws, s);
	return s;
}

std::string encode_utf8(const wchar_t* ws)
{
	string s;
	encode_utf8(wstring(ws), s);
	return s;
}


//////////////////////////////////////////////////////////////////////////
///  BIG5
//////////////////////////////////////////////////////////////////////////

// convert BIG5 code to Unicode
void decode_big5(const std::string& s, std::wstring &ws)
{
	setlocale(LC_ALL, "cht"); 
	const char* _Source = s.c_str();
	size_t _Dsize = s.size() + 1;
	wchar_t *_Dest = new wchar_t[_Dsize];
	wmemset(_Dest, 0, _Dsize);
	mbstowcs(_Dest,_Source,_Dsize);
	ws.assign(_Dest);
	delete []_Dest;
	setlocale(LC_ALL, "C");
}

// convert unicode to GBK
void encode_big5(const std::wstring &src, std::string &dst)
{
	string curLocale(setlocale(LC_ALL, NULL));        // curLocale = "C";
	setlocale(LC_ALL, "cht");
	const wchar_t* _Source = src.c_str();
	size_t _Dsize = 2 * src.size() + 1;
	char *_Dest = new char[_Dsize];
	memset(_Dest,0,_Dsize);
	wcstombs(_Dest,_Source,_Dsize);
	dst.assign(_Dest);
	delete []_Dest;
	setlocale(LC_ALL, curLocale.c_str());
}

ENC_NS_END

/*ostream& operator << (ostream &out, const wstring &str)
{
out << enc::encode_gbk(str);
return out;
}

ostream& operator << (ostream &out, const wchar_t *str)
{
out << enc::encode_gbk(str);
return out;
}
<<<<<<< .mine
*/

