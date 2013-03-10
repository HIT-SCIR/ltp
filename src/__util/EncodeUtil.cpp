#include "EncodeUtil.h"
#include <fstream>
#include "conversion_utf.h"
#include "gbk_u16.h"

using namespace std;

const char* EncodeUtil::default_charset = "gbk";

wchar_t EncodeUtil::_gbk2uni[MAX_CODE_NUM];
wchar_t EncodeUtil::_uni2gbk[MAX_CODE_NUM];

bool EncodeUtil::_bGbkUnicodeInit=false;

int EncodeUtil::hex2int(const char c)
{
	if( c>='A' && c<='F' ) return c-'A'+10;
	return c-'0';
}

wchar_t EncodeUtil::hex2int(const char *hex, const int size)
{
	wchar_t num=0;
	for(int i=0; i<size; ++i)
	{
		num <<= 4;
		num |= hex2int(hex[i]);
	}
	return num;
}

unsigned EncodeUtil::MultiByteToWideChar(const std::string &sMulti, std::wstring &wsWide)
{
	string::const_iterator i;
	unsigned char c;
	wsWide.reserve( sMulti.size() );
	wsWide.resize(0);
	for(i=sMulti.begin(); i!=sMulti.end(); ++i)
	{
		if( ((c = *i) & 0x80) == 0x80 )
		{
			i++;
			if( i!=sMulti.end() )
				wsWide.append(1, static_cast<wchar_t>( (c<<8) | (unsigned char)*i ));
			else
				return (unsigned) wsWide.size();
		}
		else
			wsWide.append(1, static_cast<wchar_t>( c ) );
	}
	return (unsigned)wsWide.size();
}

unsigned EncodeUtil::WideCharToMultiByte(const std::wstring &wsWide, std::string &sMulti)
{
	wstring::const_iterator i;
	sMulti.reserve(wsWide.size());
	sMulti.resize(0);
	wchar_t c;
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

void EncodeUtil::GbkToUnicode(const wstring &wsLocal, wstring &wsUnicode)
{
	wsUnicode.resize(wsLocal.length());
	unsigned i;
	for(i=0; i<wsLocal.size(); ++i)
	{
		wsUnicode[i] = _gbk2uni[ wsLocal[i] & 0xFFFF ];
	}
}

void EncodeUtil::UnicodeToGbk(const wstring &wsUnicode, wstring &wsLocal)
{
	wsLocal.resize(wsUnicode.length());
	unsigned i;
	for(i=0; i<wsUnicode.size(); ++i)
	{
		wsLocal[i] = _uni2gbk[ wsUnicode[i] & 0xFFFF ];
	}
}

void EncodeUtil::Utf16ToUtf8(const wstring &u16, string &u8)
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
	return;
}

void EncodeUtil::Utf8ToUtf16(const string &u8, wstring &u16)
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
	return;
}

void EncodeUtil::split_utf8(const string &utf8, std::vector<string> &vec)
{
	vec.resize(utf8.size());

	const unsigned char *pu8 = (const unsigned char*)utf8.c_str();
	size_t len=0;
	wchar_t w=0;
	const int buf_size = 10;
	char buf[buf_size];
	int j=0;
	for(size_t i=0; i<utf8.length(); i+=len)
	{
		len = g_f_u8towc(w, pu8+i);
		if( len == (size_t)-1 )
			break;
		// wide char to multi-bytes char
		size_t len2 = g_f_wctou8(buf, w);
		vec[j].assign(buf, len2);
		j++;
	}
	vec.resize(j);
	return;
}

bool EncodeUtil::IsBIG5(const wstring &str, const double SEP_VALUE)
{
	unsigned long int num=0, value=0;
	double final;
	wstring::const_iterator i;

	for( i=str.begin(); i!=str.end(); ++i )
	{
		if( *i > 256 )
		{
			value  += (*i)& 0xff;
			num ++;
		}
	}
	final = (double)value/(double)num;
	bool ret = (final>SEP_VALUE) ? false : true;
	return ret;
}

ostream& operator << (ostream &out, const wstring &str)
{
	string ms;
	EncodeUtil::WideCharToMultiByte(str, ms);
	out << ms;
	return out;
}

ostream& operator << (ostream &out, const wchar_t *str)
{
	string ms;
	EncodeUtil::WideCharToMultiByte(wstring(str), ms);
	out << ms;
	return out;
}

const char *EncodeUtil::_default_fullchar_map[] =
{
	"£°", "0",
	"£±", "1",
	"£²", "2",
	"£³", "3",
	"£´", "4",
	"£µ", "5",
	"£¶", "6",
	"£·", "7",
	"£¸", "8",
	"£¹", "9",
	"£Á", "A",
	"£Â", "B",
	"£Ã", "C",
	"£Ä", "D",
	"£Å", "E",
	"£Æ", "F",
	"£Ç", "G",
	"£È", "H",
	"£É", "I",
	"£Ê", "J",
	"£Ë", "K",
	"£Ì", "L",
	"£Í", "M",
	"£Î", "N",
	"£Ï", "O",
	"£Ð", "P",
	"£Ñ", "Q",
	"£Ò", "R",
	"£Ó", "S",
	"£Ô", "T",
	"£Õ", "U",
	"£Ö", "V",
	"£×", "W",
	"£Ø", "X",
	"£Ù", "Y",
	"£Ú", "Z",
	"£á", "a",
	"£â", "b",
	"£ã", "c",
	"£ä", "d",
	"£å", "e",
	"£æ", "f",
	"£ç", "g",
	"£è", "h",
	"£é", "i",
	"£ê", "j",
	"£ë", "k",
	"£ì", "l",
	"£í", "m",
	"£î", "n",
	"£ï", "o",
	"£ð", "p",
	"£ñ", "q",
	"£ò", "r",
	"£ó", "s",
	"£ô", "t",
	"£õ", "u",
	"£ö", "v",
	"£÷", "w",
	"£ø", "x",
	"£ù", "y",
	"£ú", "z",
	"¡¡", " ",
	"¡¯", "'",
	"¡°", "\"",
	"£¬", ",",
	"¡£", ".",
	"£¡", "!",
	"£¿", "?",

	"£¬", ",",
	"£¨", "(",
	"£©", ")",
	"£¬", ",",
	"¡£", ".",
	"¡¢", ",",
	"¡°", "\"",
	"¡±", "\"",
	"¡¡", " ",
	"\t", " ",
	"¡ª", "-",
	"£»", ";",
	"£º", ":",
	"£¿", "?",
	"¡¾", "[",
	"¡¿", "]",
	"£û", "{",
	"£ý", "}",
	"£¡", "!",
	"[", "[",
	"]", "]",
	NULL,
};

void EncodeUtil::InitFullCharMap(fullchar_map &m, const char **pFull)
{
	const char **p = _default_fullchar_map;
	m.clear();
	wchar_t t1, t2;
	wchar_t tmp1[2]={0, 0}, tmp2[2]={0,0};
	while( *p != NULL )
	{
		t1 = EncodeUtil::MultiCharToWideChar(*p++);
		t2 = EncodeUtil::MultiCharToWideChar(*p++);
		m[t1] = t2;
	}

	p=pFull;
	while( p!=NULL && *p != NULL )
	{
		t1 = EncodeUtil::MultiCharToWideChar(*p++);
		t2 = EncodeUtil::MultiCharToWideChar(*p++);
		m[t1] = t2;
	}
}

void EncodeUtil::FullToHalf(wstring &str, const fullchar_map &fm)
{
	wstring::iterator i;
	fullchar_map::const_iterator j;
	for(i=str.begin(); i!=str.end(); ++i)
	{
		if( (j=fm.find(*i)) != fm.end() )
			*i = j->second;
	}
	return;
}

void EncodeUtil::GbkToUtf8(const std::string gbk, std::string &utf8)
{
	wstring utf16, wgbk;
	EncodeUtil::MultiByteToWideChar(gbk, wgbk);
	EncodeUtil::GbkToUnicode(wgbk, utf16);
	EncodeUtil::Utf16ToUtf8(utf16, utf8);
}

void EncodeUtil::Utf8ToGbk(const std::string utf8, std::string &gbk)
{
	wstring utf16, wgbk;
	EncodeUtil::Utf8ToUtf16(utf8, utf16);
	EncodeUtil::UnicodeToGbk(utf16, wgbk);
	EncodeUtil::WideCharToMultiByte(wgbk, gbk);
}

wstring EncodeUtil::gbk_unicode(const string gbk)
{
	wstring tmp, unicode;
	EncodeUtil::MultiByteToWideChar(gbk, tmp);
	EncodeUtil::GbkToUnicode(tmp, unicode);
	return unicode;
}

string EncodeUtil::unicode_gbk(const wstring unicode)
{
	string tmp;
	wstring gbk;
	EncodeUtil::UnicodeToGbk(unicode, gbk);
	EncodeUtil::WideCharToMultiByte(gbk, tmp);
	return tmp;
}
