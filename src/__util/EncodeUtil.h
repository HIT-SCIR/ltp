#ifndef __ENCODE_UTIL_H__
#define __ENCODE_UTIL_H__

#include <string>
#include <cstdlib>
#include <map>
#include <vector>

#define HIGH_BYTE(w) ( (unsigned) ((w >> 8 ) & 0xff ) )
#define LOW_BYTE(w)  ( w & 0xff )

class EncodeUtil
{
public:
	enum _ENCODE_CONST_
	{
		MAX_CODE_NUM = 65536
	};
	static const char *default_charset;
	
	static wchar_t _gbk2uni[MAX_CODE_NUM];
	static wchar_t _uni2gbk[MAX_CODE_NUM];
	static bool _bGbkUnicodeInit;

	static void InitGbkU16();

	static int hex2int(const char c);
	static wchar_t hex2int(const char *hex, const int size);

	static int MultiByteToWideChar(int nCodePage,
		const char *pszMulti,
		const int  nMultiSize,
		wchar_t *pwzWide,
		const int  nWideSize);
	static int WideCharToMultiByte(int nCodePage,
		const wchar_t *pwzWide,
		const int  nWideSize,
		char *pszMulti,
		const int  nMultiSize);

	static unsigned MultiByteToWideChar(const std::string &sMulti,
		std::wstring &wsWide);
	static unsigned WideCharToMultiByte(const std::wstring &wsWide,
		std::string &sMulti);

	static void UnicodeToGbk(const std::wstring &wsUnicode, std::wstring &wsLocal);
	static void GbkToUnicode(const std::wstring &wsLocal, std::wstring &wsUnicode);

	static void Utf16ToUtf8(const std::wstring &u16, std::string &u8);
	static void Utf8ToUtf16(const std::string &u8, std::wstring &u16);
	static void split_utf8(const std::string &utf8, std::vector<std::string> &vec);
	
	static void GbkToUtf8(const std::string gbk, std::string &utf8);
	static void Utf8ToGbk(const std::string utf8, std::string &gbk);

	static std::wstring gbk_unicode(const std::string gbk);
	static std::string unicode_gbk(const std::wstring unicode);

	static wchar_t MultiCharToWideChar(const char *ms)
	{
		char const *p = ms;
		wchar_t w;
		w = (*p & 0x80) ? (((*p)<<8)&0xff00 | (*(p+1) & 0xff) ) : *p;
		return w;
	}

	static bool IsBIG5(const std::wstring &str, const double SEP_VALUE=184.0);

	typedef std::map<wchar_t, wchar_t> fullchar_map;	// full char to half char
	static const char* _default_fullchar_map[];
	static void InitFullCharMap(fullchar_map& m, const char **pFull=NULL);

	// full character to half character
	static void FullToHalf(std::wstring &str, const fullchar_map &fm);

};

std::ostream& operator << (std::ostream &out, const std::wstring &str);
std::ostream& operator << (std::ostream &out, const wchar_t *str);

#endif // __ENCODE_UTIL_H__
