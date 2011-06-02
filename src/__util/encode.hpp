#ifndef __ENCODE_HPP__
#define __ENCODE_HPP__

#include <string>
#include <cstdlib>
#include <map>
#include <vector>

#define ENC_NS_BEG namespace enc {
#define ENC_NS_END };

ENC_NS_BEG
// new definition

void init();		// initializing

// GBK <=> unicode
std::wstring decode_gbk(const std::string& s);
std::wstring & decode_gbk(const std::string& s, std::wstring &ws); // `s' is input, `ws' is output
std::wstring decode_gbk(const char* s);

std::string encode_gbk(const std::wstring &ws);
std::string encode_gbk(const wchar_t* ws);
std::string& encode_gbk(const std::wstring &src, std::string &dst);

// UTF8 <=> unicode
std::wstring& decode_utf8(const std::string& s, std::wstring &ws);
std::wstring decode_utf8(const std::string& s);
std::wstring decode_utf8(const char* s);
std::string& encode_utf8(const std::wstring &ws, std::string &s);
std::string encode_utf8(const std::wstring &ws);
std::string encode_utf8(const wchar_t* ws);

// BIG5 <=> unicode
void decode_big5(const std::string& s, std::wstring &ws);
std::wstring decode_big5(const std::string& s);
std::wstring decode_big5(const char* s);
void encode_big5(const std::wstring &ws, std::string &s);
std::string encode_big5(const std::wstring &ws);
std::string encode_big5(const wchar_t* ws);

std::wstring decode(const std::string enc, const std::string &s);
std::string encode(const std::string enc, const std::wstring &s);

// previous definition ...

ENC_NS_END

//std::ostream& operator << (std::ostream &out, const std::wstring &str);
//std::ostream& operator << (std::ostream &out, const wchar_t *str);

#endif // __ENCODE_HPP__
