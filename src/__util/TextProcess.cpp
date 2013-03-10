#include "TextProcess.h"
#include <string.h>
#include <ctime>

using namespace std;

namespace txtutil
{
	//把str中以token分割的字符串存入vecstr中
	bool split_by_token(vector<string>& vecstr, const string &str, const string token)
	{
		vecstr.clear();
		string::size_type LeftPst = 0;
		string::size_type RightPst = 0;

		while((RightPst = str.find(token.c_str(), LeftPst)) != string::npos && LeftPst < str.size())
		{
			if(RightPst != 0)
			{
				const string &term = str.substr(LeftPst, RightPst-LeftPst);
				if( term.length() > 0 )
					vecstr.push_back(term);
				LeftPst = RightPst + token.size();
			}
			//str以token开头
			else
			{
				LeftPst = RightPst + token.size();
			}
		}

		if(LeftPst < str.size())
		{
			const string &term = str.substr(LeftPst);
			if( term.length() > 0 )
				vecstr.push_back(term);
		}

		return (!vecstr.empty());
	}

	//把str中从start开始left_token和right_token之间的部分存入子串subsrt中
	bool SplitSubstrByTokens(const string& str, string::size_type start,
		const string& left_token, const string& right_token, string& substr)
	{
		string::size_type LeftPst;
		string::size_type RightPst;
		if((LeftPst = str.find(left_token.c_str(), start)) != string::npos)
		{
			if((RightPst = str.find(right_token.c_str(), LeftPst)) != string::npos)
			{
				substr = str.substr(LeftPst+1, RightPst-LeftPst-1);
				return true;
			}
		}
		return false;
	}

	string& trim_return(string &str)
	{
		string::size_type pos;
		if( (pos=str.find_last_of('\r')) != string::npos )
			str.resize(pos);
		else if( (pos=str.find_last_of('\n')) != string::npos )
			str.resize(pos);
		return str;
	}

	//移除末尾的回车换行，字符串以0结尾
	char* trim_return(char *str)
	{
		char *p;
		if( (p=strrchr(str, '\r')) != NULL ) *p=0;
		else if( (p=strrchr(str, '\n')) != NULL ) *p=0;
		return str;
	}

	wchar_t* trim_return(wchar_t *str)
	{
		wchar_t *p;
		if( (p=wcsrchr(str, L'\r')) != NULL ) *p=0;
		else if( (p=wcsrchr(str, L'\n')) != NULL) *p=0;
		return str;
	}

	void RemoveEndRN(char* source)
	{
		char *p;
		if((p = strrchr(source, '\r')) != 0)
		{
			*p = 0;
		}
		else if((p = strrchr(source, '\n')) !=0 )
		{
			*p = 0;
		}
	}

	//中文单词的Hash函数
	unsigned int hashChinese(const char* Word)
	{
		unsigned int c1, c2;
		if( strlen(Word) < 2 )	return 0;
		c1 = (unsigned char)*(Word);
		c2 = (unsigned char)*(Word+1);
		if( c1 < 176 )
			return 0;
		return (c1 - 176)*94 + (c2-161) + 1;
	}

	//英文单词的Hash函数
	unsigned int hashEnglish(const char* Word)
	{
		unsigned int c1, c2;
		if( strlen(Word) < 2 ) return 0;
		c1 = (unsigned char)*(Word);
		c2 = (unsigned char)*(Word + 1);
		if( c1 <'A' || c1 > 'z') return 0;
		if( c2 <'A' || c2 > 'z') return 0;
		return (c1 - 'A')*('z'-'A'+1) + (c2 - 'A') + 1;
	}

	unsigned int hashFourByte(const char *str1, const char*str2, int length, int mod)
	{
		register unsigned int h = 0;
		register unsigned char *p = (unsigned char *)str1;

		for(int i = 0;	*p && i != length; p++,i++)
		{
			h = 31 * h + *p;
		}
		p = (unsigned char*)str2;
		for(int j = 0; *p && j != length; p++,j++)
		{
			h = 31 * h + *p;
		}
		return h%mod;
	}

	std::string lowercase(const string &str)
	{
		string res;
		string::const_iterator i;
		res.reserve(str.size());
		for(i=str.begin(); i!=str.end(); ++i)
		{
			if( *i >='A' && *i<='Z' )
				res.push_back(*i-'A'+'a');
			else
				res.push_back(*i);
		}
		return res;
	}

	std::string trim(const std::string &str)
	{
		size_t i=0, j=0, size=0;
		size = str.size();
		while( i<size && str[i]==' ' ) ++i;
		j=size-1;
		while( j>0 && str[j]==' ' ) --j;
		return str.substr(i, j-i+1);
	}

	std::string trim_start(const std::string &str)
	{
		size_t size=0;
		size = str.size();
		string::size_type i=0;
		while( i<size && str[i]==' ' ) ++i;
		return str.substr(i);
	}

	int to_dec(const char c)
	{
		if( c>='0' && c<='9' ) return c-'0';
		if( c>='A' && c<='F' ) return c-'A'+10;
		if( c>='a' && c<='f' ) return c-'a'+10;
		return -1;
	}

	std::string convert_escape(const std::string &str)
	{
		string::const_iterator i;
		string r;
		char c1, c2, c;
		r.reserve( str.size() );
		for(i=str.begin(); i!=str.end(); ++i)
		{
			if( *i == '%' )
			{
				i++;
				if( i!=str.end() ) c1 = *i;
				else break;
				i++;
				if( i!=str.end() ) c2 = *i;
				else break;

				c = to_dec(c1)*16 + to_dec(c2);
				r.append(1, c);
			}
			else
				r.append(1, *i);
		}
		return r;
	}

	void GetExpireTime(const int nExpireSecond, string &strCurrent, string &strExpire)
	{
		time_t now, later;
		time(&now);
		later = now + nExpireSecond;

		struct tm *tm_now, *tm_later;
		const int buf_size = 256;
		char buf[buf_size];

		tm_now = gmtime(&now);			// convert UTC to GMT
		strftime(buf, buf_size, "%a, %d %b %Y %H:%M:%S GMT", tm_now);
		strCurrent.assign(buf);

		tm_later = gmtime(&later);
		strftime(buf, buf_size, "%a, %d %b %Y %H:%M:%S GMT", tm_later);
		strExpire.assign(buf);
	}

	void get_current_time(string &strCurrent)
	{
		time_t now;
		time(&now);

		struct tm *tm_now;
		const int buf_size = 256;
		char buf[buf_size];

		tm_now = gmtime(&now);			// convert UTC to GMT
		strftime(buf, buf_size, "%a, %d %b %Y %H:%M:%S GMT", tm_now);
		strCurrent.assign(buf);
	}

	void get_current_time_std(string &strCurrent)
	{
		time_t now;
		time(&now);

		struct tm *tm_now;
		const int buf_size = 256;
		char buf[buf_size];

		tm_now = gmtime(&now);			// convert UTC to GMT
		strftime(buf, buf_size, "%m-%d %H:%M:%S ", tm_now);	// MM:DD HH:MM:SS
		strCurrent.assign(buf);
	}


	bool isprefix(const std::wstring &src, const std::wstring &prefix)
	{
		size_t s1=prefix.size();
		size_t s2=src.size();
		if( s1>s2 ) return false;

		size_t i=0;
		while( i<s1 && src[i]==prefix[i])
			++i;
		return i == s1;
	}

	bool ispostfix(const std::wstring &src, const std::wstring &postfix)
	{
		size_t s1=postfix.size();
		size_t s2=src.size();
		if( s1>s2 ) return false;

		int i=(int)postfix.size()-1;
		int j=(int)src.size()-1;
		while( i>=0 && src[j]==postfix[i])
			--i, --j;
		return i == -1;
	}

	// support to parse_line
	static const char* parse_text(const char *buf, string &text)
	{
		const char *p=buf;
		text.resize(0);
		while(*p!=0 && *p!='>' && *p!='<' )
			text.append(1, *p++);
		return p;
	}

	// extract "<key>value</key>" to "(key, value)"
	void parse_xml_line(const char *line, string &key, string &value)
	{
		char const *p=line;
		string tag2;
		int state=0;
		while( *p!=0 )
		{
			switch(state)
			{
			case 0:
				if( *p=='<' )
				{
					p=parse_text(p+1, key);
					// change to the tag-text state when encountered the expected '>'
					if( *p=='>' )
						state=1;
				}
				break;
			case 1:
				p=parse_text(p, value);
				// get ready to parse the end tag
				if( *p=='<' )
					state=2;
				break;
			case 2:
				p=parse_text(p, tag2);
				// succeeded only when the beginning tag and ending tag matched
				if( *p=='>' && "/"+key == tag2 )
					state=3;
				break;
			case 3:
				return;
			}
			p++;
		}
//		key.resize(0);
//		value.resize(0);
	}

}
