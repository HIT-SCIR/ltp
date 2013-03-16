#ifndef _WENXU_TEXTPROCESS_H
#define _WENXU_TEXTPROCESS_H

#include <string>
#include <vector>
#include <sstream>

namespace txtutil
{
	template <typename T, typename T2>
	bool split_by_char(std::vector<T> &vecstr, const T& str, T2 token=' ', bool bAllowEmpty=false)
	{
		vecstr.clear();
		typename T::size_type LeftPst = 0;
		typename T::size_type RightPst = 0;

		while((RightPst = str.find_first_of(token, LeftPst)) != T::npos && LeftPst < str.size())
		{
			if(RightPst != 0)
			{
				const T &term( str.substr(LeftPst, RightPst-LeftPst) );
				if( bAllowEmpty || !term.empty() )
					vecstr.push_back(term);
			}
			LeftPst = RightPst + 1;
		}

		if(LeftPst < str.size())
		{
			const T& term(str.substr(LeftPst));
			if( !term.empty() )
				vecstr.push_back(term);
		}
		return (!vecstr.empty());
	}

	//把str中以token分割的字符串存入vecstr中
	bool split_by_token(std::vector<std::string>& vecstr, const std::string &str, const std::string token);

	//移除末尾的回车换行，字符串以0结尾
	void RemoveEndRN(char* source);
	char* trim_return(char *str);
	wchar_t* trim_return(wchar_t *str);
	std::string& trim_return(std::string &str);

	//英文单词的Hash函数
	unsigned int hashEnglish(const char* Word);
	//中文单词的Hash函数
	unsigned int hashChinese(const char* Word);
	//hash两个中文汉字
	unsigned int hashFourByte(const char *str1, const char*str2, int length, int mod);

	template <typename T> T lowercase(const T &str)
	{
		T res;
		typename T::const_iterator i;
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

	std::string trim(const std::string &str);
	std::string trim_start(const std::string &str);

	// for HTTP converting
	int to_dec(const char c);
	std::string convert_escape(const std::string &str);
	void GetExpireTime(const int nExpireSecond, std::string &strCurrent, std::string &strExpire);
	void get_current_time(std::string &strCurrent);
	void get_current_time_std(std::string &strCurrent);

	template<typename T>
		T dump_vector(const std::vector<T> &vec, const char* delimiter=" ")
	{
		std::stringstream ss;
		typename std::vector<T>::const_iterator j=vec.begin();
		if( j!=vec.end() )
		{
			ss << *j;
			j++;
		}
		for(; j!=vec.end(); ++j)
			ss << delimiter << *j;
		return ss.str();
	};

	template<typename T>
		T dump_vector2(const std::vector<T> &vec)
	{
		std::stringstream ss;
		typename std::vector<T>::const_iterator j=vec.begin();
		if( j!=vec.end() )
		{
			ss << "[" << *j << "]";
			j++;
		}
		for(; j!=vec.end(); ++j)
			ss << "[" << *j << "]";
		return ss.str();
	};

	bool isprefix(const std::wstring &src, const std::wstring &prefix);
	bool ispostfix(const std::wstring &src, const std::wstring &postfix);

	// extract XML-like lines (eg, "<key>value</key>") to "(key, value)"
	void parse_xml_line(const char *line, std::string &key, std::string &value);
}
#endif
