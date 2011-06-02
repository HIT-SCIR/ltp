/////////////////////////////////////////////////////////////////////////////////////
// File Name   : MyLib.cpp
// Project Name: IRLAS
// Author      : Huipeng Zhang (zhp@ir.hit.edu.cn)
// Environment : Microsoft Visual C++ 6.0
// Description : some utility functions
// Time        : 2005.9
// History     : 
// CopyRight   : HIT-IRLab (c) 2001-2005, all rights reserved.
/////////////////////////////////////////////////////////////////////////////////////
#include "MyLib.h"

void replace_char_by_char(string &str, char c1, char c2)
{
    string::size_type pos = 0;
    for (; pos < str.size(); ++pos) {
        if (str[pos] == c1) {
            str[pos] = c2;
        }
    }
}

// remove the blanks of string
void remove_space(string &str) 
{
    vector<string> vecTmp;
    split_bychar(str, vecTmp, ' ');
    join_bystr(vecTmp, str, "");
}

void join_bystr(const vector<string> &vec, string &str, const string &sep)
{
    str = "";
    if (vec.empty()) return;
    str = vec[0];
    int i = 1;
    for(; i < vec.size(); ++i)
    {
        str += sep + vec[i];
    }
}

// remove the blanks at the begin and end of string
void clean_str(string &str) 
{
    int i = 0;
    for (; i < str.size(); ++i) {
        if (str[i] != ' ' && str[i] != '\t'
                && str[i] != '\n' && str[i] != '\r')
        {
            break;
        }
    }
    if (i > 0)
    {
        str.erase(0, i);
    }

    i = str.size() - 1;
    for (; i >= 0; --i) 
    {
        if (str[i] != ' ' && str[i] != '\t'
                && str[i] != '\n' && str[i] != '\r')
        {
            break;
        }
    }
    if (i < str.size() - 1)
    {
        str.erase(i+1, str.size() - (i+1));
    }
}

/////////////////////////////////////////////////////////////////////////////////////
/// split a sentence into a vector by separator which is a char.
/////////////////////////////////////////////////////////////////////////////////////
void split_bychar(const string& str, vector<string>& vec, const char separator)
{
    assert(vec.empty());
    string::size_type pos1 = 0, pos2 = 0;
    string word;
    while((pos2 = str.find_first_of(separator, pos1)) != string::npos)
    {
        word = str.substr(pos1, pos2-pos1);
        pos1 = pos2 + 1;
        if(!word.empty()) 
            vec.push_back(word);
    }
    word = str.substr(pos1);
    if(!word.empty())
        vec.push_back(word);
}

/////////////////////////////////////////////////////////////////////////////////////
/// convert a string to a pair splited by separator which is '/' by default.
/////////////////////////////////////////////////////////////////////////////////////
void string2pair(const string& str, pair<string, string>& pairStr, const char separator)
{
    string::size_type pos;
    pos = str.find_first_of(separator, 0);
    pairStr.first = str.substr(0, pos);
    if (pos != string::npos) 
    {
        pairStr.second = str.substr(pos+1);
    }
    else 
    {
        pairStr.second = "";
    }
}

/////////////////////////////////////////////////////////////////////////////////////
/// convert every item separated by '/' in a vector to a pair.
/////////////////////////////////////////////////////////////////////////////////////
void convert_to_pair(const vector<string>& vecString, 
        vector< pair<string, string> >& vecPair)
{
    assert(vecPair.empty());
    int size = vecString.size();
    string::size_type cur;
    string strWord, strPos;
    for(int i = 0; i < size; ++i)
    {
        cur = vecString[i].rfind('/');

        if (cur == string::npos) 
        {
            strWord = vecString[i];
            strPos = "";
        }
        else {
            strWord = vecString[i].substr(0, cur);
            strPos = vecString[i].substr(cur + 1);
        }
        if (strWord.empty() || strPos.empty()) {
            cerr << "strWord: #" << strWord << "#\n"
                << "strPos: #" << strPos << "#\n";
        }

        vecPair.push_back(pair<string, string>(strWord, strPos));
    }
}

/////////////////////////////////////////////////////////////////////////////////////
/// the combination of split_bychar and convert_to_pair.
/////////////////////////////////////////////////////////////////////////////////////
void split_to_pair(const string& str, vector< pair<string, string> >& vecPair)
{
    assert(vecPair.empty());
    vector<string> vec;
    split_bychar(str, vec);
    convert_to_pair(vec, vecPair);
}

/////////////////////////////////////////////////////////////////////////////////////
/// delete the white(space, Tab or a new line) on the two sides of a string.
/////////////////////////////////////////////////////////////////////////////////////
void chomp(string& str)
{
    string white = " \t\n";
    string::size_type pos1 = str.find_first_not_of(white);
    string::size_type pos2 = str.find_last_not_of(white);
    if (pos1 == string::npos || pos2 == string::npos) 
    {
        str = "";
    }
    else
    {
        str = str.substr(pos1, pos2-pos1+1);
    }	
}

/////////////////////////////////////////////////////////////////////////////////////
/// get the length of the longest common string of two strings.
/////////////////////////////////////////////////////////////////////////////////////
int common_substr_len(string str1, string str2)
{
    string::size_type minLen;
    if (str1.length() < str2.length()) 
    {
        minLen = str1.length();
    }
    else
    {
        minLen = str2.length();
        str1.swap(str2); //make str1 the shorter string
    }

    string::size_type maxSubstrLen = 0;
    string::size_type posBeg;
    string::size_type substrLen;
    string sub;
    for (posBeg = 0; posBeg < minLen; posBeg++) 
    {
        for (substrLen = minLen-posBeg; substrLen > 0; substrLen--) 
        {
            sub = str1.substr(posBeg, substrLen);
            if (str2.find(sub) != string::npos) 
            {
                if (maxSubstrLen < substrLen) 
                {
                    maxSubstrLen = substrLen;
                }

                if (maxSubstrLen >= minLen-posBeg-1) 
                {
                    return maxSubstrLen;
                }				
            }
        }		
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////
/// compute the index of a Chinese character. 
/// the input can be any string whose length is larger than 2.
/////////////////////////////////////////////////////////////////////////////////////
int get_chinese_char_index(string& str)
{
    assert(str.size() == 2);
    return ((unsigned char)str[0]-176)*94 + (unsigned char)str[1] - 161;
}

/////////////////////////////////////////////////////////////////////////////////////
/// judge if a string is a Hanzi.
/////////////////////////////////////////////////////////////////////////////////////
bool is_chinese_char(string& str)
{
    if (str.size() != 2) 
    {
        return false;
    }
    int index = ((unsigned char)str[0]-176)*94 + (unsigned char)str[1] - 161;
    if (index >= 0 && index < 6768) 
    {
        return true;
    }
    else
    {
        return false;
    }
}

/// all defined separators
string separators = "。，？！、：—“”《》（）％￥℃／·\",.?!:'/;；()%"; 

/////////////////////////////////////////////////////////////////////////////////////
/// judge if a string is a separator.
/////////////////////////////////////////////////////////////////////////////////////
bool is_separator(string& str)
{
    if (separators.find(str) != string::npos && str.size() <= 2) 
    {
        return true;
    }
    else 
    {
        return false;
    }
}

/////////////////////////////////////////////////////////////////////////////////////
/// find GB char which is two-char-width and the first char is negative.
/// it is a little different from string::find.
/////////////////////////////////////////////////////////////////////////////////////
int find_GB_char(const string& str, string wideChar, int begPos)
{
    assert(wideChar.size() == 2 && wideChar[0] < 0); //is a GB char
    int strLen = str.size();
    if (begPos >= strLen) 
    {
        return -1;
    }
    string GBchar;
    for (int i = begPos; i < strLen-1; i++) 
    {
        if (str[i] < 0) //is a GB char 
        {
            GBchar = str.substr(i, 2);
            if (GBchar == wideChar) 
                return i;
            else 
                i++;
        }
    }
    return -1;
}

/////////////////////////////////////////////////////////////////////////////////////
/// split a line to sentences separated by period.
/////////////////////////////////////////////////////////////////////////////////////
void split_to_sentence_by_period(const string& line, vector<string>& vecSentence)
{
    assert(vecSentence.empty());
    int pos1 = 0, pos2 = 0;
    string sentence;
    while((pos2 = find_GB_char(line, "。", pos1)) != -1)
    {
        sentence = line.substr(pos1, pos2-pos1+2);
        pos1 = pos2 + 2;
        if(!sentence.empty()) 
            vecSentence.push_back(sentence);
    }
    sentence = line.substr(pos1);
    if(!sentence.empty())
        vecSentence.push_back(sentence);
}

/////////////////////////////////////////////////////////////////////////////////////
/// it is similar to split_bychar, except that the separator can be a string.
/////////////////////////////////////////////////////////////////////////////////////
void split_by_separator(const string& str, vector<string>& vec, const string separator)
{
    assert(vec.empty());
    string::size_type pos1 = 0, pos2 = 0;
    string word;
    while((pos2 = find_GB_char(str, separator, pos1)) != -1)
    {
        word = str.substr(pos1, pos2-pos1);
        pos1 = pos2 + separator.size();
        if(!word.empty()) 
            vec.push_back(word);
    }
    word = str.substr(pos1);
    if(!word.empty())
        vec.push_back(word);
}

/////////////////////////////////////////////////////////////////////////////////////
/// judge if a string is a Chinese number.
/////////////////////////////////////////////////////////////////////////////////////
bool is_chinese_number(const string& str)
{
    if (str == "一" || str == "二" || str == "三" || str == "四" || str == "五" ||
            str == "六" || str == "七" || str == "八" || str == "九" || str == "十" ||
            str == "两" || str == "几" || str == "零" || str == "〇" || str == "百" ||
            str == "千" || str == "万" || str == "亿") 
    {
        return true;
    }
    else
    {
        return false;
    }
}

/////////////////////////////////////////////////////////////////////////////////////
/// compute the total time used by a program.
/////////////////////////////////////////////////////////////////////////////////////
void compute_time()
{
    clock_t tick = clock();
    double t = (double)tick / CLK_TCK;
    cout << endl << "The time used: " << t << " seconds." << endl;
}

/////////////////////////////////////////////////////////////////////////////////////
/// for example: "高兴/a" -> "高兴".
/////////////////////////////////////////////////////////////////////////////////////
string word(string& word_pos)
{
    return word_pos.substr(0, word_pos.find("/"));
}

/////////////////////////////////////////////////////////////////////////////////////
/// judge if a string purely consist of ASCII characters.
/////////////////////////////////////////////////////////////////////////////////////
bool is_ascii_string(string& word)
{
    for (unsigned int i = 0; i < word.size(); i++)
    {
        if (word[i] < 0)
        {
            return false;
        }
    }
    return true;
}

