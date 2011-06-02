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

void split_bychar(const string& str, vector<string>& vec,
				 const char separator)
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

void convert_to_pair(vector<string>& vecString, 
					 vector< pair<string, string> >& vecPair)
{
	assert(vecPair.empty());
	int size = vecString.size();
	string::size_type cur;
	string strWord, strPos;
	for(int i = 0; i < size; ++i)
	{
		cur = vecString[i].find('/');
		
		if (cur == string::npos) 
		{
			strWord = vecString[i].substr(0);
			strPos = "";
		}
		else if (cur == vecString[i].size()-1) 
		{
			strWord = vecString[i].substr(0, cur);
			strPos = "";
		}
		else
		{
			strWord = vecString[i].substr(0, cur);
			strPos = vecString[i].substr(cur+1);
		}
		
		vecPair.push_back(pair<string, string>(strWord, strPos));
	}
}

void split_to_pair(const string& str, vector< pair<string, string> >& vecPair)
{
	assert(vecPair.empty());
	vector<string> vec;
	split_bychar(str, vec);
	convert_to_pair(vec, vecPair);
}

void split_sentence(const string& line, vector<string>& vecSentence)
{
	assert(vecSentence.empty());
	vector< pair<string, string> > vecPair;
	split_to_pair(line, vecPair);
	int size = vecPair.size();
	string sentence = "";
	for(int i = 0; i < size; i++)
	{
		if (vecPair[i].first == "。" || vecPair[i].first == "！" || vecPair[i].first == "？") 
		{
			sentence += vecPair[i].first + "/" + vecPair[i].second + " ";
			if (i+1 < size && vecPair[i+1].first == "”") 
			{
				sentence += vecPair[i+1].first + "/" + vecPair[i+1].second + " ";
				i++;
			}
			vecSentence.push_back(sentence);
			sentence = "";
		}
		else
		{
			sentence += vecPair[i].first + "/" + vecPair[i].second + " ";
		}
	}
}

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

int get_char_index(string& str)
{
	assert(str.size() == 2);
	return ((unsigned char)str[0]-176)*94 + (unsigned char)str[1] - 161;
}

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

string separators = "。，？！、：—“”《》（）％￥℃／·\",.?!:'/;；()%"; //all defined separators

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

//void compute_time()
//{
//	clock_t tick = clock();
//	double t = (double)tick / CLK_TCK;
//	cout << endl << "The time used: " << t << " seconds." << endl;
//}

string word(string& word_pos)
{
	return word_pos.substr(0, word_pos.find("/"));
}

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

//////////////////////////////////////////////////////////////////////////
// follow were addde by hjliu at 2007.4.20
//////////////////////////////////////////////////////////////////////////
bool ReadFileToVector(const char* fileName, vector<string>& vecLine)
{
	FILE *fp;
	fp = fopen(fileName, "rb");
	if (!fp)
	{
		cerr << "can't open " << fileName << endl;
		return 0;
	}

	long fileLen;
	fseek(fp, 0, SEEK_END);
	fileLen = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char *buf = new char[fileLen];
	if (!buf)
	{
		cerr << "new memory error" << endl;
		return 0;
	}
	fread(buf, fileLen, 1, fp);
	fclose(fp);

	char word[I_WORD_LEN];

	//add the features config words to vector
	int index = 0;
	int i = 0;
	while (index < fileLen)
	{
		switch(buf[index])
		{
		case 0X0D:
			word[i] = C_END_CHAR;
			i = 0;
			break;
		case 0X0A:
			vecLine.push_back(word);
			break;
		default:
			word[i] = buf[index];
			i++;
			break;
		}

		index++;
	}

	delete []buf;

	return 1;
}

//////////////////////////////////////////////////////////////////////////
void JoinVecToStrByChar(string& strResult, const vector<string>& vecIn, char separator /* =  */)
{
	vector<string>::const_iterator itIn;

	strResult = S_NULL_STR;
	if(vecIn.empty())
	{
		return;
	}

	itIn = vecIn.begin();
	while ( itIn != (vecIn.end() - 1) )
	{
		strResult.append(*itIn);
		strResult.append(1, separator);
		
		itIn++;
	}
	strResult.append(*itIn);
}

//////////////////////////////////////////////////////////////////////////
void RemoveNeighboringSameItem(vector<string>& vecRemoved, const vector<string>& vecOld)
{
	vector<string>::const_iterator itOld;
	
	vecRemoved.clear();
	if (!vecOld.empty())
	{
		itOld = vecOld.begin();
		vecRemoved.push_back(*itOld);
		itOld++;

		while (itOld != vecOld.end())
		{
			if ( (*itOld).compare(vecRemoved.back()) )
			{
				vecRemoved.push_back(*itOld);
			}

			itOld++;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
void RemoveSameItem(vector<string>& vecRemoved, const vector<string>& vecOld)
{
	set<string> setTemp;
	set<string>::iterator itTemp;
	vector<string>::const_iterator itOld;

	itOld = vecOld.begin();
	while (itOld != vecOld.end())
	{
		setTemp.insert(*itOld);
		itOld++;
	}

	vecRemoved.clear();
	itTemp = setTemp.begin();
	while (itTemp != setTemp.end())
	{
		vecRemoved.push_back(*itTemp);
		itTemp++;
	}
}

void split_bychar_to_num(const string& str, vector<int>& vec,
						 const char separator)
{
	assert(vec.empty());
	string::size_type pos1 = 0, pos2 = 0;
	string word;
	while((pos2 = str.find_first_of(separator, pos1)) != string::npos)
	{
		word = str.substr(pos1, pos2-pos1);
		pos1 = pos2 + 1;
		if(!word.empty()) 
			vec.push_back( atoi(word.c_str()) );
	}
	word = str.substr(pos1);
	if(!word.empty())
		vec.push_back( atoi(word.c_str()) );
}
