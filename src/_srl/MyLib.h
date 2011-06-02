/////////////////////////////////////////////////////////////////////////////////////
// File Name   : MyLib.h
// Project Name: IRLAS
// Author      : Huipeng Zhang (zhp@ir.hit.edu.cn)
// Environment : Microsoft Visual C++ 6.0
// Description : some utility functions 
// Time        : 2005.9
// History     : 
// CopyRight   : HIT-IRLab (c) 2001-2005, all rights reserved.
/////////////////////////////////////////////////////////////////////////////////////
#ifndef _MYLIB_H_
#define _MYLIB_H_

#pragma warning(disable:4786)
#pragma warning(disable:4250)
#pragma warning(disable:4996)
#pragma warning(disable:4018)

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <cassert>
#include <set>
#include <deque>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <strstream>
#include <stdlib.h>

#include "ConstVar.h"

using namespace std;


//split a sentence into a vector by separator which is a char
void split_bychar(const string& str, vector<string> & vec, const char separator = ' ');

//convert a string to a pair splited by separator which is '/' by default
void string2pair(const string& str, pair<string, string>& pairStr, const char separator = '/');

//convert every item separated by '/' in a vector to a pair 
void convert_to_pair(vector<string>& vecString, vector< pair<string, string> >& vecPair);

//the combination of the two functions above
void split_to_pair(const string& str, vector< pair<string, string> >& vecPair);

//split a line to sentences separated by "。", "！" and "？", it needs special 
//consideration of the situation that separator followed by quotation mark
void split_sentence(const string& line, vector<string>& vecSentence);

//it is similar to split_bychar, except that the separator can be a string
void split_by_separator(const string& str, vector<string>& vec, const string separator);

//delete the white(space, Tab or a new line) on the two sides of a string
void chomp(string& str);

//get the length of the longest common string of two strings
int common_substr_len(string str1, string str2);

//compute the index of a Chinese character, the input 
//can be any string whose length is larger than 2
int get_char_index(string& str);

//judge if a string is a Hanzi
bool is_chinese_char(string& str);

//judge if a string is a separator
bool is_separator(string& str); 

//split a line to sentences separated by period
void split_to_sentence_by_period(const string& line, vector<string>& vecSentence);

//find GB char which is two-char-width and the first char is negative
int find_GB_char(const string& str, string wideChar, int begPos);

//output a vector to screen
//template<class T>
//void output_vector(vector<T>& vec)
//{
//	copy(vec.begin(), vec.end(), ostream_iterator<T>(cout, " "));
//	cout <<endl; 
//}

//judge if a string is a Chinese number
bool is_chinese_number(const string& str);

//compute the total time used by a program
void compute_time();

//eg. "高兴/a" -> "高兴"
string word(string& word_pos);

//judge if a string purely consist of ASCII characters
bool is_ascii_string(string& word);


//////////////////////////////////////////////////////////////////////////
// follow were added by hjliu at 2007.4.20
//////////////////////////////////////////////////////////////////////////

//read the file content to a vector, one line for one item
bool ReadFileToVector(const char* fileName, vector<string>& vecLine);
void JoinVecToStrByChar(string& strResult, const vector<string>& vecIn, char separator = ' ');
void RemoveNeighboringSameItem(vector<string>& vecRemoved, const vector<string>& vecOld);
void RemoveSameItem(vector<string>& vecRemoved, const vector<string>& vecOld);

void split_bychar_to_num(const string& str, vector<int> & vec, const char separator = ' ');

#endif
