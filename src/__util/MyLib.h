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

#include <set>
#include <deque>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <cfloat>
#include <map>
#include <cassert>
#include <iterator>
using namespace std;

inline void print_time() {
//#ifdef SHOW_TIME
//	time_t lt=time(NULL); 
//	cerr << ctime(&lt) << endl;
//#endif
}

// split by each of the chars
void split_bychars(const string& str, vector<string> & vec, const char *sep = " ");

// Zhenghua Li, 2007-8-31, 15:57
void replace_char_by_char(string &str, char c1, char c2);

// Zhenghua Li, 2007-8-31, 15:57
// remove the blanks at the begin and end of string
void clean_str(string &str);

// remove the blanks of string
void remove_space_gbk(string &str);

void getCharacters_gbk(const string &str, vector<string> &vecCharacter);

void join_bystr(const vector<string> &vec, string &str, const string &sep);

void split_bystr(const string &str, vector<string> &vec, const string &sep);
inline void split_bystr(const string &str, vector<string> &vec, const char *sep) { split_bystr(str, vec, string(sep));}

void split_bychar(const string& str, vector<string> & vec,  const char separator = ' ');

void string2pair(const string& str, pair<string, string>& pairStr, const char separator = '/');

void convert_to_pair(const vector<string>& vecString, vector< pair<string, string> >& vecPair);

void split_to_pair(const string& str, vector< pair<string, string> >& vecPair);

void split_by_separator(const string& str, vector<string>& vec, const string separator);

void chomp(string& str);

int common_substr_len(string str1, string str2);

int get_chinese_char_index(string& str);

bool is_chinese_char(string& str);

bool is_separator(string& str); 

void split_to_sentence_by_period(const string& line, vector<string>& vecSentence);

int find_GB_char(const string& str, string wideChar, int begPos);

/// output a vector to console
template<class T>
void output_vector(vector<T>& vec)
{
	copy(vec.begin(), vec.end(), ostream_iterator<T>(cout, " "));
	cout <<endl; 
}

bool is_chinese_number(const string& str);

void compute_time();

string word(string& word_pos);

bool is_ascii_string(string& word);

#endif
