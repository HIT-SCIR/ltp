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
#include <cstring>
#include <sstream>
#include "MyVector.h"
using namespace std;

class string_less
{
    public:
        bool operator()(const string &str1, const string &str2) const {
            int ret = strcmp(str1.c_str(), str2.c_str());
            if (ret < 0) return true;
            else return false;
        }
};

inline void print_time() {
#ifdef SHOW_TIME
    time_t lt=time(NULL); 
    cerr << ctime(&lt) << endl;
#endif
}

inline void readObject(FILE *inf, int &obj) {
    fread(&obj, sizeof(int), 1, inf);
}

template <typename Ty>
void readObject(FILE *inf, MyVector<Ty> &obj) {
    int size = 0;
    fread(&size, sizeof(int), 1, inf);
    if (size <= 0) {
        obj.clear();
        return;
    }
    obj.resize(size);
    fread(obj.begin(), sizeof(Ty), size, inf);
}


inline void writeObject(FILE *outf, int obj) {
    fwrite(&obj, sizeof(int), 1, outf);
}

inline void writeObject(FILE *outf, const string &obj) {
    int size = obj.size() + 1;
    fwrite(&size, sizeof(int), 1, outf);
    //	cerr << "write size: " << size << endl;
    if (!obj.empty()) {
        fwrite(obj.c_str(), obj.size()*sizeof(char), 1, outf);
    }
    char end = '\0';
    fwrite(&end, sizeof(char), 1, outf);
}

template <typename Ty>
void writeObject(FILE *outf, const vector<Ty> &obj) {
    int size = obj.size();
    fwrite(&size, sizeof(size), 1, outf);
    if (0 == size) return;
    fwrite(&(*obj.begin()), sizeof(Ty), size, outf);
}



inline void readObject(ifstream &inf, int &obj) {
    inf >> obj;
    string str;
    getline(inf, str);
}
inline void readObject(ifstream &inf, vector<int> &obj) {
    int size = 0;
    inf >> size;
    obj.resize(size);
    int i = 0;
    for (; i < size; ++i) {
        inf >> obj[i];
    }
    string str;
    getline(inf, str);
}
inline void readObject(ifstream &inf, vector<double> &obj) {
    int size = 0;
    inf >> size;
    obj.resize(size);
    int i = 0;
    for (; i < size; ++i) {
        inf >> obj[i];
    }
    string str;
    getline(inf, str);
}

inline void writeObject(ofstream &outf, int obj) {
    outf << obj << endl;
}
inline void writeObject(ofstream &outf, const vector<int> &obj) {
    outf << obj.size();
    int i = 0;
    for (; i < obj.size(); ++i) {
        outf << " " << obj[i];
    }
    outf << endl;
}

inline void writeObject(ofstream &outf, const vector<double> &obj) {
    outf << obj.size();
    int i = 0;
    for (; i < obj.size(); ++i) {
        outf << " " << obj[i];
    }
    outf << endl;
}

// split by each of the chars
void split_bychars(const string& str, vector<string> & vec, const char *sep = " ");

void replace_char_by_char(string &str, char c1, char c2);

void remove_space_gbk(string &str);
void getCharacters_gbk(const string &str, vector<string> &vecCharacter);

// remove the blanks at the begin and end of string
void clean_str(string &str);
inline void remove_beg_end_spaces(string &str) { clean_str(str); }

bool my_getline(ifstream &inf, string &line);

void int2str_vec(const vector<int> &vecInt, vector<string> &vecStr); 

void str2uint_vec(const vector<string> &vecStr, vector<unsigned int> &vecInt);

void str2int_vec(const vector<string> &vecStr, vector<int> &vecInt); 

void join_bystr(const vector<string> &vec, string &str, const string &sep);

void split_bystr(const string &str, vector<string> &vec, const string &sep);
inline void split_bystr(const string &str, vector<string> &vec, const char *sep) { split_bystr(str, vec, string(sep));}

//split a sentence into a vector by separator which is a char
void split_bychar(const string& str, vector<string> & vec, const char separator = ' ');

//convert a string to a pair splited by separator which is '/' by default
void string2pair(const string& str, pair<string, string>& pairStr, const char separator = '/');

//convert every item separated by '/' in a vector to a pair 
void convert_to_pair(vector<string>& vecString, vector< pair<string, string> >& vecPair);

//the combination of the two functions above
void split_to_pair(const string& str, vector< pair<string, string> >& vecPair);

void split_pair_vector(const vector< pair<int, string> > &vecPair, vector<int> &vecInt, vector<string> &vecStr);

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
//void compute_time();

//eg. "高兴/a" -> "高兴"
string word(string& word_pos);

//judge if a string purely consist of ASCII characters
bool is_ascii_string(string& word);

#endif

