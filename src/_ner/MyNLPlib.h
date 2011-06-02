#ifndef __MYNLPLIB_H__
#define __MYNLPLIB_H__

// #define STL_USING_ALL
// #include <STL.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <string>
#include <map>

using namespace std;

struct NEnode
{
	string Word;
	string Pos;
	string NEtag;
};

typedef struct NEnode NENODE;

typedef string::size_type POSITION;

//get file length
inline int getFilelength(FILE *fp);

//encode file
void encodeFile(string& infilename, string& outfilename);

bool isNEtype(string& strIn, bool* bisNEtypeflag);

//split sentence by ' ', e.g:我 /r
void splitSenByWord(const string& strSen, vector< pair<string, string> >& Con, const char SplitChar);

//split sentence by ' ', but put NE in a whole. e.g: [哈尔滨/ns 工业/n 大学/n]ni
void splitSenByNE(const string& strSen, vector< pair<string, string> >& vecOut);

//split sentence, every element has word, pos and NE tag information
void splitSenByNETag(const string& strSen, vector<NENODE>& vecOut);

//split a sentence with a char defined by user
void splitSenByChar(const string& strSen, const char spliter, vector<string>& vecOut);

//tag NE using B I O tagset
void NEtagBIO(const vector< pair<string, string> >& vecIn, 
			  const string& NEtype, vector<NENODE>& vecOut);

//tag NE using BIESO tagset
void NEtagBIESO(const vector< pair<string, string> >& vecIn,
			  const string& NEtype, vector<NENODE>& vecOut);

//get NEtype index
inline int getNEtypeindex(string& NEtype);

//make a map for filename, its name is mapName, and its value is offered by mapValue
void makeaMapForRule(const string& filename, map<string, int>& mapName, int mapValue);

//show vector<string>'s content
inline void showvectorContent(vector<string>& vecTemp);

inline void showvec2pairContent(vector< pair<string, string> >& vec2paTemp);

//output a vector to screen
template<class T>
void output_vector(vector<T>& vec)
{
	copy(vec.begin(), vec.end(), ostream_iterator<T>(cout, " "));
	cout <<endl; 
}

inline void showvec2paContent2(vector< pair<string, double> >& vec2paTemp, ofstream& outfile);

//read strings from infile to a vector, and then sort the vector and delete the same ones.
//At last, output the result to the outfile.
void sortWithUniqueElement(ifstream& infile, ofstream& outfile);

//It is used to extract substr according the begin and end position
inline string extractSen(string& Src, POSITION begin, POSITION end);

#endif
