#include "gparser_dll.h"
#include <iostream>
#include <sstream>
#include <stdlib.h>
using namespace std;

static void str2int_vec(const vector<string> &vecStr, vector<int> &vecInt) 
{
	vecInt.resize(vecStr.size());
	int i = 0;
	for (; i < vecStr.size(); ++i)
	{
		vecInt[i] = atoi(vecStr[i].c_str());
	}
}

static void split_bychar(const string& str, vector<string>& vec,
						 const char separator)
{
	//assert(vec.empty());
	vec.clear();
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

int GParser_Parse_x(void *gparser,
					  const vector<string> &vecWord,
					  const vector<string> &vecCPOS,
					  vector<int> &vecHead,
					  vector<string> &vecLabel)
{
	int nHeadsSize = vecWord.size() * 10;
	int nLabelsSize = vecWord.size() * 30;
	char *szHeads = new char[nHeadsSize];
	char *szLabels = new char[nLabelsSize];
	int ret = GParser_Parse(gparser, vecWord, vecCPOS, szHeads, szLabels, nHeadsSize, nLabelsSize);
	if ( ret < 0 ) {
		if (ret == -11) {
			cerr << "GParser_Parse_x szHeads Size too small, should be " << nHeadsSize << endl;
		} else if (ret == -12) {
			cerr << "GParser_Parse_x szLabels Size too small, should be " << nLabelsSize << endl;
		}

		return -1;
	}
	
	vector<string> vecStrHead;
	split_bychar(string(szHeads), vecStrHead, '\t');
	str2int_vec(vecStrHead, vecHead);
	split_bychar(string(szLabels), vecLabel, '\t');
	return 0;
}
