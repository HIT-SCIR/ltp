#include "svmtagger_dll.h"

//#include <stdio.h>
//#include <stdlib.h>
//#include <string.h>
#include <string>
#include <vector>
#include <iostream>
using namespace std;

int svmtagger_PosTag(const vector<string> &vecWord, vector<string>&vecPOS)
{
	vecPOS.clear();

	if (vecWord.empty()) return 0;

	int len = vecWord.size();
	const char **wordstr = new const char*[len];
	char **pword = new char*[len];
	if (!wordstr || !pword) {
		cerr << "svmtagger Postagger(): alloc memory err." << endl;
		return -1;
	}

	int j = 0;
	for (;j < len; j++)
	{
//		wordstr[j] = new char[ vecWord[j].size() + 1 ];
		wordstr[j] = vecWord[j].c_str();
		pword[j] = new char[ 20 ];
		*pword[j] = '\0';
		if (!wordstr[j] || !pword[j]) {
			cerr << "svmtagger(): alloc memory err." << endl;
			return -1;
		}
//		strcpy(wordstr[j], vecWord[j].c_str());
	}

	int ret = svmtagger_PosTag(wordstr, pword, len);
	if (0 == ret) {
		vecPOS.resize(len);
		for (j = 0;j < len; j++)
		{
			if(*pword[j] > 0){
				vecPOS[j] = pword[j];
			} else {
				vecPOS[j] = "wp";
			}
		}
	}

	for (j = 0; j < len; j++)
	{
		delete[] pword[j];
	}
	delete[] wordstr;
	delete[] pword;
	
	return ret;
}
