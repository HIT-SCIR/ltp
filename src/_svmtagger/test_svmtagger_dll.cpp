
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <iterator>
#include "svmtagger_dll.h"
#include "MyLib.h"

using namespace std;


int main()
{
	char *szResPath= "../svmtagger_data/";
	svmtagger_LoadResource(szResPath);
	
//	char *p = new char [10000000];
	string strWords = "我 是 中国 人 。 ";
//	strWords += strWords;

	vector<string> vecWord;
	vector<string> vecPOS;
	split_bychar(strWords, vecWord, ' ');
	svmtagger_PosTag(vecWord, vecPOS);
	
	copy(vecWord.begin(), vecWord.end(), ostream_iterator<string>(cout, "\t"));
	cout << endl;
	copy(vecPOS.begin(), vecPOS.end(), ostream_iterator<string>(cout, "\t"));
	cout << endl;

	svmtagger_ReleaseResource();

	return 0;
}

